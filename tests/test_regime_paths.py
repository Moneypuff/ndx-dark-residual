"""Tests for `regime_paths.py`, the pure computation behind the regime
return-path study (`regime_paths_study.py`) and the comovement page's "How
the month unfolds" section (see `REGIME_PATHS_PLAN.md`).

All series are hand-built so every expected number can be checked by hand;
no network, no fixtures from the live pipeline.
"""
import numpy as np
import pandas as pd
import pytest

import regime_paths as R


def _dates(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B")


# ---------------------------------------------------------------------------
# path_metrics
# ---------------------------------------------------------------------------
def test_path_metrics_straight_line_up():
    # Monotonically up 1%/day: a pure trend with zero excursion below entry.
    idx = _dates(30)
    p = 100 * np.cumprod(np.r_[1.0, np.full(29, 1.01)])
    row = R.path_metrics(pd.Series(p, index=idx), horizon=21, trail=5).iloc[0]
    assert row["er"] == pytest.approx(1.0, abs=1e-9)
    assert row["taw"] == pytest.approx(100.0)
    assert row["xings"] == 0
    assert row["mae"] == pytest.approx(0.0, abs=1e-9)
    assert row["mfe_day"] == 21
    assert row["r21"] > 0


def test_path_metrics_zigzag_chop():
    # Cumulative return alternates +/-0.5% around the entry every session:
    # a pure chop with no net trend and maximally anti-correlated daily
    # returns (each day fully reverses the prior day's move).
    n = 40
    idx = _dates(n)
    c = np.zeros(n)
    c[1:] = np.where(np.arange(1, n) % 2 == 0, 0.005, -0.005)
    p = 100 * (1 + c)
    row = R.path_metrics(pd.Series(p, index=idx), horizon=21, trail=5).iloc[0]
    assert row["er"] < 0.1               # far from a straight line
    assert row["taw"] == pytest.approx(50.0, abs=5.0)
    assert row["xings"] >= 19            # sign flips almost every session
    assert row["ac1"] < -0.9             # each day reverses the last


def test_path_metrics_constant_price_no_exception():
    idx = _dates(30)
    p = np.full(30, 100.0)
    m = R.path_metrics(pd.Series(p, index=idx), horizon=21, trail=5)
    row = m.iloc[0]
    assert row["rv"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(row["rv_ratio"])     # 0/0 trailing vol -> NaN, not an error
    assert row["er"] is not None         # er is NaN too (0/0), but no exception
    assert m.notna().any().any()         # the frame is otherwise usable


def test_path_metrics_known_drawdown_and_runup():
    # Hand path: +5%, -10% (from entry), back to -5%, flat, then +12% at the
    # horizon. mae/mfe and their days should match by inspection.
    idx = _dates(8)
    p = np.array([100.0, 105.0, 90.0, 95.0, 100.0, 112.0, 112.0, 112.0])
    row = R.path_metrics(pd.Series(p, index=idx), horizon=5, trail=2).iloc[0]
    assert row["mae"] == pytest.approx(-10.0, abs=1e-9)
    assert row["mae_day"] == 2
    assert row["mfe"] == pytest.approx(12.0, abs=1e-9)
    assert row["mfe_day"] == 5


def test_path_metrics_rv_ratio_doubled_amplitude():
    # Trailing window: alternating +/-0.4% daily log returns. Forward window:
    # the identical alternating pattern at double amplitude. Realized vol
    # scales linearly with amplitude for the same on/off pattern, so the
    # ratio should land almost exactly on 2.
    trail, horizon, a = 5, 5, 0.004
    tr = np.where(np.arange(trail) % 2 == 0, a, -a)
    fwd = np.where(np.arange(horizon) % 2 == 0, 2 * a, -2 * a)
    logp = np.concatenate([[0.0], np.cumsum(tr), np.cumsum(fwd) + tr.sum()])
    p = 100 * np.exp(logp)
    idx = _dates(len(p))
    m = R.path_metrics(pd.Series(p, index=idx), horizon=horizon, trail=trail)
    row = m.iloc[trail]  # the observation day is where the forward window starts
    assert row["rv_ratio"] == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# fan_quantiles
# ---------------------------------------------------------------------------
def test_fan_quantiles_three_parallel_lines():
    # Three disjoint windows on one close series, each a constant daily
    # return (1% / 3% / 5%). With exactly three samples the 50th percentile
    # at every checkpoint is exactly the middle path's value.
    horizon = 3
    n = 30
    idx = _dates(n)
    close = pd.Series(100.0, index=idx)
    starts = {}
    for start_i, r in ((0, 0.01), (10, 0.03), (20, 0.05)):
        seg = 100 * np.cumprod(np.r_[1.0, np.full(horizon, 1 + r)])
        close.iloc[start_i:start_i + horizon + 1] = seg
        starts[r] = close.index[start_i]

    fan = R.fan_quantiles(close, list(starts.values()), horizon=horizon)
    assert fan.shape == (len(R.FAN_Q), horizon + 1)
    assert np.allclose(fan[:, 0], 0.0)  # day 0 is the entry itself: always flat

    mid_path = (np.cumprod(np.r_[1.0, np.full(horizon, 1.03)]) - 1) * 100
    q50_row = R.FAN_Q.index(50)
    assert np.allclose(fan[q50_row], mid_path, atol=1e-6)


def test_fan_quantiles_empty_dates_returns_nan_shape():
    idx = _dates(10)
    close = pd.Series(100.0, index=idx)
    fan = R.fan_quantiles(close, [], horizon=5)
    assert fan.shape == (len(R.FAN_Q), 6)
    assert np.isnan(fan).all()


# ---------------------------------------------------------------------------
# run_lengths / run_length_at
# ---------------------------------------------------------------------------
def test_run_lengths_known_episodes():
    idx = _dates(7)
    code = pd.Series(["A", "A", "B", "A", "A", "A", "C"], index=idx)
    ep = R.run_lengths(code)
    assert list(ep["code"]) == ["A", "B", "A", "C"]
    assert list(ep["length"]) == [2, 1, 3, 1]
    assert list(ep["start"]) == [idx[0], idx[2], idx[3], idx[6]]


def test_run_lengths_empty():
    ep = R.run_lengths(pd.Series([], dtype=object))
    assert ep.empty
    assert list(ep.columns) == ["code", "start", "length"]


def test_run_length_at_matches_episodes():
    idx = _dates(7)
    code = pd.Series(["A", "A", "B", "A", "A", "A", "C"], index=idx)
    rla = R.run_length_at(code)
    assert list(rla) == [1, 2, 1, 1, 2, 3, 1]


# ---------------------------------------------------------------------------
# classify (boundary cases at exactly the thresholds)
# ---------------------------------------------------------------------------
BASE = {"rv_med": 20.0, "er_med": 0.20, "r21_med": 0.0}


def test_classify_hot_trend_up_at_exact_thresholds():
    row = {"rv_med": 20.0 * R.VOL_HOT, "er_med": 0.20 * R.ER_TREND,
          "r21_med": R.DIR_MIN_PP}
    assert R.classify(row, BASE) == "hot trend up"


def test_classify_quiet_chop_down_at_exact_thresholds():
    row = {"rv_med": 20.0 * R.VOL_QUIET, "er_med": 0.20 * R.ER_CHOP,
          "r21_med": -R.DIR_MIN_PP}
    assert R.classify(row, BASE) == "quiet chop down"


def test_classify_normal_mixed_flat_just_inside_bands():
    row = {"rv_med": 20.0 * 1.05, "er_med": 0.20 * 1.05,
          "r21_med": R.DIR_MIN_PP - 0.01}
    assert R.classify(row, BASE) == "normal mixed flat"


def test_classify_missing_inputs_are_na():
    row = {"rv_med": np.nan, "er_med": np.nan, "r21_med": np.nan}
    assert R.classify(row, BASE) == "n/a n/a n/a"


def test_classify_zero_baseline_is_na_not_crash():
    base = {"rv_med": 0.0, "er_med": 0.0, "r21_med": 0.0}
    row = {"rv_med": 10.0, "er_med": 0.3, "r21_med": 2.0}
    assert R.classify(row, base) == "n/a n/a up"


# ---------------------------------------------------------------------------
# cell_table
# ---------------------------------------------------------------------------
def _synthetic_metrics(idx, rng):
    n = len(idx)
    return pd.DataFrame({
        "r21": rng.normal(1.0, 2.0, n), "rv": rng.normal(20, 3, n),
        "rv_ratio": rng.normal(1.0, 0.2, n), "mae": -np.abs(rng.normal(3, 1, n)),
        "mae_day": rng.integers(0, 21, n).astype(float),
        "mfe": np.abs(rng.normal(3, 1, n)),
        "mfe_day": rng.integers(0, 21, n).astype(float),
        "range": np.abs(rng.normal(6, 1, n)), "er": rng.uniform(0.1, 0.4, n),
        "taw": rng.uniform(30, 70, n), "xings": rng.integers(0, 20, n).astype(float),
        "ac1": rng.uniform(-0.3, 0.3, n), "big_dn": rng.integers(0, 5, n).astype(float),
        "big_up": rng.integers(0, 5, n).astype(float),
    }, index=idx)


def test_cell_table_blanks_small_cells_but_keeps_n():
    idx = _dates(60)
    rng = np.random.default_rng(0)
    code = pd.Series(np.where(np.arange(60) % 5 == 0, "XXX", "MMM"), index=idx)
    code.iloc[45:] = "MMM"  # cap XXX at 9 occurrences (< MIN_N)
    A = pd.DataFrame({"code": code})
    metrics = _synthetic_metrics(idx, rng)

    tbl = R.cell_table(A, metrics, "TEST", with_ci=False)
    by_regime = tbl.set_index("regime")
    assert by_regime.loc["XXX", "n"] == 9
    assert np.isnan(by_regime.loc["XXX", "r21_med"])
    assert np.isnan(by_regime.loc["XXX", "rv_med"])
    assert by_regime.loc["MMM", "n"] >= R.MIN_N
    assert pd.notna(by_regime.loc["MMM", "r21_med"])
    assert "BASELINE" in by_regime.index
    assert by_regime.loc["BASELINE", "n"] == 60


def test_cell_table_min_run_filters_short_episodes():
    idx = _dates(20)
    # AAA never runs longer than 2 sessions in a row.
    code = pd.Series(["AAA", "AAA", "BBB", "BBB", "BBB"] * 4, index=idx)
    A = pd.DataFrame({"code": code})
    rng = np.random.default_rng(1)
    metrics = _synthetic_metrics(idx, rng)

    full = R.cell_table(A, metrics, "TEST", min_run=1, with_ci=False)
    persistent = R.cell_table(A, metrics, "TEST", min_run=3, with_ci=False)
    assert "AAA" in set(full["regime"])
    # AAA never reaches a 3-session run, so it drops out entirely at min_run=3
    assert "AAA" not in set(persistent["regime"])
    assert "BBB" in set(persistent["regime"])


# ---------------------------------------------------------------------------
# vol_matched_baseline
# ---------------------------------------------------------------------------
def test_vol_matched_baseline_mix_sums_to_one_and_runs():
    idx = _dates(300)
    rng = np.random.default_rng(2)
    code = pd.Series(rng.choice(["AAA", "BBB", "CCC"], size=300, p=[0.3, 0.4, 0.3]),
                     index=idx)
    A = pd.DataFrame({"code": code})
    metrics = pd.DataFrame({
        "rv": rng.normal(20, 5, 300), "rv_ratio": rng.normal(1.0, 0.3, 300),
        "er": rng.uniform(0.1, 0.4, 300),
    }, index=idx)
    out = R.vol_matched_baseline(A, metrics, "AAA")
    assert out["mix"]
    assert sum(out["mix"].values()) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(out["rv"])
    assert np.isfinite(out["er"])


def _with_dummy_r1m(A, idx, rng):
    # entry_events (index_comovement_study) loops over all three indices'
    # *_r1m columns regardless of which one the caller asked about.
    out = A.copy()
    for i in ("NDX", "SPX", "IWM"):
        out[i + "_r1m"] = rng.normal(0, 1, len(idx))
    return out


def test_entry_cell_table_never_blanks_small_n():
    # Two entries of code "AAA" (21+-session cool-down keeps only the first
    # of each run), scored regardless of how far below MIN_N that leaves it.
    idx = _dates(120)
    code = pd.Series("BBB", index=idx)
    code.iloc[5:8] = "AAA"     # one run
    code.iloc[60:64] = "AAA"   # a second, well-separated run
    rng = np.random.default_rng(4)
    A = _with_dummy_r1m(pd.DataFrame({"code": code}), idx, rng)
    metrics = _synthetic_metrics(idx, rng)

    et = R.entry_cell_table(A, metrics, "TEST", ["AAA", "BBB"])
    by_regime = et.set_index("regime")
    assert by_regime.loc["AAA", "n"] == 2
    assert pd.notna(by_regime.loc["AAA", "r21_med"])  # not blanked despite n < MIN_N


def test_entry_cell_table_missing_code_reports_zero():
    idx = _dates(30)
    code = pd.Series("BBB", index=idx)
    rng = np.random.default_rng(5)
    A = _with_dummy_r1m(pd.DataFrame({"code": code}), idx, rng)
    metrics = _synthetic_metrics(idx, rng)
    et = R.entry_cell_table(A, metrics, "TEST", ["ZZZ"])
    assert et.iloc[0]["n"] == 0
    assert np.isnan(et.iloc[0]["r21_med"])


def test_vol_matched_baseline_unknown_code_returns_nan():
    idx = _dates(50)
    rng = np.random.default_rng(3)
    code = pd.Series(["AAA"] * 50, index=idx)
    A = pd.DataFrame({"code": code})
    metrics = pd.DataFrame({
        "rv": rng.normal(20, 5, 50), "rv_ratio": rng.normal(1.0, 0.3, 50),
        "er": rng.uniform(0.1, 0.4, 50),
    }, index=idx)
    out = R.vol_matched_baseline(A, metrics, "ZZZ")
    assert out["mix"] == {}
    assert np.isnan(out["rv"])
    assert np.isnan(out["er"])
