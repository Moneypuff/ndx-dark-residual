"""Tests for the pure pieces of ``intra_index_regime_study``.

The study's regime axis is built from per-name daily returns derived from the
payload's packed (split-adjusted, dividend-unadjusted) close panel, so the
things worth pinning are the silent-methodology pieces: the gap-masking in the
return construction, the average-pairwise-correlation gauge (driven with
synthetic panels whose correlation is known by construction), the breadth and
dispersion gauges, the 30/40/30 zone split on both bases (the expanding basis
must not look ahead), the close-vs-r21 validation guard that drops names whose
packed closes are on a different basis, the daily dark-flow tilt spread, the
tape taxonomy, and the entry cool-down. All synthetic; no payload required.
"""
import numpy as np
import pandas as pd
import pytest

import intra_index_regime_study as S


def _bdays(n, start="2022-01-03"):
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# daily_returns
# ---------------------------------------------------------------------------
def test_daily_returns_simple():
    idx = _bdays(4)
    close = pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0]}, index=idx)
    r = S.daily_returns(close)["A"]
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(10.0)
    assert r.iloc[2] == pytest.approx(-10.0)
    assert r.iloc[3] == pytest.approx(0.0)


def test_daily_returns_masks_across_gaps():
    # A NaN close must kill BOTH adjacent returns -- pct_change would otherwise
    # bridge the gap and manufacture a fake multi-day 'daily' move.
    idx = _bdays(4)
    close = pd.DataFrame({"A": [100.0, np.nan, 130.0, 143.0]}, index=idx)
    r = S.daily_returns(close)["A"]
    assert np.isnan(r.iloc[1]) and np.isnan(r.iloc[2])
    assert r.iloc[3] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# avg_pairwise_corr
# ---------------------------------------------------------------------------
def _panel_from_cols(cols):
    idx = _bdays(len(next(iter(cols.values()))))
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()},
                        index=idx)


def test_avg_pairwise_corr_perfect_comovement():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 40)
    ret = _panel_from_cols({f"N{i}": base * (i + 1) for i in range(5)})
    out = S.avg_pairwise_corr(ret, window=21, min_names=5)
    assert np.isnan(out.iloc[19])           # window not yet full
    assert out.iloc[-1] == pytest.approx(1.0)


def test_avg_pairwise_corr_known_mixture():
    # Two names identical, one exactly inverted: pairs (+1, -1, -1) -> mean -1/3.
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 40)
    ret = _panel_from_cols({"A": base, "B": base, "C": -base})
    out = S.avg_pairwise_corr(ret, window=21, min_names=3)
    assert out.iloc[-1] == pytest.approx(-1.0 / 3.0)


def test_avg_pairwise_corr_min_names_guard():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 40)
    ret = _panel_from_cols({"A": base, "B": base * 2})
    out = S.avg_pairwise_corr(ret, window=21, min_names=3)
    assert out.isna().all()


def test_avg_pairwise_corr_excludes_incomplete_windows():
    # A name with one NaN inside the trailing window must not enter that day's
    # correlation matrix (a partial overlap would bias the estimate).
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 40)
    noisy = rng.normal(0, 1, 40)
    cols = {"A": base, "B": base, "C": noisy}
    ret = _panel_from_cols(cols)
    ret.iloc[-5, ret.columns.get_loc("C")] = np.nan
    out = S.avg_pairwise_corr(ret, window=21, min_names=2)
    # with C excluded only the A-B pair remains -> exactly 1.0
    assert out.iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# breadth / dispersion
# ---------------------------------------------------------------------------
def test_breadth_positive_counts_fraction():
    idx = _bdays(30)
    up = np.linspace(100, 120, 30)
    down = np.linspace(100, 80, 30)
    close = pd.DataFrame({"U1": up, "U2": up * 1.01, "U3": up * 0.99, "D1": down},
                         index=idx)
    b = S.breadth_positive(close, window=21, min_names=4)
    assert b.iloc[-1] == pytest.approx(0.75)
    assert np.isnan(b.iloc[10])             # trailing window not yet available


def test_breadth_min_names_guard():
    idx = _bdays(30)
    close = pd.DataFrame({"A": np.linspace(100, 120, 30)}, index=idx)
    assert S.breadth_positive(close, window=21, min_names=2).isna().all()


def test_cross_sectional_dispersion_zero_when_identical():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 1, 40)
    ret = _panel_from_cols({f"N{i}": base for i in range(5)})
    out = S.cross_sectional_dispersion(ret, window=21, min_names=5)
    assert out.iloc[-1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# zones (30/40/30, full + expanding)
# ---------------------------------------------------------------------------
def test_zones_full_sample_cutoffs():
    s = pd.Series(np.arange(100, dtype=float), index=_bdays(100))
    z = S.zones_30_40_30(s, "full", labels=("Low", "Mid", "High"))
    assert (z.iloc[:30] == "Low").all()
    assert (z.iloc[-30:] == "High").all()
    assert (z.iloc[40:60] == "Mid").all()


def test_zones_expanding_no_lookahead():
    # The expanding zone on day t must depend only on values up to t: appending
    # a wild future value cannot change any already-assigned zone.
    rng = np.random.default_rng(5)
    v = rng.normal(0, 1, 60)
    s1 = pd.Series(v, index=_bdays(60))
    v2 = np.concatenate([v, [50.0]])
    s2 = pd.Series(v2, index=_bdays(61))
    z1 = S.zones_30_40_30(s1, "expanding", min_obs=20)
    z2 = S.zones_30_40_30(s2, "expanding", min_obs=20)
    assert (z1.to_numpy() == z2.to_numpy()[:-1]).all()


def test_zones_expanding_na_until_min_obs():
    s = pd.Series(np.arange(30, dtype=float), index=_bdays(30))
    z = S.zones_30_40_30(s, "expanding", min_obs=20)
    assert (z.iloc[:19] == "NA").all()
    # a monotonically rising series always sits at its own historical top
    assert (z.iloc[19:] == "High").all()


def test_expanding_pctile_midrank_on_ties():
    s = pd.Series([1.0, 1.0, 1.0], index=_bdays(3))
    p = S.expanding_pctile(s, min_obs=3)
    # all three equal -> mid-rank 0.5
    assert p.iloc[-1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# validate_names
# ---------------------------------------------------------------------------
def test_validate_names_drops_mismatched_basis():
    idx = _bdays(160)
    rng = np.random.default_rng(6)
    good = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 160)), index=idx)
    # BAD carries an unrepaired 10:1 split halfway through, so its compounded
    # close returns disagree with the (adjusted) r21 panel built from GOODLIKE.
    goodlike = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 160)), index=idx)
    bad = goodlike.copy()
    bad.iloc[80:] = bad.iloc[80:] / 10.0
    close = pd.DataFrame({"GOOD": good, "BAD": bad})
    r21 = pd.DataFrame({
        "GOOD": (good.shift(-21) / good - 1) * 100,
        "BAD": (goodlike.shift(-21) / goodlike - 1) * 100,
    })
    ok, dropped = S.validate_names(close, r21, min_corr=0.98)
    assert ok == ["GOOD"]
    assert "BAD" in dropped and dropped["BAD"] < 0.98


def test_validate_names_keeps_short_history():
    # Too little overlap to validate (recent IPO) -> keep rather than drop.
    idx = _bdays(160)
    rng = np.random.default_rng(7)
    s = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 160)), index=idx)
    short = s.copy()
    short.iloc[:100] = np.nan
    close = pd.DataFrame({"NEW": short})
    r21 = pd.DataFrame({"NEW": (short.shift(-21) / short - 1) * 100})
    ok, dropped = S.validate_names(close, r21, min_corr=0.98)
    assert ok == ["NEW"] and dropped == {}


# ---------------------------------------------------------------------------
# tilt_spread
# ---------------------------------------------------------------------------
def test_tilt_spread_sign_and_guard():
    # 60 names; the first 12 get a POSITIVE dark-flow tilt on the last day and
    # are assigned LOWER forward returns -> spread (top minus bottom) < 0.
    n_days, n_names = 80, 60
    idx = _bdays(n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    d = pd.DataFrame(0.4, index=idx, columns=names)
    d.iloc[-5:, :12] = 0.6            # 5d MA of these names ends above their mean
    fwd = pd.DataFrame(5.0, index=idx, columns=names)
    fwd.iloc[:, :12] = -5.0
    out = S.tilt_spread(d, fwd, names, min_names=50, min_obs=20)
    assert out["spread"].iloc[-1] < 0
    # with fewer live names than min_names the day is unscored
    out2 = S.tilt_spread(d.iloc[:, :30], fwd, names[:30], min_names=50, min_obs=20)
    assert out2["spread"].isna().all()


# ---------------------------------------------------------------------------
# tape_label
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tr,b,expect", [
    (5.0, 0.9, "broad rally"),
    (5.0, 0.1, "narrow rally"),
    (5.0, 0.5, "mid rally"),
    (-5.0, 0.1, "broad selloff"),
    (-5.0, 0.9, "selective selloff"),
    (-5.0, 0.5, "mid selloff"),
    (np.nan, 0.5, "NA"),
])
def test_tape_label_branches(tr, b, expect):
    assert S.tape_label(tr, b, blo=1 / 3, bhi=2 / 3) == expect


# ---------------------------------------------------------------------------
# entry_events
# ---------------------------------------------------------------------------
def test_entry_events_cooldown():
    idx = _bdays(60)
    M = pd.DataFrame(index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[5:8] = True     # entry at 5
    mask.iloc[10:12] = True   # re-entry at 10 -- inside the cool-down, skipped
    mask.iloc[40:42] = True   # entry at 40 -- past the cool-down, counted
    entries = S.entry_events(M, mask, min_gap=21)
    assert entries == [5, 40]


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def test_ols_nw_recovers_slope():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 1, 500)
    y = 1.5 + 2.0 * x + rng.normal(0, 0.1, 500)
    X = np.column_stack([np.ones(500), x])
    beta, se, t, p, r2 = S.ols_nw(y, X, lags=5)
    assert beta[0] == pytest.approx(1.5, abs=0.05)
    assert beta[1] == pytest.approx(2.0, abs=0.05)
    assert r2 > 0.99


def test_block_boot_ci_degenerate_and_coverage():
    assert S.block_boot_ci(np.arange(10.0)) == (pytest.approx(np.nan, nan_ok=True),
                                                pytest.approx(np.nan, nan_ok=True))
    rng = np.random.default_rng(9)
    r = rng.normal(1.0, 1.0, 500)
    lo, hi = S.block_boot_ci(r, seed=1)
    assert lo < 1.0 < hi
    assert hi - lo < 1.0
