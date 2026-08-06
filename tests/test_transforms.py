"""Tests for the core financial transforms: the small, pure DataFrame/Series
functions that turn raw FINRA/price panels into the dark-flow indicators the
whole pipeline is built on.

These are high-blast-radius: forward returns, the per-name D (dark-pool
indicator), the index-level aggregate dark ratio and dollar-DIX, the
residualisation of each name against its benchmark, and the dual-class share
merge. They have no external oracle but their arithmetic is exact, so the
tests build tiny hand-computed panels and assert the numbers, the documented
guards (zero-denominator -> NaN, 0..1 clipping, min-name / coverage floors,
missing-benchmark error), and the alignment/immutability contracts.
"""
import numpy as np
import pandas as pd
import pytest

import ndx_dark_residual as N
import earnings_dpi_study as E
import build_report as B


def _idx(n, start="2022-01-03"):
    return pd.bdate_range(start, periods=n)


# ===========================================================================
# compute_forward_return
# ===========================================================================
def test_forward_return_one_day():
    idx = _idx(4)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=idx)
    fwd = N.compute_forward_return(close, 1)
    # (11/10 - 1)*100 = 10, (12/11-1)*100 = 9.0909..., last is NaN.
    assert fwd["A"].iloc[0] == pytest.approx(10.0)
    assert fwd["A"].iloc[1] == pytest.approx(100.0 / 11.0)
    assert np.isnan(fwd["A"].iloc[-1])


def test_forward_return_multi_horizon_and_shape():
    idx = _idx(5)
    close = pd.DataFrame({"A": [10.0, 20.0, 40.0, 80.0, 160.0]}, index=idx)
    fwd = N.compute_forward_return(close, 2)
    # A doubles every step, so a 2-step forward return is +300%.
    assert fwd["A"].iloc[0] == pytest.approx(300.0)
    assert fwd.shape == close.shape
    assert list(fwd.index) == list(close.index)
    assert np.isnan(fwd["A"].iloc[-1]) and np.isnan(fwd["A"].iloc[-2])


# ===========================================================================
# compute_divergence_signal (DIX-vs-price divergence, option B)
# ===========================================================================
def test_divergence_signal_states_and_guard():
    idx = _idx(30)
    # DIX rises monotonically; price falls -> bullish divergence (+1) everywhere
    # the lookback change is defined; NaN for the first `lookback` sessions.
    dix = pd.Series(np.linspace(0.40, 0.50, 30), index=idx)
    close = pd.Series(np.linspace(100.0, 90.0, 30), index=idx)
    fwd = N.compute_forward_return(pd.DataFrame({"CLOSE": close}), 5)["CLOSE"]
    state, stats = N.compute_divergence_signal(dix, close, fwd, lookback=5)
    assert state.iloc[:5].isna().all()          # undefined until the change exists
    assert (state.iloc[5:] == 1.0).all()        # DIX up + price down = bullish
    assert stats["lookback"] == 5
    assert stats["cur"]["state"] == 1
    assert stats["cur"]["streak"] == 25         # +1 held for every defined day
    assert stats["cur"]["dixchg"] > 0 and stats["cur"]["pxchg"] < 0
    assert stats["bull"]["n"] > 0 and stats["bear"]["n"] == 0


def test_divergence_signal_bearish_and_aligned():
    idx = _idx(12)
    # DIX down while price up over the lookback -> bearish (-1)
    dix = pd.Series(np.linspace(0.50, 0.44, 12), index=idx)
    close = pd.Series(np.linspace(90.0, 100.0, 12), index=idx)
    fwd = N.compute_forward_return(pd.DataFrame({"CLOSE": close}), 3)["CLOSE"]
    state, stats = N.compute_divergence_signal(dix, close, fwd, lookback=3)
    assert (state.dropna() == -1.0).all()
    assert stats["cur"]["state"] == -1
    # both rising -> aligned (0), no divergence in either direction
    dix2 = pd.Series(np.linspace(0.40, 0.50, 12), index=idx)
    state2, stats2 = N.compute_divergence_signal(dix2, close, fwd, lookback=3)
    assert (state2.dropna() == 0.0).all()
    assert stats2["cur"]["state"] == 0
    assert stats2["bull"]["n"] == 0 and stats2["bear"]["n"] == 0


def test_divergence_conditional_stats_ignore_unrealised_forward():
    # The conditional means must use only realised (non-NaN) forward returns --
    # the last `horizon` rows have no future close and must be excluded.
    idx = _idx(20)
    dix = pd.Series(np.linspace(0.40, 0.50, 20), index=idx)
    close = pd.Series(np.linspace(100.0, 90.0, 20), index=idx)
    fwd = N.compute_forward_return(pd.DataFrame({"CLOSE": close}), 5)["CLOSE"]
    state, stats = N.compute_divergence_signal(dix, close, fwd, lookback=5)
    # bullish days = index 5..19 (15), but the last 5 have NaN forward returns
    defined_bull = int(((state == 1.0) & fwd.notna()).sum())
    assert stats["bull"]["n"] == defined_bull
    assert stats["bull"]["n"] == 10


# ===========================================================================
# finra_dpi_to_d
# ===========================================================================
def test_dpi_to_d_ratio_clip_and_zero_denominator():
    idx = _idx(6)
    short = pd.DataFrame({"A": [5.0, 10.0, 0.0, 15.0, 20.0, 25.0]}, index=idx)
    total = pd.DataFrame({"A": [10.0, 10.0, 0.0, 10.0, 10.0, 10.0]}, index=idx)
    dpi, d = N.finra_dpi_to_d(short, total, smooth=2)
    # 0.5, 1.0, (0/0 -> NaN via replace(0,NaN)), 1.5->clip 1.0, 1.0, 1.0
    got = dpi["A"].tolist()
    assert got[0] == pytest.approx(0.5)
    assert got[1] == pytest.approx(1.0)
    assert np.isnan(got[2])
    assert got[3] == pytest.approx(1.0)  # clipped from 1.5


def test_dpi_to_d_smoothing_is_min_periods_one_moving_average():
    idx = _idx(4)
    short = pd.DataFrame({"A": [2.0, 8.0, 5.0, 5.0]}, index=idx)
    total = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=idx)
    dpi, d = N.finra_dpi_to_d(short, total, smooth=2)
    assert dpi["A"].tolist() == pytest.approx([0.2, 0.8, 0.5, 0.5])
    # 2-day trailing mean, first point seeded by min_periods=1.
    assert d["A"].tolist() == pytest.approx([0.2, 0.5, 0.65, 0.5])


def test_dpi_to_d_returns_dpi_and_smoothed_pair():
    idx = _idx(3)
    short = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    total = pd.DataFrame({"A": [4.0, 4.0, 4.0]}, index=idx)
    dpi, d = N.finra_dpi_to_d(short, total, smooth=1)
    # smooth=1 -> d equals dpi.
    pd.testing.assert_frame_equal(dpi, d)


# ===========================================================================
# compute_aggregate_dark_ratio
# ===========================================================================
def test_aggregate_dark_ratio_is_volume_weighted_sum():
    idx = _idx(2)
    fx = pd.DataFrame({"A": [4.0, 4.0], "B": [6.0, 6.0], "C": [2.0, 2.0]}, index=idx)
    vol = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0], "C": [10.0, 10.0]}, index=idx)
    agg = N.compute_aggregate_dark_ratio(fx, vol, min_names=3)
    # sum(off-exch)=12 / sum(vol)=30 = 0.4 (a sum ratio, not a mean of ratios).
    assert agg.tolist() == pytest.approx([0.4, 0.4])


def test_aggregate_dark_ratio_min_names_guard():
    idx = _idx(2)
    fx = pd.DataFrame({"A": [4.0, 4.0], "B": [6.0, 6.0], "C": [2.0, 2.0]}, index=idx)
    vol = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0], "C": [10.0, 10.0]}, index=idx)
    agg = N.compute_aggregate_dark_ratio(fx, vol, min_names=4)
    assert agg.isna().all()  # only 3 contributors, below the floor


def test_aggregate_dark_ratio_excludes_bench_and_masks_missing():
    idx = _idx(2)
    fx = pd.DataFrame({"A": [4.0, 4.0], "B": [6.0, 6.0], "C": [2.0, np.nan]}, index=idx)
    vol = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0], "C": [10.0, 10.0]}, index=idx)
    agg = N.compute_aggregate_dark_ratio(fx, vol, exclude=("B",), min_names=1)
    # Excludes B; on day 2, C has no FINRA value so only A counts: 4/10 = 0.4.
    assert agg.iloc[0] == pytest.approx((4.0 + 2.0) / (10.0 + 10.0))  # day1: A,C
    assert agg.iloc[1] == pytest.approx(4.0 / 10.0)                   # day2: A only


def test_aggregate_dark_ratio_clipped_to_unit_interval():
    idx = _idx(2)
    fx = pd.DataFrame({"A": [50.0, 50.0], "B": [50.0, 50.0]}, index=idx)  # off-exch > total
    vol = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=idx)
    agg = N.compute_aggregate_dark_ratio(fx, vol, min_names=2)
    assert (agg <= 1.0).all()


# ===========================================================================
# compute_dollar_dix
# ===========================================================================
def test_dollar_dix_dollar_weighted_value():
    idx = _idx(2)
    sh = pd.DataFrame({"A": [5.0, 5.0], "B": [3.0, 3.0]}, index=idx)
    tv = pd.DataFrame({"A": [10.0, 10.0], "B": [10.0, 10.0]}, index=idx)
    px = pd.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0]}, index=idx)
    dix = N.compute_dollar_dix(sh, tv, px, min_names=2, min_coverage=0.0)
    # sum(px*short)=5*100+3*50=650 ; sum(px*total)=10*100+10*50=1500 -> 0.4333
    assert dix.tolist() == pytest.approx([650.0 / 1500.0, 650.0 / 1500.0])


def test_dollar_dix_min_names_floor():
    idx = _idx(2)
    sh = pd.DataFrame({"A": [5.0, 5.0]}, index=idx)
    tv = pd.DataFrame({"A": [10.0, 10.0]}, index=idx)
    px = pd.DataFrame({"A": [100.0, 100.0]}, index=idx)
    dix = N.compute_dollar_dix(sh, tv, px, min_names=2, min_coverage=0.0)
    assert dix.isna().all()  # only 1 contributor < min_names=2


def test_dollar_dix_coverage_guard_drops_thin_day():
    # A long run of full days sets a high "typical" contributor count; a final
    # day covered by only a few names is dropped even though it clears min_names.
    n_days, n_names = 15, 12
    idx = _idx(n_days)
    cols = [f"S{i}" for i in range(n_names)]
    sh = pd.DataFrame(1.0, index=idx, columns=cols)
    tv = pd.DataFrame(4.0, index=idx, columns=cols)
    px = pd.DataFrame(10.0, index=idx, columns=cols)
    # Last day: knock out all but 3 names (blank price -> not a contributor).
    px.iloc[-1, 3:] = np.nan
    dix = N.compute_dollar_dix(sh, tv, px, min_names=3, min_coverage=0.6)
    assert not np.isnan(dix.iloc[0])          # full early day survives
    assert dix.iloc[0] == pytest.approx(0.25)  # 1/4
    assert np.isnan(dix.iloc[-1])             # 3 of ~12 typical -> below coverage


# ===========================================================================
# compute_residuals
# ===========================================================================
def test_residuals_diff_and_regression_and_names():
    idx = _idx(60)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.uniform(0.3, 0.6, 60), index=idx)
    panel = pd.DataFrame({"BENCH": b, "X": 0.1 + 0.5 * b, "Y": b + 0.05})
    res = N.compute_residuals(panel, "BENCH", window=20, min_periods=10)
    assert set(res) == {"diff", "reg", "raw", "bench"}
    # The benchmark itself is not among the residualised names.
    assert "BENCH" not in res["diff"].columns
    assert list(res["raw"].columns) == ["X", "Y"]
    # diff is a plain level difference.
    assert res["diff"]["Y"].iloc[-1] == pytest.approx(0.05)
    # X is a perfect linear function of the benchmark -> OLS residual ~ 0.
    assert res["reg"]["X"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_residuals_missing_benchmark_raises():
    idx = _idx(30)
    panel = pd.DataFrame({"X": np.arange(30.0), "Y": np.arange(30.0)}, index=idx)
    with pytest.raises(ValueError, match="benchmark BENCH not present"):
        N.compute_residuals(panel, "BENCH")


def test_residuals_bench_series_not_required_in_panel():
    idx = _idx(40)
    rng = np.random.default_rng(1)
    b = pd.Series(rng.uniform(0.3, 0.6, 40), index=idx)
    panel = pd.DataFrame({"X": 0.2 + b, "Y": 0.3 + b})  # no BENCH column
    res = N.compute_residuals(panel, "BENCH", window=15, min_periods=8, bench_series=b * 2)
    # diff residualises against the supplied series, not a panel column.
    assert res["diff"]["X"].iloc[-1] == pytest.approx((panel["X"] - b * 2).iloc[-1])


# ===========================================================================
# merge_share_classes  (GOOG folded into GOOGL)
# ===========================================================================
def _dualclass_panels():
    idx = _idx(2)
    short = pd.DataFrame({"GOOGL": [2.0, 4.0], "GOOG": [1.0, 1.0], "AAPL": [3.0, 3.0]}, index=idx)
    total = pd.DataFrame({"GOOGL": [10.0, 10.0], "GOOG": [5.0, 5.0], "AAPL": [10.0, 10.0]}, index=idx)
    dpi, d = N.finra_dpi_to_d(short, total)
    one = pd.DataFrame({"GOOGL": [1.0, 1.0], "GOOG": [1.0, 1.0], "AAPL": [1.0, 1.0]}, index=idx)
    panels = {"short": short.copy(), "total": total.copy(), "dpi": dpi.copy(), "d": d.copy(),
              "close": one.copy(), "adjclose": one.copy(), "volume": one.copy()}
    earn = pd.DataFrame({"ticker": ["GOOGL", "GOOG", "AAPL"],
                         "report_date": pd.to_datetime(["2022-01-03"] * 3)})
    return panels, earn


def test_merge_share_classes_sums_and_rederives_dpi():
    panels, earn = _dualclass_panels()
    p, e = E.merge_share_classes(panels, earn)
    # Secondary GOOG folded away everywhere.
    for k in ("short", "total", "dpi", "d", "close", "adjclose", "volume"):
        assert "GOOG" not in p[k].columns
    assert "GOOG" not in set(e["ticker"])
    # Short/total summed across classes.
    assert p["short"]["GOOGL"].tolist() == pytest.approx([3.0, 5.0])
    assert p["total"]["GOOGL"].tolist() == pytest.approx([15.0, 15.0])
    # DPI re-derived volume-weighted from the SUMMED short/total (not an avg of ratios).
    assert p["dpi"]["GOOGL"].tolist() == pytest.approx([3.0 / 15.0, 5.0 / 15.0])
    # Unrelated names are untouched.
    assert p["total"]["AAPL"].tolist() == pytest.approx([10.0, 10.0])


def test_merge_share_classes_noop_when_secondary_absent():
    idx = _idx(2)
    short = pd.DataFrame({"GOOGL": [2.0, 4.0], "AAPL": [3.0, 3.0]}, index=idx)
    total = pd.DataFrame({"GOOGL": [10.0, 10.0], "AAPL": [10.0, 10.0]}, index=idx)
    dpi, d = N.finra_dpi_to_d(short, total)
    panels = {"short": short, "total": total, "dpi": dpi, "d": d}
    earn = pd.DataFrame({"ticker": ["GOOGL", "AAPL"],
                         "report_date": pd.to_datetime(["2022-01-03"] * 2)})
    p, e = E.merge_share_classes(panels, earn)
    assert list(p["total"].columns) == ["GOOGL", "AAPL"]
    assert len(e) == 2


# ===========================================================================
# to_yahoo_symbol
# ===========================================================================
@pytest.mark.parametrize("raw,expected", [
    ("BRK.B", "BRK-B"),
    ("brk.b", "BRK-B"),
    (" aapl ", "AAPL"),
    ("BF.B", "BF-B"),
    ("MSFT", "MSFT"),
])
def test_to_yahoo_symbol(raw, expected):
    assert N.to_yahoo_symbol(raw) == expected


# ===========================================================================
# build_report._terc  (tercile split + Welch test)
# ===========================================================================
def test_terc_buckets_means_and_up_fractions():
    ev = pd.DataFrame({"p": [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95],
                       "r": [-1.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    out = B._terc(ev, "p", "r")
    assert out["hi"] == pytest.approx(3.5)   # p>=2/3 -> r in {2,3,4,5}
    assert out["lo"] == pytest.approx(-1.5)  # p<=1/3 -> r in {-1,-2}
    assert out["mid"] == pytest.approx(0.5)  # 1/3<p<2/3 -> r in {0,1}
    assert out["hi_up"] == pytest.approx(1.0)
    assert out["lo_up"] == pytest.approx(0.0)
    assert (out["nh"], out["nm"], out["nl"]) == (4, 2, 2)


def test_terc_diff_matches_welch_when_samples_sufficient():
    rng = np.random.default_rng(2)
    p = np.concatenate([np.full(6, 0.1), np.full(6, 0.9)])
    r = np.concatenate([rng.normal(-1, 1, 6), rng.normal(2, 1, 6)])
    ev = pd.DataFrame({"p": p, "r": r})
    out = B._terc(ev, "p", "r")
    hi = r[p >= 2 / 3]
    lo = r[p <= 1 / 3]
    diff, t, pval, _, _ = E._welch(hi, lo)
    assert out["diff"] == pytest.approx(diff)
    assert out["t"] == pytest.approx(t)
    assert out["p"] == pytest.approx(pval)
