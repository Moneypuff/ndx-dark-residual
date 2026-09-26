"""Tests for ``dark_flow`` -- the relative-darkness measures and the statistics used
to evaluate them (see DARK_FLOW_FINDINGS.md).

The measures are small rolling / cross-sectional transforms whose arithmetic is
exact, so they are pinned with hand-computed panels: volume weighting, the baseline
window never overlapping the signal window, own-sigma scaling that only uses the
past, split-basis ratios, and the per-date cross-sectional algebra (ranks centred on
zero, residuals orthogonal to their regressors). The evaluation helpers are checked
against planted relationships and, for the t-statistic, scipy as an oracle.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import dark_flow as F


def _idx(n, start="2022-01-03"):
    return pd.bdate_range(start, periods=n)


# ===========================================================================
# vw_dpi
# ===========================================================================
def test_vw_dpi_is_sum_over_sum():
    idx = _idx(3)
    short = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    total = pd.DataFrame({"A": [10.0, 10.0, 20.0]}, index=idx)
    got = F.vw_dpi(short, total, window=2)["A"].tolist()
    # min_periods defaults to window-1 = 1: 1/10, (1+2)/(10+10), (2+3)/(10+20)
    assert got == pytest.approx([0.1, 0.15, 5.0 / 30.0])


def test_vw_dpi_weights_heavy_days_unlike_mean_of_ratios():
    idx = _idx(2)
    # a thin day with an extreme ratio next to a heavy normal day
    short = pd.DataFrame({"A": [9.0, 400.0]}, index=idx)
    total = pd.DataFrame({"A": [10.0, 1000.0]}, index=idx)
    vw = F.vw_dpi(short, total, window=2)["A"].iloc[-1]
    mean_of_ratios = (0.9 + 0.4) / 2
    assert vw == pytest.approx(409.0 / 1010.0)
    assert abs(vw - 0.4) < abs(mean_of_ratios - 0.4)


def test_vw_dpi_drops_one_sided_missing_cells_from_both_sums():
    idx = _idx(3)
    short = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=idx)
    total = pd.DataFrame({"A": [10.0, 1000.0, 10.0]}, index=idx)
    # the day-2 total must not enter the denominator when its short is missing
    assert F.vw_dpi(short, total, window=3)["A"].iloc[-1] == pytest.approx(4.0 / 20.0)


def test_vw_dpi_zero_total_is_nan_and_clipped():
    idx = _idx(2)
    short = pd.DataFrame({"A": [0.0, 30.0]}, index=idx)
    total = pd.DataFrame({"A": [0.0, 20.0]}, index=idx)
    got = F.vw_dpi(short, total, window=1)["A"]
    assert np.isnan(got.iloc[0])
    assert got.iloc[1] == 1.0


# ===========================================================================
# abnormal_dpi
# ===========================================================================
def _step_panel(n=60, step_at=40, lo=0.4, hi=0.6, vol=100.0):
    idx = _idx(n)
    ratio = np.where(np.arange(n) < step_at, lo, hi)
    total = pd.DataFrame({"A": vol}, index=idx)
    return total * ratio[:, None], total


def test_abnormal_dpi_zero_when_flow_is_steady():
    short, total = _step_panel(step_at=10_000)          # never steps
    abn, z = F.abnormal_dpi(short, total, window=5, base=20, sd_window=20)
    assert abn["A"].dropna().abs().max() == pytest.approx(0.0)


def test_abnormal_dpi_baseline_excludes_the_signal_window():
    short, total = _step_panel(n=80, step_at=40)
    abn, _ = F.abnormal_dpi(short, total, window=5, base=20, sd_window=20)
    # 5 sessions after the step the window is all 0.6 and the 20-session baseline that ends
    # just before it is all 0.4 -> exactly +0.2, not diluted by the window itself.
    assert abn["A"].iloc[44] == pytest.approx(0.2)
    # at session 59 the baseline (sessions 35..54) straddles the step: 5 x 0.4 + 15 x 0.6
    assert abn["A"].iloc[59] == pytest.approx(0.6 - (5 * 0.4 + 15 * 0.6) / 20)
    # once the baseline has rolled fully past the step (session >= 64) it is steady again
    assert abn["A"].iloc[64] == pytest.approx(0.0)


def test_abnormal_dpi_z_uses_only_past_sigma():
    rng = np.random.default_rng(0)
    idx = _idx(80)
    total = pd.DataFrame({"A": 100.0}, index=idx)
    short = total * pd.DataFrame({"A": rng.uniform(0.3, 0.6, 80)}, index=idx)
    _, z1 = F.abnormal_dpi(short, total, window=5, base=20, sd_window=20)
    short2 = short.copy()
    short2.iloc[60:] = short2.iloc[60:] * 1.5            # rewrite the future only
    _, z2 = F.abnormal_dpi(short2, total, window=5, base=20, sd_window=20)
    pd.testing.assert_series_equal(z1["A"].iloc[:60], z2["A"].iloc[:60])


# ===========================================================================
# dark_share / abnormal_dark_share / dark_imbalance
# ===========================================================================
def test_dark_share_sum_over_sum_and_zero_volume_guard():
    idx = _idx(3)
    total = pd.DataFrame({"A": [40.0, 60.0, 50.0]}, index=idx)
    vol = pd.DataFrame({"A": [100.0, 100.0, 0.0]}, index=idx)
    got = F.dark_share(total, vol, window=2)["A"].tolist()
    # zero-volume day is dropped from BOTH sums (not counted as a 50/0 share)
    assert got == pytest.approx([0.4, 0.5, 0.6])


def test_abnormal_dark_share_is_log_of_share_vs_baseline():
    idx = _idx(30)
    vol = pd.DataFrame({"A": 100.0}, index=idx)
    total = pd.DataFrame({"A": [30.0] * 25 + [60.0] * 5}, index=idx)
    got = F.abnormal_dark_share(total, vol, window=5, base=20)["A"]
    assert got.iloc[-1] == pytest.approx(np.log(2.0))
    assert got.iloc[24] == pytest.approx(0.0)


def test_dark_imbalance_hand_computed():
    idx = _idx(8)
    vol = pd.DataFrame({"A": 1000.0}, index=idx)
    total = pd.DataFrame({"A": 500.0}, index=idx)
    short = pd.DataFrame({"A": [200.0] * 6 + [300.0, 300.0]}, index=idx)   # DPI 0.4 -> 0.6
    got = F.dark_imbalance(short, total, vol, window=2, base=4, min_base=4)["A"]
    # last window: sum short 600, baseline DPI 0.4 x sum total 1000 = 400 -> excess 200;
    # ADV 1000 over 2 sessions -> 200 / 2000 = 0.1 (a tenth of a normal day's volume)
    assert got.iloc[-1] == pytest.approx(0.1)
    assert got.iloc[5] == pytest.approx(0.0)


# ===========================================================================
# cross-sectional helpers
# ===========================================================================
def test_xs_rank_centred_bounded_and_nan_preserving():
    idx = _idx(2)
    df = pd.DataFrame({"a": [1.0, 5.0], "b": [2.0, np.nan], "c": [3.0, 4.0], "d": [4.0, 3.0]}, index=idx)
    r = F.xs_rank(df)
    assert r.loc[idx[0]].sum() == pytest.approx(0.0)
    assert r.loc[idx[1]].dropna().sum() == pytest.approx(0.0)
    assert np.isnan(r.loc[idx[1], "b"])
    assert r.max().max() < 0.5 and r.min().min() > -0.5
    assert list(r.loc[idx[0]].rank()) == [1.0, 2.0, 3.0, 4.0]


def test_xs_residualize_matches_lstsq_and_is_orthogonal():
    rng = np.random.default_rng(1)
    idx, cols = _idx(3), [f"n{i}" for i in range(50)]
    x1 = pd.DataFrame(rng.normal(size=(3, 50)), index=idx, columns=cols)
    x2 = pd.DataFrame(rng.normal(size=(3, 50)), index=idx, columns=cols)
    y = 2 * x1 - x2 + pd.DataFrame(rng.normal(size=(3, 50)), index=idx, columns=cols)
    res = F.xs_residualize(y, [x1, x2], min_names=10)
    for d in idx:
        A = np.column_stack([np.ones(50), x1.loc[d], x2.loc[d]])
        b, *_ = np.linalg.lstsq(A, y.loc[d], rcond=None)
        np.testing.assert_allclose(res.loc[d].to_numpy(), y.loc[d] - A @ b, atol=1e-10)
        assert abs(res.loc[d].sum()) < 1e-9
        assert abs(float(res.loc[d] @ x1.loc[d])) < 1e-8


def test_xs_residualize_skips_thin_dates():
    idx = _idx(1)
    y = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}, index=idx)
    out = F.xs_residualize(y, [y * 2], min_names=10)
    assert out.isna().all().all()


def test_price_echo_split_recombines_and_hidden_is_return_orthogonal():
    rng = np.random.default_rng(2)
    idx, cols = _idx(4), [f"n{i}" for i in range(80)]
    r1 = pd.DataFrame(rng.normal(size=(4, 80)), index=idx, columns=cols)
    rk = pd.DataFrame(rng.normal(size=(4, 80)), index=idx, columns=cols)
    sig = r1 + rk + pd.DataFrame(rng.normal(size=(4, 80)), index=idx, columns=cols)
    echo, hidden = F.price_echo_split(sig, r1, rk, min_names=10)
    pd.testing.assert_frame_equal(echo + hidden, F.xs_rank(sig))
    for d in idx:
        assert abs(np.corrcoef(hidden.loc[d], F.xs_rank(r1).loc[d])[0, 1]) < 1e-8


# ===========================================================================
# evaluation helpers
# ===========================================================================
def test_forward_log_return_lag_semantics():
    idx = _idx(5)
    adj = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=idx)
    f = F.forward_log_return(adj, 2, lag=1)["A"]
    # from the close of t+1 to t+3: log(133.1/110) * 100
    assert f.iloc[0] == pytest.approx(np.log(133.1 / 110.0) * 100)
    assert np.isnan(f.iloc[2])
    f0 = F.forward_log_return(adj, 1, lag=0)["A"]
    assert f0.iloc[0] == pytest.approx(np.log(1.1) * 100)


def test_newey_west_t_zero_lags_matches_scipy_up_to_hc0_factor():
    x = np.random.default_rng(3).normal(0.2, 1.0, 200)
    mu, t, n = F.newey_west_t(x, 0)
    ref = stats.ttest_1samp(x, 0.0).statistic
    assert mu == pytest.approx(x.mean())
    assert n == 200
    # HC0 variance uses 1/n where the classic t uses 1/(n-1)
    assert t == pytest.approx(ref * np.sqrt(200 / 199), rel=1e-9)


def test_newey_west_t_too_short_is_nan():
    mu, t, n = F.newey_west_t([1.0, 2.0, 3.0], 4)
    assert np.isnan(mu) and np.isnan(t) and n == 3


def test_fama_macbeth_recovers_planted_slopes():
    rng = np.random.default_rng(4)
    idx, cols = _idx(40), [f"n{i}" for i in range(60)]
    x1 = pd.DataFrame(rng.normal(size=(40, 60)), index=idx, columns=cols)
    x2 = pd.DataFrame(rng.normal(size=(40, 60)), index=idx, columns=cols)
    market = pd.Series(rng.normal(size=40), index=idx)
    y = (0.5 * x1 - 0.25 * x2).add(market, axis=0)               # per-date level is absorbed
    res, slopes = F.fama_macbeth(y, {"x1": x1, "x2": x2}, idx, min_names=20, nw_lags=2)
    assert res.loc["x1", "coef"] == pytest.approx(0.5)
    assert res.loc["x2", "coef"] == pytest.approx(-0.25)
    assert res.loc["x1", "n_dates"] == 40
    np.testing.assert_allclose(slopes["x1"].to_numpy(), 0.5)


def test_fama_macbeth_skips_dates_below_min_names():
    idx, cols = _idx(3), list("abcde")
    x = pd.DataFrame(np.arange(15.0).reshape(3, 5), index=idx, columns=cols)
    res, slopes = F.fama_macbeth(2 * x, {"x": x}, idx, min_names=10)
    assert slopes.empty and res.loc["x", "n_dates"] == 0


def test_quantile_spread_top_minus_bottom_quintile():
    idx, cols = _idx(1), [f"n{i}" for i in range(10)]
    sig = pd.DataFrame([np.arange(10.0)], index=idx, columns=cols)
    y = pd.DataFrame([np.arange(10.0) * 2], index=idx, columns=cols)
    got = F.quantile_spread(sig, y, idx, q=5, min_names=5)
    # top quintile {8, 9} -> y mean 17; bottom {0, 1} -> 1
    assert got.iloc[0] == pytest.approx(16.0)


def test_decompose_dpi_attributes_planted_components():
    rng = np.random.default_rng(5)
    idx, n = _idx(400), 200
    mu = rng.normal(0.5, 0.05, n)
    delta = rng.normal(0.0, 0.02, 400)
    noise = rng.normal(0.0, 0.03, (400, n))
    dpi = pd.DataFrame(mu[None, :] + delta[:, None] + noise, index=idx)
    out = F.decompose_dpi(dpi)
    tot = 0.05 ** 2 + 0.02 ** 2 + 0.03 ** 2
    assert out["share_structural"] == pytest.approx(0.05 ** 2 / tot, abs=0.03)
    assert out["share_common"] == pytest.approx(0.02 ** 2 / tot, abs=0.02)
    assert out["share_idio"] == pytest.approx(0.03 ** 2 / tot, abs=0.02)
    assert abs(out["acf"][1]) < 0.02              # iid idiosyncratic noise has no memory


def test_echo_profile_detects_same_day_echo_only():
    rng = np.random.default_rng(6)
    idx, cols = _idx(60), [f"n{i}" for i in range(50)]
    ret = pd.DataFrame(rng.normal(size=(60, 50)), index=idx, columns=cols)
    prof = F.echo_profile(ret, ret, lags=(-1, 0, 1), min_names=10)
    assert prof[0] == pytest.approx(1.0)
    assert abs(prof[1]) < 0.05 and abs(prof[-1]) < 0.05


def test_index_predictability_shapes_and_planted_signal():
    rng = np.random.default_rng(7)
    idx = _idx(900)
    g = pd.Series(np.clip(0.45 + np.cumsum(rng.normal(0, 0.002, 900)), 0.3, 0.6), index=idx)
    ret = 0.8 * (g - g.mean()) / g.std() * 0.01 + rng.normal(0, 0.01, 900)   # future return tracks g
    price = pd.Series(100 * np.exp(np.cumsum(np.r_[0.0, ret[:-1]])), index=idx)
    R = F.index_predictability(g, price, horizons=(5,))
    assert set(R["signal"]) == {"level", "detrended"} and set(R["controls"]) == {False, True}
    lvl = R[(R.signal == "level") & ~R.controls].iloc[0]
    assert lvl["coef_pp_per_sd"] > 0 and lvl["t"] > 3
    assert {"t_rv21", "t_past21"} <= set(R.columns)


# ===========================================================================
# per-name heterogeneity
# ===========================================================================
def test_nw_slope_is_ols_on_standardized_x_with_hc_t():
    rng = np.random.default_rng(10)
    x = rng.normal(2.0, 3.0, 400)
    y = 0.7 * (x - x.mean()) / x.std() + rng.normal(0, 1, 400)
    b, t, n = F.nw_slope(x, y, lags=0)
    xs = (x - x.mean()) / x.std()
    assert b == pytest.approx(np.polyfit(xs, y, 1)[0])
    u = (y - y.mean()) - b * xs
    assert t == pytest.approx(b / (np.sqrt(np.sum((xs * u) ** 2)) / np.sum(xs * xs)))
    assert n == 400 and t > 10


def test_nw_slope_guards_short_constant_and_nan():
    assert np.isnan(F.nw_slope(np.arange(10.0), np.arange(10.0), 1)[1])
    assert np.isnan(F.nw_slope(np.ones(100), np.arange(100.0), 1)[1])
    x = np.r_[np.arange(50.0), [np.nan] * 5]
    y = np.r_[np.arange(50.0), np.arange(5.0)]
    assert F.nw_slope(x, y, 1)[2] == 50


def test_per_name_slopes_signs_and_min_obs():
    rng = np.random.default_rng(11)
    idx = _idx(300)
    x = pd.DataFrame(rng.normal(size=(300, 3)), index=idx, columns=["UP", "FLAT", "SHORT"])
    y = pd.DataFrame({"UP": 0.5 * x["UP"] + rng.normal(0, 1, 300),
                      "FLAT": rng.normal(0, 1, 300),
                      "SHORT": x["SHORT"]}, index=idx)
    x.iloc[:250, 2] = np.nan                                  # only 50 obs -> excluded
    out = F.per_name_slopes(x, y, lags=1, min_obs=100)
    assert set(out.index) == {"UP", "FLAT"}
    assert out.loc["UP", "t"] > 4 and abs(out.loc["FLAT", "t"]) < 3


def test_shift_names_rotates_each_column_independently():
    idx = _idx(100)
    X = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2}, index=idx)
    Xs = F.shift_names(X, np.random.default_rng(0), min_shift=10)
    for c in X:
        assert sorted(Xs[c]) == sorted(X[c])                   # same values, rotated
        k = next(s for s in range(100) if np.array_equal(np.roll(X[c].to_numpy(), s), Xs[c].to_numpy()))
        assert 10 <= k <= 90                                   # at least min_shift from alignment
    with pytest.raises(ValueError):
        F.shift_names(X.iloc[:15], np.random.default_rng(0), min_shift=10)


def test_placebo_slope_t_is_centred_even_when_real_relation_exists():
    rng = np.random.default_rng(12)
    idx, cols = _idx(400), [f"n{i}" for i in range(20)]
    X = pd.DataFrame(rng.normal(size=(400, 20)), index=idx, columns=cols)
    Y = 0.4 * X + pd.DataFrame(rng.normal(size=(400, 20)), index=idx, columns=cols)
    real = F.per_name_slopes(X, Y, lags=1)["t"]
    null = F.placebo_slope_t(X, Y, lags=1, k=5, seed=1)
    assert real.mean() > 5
    assert abs(null.mean()) < 0.5 and 0.6 < null.std() < 1.5
    assert len(null) == 5 * 20


def test_selection_pnl_rewards_persistent_name_directions_only():
    rng = np.random.default_rng(13)
    idx, n = _idx(600), 40
    cols = [f"n{i}" for i in range(n)]
    X = pd.DataFrame(rng.normal(size=(600, n)), index=idx, columns=cols)
    direction = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)      # half the names up, half down
    Y = X * (0.5 * direction) + pd.DataFrame(rng.normal(size=(600, n)), index=idx, columns=cols)
    est, test = np.arange(600) < 300, np.arange(600) >= 300
    pnl, nsel = F.selection_pnl(X, Y, est, test, lags=1)
    assert nsel == n and pnl.mean() > 0.3                      # own direction carries over
    pooled, _ = F.selection_pnl(X, Y, est, test, lags=1, orient="pooled")
    assert abs(pooled.mean()) < 0.1                            # one common direction does not
    timing, _ = F.selection_pnl(X, Y, est, test, lags=1, demean=True)
    assert timing.mean() > 0.3
    noise = pd.DataFrame(rng.normal(size=(600, n)), index=idx, columns=cols)
    p0, _ = F.selection_pnl(X, noise, est, test, lags=1, sel_t=0.0)
    assert abs(p0.mean()) < 0.1
    none, k = F.selection_pnl(X, noise, est, test, lags=1, sel_t=99)
    assert none.empty and k == 0


# ===========================================================================
# index_mechanism
# ===========================================================================
def _vol_driven_market(n=2500, seed=21):
    """A market whose forward returns depend only on a volatility state; the dark gauge merely
    co-moves with that state (plus noise and a slow upward drift) and lags returns by a day."""
    rng = np.random.default_rng(seed)
    idx = _idx(n, start="2012-01-02")
    v = np.empty(n)
    v[0] = 0.0
    for i in range(1, n):
        v[i] = 0.98 * v[i - 1] + rng.normal(0, 0.2)
    iv = pd.Series(18 + 4 * v, index=idx)                          # implied vol tracks the state
    ret = 0.05 + 0.06 * v + rng.normal(0, 1.0, n)                   # expected return rises with vol
    r = pd.Series(ret, index=idx)
    gauge = pd.Series(0.40 + np.linspace(0, 0.05, n) + 0.01 * v + 0.004 * r.shift(1).fillna(0)
                      + rng.normal(0, 0.004, n), index=idx)
    price = pd.Series(100 * np.exp(np.cumsum(ret) / 100), index=idx)
    return gauge, price, iv


def test_index_mechanism_implied_vol_absorbs_a_vol_proxy():
    gauge, price, iv = _vol_driven_market()
    M = F.index_mechanism(gauge, price, iv, horizons=(21,))
    c = M["coef"].set_index("spec")
    assert c.loc["gauge alone", "coef"] > 0
    assert c.loc["implied vol alone", "t"] > c.loc["gauge alone", "t"]
    assert abs(c.loc["gauge + implied vol", "coef"]) < 0.5 * c.loc["gauge alone", "coef"]


def test_index_mechanism_table_and_leadlag():
    gauge, price, iv = _vol_driven_market()
    M = F.index_mechanism(gauge, price, iv, horizons=(21,))
    assert list(M["table"].index) == ["low", "mid", "high"]
    assert list(M["table"].columns) == ["low", "mid", "high"]
    assert int(M["counts"].to_numpy().sum()) > 1500
    ll = M["leadlag"]
    assert ll[-1] == max(ll.values())                      # the gauge follows yesterday's return
    assert abs(ll[1]) < 0.1                                # and says nothing about tomorrow's
    assert M["corr_iv"] > 0.3                              # the planted gauge co-moves with vol
    loo = M["loo"]
    assert list(loo.columns) == ["coef", "t"]
    assert set(loo.index) == set(gauge.index.year)         # one re-estimate per calendar year
