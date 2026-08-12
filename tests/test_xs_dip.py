"""Tests for the cross-sectional 'buy the dip on high DIX' study.

The study's headline is a NULL (high per-name dark flow does not improve
dip-buying), so these tests exist mainly to prove the machinery can DETECT a
real effect when one is present -- otherwise a null could just be a bug. We
plant a large signal in a synthetic (name, day) frame and assert every analysis
recovers it, then assert a pure-noise frame comes back flat. Plus small unit
checks on the pure helpers.
"""
import numpy as np
import pandas as pd
import pytest

import spx_xs_dip_dix_study as X


def _planted_frame(n_names=40, n_days=200, effect=3.0, seed=0):
    """A long (date,name) frame where, among DIPPED rows, a high self-relative D
    percentile adds `effect` to the outcome -- and nothing elsewhere."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    dpct = rng.uniform(0, 1, m)
    xrank = rng.uniform(0, 1, m)
    dip = rng.normal(0, 5, m)                       # ~half negative (dips)
    noise = rng.normal(0, 1, m)
    is_dip = dip < 0
    is_hi = dpct >= X.HI
    xret = noise + effect * (is_dip & is_hi)        # planted only on high-D dips
    return pd.DataFrame({"dpct": dpct, "xrank": xrank, "dip": dip,
                         "fwd": xret, "xret": xret, "spyx": xret}, index=idx)


def test_planted_signal_is_recovered_end_to_end():
    long = _planted_frame(effect=3.0, seed=1)
    sel = long["dip"] < 0                           # the "downtrend" selector
    # 2x2 interaction: high-D pays much more in a downtrend than outside
    tab, inter = X.interaction_2x2(long, sel, "xret", "dpct")
    assert inter > 1.5
    hd = tab.set_index("cell").loc["highD & downtrend", "mean"]
    ld = tab.set_index("cell").loc["lowD & downtrend", "mean"]
    assert hd - ld > 1.5
    # decile within downtrend: monotone-ish, strong positive long-short
    g, ls = X.decile_within_dips(long, sel, "xret", "dpct")
    assert ls > 1.5
    assert g.loc[g.index.max(), "mean"] > g.loc[g.index.min(), "mean"]
    # daily portfolio: long-short mean strongly positive, CI clears zero
    dp = X.daily_portfolio(long, sel, "xret", "dpct")
    assert dp["long_short"]["mean"] > 1.5
    assert dp["long_short"]["ci"][0] > 0
    # 'no matter the stock': the planted effect is universal
    pn = X.per_name_edge(long, sel, "xret", "dpct")
    assert pn["pos_frac"] > 90
    assert pn["sign_p"] < 0.01


def test_null_frame_comes_back_flat():
    long = _planted_frame(effect=0.0, seed=2)      # outcome is pure noise
    sel = long["dip"] < 0
    _, inter = X.interaction_2x2(long, sel, "xret", "dpct")
    _, ls = X.decile_within_dips(long, sel, "xret", "dpct")
    dp = X.daily_portfolio(long, sel, "xret", "dpct")
    assert abs(inter) < 0.4
    assert abs(ls) < 0.4
    lo, hi = dp["long_short"]["ci"]
    assert lo < 0 < hi                              # CI straddles zero


def test_trend_slope_sign_and_lookahead():
    # a strictly rising then falling log-price -> slope > 0 then < 0; the first
    # (win-1) rows are NaN (no full trailing window yet).
    idx = pd.bdate_range("2019-01-02", periods=60)
    up = np.linspace(0, 1, 30)
    down = np.linspace(1, 0, 30)
    adj = pd.DataFrame({"A": np.exp(np.concatenate([up, down]))}, index=idx)
    slope = X.trend_slope(adj, win=10)
    assert slope["A"].iloc[:9].isna().all()
    assert slope["A"].iloc[15] > 0                 # inside the rising leg
    assert slope["A"].iloc[-1] < 0                 # inside the falling leg


def test_self_percentile_no_lookahead_and_bounds():
    # a strictly increasing series -> today is always the max of its trailing
    # window -> percentile 1.0 once min_obs is reached; NaN before.
    idx = pd.bdate_range("2019-01-02", periods=40)
    D = pd.DataFrame({"A": np.arange(40.0)}, index=idx)
    p = X.self_percentile(D, win=10, min_obs=5)
    assert p["A"].iloc[:4].isna().all()
    assert p["A"].iloc[10:].max() <= 1.0 and p["A"].iloc[10:].min() >= 0.0
    assert p["A"].iloc[-1] == pytest.approx(1.0)


def test_sign_test_symmetric_and_extremes():
    assert X.sign_test_p(50, 100) == pytest.approx(1.0, abs=1e-6)   # exactly half
    assert X.sign_test_p(100, 100) < 1e-6                            # all positive
    assert 0.0 <= X.sign_test_p(60, 100) <= 1.0
