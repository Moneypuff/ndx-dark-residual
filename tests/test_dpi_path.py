"""Tests for the DPI path study: forward path stats (MAE/MFE/timing) against
hand-computed values, NaN handling, and proof the group/diff machinery detects
a planted 'high DPI -> friendlier path' effect (so a null is real, not a bug).
"""
import numpy as np
import pandas as pd
import pytest

import spx_dpi_path_study as PT


def test_forward_path_stats_hand_computed():
    # prices 100, 90, 110, 80, 120, horizon 3:
    #  t=0 window {90,110,80}: MAE=-20%, MFE=+10%, trough at day 3, peak at day 2
    #  t=1 window {110,80,120}: MAE=80/90-1=-11.11%, MFE=120/90-1=+33.33%, trough day 2, peak day 3
    idx = pd.bdate_range("2019-01-02", periods=5)
    adj = pd.DataFrame({"A": [100.0, 90.0, 110.0, 80.0, 120.0]}, index=idx)
    ps = PT.forward_path_stats(adj, 3)
    assert ps["mae"]["A"].iloc[0] == pytest.approx(-20.0)
    assert ps["mfe"]["A"].iloc[0] == pytest.approx(10.0)
    assert ps["tmin"]["A"].iloc[0] == 3 and ps["tmax"]["A"].iloc[0] == 2
    assert ps["mae"]["A"].iloc[1] == pytest.approx(-11.1111, abs=1e-3)
    assert ps["mfe"]["A"].iloc[1] == pytest.approx(33.3333, abs=1e-3)
    assert ps["tmin"]["A"].iloc[1] == 2 and ps["tmax"]["A"].iloc[1] == 3
    # last 3 rows have no complete forward window -> NaN
    assert ps["mae"]["A"].iloc[2:].isna().all()


def test_forward_path_stats_nan_propagates():
    idx = pd.bdate_range("2019-01-02", periods=6)
    adj = pd.DataFrame({"A": [100.0, np.nan, 110.0, 80.0, 120.0, 90.0]}, index=idx)
    ps = PT.forward_path_stats(adj, 3)
    # window of t=0 contains the NaN -> all stats NaN, including timings
    for k in ("mae", "mfe", "tmin", "tmax"):
        assert np.isnan(ps[k]["A"].iloc[0])
    # t=2 window {80,120,90} is clean -> defined
    assert ps["mae"]["A"].iloc[2] == pytest.approx(80 / 110 * 100 - 100, abs=1e-3)


def _frame(hi_tilt, seed=0, n_names=40, n_days=200):
    """Downtrend panel; HIGH-p10 names optionally get a friendlier path (bigger
    MFE, shallower MAE, earlier peak). streak correlated with p10."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    names = [f"N{i}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    p10 = rng.uniform(0, 1, m)
    hi = p10 >= PT.HIQ
    mae = rng.normal(-6, 2, m) + np.where(hi, hi_tilt / 2.0, 0.0)
    mfe = rng.normal(6, 2, m) + np.where(hi, hi_tilt / 2.0, 0.0)
    streak = np.where(hi, rng.integers(5, 15, m), rng.integers(0, 3, m))
    return pd.DataFrame({
        "p10": p10, "streak": streak, "trend63": np.full(m, -1.0),
        "mae": mae, "mfe": mfe, "tilt": mae + mfe,
        "tmin": rng.integers(1, 21, m).astype(float),
        "tmax": np.where(hi, rng.integers(1, 10, m), rng.integers(10, 21, m)).astype(float),
    }, index=idx)


def test_detects_planted_friendly_path_for_high_dpi():
    long = _frame(hi_tilt=4.0, seed=1)
    d = PT.path_diffs(long)
    assert d["HIGH-LOW"]["tilt"]["ci"][0] > 0        # friendlier path detected
    assert d["HIGH-LOW"]["mae"]["ci"][0] > 0         # shallower drawdown detected
    prof = {r["group"]: r for r in PT.path_rows(long)}
    assert prof["HIGH"]["tmax"]["mean"] < prof["LOW"]["tmax"]["mean"] - 5  # earlier peak
    assert d["S10+-S0"]["tilt"]["mean"] > 1          # streak proxy inherits the effect


def test_flat_when_no_planted_difference():
    long = _frame(hi_tilt=0.0, seed=2)
    d = PT.path_diffs(long)
    for metric in ("tilt", "mae", "mfe"):
        lo, hi = d["HIGH-LOW"][metric]["ci"]
        assert lo < 0 < hi                           # CI straddles zero on a null


def test_path_reversal_control_isolates_streak_effect():
    # mfe = -0.1*rback + 1.5*streak10 + noise; the FM streak10 coef on mfe must
    # recover ~1.5 even though rback correlates with the outcome, while the mae
    # coef (no planted effect) must straddle zero.
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2019-01-02", periods=250)
    names = [f"N{i}" for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    rback = rng.normal(-4, 5, m)
    streak10 = (rng.uniform(0, 1, m) < 0.25).astype(float)
    mfe = -0.1 * rback + 1.5 * streak10 + rng.normal(6, 2, m)
    mae = -0.1 * rback + rng.normal(-6, 2, m)
    long = pd.DataFrame({"trend63": np.full(m, -1.0), "rback": rback,
                         "streak10": streak10, "mae": mae, "mfe": mfe,
                         "tilt": mae + mfe}, index=idx)
    rc = PT.path_reversal_control(long, split="2019-07-01")
    assert abs(rc["mfe"]["full"]["mean"] - 1.5) < 0.3
    assert rc["mfe"]["full"]["ci"][0] > 0
    lo, hi = rc["mae"]["full"]["ci"]
    assert lo < 0 < hi
