"""Tests for the DPI-persistence study: run-length, streak gradient, and the
per-name conditional. Guards the streak mechanics (the headline is a streak
effect, so a mis-counted streak would bias it) and proves the analyses recover a
planted streak->return relationship. Pure functions on synthetic frames.
"""
import numpy as np
import pandas as pd

import spx_dpi_persistence_study as P


def test_run_length_counts_consecutive_true():
    idx = pd.RangeIndex(7)
    b = pd.DataFrame({"A": [False, True, True, True, False, True, True],
                      "B": [True, False, True, True, True, True, False]}, index=idx)
    rl = P.run_length(b)
    assert rl["A"].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert rl["B"].tolist() == [1, 0, 1, 2, 3, 4, 0]


def _planted(streak_effect, seed=0, n_names=30, n_days=400):
    """Synthetic long frame where, among downtrend rows, a LONGER high-DPI streak
    adds proportionally more to the outcome. Bypasses build_long so we control
    streak/trend directly."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    names = [f"N{i}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    streak = rng.integers(0, 15, m)
    trend = rng.normal(0, 1, m)
    is_down = trend < 0
    base = rng.normal(0, 3, m)
    bump = streak_effect * (streak / 10.0) * is_down       # longer streak -> more, only on downtrend
    fwd = base + bump
    p10 = np.clip(0.3 + streak / 20.0 + rng.normal(0, 0.1, m), 0, 1)   # correlated with streak
    return pd.DataFrame({"p1": p10, "p10": p10, "streak": streak, "trend63": trend,
                         "fwd": fwd, "xret": fwd, "spyx": fwd, "alpha": fwd}, index=idx)


def test_streak_gradient_recovers_planted_dose_response():
    long = _planted(streak_effect=2.0, seed=1)
    cells = {c["streak"]: c for c in P.streak_gradient(long, "alpha")}
    # monotone increasing across buckets, and 10+ clears zero
    assert cells["0"]["mean"] < cells["5-9"]["mean"] < cells["10+"]["mean"]
    assert cells["10+"]["ci"][0] > 0


def test_streak_gradient_flat_on_null():
    long = _planted(streak_effect=0.0, seed=2)
    cells = {c["streak"]: c for c in P.streak_gradient(long, "alpha")}
    assert abs(cells["10+"]["mean"]) < 0.6
    assert cells["10+"]["ci"][0] < 0 < cells["10+"]["ci"][1]        # CI straddles zero


def test_per_name_conditional_runs_and_bounds():
    long = _planted(streak_effect=2.0, seed=3)
    r = P.per_name_conditional(long, "p10", "alpha")
    assert r["names"] > 0
    assert 0.0 <= r["pos_pct"] <= 100.0
    assert 0.0 <= r["sign_p"] <= 1.0
