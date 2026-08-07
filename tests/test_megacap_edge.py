"""Tests for the mega-cap edge machinery.

The study's headline is a null, so these prove the analysis DETECTS a real
within-mega-cap selection edge when one is planted (else a null could be a bug),
and returns flat on noise. Pure-function tests on a synthetic long frame -- no
network.
"""
import numpy as np
import pandas as pd

import spx_megacap_edge_study as M


def _frame(effect, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=200)
    names = [f"N{i}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    dpct = rng.uniform(0, 1, m)
    fwd = rng.normal(0, 3, m) + effect * (dpct >= M.XS.HI)   # planted on high-DPI names
    long = pd.DataFrame({"dpct": dpct, "xrank": rng.uniform(0, 1, m),
                         "trend_63": rng.normal(0, 1, m), "fwd": fwd,
                         "xret": fwd, "spyx": fwd - 1.0}, index=idx)
    weights = {n: 30 - i for i, n in enumerate(names)}
    return long, weights


def test_selection_edge_recovered_when_planted():
    long, weights = _frame(effect=2.0, seed=1)
    tab = M.megacap_edge(long, weights, cutoffs=(10, 20, 30))
    ad = tab[tab["cond"] == "all-days"]
    # planted high-DPI outperformance -> selection strongly positive, CI clears 0
    assert ad["sel_mean"].min() > 1.0
    assert (ad["sel_ci_lo"] > 0).all()
    assert (ad["ls_mean"] > 0).all()


def test_null_frame_selection_flat():
    long, weights = _frame(effect=0.0, seed=2)
    tab = M.megacap_edge(long, weights, cutoffs=(20, 30))
    ad = tab[tab["cond"] == "all-days"]
    # no planted effect -> selection stays small and its CI is not a strong,
    # decisively-positive signal (lower bound near/below zero).
    assert ad["sel_mean"].abs().max() < 0.6
    assert ad["sel_ci_lo"].min() < 0.2


def test_basket_and_longshort_respect_min_size():
    long, weights = _frame(effect=0.0, seed=3)
    sub = long[long.index.get_level_values("name").isin([f"N{i}" for i in range(30)])].copy()
    sub["mc_x"] = sub["fwd"] - sub.groupby(level="date")["fwd"].transform("mean")
    hi = sub["dpct"] >= M.XS.HI
    # a basket with an impossibly large min size yields no days
    assert M.basket_series(sub, hi, "mc_x", min_n=1000) == []
    # a normal basket yields one value per qualifying day (<= number of days)
    days = sub.index.get_level_values("date").nunique()
    assert 0 < len(M.basket_series(sub, hi, "mc_x", min_n=1)) <= days
