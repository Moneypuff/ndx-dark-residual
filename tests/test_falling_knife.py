"""Tests for the falling-knife study: group masks, loss-profile daily stats, and
HIGH-LOW / KEEP-ALL differences. Proves the analysis DETECTS a fatter left tail
in a low-DPI group when one is planted (so the study's null is real, not a bug).
"""
import numpy as np
import pandas as pd

import spx_falling_knife_study as F


def _frame(low_extra_loss, seed=0, n_names=40, n_days=200):
    """Downtrend panel where LOW-p10 names optionally suffer extra big losses."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    names = [f"N{i}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    p10 = rng.uniform(0, 1, m)
    fwd = rng.normal(1, 8, m) + np.where(p10 <= F.LOWQ, rng.normal(low_extra_loss, 3, m), 0.0)
    return pd.DataFrame({"p10": p10, "trend63": np.full(m, -1.0),
                         "rback": rng.normal(-3, 4, m), "fwd": fwd, "alpha": fwd}, index=idx)


def test_group_masks_partition_downtrend():
    long = _frame(0.0, seed=1)
    m = F.group_masks(long)
    down = long["trend63"] < 0
    # LOW + MID + HIGH partition the downtrend set; KEEP = ALL minus LOW
    assert int((m["LOW"] | m["MID"] | m["HIGH"]).sum()) == int(down.sum())
    assert int((m["LOW"] & m["HIGH"]).sum()) == 0
    assert int(m["KEEP"].sum()) == int(down.sum()) - int(m["LOW"].sum())


def test_detects_planted_fat_left_tail_in_low_dpi():
    long = _frame(low_extra_loss=-5.0, seed=2)      # LOW names get big extra losses
    prof = {r["group"]: r for r in F.profile(long, "fwd")}
    assert prof["LOW"]["loss10"]["mean"] > prof["HIGH"]["loss10"]["mean"] + 3   # LOW worse
    assert prof["LOW"]["hit"]["mean"] < prof["HIGH"]["hit"]["mean"]
    d = F.diffs(long, "fwd")
    assert d["HIGH-LOW"]["loss10"]["ci"][1] < 0     # HIGH-LOW big-loss diff clearly negative
    assert d["KEEP-ALL"]["hit"]["mean"] > 0         # filtering LOW lifts hit rate


def test_flat_when_no_planted_difference():
    long = _frame(low_extra_loss=0.0, seed=3)
    d = F.diffs(long, "fwd")
    lo, hi = d["HIGH-LOW"]["loss10"]["ci"]
    assert lo < 0 < hi                              # no difference -> CI straddles zero
