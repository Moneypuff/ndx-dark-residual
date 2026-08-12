"""Tests for the trend-reversal study: turning-point detection, the forward
reversal label's NaN edge handling, event-time stacking, and proof the event
curve and probability machinery recover planted DPI-leads-reversal effects.
"""
import numpy as np
import pandas as pd
import pytest

import spx_trend_reversal_study as R


def _v_frame(n=120, low_at=60):
    idx = pd.bdate_range("2020-01-01", periods=n)
    v = np.concatenate([np.linspace(100, 60, low_at + 1),
                        np.linspace(60, 95, n - low_at - 1 + 1)[1:]])
    adj = pd.DataFrame({"A": v}, index=idx)
    trend = pd.DataFrame({"A": np.full(n, -1.0)}, index=idx)
    return adj, trend


def test_find_turns_v_shape_trough():
    adj, trend = _v_frame()
    ev = R.find_turns(adj, trend, kind="trough")
    assert ev == [("A", 60)]
    # peaks need an uptrend; with trend<0 everywhere there are none
    assert R.find_turns(adj, trend, kind="peak") == []


def test_find_turns_deoverlap_keeps_first():
    # W shape: minima at 50 and 80 (gap 30 < MIN_GAP=42) -> only the first kept
    idx = pd.bdate_range("2020-01-01", periods=141)
    v = np.concatenate([np.linspace(100, 70, 51),
                        np.linspace(70, 85, 16)[1:],
                        np.linspace(85, 65, 16)[1:],
                        np.linspace(65, 95, 61)[1:]])
    adj = pd.DataFrame({"A": v}, index=idx)
    trend = pd.DataFrame({"A": np.full(141, -1.0)}, index=idx)
    ev = R.find_turns(adj, trend, kind="trough", min_gap=42)
    assert [i for _, i in ev] == [50]
    ev2 = R.find_turns(adj, trend, kind="trough", min_gap=10)
    assert [i for _, i in ev2] == [50, 80]


def test_turn_within_label_and_nan_edge():
    idx = pd.bdate_range("2020-01-01", periods=100)
    tw = R.turn_within([("A", 50)], idx, ["A"], h=21, wing=21)
    # days 29..49 see the turn within their next 21 sessions; day 50 itself
    # and later do not; day 28 is one session too early
    assert tw["A"].iloc[29] == 1.0 and tw["A"].iloc[49] == 1.0
    assert tw["A"].iloc[28] == 0.0 and tw["A"].iloc[50] == 0.0
    # trailing h+wing sessions are NaN, never "no reversal"
    assert tw["A"].iloc[-(21 + 21):].isna().all()
    assert tw["A"].iloc[-(21 + 21) - 1] == 0.0


def test_event_curve_recovers_planted_ramp():
    # p10 ramps 0.5 -> 0.9 over the 21 sessions into each event, else 0.5
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=600)
    names = [f"N{i}" for i in range(12)]
    p10 = pd.DataFrame(0.5 + rng.normal(0, 0.02, (600, 12)), index=idx, columns=names)
    events = [(nm, 100 + 90 * k) for k, nm in enumerate(names[:5]) for _ in [0]]
    for nm, i in events:
        p10.iloc[i - 21:i + 1, p10.columns.get_loc(nm)] += np.linspace(0, 0.4, 22)
    cur = R.event_curves(p10, events)
    mean, lo, hi = R.curve_boot(cur, [n for n, _ in events], seed=1)
    k0 = R.TAUS.tolist().index(0)
    km = R.TAUS.tolist().index(-42)
    assert mean[k0] > 0.8 and abs(mean[km] - 0.5) < 0.05
    assert lo[k0] > 0.55                       # CI at tau=0 excludes the 0.5 baseline


def _prob_frame(lift, seed=0, n_days=300, n_names=40):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    names = [f"N{i}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "name"])
    m = len(idx)
    p10 = rng.uniform(0, 1, m)
    streak = np.where(p10 >= R.HI, rng.integers(5, 15, m), 0)
    tw = (rng.uniform(0, 1, m) < 0.10 + lift * (p10 >= R.HI)).astype(float)
    return pd.DataFrame({"p10": p10, "streak": streak, "trend63": np.full(m, -1.0),
                         "rback": rng.normal(-3, 4, m), "tw": tw}, index=idx)


def test_prob_tests_detect_planted_lift():
    long = _prob_frame(lift=0.15, seed=2)
    base, rates, hl, st, fm = R.prob_tests(long, split="2019-08-01")
    assert rates["HIGH p10"][0] > rates["LOW p10"][0] + 8
    assert hl["ci"][0] > 0                              # HIGH-LOW clears zero
    assert fm["full"]["ci"][0] > 0                      # survives the rback control


def test_prob_tests_flat_on_null():
    long = _prob_frame(lift=0.0, seed=3)
    base, rates, hl, st, fm = R.prob_tests(long, split="2019-08-01")
    lo, hi = hl["ci"]
    assert lo < 0 < hi
    lo, hi = fm["full"]["ci"]
    assert lo < 0 < hi
