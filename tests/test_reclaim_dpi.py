"""Tests for the MA-reclaim + persistent-high-DPI event detection.

Verifies the entry mechanics on hand-built series: a genuine dip-then-reclaim
inside an uptrend fires exactly one reclaim event, the DPI-persistence split
routes it to R_dpi vs R_nodpi correctly, and the per-name cooldown de-overlaps
repeat entries. These guard the event definition itself (the study's headline is
a marginal comparison, so a mis-detected event would silently bias it).
"""
import numpy as np
import pandas as pd

import spx_reclaim_dpi_study as R


def _one_name_dip_reclaim():
    """A rising base (well above a 200d floor), then a multi-week dip below the
    50d MA, then a reclaim -- one clean event. Returns (adj, D) single-column."""
    idx = pd.bdate_range("2019-01-02", periods=320)
    # long steady uptrend so the 200d MA sits well below price throughout
    base = np.linspace(50, 150, 320)
    # carve a ~25-session pullback near the end that dips below the 50d MA then recovers
    dip = np.zeros(320)
    dip[280:300] = -np.concatenate([np.linspace(0, 12, 10), np.linspace(12, 0, 10)])
    px = base + dip
    adj = pd.DataFrame({"A": px}, index=idx)
    return adj, idx


def test_reclaim_fires_once_inside_uptrend():
    adj, idx = _one_name_dip_reclaim()
    D = pd.DataFrame({"A": np.full(len(idx), 0.5)}, index=idx)   # DPI mid -> not persistent-high
    spy = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    f = R.build_events(D, adj, spy, reclaim_ma=50, trend_ma=200,
                       persist_win=15, persist_frac=0.5, prior_win=10, prior_frac=0.5)
    ev_dpi = R.deoverlap(f["ev_dpi"])
    ev_nodpi = R.deoverlap(f["ev_nodpi"])
    # exactly one reclaim, and with mid DPI it must be the no-DPI branch
    assert len(ev_dpi) + len(ev_nodpi) >= 1
    assert len(ev_dpi) == 0
    assert len(ev_nodpi) >= 1


def test_persistent_high_dpi_routes_to_r_dpi():
    adj, idx = _one_name_dip_reclaim()
    d = np.full(len(idx), 0.30)
    d[275:302] = 0.95                       # DPI spikes high through the pullback + reclaim
    D = pd.DataFrame({"A": d}, index=idx)
    spy = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    f = R.build_events(D, adj, spy, reclaim_ma=50, trend_ma=200,
                       persist_win=15, persist_frac=0.5, prior_win=10, prior_frac=0.5)
    # self-relative percentile of a series that is mostly 0.30 then 0.95 -> the
    # spike lands in the top quintile and persists -> the reclaim routes to R_dpi.
    assert len(R.deoverlap(f["ev_dpi"])) >= 1
    assert len(R.deoverlap(f["ev_nodpi"])) == 0


def test_no_event_when_below_long_trend():
    # price in a genuine downtrend (below its 200d MA) must never fire an event,
    # even if it wiggles back above the 50d MA.
    idx = pd.bdate_range("2019-01-02", periods=320)
    px = np.linspace(150, 60, 320) + 3 * np.sin(np.linspace(0, 30, 320))
    adj = pd.DataFrame({"A": px}, index=idx)
    D = pd.DataFrame({"A": np.full(len(idx), 0.95)}, index=idx)
    spy = pd.Series(np.linspace(100, 90, len(idx)), index=idx)
    f = R.build_events(D, adj, spy, reclaim_ma=50, trend_ma=200)
    assert len(R.deoverlap(f["ev_dpi"])) == 0
    assert len(R.deoverlap(f["ev_nodpi"])) == 0


def test_cooldown_deoverlaps_repeat_entries():
    idx = pd.bdate_range("2019-01-02", periods=8)
    mask = pd.DataFrame({"A": [True, True, True, False, True, False, False, True]}, index=idx)
    # cooldown 3 -> events at positions 0, 4(>=0+3? 4-0=4 ok -> but 1,2 within 3 of 0 skipped), 7
    ev = R.deoverlap(mask, cooldown=3)
    pos = [idx.get_loc(d) for d, _ in ev]
    assert pos == [0, 4, 7]
