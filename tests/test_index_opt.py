"""Tests for the index-level signal optimization study: strategy accounting
(no look-ahead, cost per switch), the timing-edge stat, and proof the
walk-forward selector picks a planted dominant config out of sample.
"""
import numpy as np
import pandas as pd
import pytest

import spx_index_signal_opt_study as O


def test_strat_returns_no_lookahead_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=5)
    mkt = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=idx)
    pos = pd.Series([1.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    net, held = O.strat_returns(pos, mkt, cost_bps=10.0)
    # day0: no prior position -> flat, no cost
    assert net.iloc[0] == pytest.approx(0.0)
    # day1: holds day0's position (1) -> earns mkt[1] minus the 0->1 switch cost
    assert net.iloc[1] == pytest.approx(0.02 - 10e-4)
    # day2: holds day1's position (1), no switch -> full mkt[2]
    assert net.iloc[2] == pytest.approx(-0.01)
    # day3: day2's position was 0 -> flat but pays the 1->0 switch
    assert net.iloc[3] == pytest.approx(0.0 - 10e-4)
    # day4: day3's position was 1 -> earns mkt[4] minus 0->1 switch
    assert net.iloc[4] == pytest.approx(0.01 - 10e-4)


def test_hold_config_matches_market():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(5e-4, 0.01, 300), index=idx)
    pos = pd.Series(1.0, index=idx)
    net, held = O.strat_returns(pos, mkt, cost_bps=0.0)
    # after the first day (entry), buy-and-hold IS the market
    assert np.allclose(net.iloc[1:], mkt.iloc[1:])
    assert O.perf(net, held)["exposure"] == pytest.approx(100.0, abs=0.5)


def test_timing_edge_detects_planted_good_days():
    # market pays +30bps on signal days and -5bps otherwise -> edge must be
    # strongly positive with a CI clear of zero; a random signal must straddle.
    idx = pd.bdate_range("2018-01-01", periods=1200)
    rng = np.random.default_rng(1)
    sig = pd.Series((rng.uniform(0, 1, 1200) < 0.3).astype(float), index=idx)
    good = sig.shift(1).fillna(0.0) > 0                 # position lag mirrors strat_returns
    mkt = pd.Series(np.where(good, 30e-4, -5e-4) + rng.normal(0, 5e-4, 1200), index=idx)
    e = O.timing_edge(sig, mkt)
    assert e["edge_bps"] > 15
    assert e["ci"][0] > 0
    rand = pd.Series((rng.uniform(0, 1, 1200) < 0.3).astype(float), index=idx)
    e2 = O.timing_edge(rand, mkt)
    assert e2["ci"][0] < e2["edge_bps"] < e2["ci"][1]


def test_entry_hold_pos_hand_computed():
    idx = pd.bdate_range("2020-01-01", periods=10)
    active = pd.Series([False, True, True, False, False, False, True, False, False, False],
                       index=idx)
    pos = O.entry_hold_pos(active, 3)
    # entry at day1 -> long days 1,2,3; entry at day6 -> long days 6,7,8
    assert pos.tolist() == [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    # NaN active never triggers an entry
    active2 = pd.Series([np.nan, True] + [False] * 8, index=idx)
    assert O.entry_hold_pos(active2, 2).iloc[1] == 1.0
    assert O.entry_hold_pos(active2, 2).iloc[0] == 0.0


def test_walk_forward_picks_planted_dominant_config():
    # config GOOD is long exactly on the +40bps days; BAD is long on random
    # days. The selector must pick GOOD every year, and never peek: the pick
    # for year Y must be based only on data before Y.
    idx = pd.bdate_range("2018-01-01", periods=1500)
    rng = np.random.default_rng(2)
    good_pos = pd.Series((rng.uniform(0, 1, 1500) < 0.4).astype(float), index=idx)
    mkt = pd.Series(np.where(good_pos.shift(1).fillna(0) > 0, 40e-4, -10e-4)
                    + rng.normal(0, 10e-4, 1500), index=idx)
    df = pd.DataFrame(index=idx)
    configs = {"GOOD": good_pos,
               "BAD": pd.Series((rng.uniform(0, 1, 1500) < 0.4).astype(float), index=idx),
               "HOLD": pd.Series(1.0, index=idx)}
    stitched, picks = O.walk_forward(df, configs, mkt, start_year=2020)
    assert all(name == "GOOD" for name, _ in picks.values())
    # stitched positions equal GOOD's over the trading window
    lo = stitched.index[0]
    assert (stitched == good_pos[good_pos.index >= lo]).all()
    # and the walk-forward line beats HOLD on Sharpe over the trading window
    s_wf = O.sharpe_on(stitched, mkt[mkt.index >= lo])
    s_bh = O.sharpe_on(configs["HOLD"][lambda s: s.index >= lo], mkt[mkt.index >= lo])
    assert s_wf > s_bh
