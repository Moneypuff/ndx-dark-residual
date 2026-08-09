"""Tests for the pure pieces of ``etf_expected_move_study``.

The study's claim is a relative-value comparison: empirical fair premia
(expiry-style payoffs under the conditional return distribution) against an
implied expected move read off an option chain. Both sides are pinned here
on hand-built inputs -- the payoff identities (call + put = straddle =
E[|r|]), strike monotonicity, the expiration picker, and both branches of
the implied-move calculation (live straddle mid vs the IV fallback).
"""
import numpy as np
import pandas as pd
import pytest

import etf_expected_move_study as E


RETS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])


# ---------------------------------------------------------------------------
# fair_premia / dist_stats
# ---------------------------------------------------------------------------
def test_fair_premia_known_values():
    p = E.fair_premia(RETS, otm=(5.0, 10.0))
    assert p["straddle"] == pytest.approx(6.0)     # mean |r|
    assert p["call"] == pytest.approx(3.0)         # mean max(r, 0)
    assert p["put"] == pytest.approx(3.0)
    assert p["call5"] == pytest.approx(1.0)        # only +10 pays 5
    assert p["call10"] == pytest.approx(0.0)
    assert p["pbelow5"] == pytest.approx(20.0)     # only -10 is < -5
    assert p["pbelow10"] == pytest.approx(0.0)     # -10 is not strictly < -10


def test_fair_premia_parity_and_monotonicity():
    rng = np.random.default_rng(7)
    r = rng.normal(2, 12, 500)
    p = E.fair_premia(r, otm=(5.0, 10.0))
    # ATM call + ATM put == straddle == E|r|
    assert p["call"] + p["put"] == pytest.approx(p["straddle"])
    assert p["straddle"] == pytest.approx(np.mean(np.abs(r)))
    # call value decreases as the strike moves up
    assert p["call"] >= p["call5"] >= p["call10"] >= 0


def test_dist_stats_move_size_vs_location():
    st = E.dist_stats(RETS)
    assert st["n"] == 5
    assert st["mean"] == pytest.approx(0.0)
    assert st["med"] == pytest.approx(0.0)
    assert st["eabs"] == pytest.approx(6.0)        # move size despite zero drift
    assert st["q10"] < st["q25"] < st["q75"] < st["q90"]


# ---------------------------------------------------------------------------
# fwd_returns / uncond_returns
# ---------------------------------------------------------------------------
def _px(vals, start="2020-01-02"):
    return pd.Series(list(vals), index=pd.bdate_range(start, periods=len(vals)),
                     dtype="float64")


def test_fwd_returns_alignment_and_truncation():
    c = _px(100 * 1.01 ** np.arange(30))
    r = E.fwd_returns(c, [c.index[0], c.index[25]], horizon=10)
    # the second event's horizon runs past the data and is dropped
    assert len(r) == 1
    assert r[0] == pytest.approx((1.01 ** 10 - 1) * 100, rel=1e-9)


def test_uncond_returns_window_count():
    c = _px(np.linspace(100, 110, 50))
    r = E.uncond_returns(c, horizon=10)
    assert len(r) == 40
    assert np.all(r > 0)


# ---------------------------------------------------------------------------
# pick_expiry / implied_move
# ---------------------------------------------------------------------------
def test_pick_expiry_nearest():
    today = pd.Timestamp("2026-01-05")
    exps = [int((today + pd.Timedelta(days=d)).timestamp()) for d in (30, 85, 100, 180)]
    assert E.pick_expiry(exps, today, 91) == exps[1]
    assert E.pick_expiry(exps, today, 182) == exps[3]
    assert E.pick_expiry([], today, 91) is None


def _chain(spot, bid_ask=None, iv=None):
    leg = {"strike": spot}
    if bid_ask:
        leg.update(bid=bid_ask[0], ask=bid_ask[1])
    if iv is not None:
        leg["impliedVolatility"] = iv
    return {"calls": [dict(leg), {"strike": spot * 1.5}],
            "puts": [dict(leg), {"strike": spot * 0.5}]}


def test_implied_move_straddle_mid():
    today = pd.Timestamp("2026-01-05")
    expiry = int((today + pd.Timedelta(days=91)).timestamp())
    ch = _chain(100.0, bid_ask=(4.0, 6.0))
    # call mid 5 + put mid 5 = 10 -> 10% of spot
    assert E.implied_move(ch, 100.0, expiry, today) == pytest.approx(10.0)


def test_implied_move_iv_fallback_and_empty():
    today = pd.Timestamp("2026-01-05")
    expiry = int((today + pd.Timedelta(days=91)).timestamp())
    ch = _chain(100.0, iv=0.40)                    # no bids -> IV fallback
    expect = 0.8 * 0.40 * np.sqrt(91 / 365.25) * 100
    assert E.implied_move(ch, 100.0, expiry, today) == pytest.approx(expect, rel=1e-6)
    assert E.implied_move({"calls": [], "puts": []}, 100.0, expiry, today) is None
