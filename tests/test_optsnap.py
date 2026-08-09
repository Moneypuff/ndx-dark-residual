"""Tests for the pure policy pieces of ``snapshot_option_chains``.

The snapshot is the only unrepeatable step of the vol tracker (Yahoo keeps
no chain history), so what it decides to keep is pinned here: the expiry
policy (nearest-N + monthlies out to nine months + every January LEAP) and
the contract filter (alive within the moneyness window -- widened for
LEAPs -- plus the top-OI strikes regardless of moneyness).
"""
import pandas as pd
import pytest

import snapshot_option_chains as S


def _epoch(*dates):
    return [int(pd.Timestamp(d).timestamp()) for d in dates]


TODAY = pd.Timestamp("2026-08-10")


# ---------------------------------------------------------------------------
# select_expiries
# ---------------------------------------------------------------------------
def test_select_expiries_policy():
    exps = _epoch(
        "2026-08-14", "2026-08-21", "2026-08-28",       # weeklies + monthly
        "2026-09-18", "2026-12-18",                     # monthlies
        "2027-01-15", "2027-06-18", "2028-01-21",       # Jan LEAPs + far monthly
    )
    kept = [str(pd.Timestamp(e, unit="s").date())
            for e in S.select_expiries(exps, TODAY)]
    assert "2026-08-14" in kept and "2026-08-21" in kept   # nearest 2
    assert "2026-09-18" in kept and "2026-12-18" in kept   # monthlies <= 9M
    assert "2027-01-15" in kept and "2028-01-21" in kept   # every January
    assert "2027-06-18" not in kept                        # > 9M, not January
    assert "2026-08-28" not in kept                        # weekly, not nearest-2


def test_select_expiries_picks_third_friday_per_month():
    # two September listings: the one nearer the 3rd Friday (18th) wins
    exps = _epoch("2026-08-14", "2026-08-21", "2026-09-11", "2026-09-18")
    kept = {str(pd.Timestamp(e, unit="s").date())
            for e in S.select_expiries(exps, TODAY)}
    assert "2026-09-18" in kept and "2026-09-11" not in kept


def test_select_expiries_drops_past_and_empty():
    assert S.select_expiries([], TODAY) == []
    past = _epoch("2026-07-17")
    assert S.select_expiries(past, TODAY) == []


def test_third_friday():
    assert S.third_friday(2026, 8) == pd.Timestamp("2026-08-21")
    assert S.third_friday(2027, 1) == pd.Timestamp("2027-01-15")


def test_is_leap():
    assert S.is_leap(_epoch("2027-01-15")[0])
    assert not S.is_leap(_epoch("2026-12-18")[0])


# ---------------------------------------------------------------------------
# filter_contracts
# ---------------------------------------------------------------------------
def _c(strike, oi=0, bid=0.0):
    return {"strike": strike, "openInterest": oi, "bid": bid,
            "impliedVolatility": 0.3}


def test_filter_window_and_liveness():
    spot = 100.0
    cs = [_c(80, oi=5), _c(90, bid=1.0), _c(110, oi=1),
          _c(120, oi=0, bid=0.0),          # in window but dead
          _c(150, oi=2)]                   # alive but outside +/-25%
    kept = {c["strike"] for c in S.filter_contracts(cs, spot, leap=False, top_oi=0)}
    assert kept == {80, 90, 110}


def test_filter_leap_window_reaches_note_barriers():
    spot = 100.0
    cs = [_c(40, oi=9), _c(60, oi=9), _c(160, oi=9), _c(170, oi=9)]
    normal = {c["strike"] for c in S.filter_contracts(cs, spot, leap=False, top_oi=0)}
    leap = {c["strike"] for c in S.filter_contracts(cs, spot, leap=True, top_oi=0)}
    assert normal == set()                 # all outside +/-25%
    assert leap == {40, 60, 160}           # +/-65% catches the barrier zone
    assert 170 not in leap


def test_filter_top_oi_rescues_monster_strikes():
    spot = 100.0
    cs = [_c(95, bid=1.0), _c(300, oi=50000)]   # way outside any window
    kept = {c["strike"] for c in S.filter_contracts(cs, spot, leap=False)}
    assert kept == {95, 300}


def test_filter_empty_inputs():
    assert S.filter_contracts([], 100.0, leap=False) == []
    assert S.filter_contracts([_c(100, oi=1)], None, leap=False) == []


# ---------------------------------------------------------------------------
# contract_rows
# ---------------------------------------------------------------------------
def test_contract_rows_shape():
    e = int(pd.Timestamp("2027-01-15").timestamp())
    chain = {"calls": [_c(60, oi=7)], "puts": [_c(55, oi=3)]}
    rows = S.contract_rows(chain, "GDX", 100.0, e, "2026-08-10")
    assert {r["right"] for r in rows} == {"C", "P"}
    r = rows[0]
    assert r["symbol"] == "GDX" and r["expiry"] == "2027-01-15"
    assert set(r) == {"date", "symbol", "expiry", "right", "strike", "iv",
                      "oi", "volume", "bid", "ask", "last", "spot"}
