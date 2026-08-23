"""Tests for the pure pieces of ``backfill_optsnap_from_orats``: reshaping
ORATS rows into the Yahoo-chain-shaped dicts the live capture's own policy
functions expect, and that policy (expiry selection, moneyness window,
top-OI rescue) coming out identically on backfilled data."""
import pandas as pd
import pytest

import backfill_optsnap_from_orats as B


def _orats_row(expiry, strike, spot=100.0, c_iv=0.30, p_iv=0.30,
               c_oi=0, p_oi=0, c_bid=1.0, c_ask=1.2, p_bid=1.0, p_ask=1.2):
    return {"expirDate": pd.Timestamp(expiry), "strike": strike, "stkPx": spot,
            "cMidIv": c_iv, "cOi": c_oi, "cBidPx": c_bid, "cAskPx": c_ask, "cVolu": 0,
            "pMidIv": p_iv, "pOi": p_oi, "pBidPx": p_bid, "pAskPx": p_ask, "pVolu": 0}


# ---------------------------------------------------------------------------
# _side_contracts
# ---------------------------------------------------------------------------
def test_side_contracts_treats_zero_iv_as_no_quote():
    df = pd.DataFrame([_orats_row("2026-09-18", 100, c_iv=0.0, p_iv=0.28)])
    calls = B._side_contracts(df, "C")
    puts = B._side_contracts(df, "P")
    assert calls[0]["impliedVolatility"] is None
    assert puts[0]["impliedVolatility"] == pytest.approx(0.28)


def test_side_contracts_last_is_mid_of_tight_quote():
    df = pd.DataFrame([_orats_row("2026-09-18", 100, c_bid=2.0, c_ask=2.4)])
    calls = B._side_contracts(df, "C")
    assert calls[0]["lastPrice"] == pytest.approx(2.2)


def test_side_contracts_shape():
    df = pd.DataFrame([_orats_row("2026-09-18", 100)])
    row = B._side_contracts(df, "C")[0]
    assert set(row) == {"strike", "openInterest", "bid", "ask",
                        "impliedVolatility", "volume", "lastPrice"}


# ---------------------------------------------------------------------------
# build_day_rows
# ---------------------------------------------------------------------------
TODAY = pd.Timestamp("2026-08-10")


def test_build_day_rows_applies_expiry_and_moneyness_policy():
    rows = [
        # nearest 2 listed expiries, one strike each, near the money
        _orats_row("2026-08-14", 100, spot=100.0),
        _orats_row("2026-08-21", 100, spot=100.0),
        # a monthly within 9 months
        _orats_row("2026-09-18", 100, spot=100.0),
        # too far out and not January -> dropped entirely
        _orats_row("2027-06-18", 100, spot=100.0),
        # a January LEAP, deep strike only reachable via the wide LEAP window
        _orats_row("2027-01-15", 65, spot=100.0, c_oi=5, p_oi=5),
    ]
    df = pd.DataFrame(rows)
    out = B.build_day_rows(df, "GDX", TODAY)
    expiries = {r["expiry"] for r in out}
    assert expiries == {"2026-08-14", "2026-08-21", "2026-09-18", "2027-01-15"}
    assert all(r["symbol"] == "GDX" and r["spot"] == 100.0 for r in out)


def test_build_day_rows_drops_dead_strike_outside_window():
    rows = [
        _orats_row("2026-08-14", 100, spot=100.0, c_oi=1, p_oi=1),        # ATM, alive
        _orats_row("2026-08-14", 200, spot=100.0, c_oi=0, p_oi=0,
                   c_bid=0.0, p_bid=0.0),                                  # dead, far OTM
    ]
    df = pd.DataFrame(rows)
    out = B.build_day_rows(df, "GDX", TODAY)
    strikes = {r["strike"] for r in out}
    assert 100.0 in strikes
    assert 200.0 not in strikes


def test_build_day_rows_empty_without_spot():
    df = pd.DataFrame([_orats_row("2026-08-14", 100, spot=0.0)])
    assert B.build_day_rows(df, "GDX", TODAY) == []


def test_build_day_rows_empty_frame():
    assert B.build_day_rows(pd.DataFrame(), "GDX", TODAY) == []
