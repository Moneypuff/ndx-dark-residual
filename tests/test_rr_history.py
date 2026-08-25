"""Tests for the pure pieces of ``build_rr_history`` -- the ORATS wide ->
optsnap-shaped long reshaping and the own-IV solve. The pillar math itself
lives in build_vol_tracker and is tested there; what's pinned here is that
the backfill feeds it the exact same frame shape the live snapshots do."""
import numpy as np
import pandas as pd
import pytest

import build_rr_history as R
import trade_structures as T


def _wide(strike, spot=100.0, c_bid=1.0, c_ask=1.2, p_bid=1.0, p_ask=1.2,
          c_oi=10, p_oi=10):
    return {"trade_date": pd.Timestamp("2026-08-10"),
            "expirDate": pd.Timestamp("2026-11-20"), "strike": strike,
            "stkPx": spot, "cBidPx": c_bid, "cAskPx": c_ask, "cOi": c_oi,
            "cVolu": 0, "pBidPx": p_bid, "pAskPx": p_ask, "pOi": p_oi,
            "pVolu": 0}


def test_long_rows_otm_split():
    df = pd.DataFrame([_wide(90.0), _wide(100.0), _wide(110.0)])
    out = R.long_rows(df, "GDX")
    puts = out[out["right"] == "P"]
    calls = out[out["right"] == "C"]
    assert set(puts["strike"]) == {90.0}          # strictly below spot
    assert set(calls["strike"]) == {100.0, 110.0}  # at/above spot
    assert set(out.columns) == {"date", "symbol", "expiry", "right", "strike",
                                "iv", "oi", "volume", "bid", "ask", "last",
                                "spot"}
    assert (out["date"] == "2026-08-10").all()
    assert (out["expiry"] == "2026-11-20").all()


def test_long_rows_empty():
    assert R.long_rows(pd.DataFrame(), "GDX").empty


def test_solve_ivs_round_trips_known_price():
    t = (pd.Timestamp("2026-11-20") - pd.Timestamp("2026-08-10")).days / 365.25
    theo = T.bs_price(100.0, 110.0, t, 0.40, "C")
    df = pd.DataFrame([_wide(110.0, c_bid=theo - 0.01, c_ask=theo + 0.01)])
    out = R.solve_ivs(R.long_rows(df, "GDX"))
    call = out[out["right"] == "C"].iloc[0]
    assert call["iv"] == pytest.approx(0.40, abs=1e-3)


def test_solve_ivs_dead_quote_stays_nan():
    df = pd.DataFrame([_wide(110.0, c_bid=0.0, c_ask=0.0)])
    out = R.solve_ivs(R.long_rows(df, "GDX"))
    assert np.isnan(out[out["right"] == "C"]["iv"].iloc[0])
