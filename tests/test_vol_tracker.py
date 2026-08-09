"""Tests for the pure pieces of ``build_vol_tracker``.

The tracker's claims rest on three mechanics pinned here: the fixed-strike
identity (a contract keeps its row across days; local re-pricing is its
dIV net of its own expiry's ATM dIV), the T+1 pairing (dOI of snapshot t
scored against the local dIV of session t-1), and the aggressor quadrants.
All on hand-built snapshot frames -- no network, no real chains.
"""
import numpy as np
import pandas as pd
import pytest

import build_vol_tracker as V


def _rows(date, symbol, expiry, spot, quotes):
    """quotes: list of (right, strike, iv, oi)."""
    return [{"date": date, "symbol": symbol, "expiry": expiry, "right": r,
             "strike": k, "iv": iv, "oi": oi, "volume": 0, "bid": 1.0,
             "ask": 1.2, "last": 1.1, "spot": spot}
            for r, k, iv, oi in quotes]


def _panel_3d():
    """Three days, one symbol/expiry. ATM (interpolated flat smile) rises
    0.02/day; the 110C is re-priced +0.05/day (local +0.03); the 90P
    follows ATM exactly (local 0). OI: 110C prints +500 on day 3."""
    days = ["2026-08-10", "2026-08-11", "2026-08-12"]
    rows = []
    for i, d in enumerate(days):
        base = 0.30 + 0.02 * i
        rows += _rows(d, "GDX", "2027-01-15", 100.0, [
            ("P", 80.0, base, 1000), ("P", 90.0, base, 1000),
            ("C", 100.0, base, 1000), ("C", 105.0, base, 1000),
            ("C", 110.0, 0.30 + 0.05 * i, 2000 + (500 if i == 2 else 0)),
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# atm_iv / contract_panel / local_repricing
# ---------------------------------------------------------------------------
def test_atm_iv_interpolates_otm_smile():
    df = _panel_3d()
    day = df[df["date"] == "2026-08-10"]
    assert V.atm_iv(day) == pytest.approx(0.30, abs=1e-9)
    thin = day.iloc[:2]                       # one-sided smile -> NaN
    assert np.isnan(V.atm_iv(thin))


def test_local_repricing_nets_out_atm():
    iv, oi, atm = V.contract_panel(_panel_3d())
    dloc = V.local_repricing(iv, atm)
    k110 = ("GDX", "2027-01-15", 110.0, "C")
    k90 = ("GDX", "2027-01-15", 90.0, "P")
    # 110C: dIV +0.05 vs ATM +0.02 -> local +0.03; 90P tracks ATM -> 0
    assert dloc.loc[k110, "2026-08-11"] == pytest.approx(0.03, abs=1e-9)
    assert dloc.loc[k90, "2026-08-11"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# classify_flow / flow_table / pressure_index
# ---------------------------------------------------------------------------
def test_classify_flow_quadrants_and_threshold():
    assert V.classify_flow(500, +0.01) == "buyers_opening"
    assert V.classify_flow(500, -0.01) == "sellers_opening"
    assert V.classify_flow(-500, -0.01) == "longs_closing"
    assert V.classify_flow(-500, +0.01) == "shorts_covering"
    assert V.classify_flow(50, +0.01) is None          # under DOI_MIN
    assert V.classify_flow(500, np.nan) is None


def test_flow_table_pairs_doi_with_prior_session_iv():
    iv, oi, atm = V.contract_panel(_panel_3d())
    flows = V.flow_table(iv, oi, atm)
    # only the 110C moved OI (+500 printed on day 3), and its session-2
    # local dIV was +0.03 -> buyers opening, dated day 3
    assert len(flows) == 1
    f = flows.iloc[0]
    assert (f["strike"], f["right"], f["date"]) == (110.0, "C", "2026-08-12")
    assert f["flow"] == "buyers_opening"
    assert f["doi"] == pytest.approx(500)
    assert f["dlocal"] == pytest.approx(3.0)           # vol points


def test_pressure_index_sign():
    iv, oi, atm = V.contract_panel(_panel_3d())
    press = V.pressure_index(V.flow_table(iv, oi, atm), oi)
    row = press.iloc[0]
    assert row["symbol"] == "GDX" and row["right"] == "C"
    assert row["pressure"] == pytest.approx(500 * 3.0 / 1000)   # positive = buyers
    assert V.pressure_index(pd.DataFrame(), oi).empty


# ---------------------------------------------------------------------------
# big_oi_map / resolve_leg
# ---------------------------------------------------------------------------
def test_big_oi_map_first_vs_latest():
    m = V.big_oi_map(_panel_3d(), top=1)
    r = m.iloc[0]
    assert (r["strike"], r["right"]) == (110.0, "C")   # biggest OI line
    assert r["oi"] == 2500 and r["oi_first"] == 2000
    assert r["iv"] == pytest.approx(0.40) and r["iv_first"] == pytest.approx(0.30)
    assert r["days_seen"] == 3


def test_resolve_leg_nearest_expiry_then_strike():
    day = pd.DataFrame(
        _rows("2026-08-10", "GDX", "2026-11-20", 100.0,
              [("C", 100.0, 0.3, 10), ("C", 110.0, 0.3, 10)]) +
        _rows("2026-08-10", "GDX", "2027-02-19", 100.0,
              [("C", 100.0, 0.3, 10), ("C", 110.0, 0.3, 10),
               ("P", 90.0, 0.3, 10)]))
    leg = V.resolve_leg(day, "GDX", "C", 183, 0.10, "2026-08-10")
    assert leg["expiry"] == "2027-02-19" and leg["strike"] == 110.0
    put = V.resolve_leg(day, "GDX", "P", 91, -0.08, "2026-08-10")
    assert put["expiry"] == "2027-02-19" and put["strike"] == 90.0
    assert V.resolve_leg(day, "QQQ", "C", 91, 0.0, "2026-08-10") is None
