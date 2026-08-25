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
# delta pillars / constant maturity / rr frame / payload
# ---------------------------------------------------------------------------
# spot 100, flat 25-vol, T=30d: call deltas at 100/105/110/115 span
# ~0.51..0.03 (0.50, 0.25, 0.10 all inside); put |deltas| at 90/95/97
# span ~0.07..0.32 (0.25 and 0.10 inside).
FLAT_QUOTES = [("P", 90.0, 0.25, 100), ("P", 95.0, 0.25, 100),
               ("P", 97.0, 0.25, 100), ("C", 100.0, 0.25, 100),
               ("C", 105.0, 0.25, 100), ("C", 110.0, 0.25, 100),
               ("C", 115.0, 0.25, 100)]


def test_delta_pillars_flat_smile():
    rows = pd.DataFrame(_rows("2026-08-10", "GDX", "2026-09-09", 100.0,
                              FLAT_QUOTES))
    p = V.delta_pillars(rows, 100.0, 30 / 365.25)
    for name in V.RR_PILLARS:
        assert p[name] == pytest.approx(0.25, abs=1e-6), name


def test_delta_pillars_guards():
    # thin put side (2 points) -> put pillars NaN, call side still fills
    thin = [q for q in FLAT_QUOTES if not (q[0] == "P" and q[1] == 90.0)]
    thin = [q for q in thin if not (q[0] == "P" and q[1] == 95.0)]
    rows = pd.DataFrame(_rows("2026-08-10", "GDX", "2026-09-09", 100.0, thin))
    p = V.delta_pillars(rows, 100.0, 30 / 365.25)
    assert np.isnan(p["iv25p"]) and np.isnan(p["iv10p"])
    assert np.isfinite(p["iv50"])
    # calls capped below delta 0.50: parity-combined smile still spans it
    # (the put just below spot has call-equivalent delta just above 0.50)
    deep = [("P", 90.0, 0.25, 100), ("P", 95.0, 0.25, 100),
            ("P", 97.0, 0.25, 100), ("C", 105.0, 0.25, 100),
            ("C", 110.0, 0.25, 100), ("C", 115.0, 0.25, 100)]
    rows2 = pd.DataFrame(_rows("2026-08-10", "GDX", "2026-09-09", 100.0, deep))
    p2 = V.delta_pillars(rows2, 100.0, 30 / 365.25)
    assert p2["iv50"] == pytest.approx(0.25, abs=1e-6)
    assert np.isfinite(p2["iv25c"]) and np.isfinite(p2["iv10c"])
    # no put side at all: nothing reaches call-delta 0.50 -> iv50 NaN
    calls_only = [q for q in deep if q[0] == "C"]
    rows3 = pd.DataFrame(_rows("2026-08-10", "GDX", "2026-09-09", 100.0,
                               calls_only))
    p3 = V.delta_pillars(rows3, 100.0, 30 / 365.25)
    assert np.isnan(p3["iv50"])


def test_cm_pillars_interpolation():
    lo = dict.fromkeys(V.RR_PILLARS, 0.20)
    hi = dict.fromkeys(V.RR_PILLARS, 0.30)
    # midpoint of 20d and 40d brackets at tenor 30
    cm = V.cm_pillars([(20, lo), (40, hi)], 30)
    assert cm["iv50"] == pytest.approx(0.25)
    # exact-DTE match uses that expiry directly
    cm2 = V.cm_pillars([(30, lo), (60, hi)], 30)
    assert cm2["iv50"] == pytest.approx(0.20)
    # unbracketed (all expiries beyond the tenor) -> NaN
    cm3 = V.cm_pillars([(40, lo), (60, hi)], 30)
    assert np.isnan(cm3["iv50"])
    # NaN on one bracket side -> NaN for that pillar only
    lo_nan = dict(lo, iv10p=np.nan)
    cm4 = V.cm_pillars([(20, lo_nan), (40, hi)], 30)
    assert np.isnan(cm4["iv10p"]) and cm4["iv50"] == pytest.approx(0.25)


def test_rr_frame_end_to_end():
    rows = (_rows("2026-08-10", "GDX", "2026-08-30", 100.0, FLAT_QUOTES) +
            _rows("2026-08-10", "GDX", "2026-09-19", 100.0, FLAT_QUOTES))
    rr = V.rr_frame(pd.DataFrame(rows))
    t30 = rr[rr["tenor"] == 30]
    assert len(t30) == 1
    assert t30["iv50"].iloc[0] == pytest.approx(0.25, abs=1e-6)
    assert (rr["tenor"] == 90).sum() == 0        # 20d/40d can't bracket 90d


def test_merge_rr_seam():
    cols = ["date", "symbol", "tenor", *V.RR_PILLARS]
    bf = pd.DataFrame([{"date": d, "symbol": "GDX", "tenor": 30,
                        **dict.fromkeys(V.RR_PILLARS, 0.2)}
                       for d in ["2026-08-08", "2026-08-10", "2026-08-11"]],
                      columns=cols)
    live = pd.DataFrame([{"date": d, "symbol": "GDX", "tenor": 30,
                          **dict.fromkeys(V.RR_PILLARS, 0.3)}
                         for d in ["2026-08-10", "2026-08-12"]], columns=cols)
    m = V.merge_rr(bf, live)
    assert list(m["date"]) == ["2026-08-08", "2026-08-10", "2026-08-12"]
    assert m[m["date"] == "2026-08-10"]["iv50"].iloc[0] == 0.3   # live wins
    assert V.merge_rr(bf, live.iloc[0:0])["iv50"].eq(0.2).all()  # no live


def test_rr_payload_shape_and_hygiene():
    import json
    dates = [str(d.date()) for d in
             pd.bdate_range("2023-01-02", "2026-08-21")]
    rows = [{"date": d, "symbol": "GDX", "tenor": t,
             "iv10p": 0.30, "iv25p": 0.27, "iv50": 0.2512345,
             "iv25c": 0.23, "iv10c": np.nan}
            for d in dates for t in (30, 90)]
    pay = V.rr_payload(pd.DataFrame(rows))
    assert pay["tenors"] == [30, 90]
    assert "GDX" in pay["symbols"] and "t30" in pay["symbols"]["GDX"]
    # weekly axis before the cutoff: strictly fewer points than raw days
    assert len(pay["dates"]) < len(dates)
    # daily after the cutoff: the last 10 business days all present
    assert set(dates[-10:]) <= set(pay["dates"])
    # 2dp rounding, NaN -> None, and no NaN token leaks into the JSON
    s = pay["symbols"]["GDX"]["t30"]
    assert s["atm"][-1] == pytest.approx(25.12)
    assert all(v is None for v in s["rr10"])     # iv10c was NaN throughout
    assert "NaN" not in json.dumps(pay)
    # skew slice: 3 dates -- latest, ~1w back, ~1m back
    sk = pay["skew"]["GDX"]["t30"]
    assert sk["dates"][0] == dates[-1]
    d0 = pd.Timestamp(sk["dates"][0])
    assert abs((d0 - pd.Timestamp(sk["dates"][1])).days - 7) <= 3
    assert abs((d0 - pd.Timestamp(sk["dates"][2])).days - 30) <= 4
    assert sk["atm"][0] == pytest.approx(25.12)


def test_rr_payload_empty():
    empty = pd.DataFrame(columns=["date", "symbol", "tenor", *V.RR_PILLARS])
    pay = V.rr_payload(empty)
    assert pay["dates"] == [] and pay["symbols"] == {}


# ---------------------------------------------------------------------------
# recompute_iv
# ---------------------------------------------------------------------------
def _iv_row(bid, ask, last, spot=100.0, strike=100.0, right="C",
           date="2026-08-10", expiry="2026-11-20"):
    return {"date": date, "symbol": "GDX", "expiry": expiry, "right": right,
            "strike": strike, "iv": 0.999, "oi": 1, "volume": 0,
            "bid": bid, "ask": ask, "last": last, "spot": spot}


def test_recompute_iv_prefers_tight_bid_ask_mid():
    import trade_structures as T
    t = (pd.Timestamp("2026-11-20") - pd.Timestamp("2026-08-10")).days / 365.25
    theo = T.bs_price(100.0, 100.0, t, 0.35, "C")
    # bid/ask straddle theo tightly; a very different `last` must be ignored
    row = _iv_row(bid=round(theo - 0.05, 2), ask=round(theo + 0.05, 2), last=999.0)
    out = V.recompute_iv(pd.DataFrame([row]))
    assert out["iv"].iloc[0] == pytest.approx(0.35, abs=0.01)


def test_recompute_iv_falls_back_to_last_without_two_sided_quote():
    t = (pd.Timestamp("2026-11-20") - pd.Timestamp("2026-08-10")).days / 365.25
    import trade_structures as T
    theo = T.bs_price(100.0, 100.0, t, 0.50, "C")
    row = _iv_row(bid=0.0, ask=0.0, last=theo)
    out = V.recompute_iv(pd.DataFrame([row]))
    assert out["iv"].iloc[0] == pytest.approx(0.50, abs=0.01)


def test_recompute_iv_nan_without_any_usable_price():
    row = _iv_row(bid=0.0, ask=0.0, last=0.0)
    out = V.recompute_iv(pd.DataFrame([row]))
    assert np.isnan(out["iv"].iloc[0])


def test_recompute_iv_ignores_the_original_vendor_iv():
    # the stored `iv` (0.999, deliberately absurd) must be fully replaced
    t = (pd.Timestamp("2026-11-20") - pd.Timestamp("2026-08-10")).days / 365.25
    import trade_structures as T
    theo = T.bs_price(100.0, 100.0, t, 0.20, "C")
    row = _iv_row(bid=round(theo - 0.02, 2), ask=round(theo + 0.02, 2), last=theo)
    out = V.recompute_iv(pd.DataFrame([row]))
    assert out["iv"].iloc[0] < 0.99


def test_recompute_iv_empty_frame_is_a_noop():
    empty = pd.DataFrame(columns=["date", "symbol", "expiry", "right",
                                  "strike", "iv", "oi", "volume", "bid",
                                  "ask", "last", "spot"])
    assert V.recompute_iv(empty) is empty


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


def test_big_oi_map_drops_expired_strikes():
    days = ["2026-08-10", "2026-08-11", "2026-08-12"]
    rows = []
    for d in days:
        rows += _rows(d, "GDX", "2026-08-11", 100.0,           # expires mid-panel
                      [("C", 100.0, 0.30, 5000)])
        rows += _rows(d, "GDX", "2027-01-15", 100.0,           # still alive
                      [("C", 110.0, 0.30, 1000)])
    df = pd.DataFrame(rows)
    # as of the last snapshot date itself, the 08-11 expiry is still live (0DTE)
    still_live = V.big_oi_map(df, top=5, asof="2026-08-11")
    assert {"2026-08-11", "2027-01-15"} <= set(still_live["expiry"])
    # a week later (stale build over a weekend), the 08-11 line is gone
    later = V.big_oi_map(df, top=5, asof="2026-08-18")
    assert "2026-08-11" not in set(later["expiry"])
    assert "2027-01-15" in set(later["expiry"])


def test_chain_liquidity_gate():
    def sym_rows(sym, spot, bid, ask, oi, n=6):
        ks = np.linspace(spot * 0.92, spot * 1.08, n)
        return [{"date": "2026-08-10", "symbol": sym, "expiry": "2026-11-20",
                 "right": "C" if k >= spot else "P", "strike": float(k),
                 "iv": 0.3, "oi": oi, "volume": 1, "bid": bid, "ask": ask,
                 "last": 1.0, "spot": spot}
                for k in ks]
    day = pd.DataFrame(
        sym_rows("QQQ", 700.0, 10.0, 10.4, 100_000) +     # tight + deep
        sym_rows("VTV", 224.0, 1.0, 1.9, 1_000) +         # 62% spread, 6k OI
        sym_rows("XLE", 57.0, 0.10, 0.40, 600_000))       # wide but 3.6M OI
    L = V.chain_liquidity(day).set_index("symbol")
    assert bool(L.loc["QQQ", "liquid"])
    assert not bool(L.loc["VTV", "liquid"])               # the VTV rule
    assert bool(L.loc["XLE", "liquid"])                   # deep-OI override
    assert L.loc["VTV", "med_spread"] > 20


def _d1_sig(**over):
    s = {"symbol": "GDX", "family": "up", "event": pd.Timestamp("2026-08-10"),
         "roundtrip": True, "cond_eabs63": 13.0, "mae_q25": -16.0,
         "mfe_med": 11.0, "exit_date": None}
    s.update(over)
    return s


def test_trade_log_illiquid_gets_delta1_with_1pct_risk_sizing():
    log = V.trade_log(_panel_3d(), [_d1_sig()], liquid=set())
    r = log.iloc[0]
    assert r["structure"] == "delta-1 stock" and r["status"] == "open"
    assert r["entry_px"] == pytest.approx(100.0)
    assert r["stop_px"] == pytest.approx(84.0)          # entry * (1 - 16%)
    assert r["tp_px"] == pytest.approx(111.0)
    assert r["weight_pct"] == pytest.approx(100 * V.RISK_NAV / 16.0, abs=0.05)
    assert "1% NAV" in r["rationale"]


def _panel_spots(path):
    """3-day panel whose spot follows `path` (list of 3 prices)."""
    df = _panel_3d()
    for d, px in zip(sorted(df["date"].unique()), path):
        df.loc[df["date"] == d, "spot"] = px
    return df


def test_delta1_replay_stop_and_target():
    stopped = V.trade_log(_panel_spots([100, 92, 80]), [_d1_sig()], liquid=set()).iloc[0]
    assert stopped["status"] == "stopped"
    assert stopped["pnl_pct"] == pytest.approx(-16.0)   # exits AT the stop
    assert stopped["nav_pnl_pct"] == pytest.approx(-V.RISK_NAV, abs=0.01)
    hit = V.trade_log(_panel_spots([100, 105, 115]), [_d1_sig()], liquid=set()).iloc[0]
    assert hit["status"] == "target"
    assert hit["pnl_pct"] == pytest.approx(11.0)


def test_delta1_defaults_without_family_stats():
    sig = _d1_sig(mae_q25=np.nan, mfe_med=np.nan)
    r = V.trade_log(_panel_3d(), [sig], liquid=set()).iloc[0]
    assert r["stop_px"] == pytest.approx(100 * (1 + V.D1_STOP_DEFAULT / 100))
    assert r["tp_px"] == pytest.approx(100 * (1 + V.D1_TP_DEFAULT / 100))


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
