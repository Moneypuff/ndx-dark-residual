"""Tests for ``trade_structures`` -- the suggester and the smile mark.

The suggester's branches are the executable form of the findings docs'
structure-selection table, so each branch is pinned. The pricing claim is
that a leg's mark comes from the *smile* (interpolated IV of live
neighbors -> Black-Scholes) and quotes only bound it -- pinned by marking
a contract whose own quote is absurdly wide.
"""
import numpy as np
import pandas as pd
import pytest

import trade_structures as T


# ---------------------------------------------------------------------------
# suggest_structure
# ---------------------------------------------------------------------------
def test_chaser_rich_with_put_skew_sells_the_put_wing():
    name, legs, why = T.suggest_structure("up", False, "QQQ", 1.5, -4.0)
    assert name == "short put spread"
    assert [(L["right"], L["qty"], L["sigma"]) for L in legs] == \
        [("P", -1, -0.5), ("P", +1, -1.5)]


def test_chaser_fair_owns_the_call_spread():
    name, legs, _ = T.suggest_structure("up", False, "XLF", 1.0, -2.5)
    assert name == "call spread"
    assert [(L["right"], L["qty"]) for L in legs] == [("C", 1), ("C", -1)]
    assert all(L["tenor"] == 183 for L in legs)


def test_roundtripper_gets_the_three_legs():
    name, legs, why = T.suggest_structure("up", True, "GDX", 1.44, +4.3)
    assert name == "dip-financed call spread"
    assert [(L["right"], L["qty"], L["sigma"]) for L in legs] == \
        [("C", +1, 0.0), ("C", -1, +1.0), ("P", -1, -1.0)]
    assert legs[2]["tenor"] == 91 and legs[0]["tenor"] == 183


def test_capitulation_cheap_owns_premium_rich_sells_it():
    cheap, legs, _ = T.suggest_structure("dn", False, "QQQ", 0.9, -3.0)
    assert cheap == "long call" and legs[0]["qty"] == 1
    rich, legs2, _ = T.suggest_structure("dn", False, "SMH", 1.5, -3.0)
    assert rich == "short put spread"


def test_turn_traps_and_turn_structure():
    none, legs, why = T.suggest_structure("turn", False, "XLF", 1.0, 0.0)
    assert none is None and legs == [] and "trap" in why
    name, legs, _ = T.suggest_structure("turn", False, "SMH", 1.0, 0.0)
    assert name == "call spread (long tenor)"
    assert legs[1]["sigma"] == 1.5


def test_missing_context_defaults_are_neutral():
    # nan rich/rr must not crash and must take the fair path, not the rich one
    name, _, _ = T.suggest_structure("up", False, "XLC", np.nan, np.nan)
    assert name == "call spread"


def test_sigma_to_moneyness():
    m = T.sigma_to_moneyness(-1.0, 0.40, 365.25 / 4)
    assert m == pytest.approx(-0.20)          # 40 vol * sqrt(0.25) = 20%


# ---------------------------------------------------------------------------
# bs_price
# ---------------------------------------------------------------------------
def test_bs_parity_and_limits():
    c = T.bs_price(100, 100, 0.5, 0.3, "C")
    p = T.bs_price(100, 100, 0.5, 0.3, "P")
    assert c == pytest.approx(p)                        # ATM, r=0 parity
    assert c == pytest.approx(0.3 * np.sqrt(0.5) * 100 * 0.399, rel=0.01)
    assert T.bs_price(100, 80, 0.0, 0.3, "C") == 20.0   # expiry -> intrinsic
    assert T.bs_price(100, 80, 0.5, np.nan, "P") == 0.0
    assert T.bs_price(100, 100, 1.0, 0.6, "C") > c      # vega positive


# ---------------------------------------------------------------------------
# leg_mark / structure_mark
# ---------------------------------------------------------------------------
T_YRS = (pd.Timestamp("2026-11-20") - pd.Timestamp("2026-08-10")).days / 365.25


def _day(spot=100.0, wide_at=None):
    """Flat 40-vol chain quoted ~+/-5% around each leg's own theoretical
    value; `wide_at` gives that one contract an absurd 0.05/20.00 market."""
    rows = []
    for right, ks in (("P", [70, 80, 90, 95]), ("C", [100, 105, 110, 120])):
        for k in ks:
            theo = T.bs_price(spot, float(k), T_YRS, 0.40, right)
            wide = wide_at == (right, k)
            rows.append({"date": "2026-08-10", "symbol": "GDX",
                         "expiry": "2026-11-20", "right": right, "strike": float(k),
                         "iv": 0.40, "oi": 500, "volume": 10,
                         "bid": 0.05 if wide else round(theo * 0.95, 2),
                         "ask": 20.0 if wide else round(theo * 1.05 + 0.02, 2),
                         "last": theo, "spot": spot})
    return pd.DataFrame(rows)


def test_leg_mark_uses_smile_not_the_wide_quote():
    day = _day(wide_at=("C", 105))
    rows = day[day["expiry"] == "2026-11-20"]
    m = T.leg_mark(rows, 105.0, "C", "2026-08-10", "2026-11-20")
    theo = T.bs_price(100.0, 105.0, (pd.Timestamp("2026-11-20") -
                                     pd.Timestamp("2026-08-10")).days / 365.25,
                      0.40, "C")
    # the 0.05/20.00 quote's mid (~10) is ignored; the smile mark is theo
    assert m["mark_pct"] == pytest.approx(theo, rel=1e-6)
    assert m["iv_used"] == pytest.approx(0.40)
    assert m["spread_pct"] > 100                        # the width is reported


def test_leg_mark_clamps_into_a_tight_quote_only():
    # whole smile legitimately at 90 vol (despike keeps a flat level) but
    # the quotes still price 40 vol: a TIGHT quote bounds the mark...
    rows = _day()
    rows["iv"] = 0.90
    ask = float(rows.loc[(rows["strike"] == 110) & (rows["right"] == "C"),
                         "ask"].iloc[0]) / 100.0 * 100  # $ -> % of spot=100
    m = T.leg_mark(rows, 110.0, "C", "2026-08-10", "2026-11-20")
    assert m["clamped"] and m["mark_pct"] == pytest.approx(ask)
    # ...but a WIDE quote does not: the smile mark stands unclamped
    wide = rows.copy()
    tgt = (wide["strike"] == 110) & (wide["right"] == "C")
    wide.loc[tgt, ["bid", "ask"]] = [0.05, 20.0]
    m2 = T.leg_mark(wide, 110.0, "C", "2026-08-10", "2026-11-20")
    theo = T.bs_price(100.0, 110.0, T_YRS, 0.90, "C")
    assert not m2["clamped"] and m2["mark_pct"] == pytest.approx(theo, rel=1e-6)


def test_despike_drops_the_stale_line():
    ks = np.array([190.0, 194.0, 195.0, 198.0, 199.0, 200.0, 205.0])
    vs = np.array([0.30, 0.41, 0.26, 0.40, 0.24, 0.24, 0.21])   # VTV-style junk
    k2, v2 = T.despike_smile(ks, vs)
    assert 0.41 not in v2 and 0.40 not in v2
    assert 0.26 in v2 and 0.21 in v2                 # real slope survives
    flat_k, flat_v = T.despike_smile(ks, np.full(7, 0.40))
    assert len(flat_v) == 7                          # clean smile untouched


def test_leg_mark_interpolates_past_a_junk_neighbor():
    day = _day()
    rows = day.copy()
    # poison one strike's IV; its own mark must come from the neighbors
    rows.loc[(rows["strike"] == 105) & (rows["right"] == "C"), "iv"] = 0.95
    rows.loc[(rows["strike"] == 105) & (rows["right"] == "C"), ["bid", "ask"]] = [0.0, 0.0]
    m = T.leg_mark(rows, 105.0, "C", "2026-08-10", "2026-11-20")
    assert m["iv_used"] == pytest.approx(0.40, abs=0.01)


def test_structure_mark_signs():
    day = _day()
    legs = [{"expiry": "2026-11-20", "strike": 100.0, "right": "C", "qty": 1},
            {"expiry": "2026-11-20", "strike": 110.0, "right": "C", "qty": -1}]
    net, det = T.structure_mark(day, "GDX", legs, "2026-08-10")
    assert net is not None and len(det) == 2
    long_leg = next(d for d in det if d["qty"] == 1)
    short_leg = next(d for d in det if d["qty"] == -1)
    assert net == pytest.approx(long_leg["mark_pct"] - short_leg["mark_pct"])
    assert net > 0                                     # debit call spread
