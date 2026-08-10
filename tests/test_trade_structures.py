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
# bs_delta / delta_strike
# ---------------------------------------------------------------------------
def test_bs_delta_basics():
    assert T.bs_delta(100, 100, 0.25, 0.30, "C") > 0.5      # ATM call slightly ITM-forward
    assert T.bs_delta(100, 100, 0.25, 0.30, "P") == pytest.approx(
        T.bs_delta(100, 100, 0.25, 0.30, "C") - 1.0)        # parity
    assert T.bs_delta(100, 40, 1.0, 0.30, "C") > 0.99       # deep ITM
    assert np.isnan(T.bs_delta(100, 100, 0.0, 0.30, "C"))


def test_delta_strike_matches_flat_smile_closed_form():
    iv, tyr, spot = 0.40, 0.25, 100.0
    ks = np.linspace(50, 160, 45)
    vs = np.full_like(ks, iv)
    # flat smile: N(d1)=0.25 -> ln(S/K) = z25*iv*sqrt(T) - iv^2 T/2, z25=-0.6745
    z25 = -0.674489750196
    kc_true = spot * np.exp(-(z25 * iv * np.sqrt(tyr) - 0.5 * iv * iv * tyr))
    kc = T.delta_strike(ks, vs, spot, tyr, +0.25, "C")
    assert kc == pytest.approx(kc_true, rel=2e-3)
    assert T.bs_delta(spot, kc, tyr, iv, "C") == pytest.approx(0.25, abs=5e-3)
    kp = T.delta_strike(ks, vs, spot, tyr, -0.25, "P")
    assert T.bs_delta(spot, kp, tyr, iv, "P") == pytest.approx(-0.25, abs=5e-3)
    assert kp < spot < kc


def test_delta_strike_outside_smile_is_nan():
    ks = np.array([98.0, 99.0, 100.0, 101.0, 102.0])   # too narrow for 25d
    vs = np.full_like(ks, 0.40)
    assert np.isnan(T.delta_strike(ks, vs, 100.0, 0.25, +0.25, "C"))


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
    theo = T.bs_price(100.0, 105.0, T_YRS, 0.40, "C")
    # the 0.05/20.00 quote's mid (~10) is ignored; the smile mark ~ theo
    assert m["mark_pct"] == pytest.approx(theo, rel=0.02)
    assert m["iv_used"] == pytest.approx(0.40, abs=0.01)
    assert m["spread_pct"] > 100                        # the width is reported


def test_leg_mark_clamps_into_a_tight_quote_only():
    # neighbors quote 90-vol mids; the target strike's own market is a
    # TIGHT 40-vol quote -> the interpolated smile mark clamps into it...
    rows = _day()
    tgt = (rows["strike"] == 110) & (rows["right"] == "C")
    for i, r in rows[~tgt].iterrows():
        theo = T.bs_price(100.0, r["strike"], T_YRS, 0.90, r["right"])
        rows.loc[i, ["bid", "ask"]] = [round(theo * 0.98, 2),
                                       round(theo * 1.02 + 0.02, 2)]
    ask = float(rows.loc[tgt, "ask"].iloc[0])           # still 40-vol based
    m = T.leg_mark(rows, 110.0, "C", "2026-08-10", "2026-11-20")
    assert m["clamped"] and m["mark_pct"] == pytest.approx(ask, rel=1e-6)
    # ...but with a WIDE own-quote the smile mark stands unclamped
    wide = rows.copy()
    wide.loc[tgt, ["bid", "ask"]] = [0.05, 20.0]
    m2 = T.leg_mark(wide, 110.0, "C", "2026-08-10", "2026-11-20")
    theo90 = T.bs_price(100.0, 110.0, T_YRS, 0.90, "C")
    assert not m2["clamped"] and m2["mark_pct"] == pytest.approx(theo90, rel=0.05)


def test_implied_forward_recovers_carry():
    # chain priced off F = 104, D = 0.97: parity must recover both, and
    # re-inverted call/put IVs at the same strike must agree again
    F, D, iv, spot = 104.0, 0.97, 0.35, 100.0
    rows = []
    for k in np.linspace(80, 125, 19):
        for right in ("C", "P"):
            px = D * T.bs_price(F, float(k), T_YRS, iv, right)
            rows.append({"date": "2026-08-10", "symbol": "X",
                         "expiry": "2026-11-20", "right": right,
                         "strike": float(k), "iv": 0.99, "oi": 100,
                         "volume": 1, "bid": round(px * 0.99, 4),
                         "ask": round(px * 1.01 + 0.001, 4), "last": px,
                         "spot": spot})
    day = pd.DataFrame(rows)
    f_hat, d_hat = T.implied_forward(day, spot)
    assert f_hat == pytest.approx(F, rel=1e-3)
    assert d_hat == pytest.approx(D, abs=0.01)
    ks, vs, f2, _ = T.forward_smile(day, spot, T_YRS)
    # the garbage feed IV (0.99) is ignored; recovered smile is flat 0.35
    assert np.allclose(vs, iv, atol=0.01)
    assert f2 == pytest.approx(F, rel=1e-3)


def test_invert_iv_round_trip():
    px = 0.97 * T.bs_price(104.0, 110.0, 0.5, 0.42, "C")
    assert T.invert_iv(px, 104.0, 110.0, 0.5, "C", 0.97) == pytest.approx(
        0.42, abs=1e-3)
    assert np.isnan(T.invert_iv(0.0, 104.0, 110.0, 0.5, "C"))


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
