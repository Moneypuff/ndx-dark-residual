#!/usr/bin/env python3
"""
Structure suggestion and smile-marked pricing for the vol tracker.

Two jobs, both pure and unit-tested:

1. suggest_structure() -- map a live signal's context (family, playbook
   class, how rich ~3M implied is vs the signal-conditional expected move,
   and the risk reversal) to a concrete option structure. Strikes are
   specified in SIGMA-MOVE units (sigma = ATM IV * sqrt(T)), so "-1 sigma"
   means the same thing on a 15-vol and a 47-vol name; they resolve to %
   moneyness and then to listed strikes at suggestion time. The rules are
   EXPECTED_MOVE_FINDINGS.md's structure-selection table made executable.

2. leg_mark() -- price a leg without trusting the closing quote. Closing
   bid/ask widens and single prints go stale, so the mark is BUILT FROM
   THE SMILE: interpolate IV at the leg's strike from the surrounding
   *live* strikes of the same expiry (the same liveness filter the skew
   study uses), price with Black-Scholes (r=0), then clamp into [bid, ask]
   when both sides are live -- the surface supplies the level, the quotes
   only bound it. The mark is a valuation, not an executable price; the
   spread is reported alongside so wide markets are visible.
"""
import math

import numpy as np
import pandas as pd

# families x context -> structure. qty +1 long / -1 short; sigma = strike
# offset in sigma-move units (0 = ATM); tenor in target calendar days.
RICH = 1.2          # implied/conditional ratio above this = premium is rich
FAIR = 1.1          # below this = premium fair-to-cheap vs the signal
PUT_SKEW = -2.0     # rr at/below this = the put wing carries real premium
TURN_TRAPS = {"XLF", "KRE"}   # financial turns are the study's value traps


def _legs(*specs):
    return [{"right": r, "tenor": t, "sigma": s, "qty": q,
             "label": lbl} for r, t, s, q, lbl in specs]


def suggest_structure(family, roundtrip, symbol, rich_ratio, rr):
    """(name, legs, rationale) or (None, [], reason) when no trade.
    `rich_ratio` = implied ~3M move / conditional E|move| 63d (1.0 when
    unknown); `rr` = 3M risk reversal in vol points (0 when unknown)."""
    rich = rich_ratio if np.isfinite(rich_ratio) else 1.0
    rr = rr if np.isfinite(rr) else 0.0

    if family == "turn" and symbol in TURN_TRAPS:
        return None, [], ("financial turn signals were the study's value "
                          "traps (XLF med -7.8%, q25 excursion -36%) -- no trade")
    if family == "up" and roundtrip:
        return ("dip-financed call spread", _legs(
            ("C", 183, 0.0, +1, "own the 6M drift"),
            ("C", 183, +1.0, -1, "sell the wing into the speculative bid"),
            ("P", 91, -1.0, -1, "the paid dip-limit order")),
            "round-tripper: flat median + right tail; never chase full "
            "premium -- own 6M spread, get paid to rest the dip bid")
    if family == "up":
        if rich >= RICH and rr <= PUT_SKEW:
            return ("short put spread", _legs(
                ("P", 91, -0.5, -1, "sell the rich wing under the drift"),
                ("P", 91, -1.5, +1, "tail stays covered")),
                f"premium {rich:.1f}x rich vs the signal move and the put "
                "wing carries it; chaser drift does the rest")
        return ("call spread", _legs(
            ("C", 183, 0.0, +1, "own the 6M drift"),
            ("C", 183, +1.0, -1, "cap where the median path tops")),
            f"premium near fair ({rich:.1f}x); intrinsic-heavy expression "
            "of the chase" + (", short wing sold into a bid skew" if rr > 0 else ""))
    if family == "dn":
        if rich < FAIR:
            return ("long call", _legs(
                ("C", 91, 0.0, +1, "capitulation move is under-priced")),
                f"conditional move exceeds implied ({rich:.1f}x) -- the one "
                "spot where owning premium after the signal is fair")
        return ("short put spread", _legs(
            ("P", 91, -0.5, -1, "monetize spiked IV + positive drift"),
            ("P", 91, -1.5, +1, "tail stays covered")),
            f"capitulation drift with premium {rich:.1f}x rich -- sell the "
            "fear, keep the tail covered")
    # turn (non-financial)
    return ("call spread (long tenor)", _legs(
        ("C", 183, 0.0, +1, "the move lives in the second quarter"),
        ("C", 183, +1.5, -1, "wide cap; turns run")),
        "turns start quiet then widen -- 6M tenor so the structure "
        "survives the flat first quarter")


def sigma_to_moneyness(sigma_units, atm_iv, tenor_days):
    """Strike offset in sigma-move units -> fractional moneyness."""
    return sigma_units * atm_iv * math.sqrt(max(tenor_days, 1) / 365.25)


# ----------------------------------------------------------------------------
# Smile-marked pricing
# ----------------------------------------------------------------------------
def bs_price(spot, strike, t_years, iv, right):
    """Black-Scholes (r=0, q=0) -- the marking model, not a claim about
    carry. Degenerates to intrinsic at t or iv ~ 0."""
    if spot <= 0 or strike <= 0:
        return np.nan
    intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
    if t_years <= 0 or not np.isfinite(iv) or iv <= 0:
        return intrinsic
    v = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    call = spot * N(d1) - strike * N(d2)
    return call if right == "C" else call - spot + strike


def _norm_cdf(x):
    """Vectorized standard normal CDF (Abramowitz & Stegun 7.1.26, a numpy
    stand-in for scipy.special.erf so implied_vol() needs no dependency
    beyond numpy/pandas; |error| < 1.5e-7, plenty for a vol solver)."""
    a1, a2, a3, a4, a5, p = (0.254829592, -0.284496736, 1.421413741,
                              -1.453152027, 1.061405429, 0.3275911)
    ax = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf = 1.0 - poly * np.exp(-ax * ax)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _bs_price_vec(spot, strike, t_years, iv, is_call):
    """Vectorized bs_price (r=0, q=0) -- same model, array-shaped inputs,
    used by implied_vol()'s bisection inner loop."""
    v = np.maximum(iv * np.sqrt(t_years), 1e-12)
    d1 = (np.log(spot / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    call = spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return np.where(is_call, call, call - spot + strike)


def bs_delta(spot, strike, t_years, iv, right):
    """Vectorized Black-Scholes (r=0, q=0) delta: N(d1) for calls,
    N(d1) - 1 for puts -- computed from OUR OWN iv (implied_vol), never a
    vendor greek, so an ORATS-backfilled day and a live-captured day
    carry the same delta convention. NaN wherever iv/t/spot/strike can't
    support a d1 (non-positive, or NaN in)."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t_years = np.asarray(t_years, dtype=float)
    iv = np.asarray(iv, dtype=float)
    is_call = np.asarray(right) == "C"

    ok = (np.isfinite(iv) & (iv > 0) & (t_years > 0) & (spot > 0) & (strike > 0))
    s = np.where(ok, spot, 1.0)
    k = np.where(ok, strike, 1.0)
    v = np.where(ok, iv, 1.0) * np.sqrt(np.where(ok, t_years, 1.0))
    d1 = (np.log(s / k) + 0.5 * v * v) / v
    delta = np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
    return np.where(ok, delta, np.nan)


IV_LO, IV_HI, IV_ITERS = 1e-4, 5.0, 60   # solver bracket: 0.01%-500% vol


def implied_vol(price, spot, strike, t_years, right):
    """Black-Scholes (r=0, q=0) implied vol via bisection, vectorized over
    numpy arrays (a full snapshot history is millions of contract-days).
    This is OUR read on IV rather than trusting each vendor's own
    convention (Yahoo's impliedVolatility, ORATS' quoted mid vol) -- one
    consistent methodology across every source, so a backfilled day and a
    live-captured day carry no vendor seam between them.

    NaN wherever a solution can't be bracketed within [IV_LO, IV_HI]: a
    non-positive price/spot/strike/t_years, or a price outside what the
    model can produce in that vol range (already-expired rows, a crossed
    or otherwise nonsensical quote)."""
    price = np.asarray(price, dtype=float)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t_years = np.asarray(t_years, dtype=float)
    is_call = np.asarray(right) == "C"

    ok = np.isfinite(price) & (price > 0) & (spot > 0) & (strike > 0) & (t_years > 0)
    s = np.where(ok, spot, 1.0)
    k = np.where(ok, strike, 1.0)
    t = np.where(ok, t_years, 1.0)
    p = np.where(ok, price, 1.0)

    lo = np.full_like(s, IV_LO)
    hi = np.full_like(s, IV_HI)
    for _ in range(IV_ITERS):
        mid = (lo + hi) / 2.0
        too_high = _bs_price_vec(s, k, t, mid, is_call) > p
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)

    bracketed = ((p > _bs_price_vec(s, k, t, IV_LO, is_call)) &
                (p < _bs_price_vec(s, k, t, IV_HI, is_call)))
    return np.where(ok & bracketed, (lo + hi) / 2.0, np.nan)


def despike_smile(ks, vs, tol_floor=0.06, tol_frac=0.25, window=7):
    """Drop IV points that spike away from their rolling-median neighbors
    -- thin chains carry stale lines whose IV sits 10+ vol points off the
    strikes either side (e.g. a 40-vol print between 24-vol neighbors),
    and interpolating through them poisons every mark nearby. The window
    is wide (7) so a *pair* of adjacent junk lines can't dominate its own
    median; tolerance scales with the local level so genuinely steep
    high-vol wings survive."""
    if len(vs) < window:
        return ks, vs
    s = pd.Series(vs)
    med = s.rolling(window, center=True, min_periods=3).median()
    keep = ((s - med).abs() <= np.maximum(tol_floor, tol_frac * med)).to_numpy()
    return ks[keep], vs[keep]


def smile_points(expiry_rows, spot):
    """(strikes, ivs) of the live, despiked OTM smile inside one
    (symbol, expiry) day-group of snapshot rows -- puts below spot, calls
    at/above, only contracts with a bid or OI (dead wings carry garbage
    IVs), then despike_smile() for the stale lines that pass liveness."""
    r = expiry_rows
    otm = r[(((r["right"] == "P") & (r["strike"] < spot)) |
             ((r["right"] == "C") & (r["strike"] >= spot))) &
            ((r["bid"].fillna(0) > 0) | (r["oi"].fillna(0) > 0))]
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    return despike_smile(otm["strike"].to_numpy(dtype=float),
                         otm["iv"].to_numpy(dtype=float))


CLAMP_MAX_SPREAD = 25.0   # only a tight market (spread% <= this) bounds the mark


def leg_mark(expiry_rows, strike, right, asof, expiry):
    """Smile-based mark for one leg, % of spot. Returns dict with the
    mark, the IV used, the quote mid/spread, and whether the surface mark
    was clamped into a live bid/ask. Clamping only applies when the quote
    is TIGHT (spread <= CLAMP_MAX_SPREAD%): a wide or stale market is
    exactly the case the smile mark exists to overrule."""
    if expiry_rows.empty:
        return None
    spot = float(expiry_rows["spot"].iloc[0])
    ks, vs = smile_points(expiry_rows, spot)
    t_years = max((pd.Timestamp(expiry) - pd.Timestamp(asof)).days, 0) / 365.25
    iv = float(np.interp(strike, ks, vs)) if len(ks) >= 4 else np.nan
    px = bs_price(spot, strike, t_years, iv, right)

    row = expiry_rows[(expiry_rows["strike"] == strike) &
                      (expiry_rows["right"] == right)]
    bid = float(row["bid"].iloc[0]) if len(row) and pd.notna(row["bid"].iloc[0]) else 0.0
    ask = float(row["ask"].iloc[0]) if len(row) and pd.notna(row["ask"].iloc[0]) else 0.0
    mid = (bid + ask) / 2 if bid > 0 and ask >= bid else np.nan
    spread = (ask - bid) / mid * 100 if np.isfinite(mid) and mid > 0 else np.nan
    clamped = False
    if not np.isfinite(px):
        px = mid                       # thin smile: quote is all we have
    elif (np.isfinite(spread) and spread <= CLAMP_MAX_SPREAD
          and not (bid <= px <= ask)):
        px, clamped = float(np.clip(px, bid, ask)), True
    if not np.isfinite(px):
        return None
    return {"mark_pct": px / spot * 100, "iv_used": iv,
            "mid_pct": mid / spot * 100 if np.isfinite(mid) else np.nan,
            "spread_pct": spread, "clamped": clamped, "spot": spot}


def structure_mark(day_df, symbol, legs, asof):
    """Net mark of a resolved structure (list of legs carrying expiry/
    strike/right/qty), % of spot. Positive = debit. None if any leg is
    unmarkable."""
    total, details = 0.0, []
    for L in legs:
        rows = day_df[(day_df["symbol"] == symbol) &
                      (day_df["expiry"] == L["expiry"])]
        m = leg_mark(rows, L["strike"], L["right"], asof, L["expiry"])
        if m is None:
            return None, []
        total += L["qty"] * m["mark_pct"]
        details.append({**L, **m})
    return total, details
