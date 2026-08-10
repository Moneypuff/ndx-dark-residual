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


def bs_delta(spot, strike, t_years, iv, right):
    """Black-Scholes delta (r=q=0): N(d1) for calls, N(d1)-1 for puts.
    Pass the implied forward as `spot` for a Black-76 (forward) delta."""
    if (spot <= 0 or strike <= 0 or t_years <= 0 or not np.isfinite(iv)
            or iv <= 0):
        return np.nan
    v = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * v * v) / v
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return nd1 if right == "C" else nd1 - 1.0


def implied_forward(expiry_rows, spot, moneyness=0.10, max_spread=40.0):
    """(forward, discount) for one expiry, extracted from the chain itself:
    put-call parity says C - P = D*(F - K), so regressing the tight-quoted
    call-put mid differences on strike gives D (the slope, negated) and F
    (the zero-crossing) with no rate or dividend assumptions. Falls back
    to (spot, 1.0) when fewer than 3 usable pairs exist or the fit is
    implausible -- Yahoo's own long-dated IVs are inverted off a spot-like
    forward, which splits same-strike call/put IVs by many vol points;
    this is the correction."""
    r = expiry_rows
    live = r[(r["bid"].fillna(0) > 0) & (r["ask"] >= r["bid"])].copy()
    if not len(live) or not spot:
        return spot, 1.0
    live["mid"] = (live["bid"] + live["ask"]) / 2
    live["spr"] = (live["ask"] - live["bid"]) / live["mid"] * 100
    live = live[(live["spr"] <= max_spread) &
                ((live["strike"] / spot - 1).abs() <= moneyness)]
    piv = live.pivot_table(index="strike", columns="right", values="mid",
                           aggfunc="last").dropna()
    if len(piv) < 3:
        return spot, 1.0
    k = piv.index.to_numpy(dtype=float)
    cp = (piv["C"] - piv["P"]).to_numpy(dtype=float)
    slope, intercept = np.polyfit(k, cp, 1)
    disc, fwd = -slope, intercept / -slope if slope < 0 else (np.nan, np.nan)
    if not (np.isfinite(disc) and np.isfinite(fwd)
            and 0.6 < disc <= 1.05 and 0.7 < fwd / spot < 1.3):
        return spot, 1.0
    return float(fwd), float(min(disc, 1.0))


def invert_iv(price, forward, strike, t_years, right, discount=1.0):
    """Black-76 implied vol from an option price by bisection (the price is
    discounted; undiscount via `discount`)."""
    if price <= 0 or t_years <= 0 or forward <= 0 or strike <= 0:
        return np.nan
    tgt = price / discount
    intrinsic = max(forward - strike, 0.0) if right == "C" else max(strike - forward, 0.0)
    if tgt <= intrinsic:
        return np.nan
    lo, hi = 0.005, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(forward, strike, t_years, mid, right) > tgt:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def delta_strike(ks, vs, spot, t_years, target, right):
    """Strike on the interpolated smile where the BS delta equals `target`
    (e.g. +0.25 for the 25-delta call, -0.25 for the 25-delta put) --
    smile-consistent: each candidate strike is priced at its own
    interpolated IV. Solved on a dense grid inside the smile's strike
    range (delta is monotone in strike); NaN when the target delta falls
    outside the live smile."""
    if len(ks) < 4:
        return np.nan
    side = ks[(ks >= spot)] if right == "C" else ks[(ks <= spot)]
    if len(side) < 2:
        return np.nan
    grid = np.linspace(side.min(), side.max(), 200)
    deltas = np.array([bs_delta(spot, k, t_years, float(np.interp(k, ks, vs)),
                                right) for k in grid])
    ok = np.isfinite(deltas)
    grid, deltas = grid[ok], deltas[ok]
    if len(grid) < 2 or not (deltas.min() <= target <= deltas.max()):
        return np.nan
    order = np.argsort(deltas)
    return float(np.interp(target, deltas[order], grid[order]))


def smile_points(expiry_rows, spot):
    """(strikes, ivs) of the live, despiked OTM smile inside one
    (symbol, expiry) day-group of snapshot rows -- puts below spot, calls
    at/above, only contracts with a bid or OI (dead wings carry garbage
    IVs), then despike_smile() for the stale lines that pass liveness.
    Uses the feed's own IVs; prefer forward_smile() wherever IV *levels*
    are compared across strikes or tenors."""
    r = expiry_rows
    otm = r[(((r["right"] == "P") & (r["strike"] < spot)) |
             ((r["right"] == "C") & (r["strike"] >= spot))) &
            ((r["bid"].fillna(0) > 0) | (r["oi"].fillna(0) > 0))]
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    return despike_smile(otm["strike"].to_numpy(dtype=float),
                         otm["iv"].to_numpy(dtype=float))


def forward_smile(expiry_rows, spot, t_years, max_spread=40.0):
    """(strikes, ivs, forward, discount): the OTM smile with IVs
    RE-INVERTED against the chain's own implied forward (Black-76).

    Yahoo's IV field is inverted off a spot-like forward, which splits
    same-strike call/put IVs by +3.5 vol pts at ~3M up to ~9 at LEAPs and
    contaminates any cross-tenor skew read. Here: OTM side chosen at the
    implied forward; contracts with a tight live quote get iv from their
    own mid via invert_iv(); quoteless-but-open contracts fall back to
    the feed IV; then despike."""
    F, D = implied_forward(expiry_rows, spot)
    r = expiry_rows
    otm = r[(((r["right"] == "P") & (r["strike"] < F)) |
             ((r["right"] == "C") & (r["strike"] >= F))) &
            ((r["bid"].fillna(0) > 0) | (r["oi"].fillna(0) > 0))]
    pts = []
    for _, c in otm.sort_values("strike").iterrows():
        bid, ask = c["bid"] or 0, c["ask"] or 0
        iv = np.nan
        if bid > 0 and ask >= bid:
            mid = (bid + ask) / 2
            if (ask - bid) / mid * 100 <= max_spread:
                iv = invert_iv(mid, F, float(c["strike"]), t_years,
                               c["right"], D)
        if not np.isfinite(iv):
            iv = c["iv"] if np.isfinite(c["iv"] or np.nan) else np.nan
        if np.isfinite(iv) and 0.01 < iv < 5.0:
            pts.append((float(c["strike"]), float(iv)))
    if not pts:
        return np.array([]), np.array([]), F, D
    ks, vs = despike_smile(np.array([p[0] for p in pts]),
                           np.array([p[1] for p in pts]))
    return ks, vs, F, D


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
    t_years = max((pd.Timestamp(expiry) - pd.Timestamp(asof)).days, 0) / 365.25
    ks, vs, F, D = forward_smile(expiry_rows, spot, t_years)
    iv = float(np.interp(strike, ks, vs)) if len(ks) >= 4 else np.nan
    px = D * bs_price(F, strike, t_years, iv, right)

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
