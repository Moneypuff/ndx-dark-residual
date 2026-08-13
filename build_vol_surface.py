#!/usr/bin/env python3
"""
3D vol surface viewer -- gridded IV surfaces from the optsnap capture.

Consumes the same daily chain snapshots as build_vol_tracker.py
(snapshot_option_chains.py, the optsnap-data branch) and, for each
(symbol, snapshot date), stacks every captured expiry's live despiked
smile (trade_structures.smile_points) onto a common moneyness grid --
the fixed grid is what makes surfaces comparable across dates and
across symbols of very different vol level. The browser only projects
and draws the pre-gridded numbers; no financial math happens in JS.

Grid decisions:
  * moneyness grid: 60%-140% of spot in 2.5pt steps (33 points) --
    covers the +/-25% regular capture window fully and the meat of the
    +/-65% January-LEAP window.
  * no extrapolation: a grid cell outside the smile's own strike range
    is left null. A flat-extrapolated wing looks like data and isn't;
    the renderer draws an honest hole instead.
  * up to 14 expiries per surface (nearest-first by DTE), capping
    payload size while keeping every monthly plus the near LEAPs.
  * the payload carries the last 10 snapshot days -- the scrubber's
    replay window. Rebuild with --days for more history.

Model views (per symbol/date, when the prior snapshot day has a surface
for that symbol): a one-day-ahead prediction of today's grid from
yesterday's, under each of three smile-dynamics assumptions --

  * sticky strike: IV(K) is unchanged -- today's cell samples yesterday's
    smile at the shifted moneyness m*(S_t/S_{t-1}).
  * sticky delta: IV(moneyness) is unchanged -- yesterday's row verbatim.
    (Implemented as sticky-*moneyness*, the standard practical proxy for
    sticky delta; a true delta bucketing needs a full BS delta inversion
    and adds nothing at daily granularity here.)
  * hybrid: blends the two by distance from ATM in sigma-move units
    (W_LO/W_HI below) -- near ATM the smile is closer to sticky-delta,
    in the wings closer to sticky-strike, per the empirical finding this
    view exists to check.

Predictions never see today's actual smile -- only yesterday's grid plus
today's spot -- so the residual (observed - predicted) is an honest,
falsifiable miss, not something tuned to look good.

    python build_vol_surface.py --snap-dir optsnap --docs-out docs/vol_surface.html
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import trade_structures as T
from build_vol_tracker import atm_iv, load_snapshots

M_GRID = np.arange(0.60, 1.4001, 0.025)   # 33 moneyness points, 60%..140%
MAX_EXPIRIES = 14      # per symbol/date, nearest-first by DTE
SURF_DAYS = 10         # snapshot days carried in the payload (scrubber window)

# Hybrid model blend, in sigma-move units of distance from ATM (sigma =
# atm_iv * sqrt(T)): pure sticky-delta inside W_LO, pure sticky-strike
# beyond W_HI, smoothstepped between. Calibration knobs for a later
# phase -- not fit from data here.
W_LO = 0.5
W_HI = 1.5


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def surface_grid(expiry_rows, spot):
    """One (date, symbol, expiry) group of snapshot rows -> the live
    despiked smile (trade_structures.smile_points) resampled onto
    M_GRID, in spot-relative moneyness. None when the smile is too
    thin or one-sided (same gate atm_iv uses). Grid cells outside the
    smile's own strike range are NaN -- np.interp would otherwise
    flat-extrapolate them, which is not data."""
    ks, vs = T.smile_points(expiry_rows, spot)
    if len(ks) < 4 or not (ks[0] < spot < ks[-1]):
        return None
    strikes = M_GRID * spot
    iv = np.interp(strikes, ks, vs)
    iv[(strikes < ks[0]) | (strikes > ks[-1])] = np.nan
    return iv


def symbol_surface(day_df, symbol, asof):
    """One snapshot day, one symbol -> the surface dict, or None when
    no expiry survives surface_grid. Expiries are capped to
    MAX_EXPIRIES (nearest-first) and returned sorted by DTE ascending."""
    g = day_df[day_df["symbol"] == symbol]
    if g.empty:
        return None
    expiries = sorted(g["expiry"].unique(), key=lambda e: pd.Timestamp(e))
    rows = []
    for exp in expiries:
        er = g[g["expiry"] == exp]
        spot = float(er["spot"].iloc[0])
        grid = surface_grid(er, spot)
        if grid is None:
            continue
        dte = max((pd.Timestamp(exp) - pd.Timestamp(asof)).days, 1)
        a = atm_iv(er)
        rows.append((dte, exp, spot, a, grid))
        if len(rows) >= MAX_EXPIRIES:
            break
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    spot = rows[0][2]
    return {
        "spot": spot,
        "expiries": [r[1] for r in rows],
        "dtes": [r[0] for r in rows],
        "atm": [None if not np.isfinite(r[3]) else round(float(r[3]), 4)
                for r in rows],
        "iv": [[None if not np.isfinite(v) else round(float(v), 4)
                for v in r[4]] for r in rows],
    }


def hybrid_weight(m, atm_iv_prev, dte_prev):
    """Hybrid-model blend weight at grid moneyness `m`: 0 = pure sticky
    delta, 1 = pure sticky strike. Distance from ATM in sigma-move units
    (sigma = atm_iv_prev * sqrt(dte_prev/365.25)), smoothstepped between
    W_LO and W_HI. A non-positive or non-finite sigma (no usable prior
    ATM) can't place a distance -- treated as fully in the wing (1.0)
    rather than guessing."""
    if not np.isfinite(atm_iv_prev) or atm_iv_prev <= 0 or dte_prev <= 0:
        return 1.0
    sigma = atm_iv_prev * math.sqrt(dte_prev / 365.25)
    if sigma <= 0:
        return 1.0
    d = abs(m - 1.0) / sigma
    if d <= W_LO:
        return 0.0
    if d >= W_HI:
        return 1.0
    u = (d - W_LO) / (W_HI - W_LO)
    return 3 * u * u - 2 * u ** 3


def predict_row_strike(prev_row, ratio):
    """Sticky-strike prediction of one grid row from yesterday's
    (`prev_row`, payload form: list of float/None): today's cell at
    moneyness m reads yesterday's smile at m*ratio (ratio =
    S_t/S_{t-1}), linearly interpolated over yesterday's own contiguous
    non-null block -- null outside that block's span, same no-
    extrapolation rule as surface_grid. None (all-null row) when fewer
    than 2 cells are usable."""
    block = [(float(M_GRID[i]), v) for i, v in enumerate(prev_row) if v is not None]
    if len(block) < 2:
        return [None] * len(M_GRID)
    ks = np.array([b[0] for b in block])
    vs = np.array([b[1] for b in block])
    targets = M_GRID * ratio
    iv = np.interp(targets, ks, vs)
    iv[(targets < ks[0]) | (targets > ks[-1])] = np.nan
    return [None if not np.isfinite(v) else round(float(v), 4) for v in iv]


def _nearest_atm_cell(row):
    """Fallback ATM level when a row's own atm_iv is null: the row's
    grid cell nearest moneyness 1.0, searching outward. None if the
    whole row is null."""
    center = int(np.argmin(np.abs(M_GRID - 1.0)))
    if row[center] is not None:
        return row[center]
    for d in range(1, len(row)):
        for idx in (center - d, center + d):
            if 0 <= idx < len(row) and row[idx] is not None:
                return row[idx]
    return None


def model_grids(prev_surf, cur_surf):
    """(symbol, date t)'s "models" payload: sticky-strike / sticky-delta
    / hybrid one-day-ahead predictions of `cur_surf`'s grid, built ONLY
    from `prev_surf` (day t-1's surface) and the two spots -- today's
    actual smile never leaks into its own prediction. Rows align to
    cur_surf's expiries; an expiry absent from prev_surf predicts null
    throughout. None when every predicted cell is null (nothing to
    show)."""
    n = len(M_GRID)
    prev_index = {exp: i for i, exp in enumerate(prev_surf["expiries"])}
    ratio = cur_surf["spot"] / prev_surf["spot"]
    strike_rows, delta_rows, hybrid_rows = [], [], []
    any_data = False
    for exp in cur_surf["expiries"]:
        k = prev_index.get(exp)
        if k is None:
            strike_rows.append([None] * n)
            delta_rows.append([None] * n)
            hybrid_rows.append([None] * n)
            continue
        prev_row = prev_surf["iv"][k]
        strike_row = predict_row_strike(prev_row, ratio)
        delta_row = list(prev_row)
        atm_prev = prev_surf["atm"][k]
        if atm_prev is None:
            atm_prev = _nearest_atm_cell(prev_row)
        dte_prev = prev_surf["dtes"][k]
        hybrid_row = []
        for j, m in enumerate(M_GRID):
            sv, dv = strike_row[j], delta_row[j]
            if sv is None or dv is None or atm_prev is None:
                hybrid_row.append(None)
                continue
            w = hybrid_weight(float(m), atm_prev, dte_prev)
            hybrid_row.append(round(w * sv + (1 - w) * dv, 4))
        strike_rows.append(strike_row)
        delta_rows.append(delta_row)
        hybrid_rows.append(hybrid_row)
        if any(v is not None for v in strike_row + delta_row + hybrid_row):
            any_data = True
    if not any_data:
        return None
    return {"strike": strike_rows, "delta": delta_rows, "hybrid": hybrid_rows}


def build_payload(df, days=SURF_DAYS):
    """Full snapshot frame -> the template payload: the shared
    moneyness grid, the carried date window (last `days` snapshot
    days, ascending), the symbols that produced at least one surface
    anywhere in that window, and surfaces[symbol][date] for every
    (symbol, date) pair that produced one (missing = no capture that
    day, handled client-side)."""
    dates = sorted(df["date"].unique())[-days:]
    surfaces = {}
    symbols = set()
    # The first carried date never gets models, even if older snapshots
    # exist on disk outside this window -- the payload window is the
    # whole universe the page can show, so predicting across its edge
    # from data the page never displays would be unverifiable there.
    prev_by_symbol = {}
    for d in dates:
        day_df = df[df["date"] == d]
        for sym in sorted(day_df["symbol"].unique()):
            surf = symbol_surface(day_df, sym, d)
            if surf is None:
                continue
            if sym in prev_by_symbol:
                models = model_grids(prev_by_symbol[sym], surf)
                if models is not None:
                    surf["models"] = models
            prev_by_symbol[sym] = surf
            surfaces.setdefault(sym, {})[d] = surf
            symbols.add(sym)
    return {
        "m_grid": [round(float(m), 4) for m in M_GRID],
        "dates": dates,
        "symbols": sorted(symbols),
        "surfaces": surfaces,
    }


# ----------------------------------------------------------------------------
def render_page(template, docs_out, ctx, surf):
    """docs/vol_surface.html -- payloads injected into the template the
    same way the other pages do it."""
    body = (Path(template).read_text(encoding="utf-8")
            .replace("/*__CTX__*/", json.dumps(ctx, separators=(",", ":")))
            .replace("/*__SURF__*/", json.dumps(surf, separators=(",", ":"))))
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>Vol Surface</title></head>'
           '<body>\n' + body + '\n</body></html>\n')
    out = Path(docs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc) // 1024} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap-dir", default="optsnap")
    ap.add_argument("--docs-out", default="docs/vol_surface.html")
    ap.add_argument("--template", default="vol_surface_template.html")
    ap.add_argument("--days", type=int, default=SURF_DAYS)
    args = ap.parse_args()

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = load_snapshots(args.snap_dir)
    if df.empty:
        print(f"no snapshots under {args.snap_dir}")
        render_page(args.template, args.docs_out, {
            "built": built, "days": 0,
            "note": "No chain capture yet -- the optsnap workflow appends "
                    "the first snapshot after the next close."}, {})
        return

    payload = build_payload(df, days=args.days)
    n_surf = sum(len(v) for v in payload["surfaces"].values())
    print(f"{len(payload['dates'])} day(s), {len(payload['symbols'])} symbol(s), "
          f"{n_surf} surface(s)")

    render_page(args.template, args.docs_out, {
        "built": built,
        "days": len(payload["dates"]),
        "first": payload["dates"][0] if payload["dates"] else None,
        "last": payload["dates"][-1] if payload["dates"] else None,
        "nSymbols": len(payload["symbols"]),
        "note": None,
    }, payload)


if __name__ == "__main__":
    main()
