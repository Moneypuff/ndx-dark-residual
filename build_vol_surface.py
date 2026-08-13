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

    python build_vol_surface.py --snap-dir optsnap --docs-out docs/vol_surface.html
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import trade_structures as T
from build_vol_tracker import atm_iv, load_snapshots

M_GRID = np.arange(0.60, 1.4001, 0.025)   # 33 moneyness points, 60%..140%
MAX_EXPIRIES = 14      # per symbol/date, nearest-first by DTE
SURF_DAYS = 10         # snapshot days carried in the payload (scrubber window)


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
    for d in dates:
        day_df = df[df["date"] == d]
        for sym in sorted(day_df["symbol"].unique()):
            surf = symbol_surface(day_df, sym, d)
            if surf is None:
                continue
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
