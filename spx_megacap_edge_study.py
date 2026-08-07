#!/usr/bin/env python3
"""
Mega-cap edge test: does dark-flow selection beat beta on the HIGH-WEIGHT names?
==============================================================================

Hypothesis (per request): the index-level DIX signal works, and the index is
dollar-weighted -- so it is essentially the mega-caps. If dark flow carries
anything at the single-stock level it should show up in the highest-weight
names, not the long tail. Restrict the universe to the top-N S&P 500 (IVV)
constituents by index weight and ask: can a high-DPI / dip setup produce a small
edge OVER BETA there?

Signals (real-time, per name)
-----------------------------
* dpct   -- SELF-relative DPI percentile within the name's own trailing year
            (top quintile >= 0.80 = "high DIX for this stock").
* trend_63 -- 3-month OLS log-price slope (< 0 = in a downtrend / dip context).

Edge-over-beta metrics (per cutoff, cross-sectionally within the mega-cap set)
-----------------------------------------------------------------------------
For each top-N cutoff we form a DAILY equal-weight basket and score one return
per day (overlapping 21d), then a moving-block-bootstrap 95% CI:

  1. LONG-SHORT  = darkest-quintile minus least-dark-quintile mega-caps -- fully
     market-neutral (no beta), so a non-zero mean is pure dark-flow selection.
  2. SELECTION   = high-DPI mega-caps' forward return MINUS the mega-cap basket's
     own daily mean (within-set cross-sectional demean) -- alpha over "just own
     the mega-caps", which is ~beta-neutral since all are high-cap.
  3. vs SPY      = high-DPI basket's excess over SPY, and the ALL-mega-cap basket's
     excess over SPY (the mega-cap tilt / beta term) -- reported so (2) is read
     as edge-over-beta, not mega-cap tilt.

Each is shown unconditionally and restricted to names in a 3-month downtrend
(the "buy the dip on high DIX" case). Bar to clear: a small but CI-positive
LONG-SHORT / SELECTION mean that survives as N shrinks toward the mega-caps.

Reproduce
---------
    python spx_megacap_edge_study.py --start 2019-01-01 --out spx_megacap_edge.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import spx_xs_dip_dix_study as XS   # build_frames, self_percentile, block_boot_ci, HI/LO

CUTOFFS = (15, 25, 50, 100, 200)
MIN_BASKET = 3          # skip days with fewer than this many names in a leg
HORIZON = 21


def load_megacaps(start, end, cache_dir, workers, max_cut, refresh=False):
    syms, wmap = N.fetch_ishares_holdings(N.IVV_PORTFOLIO_ID, label="IVV S&P 500",
                                          return_weights=True)
    wmap = {t: float(w) for t, w in wmap.items()
            if t and t.isalpha() and np.isfinite(w) and w > 0}
    top = [t for t, _ in sorted(wmap.items(), key=lambda kv: -kv[1])][:max_cut]
    print(f"IVV weights: {len(wmap)} names; building the top {len(top)} by weight "
          f"(largest: {', '.join(top[:8])} ...)", flush=True)
    SP = N.build_universe_panels(top, start, end, workers=workers, cache_dir=cache_dir,
                                 ns="sp500", refresh=refresh, label="mega-cap S&P 500")
    spy = N.load_yahoo_panels(["SPY"], start, end, workers=1, cache_dir=cache_dir,
                              refresh=refresh, label="SPY")["adjclose"]["SPY"]
    return SP["d"], SP["adjclose"], spy, wmap


def _daily(series_by_day):
    a = np.asarray(series_by_day, dtype=float)
    a = a[np.isfinite(a)]
    return a


def _stat(daily, seed=0):
    a = _daily(daily)
    if len(a) == 0:
        return {"days": 0, "mean": np.nan, "hit": np.nan, "ci": None}
    ci = XS.block_boot_ci(a, seed=seed)
    return {"days": int(len(a)), "mean": round(float(a.mean()), 3),
            "hit": round(float((a > 0).mean() * 100)),
            "ci": (round(ci[0], 3), round(ci[1], 3)) if np.isfinite(ci[0]) else None}


def basket_series(sub, mask, col, min_n=MIN_BASKET):
    """One equal-weight basket return per day from `col`, over rows where `mask`."""
    d = sub[mask]
    out = []
    for _, g in d.groupby(level="date"):
        r = g[col].dropna()
        if len(r) >= min_n:
            out.append(float(r.mean()))
    return out


def longshort_series(sub, sig, col, min_n=MIN_BASKET):
    out = []
    for _, g in sub.groupby(level="date"):
        hi = g[g[sig] >= XS.HI][col].dropna()
        lo = g[g[sig] <= XS.LO][col].dropna()
        if len(hi) >= min_n and len(lo) >= min_n:
            out.append(float(hi.mean() - lo.mean()))
    return out


def megacap_edge(long, weights, cutoffs=CUTOFFS, sig="dpct", trend_col="trend_63"):
    names_have = set(long.index.get_level_values("name").unique())
    ranked = [t for t, _ in sorted(weights.items(), key=lambda kv: -kv[1]) if t in names_have]
    rows = []
    for ncut in cutoffs:
        top = ranked[:ncut]
        if len(top) < 5:
            continue
        sub = long[long.index.get_level_values("name").isin(top)].copy()
        # within-mega-cap cross-sectional demean of the forward return (~beta-neutral)
        sub["mc_x"] = sub["fwd"] - sub.groupby(level="date")["fwd"].transform("mean")
        hi = sub[sig] >= XS.HI
        down = sub[trend_col] < 0
        for clab, cmask in [("all-days", pd.Series(True, index=sub.index)),
                            ("in-3mo-downtrend", down)]:
            ls = _stat(longshort_series(sub[cmask.reindex(sub.index).fillna(False)], sig, "mc_x"), seed=1)
            sel = _stat(basket_series(sub, hi & cmask, "mc_x"), seed=2)
            spy_hi = _stat(basket_series(sub, hi & cmask, "spyx"), seed=3)
            spy_all = _stat(basket_series(sub, cmask, "spyx", min_n=max(5, MIN_BASKET)), seed=4)
            rows.append({
                "top_n": ncut, "names": len(top), "cond": clab,
                "ls_mean": ls["mean"], "ls_ci_lo": ls["ci"][0] if ls["ci"] else None,
                "ls_ci_hi": ls["ci"][1] if ls["ci"] else None, "ls_hit": ls["hit"],
                "sel_mean": sel["mean"], "sel_ci_lo": sel["ci"][0] if sel["ci"] else None,
                "sel_ci_hi": sel["ci"][1] if sel["ci"] else None, "sel_days": sel["days"],
                "highDPI_vs_spy": spy_hi["mean"], "allmega_vs_spy": spy_all["mean"],
                "edge_over_beta": (round(spy_hi["mean"] - spy_all["mean"], 3)
                                   if np.isfinite(spy_hi["mean"]) and np.isfinite(spy_all["mean"]) else np.nan),
            })
    return pd.DataFrame(rows)


def _ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None and np.isfinite(lo) else "--"


def report(tab):
    out = []
    for cond in ("all-days", "in-3mo-downtrend"):
        sub = tab[tab["cond"] == cond]
        out.append(f"\n=== {cond} ===")
        out.append(" topN  names |  LONG-SHORT (darkest-least) [95% CI]      | "
                   "SELECTION vs mega basket [95% CI]     | highDPI-vSPY  allmega-vSPY  edge")
        for _, r in sub.iterrows():
            out.append(
                f" {int(r['top_n']):>4}  {int(r['names']):>5} | "
                f"{r['ls_mean']:+7.3f}% {_ci(r['ls_ci_lo'], r['ls_ci_hi']):>20} | "
                f"{r['sel_mean']:+7.3f}% {_ci(r['sel_ci_lo'], r['sel_ci_hi']):>20} | "
                f"{r['highDPI_vs_spy']:+7.3f}%  {r['allmega_vs_spy']:+7.3f}%  {r['edge_over_beta']:+.3f}")
    out.append("\n  LONG-SHORT & SELECTION are ~beta-neutral (within the mega-cap set); a")
    out.append("  CI-positive mean there is a real dark-flow edge over beta. 'edge' = highDPI")
    out.append("  basket's excess-vs-SPY minus the all-mega-cap basket's -- selection on top of tilt.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-cut", type=int, default=max(CUTOFFS),
                    help="build this many top-weight names (the largest cutoff)")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    D, adj, spy, weights = load_megacaps(start, end, args.cache_dir, args.workers,
                                         args.max_cut, refresh=args.refresh)
    long = XS.build_frames(D, adj, spy, fwd_h=args.horizon, trend_windows=(21, 63))
    nnames = long.index.get_level_values("name").nunique()
    ndays = long.index.get_level_values("date").nunique()
    print(f"\nPanel: {len(long):,} (name,day) obs · {nnames} names · {ndays} days "
          f"[{long.index.get_level_values('date').min().date()} -> "
          f"{long.index.get_level_values('date').max().date()}]")
    print(f"Outcome horizon: {args.horizon}d. 'high DIX' = self-relative DPI top quintile.")
    tab = megacap_edge(long, weights)
    print(report(tab))
    if args.out:
        tab.to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
