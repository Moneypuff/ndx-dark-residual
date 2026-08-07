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
SPLIT_DATE = "2023-01-01"   # in-sample (< split) vs out-of-sample (>= split)


def add_capm_alpha(long, adj, spy, horizon, beta_win=252, beta_min=120):
    """Add a per-(name,day) forward CAPM ALPHA column to `long`:

        beta_i,t  = cov(r_i, r_spy)/var(r_spy) over the trailing `beta_win`
                    DAILY returns (min `beta_min`) -- estimated real-time, no
                    look-ahead.
        alpha_i,t = fwd_ret_i  -  beta_i,t * fwd_ret_spy   (both h-day forward, %)

    Alpha is the beta-adjusted forward return -- what's left after the stock's own
    market exposure is priced out, so a positive high-DPI alpha is a real edge
    over beta, not a high-beta name riding an up market."""
    cols = adj.columns
    r = adj.pct_change()
    rs = spy.reindex(adj.index).pct_change()
    Er = r.rolling(beta_win, min_periods=beta_min).mean()
    Ers = rs.rolling(beta_win, min_periods=beta_min).mean()
    Errs = r.mul(rs, axis=0).rolling(beta_win, min_periods=beta_min).mean()
    Erss = (rs * rs).rolling(beta_win, min_periods=beta_min).mean()
    cov = Errs.sub(Er.mul(Ers, axis=0))
    var = (Erss - Ers * Ers)
    beta = cov.div(var, axis=0)
    fwd = N.compute_forward_return(adj, horizon)
    spy_fwd = (spy.reindex(adj.index).shift(-horizon) / spy.reindex(adj.index) - 1.0) * 100.0
    alpha = fwd.sub(beta.mul(spy_fwd, axis=0))
    long = long.copy()
    long["beta"] = beta.stack().reindex(long.index)
    long["alpha"] = alpha.stack().reindex(long.index)
    return long


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
            row = {
                "top_n": ncut, "names": len(top), "cond": clab,
                "ls_mean": ls["mean"], "ls_ci_lo": ls["ci"][0] if ls["ci"] else None,
                "ls_ci_hi": ls["ci"][1] if ls["ci"] else None, "ls_hit": ls["hit"],
                "sel_mean": sel["mean"], "sel_ci_lo": sel["ci"][0] if sel["ci"] else None,
                "sel_ci_hi": sel["ci"][1] if sel["ci"] else None, "sel_days": sel["days"],
                "highDPI_vs_spy": spy_hi["mean"], "allmega_vs_spy": spy_all["mean"],
                "edge_over_beta": (round(spy_hi["mean"] - spy_all["mean"], 3)
                                   if np.isfinite(spy_hi["mean"]) and np.isfinite(spy_all["mean"]) else np.nan),
            }
            if "alpha" in sub.columns:
                a_lo = _stat(basket_series(sub, hi & cmask, "alpha"), seed=5)
                a_ls = _stat(longshort_series(sub[cmask.reindex(sub.index).fillna(False)], sig, "alpha"), seed=6)
                row.update({
                    "alpha_lo": a_lo["mean"], "alpha_lo_ci_lo": a_lo["ci"][0] if a_lo["ci"] else None,
                    "alpha_lo_ci_hi": a_lo["ci"][1] if a_lo["ci"] else None, "alpha_lo_hit": a_lo["hit"],
                    "alpha_ls": a_ls["mean"], "alpha_ls_ci_lo": a_ls["ci"][0] if a_ls["ci"] else None,
                    "alpha_ls_ci_hi": a_ls["ci"][1] if a_ls["ci"] else None,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _slice(long, lo=None, hi=None):
    d = long.index.get_level_values("date")
    m = np.ones(len(long), dtype=bool)
    if lo is not None:
        m &= np.asarray(d >= pd.Timestamp(lo))
    if hi is not None:
        m &= np.asarray(d < pd.Timestamp(hi))
    return long[m]


def oos_report(long, weights, split=SPLIT_DATE, cutoffs=CUTOFFS):
    """Same configuration evaluated in-sample (< split) vs out-of-sample (>=
    split). The honest test of the suggestive full-sample cells: do they
    reproduce, or were they a pre-split artifact?"""
    pre = megacap_edge(_slice(long, hi=split), weights, cutoffs)
    post = megacap_edge(_slice(long, lo=split), weights, cutoffs)
    pre["period"], post["period"] = f"IS(<{split[:4]})", f"OOS(>={split[:4]})"
    out = [f"OUT-OF-SAMPLE SPLIT at {split}:  IS = 2019-2022, OOS = 2023-2026",
           "  (SELECTION = high-DPI mega-caps demeaned within the set; ~beta-neutral)"]
    for cond in ("all-days", "in-3mo-downtrend"):
        out.append(f"\n=== {cond} ===")
        out.append(" topN |     IS selection [95% CI]            |    OOS selection [95% CI]           "
                   "|  IS L/S    OOS L/S")
        for ncut in cutoffs:
            a = pre[(pre["top_n"] == ncut) & (pre["cond"] == cond)]
            b = post[(post["top_n"] == ncut) & (post["cond"] == cond)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            out.append(
                f" {ncut:>4} | {a['sel_mean']:+7.3f}% {_ci(a['sel_ci_lo'], a['sel_ci_hi']):>20} "
                f"| {b['sel_mean']:+7.3f}% {_ci(b['sel_ci_lo'], b['sel_ci_hi']):>20} "
                f"| {a['ls_mean']:+6.3f}%  {b['ls_mean']:+6.3f}%")
    out.append("\n  A cell 'survives' only if BOTH halves are positive and the OOS CI clears zero.")
    return "\n".join(out), pd.concat([pre, post], ignore_index=True)


def alpha_oos_report(long, weights, split=SPLIT_DATE, cutoffs=CUTOFFS):
    """CAPM-alpha version of the OOS split. Reports the high-DPI basket's
    beta-adjusted forward alpha (long-only) and the darkest-minus-least-dark
    alpha (long-short), full sample + IS/OOS -- the clean 'edge over beta' test."""
    full = megacap_edge(long, weights, cutoffs)
    pre = megacap_edge(_slice(long, hi=split), weights, cutoffs)
    post = megacap_edge(_slice(long, lo=split), weights, cutoffs)
    out = ["CAPM-ALPHA edge over beta (alpha = fwd - beta*SPY_fwd; beta on trailing 252d):",
           f"  full = 2019-2026, IS = 2019-{int(split[:4])-1}, OOS = {split[:4]}-2026"]
    for cond in ("all-days", "in-3mo-downtrend"):
        out.append(f"\n=== {cond}  --  ALPHA long-only (high-DPI basket) [95% CI] ===")
        out.append(" topN |        full                 |         IS                  |        OOS")
        for ncut in cutoffs:
            def cell(tab):
                r = tab[(tab["top_n"] == ncut) & (tab["cond"] == cond)]
                if r.empty or "alpha_lo" not in r or not np.isfinite(r.iloc[0].get("alpha_lo", np.nan)):
                    return f"{'--':>27}"
                r = r.iloc[0]
                return f"{r['alpha_lo']:+6.3f}% {_ci(r['alpha_lo_ci_lo'], r['alpha_lo_ci_hi']):>18}"
            out.append(f" {ncut:>4} | {cell(full)} | {cell(pre)} | {cell(post)}")
        out.append(f"  -- {cond}: ALPHA long-short (darkest-least-dark) --")
        for ncut in cutoffs:
            def cell_ls(tab):
                r = tab[(tab["top_n"] == ncut) & (tab["cond"] == cond)]
                if r.empty or not np.isfinite(r.iloc[0].get("alpha_ls", np.nan)):
                    return f"{'--':>27}"
                r = r.iloc[0]
                return f"{r['alpha_ls']:+6.3f}% {_ci(r['alpha_ls_ci_lo'], r['alpha_ls_ci_hi']):>18}"
            out.append(f" {ncut:>4} | {cell_ls(full)} | {cell_ls(pre)} | {cell_ls(post)}")
    out.append("\n  Edge over beta is real only where ALPHA (long-only or long-short) is")
    out.append("  positive AND its CI clears zero -- in BOTH the full sample and the OOS half.")
    full["period"] = "full"
    pre["period"] = f"IS(<{split[:4]})"
    post["period"] = f"OOS(>={split[:4]})"
    return "\n".join(out), pd.concat([full, pre, post], ignore_index=True)


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
    ap.add_argument("--split-date", default=SPLIT_DATE,
                    help="in-sample (<) vs out-of-sample (>=) boundary (default 2023-01-01)")
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
    long = add_capm_alpha(long, adj, spy, args.horizon)
    nnames = long.index.get_level_values("name").nunique()
    ndays = long.index.get_level_values("date").nunique()
    print(f"\nPanel: {len(long):,} (name,day) obs · {nnames} names · {ndays} days "
          f"[{long.index.get_level_values('date').min().date()} -> "
          f"{long.index.get_level_values('date').max().date()}]")
    print(f"Outcome horizon: {args.horizon}d. 'high DIX' = self-relative DPI top quintile.")
    tab = megacap_edge(long, weights)
    tab["period"] = "full"
    print(report(tab))
    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE SPLIT")
    print("=" * 78)
    oos_txt, oos_tab = oos_report(long, weights, split=args.split_date)
    print(oos_txt)
    print("\n" + "=" * 78)
    print("CAPM-ALPHA OUT-OF-SAMPLE SPLIT (the confirmation test)")
    print("=" * 78)
    alpha_txt, alpha_tab = alpha_oos_report(long, weights, split=args.split_date)
    print(alpha_txt)
    if args.out:
        alpha_tab.to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
