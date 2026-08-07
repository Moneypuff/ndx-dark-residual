#!/usr/bin/env python3
"""
Cross-sectional "buy the dip on high DIX" study over the S&P 500 universe.
=========================================================================

Question (per request): expand option B (DIX-vs-price divergence) from the index
level to the WHOLE S&P 500 single-stock universe. Does *buying a dip when the
stock's dark-flow is high* have a cross-sectional edge -- no matter the stock?

Per-name signal (the single-stock analog of DIX), all real-time / no look-ahead
------------------------------------------------------------------------------
D_i,t = ShortVolume / TotalVolume from FINRA's daily consolidated off-exchange
(CNMSshvol) file for name i, 5-day moving average (0..1) -- the same per-name
dark ratio the dashboard uses. "High DIX for a single stock" is framed two ways:

  * SELF-relative  : D_i,t's percentile within name i's OWN trailing 252 sessions
                     (min 120) -- "this stock is unusually dark vs its own norm".
                     This is the primary lens ("no matter the stock").
  * CROSS-SECTIONAL: D_i,t ranked across all names that day -- "the darkest names
                     today". Reported as a cross-check.

"Dip" = a recent price pullback: the name's trailing DIP_LOOKBACK-session return
is negative (default 21 sessions -- a one-month dip). Graded by depth too.

Outcome (no look-ahead)
-----------------------
Forward FWD-session return (default 21), CROSS-SECTIONALLY DEMEANED: each day we
subtract the universe's mean forward return, so the outcome measures relative
stock selection, not market beta (a rising-tide bull sample would otherwise make
every dip look good). A market-excess-vs-SPY twin is also reported.

Inference (honest about overlap + cross-name correlation)
---------------------------------------------------------
Pooled (name, day) observations are massively overlapping and cross-correlated,
so pooled t-stats are meaningless. The headline is instead a DAILY PORTFOLIO:
each day form the equal-weight basket the rule would hold, giving one return per
day; that daily series then gets a moving-block bootstrap CI. Plus:
  * a 2x2 interaction table (high/low D x dip/no-dip),
  * forward return by D-decile WITHIN dipped names (is it monotone?),
  * a per-NAME breakdown -- the fraction of names for which high-D dips beat
    their other dips (the literal "no matter the stock" test), with a sign test,
  * a pre-2021 vs 2021+ regime split.

Data
----
FINRA CNMSshvol daily files (one per day, all tickers) -> per-name Short/Total;
Yahoo adjusted closes for the current S&P 500 (IVV) constituents. Both are
fetched live (FINRA cached on disk); survivorship is a disclosed caveat -- the
universe is today's members, so names that left the index are absent.

Reproduce
---------
    python spx_xs_dip_dix_study.py --start 2019-01-01
    python spx_xs_dip_dix_study.py --start 2019-01-01 --out spx_xs_dip_dix.csv
"""
import argparse
import math

import numpy as np
import pandas as pd

# Reuse the main pipeline's data acquisition + stats (DRY, same as the other
# study scripts). Names are resolved lazily in main() so the pure-analysis
# functions below stay importable/testable without a network.
import ndx_dark_residual as N

DIP_LOOKBACK = 21       # trailing sessions defining a "dip" (1 month)
SELF_WIN = 252          # trailing window for the self-relative D percentile
SELF_MIN = 120          # min obs before the self-relative percentile is defined
HI, LO = 0.80, 0.20     # high/low cutoffs on the D percentile (top/bottom quintile)
REGIME_SPLIT = "2021-01-01"
BOOT_B, BOOT_L = 2000, 21


# ---------------------------------------------------------------------------
# Stats helpers (mirror the repo's conventions)
# ---------------------------------------------------------------------------
def block_boot_ci(r, B=BOOT_B, L=BOOT_L, seed=0, levels=(2.5, 97.5)):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2 * L:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    means = r[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, levels))


def sign_test_p(k, n):
    """Two-sided binomial sign test p-value for k of n positive under p=0.5,
    via a normal approximation (n is large here)."""
    if n == 0:
        return np.nan
    z = (k - n / 2) / (math.sqrt(n) / 2)
    return 2 * (0.5 * math.erfc(abs(z) / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Signal / outcome construction  (pure: wide DataFrames in, wide frames out)
# ---------------------------------------------------------------------------
def self_percentile(D, win=SELF_WIN, min_obs=SELF_MIN):
    """Per-name trailing percentile of D within its own last `win` sessions
    (inclusive, mid-rank on ties), 0..1. NaN until `min_obs` obs. No look-ahead:
    each day's D is ranked only against its own trailing window."""
    return D.rolling(win, min_periods=min_obs).rank(pct=True)


def build_frames(D, adj, spy, fwd_h=21, dip_lb=DIP_LOOKBACK):
    """Assemble the aligned per-(name,day) fields as a tidy long DataFrame:
      dpct   -- SELF-relative D percentile within the name's own trailing year (0..1)
      xrank  -- CROSS-SECTIONAL D percentile across all names that day (0..1)
      dip    -- trailing `dip_lb`-session return, % (negative = a dip)
      fwd    -- forward `fwd_h`-session return, %
      xret   -- fwd cross-sectionally demeaned (minus the universe's daily mean)
      spyx   -- fwd minus SPY's forward return over the same window (market excess)
    Only rows with all of {dpct, dip, fwd} present are kept."""
    cols = D.columns.intersection(adj.columns)
    idx = D.index.intersection(adj.index)                   # shared trading calendar
    D, adj = D.loc[idx, cols], adj.loc[idx, cols]
    fwd = N.compute_forward_return(adj, fwd_h)
    back = (adj / adj.shift(dip_lb) - 1.0) * 100.0
    dpct = self_percentile(D)
    xrank = D.rank(axis=1, pct=True)                         # cross-sectional, per day
    xret = fwd.sub(fwd.mean(axis=1), axis=0)                 # cross-sectional demean
    spy = spy.reindex(idx)
    spy_fwd = (spy.shift(-fwd_h) / spy - 1.0) * 100.0
    spyx = fwd.sub(spy_fwd, axis=0)

    # pandas 3.0 stack() keeps the full grid; DataFrame construction aligns the
    # six columns on the shared (date, name) MultiIndex.
    long = pd.DataFrame({
        "dpct": dpct.stack(), "xrank": xrank.stack(), "dip": back.stack(),
        "fwd": fwd.stack(), "xret": xret.stack(), "spyx": spyx.stack(),
    })
    long.index.set_names(["date", "name"], inplace=True)
    long = long.dropna(subset=["dpct", "dip", "fwd"])
    return long


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def interaction_2x2(long, outcome="xret", sig="dpct"):
    """Mean outcome across (high/low D) x (dip / no-dip), plus the interaction:
    does high-vs-low D pay MORE among dipped names than among non-dipped?"""
    hi = long[sig] >= HI
    lo = long[sig] <= LO
    dip = long["dip"] < 0
    rows = []
    for dlab, dmask in [("dip", dip), ("no-dip", ~dip)]:
        for glab, gmask in [("highD", hi), ("lowD", lo)]:
            r = long.loc[dmask & gmask, outcome]
            rows.append({"cell": f"{glab} & {dlab}", "n": int(len(r)),
                         "mean": round(float(r.mean()), 3) if len(r) else np.nan,
                         "hit": round(float((r > 0).mean() * 100)) if len(r) else np.nan})
    tab = pd.DataFrame(rows)
    def m(g, d):
        return long.loc[(hi if g == "hi" else lo) & (dip if d else ~dip), outcome].mean()
    inter = (m("hi", True) - m("lo", True)) - (m("hi", False) - m("lo", False))
    return tab, float(inter)


def decile_within_dips(long, outcome="xret", sig="dpct", n=10):
    """Outcome by decile of the D signal, restricted to dipped names."""
    d = long[long["dip"] < 0].copy()
    d["dec"] = pd.qcut(d[sig], n, labels=False, duplicates="drop") + 1
    g = d.groupby("dec")[outcome].agg(["mean", "median", "count"])
    top, bot = int(g.index.max()), int(g.index.min())
    ls = float(g.loc[top, "mean"] - g.loc[bot, "mean"])
    return g, ls


def daily_portfolio(long, outcome="xret", sig="dpct", seed=0):
    """The honest, tradeable view. Each day, among names that dipped, take the
    top-D-quintile long and bottom-D-quintile short (equal weight) -> one L/S
    return per day; also a long-only 'darkest dips vs universe' leg. Returns a
    dict of the daily series' mean + block-bootstrap CI + hit rate."""
    dips = long[long["dip"] < 0]
    ls_daily, lo_daily = [], []
    for date, g in dips.groupby(level="date"):
        if len(g) < 10:
            continue
        hiq = g[g[sig] >= HI][outcome]
        loq = g[g[sig] <= LO][outcome]
        if len(hiq) and len(loq):
            ls_daily.append(float(hiq.mean() - loq.mean()))
        if len(hiq):
            lo_daily.append(float(hiq.mean()))
    out = {}
    for lab, ser in [("long_short", ls_daily), ("long_only", lo_daily)]:
        a = np.asarray(ser, dtype=float)
        ci = block_boot_ci(a, seed=seed + (0 if lab == "long_short" else 1))
        out[lab] = {"days": int(len(a)), "mean": round(float(np.nanmean(a)), 3) if len(a) else np.nan,
                    "hit": round(float((a > 0).mean() * 100)) if len(a) else np.nan,
                    "ci": (round(ci[0], 3), round(ci[1], 3)) if np.isfinite(ci[0]) else None}
    return out


def per_name_edge(long, outcome="xret", sig="dpct"):
    """The literal 'no matter the stock' test. For each name, compare its mean
    outcome on HIGH-D dips vs its mean outcome on its OTHER dips (low/mid-D
    dips). Report the distribution of that per-name edge and a sign test on how
    many names are positive (each name weighted once, so a few names can't carry
    the result)."""
    dips = long[long["dip"] < 0]
    edges = []
    for name, g in dips.groupby(level="name"):
        hiq = g[g[sig] >= HI][outcome]
        rest = g[g[sig] < HI][outcome]
        if len(hiq) >= 5 and len(rest) >= 5:
            edges.append(float(hiq.mean() - rest.mean()))
    e = np.asarray(edges, dtype=float)
    if not len(e):
        return {"names": 0}
    k = int((e > 0).sum())
    return {"names": int(len(e)), "pos_frac": round(k / len(e) * 100, 1),
            "median_edge": round(float(np.median(e)), 3),
            "mean_edge": round(float(np.mean(e)), 3),
            "iqr": (round(float(np.percentile(e, 25)), 3), round(float(np.percentile(e, 75)), 3)),
            "sign_p": round(sign_test_p(k, len(e)), 4)}


def regime_split(long, outcome="xret", sig="dpct"):
    """Daily-portfolio long-short mean, pre-2021 vs 2021+."""
    out = {}
    for lab, sub in [("pre-2021", long[long.index.get_level_values("date") < REGIME_SPLIT]),
                     ("2021+", long[long.index.get_level_values("date") >= REGIME_SPLIT])]:
        dp = daily_portfolio(sub, outcome, sig=sig)
        out[lab] = dp["long_short"]
    return out


# ---------------------------------------------------------------------------
# Data acquisition (reuses the main pipeline; FINRA cached on disk)
# ---------------------------------------------------------------------------
def load_universe(start, end, cache_dir, workers, max_names=None, refresh=False):
    """Build the full daily S&P 500 per-name panel (D + adjclose) plus SPY, via
    ndx_dark_residual's FINRA/Yahoo machinery. Returns (D, adj, spy, syms)."""
    syms = N.fetch_ishares_holdings(N.IVV_PORTFOLIO_ID, label="IVV S&P 500")
    syms = [s for s in dict.fromkeys(syms) if s and s.isalpha()]   # de-dupe, drop odd tickers
    if max_names:
        syms = syms[:max_names]
    print(f"S&P 500 constituents (IVV): {len(syms)} names", flush=True)
    SP = N.build_universe_panels(syms, start, end, workers=workers, cache_dir=cache_dir,
                                 ns="sp500", refresh=refresh, label="S&P 500")
    D, adj = SP["d"], SP["adjclose"]
    spy_panels = N.load_yahoo_panels(["SPY"], start, end, workers=1,
                                     cache_dir=cache_dir, refresh=refresh, label="SPY")
    spy = spy_panels["adjclose"]["SPY"]
    return D, adj, spy, syms


# ---------------------------------------------------------------------------
def _fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "--"


def report(long, outcome="xret", sig="dpct", sig_label="self-relative"):
    out = [f"### signal = {sig_label} D percentile   |   outcome = "
           + ("cross-sectionally demeaned" if outcome == "xret" else "excess vs SPY")
           + f" forward return  ({outcome})"]
    tab, inter = interaction_2x2(long, outcome, sig)
    out.append("\n2x2  (high D = top quintile of the signal, low D = bottom quintile):")
    for _, r in tab.iterrows():
        out.append(f"   {r['cell']:16s} n={int(r['n']):>7}  mean {r['mean']:+.3f}%  hit {r['hit']:.0f}%")
    out.append(f"   INTERACTION (highD-lowD | dip) - (highD-lowD | no-dip) = {inter:+.3f} pp"
               "   <- is dark flow MORE informative on a dip?")
    g, ls = decile_within_dips(long, outcome, sig)
    out.append("\nForward return by D decile, WITHIN dipped names only:")
    for dec, row in g.iterrows():
        out.append(f"   D{int(dec):<2d} mean {row['mean']:+.3f}%  med {row['median']:+.3f}%  n={int(row['count'])}")
    out.append(f"   long-short (darkest - least-dark dips): {ls:+.3f} pp")
    dp = daily_portfolio(long, outcome, sig)
    out.append("\nDaily portfolio (each day, among dips; equal-weight; overlapping 21d holds):")
    out.append(f"   LONG-SHORT (top-Q darkest dips - bottom-Q): "
               f"mean {dp['long_short']['mean']:+.3f}%  hit {dp['long_short']['hit']:.0f}%  "
               f"95% CI {_fmt_ci(dp['long_short']['ci'])}  ({dp['long_short']['days']} days)")
    out.append(f"   LONG-ONLY  (darkest dips vs universe):      "
               f"mean {dp['long_only']['mean']:+.3f}%  hit {dp['long_only']['hit']:.0f}%  "
               f"95% CI {_fmt_ci(dp['long_only']['ci'])}")
    pn = per_name_edge(long, outcome, sig)
    if pn.get("names"):
        out.append("\n'No matter the stock' -- per-NAME edge (high-D dips vs the name's other dips):")
        out.append(f"   {pn['names']} names with enough obs | {pn['pos_frac']}% positive "
                   f"(sign-test p={pn['sign_p']}) | median edge {pn['median_edge']:+.3f}pp "
                   f"IQR [{pn['iqr'][0]:+.3f}, {pn['iqr'][1]:+.3f}]")
    rs = regime_split(long, outcome, sig)
    out.append("\nRegime split (daily-portfolio long-short mean):")
    for lab, s in rs.items():
        out.append(f"   {lab:9s} mean {s['mean']:+.3f}%  hit {s['hit']:.0f}%  "
                   f"95% CI {_fmt_ci(s['ci'])}  ({s['days']} days)")
    return "\n".join(out)


def summary_rows(long):
    """Compact machine-readable rows for the results CSV: the daily-portfolio
    long-short and long-only headline for each (signal, outcome, period)."""
    rows = []
    for sig, slab in [("dpct", "self"), ("xrank", "xsect")]:
        for outcome in ("xret", "spyx"):
            for lab, sub in [("all", long),
                             ("pre2021", long[long.index.get_level_values("date") < REGIME_SPLIT]),
                             ("post2021", long[long.index.get_level_values("date") >= REGIME_SPLIT])]:
                dp = daily_portfolio(sub, outcome, sig)
                for leg in ("long_short", "long_only"):
                    d = dp[leg]
                    rows.append({"signal": slab, "outcome": outcome, "period": lab, "leg": leg,
                                 "days": d["days"], "mean_pp": d["mean"], "hit_pct": d["hit"],
                                 "ci_lo": d["ci"][0] if d["ci"] else None,
                                 "ci_hi": d["ci"][1] if d["ci"] else None})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-01-01", help="history start (default 2019-01-01)")
    ap.add_argument("--end", default=None, help="history end (default today)")
    ap.add_argument("--horizon", type=int, default=21, help="forward horizon in sessions (default 21)")
    ap.add_argument("--dip-lookback", type=int, default=DIP_LOOKBACK,
                    help="trailing sessions defining a dip (default 21)")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR, help="FINRA/Yahoo cache dir")
    ap.add_argument("--workers", type=int, default=10, help="fetch workers (default 10)")
    ap.add_argument("--max-names", type=int, default=None, help="cap universe size (smoke tests)")
    ap.add_argument("--refresh", action="store_true", help="re-poll caches to today")
    ap.add_argument("--out", default=None, help="write the summary table to this CSV")
    ap.add_argument("--frame-out", default=None, help="optional: dump the per-(name,day) long frame")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    D, adj, spy, syms = load_universe(start, end, args.cache_dir, args.workers,
                                      max_names=args.max_names, refresh=args.refresh)
    long = build_frames(D, adj, spy, fwd_h=args.horizon, dip_lb=args.dip_lookback)
    ndays = long.index.get_level_values("date").nunique()
    nnames = long.index.get_level_values("name").nunique()
    ndips = int((long["dip"] < 0).sum())
    print(f"\nAligned panel: {len(long):,} (name,day) obs · {nnames} names · {ndays} days "
          f"[{long.index.get_level_values('date').min().date()} -> "
          f"{long.index.get_level_values('date').max().date()}]")
    print(f"Dips (trailing {args.dip_lookback}d return < 0): {ndips:,} obs "
          f"({ndips / len(long) * 100:.0f}% of the panel)\n")
    print("=" * 78)
    print("PRIMARY -- self-relative high-D dips, cross-sectionally demeaned outcome")
    print("=" * 78)
    print(report(long, "xret", "dpct", "self-relative"))
    print("\n" + "=" * 78)
    print("CROSS-CHECK -- cross-sectional darkest-names ranking (xrank)")
    print("=" * 78)
    print(report(long, "xret", "xrank", "cross-sectional"))

    if args.out:
        summary_rows(long).to_csv(args.out, index=False)
        print(f"\nWrote summary table -> {args.out}")
    if args.frame_out:
        long.reset_index().to_csv(args.frame_out, index=False)
        print(f"Wrote per-(name,day) frame -> {args.frame_out}")


if __name__ == "__main__":
    main()
