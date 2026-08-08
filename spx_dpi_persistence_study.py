#!/usr/bin/env python3
"""
DPI PERSISTENCE on a downtrend: does a sustained high-decile DPI (10-day average
in the higher deciles / a high-DPI streak) beat a single print?
==============================================================================

Pushback tested (per request): the single-stock tests used a single-day DPI
reading and found no edge -- but a stock on a high-decile DPI *streak* may do
better over the 1-2 month range. So on a downtrend, require the **10-day average
DPI to sit in the higher deciles** (persistent dark accumulation), not one print.

Signals (real-time, per name; self-relative = ranked within the name's own year)
-------------------------------------------------------------------------------
* p1     -- self-relative percentile of the 5-day-MA dark ratio D (the SINGLE
            print used before, for comparison).
* p10    -- self-relative percentile of the 10-DAY AVERAGE of the raw daily DPI
            (the requested persistence signal). "Higher deciles" = top quintile.
* streak -- run length (consecutive sessions) of p1 in the top quintile (the
            "high-decile DPI streak").
"High DPI" = percentile >= 0.80 (top quintile). "Downtrend" = 3-month OLS
log-price slope < 0.

Outcomes (no look-ahead)
------------------------
Forward 21/42/63-session return, three ways: RAW, EXCESS vs SPY, and CAPM ALPHA
(fwd - beta*SPY_fwd, beta on trailing 252 daily returns). RAW answers "does the
stock tend to do better"; ALPHA answers "is any of it an edge over beta".

Tests
-----
1. STREAK-LENGTH GRADIENT -- forward return by DPI-streak bucket (0 / 1-4 / 5-9 /
   10+ days) among downtrend names. Does a longer streak pay more? (raw & alpha)
2. PER-NAME CONDITIONAL (the pushback's framing) -- for each name, mean forward
   return on [signal-high & downtrend] minus its mean over ALL its downtrend
   days; fraction of names positive + sign test. Run for the single print (p1)
   vs the 10-day average (p10), so we see if persistence adds anything.
3. BETA-NEUTRAL LONG-SHORT -- darkest-minus-least-dark (by p10) among downtrend
   names, on CAPM alpha, full sample + 2019-22 / 2023+ split (the tradeable,
   overfitting-disciplined test).

Reproduce
---------
    python spx_dpi_persistence_study.py --start 2019-01-01 --out spx_dpi_persistence.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import spx_xs_dip_dix_study as XS   # self_percentile, block_boot_ci, sign_test_p, trend_slope, HI/LO

HI = XS.HI               # 0.80 top-quintile cutoff
STREAK_WIN = 10          # "10-day average DPI"
SELF_WIN, SELF_MIN = 252, 120
HORIZONS = (21, 42, 63)
SPLIT_DATE = "2023-01-01"
MIN_BASKET = 3


def run_length(bool_df):
    """Per-column consecutive-True run length ending at each row (0 where False)."""
    def _rl(s):
        b = s.fillna(False).to_numpy()
        out = np.zeros(len(b), dtype=int)
        c = 0
        for i, v in enumerate(b):
            c = c + 1 if v else 0
            out[i] = c
        return pd.Series(out, index=s.index)
    return bool_df.apply(_rl)


def build_long(SP, spy, fwd_h):
    dpi, d5, adj = SP["dpi"], SP["d"], SP["adjclose"]
    cols = dpi.columns.intersection(adj.columns)
    idx = dpi.index.intersection(adj.index)
    dpi, d5, adj = dpi.loc[idx, cols], d5.loc[idx, cols], adj.loc[idx, cols]

    p1 = XS.self_percentile(d5, SELF_WIN, SELF_MIN)                         # single print
    dpi10 = dpi.rolling(STREAK_WIN, min_periods=STREAK_WIN // 2 + 1).mean()
    p10 = XS.self_percentile(dpi10, SELF_WIN, SELF_MIN)                     # 10-day average
    streak = run_length(p1 >= HI)                                          # high-decile streak
    trend63 = XS.trend_slope(adj, 63)

    rback = (adj / adj.shift(21) - 1.0) * 100.0                            # prior-21d return = the STR characteristic
    fwd = N.compute_forward_return(adj, fwd_h)
    xret = fwd.sub(fwd.mean(axis=1), axis=0)
    spy_i = spy.reindex(idx)
    spy_fwd = (spy_i.shift(-fwd_h) / spy_i - 1.0) * 100.0
    spyx = fwd.sub(spy_fwd, axis=0)
    # CAPM beta (trailing 252 daily returns) -> forward alpha
    r, rs = adj.pct_change(), spy_i.pct_change()
    Er, Ers = r.rolling(252, min_periods=120).mean(), rs.rolling(252, min_periods=120).mean()
    Errs = r.mul(rs, axis=0).rolling(252, min_periods=120).mean()
    Erss = (rs * rs).rolling(252, min_periods=120).mean()
    beta = Errs.sub(Er.mul(Ers, axis=0)).div(Erss - Ers * Ers, axis=0)
    alpha = fwd.sub(beta.mul(spy_fwd, axis=0))

    long = pd.DataFrame({
        "p1": p1.stack(), "p10": p10.stack(), "streak": streak.stack(),
        "trend63": trend63.stack(), "rback": rback.stack(), "fwd": fwd.stack(),
        "xret": xret.stack(), "spyx": spyx.stack(), "alpha": alpha.stack(),
    })
    long.index.set_names(["date", "name"], inplace=True)
    long["streak10"] = (long["streak"] >= 10).astype(float)
    return long.dropna(subset=["p1", "p10", "fwd", "trend63"])


# ---------------------------------------------------------------------------
def _daily(long, mask, col, min_n=MIN_BASKET):
    d = long[mask]
    out = []
    for _, g in d.groupby(level="date"):
        r = g[col].dropna()
        if len(r) >= min_n:
            out.append(float(r.mean()))
    return np.asarray(out, dtype=float)


def _stat(a, seed=0):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"days": 0, "mean": np.nan, "hit": np.nan, "ci": None}
    ci = XS.block_boot_ci(a, seed=seed)
    return {"days": int(len(a)), "mean": round(float(a.mean()), 3),
            "hit": round(float((a > 0).mean() * 100)),
            "ci": (round(ci[0], 3), round(ci[1], 3)) if np.isfinite(ci[0]) else None}


def streak_gradient(long, outcome, seed=0):
    down = long["trend63"] < 0
    buckets = [("0", long["streak"] == 0), ("1-4", long["streak"].between(1, 4)),
               ("5-9", long["streak"].between(5, 9)), ("10+", long["streak"] >= 10)]
    rows = []
    for lab, m in buckets:
        s = _stat(_daily(long, down & m, outcome), seed=seed + len(lab))
        rows.append({"streak": lab, "days": s["days"], "mean": s["mean"],
                     "hit": s["hit"], "ci": s["ci"]})
    return rows


def per_name_conditional(long, sig_col, outcome, thr=HI, min_ev=5):
    """For each name: mean outcome on [sig>=thr & downtrend] minus its mean over
    ALL its downtrend days. Fraction of names positive + sign test."""
    down = long[long["trend63"] < 0]
    edges = []
    for name, g in down.groupby(level="name"):
        hi = g[g[sig_col] >= thr][outcome].dropna()
        base = g[outcome].dropna()
        if len(hi) >= min_ev and len(base) >= 2 * min_ev:
            edges.append(float(hi.mean() - base.mean()))
    e = np.asarray(edges, dtype=float)
    if not len(e):
        return {"names": 0}
    k = int((e > 0).sum())
    return {"names": len(e), "pos_pct": round(k / len(e) * 100, 1),
            "median_edge": round(float(np.median(e)), 3),
            "mean_edge": round(float(np.mean(e)), 3),
            "sign_p": round(XS.sign_test_p(k, len(e)), 4)}


def longshort(long, sig_col, outcome, seed=0):
    down = long[long["trend63"] < 0]
    daily = []
    for _, g in down.groupby(level="date"):
        hi = g[g[sig_col] >= HI][outcome].dropna()
        lo = g[g[sig_col] <= XS.LO][outcome].dropna()
        if len(hi) >= MIN_BASKET and len(lo) >= MIN_BASKET:
            daily.append(float(hi.mean() - lo.mean()))
    return _stat(np.asarray(daily, dtype=float), seed=seed)


def fama_macbeth(long, ycol, xcols, mask, seed=0, min_n=8):
    """Daily cross-sectional OLS of `ycol` on [const]+`xcols` over `mask` rows,
    then the mean of each daily coefficient with a block-bootstrap CI (the
    Fama-MacBeth estimator). Used to ask: does the streak dummy still carry
    forward alpha once the short-term-reversal characteristic (rback) is in the
    regression alongside it?"""
    d = long[mask].dropna(subset=[ycol] + xcols)
    series = {c: [] for c in ["const"] + xcols}
    for _, g in d.groupby(level="date"):
        if len(g) < len(xcols) + min_n or g[xcols].std().min() == 0:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in xcols])
        y = g[ycol].to_numpy()
        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        series["const"].append(b[0])
        for i, c in enumerate(xcols):
            series[c].append(b[i + 1])
    out = {}
    for i, c in enumerate(["const"] + xcols):
        a = np.asarray(series[c], dtype=float)
        ci = XS.block_boot_ci(a, seed=seed + i) if len(a) else (np.nan, np.nan)
        out[c] = {"mean": round(float(a.mean()), 4) if len(a) else np.nan,
                  "ci": (round(ci[0], 4), round(ci[1], 4)) if np.isfinite(ci[0]) else None,
                  "days": int(len(a))}
    return out


def reversal_double_sort(long, outcome="alpha"):
    """Hold reversal ~constant: among downtrend names, within each recent-return
    (rback) tercile, the 10+ streak minus no-streak forward outcome. If the
    streak still pays inside every reversal bucket, it is not just reversal."""
    d = long[(long["trend63"] < 0)].dropna(subset=["rback", outcome]).copy()
    # daily rback terciles so the sort is real-time
    d["rev_t"] = d.groupby(level="date")["rback"].transform(
        lambda s: pd.qcut(s, 3, labels=["deepest", "mid", "shallow"], duplicates="drop")
        if s.nunique() >= 3 else np.nan)
    rows = []
    for t in ("deepest", "mid", "shallow"):
        sub = d[d["rev_t"] == t]
        hi = sub[sub["streak"] >= 10][outcome]
        no = sub[sub["streak"] == 0][outcome]
        rows.append({"rev_tercile": t, "streak10_mean": round(float(hi.mean()), 3) if len(hi) else np.nan,
                     "nostreak_mean": round(float(no.mean()), 3) if len(no) else np.nan,
                     "diff": round(float(hi.mean() - no.mean()), 3) if len(hi) and len(no) else np.nan,
                     "n_streak10": int(len(hi))})
    return rows


def _slice(long, lo=None, hi=None):
    d = long.index.get_level_values("date")
    m = np.ones(len(long), dtype=bool)
    if lo is not None:
        m &= np.asarray(d >= pd.Timestamp(lo))
    if hi is not None:
        m &= np.asarray(d < pd.Timestamp(hi))
    return long[m]


def _ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "--"


def report(long_by_h, split=SPLIT_DATE):
    out = []
    for h, long in long_by_h.items():
        out.append("\n" + "=" * 84)
        out.append(f"HORIZON {h}d")
        out.append("=" * 84)
        out.append("1) STREAK-LENGTH GRADIENT among downtrend names (does a longer high-DPI streak pay?)")
        for lab, oc in [("RAW", "fwd"), ("ALPHA", "alpha")]:
            cells = streak_gradient(long, oc)
            out.append(f"   {lab:5s}: " + " | ".join(
                f"streak {c['streak']:>3} {c['mean']:+.2f}% (n={c['days']}, {_ci(c['ci'])})" for c in cells))
        out.append("   1b) 10+ streak ALPHA long-only, OOS split (the confirmation of the streak effect):")
        for lab, sub in [("full", long), (f"IS<{split[:4]}", _slice(long, hi=split)),
                         (f"OOS>={split[:4]}", _slice(long, lo=split))]:
            cells = {c["streak"]: c for c in streak_gradient(sub, "alpha", seed=7)}
            c = cells.get("10+", {})
            out.append(f"        {lab:10s} 10+ streak alpha {c.get('mean', float('nan')):+.3f}%  "
                       f"hit {c.get('hit', float('nan'))}%  95% CI {_ci(c.get('ci'))}  ({c.get('days', 0)} days)")
        out.append("\n2) PER-NAME CONDITIONAL: mean outcome on [high & downtrend] minus the name's "
                   "downtrend baseline")
        out.append(f"   {'signal':16s} {'outcome':7s}  names  pos%   median-edge  sign-p")
        for sig, slab in [("p1", "single-print"), ("p10", "10d-avg (requested)")]:
            for oc in ("fwd", "spyx", "alpha"):
                r = per_name_conditional(long, sig, oc)
                if r.get("names"):
                    out.append(f"   {slab:16s} {oc:7s}  {r['names']:>4}  {r['pos_pct']:>5}%  "
                               f"{r['median_edge']:+.3f}pp     {r['sign_p']}")
        out.append("\n3) BETA-NEUTRAL LONG-SHORT (p10 darkest-least-dark, downtrend), CAPM alpha:")
        for lab, sub in [("full", long), (f"IS<{split[:4]}", _slice(long, hi=split)),
                         (f"OOS>={split[:4]}", _slice(long, lo=split))]:
            ls = longshort(sub, "p10", "alpha", seed=hash((h, lab)) % 1000)
            out.append(f"   {lab:10s} alpha L/S {ls['mean']:+.3f}%  hit {ls['hit']}%  "
                       f"95% CI {_ci(ls['ci'])}  ({ls['days']} days)")

        out.append("\n4) SHORT-TERM-REVERSAL CONTROL (among downtrend names)")
        down = long["trend63"] < 0
        rb_s = long.loc[down & (long["streak"] >= 10), "rback"].mean()
        rb_n = long.loc[down & (long["streak"] == 0), "rback"].mean()
        out.append(f"   confound check -- mean prior-21d return: 10+ streak {rb_s:+.2f}%  vs  "
                   f"no-streak {rb_n:+.2f}%  (are streak names deeper losers?)")
        out.append("   Fama-MacBeth: alpha ~ const + rback(reversal) + streak10 dummy  "
                   "(coef on streak10 = alpha the streak adds BEYOND reversal):")
        for lab, sub in [("full", long), (f"IS<{split[:4]}", _slice(long, hi=split)),
                         (f"OOS>={split[:4]}", _slice(long, lo=split))]:
            fm = fama_macbeth(sub, "alpha", ["rback", "streak10"], sub["trend63"] < 0,
                              seed=hash((h, lab, "fm")) % 1000)
            s10, rbk = fm["streak10"], fm["rback"]
            out.append(f"   {lab:10s} streak10 {s10['mean']:+.3f}% {_ci(s10['ci'])}  |  "
                       f"rback(reversal) {rbk['mean']:+.4f} {_ci(rbk['ci'])}  ({s10['days']} days)")
        out.append("   double-sort (10+ streak - no-streak alpha, within reversal terciles):")
        for r in reversal_double_sort(long, "alpha"):
            out.append(f"      rback {r['rev_tercile']:>8}: streak10 {r['streak10_mean']:+.3f}%  "
                       f"no-streak {r['nostreak_mean']:+.3f}%  diff {r['diff']:+.3f}pp "
                       f"(n={r['n_streak10']})")
    return "\n".join(out)


def summary_rows(long_by_h, split=SPLIT_DATE):
    rows = []
    for h, long in long_by_h.items():
        for lab, oc in [("raw", "fwd"), ("alpha", "alpha")]:
            for c in streak_gradient(long, oc):
                rows.append({"horizon": h, "test": "streak_gradient", "outcome": lab,
                             "bucket": c["streak"], "n": c["days"], "mean_pp": c["mean"],
                             "hit": c["hit"], "ci_lo": c["ci"][0] if c["ci"] else None,
                             "ci_hi": c["ci"][1] if c["ci"] else None})
        for sig in ("p1", "p10"):
            for oc in ("fwd", "spyx", "alpha"):
                r = per_name_conditional(long, sig, oc)
                if r.get("names"):
                    rows.append({"horizon": h, "test": f"pername_{sig}", "outcome": oc,
                                 "bucket": "high&downtrend", "n": r["names"],
                                 "mean_pp": r["mean_edge"], "hit": r["pos_pct"],
                                 "ci_lo": None, "ci_hi": None, "sign_p": r["sign_p"]})
        for lab, sub in [("full", long), ("IS", _slice(long, hi=split)), ("OOS", _slice(long, lo=split))]:
            ls = longshort(sub, "p10", "alpha")
            rows.append({"horizon": h, "test": "ls_alpha_p10", "outcome": lab,
                         "bucket": "downtrend", "n": ls["days"], "mean_pp": ls["mean"],
                         "hit": ls["hit"], "ci_lo": ls["ci"][0] if ls["ci"] else None,
                         "ci_hi": ls["ci"][1] if ls["ci"] else None})
        for lab, sub in [("full", long), ("IS", _slice(long, hi=split)), ("OOS", _slice(long, lo=split))]:
            c = {x["streak"]: x for x in streak_gradient(sub, "alpha", seed=7)}.get("10+", {})
            rows.append({"horizon": h, "test": "streak10_alpha_lo", "outcome": lab,
                         "bucket": "10+&downtrend", "n": c.get("days", 0), "mean_pp": c.get("mean"),
                         "hit": c.get("hit"), "ci_lo": (c.get("ci") or [None, None])[0],
                         "ci_hi": (c.get("ci") or [None, None])[1]})
        # short-term-reversal control: Fama-MacBeth streak10 coefficient
        for lab, sub in [("full", long), ("IS", _slice(long, hi=split)), ("OOS", _slice(long, lo=split))]:
            fm = fama_macbeth(sub, "alpha", ["rback", "streak10"], sub["trend63"] < 0)
            s10 = fm["streak10"]
            rows.append({"horizon": h, "test": "streak10_alpha_revadj", "outcome": lab,
                         "bucket": "FM_coef", "n": s10["days"], "mean_pp": s10["mean"],
                         "hit": None, "ci_lo": (s10["ci"] or [None, None])[0],
                         "ci_hi": (s10["ci"] or [None, None])[1]})
    return pd.DataFrame(rows)


def load_full(start, end, cache_dir, workers, max_names=None, refresh=False):
    syms = N.fetch_ishares_holdings(N.IVV_PORTFOLIO_ID, label="IVV S&P 500")
    syms = [s for s in dict.fromkeys(syms) if s and s.isalpha()]
    if max_names:
        syms = syms[:max_names]
    print(f"S&P 500 constituents: {len(syms)} names", flush=True)
    SP = N.build_universe_panels(syms, start, end, workers=workers, cache_dir=cache_dir,
                                 ns="sp500", refresh=refresh, label="S&P 500")
    spy = N.load_yahoo_panels(["SPY"], start, end, workers=1, cache_dir=cache_dir,
                              refresh=refresh, label="SPY")["adjclose"]["SPY"]
    return SP, spy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--split-date", default=SPLIT_DATE)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    SP, spy = load_full(start, end, args.cache_dir, args.workers,
                        max_names=args.max_names, refresh=args.refresh)
    long_by_h = {h: build_long(SP, spy, h) for h in HORIZONS}
    any_long = long_by_h[HORIZONS[0]]
    print(f"\nPanel: {len(any_long):,} (name,day) obs · "
          f"{any_long.index.get_level_values('name').nunique()} names · "
          f"{any_long.index.get_level_values('date').nunique()} days")
    print(f"Downtrend days (3mo slope<0): {int((any_long['trend63'] < 0).mean()*100)}% of panel; "
          f"10d-avg DPI in top quintile on {int((any_long['p10'] >= HI).mean()*100)}% of days.")
    print(report(long_by_h, split=args.split_date))
    if args.out:
        summary_rows(long_by_h, split=args.split_date).to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
