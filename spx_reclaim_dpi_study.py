#!/usr/bin/env python3
"""
"GOOGL-like reclaim" study: MA reclaim + persistent high DPI as a bottoming
signal, cross-sectional over the S&P 500.
===========================================================================

Motivating observation (per request): a name rallies earlier in the year, then
pulls back into a *mild* downtrend; through the pullback its DPI **rises sharply
and stays in the upper deciles for a few weeks**; then it **reclaims its moving
average** and rallies again. The prior generic "downtrend + high D" state was a
null -- this is a much more specific, conjunctive ENTRY pattern (a bottoming /
reclaim), so it gets an event study, and the real question is whether the
persistent-high-DPI condition ADDS anything to a plain MA reclaim.

Entry rule (real-time, no look-ahead), per name on day t
--------------------------------------------------------
A "reclaim" event fires when ALL hold:
  1. RECLAIM   : close crosses back ABOVE its `reclaim_ma`-day MA today
                 (close_t > MA_t and close_{t-1} <= MA_{t-1}).
  2. UPTREND   : close_t > its `trend_ma`-day MA  -- the longer trend is still up
                 ("rallied earlier in the year"; a pullback within an uptrend,
                 not a broken name).
  3. REAL DIP  : it had actually been below the `reclaim_ma`-day MA for >= half of
                 the prior `prior_win` sessions (a genuine multi-week pullback,
                 not one-day chop).
Events are then split by the dark-flow condition:
  4. DPI-hi    : the name's SELF-relative DPI percentile (within its own trailing
                 year) sat in the top quintile (>= `dpi_hi`) on >= half of the
                 last `persist_win` sessions -- "DPI rose and STAYED in the upper
                 deciles for a few weeks" through the pullback.

  * R_dpi   = reclaims WITH persistent high DPI (the requested setup)
  * R_nodpi = reclaims WITHOUT it (a plain reclaim) -- the control that isolates
              what the DPI condition is actually worth.

Outcome / inference
-------------------
Two questions. (1) MEAN: forward 21/42/63-session return, three ways -- RAW,
EXCESS vs SPY, cross-sectionally DEMEANED -- does high DPI pick the bigger
winner? (2) STICK (the conviction reframe): conditional on the reclaim, does high
DPI make the bounce HOLD -- path metrics over the next N sessions (whipsaw back
below the MA, give-back / max adverse excursion, blow-up <-10%, fraction of days
above the MA, still-above-MA at N)? Events are DE-OVERLAPPED per name (a
`cooldown`-session gap). Because reclaims cluster in calendar time (they fire
market-wide after a market dip), the honest lenses are: a moving-block bootstrap
over date-sorted events, a by-NAME cluster bootstrap on the stick metrics, a
per-NAME sign test, a pre-2021 vs 2021+ split, and -- above all -- the R_dpi
MINUS R_nodpi gap, since any market-timing/overlap bias hits both equally.

Data: reuses spx_xs_dip_dix_study.load_universe (FINRA D + Yahoo adjclose over
the current S&P 500, cached).

Reproduce
---------
    python spx_reclaim_dpi_study.py --start 2019-01-01
    python spx_reclaim_dpi_study.py --start 2019-01-01 --out spx_reclaim_dpi.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import spx_xs_dip_dix_study as XS   # load_universe, self_percentile, block_boot_ci, sign_test_p

RECLAIM_MA = 50         # the MA whose reclaim triggers entry
TREND_MA = 200          # longer trend that must still be up
DPI_HI = 0.80           # top-quintile self-relative DPI percentile
PERSIST_WIN = 15        # window over which DPI must have stayed high (~3 weeks)
PERSIST_FRAC = 0.5      # fraction of that window with DPI in the top quintile
PRIOR_WIN = 10          # window before the reclaim that must show a real pullback
PRIOR_FRAC = 0.5        # fraction of it spent below the reclaim MA
COOLDOWN = 21           # min sessions between a name's own entries
HORIZONS = (21, 42, 63)
REGIME_SPLIT = "2021-01-01"


def build_events(D, adj, spy, reclaim_ma=RECLAIM_MA, trend_ma=TREND_MA,
                 dpi_hi=DPI_HI, persist_win=PERSIST_WIN, persist_frac=PERSIST_FRAC,
                 prior_win=PRIOR_WIN, prior_frac=PRIOR_FRAC):
    """Return wide boolean masks (dates x names) for R_dpi and R_nodpi plus the
    forward-return / excess frames they'll be scored against."""
    cols = D.columns.intersection(adj.columns)
    idx = D.index.intersection(adj.index)
    D, adj = D.loc[idx, cols], adj.loc[idx, cols]
    ma_r = adj.rolling(reclaim_ma, min_periods=reclaim_ma // 2).mean()
    ma_t = adj.rolling(trend_ma, min_periods=trend_ma // 2).mean()
    dpct = XS.self_percentile(D)                              # trailing-year self-relative DPI %ile
    above = adj > ma_r
    reclaim = above & (~above.shift(1).fillna(False))         # cross above the reclaim MA today
    uptrend = adj > ma_t
    prior_pullback = (~above).rolling(prior_win).mean().shift(1) >= prior_frac
    base = reclaim & uptrend & prior_pullback.fillna(False) & ma_t.notna()
    persist_hi = (dpct >= dpi_hi).rolling(persist_win).mean() >= persist_frac
    ev_dpi = (base & persist_hi.fillna(False))
    ev_nodpi = (base & ~persist_hi.fillna(False))

    fwd = {h: N.compute_forward_return(adj, h) for h in HORIZONS}
    spy = spy.reindex(idx)
    spy_fwd = {h: (spy.shift(-h) / spy - 1.0) * 100.0 for h in HORIZONS}
    excess = {h: fwd[h].sub(spy_fwd[h], axis=0) for h in HORIZONS}
    xdemean = {h: fwd[h].sub(fwd[h].mean(axis=1), axis=0) for h in HORIZONS}
    return {"ev_dpi": ev_dpi, "ev_nodpi": ev_nodpi, "base": base,
            "fwd": fwd, "excess": excess, "xdemean": xdemean, "index": idx,
            "adj": adj, "ma_r": ma_r}


def deoverlap(mask, cooldown=COOLDOWN):
    """List of (date, name) events with a per-name `cooldown`-session gap."""
    entries = []
    idx = mask.index
    for name in mask.columns:
        col = mask[name].to_numpy()
        last = -10 ** 9
        for p in np.where(col)[0]:
            if p - last >= cooldown:
                entries.append((idx[p], name))
                last = p
    return entries


def _series_for(events, frame):
    """Pull the frame's value at each (date, name) event into a date-sorted Series."""
    if not events:
        return pd.Series(dtype=float)
    vals = [(d, frame.at[d, n]) for d, n in events]
    s = pd.Series([v for _, v in vals], index=pd.DatetimeIndex([d for d, _ in vals]))
    return s.sort_index().dropna()


def score(events, frames, h, seed=0):
    """Raw / excess / cross-sectional-demeaned forward-return stats for an event
    list at horizon h, with a moving-block bootstrap CI on the excess mean and a
    per-name fraction-positive sign test."""
    raw = _series_for(events, frames["fwd"][h])
    exc = _series_for(events, frames["excess"][h])
    xdm = _series_for(events, frames["xdemean"][h])
    ci = XS.block_boot_ci(exc.to_numpy(), seed=seed) if len(exc) else (np.nan, np.nan)
    # per-name fraction positive (excess), one vote per name
    dfe = pd.DataFrame(events, columns=["date", "name"])
    dfe["exc"] = [frames["excess"][h].at[d, n] for d, n in events]
    dfe = dfe.dropna(subset=["exc"])
    per = dfe.groupby("name")["exc"].mean()
    k = int((per > 0).sum())
    return {
        "n": int(len(raw)), "n_names": int(per.shape[0]),
        "raw": round(float(raw.mean()), 3) if len(raw) else np.nan,
        "raw_hit": round(float((raw > 0).mean() * 100)) if len(raw) else np.nan,
        "exc": round(float(exc.mean()), 3) if len(exc) else np.nan,
        "exc_hit": round(float((exc > 0).mean() * 100)) if len(exc) else np.nan,
        "exc_ci": (round(ci[0], 3), round(ci[1], 3)) if np.isfinite(ci[0]) else None,
        "xdm": round(float(xdm.mean()), 3) if len(xdm) else np.nan,
        "names_pos": round(k / per.shape[0] * 100, 1) if per.shape[0] else np.nan,
        "sign_p": round(XS.sign_test_p(k, per.shape[0]), 4) if per.shape[0] else np.nan,
    }


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "--"


def report(frames, cooldown=COOLDOWN):
    ev_dpi = deoverlap(frames["ev_dpi"], cooldown)
    ev_nodpi = deoverlap(frames["ev_nodpi"], cooldown)
    out = [f"Events (de-overlapped, {cooldown}d per-name cooldown):  "
           f"R_dpi = {len(ev_dpi)}   R_nodpi = {len(ev_nodpi)}"]
    for h in HORIZONS:
        a = score(ev_dpi, frames, h, seed=1)
        b = score(ev_nodpi, frames, h, seed=2)
        out.append(f"\n--- forward {h}d ---")
        out.append(f"  R_dpi   (n={a['n']:>4}, {a['n_names']} names): "
                   f"raw {a['raw']:+.3f}% (hit {a['raw_hit']:.0f}%) | "
                   f"excess-SPY {a['exc']:+.3f}% (hit {a['exc_hit']:.0f}%, 95% CI {_fmt_ci(a['exc_ci'])}) | "
                   f"xdemean {a['xdm']:+.3f}% | names+ {a['names_pos']}% (p={a['sign_p']})")
        out.append(f"  R_nodpi (n={b['n']:>4}, {b['n_names']} names): "
                   f"raw {b['raw']:+.3f}% (hit {b['raw_hit']:.0f}%) | "
                   f"excess-SPY {b['exc']:+.3f}% (hit {b['exc_hit']:.0f}%, 95% CI {_fmt_ci(b['exc_ci'])}) | "
                   f"xdemean {b['xdm']:+.3f}% | names+ {b['names_pos']}%")
        if np.isfinite(a["exc"]) and np.isfinite(b["exc"]):
            out.append(f"  --> DPI MARGINAL (R_dpi - R_nodpi), excess-SPY: {a['exc'] - b['exc']:+.3f} pp")
    # regime split on the 21d excess of R_dpi
    out.append("\nRegime split (R_dpi, 21d excess-SPY):")
    for lab, keep in [("pre-2021", lambda d: d < pd.Timestamp(REGIME_SPLIT)),
                      ("2021+", lambda d: d >= pd.Timestamp(REGIME_SPLIT))]:
        sub = [(d, n) for d, n in ev_dpi if keep(d)]
        s = score(sub, frames, 21, seed=3)
        out.append(f"  {lab:9s} n={s['n']:>4}  excess {s['exc']:+.3f}%  hit {s['exc_hit']:.0f}%  "
                   f"95% CI {_fmt_ci(s['exc_ci'])}")
    return "\n".join(out), ev_dpi, ev_nodpi


def summary_rows(frames, cooldown=COOLDOWN, reclaim_ma=RECLAIM_MA):
    ev = {"R_dpi": deoverlap(frames["ev_dpi"], cooldown),
          "R_nodpi": deoverlap(frames["ev_nodpi"], cooldown)}
    rows = []
    for grp, events in ev.items():
        for h in HORIZONS:
            s = score(events, frames, h, seed=hash((grp, h)) % 1000)
            rows.append({"reclaim_ma": reclaim_ma, "group": grp, "horizon": h,
                         "n": s["n"], "n_names": s["n_names"], "raw_pp": s["raw"],
                         "raw_hit": s["raw_hit"], "excess_spy_pp": s["exc"],
                         "excess_hit": s["exc_hit"],
                         "exc_ci_lo": s["exc_ci"][0] if s["exc_ci"] else None,
                         "exc_ci_hi": s["exc_ci"][1] if s["exc_ci"] else None,
                         "xdemean_pp": s["xdm"], "names_pos_pct": s["names_pos"],
                         "sign_p": s["sign_p"]})
    return pd.DataFrame(rows)


def stick_paths(events, adj, ma_r, N):
    """Path-dependent 'did the bounce STICK' metrics for each event over the next
    N sessions -- conditional on the reclaim, not about the mean drift. Returns a
    per-event DataFrame with:
      fail       -- did close fall back BELOW the reclaimed MA at least once (whipsaw)
      maxdd      -- worst entry-relative return over the window, % (give-back; ~0 = sticky)
      mfe        -- best entry-relative return over the window, %
      frac_above -- fraction of the N days held above the reclaimed MA
      hold       -- still above the MA at day N
      blow10     -- suffered a >10% drawdown from entry (a failed bounce)
      endret     -- entry-relative return at day N, %
    """
    posmap = {d: i for i, d in enumerate(adj.index)}
    rows = []
    from collections import defaultdict
    byname = defaultdict(list)
    for d, n in events:
        byname[n].append(d)
    for n, dates in byname.items():
        pv = adj[n].to_numpy(dtype=float)
        mv = ma_r[n].to_numpy(dtype=float)
        for dt in dates:
            p = posmap[dt]
            e = pv[p]
            if not np.isfinite(e) or e <= 0:
                continue
            path = pv[p + 1: p + 1 + N]
            mpath = mv[p + 1: p + 1 + N]
            valid = np.isfinite(path) & np.isfinite(mpath)
            if valid.sum() < max(5, N // 3):
                continue
            path, mpath = path[valid], mpath[valid]
            cr = path / e - 1.0
            above = path > mpath
            rows.append({
                "name": n,
                "fail": bool((~above).any()),
                "maxdd": float(cr.min()) * 100.0,
                "mfe": float(cr.max()) * 100.0,
                "frac_above": float(above.mean()),
                "hold": bool(above[-1]),
                "blow10": bool(cr.min() < -0.10),
                "endret": float(cr[-1]) * 100.0,
            })
    return pd.DataFrame(rows)


def _agg(df):
    """Group-level 'stick' summary from a per-event stick DataFrame."""
    if df.empty:
        return {"n": 0}
    return {"n": int(len(df)), "fail_pct": round(df["fail"].mean() * 100, 1),
            "maxdd_mean": round(df["maxdd"].mean(), 2), "maxdd_med": round(df["maxdd"].median(), 2),
            "blow10_pct": round(df["blow10"].mean() * 100, 1),
            "frac_above": round(df["frac_above"].mean() * 100, 1),
            "hold_pct": round(df["hold"].mean() * 100, 1),
            "endret_mean": round(df["endret"].mean(), 2)}


def cluster_boot_diff(dfa, dfb, col, is_bool, B=2000, seed=0):
    """95% CI on (metric_A - metric_B) via a by-NAME cluster bootstrap (resample
    names, pool their events) -- respects within-name correlation. For a bool col
    the metric is a rate (%), for a float col it is the pooled mean."""
    names = np.array(sorted(set(dfa["name"]) | set(dfb["name"])))
    if len(names) == 0:
        return None
    def vecs(df):
        g = df.groupby("name")[col]
        s = (g.sum() if is_bool else g.sum()).reindex(names).fillna(0).to_numpy(dtype=float)
        c = g.size().reindex(names).fillna(0).to_numpy(dtype=float)
        return s, c
    sa, ca = vecs(dfa)
    sb, cb = vecs(dfb)
    scale = 100.0 if is_bool else 1.0
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(names), size=(B, len(names)))
    A = sa[draw].sum(1) / np.maximum(ca[draw].sum(1), 1e-9) * scale
    Bv = sb[draw].sum(1) / np.maximum(cb[draw].sum(1), 1e-9) * scale
    d = A - Bv
    return round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3), round(float(d.mean()), 3)


def paired_by_name(dfa, dfb, col, min_ev=3, stickier="lower"):
    """Paired 'no matter the stock' test: among names with >= min_ev events in
    BOTH groups, in what fraction is R_dpi stickier than R_nodpi on `col`?
    `stickier`='lower' for fail/blow10/maxdd-as-drawdown-magnitude... but note we
    compare the metric directly: for fail_pct lower is better; for frac_above/hold
    higher is better; for maxdd (a negative %) higher (less negative) is better."""
    a = dfa.groupby("name")[col].mean()
    b = dfb.groupby("name")[col].mean()
    na = dfa.groupby("name").size()
    nb = dfb.groupby("name").size()
    common = [n for n in a.index if n in b.index and na[n] >= min_ev and nb[n] >= min_ev]
    if not common:
        return {"names": 0}
    diff = np.array([a[n] - b[n] for n in common])          # R_dpi - R_nodpi
    better = (diff < 0) if stickier == "lower" else (diff > 0)
    k = int(better.sum())
    return {"names": len(common), "dpi_stickier_pct": round(k / len(common) * 100, 1),
            "sign_p": round(XS.sign_test_p(k, len(common)), 4),
            "median_diff": round(float(np.median(diff)), 3)}


def stick_report(frames, N, cooldown=COOLDOWN):
    ev_dpi = deoverlap(frames["ev_dpi"], cooldown)
    ev_nodpi = deoverlap(frames["ev_nodpi"], cooldown)
    da = stick_paths(ev_dpi, frames["adj"], frames["ma_r"], N)
    db = stick_paths(ev_nodpi, frames["adj"], frames["ma_r"], N)
    A, B = _agg(da), _agg(db)
    out = [f"=== DOES THE BOUNCE STICK?  forward window N={N} sessions "
           f"({round(N/21)}mo) ===",
           "  ('stick' = holds above the reclaimed MA, shallow give-back, no blow-up "
           "-- NOT mean drift)",
           f"  {'metric':22s} {'R_dpi':>10} {'R_nodpi':>10}   diff (R_dpi-R_nodpi) [95% CI, by-name cluster boot]"]
    specs = [("fail_pct", "fail % (whipsaw)", "fail", True, "lower"),
             ("blow10_pct", "blow-up % (<-10%)", "blow10", True, "lower"),
             ("maxdd_mean", "mean give-back %", "maxdd", False, "higher"),
             ("frac_above", "% days above MA", "frac_above", False, "higher"),
             ("hold_pct", "still above MA @N %", "hold", True, "higher"),
             ("endret_mean", "end return %", "endret", False, "higher")]
    for key, lab, col, is_bool, _dir in specs:
        ci = cluster_boot_diff(da, db, col, is_bool) if (len(da) and len(db)) else None
        av, bv = A.get(key, np.nan), B.get(key, np.nan)
        cistr = f"{ci[2]:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "--"
        out.append(f"  {lab:22s} {av:>10} {bv:>10}   {cistr}")
    out.append(f"  (n: R_dpi={A.get('n',0)}, R_nodpi={B.get('n',0)})")
    # paired 'no matter the stock' on the two headline stick metrics
    pf = paired_by_name(da, db, "fail", stickier="lower")
    pd_ = paired_by_name(da, db, "maxdd", stickier="higher")
    if pf.get("names"):
        out.append(f"  paired-by-name (>=3 events each): R_dpi has LOWER fail% in "
                   f"{pf['dpi_stickier_pct']}% of {pf['names']} names (sign-p={pf['sign_p']}); "
                   f"shallower give-back in {pd_['dpi_stickier_pct']}% (sign-p={pd_['sign_p']})")
    return "\n".join(out)


def stick_summary_rows(frames, windows=(21, 42, 63), cooldown=COOLDOWN):
    """Tidy stick metrics per (window) for R_dpi vs R_nodpi plus the by-name
    cluster-bootstrap diff and CI on the two headline metrics."""
    ev_dpi = deoverlap(frames["ev_dpi"], cooldown)
    ev_nodpi = deoverlap(frames["ev_nodpi"], cooldown)
    rows = []
    for N in windows:
        da = stick_paths(ev_dpi, frames["adj"], frames["ma_r"], N)
        db = stick_paths(ev_nodpi, frames["adj"], frames["ma_r"], N)
        A, B = _agg(da), _agg(db)
        fail_ci = cluster_boot_diff(da, db, "fail", True) if (len(da) and len(db)) else None
        dd_ci = cluster_boot_diff(da, db, "maxdd", False) if (len(da) and len(db)) else None
        rows.append({
            "window": N, "months": round(N / 21),
            "n_dpi": A.get("n", 0), "n_nodpi": B.get("n", 0),
            "fail_pct_dpi": A.get("fail_pct"), "fail_pct_nodpi": B.get("fail_pct"),
            "fail_diff": fail_ci[2] if fail_ci else None,
            "fail_ci_lo": fail_ci[0] if fail_ci else None, "fail_ci_hi": fail_ci[1] if fail_ci else None,
            "giveback_dpi": A.get("maxdd_mean"), "giveback_nodpi": B.get("maxdd_mean"),
            "giveback_diff": dd_ci[2] if dd_ci else None,
            "giveback_ci_lo": dd_ci[0] if dd_ci else None, "giveback_ci_hi": dd_ci[1] if dd_ci else None,
            "blow10_dpi": A.get("blow10_pct"), "blow10_nodpi": B.get("blow10_pct"),
            "holdN_dpi": A.get("hold_pct"), "holdN_nodpi": B.get("hold_pct"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--reclaim-ma", type=int, default=RECLAIM_MA,
                    help="MA whose reclaim triggers entry (default 50; try 20)")
    ap.add_argument("--trend-ma", type=int, default=TREND_MA)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    ap.add_argument("--stick-window", type=int, default=42,
                    help="forward window (sessions) for the 'does it stick' path metrics (default 42)")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    D, adj, spy, syms = XS.load_universe(start, end, args.cache_dir, args.workers,
                                         max_names=args.max_names, refresh=args.refresh)
    frames = build_events(D, adj, spy, reclaim_ma=args.reclaim_ma, trend_ma=args.trend_ma)
    print(f"\nUniverse: {len(frames['index'])} sessions "
          f"[{frames['index'].min().date()} -> {frames['index'].max().date()}], "
          f"{frames['ev_dpi'].shape[1]} names")
    print(f"Setup: reclaim {args.reclaim_ma}d MA, uptrend > {args.trend_ma}d MA, "
          f"persistent DPI top-quintile >= {int(PERSIST_FRAC*100)}% of {PERSIST_WIN}d.\n")
    print("=" * 78)
    print("RECLAIM + PERSISTENT-HIGH-DPI (R_dpi) vs plain reclaim (R_nodpi)")
    print("=" * 78)
    txt, _, _ = report(frames, args.cooldown)
    print(txt)
    print("\n" + "=" * 78)
    print("CONVICTION / 'DOES THE BOUNCE STICK' -- path metrics conditional on the reclaim")
    print("=" * 78)
    for nwin in dict.fromkeys([21, args.stick_window, 63]):
        print(stick_report(frames, nwin, args.cooldown))
        print()

    if args.out:
        summary_rows(frames, args.cooldown, args.reclaim_ma).to_csv(args.out, index=False)
        print(f"\nWrote summary -> {args.out}")
        stick_path = args.out.replace(".csv", "") + "_stick.csv"
        stick_summary_rows(frames, cooldown=args.cooldown).to_csv(stick_path, index=False)
        print(f"Wrote stick summary -> {stick_path}")


if __name__ == "__main__":
    main()
