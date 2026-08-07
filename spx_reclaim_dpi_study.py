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
Forward 21/42/63-session return, three ways: RAW, EXCESS vs SPY, and cross-
sectionally DEMEANED. Events are DE-OVERLAPPED per name (a `cooldown`-session gap
between a name's own entries). Because reclaims cluster in calendar time (they
fire market-wide after a market dip), the honest lenses are: a moving-block
bootstrap over date-sorted events, a per-NAME fraction-positive with a sign
test, a pre-2021 vs 2021+ split, and -- above all -- the R_dpi MINUS R_nodpi gap,
since any market-timing/overlap bias hits both equally.

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
            "fwd": fwd, "excess": excess, "xdemean": xdemean, "index": idx}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--reclaim-ma", type=int, default=RECLAIM_MA,
                    help="MA whose reclaim triggers entry (default 50; try 20)")
    ap.add_argument("--trend-ma", type=int, default=TREND_MA)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
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

    if args.out:
        summary_rows(frames, args.cooldown, args.reclaim_ma).to_csv(args.out, index=False)
        print(f"\nWrote summary -> {args.out}")


if __name__ == "__main__":
    main()
