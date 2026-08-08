#!/usr/bin/env python3
"""
Falling-knife filter: do LOW-DPI downtrend stocks keep selling off?
==================================================================

Reframe (per request): downtrend stocks bounce on average, but catching a falling
knife is dangerous. Instead of asking whether HIGH DPI wins (null), ask whether
LOW 10-day-average DPI in a downtrend identifies the knives that keep dropping --
worse hit rate, fatter left tail. If DPI can SCREEN OUT losers, that loss
reduction is alpha even when the mean is unchanged.

Groups (among names in a 3-month downtrend; DPI = self-relative percentile of the
10-day average DPI, no look-ahead):
  LOW   = bottom quintile (p10 <= 0.20)   -- the falling-knife candidates
  HIGH  = top quintile    (p10 >= 0.80)   -- dark accumulation
  ALL   = every downtrend name            -- the unfiltered baseline
  KEEP  = ALL minus LOW (p10 > 0.20)      -- the filtered book

Outcomes (forward 21/42d), RAW and CAPM ALPHA (fwd - beta*SPY_fwd):
  mean (bounce size) | hit% (fraction up) | P(< -10%), P(< -20%) (falling-knife
  rate) | 5th-percentile & worst-decile mean (left tail).
Inference: daily cross-sectional value -> block-bootstrap CI (overlap-aware);
HIGH-minus-LOW and KEEP-minus-ALL differences with CIs; a short-term-reversal
control (are LOW names just deeper recent losers?) via Fama-MacBeth of a big-loss
dummy on the reversal characteristic + a LOW dummy; and a 2019-22 / 2023+ split.

Reproduce
---------
    python spx_falling_knife_study.py --start 2019-01-01 --out spx_falling_knife.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import spx_xs_dip_dix_study as XS
import spx_dpi_persistence_study as P   # load_full, build_long, fama_macbeth

LOWQ, HIQ = 0.20, 0.80
MIN_BASKET = 5
SPLIT_DATE = "2023-01-01"
HORIZONS = (21, 42)


def _daily(long, mask, col, fn, min_n=MIN_BASKET):
    d = long[mask]
    out = []
    for _, g in d.groupby(level="date"):
        v = g[col].dropna().to_numpy()
        if len(v) >= min_n:
            out.append(float(fn(v)))
    return np.asarray(out, dtype=float)


def _daily_diff(long, mA, mB, col, fn, min_n=MIN_BASKET):
    a = long[mA].groupby(level="date")
    b = {d: g for d, g in long[mB].groupby(level="date")}
    out = []
    for d, ga in a:
        if d not in b:
            continue
        va, vb = ga[col].dropna().to_numpy(), b[d][col].dropna().to_numpy()
        if len(va) >= min_n and len(vb) >= min_n:
            out.append(float(fn(va) - fn(vb)))
    return np.asarray(out, dtype=float)


def _stat(a, seed=0):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"days": 0, "mean": np.nan, "ci": None}
    ci = XS.block_boot_ci(a, seed=seed)
    return {"days": int(len(a)), "mean": round(float(a.mean()), 3),
            "ci": (round(ci[0], 3), round(ci[1], 3)) if np.isfinite(ci[0]) else None}


_MEAN = np.mean
def _hit(v): return (v > 0).mean() * 100.0
def _loss10(v): return (v < -10).mean() * 100.0
def _loss20(v): return (v < -20).mean() * 100.0


def forward_path_min(adj, horizon):
    """Forward MAX ADVERSE EXCURSION: min over the next `horizon` sessions of
    adj[t+k]/adj[t] - 1 (k = 1..horizon), in %. This is the PATH answer to
    'did it keep selling off' -- an endpoint return can hide a deep intraperiod
    knife that would have stopped a real position out. Computed by a reversed
    trailing rolling-min (vectorised), shifted so today's own close is excluded."""
    rev_min = adj.iloc[::-1].rolling(horizon, min_periods=max(3, horizon // 2)).min().iloc[::-1]
    fmin = rev_min.shift(-1)                     # window becomes t+1 .. t+horizon
    return (fmin / adj - 1.0) * 100.0


def _mae10(v): return (v < -10).mean() * 100.0
def _mae15(v): return (v < -15).mean() * 100.0


def path_profile(long, mae_col="mae"):
    """Path-based falling-knife profile by DPI group: mean forward MAE, the
    probability the path dips below -10% / -15% at ANY point in the window, and
    the share of 'hidden knives' (path < -10% but endpoint positive) that the
    endpoint tables cannot see."""
    m = group_masks(long)
    rows = []
    hidden = pd.Series(
        np.where(long[mae_col].isna() | long["fwd"].isna(), np.nan,
                 ((long[mae_col] < -10) & (long["fwd"] > 0)).astype(float)),
        index=long.index)
    for g in ("LOW", "MID", "HIGH", "ALL", "KEEP"):
        mask = m[g]
        pooled = long.loc[mask, mae_col].dropna()
        hid = _stat(_daily(long.assign(_h=hidden), mask, "_h", _MEAN), seed=4)
        rows.append({
            "group": g, "n": int(len(pooled)),
            "mae_mean": _stat(_daily(long, mask, mae_col, _MEAN), seed=1),
            "mae10": _stat(_daily(long, mask, mae_col, _mae10), seed=2),
            "mae15": _stat(_daily(long, mask, mae_col, _mae15), seed=3),
            "hidden": {"days": hid["days"],
                       "mean": round(hid["mean"] * 100, 2) if np.isfinite(hid["mean"]) else np.nan,
                       "ci": (round(hid["ci"][0] * 100, 2), round(hid["ci"][1] * 100, 2))
                             if hid["ci"] else None},
            "mae_p5": round(float(np.percentile(pooled, 5)), 2) if len(pooled) else np.nan,
        })
    dif = {}
    for lab, mA, mB in [("HIGH-LOW", m["HIGH"], m["LOW"]), ("KEEP-ALL", m["KEEP"], m["ALL"])]:
        dif[lab] = {
            "mae_mean": _stat(_daily_diff(long, mA, mB, mae_col, _MEAN), seed=5),
            "mae10": _stat(_daily_diff(long, mA, mB, mae_col, _mae10), seed=6),
        }
    return rows, dif


def group_masks(long):
    down = long["trend63"] < 0
    return {"LOW": down & (long["p10"] <= LOWQ), "HIGH": down & (long["p10"] >= HIQ),
            "MID": down & (long["p10"] > LOWQ) & (long["p10"] < HIQ),
            "ALL": down, "KEEP": down & (long["p10"] > LOWQ)}


def profile(long, outcome):
    """Per-group daily-mean stats for the loss/bounce profile."""
    m = group_masks(long)
    rows = []
    for g in ("LOW", "MID", "HIGH", "ALL", "KEEP"):
        mask = m[g]
        # pooled tail stats (descriptive)
        pooled = long.loc[mask, outcome].dropna().to_numpy()
        p5 = float(np.percentile(pooled, 5)) if len(pooled) else np.nan
        worst = pooled[pooled <= np.percentile(pooled, 10)] if len(pooled) else np.array([])
        rows.append({
            "group": g, "n": int(len(pooled)),
            "mean": _stat(_daily(long, mask, outcome, _MEAN), seed=1),
            "hit": _stat(_daily(long, mask, outcome, _hit), seed=2),
            "loss10": _stat(_daily(long, mask, outcome, _loss10), seed=3),
            "loss20": _stat(_daily(long, mask, outcome, _loss20), seed=4),
            "p5": round(p5, 2), "cvar10": round(float(worst.mean()), 2) if len(worst) else np.nan,
        })
    return rows


def _ci(ci):
    return f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "--"


def diffs(long, outcome):
    """HIGH-minus-LOW and KEEP-minus-ALL daily-difference stats (with CIs)."""
    m = group_masks(long)
    out = {}
    for lab, mA, mB in [("HIGH-LOW", m["HIGH"], m["LOW"]), ("KEEP-ALL", m["KEEP"], m["ALL"])]:
        out[lab] = {
            "mean": _stat(_daily_diff(long, mA, mB, outcome, _MEAN), seed=5),
            "hit": _stat(_daily_diff(long, mA, mB, outcome, _hit), seed=6),
            "loss10": _stat(_daily_diff(long, mA, mB, outcome, _loss10), seed=7),
        }
    return out


def reversal_control(long):
    """Is LOW DPI just a deeper recent loser? Mean rback by group, and a
    Fama-MacBeth linear-probability model of a big-loss dummy (fwd < -10) on
    [rback, LOW dummy] among downtrend names -- LOW coef = extra falling-knife
    probability beyond recent return."""
    down = long["trend63"] < 0
    d = long.copy()
    d["lowdpi"] = (d["p10"] <= LOWQ).astype(float)
    # NaN forward returns (the last h sessions) must stay NaN, not count as
    # "no big loss" -- fama_macbeth then drops them via its dropna.
    d["bigloss"] = np.where(d["fwd"].isna(), np.nan, (d["fwd"] < -10).astype(float))
    rb = {g: round(float(long.loc[m & down, "rback"].mean()), 2)
          for g, m in [("LOW", long["p10"] <= LOWQ), ("HIGH", long["p10"] >= HIQ)]}
    fm = P.fama_macbeth(d, "bigloss", ["rback", "lowdpi"], down)
    return rb, fm["lowdpi"], fm["rback"]


def _slice(long, lo=None, hi=None):
    dts = long.index.get_level_values("date")
    m = np.ones(len(long), dtype=bool)
    if lo is not None:
        m &= np.asarray(dts >= pd.Timestamp(lo))
    if hi is not None:
        m &= np.asarray(dts < pd.Timestamp(hi))
    return long[m]


def report(long_by_h, split=SPLIT_DATE):
    out = []
    for h, long in long_by_h.items():
        out.append("\n" + "=" * 90)
        out.append(f"HORIZON {h}d   (RAW = does it bounce; ALPHA = beta-adjusted)")
        out.append("=" * 90)
        for oc, olab in [("fwd", "RAW"), ("alpha", "ALPHA")]:
            out.append(f"\n-- {olab} forward return, by DPI group among downtrend names --")
            out.append(f"   {'grp':4s} {'n':>7}  {'mean':>16}  {'hit%':>15}  "
                       f"{'P(<-10%)':>15}  {'P(<-20%)':>14}   p5    cvar10")
            for r in profile(long, oc):
                out.append(
                    f"   {r['group']:4s} {r['n']:>7}  {r['mean']['mean']:+6.2f} {_ci(r['mean']['ci']):>9}  "
                    f"{r['hit']['mean']:5.1f} {_ci(r['hit']['ci']):>9}  "
                    f"{r['loss10']['mean']:5.1f} {_ci(r['loss10']['ci']):>9}  "
                    f"{r['loss20']['mean']:4.1f}   {r['p5']:+6.1f} {r['cvar10']:+6.1f}")
            df = diffs(long, oc)
            for lab in ("HIGH-LOW", "KEEP-ALL"):
                d = df[lab]
                out.append(f"   >> {lab}: mean {d['mean']['mean']:+.2f} {_ci(d['mean']['ci'])} | "
                           f"hit% {d['hit']['mean']:+.2f} {_ci(d['hit']['ci'])} | "
                           f"P(<-10%) {d['loss10']['mean']:+.2f} {_ci(d['loss10']['ci'])}")
        # PATH view: forward max adverse excursion (the actual "kept selling off" test)
        if "mae" in long.columns:
            rows_p, dif_p = path_profile(long)
            out.append(f"\n-- PATH: forward MAX ADVERSE EXCURSION over {h}d "
                       "(min of the path, not the endpoint) --")
            out.append(f"   {'grp':4s} {'n':>7}  {'mean MAE':>16}  {'P(path<-10%)':>16}  "
                       f"{'P(path<-15%)':>15}  {'hidden knives%':>15}  MAE p5")
            for r in rows_p:
                out.append(
                    f"   {r['group']:4s} {r['n']:>7}  {r['mae_mean']['mean']:+6.2f} "
                    f"{_ci(r['mae_mean']['ci']):>9}  {r['mae10']['mean']:5.1f} {_ci(r['mae10']['ci']):>10}  "
                    f"{r['mae15']['mean']:5.1f} {_ci(r['mae15']['ci']):>9}  "
                    f"{r['hidden']['mean']:5.1f} {_ci(r['hidden']['ci']):>9}  {r['mae_p5']:+7.1f}")
            for lab in ("HIGH-LOW", "KEEP-ALL"):
                d = dif_p[lab]
                out.append(f"   >> {lab}: mean MAE {d['mae_mean']['mean']:+.2f} "
                           f"{_ci(d['mae_mean']['ci'])} | P(path<-10%) {d['mae10']['mean']:+.2f} "
                           f"{_ci(d['mae10']['ci'])}")
        # reversal control (raw big-loss)
        rb, low_fm, rb_fm = reversal_control(long)
        out.append(f"\n-- falling-knife reversal control (fwd {h}d) --")
        out.append(f"   mean prior-21d return: LOW {rb['LOW']:+.2f}%  HIGH {rb['HIGH']:+.2f}%  "
                   "(is LOW just a deeper recent loser?)")
        out.append(f"   Fama-MacBeth  P(loss<-10%) ~ rback + LOW-dummy:  LOW coef "
                   f"{low_fm['mean']:+.3f} {_ci(low_fm['ci'])} (pp of big-loss prob beyond reversal)")
        # OOS split on the headline: KEEP-ALL hit-rate improvement (raw) + HIGH-LOW loss10 (raw)
        out.append(f"   OOS split (raw): KEEP-ALL hit% and HIGH-LOW P(<-10%):")
        for lab, sub in [("full", long), (f"IS<{split[:4]}", _slice(long, hi=split)),
                         (f"OOS>={split[:4]}", _slice(long, lo=split))]:
            dd = diffs(sub, "fwd")
            out.append(f"      {lab:10s} KEEP-ALL hit% {dd['KEEP-ALL']['hit']['mean']:+.2f} "
                       f"{_ci(dd['KEEP-ALL']['hit']['ci'])} | HIGH-LOW P(<-10%) "
                       f"{dd['HIGH-LOW']['loss10']['mean']:+.2f} {_ci(dd['HIGH-LOW']['loss10']['ci'])}")
    return "\n".join(out)


def summary_rows(long_by_h):
    rows = []
    for h, long in long_by_h.items():
        for oc in ("fwd", "alpha"):
            for r in profile(long, oc):
                rows.append({"horizon": h, "outcome": oc, "group": r["group"], "n": r["n"],
                             "mean": r["mean"]["mean"], "hit_pct": r["hit"]["mean"],
                             "loss10_pct": r["loss10"]["mean"], "loss20_pct": r["loss20"]["mean"],
                             "p5": r["p5"], "cvar10": r["cvar10"]})
        if "mae" in long.columns:
            rows_p, _ = path_profile(long)
            for r in rows_p:
                rows.append({"horizon": h, "outcome": "path_mae", "group": r["group"],
                             "n": r["n"], "mean": r["mae_mean"]["mean"],
                             "hit_pct": None, "loss10_pct": r["mae10"]["mean"],
                             "loss20_pct": r["mae15"]["mean"], "p5": r["mae_p5"],
                             "cvar10": r["hidden"]["mean"]})
    return pd.DataFrame(rows)


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
    SP, spy = P.load_full(start, end, args.cache_dir, args.workers,
                          max_names=args.max_names, refresh=args.refresh)
    long_by_h = {}
    for h in HORIZONS:
        lg = P.build_long(SP, spy, h)
        lg["mae"] = forward_path_min(SP["adjclose"], h).stack().reindex(lg.index)
        long_by_h[h] = lg
    a = long_by_h[HORIZONS[0]]
    down = a["trend63"] < 0
    print(f"\nPanel: {len(a):,} obs · {a.index.get_level_values('name').nunique()} names · "
          f"{a.index.get_level_values('date').nunique()} days. Downtrend = {int(down.mean()*100)}% of panel; "
          f"of those, LOW 10d-avg DPI (<= {LOWQ:.0%}) on {int((a.loc[down,'p10']<=LOWQ).mean()*100)}%.")
    print(report(long_by_h, split=args.split_date))
    if args.out:
        summary_rows(long_by_h).to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
