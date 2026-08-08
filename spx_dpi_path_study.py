#!/usr/bin/env python3
"""
PATH study: HIGH 10-day-average DPI in a downtrend -- the shape of what happens
next, not just where it ends.
==============================================================================

Request: "test the path of a high 10 day average dpi in a downtrend". The
falling-knife study measured the DOWNSIDE of the path (forward max adverse
excursion) and found HIGH vs LOW DPI indistinguishable. This study characterises
the WHOLE path for the persistence setup:

  * MAE  -- forward max adverse excursion: min of adj[t+k]/adj[t]-1, k=1..h (%).
  * MFE  -- forward max favorable excursion: max over the same window (%).
  * TILT -- MFE + MAE: the path's bounce-vs-drawdown asymmetry (>0 = the best
            moment beats the worst moment in magnitude).
  * t_trough / t_peak -- sessions until the path min / max (does the bounce
            arrive earlier, i.e. does high dark flow mark a nearby bottom?).

Groups among 3-month-downtrend names: the 10-day-average-DPI self-percentile
quintiles (LOW <= 0.20 / MID / HIGH >= 0.80) AND the high-DPI streak buckets
from the persistence study (0 / 1-4 / 5-9 / 10+ consecutive top-quintile days).
Daily cross-sectional aggregation -> block-bootstrap CIs; HIGH-LOW and
S10+-minus-S0 differences; 2019-22 / 2023+ split on the headline differences.

Full windows only: rows in the last h sessions have no complete forward path
and are NaN (so MAE here can differ a hair from the falling-knife study, which
allowed half windows at the sample edge).

Reproduce
---------
    python spx_dpi_path_study.py --start 2019-01-01 --out spx_dpi_path.csv
"""
import argparse

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import ndx_dark_residual as N
import spx_xs_dip_dix_study as XS
import spx_dpi_persistence_study as P   # load_full, build_long (p10/streak/trend63)
import spx_falling_knife_study as F     # _daily, _daily_diff, _stat, LOWQ/HIQ

LOWQ, HIQ = F.LOWQ, F.HIQ
HORIZONS = (21, 42)
SPLIT_DATE = "2023-01-01"
DPI_GROUPS = ("LOW", "MID", "HIGH", "ALL")
STREAK_GROUPS = ("S0", "S1-4", "S5-9", "S10+")


def forward_path_stats(adj, horizon):
    """MAE / MFE / sessions-to-trough / sessions-to-peak over the FORWARD window
    t+1..t+horizon, vectorised via a sliding window. Full windows only; any NaN
    price inside the window makes all four stats NaN for that row."""
    A = adj.to_numpy(dtype=float)
    T, M = A.shape
    out = {k: np.full((T, M), np.nan) for k in ("mae", "mfe", "tmin", "tmax")}
    if T > horizon:
        W = sliding_window_view(A, horizon, axis=0)      # W[i] = A[i .. i+h-1]
        fwd = W[1:]                                      # window of t is W[t+1]
        base = A[: T - horizon]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mae"][: T - horizon] = (fwd.min(axis=2) / base - 1.0) * 100.0
            out["mfe"][: T - horizon] = (fwd.max(axis=2) / base - 1.0) * 100.0
            out["tmin"][: T - horizon] = np.argmin(fwd, axis=2) + 1.0
            out["tmax"][: T - horizon] = np.argmax(fwd, axis=2) + 1.0
        bad = ~np.isfinite(out["mae"])                   # NaN price in window/base
        for k in ("mfe", "tmin", "tmax"):
            out[k][bad] = np.nan
    return {k: pd.DataFrame(v, index=adj.index, columns=adj.columns)
            for k, v in out.items()}


def group_masks(long):
    down = long["trend63"] < 0
    return {"LOW": down & (long["p10"] <= LOWQ), "HIGH": down & (long["p10"] >= HIQ),
            "MID": down & (long["p10"] > LOWQ) & (long["p10"] < HIQ), "ALL": down,
            "S0": down & (long["streak"] == 0),
            "S1-4": down & long["streak"].between(1, 4),
            "S5-9": down & long["streak"].between(5, 9),
            "S10+": down & (long["streak"] >= 10)}


_MEAN = np.mean
def _p_mae10(v): return (v < -10).mean() * 100.0
def _p_mfe10(v): return (v > 10).mean() * 100.0


def path_rows(long):
    """Per-group path profile. Streak baskets are thinner than quintiles, so
    they use the persistence study's min basket (3) instead of 5."""
    m = group_masks(long)
    rows = []
    for g in DPI_GROUPS + STREAK_GROUPS:
        mask, mn = m[g], (P.MIN_BASKET if g in STREAK_GROUPS else F.MIN_BASKET)
        rows.append({
            "group": g, "n": int(long.loc[mask, "mae"].notna().sum()),
            "mae": F._stat(F._daily(long, mask, "mae", _MEAN, min_n=mn), seed=1),
            "mfe": F._stat(F._daily(long, mask, "mfe", _MEAN, min_n=mn), seed=2),
            "tilt": F._stat(F._daily(long, mask, "tilt", _MEAN, min_n=mn), seed=3),
            "p_mae10": F._stat(F._daily(long, mask, "mae", _p_mae10, min_n=mn), seed=4),
            "p_mfe10": F._stat(F._daily(long, mask, "mfe", _p_mfe10, min_n=mn), seed=5),
            "tmin": F._stat(F._daily(long, mask, "tmin", _MEAN, min_n=mn), seed=6),
            "tmax": F._stat(F._daily(long, mask, "tmax", _MEAN, min_n=mn), seed=7),
        })
    return rows


def path_diffs(long):
    m = group_masks(long)
    out = {}
    for lab, mA, mB, mn in [("HIGH-LOW", m["HIGH"], m["LOW"], F.MIN_BASKET),
                            ("S10+-S0", m["S10+"], m["S0"], P.MIN_BASKET)]:
        out[lab] = {
            "mae": F._stat(F._daily_diff(long, mA, mB, "mae", _MEAN, min_n=mn), seed=8),
            "mfe": F._stat(F._daily_diff(long, mA, mB, "mfe", _MEAN, min_n=mn), seed=9),
            "tilt": F._stat(F._daily_diff(long, mA, mB, "tilt", _MEAN, min_n=mn), seed=10),
            "p_mae10": F._stat(F._daily_diff(long, mA, mB, "mae", _p_mae10, min_n=mn), seed=11),
            "tmin": F._stat(F._daily_diff(long, mA, mB, "tmin", _MEAN, min_n=mn), seed=12),
        }
    return out


def path_reversal_control(long, split=SPLIT_DATE):
    """The endpoint streak alpha deflated under the reversal control, so the
    path metrics get the same test: Fama-MacBeth of MAE / MFE / TILT on
    [rback, streak10] among downtrend names -- the streak10 coefficient is the
    path effect BEYOND the recent-return characteristic, full/IS/OOS."""
    out = {}
    for y in ("mae", "mfe", "tilt"):
        out[y] = {}
        for lab, sub in [("full", long), ("IS", _slice(long, hi=split)),
                         ("OOS", _slice(long, lo=split))]:
            fm = P.fama_macbeth(sub, y, ["rback", "streak10"], sub["trend63"] < 0,
                                seed=hash((y, lab)) % 1000)
            out[y][lab] = fm["streak10"]
    return out


def _slice(long, lo=None, hi=None):
    dts = long.index.get_level_values("date")
    m = np.ones(len(long), dtype=bool)
    if lo is not None:
        m &= np.asarray(dts >= pd.Timestamp(lo))
    if hi is not None:
        m &= np.asarray(dts < pd.Timestamp(hi))
    return long[m]


def _ci(ci):
    return f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "--"


def report(long_by_h, split=SPLIT_DATE):
    out = []
    for h, long in long_by_h.items():
        out.append("\n" + "=" * 96)
        out.append(f"HORIZON {h}d  --  forward PATH shape among downtrend names")
        out.append("=" * 96)
        out.append(f"   {'grp':5s} {'n':>7}  {'mean MAE':>15}  {'mean MFE':>15}  "
                   f"{'TILT=MFE+MAE':>15}  {'P(mae<-10)':>10} {'P(mfe>+10)':>10}  "
                   f"{'t-trough':>8} {'t-peak':>7}")
        for r in path_rows(long):
            out.append(
                f"   {r['group']:5s} {r['n']:>7}  {r['mae']['mean']:+6.2f} {_ci(r['mae']['ci']):>8}  "
                f"{r['mfe']['mean']:+6.2f} {_ci(r['mfe']['ci']):>8}  "
                f"{r['tilt']['mean']:+6.2f} {_ci(r['tilt']['ci']):>8}  "
                f"{r['p_mae10']['mean']:>9.1f}% {r['p_mfe10']['mean']:>9.1f}%  "
                f"{r['tmin']['mean']:>8.1f} {r['tmax']['mean']:>7.1f}")
        d = path_diffs(long)
        for lab in ("HIGH-LOW", "S10+-S0"):
            x = d[lab]
            out.append(f"   >> {lab}: MAE {x['mae']['mean']:+.2f} {_ci(x['mae']['ci'])} | "
                       f"MFE {x['mfe']['mean']:+.2f} {_ci(x['mfe']['ci'])} | "
                       f"TILT {x['tilt']['mean']:+.2f} {_ci(x['tilt']['ci'])} | "
                       f"P(mae<-10) {x['p_mae10']['mean']:+.2f} {_ci(x['p_mae10']['ci'])} | "
                       f"t-trough {x['tmin']['mean']:+.2f} {_ci(x['tmin']['ci'])}")
        out.append(f"   OOS split on the headline differences (TILT and mean MAE):")
        for lab, sub in [("full", long), (f"IS<{split[:4]}", _slice(long, hi=split)),
                         (f"OOS>={split[:4]}", _slice(long, lo=split))]:
            dd = path_diffs(sub)
            out.append(f"      {lab:10s} S10+-S0 TILT {dd['S10+-S0']['tilt']['mean']:+.2f} "
                       f"{_ci(dd['S10+-S0']['tilt']['ci'])}  MAE {dd['S10+-S0']['mae']['mean']:+.2f} "
                       f"{_ci(dd['S10+-S0']['mae']['ci'])} | HIGH-LOW TILT "
                       f"{dd['HIGH-LOW']['tilt']['mean']:+.2f} {_ci(dd['HIGH-LOW']['tilt']['ci'])}")
        out.append("   REVERSAL CONTROL -- Fama-MacBeth path-metric ~ rback + streak10, "
                   "streak10 coefficient (pp beyond recent return):")
        rc = path_reversal_control(long, split=split)
        for y in ("mae", "mfe", "tilt"):
            cells = "  ".join(f"{lab} {rc[y][lab]['mean']:+.3f} {_ci(rc[y][lab]['ci'])}"
                              for lab in ("full", "IS", "OOS"))
            out.append(f"      {y:4s}: {cells}")
    return "\n".join(out)


def summary_rows(long_by_h):
    rows = []
    for h, long in long_by_h.items():
        for r in path_rows(long):
            rows.append({"horizon": h, "kind": "group", "group": r["group"], "n": r["n"],
                         "mae": r["mae"]["mean"], "mfe": r["mfe"]["mean"],
                         "tilt": r["tilt"]["mean"], "p_mae10": r["p_mae10"]["mean"],
                         "p_mfe10": r["p_mfe10"]["mean"], "t_trough": r["tmin"]["mean"],
                         "t_peak": r["tmax"]["mean"],
                         "tilt_ci_lo": (r["tilt"]["ci"] or [None, None])[0],
                         "tilt_ci_hi": (r["tilt"]["ci"] or [None, None])[1]})
        for lab, x in path_diffs(long).items():
            rows.append({"horizon": h, "kind": "diff", "group": lab, "n": x["mae"]["days"],
                         "mae": x["mae"]["mean"], "mfe": x["mfe"]["mean"],
                         "tilt": x["tilt"]["mean"], "p_mae10": x["p_mae10"]["mean"],
                         "p_mfe10": None, "t_trough": x["tmin"]["mean"], "t_peak": None,
                         "tilt_ci_lo": (x["tilt"]["ci"] or [None, None])[0],
                         "tilt_ci_hi": (x["tilt"]["ci"] or [None, None])[1]})
        rc = path_reversal_control(long)
        for y in ("mae", "mfe", "tilt"):
            for lab, s in rc[y].items():
                rows.append({"horizon": h, "kind": "fm_revadj", "group": f"{y}_{lab}",
                             "n": s["days"], "mae": None, "mfe": None, "tilt": s["mean"],
                             "p_mae10": None, "p_mfe10": None, "t_trough": None,
                             "t_peak": None,
                             "tilt_ci_lo": (s["ci"] or [None, None])[0],
                             "tilt_ci_hi": (s["ci"] or [None, None])[1]})
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
        ps = forward_path_stats(SP["adjclose"], h)
        for k in ("mae", "mfe", "tmin", "tmax"):
            lg[k] = ps[k].stack().reindex(lg.index)
        lg["tilt"] = lg["mfe"] + lg["mae"]
        long_by_h[h] = lg
    a = long_by_h[HORIZONS[0]]
    down = a["trend63"] < 0
    print(f"\nPanel: {len(a):,} obs · {a.index.get_level_values('name').nunique()} names · "
          f"{a.index.get_level_values('date').nunique()} days. Downtrend = "
          f"{int(down.mean() * 100)}% of panel; of those, HIGH 10d-avg DPI (>= {HIQ:.0%}) on "
          f"{int((a.loc[down, 'p10'] >= HIQ).mean() * 100)}%, 10+ streak on "
          f"{(a.loc[down, 'streak'] >= 10).mean() * 100:.1f}%.")
    print(report(long_by_h, split=args.split_date))
    if args.out:
        summary_rows(long_by_h).to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
