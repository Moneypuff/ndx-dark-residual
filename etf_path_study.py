#!/usr/bin/env python3
"""
Post-signal path study across the regime-log leaderboard universe -- the
GDX_CHASE study (gdx_chase_study.py) generalized to every sector/theme ETF.

For each of the 21 leaderboard ETFs and three event families

  * up-breaks   -- high-conviction (score >= 3) breaks with the 21d move up
                   (the "chase" entry the leaderboard's follow-through column
                   describes),
  * down-breaks -- the same score in a down move (capitulation),
  * turns       -- detect_breaks on the ETF-vs-SPY 63d spread, sgn=-1 (an
                   entrenched underperformance regime cracking upward),

this maps the forward own-price path and prints, per ETF:

  n, median forward return at 21/63/126 sessions, hit rate and mean at 63
  (mean - median = the tail skew the sizing section of the playbook uses),
  median / q25 max adverse excursion inside the first quarter and its trough
  day, median max favorable excursion and its peak day, the share of events
  dipping >5% / >8% below the entry, the day the median path tops (the
  natural hold period), and for up-breaks the dip-entry alternative (limit 8%
  under the event close working 42 sessions, exit 63 sessions after fill).

Classification used by ETF_PATH_PLAYBOOK.md (computed here, thresholds in
class_of()): CHASER (median d63 >= +2 and hit >= 55), FADER (median d63 <=
-2), else ROUND-TRIP. --csv-out writes the tidy stats table; --paths-out
writes each ETF's median up-break path (for charting).

    python etf_path_study.py --cache-dir .ndx_dark_cache --csv-out etf_path_stats.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
from build_regime_log import (SECTORS, THEMES, conviction_frame,
                              conviction_events, detect_breaks, spread_frames)
from gdx_chase_study import H, LIMIT_WINDOW, path_matrix

CHECKPOINTS = (21, 63, 126)


def family_stats(close, dates, horizon=H):
    """One stats row for a set of event dates on one ETF's own price."""
    dates = [d for d in dates]
    if len(dates) < 8:
        return None
    P = path_matrix(close, dates, horizon)
    M = P.to_numpy()
    row = {"n": M.shape[1]}
    for k in CHECKPOINTS:
        row[f"med{k}"] = float(np.nanmedian(M[k]))
    row["hit63"] = float(np.nanmean(M[63] > 0) * 100)
    row["mean63"] = float(np.nanmean(M[63]))
    row["skew63"] = row["mean63"] - row["med63"]
    sub = M[1:64]
    ok = ~np.all(np.isnan(sub), axis=0)
    mins = np.nanmin(sub[:, ok], axis=0)
    tmin = np.nanargmin(np.where(np.isnan(sub[:, ok]), np.inf, sub[:, ok]), axis=0) + 1
    maxs = np.nanmax(sub[:, ok], axis=0)
    tmax = np.nanargmax(np.where(np.isnan(sub[:, ok]), -np.inf, sub[:, ok]), axis=0) + 1
    row.update(mae_med=float(np.median(mins)), mae_q25=float(np.percentile(mins, 25)),
               trough_d=float(np.median(tmin)), mfe_med=float(np.median(maxs)),
               peak_d=float(np.median(tmax)),
               dip5=float(np.mean(mins < -5) * 100), dip8=float(np.mean(mins < -8) * 100))
    med_path = np.nanmedian(M, axis=1)
    row["hold_d"] = int(np.nanargmax(med_path[1:]) + 1)
    row["hold_med"] = float(np.nanmax(med_path[1:]))
    return row, med_path


def dip_entry(close, dates, x=8, limit_window=LIMIT_WINDOW):
    """Limit -x% under the event close working `limit_window` sessions, exit
    63 sessions after the fill: (fill%, median%, hit%)."""
    fills, n = [], 0
    for d in dates:
        i = close.index.get_loc(d)
        if i + 63 >= len(close):
            continue
        n += 1
        e = float(close.iloc[i])
        work = (close.iloc[i + 1:i + 1 + limit_window] / e - 1).to_numpy()
        hit = np.where(work <= -x / 100)[0]
        if len(hit) and i + hit[0] + 1 + 63 < len(close):
            level = e * (1 - x / 100)
            fills.append(float(close.iloc[i + hit[0] + 1 + 63] / level - 1) * 100)
    if not fills or not n:
        return None
    a = np.asarray(fills)
    return (100 * len(a) / n, float(np.median(a)), float(np.mean(a > 0) * 100))


def class_of(row):
    if row["med63"] >= 2 and row["hit63"] >= 55:
        return "CHASER"
    if row["med63"] <= -2:
        return "FADER"
    return "ROUND-TRIP"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--csv-out", default=None, help="write the tidy stats table here")
    ap.add_argument("--paths-out", default=None,
                    help="write per-ETF median up-break paths (wide CSV) here")
    args = ap.parse_args()

    univ = {**SECTORS, **THEMES}
    syms = sorted(set(univ) | {"SPY"})
    panels = N.load_yahoo_panels(syms, "2004-01-01", pd.Timestamp.today().normalize(),
                                 cache_dir=args.cache_dir or None,
                                 refresh=args.refresh, label="ETFPATH")
    adj = panels["adjclose"].dropna(how="all")
    volp = panels["volume"]
    spy = adj["SPY"].dropna()

    rows, med_paths = [], {}
    for t in sorted(univ):
        if t not in adj:
            continue
        c = adj[t].dropna()
        conv = conviction_frame(c, volp[t])
        ev = conviction_events(c, conv, min_score=3, cooldown=21, horizon=63)
        s21, s63, z, _ = spread_frames(c, spy)
        fams = {
            "up": list(ev[ev["dir"] == 1]["date"]),
            "dn": list(ev[ev["dir"] == -1]["date"]),
            "turn": [d for d, s in detect_breaks(s63, z) if s == -1],
        }
        for fam, dates in fams.items():
            st = family_stats(c, dates)
            if st is None:
                continue
            row, med_path = st
            row.update(etf=t, name=univ[t], family=fam)
            if fam == "up":
                med_paths[t] = med_path
                de = dip_entry(c, dates)
                if de:
                    row["dip8_fill"], row["dip8_med"], row["dip8_hit"] = de
                row["cls"] = class_of(row)
            rows.append(row)

    R = pd.DataFrame(rows).set_index(["etf", "family"])
    cols = ["n", "med21", "med63", "med126", "hit63", "mean63", "skew63",
            "mae_med", "mae_q25", "trough_d", "mfe_med", "peak_d",
            "dip5", "dip8", "hold_d", "hold_med"]

    for fam, label in (("up", "UP-BREAKS (chase)"), ("dn", "DOWN-BREAKS (capitulation)"),
                       ("turn", "TURN SIGNALS (vs SPY)")):
        F = R.xs(fam, level="family").sort_values("med63", ascending=False)
        print(f"\n== {label} ==")
        with pd.option_context("display.width", 200):
            out = F[cols].round(1)
            if fam == "up":
                out = out.join(F[["cls", "dip8_fill", "dip8_med"]].round(1))
            print(out.to_string())

    if args.csv_out:
        R.round(2).to_csv(args.csv_out)
        print(f"\nwrote {args.csv_out}")
    if args.paths_out:
        pd.DataFrame(med_paths).round(2).to_csv(args.paths_out, index=False)
        print(f"wrote {args.paths_out}")


if __name__ == "__main__":
    main()
