#!/usr/bin/env python3
"""
GDX post-signal path study -- what actually happens after you "chase" a GDX
break, and what the 5-for-5 turn-signal record is made of.

Companion to the regime log's conviction column (GEX_DISPERSION_GUIDE.md §7):
the leaderboard says GDX punishes chasers (-0.5% median, 48%) while its
relative-strength turn signals are 5-for-5. This study maps the forward price
paths behind both stats and tests entry rules against each other, using the
same detectors the page runs (conviction_frame / conviction_events /
detect_breaks out of build_regime_log).

Three event families, all on GDX:

  * up-breaks   -- high-conviction (score >= 3) breaks in an up 21d move: the
                   "chase" entry the leaderboard warns about.
  * down-breaks -- the same score in a down move: capitulation.
  * turns       -- detect_breaks on the GDX-vs-SPY 63d spread, sgn=-1: an
                   entrenched underperformance regime cracking upward (the
                   family behind the 5-for-5).

For each family it prints the median forward path (5/10/21/42/63/126
sessions), the adverse/favorable excursion inside the first quarter, splits of
the chase outcome by era / GLD-trend / score / extension, and a comparison of
entry rules (chase at the close, resting limits below, wait-21d, stops).

    python gdx_chase_study.py --cache-dir .ndx_dark_cache
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
from build_regime_log import (conviction_frame, conviction_events,
                              detect_breaks, spread_frames)

H = 126            # path horizon, sessions
LIMIT_WINDOW = 42  # sessions a resting limit order works after an event
CHECKPOINTS = (5, 10, 21, 42, 63, 126)


# ----------------------------------------------------------------------------
# Pure computation
# ----------------------------------------------------------------------------
def path_matrix(close, dates, horizon=H):
    """(horizon+1) x n matrix of forward %-from-event-close paths."""
    cols = {}
    for d in dates:
        i = close.index.get_loc(d)
        seg = close.iloc[i:i + horizon + 1]
        p = (seg.to_numpy() / seg.iloc[0] - 1) * 100
        cols[pd.Timestamp(d)] = np.concatenate([p, np.full(horizon + 1 - len(p), np.nan)])
    return pd.DataFrame(cols)


def path_stats(P):
    """Median/hit at the checkpoints plus first-quarter excursion stats."""
    M = P.to_numpy()
    out = {"n": M.shape[1],
           "med": {k: float(np.nanmedian(M[k])) for k in CHECKPOINTS},
           "pos": {k: float(np.nanmean(M[k] > 0) * 100) for k in CHECKPOINTS}}
    sub = M[1:64]
    ok = ~np.all(np.isnan(sub), axis=0)
    mins = np.nanmin(sub[:, ok], axis=0)
    tmin = np.nanargmin(np.where(np.isnan(sub[:, ok]), np.inf, sub[:, ok]), axis=0) + 1
    maxs = np.nanmax(sub[:, ok], axis=0)
    tmax = np.nanargmax(np.where(np.isnan(sub[:, ok]), -np.inf, sub[:, ok]), axis=0) + 1
    out["mae"] = {"med": float(np.median(mins)),
                  "q25": float(np.percentile(mins, 25)),
                  "q75": float(np.percentile(mins, 75)),
                  "tmed": float(np.median(tmin)),
                  "dip5": float(np.mean(mins < -5) * 100),
                  "dip8": float(np.mean(mins < -8) * 100)}
    out["mfe"] = {"med": float(np.median(maxs)), "tmed": float(np.median(tmax))}
    return out


def entry_rules(close, dates, limits=(4, 6, 8), limit_window=LIMIT_WINDOW):
    """Compare entries after up-break events. Returns rows of
    (rule, fill%, median%, hit%): chase at the event close (exit day 63),
    resting limit -x% filled within `limit_window` sessions (exit 63 sessions
    after the FILL), wait-21-sessions-then-buy (exit day 63), and chase with a
    -12% stop."""
    rows = []
    chase, wait21, stopped = [], [], []
    lim = {x: [] for x in limits}
    for d in dates:
        i = close.index.get_loc(d)
        if i + 63 >= len(close):
            continue
        e = float(close.iloc[i])
        p63 = float(close.iloc[i + 63] / e - 1) * 100
        chase.append(p63)
        if i + 21 < len(close):
            wait21.append(float(close.iloc[i + 63] / close.iloc[i + 21] - 1) * 100)
        seg = (close.iloc[i + 1:i + 64] / e - 1).to_numpy()
        breach = np.where(seg <= -0.12)[0]
        stopped.append(-12.0 if len(breach) else p63)
        work = (close.iloc[i + 1:i + 1 + limit_window] / e - 1).to_numpy()
        for x in limits:
            hit = np.where(work <= -x / 100)[0]
            if len(hit) and i + hit[0] + 1 + 63 < len(close):
                level = e * (1 - x / 100)
                lim[x].append(float(close.iloc[i + hit[0] + 1 + 63] / level - 1) * 100)

    def row(name, vals, fill=100.0):
        a = np.asarray(vals, float)
        return (name, fill, float(np.median(a)), float(np.mean(a > 0) * 100))

    rows.append(row("chase day-0 close, exit d63", chase))
    for x in limits:
        rows.append(row(f"limit -{x}% (d1-{limit_window}), exit fill+63",
                        lim[x], 100 * len(lim[x]) / max(len(chase), 1)))
    rows.append(row("wait 21 sessions, buy close, exit d63", wait21))
    rows.append(row("chase with -12% stop", stopped))
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    panels = N.load_yahoo_panels(["GDX", "GLD", "SPY"], "2004-01-01",
                                 pd.Timestamp.today().normalize(),
                                 cache_dir=args.cache_dir or None,
                                 refresh=args.refresh, label="GDXCHASE")
    adj = panels["adjclose"].dropna(how="all")
    gdx, gld, spy = adj["GDX"].dropna(), adj["GLD"].dropna(), adj["SPY"].dropna()

    conv = conviction_frame(gdx, panels["volume"]["GDX"])
    ev = conviction_events(gdx, conv, min_score=3, cooldown=21, horizon=63)
    up = ev[ev["dir"] == 1]["date"]
    dn = ev[ev["dir"] == -1]["date"]
    s21, s63, z, _ = spread_frames(gdx, spy)
    turns = [d for d, s in detect_breaks(s63, z) if s == -1]

    print(f"data through {adj.index.max().date()}")
    for name, dates in (("UP-BREAKS (chase)", up), ("DOWN-BREAKS (capitulation)", dn),
                        ("TURN SIGNALS (vs SPY)", turns)):
        st = path_stats(path_matrix(gdx, dates))
        print(f"\n== {name}: n={st['n']} ==")
        print("day:    " + "  ".join(f"{k:>6}" for k in CHECKPOINTS))
        print("median: " + "  ".join(f"{st['med'][k]:>+6.1f}" for k in CHECKPOINTS))
        print("%>0:    " + "  ".join(f"{st['pos'][k]:>6.0f}" for k in CHECKPOINTS))
        m = st["mae"]
        print(f"max drawdown (d1-63): med {m['med']:+.1f}% [q25 {m['q25']:+.1f}, q75 {m['q75']:+.1f}]"
              f"  trough ~d{m['tmed']:.0f};  dips >5%: {m['dip5']:.0f}%, >8%: {m['dip8']:.0f}%")
        print(f"max favorable (d1-63): med {st['mfe']['med']:+.1f}%  peak ~d{st['mfe']['tmed']:.0f}")

    # conditional splits of the chase outcome
    U = ev[ev["dir"] == 1].dropna(subset=["fwd"]).copy()
    g200 = gld.rolling(200).mean()
    U["gld_above"] = [bool(gld.asof(d) > g200.asof(d)) for d in U["date"]]
    U["ext21"] = [float(gdx.loc[:d].iloc[-1] / gdx.loc[:d].iloc[-22] - 1) * 100
                  for d in U["date"]]

    def stat(s, label):
        a = np.asarray(s, float)
        if not len(a):
            print(f"  {label:<36} n=0")
            return
        print(f"  {label:<36} n={len(a):<4} med {np.median(a):+5.1f}%  "
              f"hit {np.mean(a > 0) * 100:3.0f}%  mean {np.mean(a):+5.1f}%")

    print("\n== chase (day0 -> d63) conditioned ==")
    stat(U["fwd"], "all")
    stat(U[U["date"] < "2016-01-01"]["fwd"], "2006-2015 era")
    stat(U[U["date"] >= "2016-01-01"]["fwd"], "2016- era")
    stat(U[U["gld_above"]]["fwd"], "GLD above its 200d")
    stat(U[~U["gld_above"]]["fwd"], "GLD below its 200d")
    stat(U[U["score"] == 4]["fwd"], "score 4/4")
    stat(U[U["ext21"] >= 15]["fwd"], "already +15%/21d at event")

    print("\n== entry rules on up-breaks ==")
    print(f"  {'rule':<38}{'fill%':>7}{'median%':>9}{'hit%':>7}")
    for name, fill, medr, hit in entry_rules(gdx, up):
        print(f"  {name:<38}{fill:>7.0f}{medr:>9.1f}{hit:>7.0f}")

    cur = conv.iloc[-1]
    print(f"\ntoday: conviction {int(cur['score'])}/4 dir {int(cur['dir']):+d}  "
          f"last up-break {up.iloc[-1].date() if len(up) else None}")


if __name__ == "__main__":
    main()
