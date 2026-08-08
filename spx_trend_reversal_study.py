#!/usr/bin/env python3
"""
Does per-name DPI lead single-stock TREND REVERSALS?
==============================================================================

Request: in the rotation-heavy tape (indices flat, stocks moving against each
other), is there a DPI signal for WHEN a stock's trend reverses? The earlier
studies conditioned on DPI and looked forward (null). This one flips the lens:

1. EVENT STUDY around EX-POST turning points. A TROUGH is a local price minimum
   over +/-21 sessions reached while the 3-month trend was negative (a PEAK is
   the mirror image in an uptrend), de-overlapped 42 sessions per name. Stack
   the self-relative 10-day-average-DPI percentile (p10) in event time
   tau = -42..+21 around the turns. If dark accumulation front-runs reversals,
   p10 must climb into troughs (fall into peaks) relative to its regime
   baseline. Turning points use FUTURE data by construction -- this is a
   diagnostic of whether the information exists at all, not a signal.
2. The TRADEABLE version: among downtrend name-days, does a high p10 (or a 10+
   high-DPI streak) raise P(trough within the next 21 sessions) vs the
   downtrend base rate? Daily cross-sectional HIGH-LOW differences with
   block-bootstrap CIs, plus a Fama-MacBeth linear-probability control for the
   recent-return characteristic, full / pre-2023 / 2023+ (the requested
   "until ~2 years ago" split).

Inference: event curves use a cluster bootstrap BY NAME (events of one name
share flow habits); labels near the sample edge are NaN, never "no reversal".

Reproduce
---------
    python spx_trend_reversal_study.py --start 2019-01-01 --out spx_trend_reversal.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import spx_xs_dip_dix_study as XS
import spx_dpi_persistence_study as P   # load_full, build_long, fama_macbeth

WING = 21          # +/- window defining a local extremum
MIN_GAP = 42       # de-overlap between same-name events
TAUS = np.arange(-42, 22)
SHOW_TAUS = (-42, -31, -21, -10, -5, -1, 0, 5, 10, 21)
HZ = 21            # "reversal within the next HZ sessions"
SPLIT_DATE = "2023-01-01"
HI, LO = XS.HI, XS.LO
BOOT_B = 1000


def find_turns(adj, trend, kind="trough", wing=WING, min_gap=MIN_GAP):
    """(name, position) list of de-overlapped local extrema with the required
    prior trend. Full +/-wing windows only (edges yield no events)."""
    w = 2 * wing + 1
    if kind == "trough":
        ext = adj.rolling(w, center=True, min_periods=w).min()
        hit = (adj == ext) & (trend < 0)
    else:
        ext = adj.rolling(w, center=True, min_periods=w).max()
        hit = (adj == ext) & (trend > 0)
    events = []
    H = hit.fillna(False).to_numpy()
    for j, name in enumerate(adj.columns):
        last = -10 ** 9
        for i in np.flatnonzero(H[:, j]):
            if i - last >= min_gap:
                events.append((name, int(i)))
                last = i
    return events


def event_curves(sig, events, taus=TAUS):
    """events x taus matrix of `sig` values in event time."""
    A = sig.to_numpy(dtype=float)
    cols = {c: k for k, c in enumerate(sig.columns)}
    T = len(sig)
    out = np.full((len(events), len(taus)), np.nan)
    for e, (name, i) in enumerate(events):
        j = cols[name]
        for k, tau in enumerate(taus):
            t = i + tau
            if 0 <= t < T:
                out[e, k] = A[t, j]
    return out


def curve_boot(curves, names, B=BOOT_B, seed=0):
    """Mean event-time curve with a cluster bootstrap BY NAME: resample names
    with replacement, average their events' curves. Returns (mean, lo, hi)."""
    names = np.asarray(names)
    uniq = np.unique(names)
    groups = {u: np.flatnonzero(names == u) for u in uniq}
    mean = np.nanmean(curves, axis=0)
    rng = np.random.default_rng(seed)
    boots = np.full((B, curves.shape[1]), np.nan)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([groups[u] for u in pick])
        boots[b] = np.nanmean(curves[rows], axis=0)
    return mean, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def turn_within(turn_positions, index, columns, h=HZ, wing=WING):
    """0/1 panel: does a turn occur within t+1..t+h? NaN over the trailing
    h+wing sessions, where the label cannot be known (a turn needs +wing future
    sessions to even be defined) -- NaN, never counted as 'no reversal'."""
    M = pd.DataFrame(0.0, index=index, columns=columns)
    for name, i in turn_positions:
        M.iloc[i, M.columns.get_loc(name)] = 1.0
    fwd = M.iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1].shift(-1)
    fwd.iloc[-(h + wing):] = np.nan
    return fwd


def _daily_diff(long, mA, mB, col, min_n=5):
    a = long[mA].groupby(level="date")
    b = {d: g for d, g in long[mB].groupby(level="date")}
    out = []
    for d, ga in a:
        if d not in b:
            continue
        va, vb = ga[col].dropna(), b[d][col].dropna()
        if len(va) >= min_n and len(vb) >= min_n:
            out.append(float(va.mean() - vb.mean()))
    return np.asarray(out, dtype=float)


def _stat(a, seed=0):
    a = a[np.isfinite(a)]
    if len(a) < 30:
        return {"days": int(len(a)), "mean": np.nan, "ci": None}
    ci = XS.block_boot_ci(a, seed=seed)
    return {"days": int(len(a)), "mean": round(float(a.mean()), 4),
            "ci": (round(ci[0], 4), round(ci[1], 4)) if np.isfinite(ci[0]) else None}


def prob_tests(long, split=SPLIT_DATE):
    """Among downtrend days: P(trough within HZ) by p10 group / streak, the
    HIGH-LOW daily difference, and the FM control tw ~ rback + hi-dummy."""
    down = long["trend63"] < 0
    d = long[down].copy()
    base = float(d["tw"].mean() * 100)
    grp = {
        "LOW p10": d.loc[d["p10"] <= LO, "tw"],
        "HIGH p10": d.loc[d["p10"] >= HI, "tw"],
        "no streak": d.loc[d["streak"] == 0, "tw"],
        "streak 10+": d.loc[d["streak"] >= 10, "tw"],
    }
    rates = {k: (round(float(v.mean() * 100), 2), int(v.notna().sum())) for k, v in grp.items()}
    hl = _stat(_daily_diff(long, down & (long["p10"] >= HI), down & (long["p10"] <= LO), "tw"),
               seed=1)
    st = _stat(_daily_diff(long, down & (long["streak"] >= 10), down & (long["streak"] == 0),
                           "tw"), seed=2)
    fm = {}
    dd = long.copy()
    dd["hi10"] = (dd["p10"] >= HI).astype(float)
    for lab, sub in [("full", dd), ("IS", P._slice(dd, hi=split)), ("OOS", P._slice(dd, lo=split))]:
        f = P.fama_macbeth(sub, "tw", ["rback", "hi10"], sub["trend63"] < 0,
                           seed=hash(lab) % 997)
        fm[lab] = f["hi10"]
    return base, rates, hl, st, fm


def _fmt_curve(label, taus, mean, lo, hi, base):
    keep = [k for k, t in enumerate(taus) if t in SHOW_TAUS]
    head = "   tau:    " + "".join(f"{int(taus[k]):>8}" for k in keep)
    m = f"   {label:7s} " + "".join(f"{mean[k]:>8.3f}" for k in keep)
    c = "   95% CI  " + "".join(f"{'*' if (lo[k] > base or hi[k] < base) else ' ':>8}" for k in keep)
    return "\n".join([head, m, c + f"   (* = CI excludes the regime baseline {base:.3f})"])


def report(SP, long, split=SPLIT_DATE):
    out = []
    adj = SP["adjclose"]
    p10 = long["p10"].unstack()
    tr63 = long["trend63"].unstack()
    adj = adj.reindex(index=p10.index, columns=p10.columns)
    for kind, tlab in (("trough", "TROUGH (downtrend -> local low)"),
                       ("peak", "PEAK (uptrend -> local high)")):
        ev = find_turns(adj, tr63, kind=kind)
        names = [n for n, _ in ev]
        cur = event_curves(p10, ev)
        base_mask = (tr63 < 0) if kind == "trough" else (tr63 > 0)
        base = float(p10[base_mask].stack().mean())
        mean, lo, hi = curve_boot(cur, names, seed=7)
        out.append(f"\n== EVENT STUDY: p10 around {tlab}  ({len(ev)} events, "
                   f"{len(set(names))} names; cluster bootstrap by name) ==")
        out.append(_fmt_curve("p10", TAUS, mean, lo, hi, base))
        out.append(f"   drift into the turn: mean p10 at tau=-42 {mean[0]:.3f} -> tau=0 "
                   f"{mean[TAUS.tolist().index(0)]:.3f}  (regime baseline {base:.3f})")
    out.append(f"\n== TRADEABLE VERSION: P(trough within {HZ} sessions) among downtrend days ==")
    base, rates, hl, st, fm = prob_tests(long, split=split)
    out.append(f"   downtrend base rate: {base:.2f}%")
    for k, (r, n) in rates.items():
        out.append(f"   {k:11s}: {r:5.2f}%  (n={n:,})")
    out.append(f"   HIGH-LOW p10 daily diff: {hl['mean'] * 100 if hl['mean'] is not np.nan else float('nan'):+.2f}pp  "
               f"CI [{hl['ci'][0] * 100:+.2f},{hl['ci'][1] * 100:+.2f}]" if hl["ci"] else
               f"   HIGH-LOW p10 daily diff: n/a")
    out.append(f"   streak10-vs-none diff:   {st['mean'] * 100:+.2f}pp  "
               f"CI [{st['ci'][0] * 100:+.2f},{st['ci'][1] * 100:+.2f}]" if st["ci"] else
               "   streak10-vs-none diff: n/a")
    out.append("   FM control  tw ~ rback + HIGH-dummy  (HIGH coef, pp):")
    for lab, s in fm.items():
        ci = f"[{s['ci'][0] * 100:+.2f},{s['ci'][1] * 100:+.2f}]" if s["ci"] else "--"
        out.append(f"      {lab:5s} {s['mean'] * 100:+.3f}pp {ci}  ({s['days']} days)")
    return "\n".join(out), {"base": base, "rates": rates, "hl": hl, "st": st, "fm": fm}


def summary_rows(SP, long, res):
    rows = []
    for k, (r, n) in res["rates"].items():
        rows.append({"test": "p_trough_within", "group": k, "n": n, "value": r,
                     "ci_lo": None, "ci_hi": None})
    for lab, s in (("HIGH-LOW", res["hl"]), ("streak10-none", res["st"])):
        rows.append({"test": "daily_diff_pp", "group": lab, "n": s["days"],
                     "value": None if s["mean"] is np.nan else round(s["mean"] * 100, 3),
                     "ci_lo": round(s["ci"][0] * 100, 3) if s["ci"] else None,
                     "ci_hi": round(s["ci"][1] * 100, 3) if s["ci"] else None})
    for lab, s in res["fm"].items():
        rows.append({"test": "fm_hi10_pp", "group": lab, "n": s["days"],
                     "value": round(s["mean"] * 100, 3) if np.isfinite(s["mean"]) else None,
                     "ci_lo": round(s["ci"][0] * 100, 3) if s["ci"] else None,
                     "ci_hi": round(s["ci"][1] * 100, 3) if s["ci"] else None})
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
    long = P.build_long(SP, spy, HZ)
    # forward reversal label, aligned to the long frame
    adj = SP["adjclose"]
    p10w = long["p10"].unstack()
    adj = adj.reindex(index=p10w.index, columns=p10w.columns)
    tr63 = long["trend63"].unstack()
    troughs = find_turns(adj, tr63, kind="trough")
    tw = turn_within(troughs, adj.index, adj.columns)
    long["tw"] = tw.stack().reindex(long.index)
    print(f"\nPanel: {len(long):,} obs. Troughs: {len(troughs):,}; "
          f"peaks: {len(find_turns(adj, tr63, kind='peak')):,}")
    text, res = report(SP, long, split=args.split_date)
    print(text)
    if args.out:
        summary_rows(SP, long, res).to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
