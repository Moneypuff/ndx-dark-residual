#!/usr/bin/env python3
"""
DPI-into-earnings vs. post-earnings performance study.
=======================================================

Question (per request): for major NDX names, does an ELEVATED dark-pool
indicator (DPI) heading INTO an earnings report line up with the stock's
performance AFTER the report?

Signal (pre-earnings, no look-ahead)
------------------------------------
Per-name DPI_t = short / total off-exchange volume (FINRA), 0..1 -- the same
per-name "D" construction used across this repo (see ndx_dark_residual.py).
Because most of these companies report *after the close*, the DPI cut-off is
the day BEFORE the report date (T-1):

    DPI5  = mean(DPI over the 5  trading days ending T-1)
    DPI10 = mean(DPI over the 10 trading days ending T-1)

Outcome (post-earnings, split-adjusted closes), timing-aware
------------------------------------------------------------
T = last clean pre-news close. For an after-hours (AMC) report T is the report
day; for a before-open (BMO) report T is the prior session -- so the reaction is
always the first full session on the news.
    next_day_ret = adjclose(T+1) / adjclose(T) - 1      # the earnings reaction
    m1_ret       = adjclose(T+MONTH) / adjclose(T) - 1  # ~1 month later
MONTH defaults to 21 trading sessions.

Dates
-----
Report dates + AMC/BMO timing come from SEC EDGAR 8-K Item 2.02 filings (see
fetch_earnings_edgar.py -> earnings_dates_edgar.csv), matched to each 10-Q/10-K
to isolate the quarterly earnings release. A hand-curated earnings_dates.csv (20
mega-caps) is kept as a fallback example. An optional --anchor mode can snap T to
the nearest price reaction; it is off by default (biased toward large moves) and
unnecessary given authoritative dates.

Usage
-----
    python fetch_earnings_edgar.py --out earnings_dates_edgar.csv
    python earnings_dpi_study.py --earnings earnings_dates_edgar.csv \
        --cache-dir ~/.ndx_dark_cache --out-prefix earnings_dpi

Outputs <out-prefix>_events.csv (one row per event) and prints a summary.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N


# ----------------------------------------------------------------------------
# Event construction
# ----------------------------------------------------------------------------
def load_earnings(path):
    df = pd.read_csv(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["timing"] = df.get("timing", "amc").fillna("amc").str.lower()
    return df.sort_values(["ticker", "report_date"]).reset_index(drop=True)


def _pos_at_or_after(index, ts):
    """Index position of the first trading day >= ts (or None)."""
    pos = index.searchsorted(pd.Timestamp(ts), side="left")
    return int(pos) if pos < len(index) else None


def anchor_to_reaction(ret, close_index, approx_pos, sigma,
                       search=(-2, 4), min_abs=0.03, min_z=2.5, dominance=1.3):
    """Return (T_pos, anchored_bool). Anchor T so T+1 is the dominant nearby move.

    `approx_pos` is the index position of the curated report date (a trading day
    at/after the curated calendar date). We look for the earnings *reaction* --
    the largest |return| day -- in reaction slots [approx_pos+search[0]+1,
    approx_pos+search[1]+1]. If one move clearly dominates (exceeds an absolute
    and a volatility-relative floor, and is `dominance`x the next-largest in the
    window) we set T = reaction-1. Otherwise fall back to the curated position.
    """
    lo = max(1, approx_pos + search[0] + 1)
    hi = min(len(ret) - 1, approx_pos + search[1] + 1)
    if hi <= lo:
        return approx_pos, False
    slots = list(range(lo, hi + 1))
    mags = np.array([abs(ret.iloc[s]) for s in slots])
    order = np.argsort(mags)[::-1]
    top = slots[order[0]]
    top_mag = mags[order[0]]
    second = mags[order[1]] if len(order) > 1 else 0.0
    floor = max(min_abs, min_z * (sigma if sigma and np.isfinite(sigma) else np.inf))
    if top_mag >= floor and top_mag >= dominance * max(second, 1e-9):
        return top - 1, True
    return approx_pos, False


def build_events(earnings, panels, month_sessions=21, dpi_windows=(5, 10), anchor=False):
    """Build one row per earnings event, timing-aware.

    A = the announce session (first trading day on/after the report date). The
    base close T (the last clean pre-news close) depends on when the news hit:

        amc  (after close of A) : T = A       -> reaction is the A -> A+1 gap
        bmo  (before open of A) : T = A - 1   -> reaction is the (A-1) -> A move
        intraday / non-session  : T = A - 1

    Either way next-day = close(T+1)/close(T) is the first full session's
    reaction, and the pre-earnings DPI window always ends on the last session
    before A (the day before the report), matching the requested cut-off.

    `anchor=True` snaps T to the dominant nearby move -- diagnostic only; it
    biases the sample toward large (disproportionately negative) moves, off by
    default. With authoritative EDGAR dates it is unnecessary.
    """
    dpi = panels["dpi"]
    adj = panels["adjclose"]
    ret = adj.pct_change()
    idx = adj.index
    rows = []
    for _, e in earnings.iterrows():
        tk = e["ticker"]
        if tk not in adj.columns:
            continue
        A = e["report_date"]
        fp = _pos_at_or_after(idx, A)          # announce-session position
        if fp is None or fp < 1:
            continue
        timing = str(e.get("timing", "amc")).lower()
        exact = idx[fp].normalize() == A.normalize()
        t_pos = fp if (timing == "amc" and exact) else fp - 1
        if t_pos < 1:
            continue
        r = ret[tk]; a = adj[tk]; d = dpi[tk]
        T = idx[t_pos]
        # trailing vol (ending before the DPI window) for the audit flag only
        pre = r.iloc[max(0, fp - 65):max(1, fp - 3)]
        sigma = float(pre.std()) if pre.notna().sum() > 10 else np.nan
        if anchor:
            t_pos, anchored = anchor_to_reaction(r, idx, t_pos, sigma)
            T = idx[t_pos]
        else:
            anchored = False
        nxt_ret = r.iloc[t_pos + 1] if t_pos + 1 < len(r) else np.nan
        looks_reaction = int(np.isfinite(nxt_ret) and np.isfinite(sigma)
                             and abs(nxt_ret) >= max(0.02, 2.0 * sigma))

        # --- pre-earnings DPI: w sessions ending the day before A ---
        win = {}
        for w in dpi_windows:
            seg = d.iloc[max(0, fp - w):fp]
            n_ok = int(seg.notna().sum())
            win[f"dpi{w}"] = float(seg.mean()) if n_ok >= max(3, w - 2) else np.nan

        # --- post-earnings returns (split-adjusted) ---
        base = a.iloc[t_pos]
        nd = a.iloc[t_pos + 1] if t_pos + 1 < len(a) else np.nan
        mm = a.iloc[t_pos + month_sessions] if t_pos + month_sessions < len(a) else np.nan
        next_day = (nd / base - 1) if np.isfinite(base) and np.isfinite(nd) else np.nan
        m1 = (mm / base - 1) if np.isfinite(base) and np.isfinite(mm) else np.nan

        rows.append({
            "ticker": tk,
            "report_date": A.date().isoformat(),
            "timing": timing,
            "base_T": T.date().isoformat(),
            "anchored": int(anchored),
            **win,
            "next_day_ret": next_day,
            "m1_ret": m1,
            "looks_reaction": looks_reaction,
            "has_data": int(np.isfinite(next_day)),
        })
    ev = pd.DataFrame(rows)
    # within-name DPI percentile ranks (0..1): "is this event's run-in DPI high
    # *for this name*?" -- removes cross-sectional level differences between names
    for w in dpi_windows:
        ev[f"dpi{w}_pct"] = ev.groupby("ticker")[f"dpi{w}"].rank(pct=True)
    return ev


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def _pearson(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    x, y = x[m].to_numpy(), y[m].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    # two-sided p via t-approximation
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * np.sqrt((n - 2) / (1 - r * r))
        p = _t_sf(abs(t), n - 2) * 2
    return r, p, n


def _spearman(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    xr = x[m].rank()
    yr = y[m].rank()
    return _pearson(xr, yr)


def _t_sf(t, df):
    """Survival function of Student-t via regularized incomplete beta (no scipy)."""
    if df <= 0:
        return np.nan
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b) via continued fraction (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = _gammaln(a) + _gammaln(b) - _gammaln(a + b)
    bt = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _gammaln(x):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + np.log(2.5066282746310005 * ser / x)


def _welch(a, b):
    """Welch two-sample t-test; returns (mean_a-mean_b, t, p, na, nb)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return np.nan, np.nan, np.nan, na, nb
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return a.mean() - b.mean(), np.nan, np.nan, na, nb
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = _t_sf(abs(t), df) * 2
    return a.mean() - b.mean(), t, p, na, nb


def summarize(ev, dpi_windows=(5, 10)):
    out = []
    out.append("=" * 78)
    out.append("DPI-INTO-EARNINGS  vs  POST-EARNINGS PERFORMANCE")
    out.append("=" * 78)
    n_names = ev["ticker"].nunique()
    out.append(f"Events: {len(ev)}   Names: {n_names}   "
               f"T+1 looks like a real earnings reaction: "
               f"{int(ev['looks_reaction'].sum())}/{len(ev)} "
               f"({100*ev['looks_reaction'].mean():.0f}%)")
    for col, lbl in [("next_day_ret", "NEXT-DAY  (close T -> close T+1)"),
                     ("m1_ret", "1-MONTH   (close T -> close T+~21 sessions)")]:
        s = ev[col].dropna()
        out.append("")
        out.append(f"--- {lbl} ---   n={len(s)}")
        out.append(f"    mean {s.mean()*100:+.2f}%   median {s.median()*100:+.2f}%   "
                   f"std {s.std()*100:.2f}%   %positive {100*(s>0).mean():.0f}%")
        for w in dpi_windows:
            sig = ev[f"dpi{w}"]
            pr, pp, pn = _pearson(sig, ev[col])
            sr, sp, sn = _spearman(sig, ev[col])
            prp, ppp, _ = _pearson(ev[f"dpi{w}_pct"], ev[col])
            out.append(f"    DPI{w:<2}  Pearson r={pr:+.3f} (p={pp:.3f}, n={pn})   "
                       f"Spearman r={sr:+.3f} (p={sp:.3f})   "
                       f"within-name r={prp:+.3f} (p={ppp:.3f})")

    # bucket analysis on within-name DPI percentile (top vs bottom tercile)
    out.append("")
    out.append("--- TERCILE BUCKETS on within-name DPI percentile ---")
    for w in dpi_windows:
        p = ev[f"dpi{w}_pct"]
        hi = ev[p >= 2 / 3]
        lo = ev[p <= 1 / 3]
        out.append(f"  DPI{w}:  low-DPI n={len(lo)}   high-DPI n={len(hi)}")
        for col, lbl in [("next_day_ret", "next-day"), ("m1_ret", "1-month")]:
            hh = hi[col].to_numpy(); ll = lo[col].to_numpy()
            diff, t, pv, nh, nl = _welch(hh, ll)
            hm = np.nanmean(hh) if np.isfinite(hh).any() else np.nan
            lm = np.nanmean(ll) if np.isfinite(ll).any() else np.nan
            hpos = 100 * np.nanmean((hh > 0)) if np.isfinite(hh).any() else np.nan
            lpos = 100 * np.nanmean((ll > 0)) if np.isfinite(ll).any() else np.nan
            out.append(f"     {lbl:8s}: high {hm*100:+.2f}% ({hpos:.0f}% up)   "
                       f"low {lm*100:+.2f}% ({lpos:.0f}% up)   "
                       f"high-low {diff*100:+.2f}pp (t={t:+.2f}, p={pv:.3f})")

    # robustness: does DPI10 vs next-day hold across timing and sub-periods?
    def _cut(mask, name):
        d = ev[mask]
        r, p, n = _pearson(d["dpi10"], d["next_day_ret"])
        out.append(f"    {name:20s} r={r:+.3f} (p={p:.3f}, n={n})")
    out.append("")
    out.append("--- ROBUSTNESS: DPI10 vs next-day across cuts ---")
    if "timing" in ev.columns:
        for tm in ["amc", "bmo"]:
            _cut(ev["timing"] == tm, f"timing = {tm}")
    if "report_date" in ev.columns:
        yr = pd.to_datetime(ev["report_date"]).dt.year
        mid = int(yr.median())
        _cut(yr <= mid, f"reports <= {mid}")
        _cut(yr > mid, f"reports >  {mid}")
    out.append("=" * 78)
    return "\n".join(out)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--earnings", default="earnings_dates_edgar.csv")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-prefix", default="earnings_dpi")
    ap.add_argument("--month-sessions", type=int, default=21)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--refresh", action="store_true", default=False)
    ap.add_argument("--anchor", action="store_true", default=False,
                    help="snap dates to nearest price reaction (diagnostic; biased -- off by default)")
    args = ap.parse_args()

    earn = load_earnings(args.earnings)
    syms = sorted(earn["ticker"].unique())
    pad = pd.Timedelta(days=25)
    start = (earn["report_date"].min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (earn["report_date"].max() + pad + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    print(f"Universe: {len(syms)} names   window {start} -> {end}", file=sys.stderr)

    panels = N.build_universe_panels(syms, start, end, workers=args.workers,
                                     cache_dir=args.cache_dir or None, ns="earn",
                                     refresh=args.refresh, label="EARN")

    ev = build_events(earn, panels, month_sessions=args.month_sessions, anchor=args.anchor)
    out_csv = f"{args.out_prefix}_events.csv"
    ev.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(ev)} events)", file=sys.stderr)

    print(summarize(ev))
    return ev, panels


if __name__ == "__main__":
    main()
