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

Second question: what happens to DPI AFTER the report
-----------------------------------------------------
The same DPI windows are also measured on the far side of the news --

    dpi_post_w1/w2/m1 = mean DPI over T+1..T+5 / T+10 / T+21
    d_dpi_w1/w2/m1    = dpi_post_* - DPI10        (the change, in ratio points)
    dpi_base60        = mean DPI over the 60 sessions ending the day before the
                        report, i.e. the name's ordinary level; adpi_* measure
                        the pre- and post-windows against it

-- plus non-overlapping slices (week 1 / week 2 / weeks 3-4) to show how the
change decays, and returns measured FROM each horizon (fwd_w1_m1 = T+5 -> T+21,
fwd_w2_m1, fwd_m1_m2) so a DPI change can be tested against what happens after
it is observable, with no look-ahead. See summarize_dpi_change().

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


# Dual-class duplicates in the index: fold the secondary share class into the
# primary so a company isn't counted twice. DPI is re-derived volume-weighted
# from the SUMMED off-exchange short/total across classes (not an average of the
# two ratios); the primary's prices/returns are kept. GOOG/GOOGL is the NDX-100's
# only such pair -- both are Alphabet (same CIK -> same report dates).
SHARE_CLASS_MERGES = {"GOOGL": ["GOOG"]}   # primary <- [secondaries]


def merge_share_classes(panels, earnings, merges=SHARE_CLASS_MERGES):
    """Collapse dual-class tickers in-place. Returns (panels, earnings) with the
    secondary classes removed and the primary's DPI recomputed volume-weighted."""
    short, total = panels["short"], panels["total"]
    drop = []
    for primary, secs in merges.items():
        secs = [s for s in secs if s in total.columns]
        if primary not in total.columns or not secs:
            continue
        for s in secs:
            short[primary] = short[primary].add(short[s], fill_value=0)
            total[primary] = total[primary].add(total[s], fill_value=0)
            drop.append(s)
        dpi, d = N.finra_dpi_to_d(short[[primary]], total[[primary]])
        panels["dpi"][primary] = dpi[primary]
        panels["d"][primary] = d[primary]
    if drop:
        for k in ("short", "total", "dpi", "d", "close", "adjclose", "volume"):
            if k in panels:
                panels[k] = panels[k].drop(columns=[c for c in drop if c in panels[k].columns])
        earnings = earnings[~earnings["ticker"].isin(drop)].reset_index(drop=True)
    return panels, earnings


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


# post-earnings horizons, in trading sessions after the base close T
HORIZONS = {"next_day": 1, "w1": 5, "w2": 10, "m1": 21}
HZ_LABEL = {"next_day": "NEXT-DAY (T->T+1)", "w1": "1-WEEK (T->T+5)",
            "w2": "2-WEEK (T->T+10)", "m1": "1-MONTH (T->T+21)"}

# Sessions of DPI history used for the "normal level for this name" baseline that
# the pre- and post-earnings windows are measured against (ends the day before the
# report, so it is pre-announcement like the DPI5/DPI10 signal).
BASELINE_SESSIONS = 60

# Forward returns measured FROM a post-earnings horizon (not from T). These are the
# outcomes a DPI *change* could plausibly trade: the change over T+1..T+5 is only
# known at T+5, so the honest test is what the stock does from T+5 onward.
FWD_RETURNS = {"w1_m1": ("w1", "m1"), "w2_m1": ("w2", "m1"), "m1_m2": ("m1", None)}


def build_events(earnings, panels, horizons=HORIZONS, dpi_windows=(5, 10), anchor=False):
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
    adj = panels["adjclose"]
    # Align the DPI panel to the price calendar. FINRA data only starts at
    # FINRA_MIN_DATE (2018-08), so when the price history reaches further back (it
    # starts ~40 days before the earliest earnings date, which for the full universe
    # predates FINRA), panels["dpi"] has fewer, later-starting rows than adjclose.
    # `fp` below is a POSITION in adj.index, but the pre-fix code sliced dpi with that
    # same position (`d.iloc[fp-w:fp]`) despite dpi's shorter index -- a constant
    # offset (~160 sessions) that read the wrong window for every event and ran off
    # the end for the most recent ones (2026 events came out all-NaN). Reindexing onto
    # adj.index makes every positional slice consistent; pre-FINRA dates become NaN
    # (correctly, there is no dark-pool data there) and are simply skipped by n_ok.
    dpi = panels["dpi"].reindex(adj.index)
    ret = adj.pct_change()
    lret = np.log(adj).diff()          # daily log returns, for realized vol
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

        # --- the name's ordinary DPI level going into the event (baseline) ---
        bseg = d.iloc[max(0, fp - BASELINE_SESSIONS):fp]
        dpi_base = (float(bseg.mean())
                    if int(bseg.notna().sum()) >= BASELINE_SESSIONS // 2 else np.nan)
        # An EARLIER window that does not overlap DPI10 (sessions -60..-11). Because
        # `d_dpi = post - DPI10` contains DPI10's own measurement noise with a minus
        # sign, corr(DPI10, d_dpi) is mechanically negative even with no true mean
        # reversion. `dpi_prior` shares no observations with either side, so ranking
        # events by it gives an unbiased read on how much reversion is real.
        pseg = d.iloc[max(0, fp - BASELINE_SESSIONS):max(0, fp - dpi_windows[-1])]
        dpi_prior = float(pseg.mean()) if int(pseg.notna().sum()) >= 20 else np.nan

        # --- post-earnings DPI: same windows, now AFTER the news ---
        # Cumulative windows T+1..T+h mirror the return horizons, so `d_dpi_h` is the
        # change in average dark-pool positioning from the run-in (DPI10) to the h
        # sessions after the report. Session T itself is excluded on both sides: for
        # an AMC reporter it is the pre-window's cut-off day, and its tape is a mix of
        # pre-news trading and the after-hours reaction.
        post = {}
        v1 = d.iloc[t_pos + 1] if t_pos + 1 < len(d) else np.nan
        post["dpi_post_next_day"] = float(v1) if np.isfinite(v1) else np.nan
        for name, h in horizons.items():
            if name == "next_day":
                continue
            seg = d.iloc[t_pos + 1:t_pos + h + 1]
            post[f"dpi_post_{name}"] = (float(seg.mean())
                                        if int(seg.notna().sum()) >= max(3, int(0.6 * h))
                                        else np.nan)
        # change vs the pre-earnings run-in, and vs the name's own baseline level
        for name in horizons:
            post[f"d_dpi_{name}"] = post[f"dpi_post_{name}"] - win["dpi10"]
            post[f"adpi_{name}"] = post[f"dpi_post_{name}"] - dpi_base
        post["dpi_base60"] = dpi_base
        post["dpi_prior"] = dpi_prior
        post["adpi_pre"] = win["dpi10"] - dpi_base

        # non-overlapping slices of the post-earnings month, to see how the change
        # decays: week 1, week 2, then weeks 3-4
        bounds = [1] + [horizons[k] + 1 for k in ("w1", "w2") if k in horizons]
        edges = {"sw1": (bounds[0], horizons.get("w1", 5)),
                 "sw2": (bounds[1] if len(bounds) > 1 else 6, horizons.get("w2", 10)),
                 "sw34": (bounds[2] if len(bounds) > 2 else 11, horizons.get("m1", 21))}
        for name, (lo, hi) in edges.items():
            seg = d.iloc[t_pos + lo:t_pos + hi + 1]
            n = hi - lo + 1
            post[f"dpi_{name}"] = (float(seg.mean())
                                   if int(seg.notna().sum()) >= max(3, int(0.6 * n)) else np.nan)
            post[f"d_dpi_{name}"] = post[f"dpi_{name}"] - win["dpi10"]

        # --- post-earnings returns (split-adjusted), one per horizon ---
        base = a.iloc[t_pos]
        rets = {}
        for name, h in horizons.items():
            v = a.iloc[t_pos + h] if t_pos + h < len(a) else np.nan
            rets[f"{name}_ret"] = (v / base - 1) if np.isfinite(base) and np.isfinite(v) else np.nan

        # --- returns measured from a later horizon (what a DPI change could trade) ---
        for name, (frm, to) in FWD_RETURNS.items():
            s = horizons[frm]
            e_ = horizons[to] if to else 2 * horizons[frm]   # m1 -> m2 = one more month
            v0 = a.iloc[t_pos + s] if t_pos + s < len(a) else np.nan
            v1_ = a.iloc[t_pos + e_] if t_pos + e_ < len(a) else np.nan
            rets[f"fwd_{name}"] = (v1_ / v0 - 1) if np.isfinite(v0) and np.isfinite(v1_) else np.nan

        # --- realized volatility over each post-earnings window (annualized %) ---
        # RV_h = sqrt(252/h * sum_{i=1..h} r_i^2), r_i = daily log return on session T+i.
        lr = lret[tk]
        rvol = {}
        for name, h in horizons.items():
            seg = lr.iloc[t_pos + 1:t_pos + h + 1]
            rvol[f"{name}_rvol"] = (float(np.sqrt((seg ** 2).sum() * 252.0 / h) * 100)
                                    if int(seg.notna().sum()) == h else np.nan)

        rows.append({
            "ticker": tk,
            "report_date": A.date().isoformat(),
            "timing": timing,
            "base_T": T.date().isoformat(),
            "anchored": int(anchored),
            **win,
            **post,
            **rets,
            **rvol,
            "looks_reaction": looks_reaction,
            "has_data": int(np.isfinite(rets["next_day_ret"])),
        })
    ev = pd.DataFrame(rows)
    # within-name DPI percentile ranks (0..1): "is this event's run-in DPI high
    # *for this name*?" -- removes cross-sectional level differences between names
    for w in dpi_windows:
        ev[f"dpi{w}_pct"] = ev.groupby("ticker")[f"dpi{w}"].rank(pct=True)
    # same for the post-earnings DPI *change*, so "DPI rose a lot" is judged against
    # how much this name's DPI usually moves after its own reports
    for name in horizons:
        col = f"d_dpi_{name}"
        if col in ev.columns:
            ev[f"{col}_pct"] = ev.groupby("ticker")[col].rank(pct=True)
    if "dpi_prior" in ev.columns:
        ev["dpi_prior_pct"] = ev.groupby("ticker")["dpi_prior"].rank(pct=True)
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


def _ttest1(x):
    """One-sample t-test against zero; returns (mean, t, p, n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    sd = x.std(ddof=1)
    if sd == 0:
        return float(x.mean()), np.nan, np.nan, n
    t = float(x.mean()) / (sd / np.sqrt(n))
    return float(x.mean()), t, _t_sf(abs(t), n - 1) * 2, n


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
    for hz in HORIZONS:
        col, lbl = f"{hz}_ret", HZ_LABEL[hz]
        if col not in ev.columns:
            continue
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
        for hz in HORIZONS:
            col, lbl = f"{hz}_ret", hz
            if col not in ev.columns:
                continue
            hh = hi[col].to_numpy(); ll = lo[col].to_numpy()
            diff, t, pv, nh, nl = _welch(hh, ll)
            hm = np.nanmean(hh) if np.isfinite(hh).any() else np.nan
            lm = np.nanmean(ll) if np.isfinite(ll).any() else np.nan
            hpos = 100 * np.nanmean((hh > 0)) if np.isfinite(hh).any() else np.nan
            lpos = 100 * np.nanmean((ll > 0)) if np.isfinite(ll).any() else np.nan
            out.append(f"     {lbl:8s}: high {hm*100:+.2f}% ({hpos:.0f}% up)   "
                       f"low {lm*100:+.2f}% ({lpos:.0f}% up)   "
                       f"high-low {diff*100:+.2f}pp (t={t:+.2f}, p={pv:.3f})")

    # realized volatility over each post-earnings window (annualized), and whether
    # pre-earnings DPI relates to how much the stock actually moves afterwards
    if "m1_rvol" in ev.columns:
        out.append("")
        out.append("--- REALIZED VOLATILITY (annualized) over each post-earnings window ---")
        for hz in HORIZONS:
            col = f"{hz}_rvol"
            if col not in ev.columns:
                continue
            s = ev[col].dropna()
            p = ev["dpi10_pct"]
            hi = ev[p >= 2 / 3][col]; lo = ev[p <= 1 / 3][col]
            pr, pp, _ = _pearson(ev["dpi10"], ev[col])
            out.append(f"  {HZ_LABEL[hz]:20s} mean {s.mean():5.1f}%  median {s.median():5.1f}%   "
                       f"DPI10 corr r={pr:+.3f} (p={pp:.3f})   "
                       f"high-DPI {hi.mean():.1f}% vs low-DPI {lo.mean():.1f}%")

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


CHG_HZ = [("w1", "1 WEEK  (T+1..T+5)"), ("w2", "2 WEEKS (T+1..T+10)"),
          ("m1", "1 MONTH (T+1..T+21)")]


def summarize_dpi_change(ev):
    """How DPI itself moves AFTER the report: level change at 1 week / 2 weeks /
    1 month, what drives it, and whether the change carries any forward signal."""
    out = []
    out.append("=" * 78)
    out.append("POST-EARNINGS CHANGE IN DPI  (1 week / 2 weeks / 1 month after the report)")
    out.append("=" * 78)
    pre = ev["dpi10"]
    out.append(f"Pre-earnings DPI10 mean {pre.mean():.4f}   "
               f"name-baseline DPI60 mean {ev['dpi_base60'].mean():.4f}   "
               f"run-in premium {100*ev['adpi_pre'].mean():+.2f} pp "
               f"(t={_ttest1(ev['adpi_pre'])[1]:+.2f})")
    out.append("  (DPI is a ratio 0..1; changes are reported in percentage points of that ratio.)")

    # --- level in / level out / change, per horizon -------------------------
    out.append("")
    out.append("--- LEVEL AND CHANGE ---")
    out.append(f"  {'window':22s} {'pre':>7s} {'post':>7s} {'change':>9s} {'median':>8s} "
               f"{'%up':>5s} {'t':>7s} {'p':>7s}   n")
    if "dpi_post_next_day" in ev.columns:
        chg = ev["d_dpi_next_day"]
        m, t, p, n = _ttest1(chg)
        out.append(f"  {'REACTION DAY (T+1)':22s} {pre.mean():7.4f} "
                   f"{ev['dpi_post_next_day'].mean():7.4f} {100*m:+8.2f}pp "
                   f"{100*chg.median():+7.2f}pp {100*(chg > 0).mean():4.0f}% "
                   f"{t:+7.2f} {p:7.4f}   {n}")
    for k, lbl in CHG_HZ:
        chg = ev[f"d_dpi_{k}"]
        m, t, p, n = _ttest1(chg)
        out.append(f"  {lbl:22s} {pre.mean():7.4f} {ev[f'dpi_post_{k}'].mean():7.4f} "
                   f"{100*m:+8.2f}pp {100*chg.median():+7.2f}pp "
                   f"{100*(chg > 0).mean():4.0f}% {t:+7.2f} {p:7.4f}   {n}")

    # --- non-overlapping slices: how the change decays ----------------------
    out.append("")
    out.append("--- NON-OVERLAPPING SLICES (change vs pre-earnings DPI10) ---")
    for k, lbl in [("sw1", "week 1   (T+1..T+5)"), ("sw2", "week 2   (T+6..T+10)"),
                   ("sw34", "weeks 3-4 (T+11..T+21)")]:
        col = f"d_dpi_{k}"
        if col not in ev.columns:
            continue
        m, t, p, n = _ttest1(ev[col])
        out.append(f"  {lbl:24s} {100*m:+6.2f}pp   (t={t:+.2f}, p={p:.4f}, n={n})   "
                   f"level {ev[f'dpi_{k}'].mean():.4f}")

    # --- what the change is made of ----------------------------------------
    out.append("")
    out.append("--- WHAT PREDICTS THE CHANGE ---")
    for k, lbl in CHG_HZ:
        chg = ev[f"d_dpi_{k}"]
        r_pre, p_pre, n = _pearson(pre, chg)
        r_ab, p_ab, _ = _pearson(ev["adpi_pre"], chg)
        r_rx, p_rx, _ = _pearson(ev["next_day_ret"], chg)
        r_pers, p_pers, _ = _pearson(pre, ev[f"dpi_post_{k}"])
        r_ind, p_ind, _ = _pearson(ev["dpi_prior"], chg)
        out.append(f"  {lbl}")
        out.append(f"     vs pre-earnings DPI10   r={r_pre:+.3f} (p={p_pre:.4f}, n={n})"
                   f"    [persistence: corr(pre, post) = {r_pers:+.3f}]")
        out.append(f"     vs prior window (-60..-11) r={r_ind:+.3f} (p={p_ind:.4f})"
                   "  <- no shared observations: unbiased reversion read")
        out.append(f"     vs run-in premium       r={r_ab:+.3f} (p={p_ab:.4f})"
                   "     <- run-in above the name's own 60d baseline")
        out.append(f"     vs earnings reaction    r={r_rx:+.3f} (p={p_rx:.4f})")
        up = ev[ev["next_day_ret"] > 0][f"d_dpi_{k}"].to_numpy()
        dn = ev[ev["next_day_ret"] <= 0][f"d_dpi_{k}"].to_numpy()
        diff, t, pv, nu, nd = _welch(up, dn)
        out.append(f"     after an UP reaction {100*np.nanmean(up):+6.2f}pp (n={nu})   "
                   f"after a DOWN reaction {100*np.nanmean(dn):+6.2f}pp (n={nd})   "
                   f"diff {100*diff:+.2f}pp (t={t:+.2f}, p={pv:.4f})")

    # --- change by pre-earnings DPI tercile (mean reversion, visible) -------
    out.append("")
    out.append("--- CHANGE BY PRE-EARNINGS DPI TERCILE (within-name) ---")
    for pcol, tag in [("dpi10_pct", "sorted on DPI10 (shares noise with the change)"),
                      ("dpi_prior_pct", "sorted on the prior -60..-11 window (unbiased)")]:
        if pcol not in ev.columns:
            continue
        out.append(f"  {tag}")
        p10 = ev[pcol]
        hi_m, lo_m = ev[p10 >= 2 / 3], ev[p10 <= 1 / 3]
        for k, lbl in CHG_HZ:
            hh, ll = hi_m[f"d_dpi_{k}"].to_numpy(), lo_m[f"d_dpi_{k}"].to_numpy()
            diff, t, pv, nh, nl = _welch(hh, ll)
            out.append(f"    {lbl}:  high-DPI events {100*np.nanmean(hh):+6.2f}pp (n={nh})   "
                       f"low-DPI events {100*np.nanmean(ll):+6.2f}pp (n={nl})   "
                       f"high-low {100*diff:+.2f}pp (t={t:+.2f}, p={pv:.4f})")

    # --- does the change carry forward signal? ------------------------------
    # Each change is only observable at the end of its own window, so it is tested
    # against the return FROM that point on -- no look-ahead.
    out.append("")
    out.append("--- DOES THE CHANGE PREDICT WHAT COMES NEXT? ---")
    tests = [("w1", "fwd_w1_m1", "dDPI over week 1   -> return T+5 -> T+21"),
             ("w2", "fwd_w2_m1", "dDPI over 2 weeks  -> return T+10 -> T+21"),
             ("m1", "fwd_m1_m2", "dDPI over 1 month  -> return T+21 -> T+42")]
    for k, rcol, lbl in tests:
        if rcol not in ev.columns:
            continue
        chg, fwd = ev[f"d_dpi_{k}"], ev[rcol]
        r, p, n = _pearson(chg, fwd)
        sr, sp, _ = _spearman(chg, fwd)
        pc = ev[f"d_dpi_{k}_pct"]
        hh = ev[pc >= 2 / 3][rcol].to_numpy()
        ll = ev[pc <= 1 / 3][rcol].to_numpy()
        diff, t, pv, nh, nl = _welch(hh, ll)
        # for reference: the pre-earnings level over the same forward window
        rp, pp_, _ = _pearson(ev["dpi10"], fwd)
        out.append(f"  {lbl}")
        out.append(f"     Pearson r={r:+.3f} (p={p:.4f}, n={n})   Spearman r={sr:+.3f} (p={sp:.4f})"
                   f"   [pre-earnings DPI10 over the same window: r={rp:+.3f} (p={pp_:.4f})]")
        out.append(f"     rising-DPI tercile {100*np.nanmean(hh):+.2f}% (n={nh})   "
                   f"falling-DPI tercile {100*np.nanmean(ll):+.2f}% (n={nl})   "
                   f"spread {100*diff:+.2f}pp (t={t:+.2f}, p={pv:.4f})")

    # --- double sort: run-in level x post-earnings change -------------------
    out.append("")
    out.append("--- DOUBLE SORT: pre-earnings DPI10 tercile x week-1 DPI change tercile ---")
    out.append("    cell = mean return T+5 -> T+21 (n)")
    lab = ["low", "mid", "high"]
    def _terc_mask(col):
        p = ev[col]
        return [p <= 1 / 3, (p > 1 / 3) & (p < 2 / 3), p >= 2 / 3]
    rows_m = _terc_mask("dpi10_pct")
    cols_m = _terc_mask("d_dpi_w1_pct")
    out.append("    " + "run-in / dDPI".ljust(16) + "".join(f"{c:>16s}" for c in lab))
    for i, rm in enumerate(rows_m):
        cells = []
        for cm in cols_m:
            s = ev[rm & cm]["fwd_w1_m1"].dropna()
            cells.append(f"{100*s.mean():+7.2f}% ({len(s):4d})" if len(s) else f"{'--':>15s}")
        out.append(f"    {lab[i]:16s}" + "".join(f"{c:>16s}" for c in cells))
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
    ap.add_argument("--no-merge-classes", action="store_true", default=False,
                    help="keep dual-class tickers (GOOG & GOOGL) separate instead of merging")
    args = ap.parse_args()

    earn = load_earnings(args.earnings)
    syms = sorted(earn["ticker"].unique())
    # ~80 sessions of lead-in so the 60-session DPI baseline is populated for the
    # earliest events; ~100 calendar days of tail so the last events still have
    # T+42 (the second post-earnings month) available
    start = (earn["report_date"].min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (earn["report_date"].max() + pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    print(f"Universe: {len(syms)} names   window {start} -> {end}", file=sys.stderr)

    panels = N.build_universe_panels(syms, start, end, workers=args.workers,
                                     cache_dir=args.cache_dir or None, ns="earn",
                                     refresh=args.refresh, label="EARN")
    if not args.no_merge_classes:
        panels, earn = merge_share_classes(panels, earn)

    horizons = dict(HORIZONS); horizons["m1"] = args.month_sessions
    ev = build_events(earn, panels, horizons=horizons, anchor=args.anchor)
    out_csv = f"{args.out_prefix}_events.csv"
    ev.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(ev)} events)", file=sys.stderr)

    print(summarize(ev))
    print()
    print(summarize_dpi_change(ev))
    return ev, panels


if __name__ == "__main__":
    main()
