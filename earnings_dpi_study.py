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

Each horizon also gets a MARKET-EXCESS twin (`*_xret` = the name's return minus
QQQ's return over the identical window) -- the headline outcome, since raw
returns in a mostly-bull sample partly just ride beta -- plus:
    pre_m1_ret  = the 21 sessions INTO the report (momentum control)
    m1_post_ret = T+1 -> T+21 (drift AFTER the first reaction session)

Inference upgrades
------------------
* within-name DPI percentiles come in two flavours: full-history (original,
  mild look-ahead) and EXPANDING (each event ranked only against that name's
  prior events -- what was knowable on the day; needs >= 8 prior events).
* earnings cluster in reporting weeks, so pooled p-values overstate precision:
  a CLUSTER BOOTSTRAP by calendar quarter (the earnings season) is reported
  next to every headline stat. ~2,700 events collapse to ~32 seasons.
* a DOUBLE SORT on pre-earnings momentum terciles x DPI terciles separates the
  DPI effect from "it already ran up into the print".
* a GAP-DIRECTION split tests whether high pre-report DPI predicts drift
  regardless of the news, or specifically the recovery of gapped-down names
  (the "informed dark accumulation" tell).

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


BENCH_TICKER = "QQQ"   # market proxy for the excess-return outcome


def build_events(earnings, panels, horizons=HORIZONS, dpi_windows=(5, 10), anchor=False,
                 bench=BENCH_TICKER):
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
    bench_a = adj[bench] if bench in adj.columns else None
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

        # --- post-earnings returns (split-adjusted), one per horizon, raw and
        # in EXCESS of the market proxy over the identical window ---
        base = a.iloc[t_pos]
        bbase = bench_a.iloc[t_pos] if bench_a is not None else np.nan
        rets = {}
        for name, h in horizons.items():
            v = a.iloc[t_pos + h] if t_pos + h < len(a) else np.nan
            r_own = (v / base - 1) if np.isfinite(base) and np.isfinite(v) else np.nan
            rets[f"{name}_ret"] = r_own
            bv = (bench_a.iloc[t_pos + h]
                  if bench_a is not None and t_pos + h < len(bench_a) else np.nan)
            r_b = (bv / bbase - 1) if np.isfinite(bbase) and np.isfinite(bv) else np.nan
            rets[f"{name}_xret"] = (r_own - r_b
                                    if np.isfinite(r_own) and np.isfinite(r_b) else np.nan)

        # momentum INTO the report (last 21 sessions ending at T) and the drift
        # AFTER the first reaction session (T+1 -> T+21)
        pre_v = a.iloc[t_pos - 21] if t_pos - 21 >= 0 else np.nan
        pre_m1 = (base / pre_v - 1) if np.isfinite(base) and np.isfinite(pre_v) else np.nan
        t1 = a.iloc[t_pos + 1] if t_pos + 1 < len(a) else np.nan
        vm = a.iloc[t_pos + horizons["m1"]] if t_pos + horizons["m1"] < len(a) else np.nan
        m1_post = (vm / t1 - 1) if np.isfinite(t1) and np.isfinite(vm) else np.nan

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
            **rets,
            "pre_m1_ret": pre_m1,
            "m1_post_ret": m1_post,
            **rvol,
            "looks_reaction": looks_reaction,
            "has_data": int(np.isfinite(rets["next_day_ret"])),
        })
    ev = pd.DataFrame(rows)
    if ev.empty:                                # no event produced a row: nothing to rank
        return ev
    # within-name DPI percentile ranks (0..1): "is this event's run-in DPI high
    # *for this name*?" -- removes cross-sectional level differences between names.
    # Two flavours: full-history (original; mild look-ahead) and EXPANDING (each
    # event ranked only against the same name's PRIOR events + itself, so the
    # tercile was knowable on the day; NaN until >= EXPAND_MIN_EVENTS history).
    ev = ev.sort_values(["ticker", "report_date"]).reset_index(drop=True)
    for w in dpi_windows:
        ev[f"dpi{w}_pct"] = ev.groupby("ticker")[f"dpi{w}"].rank(pct=True)
        ev[f"dpi{w}_pct_exp"] = (
            ev.groupby("ticker")[f"dpi{w}"]
              .transform(lambda s: _expanding_pct(s, EXPAND_MIN_EVENTS)))
    return ev


EXPAND_MIN_EVENTS = 8   # ~2 years of quarters before an expanding rank is scored


def _expanding_pct(s, min_events=EXPAND_MIN_EVENTS):
    """Percentile (0..1, mid-rank) of each value within the series up to and
    including itself; NaN until `min_events` non-NaN observations have accrued."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    seen = []
    for i, x in enumerate(v):
        if np.isfinite(x):
            seen.append(x)
            if len(seen) >= min_events:
                arr = np.asarray(seen)
                out[i] = ((arr < x).mean() + 0.5 * (arr == x).mean())
    return pd.Series(out, index=s.index)


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


def _season_key(ev):
    """Calendar quarter of the report date -- the earnings 'season' cluster."""
    d = pd.to_datetime(ev["report_date"])
    return (d.dt.year * 10 + d.dt.quarter).to_numpy()


def cluster_boot_corr(ev, xcol, ycol, B=4000, seed=0):
    """Pearson r of xcol vs ycol with a cluster bootstrap over earnings seasons
    (calendar quarters resampled with replacement). Events inside a season are
    cross-sectionally correlated, so the honest unit is the ~32 seasons, not the
    ~2,700 events. Vectorized via per-quarter sufficient statistics.
    Returns (r, ci_lo, ci_hi, p_boot, n_events, n_quarters)."""
    d = ev[[xcol, ycol, "report_date"]].dropna()
    if len(d) < 30:
        return (np.nan,) * 4 + (len(d), 0)
    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    q = _season_key(d)
    suff = pd.DataFrame({"q": q, "n": 1.0, "sx": x, "sy": y,
                         "sxx": x * x, "syy": y * y, "sxy": x * y})
    G = suff.groupby("q").sum().to_numpy()
    k = len(G)

    def corr_of(T):
        n, sx, sy, sxx, syy, sxy = T.T if T.ndim == 2 else T
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return cov / np.sqrt(vx * vy)

    r = float(corr_of(G.sum(axis=0)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(B, k))
    rb = corr_of(G[draws].sum(axis=1))
    rb = rb[np.isfinite(rb)]
    if not len(rb):
        return r, np.nan, np.nan, np.nan, len(d), k
    lo, hi = np.percentile(rb, [2.5, 97.5])
    p = 2 * min((rb <= 0).mean(), (rb >= 0).mean())
    return r, float(lo), float(hi), float(min(1.0, p)), len(d), k


def cluster_boot_spread(ev, pcol, rcol, B=4000, seed=0):
    """High-minus-low tercile mean spread of rcol (terciles of pcol, cutoffs at
    1/3 and 2/3 as elsewhere) with the same season-cluster bootstrap.
    Returns (spread, ci_lo, ci_hi, p_boot, n_hi, n_lo, n_quarters)."""
    d = ev[[pcol, rcol, "report_date"]].dropna()
    if len(d) < 30:
        return (np.nan,) * 4 + (0, 0, 0)
    hi_m = (d[pcol] >= 2 / 3).to_numpy()
    lo_m = (d[pcol] <= 1 / 3).to_numpy()
    r = d[rcol].to_numpy(dtype=float)
    q = _season_key(d)
    suff = pd.DataFrame({"q": q,
                         "nh": hi_m.astype(float), "sh": np.where(hi_m, r, 0.0),
                         "nl": lo_m.astype(float), "sl": np.where(lo_m, r, 0.0)})
    G = suff.groupby("q").sum().to_numpy()
    k = len(G)

    def spread_of(T):
        nh, sh, nl, sl = T.T if T.ndim == 2 else T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((nh > 0) & (nl > 0), sh / nh - sl / nl, np.nan)

    est = float(spread_of(G.sum(axis=0)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(B, k))
    sb = spread_of(G[draws].sum(axis=1))
    sb = sb[np.isfinite(sb)]
    if not len(sb):
        return est, np.nan, np.nan, np.nan, int(hi_m.sum()), int(lo_m.sum()), k
    lo, hi = np.percentile(sb, [2.5, 97.5])
    p = 2 * min((sb <= 0).mean(), (sb >= 0).mean())
    return (est, float(lo), float(hi), float(min(1.0, p)),
            int(hi_m.sum()), int(lo_m.sum()), k)


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
    has_x = "m1_xret" in ev.columns and ev["m1_xret"].notna().any()
    for hz in HORIZONS:
        col, lbl = f"{hz}_ret", HZ_LABEL[hz]
        if col not in ev.columns:
            continue
        s = ev[col].dropna()
        out.append("")
        out.append(f"--- {lbl} ---   n={len(s)}")
        out.append(f"    mean {s.mean()*100:+.2f}%   median {s.median()*100:+.2f}%   "
                   f"std {s.std()*100:.2f}%   %positive {100*(s>0).mean():.0f}%")
        if has_x:
            sx = ev[f"{hz}_xret"].dropna()
            out.append(f"    EXCESS vs QQQ: mean {sx.mean()*100:+.2f}%   "
                       f"median {sx.median()*100:+.2f}%   "
                       f"%positive {100*(sx>0).mean():.0f}%")
        for w in dpi_windows:
            sig = ev[f"dpi{w}"]
            pr, pp, pn = _pearson(sig, ev[col])
            sr, sp, sn = _spearman(sig, ev[col])
            prp, ppp, _ = _pearson(ev[f"dpi{w}_pct"], ev[col])
            out.append(f"    DPI{w:<2}  Pearson r={pr:+.3f} (p={pp:.3f}, n={pn})   "
                       f"Spearman r={sr:+.3f} (p={sp:.3f})   "
                       f"within-name r={prp:+.3f} (p={ppp:.3f})")
        if has_x:
            for w in dpi_windows:
                xr, xp, _ = _pearson(ev[f"dpi{w}"], ev[f"{hz}_xret"])
                cr, clo, chi, cp, cn, ck = cluster_boot_corr(ev, f"dpi{w}", f"{hz}_xret")
                out.append(f"    DPI{w:<2} vs EXCESS  r={xr:+.3f} (p={xp:.3f})   "
                           f"season-cluster boot: r={cr:+.3f} "
                           f"[95% CI {clo:+.3f},{chi:+.3f}] p={cp:.3f} "
                           f"({ck} quarters)")

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

    # the same spreads measured honestly: excess of QQQ, season-clustered p,
    # and (separately) EXPANDING within-name percentiles (no look-ahead)
    if has_x:
        out.append("")
        out.append("--- HEADLINE SPREADS, EXCESS vs QQQ + season-cluster bootstrap ---")
        for pcol, tag in [("dpi10_pct", "full-history pct (look-ahead)"),
                          ("dpi10_pct_exp", f"EXPANDING pct (>= {EXPAND_MIN_EVENTS} "
                                            "prior events; tradable)")]:
            if pcol not in ev.columns or not ev[pcol].notna().any():
                continue
            out.append(f"  DPI10 terciles on {tag}:")
            for hz in HORIZONS:
                xcol = f"{hz}_xret"
                if xcol not in ev.columns:
                    continue
                sp, lo_, hi_, pb, nh, nl, k = cluster_boot_spread(ev, pcol, xcol)
                out.append(f"     {hz:8s}: high-low {sp*100:+.2f}pp excess   "
                           f"[95% CI {lo_*100:+.2f},{hi_*100:+.2f}] "
                           f"cluster p={pb:.3f}   (n hi/lo={nh}/{nl}, {k} quarters)")

    # momentum double-sort: does high DPI add anything beyond the run-in?
    if has_x and "pre_m1_ret" in ev.columns:
        out.append("")
        out.append("--- DOUBLE SORT: pre-earnings momentum terciles x DPI10 terciles ---")
        out.append("    (cells = mean 1-month EXCESS return; momentum = 21 sessions into T)")
        d = ev.dropna(subset=["pre_m1_ret", "dpi10_pct", "m1_xret"]).copy()
        if len(d) >= 90:
            d["mom_t"] = pd.qcut(d["pre_m1_ret"], 3, labels=["down", "flat", "up"])
            for mt in ["down", "flat", "up"]:
                g = d[d["mom_t"] == mt]
                hi = g[g["dpi10_pct"] >= 2 / 3]["m1_xret"]
                md = g[(g["dpi10_pct"] > 1 / 3) & (g["dpi10_pct"] < 2 / 3)]["m1_xret"]
                lo = g[g["dpi10_pct"] <= 1 / 3]["m1_xret"]
                diff, t, pv, _, _ = _welch(hi.to_numpy(), lo.to_numpy())
                pre = g["pre_m1_ret"].mean()
                out.append(f"     momentum {mt:5s} (avg run-in {pre*100:+.1f}%, n={len(g)}): "
                           f"loD {lo.mean()*100:+.2f}%  midD {md.mean()*100:+.2f}%  "
                           f"hiD {hi.mean()*100:+.2f}%   "
                           f"hi-lo {diff*100:+.2f}pp (p={pv:.3f})")
        else:
            out.append("     insufficient data")

    # gap-direction split: drift after the reaction, conditional on the reaction
    if "m1_post_ret" in ev.columns:
        out.append("")
        out.append("--- GAP-DIRECTION SPLIT: post-reaction drift (T+1 -> T+21) by DPI10 ---")
        out.append("    ('informed dark accumulation' would show up as high-DPI names "
                   "recovering after a down gap)")
        d = ev.dropna(subset=["next_day_ret", "m1_post_ret", "dpi10_pct"])
        cuts = [("gap DOWN < -2%", d["next_day_ret"] < -0.02),
                ("flat +/-2%", d["next_day_ret"].abs() <= 0.02),
                ("gap UP > +2%", d["next_day_ret"] > 0.02)]
        for lbl, m in cuts:
            g = d[m]
            if len(g) < 30:
                continue
            hi = g[g["dpi10_pct"] >= 2 / 3]["m1_post_ret"]
            lo = g[g["dpi10_pct"] <= 1 / 3]["m1_post_ret"]
            diff, t, pv, _, _ = _welch(hi.to_numpy(), lo.to_numpy())
            r, rp, _ = _pearson(g["dpi10"], g["m1_post_ret"])
            out.append(f"     {lbl:16s} n={len(g):4d}: drift hiD {hi.mean()*100:+.2f}% "
                       f"vs loD {lo.mean()*100:+.2f}%  ({diff*100:+.2f}pp, p={pv:.3f})   "
                       f"corr(DPI10, drift) r={r:+.3f} (p={rp:.3f})")

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
    ap.add_argument("--summary-out", default="",
                    help="also write the text summary to this file (committed by the "
                         "refresh workflow so the findings doc can be updated from it)")
    args = ap.parse_args()

    earn = load_earnings(args.earnings)
    # QQQ rides along in the panel build purely as the market proxy for the
    # excess-return outcome (it never appears in the earnings CSV itself)
    syms = sorted(set(earn["ticker"].unique()) | {BENCH_TICKER})
    pad = pd.Timedelta(days=25)
    start = (earn["report_date"].min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (earn["report_date"].max() + pad + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
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

    text = summarize(ev)
    print(text)
    if args.summary_out:
        Path(args.summary_out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.summary_out}", file=sys.stderr)
    return ev, panels


if __name__ == "__main__":
    main()
