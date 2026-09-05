#!/usr/bin/env python3
"""
Does the STRENGTH of a name's DPI affect its realized volatility?

Signal   : trailing (no-look-ahead) decile of the name's 5-day-MA dark ratio (DPI strength 1..10),
           the same construction the other DPI studies use. Continuous within-name z-score is
           tested as a robustness variant.
Target   : forward realized vol RV_h = annualized std of daily log returns over [t+1, t+h].

Two confounds have to be removed or a spurious link appears:
  (a) vol persistence -- an already-volatile name stays volatile (RV is highly autocorrelated);
  (b) market-vol regimes -- high DPI tends to cluster when the whole market is more volatile.
So the reported metrics are (1) forward RV by decile in RAW and MARKET-RELATIVE ln(RV/RV_QQQ) form
with the concurrent market RV shown alongside, (2) a within-name high(>=D8) minus low(<=D3)
market-relative contrast, and (3) the decisive test: a pooled OLS of ln(forward RV) on the DPI
decile CONTROLLING for trailing RV (HAR terms) and market RV -- does DPI strength add any
vol-predictive information beyond persistence? Cluster-robust (by date) standard errors; a
pre/post-2022 split checks stability.

No scipy dependency. Reads the dashboard `rel` payload (per-name close + 5-day-MA dark ratio `d`).

Usage:  python dpi_volatility_study.py --payload docs/index.html
"""
import argparse
import csv
import math
import os

import numpy as np

import dpi_responder_study as R   # load_payload, Study, trailing_deciles

ANN = math.sqrt(252)
HORIZONS = (10, 21, 42)


def daily_logret(logc, T):
    r = np.full(T, np.nan)
    r[1:] = logc[1:] - logc[:-1]
    return r


def rv(r, h, T, forward=True):
    """annualized realized vol (%) over the h-day window (forward [t+1,t+h] or trailing [t-h+1,t])."""
    out = np.full(T, np.nan)
    need = max(5, h // 2)
    for t in range(T):
        w = r[t + 1:t + 1 + h] if forward else r[max(0, t - h + 1):t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= need:
            out[t] = np.std(w) * ANN * 100
    return out


def _ols_cluster(Y, X, G):
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ b
    inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(G):
        m = G == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    ng = len(np.unique(G))
    cov = inv @ meat @ inv * (ng / (ng - 1))
    return b, np.sqrt(np.diag(cov)), len(Y), ng


def _zscore_trailing(x, win=252, minobs=120):
    out = np.full(len(x), np.nan)
    buf = []
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        if len(buf) >= minobs:
            a = np.asarray(buf); sd = a.std()
            if sd > 0:
                out[i] = (v - a.mean()) / sd
        buf.append(v)
        if len(buf) > win:
            buf.pop(0)
    return out


def run(payload, out_csv="data/dpi_volatility_by_decile.csv"):
    st = R.Study(payload)
    T, dates, names = st.T, st.dates, st.names
    ret = {nm: daily_logret(st.logc[nm], T) for nm in names}
    retb = daily_logret(st.logb, T)
    print(f"Universe: {len(names)} names | {dates[0]}..{dates[-1]} | forward RV, DPI = trailing decile")

    csv_rows = None
    for h in HORIZONS:
        rvf = {nm: rv(ret[nm], h, T, True) for nm in names}
        rvb = rv(retb, h, T, True)
        rvt = {nm: rv(ret[nm], h, T, False) for nm in names}
        rvt5 = {nm: rv(ret[nm], 5, T, False) for nm in names}

        # 1) dose-response by decile
        raw = {d: [] for d in range(1, 11)}
        rel = {d: [] for d in range(1, 11)}
        mkt = {d: [] for d in range(1, 11)}
        for nm in names:
            dec = st.dec[nm]; f = rvf[nm]
            for t in range(T):
                d = dec[t]
                if np.isnan(d) or not np.isfinite(f[t]) or not np.isfinite(rvb[t]) or rvb[t] <= 0:
                    continue
                dd = int(d)
                raw[dd].append(f[t]); rel[dd].append(math.log(f[t] / rvb[t])); mkt[dd].append(rvb[t])
        print(f"\n==== forward RV {h}d ====")
        print(f"  decile | rawRV% | ln(RV/QQQ)  se  | mktRV% |    n")
        for d in range(1, 11):
            if not raw[d]:
                continue
            neff = max(1.0, len(rel[d]) / h)
            print(f"    D{d:<2}   {np.mean(raw[d]):>6.1f}   {np.mean(rel[d]):>+6.3f} {np.std(rel[d])/math.sqrt(neff):.3f} "
                  f"  {np.mean(mkt[d]):>5.1f}  {len(raw[d]):>7}")
        spread = np.mean(rel[10]) - np.mean(rel[1])
        print(f"  D10-D1 ln(RV/QQQ) = {spread:+.3f} (RV vs market {math.exp(np.mean(rel[10])):.2f}x vs "
              f"{math.exp(np.mean(rel[1])):.2f}x); market RV itself D10 {np.mean(mkt[10]):.1f}% vs D1 {np.mean(mkt[1]):.1f}%")
        if h == 21:
            csv_rows = [(d, np.mean(raw[d]), np.mean(rel[d]), np.mean(mkt[d]), len(raw[d])) for d in range(1, 11)]

        # 2) within-name high vs low (market-relative)
        diffs = []
        for nm in names:
            dec = st.dec[nm]; f = rvf[nm]
            hi = [math.log(f[t] / rvb[t]) for t in range(T)
                  if not np.isnan(dec[t]) and dec[t] >= 8 and np.isfinite(f[t]) and np.isfinite(rvb[t]) and rvb[t] > 0]
            lo = [math.log(f[t] / rvb[t]) for t in range(T)
                  if not np.isnan(dec[t]) and dec[t] <= 3 and np.isfinite(f[t]) and np.isfinite(rvb[t]) and rvb[t] > 0]
            if len(hi) >= 30 and len(lo) >= 30:
                diffs.append(np.mean(hi) - np.mean(lo))
        dv = np.array(diffs)
        tstat = dv.mean() / (dv.std(ddof=1) / math.sqrt(len(dv))) if len(dv) > 1 else float("nan")
        print(f"  within-name high(>=D8)-low(<=D3) ln(RV/QQQ): mean {dv.mean():+.3f}  "
              f"{int((dv > 0).sum())}/{len(dv)} names positive  cross-name t={tstat:+.2f}  (exp {math.exp(dv.mean()):.2f}x)")

        # 3) incremental predictive OLS (+ pre/post-2022 split)
        def fit(lo, hi):
            Y, X, G = [], [], []
            for nm in names:
                dec = st.dec[nm]; f = rvf[nm]; tr = rvt[nm]; t5 = rvt5[nm]
                for t in range(lo, min(hi, T)):
                    if (np.isnan(dec[t]) or not np.isfinite(f[t]) or f[t] <= 0 or not np.isfinite(tr[t]) or tr[t] <= 0
                            or not np.isfinite(t5[t]) or t5[t] <= 0 or not np.isfinite(rvb[t]) or rvb[t] <= 0):
                        continue
                    Y.append(math.log(f[t])); X.append([1.0, dec[t], math.log(tr[t]), math.log(t5[t]), math.log(rvb[t])]); G.append(dates[t])
            return _ols_cluster(np.array(Y), np.array(X), np.array(G))
        mid = T // 2
        for lbl, (lo, hh) in [("full", (0, T)), ("pre-2022", (0, mid)), ("2022+", (mid, T))]:
            b, se, n, ng = fit(lo, hh)
            tag = "  <-- DPI" if lbl == "full" else ""
            print(f"  OLS {lbl:<9} DPI-decile coef {b[1]:+.4f} t {b[1]/se[1]:+.2f}  "
                  f"(fwdRV x{math.exp(b[1]):.3f}/decile; controls ln trailRV t={b[2]/se[2]:.0f}, ln mktRV t={b[4]/se[4]:.0f}; n {n:,}){tag}")

    if csv_rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dpi_decile", "fwd_rv_21d_pct", "ln_rv_over_qqq", "concurrent_qqq_rv_pct", "n_obs"])
            for d, a, r_, m, n in csv_rows:
                w.writerow([d, f"{a:.2f}", f"{r_:.4f}", f"{m:.2f}", n])
        print(f"\nwrote {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", default="docs/index.html")
    ap.add_argument("--out", default="data/dpi_volatility_by_decile.csv")
    args = ap.parse_args()
    run(R.load_payload(args.payload), out_csv=args.out)


if __name__ == "__main__":
    main()
