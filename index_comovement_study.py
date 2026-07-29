#!/usr/bin/env python3
"""
Cross-index DIX comovement -> 1-month forward return study.
===========================================================

Question (per request): using a 5-day moving average of the dollar-DIX for all
three index gauges (NDX-100, S&P 500, Russell 2000 / IWM), how do the indices'
own 1-month forward returns behave when the three DIX gauges COMOVE vs. DIVERGE?
For example: when SPX-DIX and NDX-DIX sit in their lower deciles while IWM-DIX
sits in its upper deciles, what are the subsequent 1-month returns of each index?

Signal (per index, no look-ahead in the DIX itself)
---------------------------------------------------
DIX_t = dollar-weighted Sum($ short vol) / Sum($ off-exchange vol) across the
index constituents each day (SqueezeMetrics' construction; see
`compute_dollar_dix` in ndx_dark_residual.py). We smooth it with a 5-day moving
average (the same 5d MA the dashboard uses as its residual benchmark):

    DIX5_t = mean(DIX over the 5 trading days ending t)      # min 3 obs

Each index's DIX5 is then bucketed into deciles over the common sample, and the
deciles collapsed to three regimes per index:

    Low  = deciles 1-3   (DIX5 in the bottom 30% of its own history)
    Mid  = deciles 4-7
    High = deciles 8-10  (DIX5 in the top 30% of its own history)

Outcome (per index)
-------------------
The 1-month (21 trading day) forward return of each index's own price proxy:
QQQ for NDX, SPY for SPX, IWM for the Russell 2000 -- exactly the forward
returns already packed into the dashboard payload (`compute_forward_return`,
in percent). A regime observed on date t is scored by the return realised over
the following 21 sessions, so there is no look-ahead in the outcome either.

Data source
-----------
The three DIX series and their 1-month forward returns are read straight from
the JSON payload embedded in the built dashboard (`docs/index.html`, the
`const P = {...}` blob) so the study is reproducible offline against the latest
refresh, with no live FINRA re-fetch. Pass --html to point at another build.

    NDX:  P.rel.ndx_dix              vs  P.rel.r21[P.bench]   (QQQ)
    SPX:  P.spx.dix                  vs  P.spx.r21            (SPY)
    IWM:  P.iwm.d                    vs  P.iwm.r21            (IWM)

Usage
-----
    python index_comovement_study.py                       # text report
    python index_comovement_study.py --html docs/index.html --csv out.csv
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

IDX = ["NDX", "SPX", "IWM"]


def load_payload(html_path):
    """Pull the `const P = {...};` JSON blob out of a built dashboard HTML."""
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    m = re.search(r"const P = (\{.*?\});", html, re.S)
    if not m:
        sys.exit(f"Could not find the embedded payload in {html_path}")
    return json.loads(m.group(1))


def build_aligned(P, ma_window=5, min_periods=3):
    """Aligned frame of each index's DIX 5d MA + its 1-month forward return (%),
    restricted to the dates on which all three DIX gauges are available."""
    def series(dates, dix, ret):
        return pd.DataFrame(
            {"date": pd.to_datetime(dates), "dix": dix, "r1m": ret}
        ).set_index("date")

    raw = {
        "NDX": series(P["rel"]["dates"], P["rel"]["ndx_dix"], P["rel"]["r21"][P["bench"]]),
        "SPX": series(P["spx"]["dates"], P["spx"]["dix"], P["spx"]["r21"]),
        "IWM": series(P["iwm"]["dates"], P["iwm"]["d"], P["iwm"]["r21"]),
    }
    cols = {}
    for name, df in raw.items():
        cols[name + "_dix5"] = df["dix"].rolling(ma_window, min_periods=min_periods).mean()
        cols[name + "_r1m"] = df["r1m"]  # already in percent
    A = pd.DataFrame(cols).sort_index()
    A = A[A[[f"{i}_dix5" for i in IDX]].notna().all(axis=1)].copy()

    for i in IDX:
        A[i + "_dec"] = pd.qcut(A[i + "_dix5"], 10, labels=False, duplicates="drop") + 1
        A[i + "_z"] = np.where(A[i + "_dec"] <= 3, "Low",
                               np.where(A[i + "_dec"] >= 8, "High", "Mid"))
    return A


def regime_stats(A, mask, label):
    sub = A[mask]
    out = {"regime": label, "n_days": int(len(sub))}
    for i in IDX:
        r = sub[i + "_r1m"].dropna()
        out[i + "_mean"] = round(float(r.mean()), 2) if len(r) else np.nan
        out[i + "_med"] = round(float(r.median()), 2) if len(r) else np.nan
        out[i + "_hit"] = round(float((r > 0).mean() * 100), 0) if len(r) else np.nan
    out["ret_n"] = int(sub[[f"{i}_r1m" for i in IDX]].notna().all(axis=1).sum())
    return out


def all_regimes(A):
    rows = []
    for sn in ("Low", "Mid", "High"):
        for ss in ("Low", "Mid", "High"):
            for si in ("Low", "Mid", "High"):
                m = (A["NDX_z"] == sn) & (A["SPX_z"] == ss) & (A["IWM_z"] == si)
                if m.sum() == 0:
                    continue
                rows.append(regime_stats(A, m, f"N={sn},S={ss},I={si}"))
    return pd.DataFrame(rows).sort_values("n_days", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="docs/index.html",
                    help="built dashboard HTML carrying the payload (default docs/index.html)")
    ap.add_argument("--csv", default=None, help="write the 27-regime table to this CSV")
    ap.add_argument("--aligned-csv", default=None, help="write the aligned per-day frame here")
    args = ap.parse_args()

    P = load_payload(args.html)
    A = build_aligned(P)
    if args.aligned_csv:
        A.to_csv(args.aligned_csv)

    print(f"Payload generated: {P.get('generated')}")
    print(f"Common DIX dates : {len(A)}  "
          f"[{A.index.min().date()} -> {A.index.max().date()}]")
    print("Deciles per index taken over this common sample; "
          "Low=dec1-3, Mid=dec4-7, High=dec8-10.\n")

    base = regime_stats(A, pd.Series(True, index=A.index), "BASELINE (all days)")
    example = regime_stats(A, (A["SPX_z"] == "Low") & (A["NDX_z"] == "Low")
                           & (A["IWM_z"] == "High"), "SPX Low, NDX Low, IWM High")

    def line(s):
        return (f"  n_days={s['n_days']:>4} (ret n={s['ret_n']:>4})   "
                f"NDX {s['NDX_mean']:>+6.2f}% (med {s['NDX_med']:>+5.2f}, hit {s['NDX_hit']:>3.0f}%)   "
                f"SPX {s['SPX_mean']:>+6.2f}% (med {s['SPX_med']:>+5.2f}, hit {s['SPX_hit']:>3.0f}%)   "
                f"IWM {s['IWM_mean']:>+6.2f}% (med {s['IWM_med']:>+5.2f}, hit {s['IWM_hit']:>3.0f}%)")

    print("=== BASELINE ===")
    print(line(base))
    print("\n=== REQUESTED DIVERGENCE: SPX & NDX DIX Low, IWM DIX High ===")
    print(line(example))

    reg = all_regimes(A)
    if args.csv:
        reg.to_csv(args.csv, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== ALL 27 COMOVEMENT REGIMES (1-month forward return %, by frequency) ===")
    print(reg.to_string(index=False))


if __name__ == "__main__":
    main()
