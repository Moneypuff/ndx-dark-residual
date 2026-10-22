#!/usr/bin/env python3
"""
Plot constant-maturity (30d / 90d) skew history for one symbol from the
optsnap daily chain snapshots.

Uses the vol tracker's own pipeline (recompute_iv -> delta_pillars ->
cm_pillars via rr_frame), so the numbers match the RR/skew card. Skew is
shown as RR25 = 25d call IV - 25d put IV and the 25d put-minus-ATM wing
(negative RR = put wing bid), in vol points.

    python tools/plot_cm_skew.py --snap-dir optsnap --symbol TLT --out tlt_skew.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_vol_tracker as V  # noqa: E402


def skew_frame(snap_dir, symbol):
    files = sorted(Path(snap_dir).glob("????-??-??.csv.gz"))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d["symbol"] == symbol]
        if not d.empty:
            frames.append(d)
    if not frames:
        raise SystemExit(f"{symbol}: no rows in {snap_dir}")
    df = V.recompute_iv(pd.concat(frames, ignore_index=True))
    rr = V.rr_frame(df)
    rr = rr[rr["symbol"] == symbol].copy()
    rr["date"] = pd.to_datetime(rr["date"])
    for c in V.RR_PILLARS:
        rr[c] *= 100                      # vol points
    rr["rr25"] = rr["iv25c"] - rr["iv25p"]
    rr["rr10"] = rr["iv10c"] - rr["iv10p"]
    rr["put25_wing"] = rr["iv25p"] - rr["iv50"]
    return rr


def plot(rr, symbol, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    style = {30: ("#1f77b4", "30d"), 90: ("#d62728", "90d")}
    panels = [("rr25", "RR25 = 25Δ call IV − 25Δ put IV (vol pts)"),
              ("put25_wing", "25Δ put IV − ATM IV (vol pts)"),
              ("iv50", "ATM (50Δ) IV (vol pts)")]
    for ax, (col, title) in zip(axes, panels):
        for tenor, (color, label) in style.items():
            s = rr[rr["tenor"] == tenor].set_index("date")[col].dropna()
            ax.plot(s.index, s.values, color=color, lw=1.4, label=label)
        if col == "rr25":
            ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
    fig.suptitle(f"{symbol} constant-maturity skew — 30d vs 90d", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", required=True)
    ap.add_argument("--symbol", default="TLT")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None, help="also write the series here")
    a = ap.parse_args()
    rr = skew_frame(a.snap_dir, a.symbol)
    if a.csv:
        rr.to_csv(a.csv, index=False)
    plot(rr, a.symbol, a.out or f"{a.symbol.lower()}_cm_skew.png")
    print(rr.groupby("tenor")[["rr25", "put25_wing", "iv50"]].describe().T.round(2))


if __name__ == "__main__":
    main()
