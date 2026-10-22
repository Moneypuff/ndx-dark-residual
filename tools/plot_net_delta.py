#!/usr/bin/env python3
"""
Plot the open-interest net delta of one symbol's listed options, per day.

Per contract: delta (our own Black-Scholes, r=0, from recompute_iv's IV) x
OI x 100 shares. Calls are positive, puts negative, so the sum is the
share-equivalent exposure of the open-interest book as if all of it were
held long by customers; dealers hold the opposite. Contracts with no OI or
no solvable IV/delta are dropped. The snapshots only keep contracts near
the money plus the top-OI strikes, so deep-wing OI is under-counted.

    python tools/plot_net_delta.py --snap-dir optsnap --symbol TLT --out tlt_net_delta.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_vol_tracker as V  # noqa: E402
import trade_structures as T  # noqa: E402


def net_delta_frame(snap_dir, symbol):
    frames = []
    for f in sorted(Path(snap_dir).glob("????-??-??.csv.gz")):
        d = pd.read_csv(f)
        d = d[d["symbol"] == symbol]
        if not d.empty:
            frames.append(d)
    if not frames:
        raise SystemExit(f"{symbol}: no rows in {snap_dir}")
    df = V.recompute_iv(pd.concat(frames, ignore_index=True))
    dte = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"])).dt.days
    df = df[(dte > 0) & (df["oi"] > 0)].copy()
    dte = dte.loc[df.index]
    df["delta"] = T.bs_delta(df["spot"], df["strike"], dte / 365.25,
                             df["iv"], df["right"])
    df = df[np.isfinite(df["delta"])]
    df["sh"] = df["delta"] * df["oi"] * 100
    g = df.groupby("date")
    out = pd.DataFrame({
        "spot": g["spot"].first(),
        "call_delta": df[df["right"] == "C"].groupby("date")["sh"].sum(),
        "put_delta": df[df["right"] == "P"].groupby("date")["sh"].sum(),
    }).fillna(0.0)
    out["net_delta"] = out["call_delta"] + out["put_delta"]
    out["net_delta_usd"] = out["net_delta"] * out["spot"]
    out.index = pd.to_datetime(out.index)
    return out


def plot(nd, symbol, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.2]})
    ax = axes[0]
    ax.plot(nd.index, nd["net_delta"] / 1e6, color="#1f77b4", lw=1.6)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Net delta of OI (calls + puts), millions of shares", loc="left", fontsize=10)
    ax = axes[1]
    ax.plot(nd.index, nd["call_delta"] / 1e6, color="#2ca02c", lw=1.3, label="calls")
    ax.plot(nd.index, nd["put_delta"] / 1e6, color="#d62728", lw=1.3, label="puts")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Call and put delta, millions of shares", loc="left", fontsize=10)
    ax.legend(loc="upper left", frameon=False)
    ax = axes[2]
    ax.plot(nd.index, nd["spot"], color="black", lw=1.3)
    ax.set_title(f"{symbol} spot", loc="left", fontsize=10)
    for a in axes:
        a.grid(alpha=0.25)
    fig.suptitle(f"{symbol} open-interest net delta (long-customer convention)", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", required=True)
    ap.add_argument("--symbol", default="TLT")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    nd = net_delta_frame(a.snap_dir, a.symbol)
    if a.csv:
        nd.to_csv(a.csv)
    plot(nd, a.symbol, a.out or f"{a.symbol.lower()}_net_delta.png")
    print(nd.tail(8).round(0).to_string())


if __name__ == "__main__":
    main()
