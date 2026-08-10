#!/usr/bin/env python3
"""
One intraday capture cycle for the vol tracker (interim, session-driven --
see VOL_TRACKER.md; the scheduled CI capture takes over once merged).

Captures the full chain snapshot, stamps it with the ET time, sanity-checks
it (pre/post-market junk is refused), and distills the small per-symbol x
tenor board (spot, ATM IV, 25-delta rr, volumes, OI) that gets mirrored to
Google Drive. Raw files are committed to the data branch by the calling
agent/workflow -- this script only produces them.

    python intraday_capture.py --raw-dir optsnap-intraday --distill-out board.csv
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_vol_tracker import expiry_stats, sane_day
from snapshot_option_chains import DEFAULT_UNIVERSE, snapshot

TENORS = (30, 60, 91, 182)


def et_stamp():
    return subprocess.run(["date", "+%Y-%m-%d_%H%M"],
                          env={**os.environ, "TZ": "America/New_York"},
                          capture_output=True, text=True).stdout.strip()


def distill(df, stamp):
    """Per symbol x tenor board rows from one snapshot frame."""
    rows = []
    day = df["date"].iloc[0] if len(df) else None
    for sym in sorted(df["symbol"].unique()):
        for tgt in TENORS:
            st = expiry_stats(df, sym, tgt, day)
            if not st:
                continue
            e = df[(df["symbol"] == sym) & (df["expiry"] == st["expiry"])]
            rows.append({
                "stamp_et": stamp, "symbol": sym, "tenor_d": tgt,
                "expiry": st["expiry"], "dte": st["dte"],
                "spot": round(st["spot"], 2),
                "atm_iv": round(st["atm_iv"] * 100, 2) if np.isfinite(st["atm_iv"]) else None,
                "rr25": round(st["rr"], 2) if np.isfinite(st["rr"]) else None,
                "call_vol": int(e[e["right"] == "C"]["volume"].fillna(0).sum()),
                "put_vol": int(e[e["right"] == "P"]["volume"].fillna(0).sum()),
                "call_oi": int(e[e["right"] == "C"]["oi"].fillna(0).sum()),
                "put_oi": int(e[e["right"] == "P"]["oi"].fillna(0).sum()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="optsnap-intraday")
    ap.add_argument("--distill-out", default=None)
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    args = ap.parse_args()

    stamp = et_stamp()
    symbols = list(pd.read_csv(args.universe)["symbol"].str.upper())
    df, status = snapshot(symbols, args.raw_dir)
    if not sane_day(df):
        print(f"NOT SANE at {stamp} ET -- market closed or junk feed; "
              "raw file removed, nothing to commit.")
        for f in Path(args.raw_dir).glob(f"{df['date'].iloc[0]}*.csv.gz"):
            f.unlink()
        sys.exit(3)

    src = Path(args.raw_dir) / f"{df['date'].iloc[0]}.csv.gz"
    dst = Path(args.raw_dir) / f"{stamp}ET.csv.gz"
    src.rename(dst)
    print(f"raw: {dst}  ({dst.stat().st_size // 1024} KB)")

    if args.distill_out:
        board = distill(df, stamp)
        board.to_csv(args.distill_out, index=False)
        print(f"distilled: {args.distill_out}  ({len(board)} rows, "
              f"{Path(args.distill_out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
