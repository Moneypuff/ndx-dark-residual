"""
Export the point-in-time, dollar-weighted index DIX history into data/pit_dix_history.csv.

The dashboard splices this file in front of its live computation (see load_pit_dix_history /
splice_dix_history in ndx_dark_residual.py): dates up to the file's last row come from here,
later dates from the nightly live build. GitHub Actions can't rebuild the history itself -- it
needs FINRA Reg SHO files back to 2009 and point-in-time membership -- so it is produced
locally and committed.

Sources (built on the maintainer's machine, 2026-09):
  ndx  D:/Data/ndx_dix_history.parquet  (col ndx_dix_dollar)  Nasdaq-100 PIT membership from
       contemporaneous Wikipedia snapshots; kept current nightly by dix-event-tracker
  spx  D:/Data/spx_dix_history.parquet  (col spx_dix_dollar)  S&P 500 PIT ("SPX PIT" table);
       reproduces SqueezeMetrics' published DIX at 0.994 daily / 0.996 5d MA (2011-2026)
  iwm  D:/Data/rut_dix_history.parquet  (col rut_dix)         Russell 2000 PIT from IWM's SEC
       N-Q (2009-2018) and N-PORT (2019+) holdings
All three: sum(raw price x FINRA short) / sum(raw price x FINRA off-exchange total), FINRA
facility files summed before 2018-08 (identical to the consolidated file where both exist).

Usage:  python tools/export_pit_dix_history.py [--data D:/Data] [--out data/pit_dix_history.csv]
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:/Data")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "pit_dix_history.csv"))
    a = ap.parse_args()
    d = Path(a.data)
    H = pd.DataFrame({
        "ndx": pd.read_parquet(d / "ndx_dix_history.parquet")["ndx_dix_dollar"],
        "spx": pd.read_parquet(d / "spx_dix_history.parquet")["spx_dix_dollar"],
        "iwm": pd.read_parquet(d / "rut_dix_history.parquet")["rut_dix"],
    }).sort_index()
    H.index = pd.to_datetime(H.index).normalize()
    H.index.name = "date"
    H = H[H.notna().any(axis=1)].round(5)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    H.to_csv(a.out, date_format="%Y-%m-%d")
    last = {c: H[c].last_valid_index().date() for c in H}
    print(f"wrote {a.out}: {len(H)} rows, {H.index.min().date()} -> {H.index.max().date()}; last valid {last}")


if __name__ == "__main__":
    main()
