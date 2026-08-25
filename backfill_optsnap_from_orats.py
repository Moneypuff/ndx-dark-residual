#!/usr/bin/env python3
"""
Backfill optsnap/*.csv.gz history from the local ORATS duckdb, for dates
before live Yahoo capture (snapshot_option_chains.py) started.

Yahoo has no chain history, so the optsnap-data branch only goes back to
whenever the nightly workflow first ran (VOL_TRACKER.md's documented "no
backfill" gap). The ORATS duckdb built by build_orats_duckdb.py (see
orats_duckdb.py) carries real EOD quotes/IV/OI for this same symbol
universe for years -- not Yahoo, but the same fields the tracker consumes,
so it can fill that gap.

Reuses the live capture's own policy functions (select_expiries,
filter_contracts, contract_rows from snapshot_option_chains.py) so a
backfilled day and a live day are policy-identical and load_snapshots()
can concatenate them with no special-casing. Each ORATS row already has
call and put quotes side by side; _side_contracts reshapes one side into
the {strike, openInterest, bid, ask, impliedVolatility, volume, lastPrice}
dicts that filter_contracts/contract_rows expect from a Yahoo chain.

Only cMidIv/pMidIv (per-contract quoted vol, matching Yahoo's
impliedVolatility) are used for iv -- not smoothSmvVol -- since deep
ITM/OTM legs carry near-meaningless quoted IV and the pipeline's own
despike_smile step exists to handle noisy wings; smile_points also only
ever reads the OTM side of each expiry, so the noisiest (ITM) leg's iv is
never touched downstream. A quoted iv of exactly 0.0 is ORATS' sentinel
for "no valid two-sided quote", not a real 0% vol, so it is treated the
same as a null.

    python backfill_optsnap_from_orats.py --out optsnap --start 2025-08-10 --end 2026-08-09
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import snapshot_option_chains as S

DEFAULT_DB = "E:/selected-full-history-dump/orats.duckdb"
DEFAULT_UNIVERSE = Path(__file__).resolve().parent / "data" / "optsnap_universe.csv"
ORATS_COLUMNS = ["ticker", "expirDate", "strike", "stkPx",
                 "cMidIv", "cOi", "cBidPx", "cAskPx", "cVolu",
                 "pMidIv", "pOi", "pBidPx", "pAskPx", "pVolu"]


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def _side_contracts(expiry_df, right):
    """ORATS rows for one (symbol, date, expiry) -> the list of per-contract
    dicts filter_contracts/contract_rows expect from a Yahoo chain side
    ('C' or 'P'). A quoted iv of 0.0 is ORATS' no-quote sentinel, not a
    real vol -- dropped to None same as a null."""
    p = "c" if right == "C" else "p"
    iv_col, oi_col, bid_col, ask_col, volu_col = (
        f"{p}MidIv", f"{p}Oi", f"{p}BidPx", f"{p}AskPx", f"{p}Volu")
    out = []
    for row in expiry_df.itertuples(index=False):
        bid = getattr(row, bid_col)
        ask = getattr(row, ask_col)
        iv = getattr(row, iv_col)
        oi = getattr(row, oi_col)
        last = ((bid + ask) / 2
                if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid
                else None)
        out.append({
            "strike": float(getattr(row, "strike")),
            "openInterest": int(oi) if pd.notna(oi) else 0,
            "bid": float(bid) if pd.notna(bid) else 0.0,
            "ask": float(ask) if pd.notna(ask) else None,
            "impliedVolatility": float(iv) if pd.notna(iv) and iv > 0 else None,
            "volume": getattr(row, volu_col),
            "lastPrice": last,
        })
    return out


def build_day_rows(day_df, symbol, trade_date):
    """One ORATS trading day's rows for one symbol (columns: expirDate,
    strike, stkPx, cMidIv, cOi, cBidPx, cAskPx, cVolu, pMidIv, pOi, pBidPx,
    pAskPx, pVolu) -> optsnap-schema rows, applying the live capture's own
    expiry and contract policy as if `trade_date` were the capture day.
    Empty list when the symbol has no rows or no usable spot that day."""
    if day_df.empty:
        return []
    spot = float(day_df["stkPx"].iloc[0])
    if not spot:
        return []
    snap_date = str(pd.Timestamp(trade_date).date())
    expiries = sorted({pd.Timestamp(e).timestamp() for e in day_df["expirDate"].unique()})
    kept = S.select_expiries(expiries, trade_date)
    rows = []
    exp_dates = day_df["expirDate"].dt.date if hasattr(day_df["expirDate"], "dt") \
        else day_df["expirDate"]
    for e in kept:
        exp_date = pd.Timestamp(e, unit="s").date()
        g = day_df[exp_dates == exp_date]
        if g.empty:
            continue
        chain = {"calls": _side_contracts(g, "C"), "puts": _side_contracts(g, "P")}
        rows.extend(S.contract_rows(chain, symbol, spot, e, snap_date))
    return rows


# ----------------------------------------------------------------------------
# Backfill (duckdb + disk)
# ----------------------------------------------------------------------------
def trading_dates(con, symbols, start, end):
    q = con.sql(
        "SELECT DISTINCT trade_date FROM options "
        "WHERE ticker IN ({}) AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date".format(",".join("?" * len(symbols))),
        params=[*symbols, start, end],
    ).df()
    return list(q["trade_date"])


def backfill_day(con, symbols, trade_date):
    """One day, every symbol -> the day's optsnap DataFrame (possibly
    empty if nothing survives the policy filters)."""
    day = con.sql(
        "SELECT ticker, {} FROM options WHERE ticker IN ({}) AND trade_date = ?"
        .format(", ".join(ORATS_COLUMNS[1:]), ",".join("?" * len(symbols))),
        params=[*symbols, trade_date],
    ).df()
    all_rows = []
    for sym, g in day.groupby("ticker"):
        all_rows.extend(build_day_rows(g, sym, trade_date))
    return pd.DataFrame(all_rows, columns=["date", "symbol", "expiry", "right",
                                            "strike", "iv", "oi", "volume",
                                            "bid", "ask", "last", "spot"])


def backfill(db_path, symbols, start, end, out_dir, skip_existing=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)
    try:
        dates = trading_dates(con, symbols, start, end)
        print(f"{len(dates)} trading days in [{start}, {end}] for {len(symbols)} symbols")
        for d in dates:
            day_path = out / f"{pd.Timestamp(d).date()}.csv.gz"
            if skip_existing and day_path.exists():
                print(f"  skip {day_path.name} (already exists)")
                continue
            df = backfill_day(con, symbols, d)
            df.to_csv(day_path, index=False, compression="gzip")
            print(f"  wrote {day_path.name}  ({len(df)} rows, "
                  f"{day_path.stat().st_size // 1024} KB)")
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="optsnap", help="snapshot output directory")
    ap.add_argument("--db", default=DEFAULT_DB, help="path to orats.duckdb")
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE),
                    help="CSV with a `symbol` column")
    ap.add_argument("--start", required=True, help="first trade_date, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last trade_date, YYYY-MM-DD")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="overwrite a day's file even if it already exists in --out")
    args = ap.parse_args()
    symbols = list(pd.read_csv(args.universe)["symbol"].str.upper())
    backfill(args.db, symbols, args.start, args.end, args.out,
             skip_existing=not args.no_skip_existing)


if __name__ == "__main__":
    main()
