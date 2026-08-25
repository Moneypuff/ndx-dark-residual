#!/usr/bin/env python3
"""
One-time delta-pillar risk-reversal backfill from the local ORATS duckdb.

The vol tracker's RR/skew study (VOL_TRACKER.md "Risk reversals & skew")
charts constant-maturity delta-pillar IVs. The nightly build derives them
from the daily chain snapshots, whose history only reaches back to the
first capture; this LOCAL-ONLY tool computes the same pillars from an
ORATS options-history duckdb (build_orats_duckdb.py; ~20 years of EOD
quotes) and writes the deep history the runners can't compute themselves.

Data channel: the output file is pushed to the `optsnap-data` branch as
`optsnap/rr_history.csv.gz`, which refresh.yml already fetches alongside
the snapshots; build_vol_tracker.load_rr_backfill picks it up from the
snap dir and merge_rr splices it in strictly before the first
snapshot-derived date (the live pipeline wins at the seam).

Methodology is deliberately IDENTICAL to the live path: our own implied
vol (trade_structures.implied_vol) inverted from the mid of a two-sided
quote -- ORATS' own cMidIv/pMidIv are never used -- then our own BS delta
and build_vol_tracker's delta_pillars/cm_pillars. One code path, no
vendor seam. The SQL prefilters to expiries that can bracket the 30/90d
tenors (5..150 DTE) and a coarse log-moneyness band wide enough to hold
the 10-delta wings of even a 100-vol name at 150 DTE, which cuts ~110M
rows to a solvable few million per symbol.

    python build_rr_history.py --out rr_history.csv.gz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import build_vol_tracker as V
import trade_structures as T
# duckdb is imported lazily in main() -- it is a LOCAL-ONLY dependency
# (not in requirements) and the CI runners import this module's pure
# functions through the test suite without ever touching the database.

DEFAULT_DB = "E:/selected-full-history-dump/orats.duckdb"
DEFAULT_UNIVERSE = Path(__file__).resolve().parent / "data" / "optsnap_universe.csv"
DTE_LO, DTE_HI = V.RR_MIN_DTE, 150     # only expiries that can bracket 30/90d
LOGM_LO, LOGM_HI = -1.05, 0.60         # coarse ln(K/S) prefilter band


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def long_rows(sym_df, symbol):
    """ORATS wide rows (one row = a strike with c*/p* quote columns) ->
    the long optsnap-shaped OTM frame [date, symbol, expiry, right,
    strike, iv, oi, volume, bid, ask, last, spot] that delta_pillars
    consumes -- puts strictly below spot, calls at/above, iv left NaN for
    solve_ivs. ORATS quotes a 0.0 bid on dead lines; those survive here
    (delta_pillars' liveness check needs bid>0 OR oi>0, same as live)."""
    if sym_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "expiry", "right",
                                     "strike", "iv", "oi", "volume", "bid",
                                     "ask", "last", "spot"])
    out = []
    for right, p in (("C", "c"), ("P", "p")):
        side = sym_df[(sym_df["strike"] >= sym_df["stkPx"]) if right == "C"
                      else (sym_df["strike"] < sym_df["stkPx"])]
        out.append(pd.DataFrame({
            "date": side["trade_date"].astype(str).str[:10],
            "symbol": symbol,
            "expiry": side["expirDate"].astype(str).str[:10],
            "right": right,
            "strike": side["strike"].astype(float),
            "iv": np.nan,
            "oi": side[f"{p}Oi"],
            "volume": side[f"{p}Volu"],
            "bid": side[f"{p}BidPx"].astype(float),
            "ask": side[f"{p}AskPx"].astype(float),
            "last": np.nan,
            "spot": side["stkPx"].astype(float),
        }))
    return pd.concat(out, ignore_index=True)


def solve_ivs(df):
    """iv from OUR solver -- mid of a two-sided quote -> implied_vol,
    exactly the price recompute_iv would use on a live snapshot (there is
    no `last` in the ORATS extract, so one-sided/dead quotes stay NaN)."""
    if df.empty:
        return df
    bid = df["bid"].to_numpy(dtype=float)
    ask = df["ask"].to_numpy(dtype=float)
    two_sided = (bid > 0) & (ask >= bid) & (ask > 0)
    mid = np.where(two_sided, (bid + ask) / 2.0, np.nan)
    t_years = ((pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"]))
              .dt.days.to_numpy(dtype=float)) / 365.25
    df = df.copy()
    df["iv"] = T.implied_vol(mid, df["spot"].to_numpy(dtype=float),
                             df["strike"].to_numpy(dtype=float),
                             t_years, df["right"].to_numpy())
    return df


# ----------------------------------------------------------------------------
# duckdb + disk
# ----------------------------------------------------------------------------
def fetch_symbol(con, symbol, start=None, end=None):
    # moneyness band phrased multiplicatively -- ln() would fault on the
    # zero strikes/spots that sneak into the raw data (SQL WHERE clauses
    # have no evaluation-order guarantee, so a `stkPx > 0` guard doesn't
    # protect a ln() in a sibling condition)
    conds = ["ticker = ?",
             f"datediff('day', trade_date, expirDate) BETWEEN {DTE_LO} AND {DTE_HI}",
             "stkPx > 0", "strike > 0",
             f"strike >= stkPx * exp({LOGM_LO})",
             f"strike <= stkPx * exp({LOGM_HI})"]
    params = [symbol]
    if start:
        conds.append("trade_date >= ?")
        params.append(start)
    if end:
        conds.append("trade_date <= ?")
        params.append(end)
    return con.sql(
        "SELECT trade_date, expirDate, strike, stkPx, "
        "cBidPx, cAskPx, cOi, cVolu, pBidPx, pAskPx, pOi, pVolu "
        f"FROM options WHERE {' AND '.join(conds)}",
        params=params).df()


def build_symbol(con, symbol, start=None, end=None):
    wide = fetch_symbol(con, symbol, start, end)
    long_df = solve_ivs(long_rows(wide, symbol))
    return V.rr_frame(long_df)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help="path to orats.duckdb")
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE),
                    help="CSV with a `symbol` column")
    ap.add_argument("--out", default="rr_history.csv.gz")
    ap.add_argument("--symbols", default=None,
                    help="comma list overriding the universe file (resume)")
    ap.add_argument("--start", default=None, help="first trade_date, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="last trade_date, YYYY-MM-DD")
    ap.add_argument("--partials-dir", default="rr_partials",
                    help="per-symbol partial CSVs (existing ones are reused)")
    args = ap.parse_args()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(pd.read_csv(args.universe)["symbol"].str.upper())

    import duckdb
    partials = Path(args.partials_dir)
    partials.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)
    try:
        for sym in symbols:
            part = partials / f"{sym}.csv"
            if part.exists():
                print(f"  {sym}: partial exists, skipping")
                continue
            rr = build_symbol(con, sym, args.start, args.end)
            rr.round(4).to_csv(part, index=False)
            print(f"  {sym}: {len(rr)} tenor-days "
                  f"({rr['date'].min() if len(rr) else '-'} .. "
                  f"{rr['date'].max() if len(rr) else '-'})")
    finally:
        con.close()

    frames = [pd.read_csv(partials / f"{s}.csv") for s in symbols
              if (partials / f"{s}.csv").exists()]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = out.sort_values(["symbol", "tenor", "date"], ignore_index=True)
    out.to_csv(args.out, index=False, compression="gzip")
    print(f"wrote {args.out}  ({len(out)} rows, "
          f"{Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
