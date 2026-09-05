#!/usr/bin/env python3
"""
Durable equity-price store (Yahoo daily close / adj-close / volume).

A DEDICATED store, deliberately kept SEPARATE from the ORATS options duckdb
(build_orats_duckdb.py / orats.duckdb): equity EOD prices and options quotes
never share a file here. Its whole reason to exist is so that once a symbol's
history has been pulled from Yahoo it is persisted and reused, and later runs
never re-query Yahoo for data already on disk -- which is what was getting the
pipeline rate-limited (HTTP 429).

Backing store: a duckdb file (default `<cache>/equity_prices.duckdb`) with one
table

    equity_eod(ticker VARCHAR, date DATE, close DOUBLE, adj_close DOUBLE,
               volume DOUBLE, PRIMARY KEY (ticker, date))

plus a committed columnar mirror `data/equity_prices.parquet` so the store is
durable across ephemeral sessions and importable into any database with a single
`read_parquet(...)`. duckdb is a LOCAL-ONLY optional dependency (as it already is
for the ORATS scripts); nothing in the nightly dashboard pipeline needs it -- the
read/write-through in ndx_dark_residual.load_yahoo_panels imports it lazily and
no-ops if it is missing, so the store only ever accelerates, never gates.

Refusing to touch orats.duckdb is enforced: opening a path whose name looks like
the ORATS database raises.
"""

import os
import sys
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = str(Path.home() / ".ndx_dark_cache")
STORE_FILENAME = "equity_prices.duckdb"
PARQUET_MIRROR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "equity_prices.parquet")
FIELDS = ("close", "adjclose", "volume")   # wide-panel field names used by the pipeline
_COLMAP = {"close": "close", "adjclose": "adj_close", "volume": "volume"}


def default_store_path(cache_dir=DEFAULT_CACHE_DIR):
    return os.path.join(cache_dir or DEFAULT_CACHE_DIR, STORE_FILENAME)


def _guard_not_orats(path):
    base = os.path.basename(str(path)).lower()
    if "orats" in base:
        raise ValueError(f"refusing to use the ORATS options database as an equity store: {path}")


def _connect(path):
    """Open (creating if needed) the equity duckdb and ensure the schema. Lazy duckdb import."""
    _guard_not_orats(path)
    try:
        import duckdb
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError("duckdb is required for the equity store (pip install duckdb)") from e
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS equity_eod ("
        "  ticker VARCHAR NOT NULL, date DATE NOT NULL,"
        "  close DOUBLE, adj_close DOUBLE, volume DOUBLE,"
        "  PRIMARY KEY (ticker, date))")
    return con


def _long_from_panels(panels):
    """{'close','adjclose','volume': wide df} -> long df[ticker,date,close,adj_close,volume]."""
    frames = []
    for field in FIELDS:
        df = panels.get(field)
        if df is None or df.empty:
            continue
        s = df.stack().dropna().rename(_COLMAP[field])
        s.index = s.index.set_names(["date", "ticker"])
        frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "adj_close", "volume"])
    long = pd.concat(frames, axis=1).reset_index()
    long["date"] = pd.to_datetime(long["date"]).dt.date
    for c in ("close", "adj_close", "volume"):
        if c not in long.columns:
            long[c] = pd.NA
    return long[["ticker", "date", "close", "adj_close", "volume"]]


def upsert_panels(panels, path=None, cache_dir=DEFAULT_CACHE_DIR, mirror=True):
    """Merge wide {close,adjclose,volume} panels into the store (upsert on (ticker,date)).
    Returns the number of rows written. No-op (returns 0) if there is nothing to write."""
    path = path or default_store_path(cache_dir)
    long = _long_from_panels(panels)
    long = long.dropna(subset=["close", "adj_close", "volume"], how="all")
    if long.empty:
        return 0
    con = _connect(path)
    try:
        con.register("_incoming", long)
        con.execute(
            "INSERT INTO equity_eod SELECT ticker, date, close, adj_close, volume FROM _incoming "
            "ON CONFLICT (ticker, date) DO UPDATE SET "
            "  close = COALESCE(excluded.close, equity_eod.close),"
            "  adj_close = COALESCE(excluded.adj_close, equity_eod.adj_close),"
            "  volume = COALESCE(excluded.volume, equity_eod.volume)")
        con.unregister("_incoming")
        n = len(long)
    finally:
        con.close()
    if mirror:
        try:
            export_parquet(path=path)
        except Exception as e:  # noqa: BLE001
            print(f"  ! equity store: parquet mirror failed ({e})", file=sys.stderr)
    return n


def _ensure_hydrated(path):
    """If the local duckdb store is absent but the committed parquet mirror exists, hydrate the
    store from it -- so a fresh (ephemeral) session inherits the durable prices and does not
    re-query Yahoo. No-op once the store exists."""
    # Only the default store is seeded from the committed mirror; a custom/temp store path
    # (tests, ad-hoc) stays isolated so it never inherits the repo's committed prices.
    if (os.path.abspath(path) == os.path.abspath(default_store_path())
            and not os.path.exists(path) and os.path.exists(PARQUET_MIRROR)):
        try:
            import_parquet(path=path)
        except Exception as e:  # noqa: BLE001
            print(f"  ! equity store hydrate-from-parquet failed ({e})", file=sys.stderr)


def load_panels(tickers=None, path=None, cache_dir=DEFAULT_CACHE_DIR):
    """Return {'close','adjclose','volume': wide df} for `tickers` (all if None) from the store.
    Returns empty panels if the store does not exist yet."""
    path = path or default_store_path(cache_dir)
    _ensure_hydrated(path)
    if not os.path.exists(path):
        return {f: pd.DataFrame() for f in FIELDS}
    con = _connect(path)
    try:
        if tickers:
            con.register("_want", pd.DataFrame({"ticker": sorted(set(tickers))}))
            long = con.execute(
                "SELECT e.ticker, e.date, e.close, e.adj_close, e.volume FROM equity_eod e "
                "SEMI JOIN _want w ON e.ticker = w.ticker").fetchdf()
            con.unregister("_want")
        else:
            long = con.execute("SELECT ticker, date, close, adj_close, volume FROM equity_eod").fetchdf()
    finally:
        con.close()
    out = {}
    if long.empty:
        return {f: pd.DataFrame() for f in FIELDS}
    long["date"] = pd.to_datetime(long["date"])
    for field, col in _COLMAP.items():
        out[field] = long.pivot(index="date", columns="ticker", values=col).sort_index()
    return out


def coverage(path=None, cache_dir=DEFAULT_CACHE_DIR):
    """{ticker: (min_date, max_date, n_rows)} for what the store already holds."""
    path = path or default_store_path(cache_dir)
    _ensure_hydrated(path)
    if not os.path.exists(path):
        return {}
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT ticker, min(date), max(date), count(*) FROM equity_eod GROUP BY ticker").fetchall()
    finally:
        con.close()
    return {t: (lo, hi, n) for t, lo, hi, n in rows}


def export_parquet(path=None, cache_dir=DEFAULT_CACHE_DIR, parquet=None):
    """Write the whole store to the committed parquet mirror (durable, DB-importable)."""
    parquet = parquet or PARQUET_MIRROR
    path = path or default_store_path(cache_dir)
    if not os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(parquet)) or ".", exist_ok=True)
    con = _connect(path)
    try:
        con.execute(
            "COPY (SELECT ticker, date, close, adj_close, volume FROM equity_eod ORDER BY ticker, date) "
            f"TO '{parquet}' (FORMAT parquet)")
    finally:
        con.close()


def import_parquet(path=None, cache_dir=DEFAULT_CACHE_DIR, parquet=None):
    """Hydrate the duckdb store from the committed parquet mirror (e.g. on a fresh session).
    Upserts, so it is safe to run over an existing store. Returns rows imported."""
    parquet = parquet or PARQUET_MIRROR
    path = path or default_store_path(cache_dir)
    if not os.path.exists(parquet):
        return 0
    con = _connect(path)
    try:
        con.execute(
            "INSERT INTO equity_eod SELECT ticker, date, close, adj_close, volume "
            f"FROM read_parquet('{parquet}') "
            "ON CONFLICT (ticker, date) DO UPDATE SET "
            "  close = COALESCE(excluded.close, equity_eod.close),"
            "  adj_close = COALESCE(excluded.adj_close, equity_eod.adj_close),"
            "  volume = COALESCE(excluded.volume, equity_eod.volume)")
        n = con.execute("SELECT count(*) FROM equity_eod").fetchone()[0]
    finally:
        con.close()
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect / mirror the equity-price store.")
    ap.add_argument("--store", default=default_store_path())
    ap.add_argument("--export", action="store_true", help="write the parquet mirror")
    ap.add_argument("--import-parquet", action="store_true", help="hydrate the store from the mirror")
    args = ap.parse_args()
    if args.import_parquet:
        print(f"imported -> {import_parquet(path=args.store)} rows total")
    if args.export:
        export_parquet(path=args.store)
        print(f"exported store -> {PARQUET_MIRROR}")
    cov = coverage(path=args.store)
    print(f"store {args.store}: {len(cov)} tickers")
    for t in sorted(cov)[:12]:
        lo, hi, n = cov[t]
        print(f"  {t:<7} {lo} .. {hi}  ({n} rows)")
