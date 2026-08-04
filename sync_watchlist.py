#!/usr/bin/env python3
"""
Sync data/watchlist.csv from a published Google Sheet (or any public CSV URL).
=============================================================================

The dashboard's "Watchlist" small-multiples tab is driven by `data/watchlist.csv`
(column 1 = ticker, optional column 2 = sector). That CSV is the source of truth the
build reads -- no data is fetched when the page is viewed; the CSV only decides WHICH
names appear. This helper lets you keep the list in a Google Sheet and pull it into the
CSV, so the sheet becomes the place you edit and a rebuild picks up the change.

One-time setup for the sheet
----------------------------
  1. Put your tickers in column A (one per row), optional sector in column B.
     A header row named `ticker` / `symbol` in A1 is fine -- it's skipped.
  2. File -> Share -> Publish to web -> pick the sheet/tab -> "Comma-separated values (.csv)".
  3. Copy the published URL. It looks like:
       https://docs.google.com/spreadsheets/d/e/<long-id>/pub?gid=0&single=true&output=csv

Usage
-----
  python sync_watchlist.py --url "<published csv url>"          # writes data/watchlist.csv
  python sync_watchlist.py --url "$WATCHLIST_SHEET_CSV_URL"     # from an env var

No credentials are used -- the sheet must be publicly published (matching this repo's
no-API-key design). If the fetch fails or returns no tickers, the existing CSV is left
untouched so a bad export can never blank out the watchlist.
"""
import argparse
import io
import sys
from pathlib import Path

import requests

# Reuse the build's own parser so "valid" here means exactly what the build accepts.
from ndx_dark_residual import WATCHLIST_CSV, load_watchlist


def fetch_csv(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True,
                    help="Published Google Sheet CSV URL (Publish to web -> CSV), or any public "
                         "CSV whose first column is the ticker and optional second is the sector.")
    ap.add_argument("--out", default=WATCHLIST_CSV,
                    help=f"Destination CSV the build reads. Default: {WATCHLIST_CSV}")
    args = ap.parse_args()

    try:
        text = fetch_csv(args.url)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"! could not fetch the sheet ({e}); leaving {args.out} unchanged")

    # Validate by round-tripping through the build's loader on a temp copy in memory: write to a
    # sibling .tmp, parse it, and only promote it if it yields at least one ticker.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tickers, sectors = load_watchlist(str(tmp))
    if not tickers:
        tmp.unlink(missing_ok=True)
        sys.exit(f"! sheet parsed to zero tickers; leaving {args.out} unchanged")

    # Re-serialize canonically (ticker,sector) so the committed CSV is clean regardless of the
    # sheet's exact column layout / stray columns.
    buf = io.StringIO()
    buf.write("ticker,sector\n")
    for t in tickers:
        buf.write(f"{t},{sectors.get(t, '')}\n")
    out.write_text(buf.getvalue(), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    print(f"Wrote {out} with {len(tickers)} tickers: {', '.join(tickers)}", file=sys.stderr)


if __name__ == "__main__":
    main()
