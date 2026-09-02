#!/usr/bin/env python3
"""
Resumable Yahoo equity-price fetch into the durable equity store (equity_store.py).

Fetches daily close / adj-close / volume for a set of tickers and upserts them into
`equity_prices.duckdb` (a store kept SEPARATE from the ORATS options duckdb), so the
study reads prices from disk and stops re-querying Yahoo -- which is what triggers the
HTTP 429 rate limiting.

Default target: the NDX-100 members that LEFT the index in-window (the price-only
baseline peers wired in ndx_dark_residual.fetch_exited_price_panels). Pass --tickers to
override.

Rate-limit aware and RESUMABLE:
  * symbols already covered in the store (history reaching within --stale-days of the
    end) are skipped, so a re-run only fetches what is missing;
  * progress is upserted incrementally, so an interrupted run is never lost;
  * on a hard Yahoo 429 that will not clear within the run, the still-pending tickers are
    written to a resume-state file and the process exits 75 (EX_TEMPFAIL) -- re-run it (or
    let the scheduled wake re-run it) once the limit wears off and it picks up where it
    stopped.

Exit codes: 0 = complete (nothing pending), 75 = rate-limited, resume later.
"""

import argparse
import json
import os
import sys
import time
import datetime as dt

import pandas as pd

import ndx_dark_residual as N
import equity_store as ES

STATE_FILENAME = "equity_fetch_state.json"
RESUME_EXIT = 75   # EX_TEMPFAIL -- "try again later"


def _state_path(cache_dir):
    return os.path.join(cache_dir or ES.DEFAULT_CACHE_DIR, STATE_FILENAME)


def default_targets():
    """NDX-100 members that left the index in-window (price-only baseline peers)."""
    import csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "ndx_historical_membership.csv")
    current = [r["ticker"] for r in csv.DictReader(open(path, newline="")) if not r["removed"]]
    return sorted(N.ndx_exited_members(current))


def already_covered(cov, ticker, end, stale_days):
    """True if the store already holds this ticker with history reaching near `end`."""
    if ticker not in cov:
        return False
    _lo, hi, _n = cov[ticker]
    hi = pd.Timestamp(hi)
    return (pd.Timestamp(end).normalize() - hi).days <= stale_days


def _overlaps_window(series, spans, min_rows=5):
    """True if `series` has >= min_rows non-null observations inside any membership stint --
    guards against a recycled/renamed ticker returning wrong-era data (e.g. a reused 'FB')."""
    if not spans:
        return True
    for a, r in spans:
        win = series.loc[(series.index >= a) & ((series.index < r) if r is not None else True)]
        if win.notna().sum() >= min_rows:
            return True
    return False


def run(tickers, start, end, cache_dir, stale_days=7, pause=1.0,
        ratelimit_backoffs=(30, 60, 120), batch=10, member_spans=None):
    cov = ES.coverage(cache_dir=cache_dir)
    pending = [t for t in tickers if not already_covered(cov, t, end, stale_days)]
    skipped = len(tickers) - len(pending)
    print(f"targets {len(tickers)} | already in store {skipped} | to fetch {len(pending)}",
          file=sys.stderr)

    done, buf, rate_limited = [], {"close": {}, "adjclose": {}, "volume": {}}, False

    def flush():
        panels = {f: pd.DataFrame(buf[f]) for f in buf if buf[f]}
        if panels:
            ES.upsert_panels(panels, cache_dir=cache_dir, mirror=False)
        for f in buf:
            buf[f] = {}

    for i, t in enumerate(pending):
        got = None
        for attempt, wait in enumerate((0,) + tuple(ratelimit_backoffs)):
            if wait:
                print(f"  429 on {t}; backing off {wait}s (attempt {attempt})", file=sys.stderr)
                time.sleep(wait)
            try:
                got = N.fetch_yahoo_one(t, start, end, raise_on_ratelimit=True)
                break
            except N.YahooRateLimited:
                got = None
                continue
        if got is None:                        # still rate-limited after all backoffs
            rate_limited = True
            break
        spans = (member_spans or {}).get(t)
        if not got.empty and _overlaps_window(got.get("adjclose", got.get("close")), spans):
            for f in ("close", "adjclose", "volume"):
                if f in got.columns:
                    buf[f][t] = got[f]
            print(f"  [{i+1}/{len(pending)}] {t}: {len(got)} rows", file=sys.stderr)
        elif not got.empty:
            print(f"  [{i+1}/{len(pending)}] {t}: {len(got)} rows but NO membership-window overlap "
                  f"(recycled/renamed ticker) -- not stored", file=sys.stderr)
            done.append(t); time.sleep(pause); continue
        else:
            print(f"  [{i+1}/{len(pending)}] {t}: no data (delisted/renamed) -- skipping",
                  file=sys.stderr)
        done.append(t)
        if len(done) % batch == 0:
            flush()
        time.sleep(pause)

    flush()
    still_pending = [t for t in pending if t not in done]
    state_path = _state_path(cache_dir)
    if still_pending:
        with open(state_path, "w") as fh:
            json.dump({"pending": still_pending, "start": str(start), "end": str(end),
                       "updated": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "rate_limited": rate_limited}, fh, indent=2)
        print(f"\n{len(done)} fetched this run; {len(still_pending)} pending -> {state_path}",
              file=sys.stderr)
        return RESUME_EXIT if rate_limited else 1
    # complete: refresh the durable parquet mirror and clear the resume state
    ES.export_parquet(cache_dir=cache_dir)
    if os.path.exists(state_path):
        os.remove(state_path)
    print(f"\ncomplete: {len(done)} tickers fetched; store mirrored to {ES.PARQUET_MIRROR}",
          file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default="",
                    help="comma-separated tickers to fetch (default: NDX exited members)")
    ap.add_argument("--start", default=str(N.FINRA_MIN_DATE.date()))
    ap.add_argument("--end", default=str(pd.Timestamp.today().normalize().date()))
    ap.add_argument("--cache-dir", default=ES.DEFAULT_CACHE_DIR)
    ap.add_argument("--stale-days", type=int, default=7,
                    help="a ticker whose stored history reaches within this many days of --end "
                         "is considered covered and skipped")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between symbols")
    ap.add_argument("--resume", action="store_true",
                    help="fetch only the tickers left pending by a prior rate-limited run")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.resume:
        sp = _state_path(args.cache_dir)
        if not os.path.exists(sp):
            print("no resume state -- nothing pending.", file=sys.stderr)
            return 0
        tickers = json.load(open(sp)).get("pending", [])
    else:
        tickers = default_targets()

    if not tickers:
        print("no target tickers.", file=sys.stderr)
        return 0

    import csv
    _mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ndx_historical_membership.csv")
    _cur = [r["ticker"] for r in csv.DictReader(open(_mpath, newline="")) if not r["removed"]]
    member_spans = N.ndx_exited_members(_cur)
    rc = run(tickers, start, end, args.cache_dir, stale_days=args.stale_days, pause=args.pause,
             member_spans=member_spans)
    if rc == RESUME_EXIT:
        print("RATE-LIMITED: re-run with --resume once Yahoo's limit clears.", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
