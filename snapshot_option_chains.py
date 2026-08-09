#!/usr/bin/env python3
"""
Daily option-chain snapshot -- the capture leg of the fixed-strike vol
tracker (VOL_TRACKER.md).

Yahoo has no option-chain history: a chain not saved today is gone. This
script pulls the chains for every symbol in data/optsnap_universe.csv and
appends one compressed CSV per trading day to an output directory (the
optsnap.yml workflow points it at the `optsnap-data` branch). Everything
downstream -- fixed-strike IV series, dOI x dIV aggressor classification,
the post-signal trade monitors -- is derived from these files and can be
rebuilt at any time; only this capture is unrepeatable.

Expiry policy (per symbol):
  * the nearest 2 listed expirations (whatever they are),
  * one expiry per calendar month out to ~9 months -- the listing nearest
    that month's third Friday (the monthly),
  * every JANUARY expiration at any horizon: the LEAPs. Structured-note
    flows (autocallables, buffered notes) concentrate there, which is
    exactly the standing OI whose fixed-strike IV the tracker watches.

Contract policy (per expiry):
  * alive contracts (a bid or open interest) within +/-25% moneyness,
  * widened to +/-65% for January LEAPs -- note barriers sit 30-50% below
    spot and would be lost to the normal window,
  * plus the top 20 open-interest strikes per side regardless of
    moneyness, so a monster strike is never dropped.

Each day writes  <out>/YYYY-MM-DD.csv.gz  (columns: date, symbol, expiry,
right, strike, iv, oi, volume, bid, ask, last, spot) and a per-symbol
capture report  <out>/_status-YYYY-MM-DD.json . Same-day reruns overwrite
(idempotent). Failures are per-symbol, never fatal.

    python snapshot_option_chains.py --out optsnap
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import ndx_dark_residual as N
from etf_expected_move_study import CHAIN_URL, CRUMB_SEED_URL, CRUMB_URL, UA

MAX_MONTHLY_MONTHS = 9     # keep one monthly expiry per month out to here
NEAREST_N = 2              # always keep the nearest N listed expirations
TOP_OI_PER_SIDE = 20       # never drop a big-OI strike, whatever its moneyness
MONEYNESS = 0.25           # regular expiries: strikes within +/-25% of spot
MONEYNESS_LEAP = 0.65      # January LEAPs: structured-note barriers run deep
REQUEST_GAP_S = 0.15       # be polite to the endpoint
DEFAULT_UNIVERSE = Path(__file__).resolve().parent / "data" / "optsnap_universe.csv"


# ----------------------------------------------------------------------------
# Pure policy (unit tested)
# ----------------------------------------------------------------------------
def third_friday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1)
    fridays = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]


def select_expiries(expirations, today, max_months=MAX_MONTHLY_MONTHS,
                    nearest_n=NEAREST_N):
    """Subset of expiration epochs per the policy above: nearest N, one
    per-month monthly out to `max_months`, and every January at any
    horizon. Returns sorted epochs."""
    today = pd.Timestamp(today).normalize()
    future = sorted(e for e in expirations
                    if pd.Timestamp(e, unit="s") >= today)
    keep = set(future[:nearest_n])
    horizon = today + pd.DateOffset(months=max_months)
    by_month = {}
    for e in future:
        d = pd.Timestamp(e, unit="s")
        if d.month == 1:                      # LEAP: keep every January
            keep.add(e)
        if d <= horizon:
            by_month.setdefault((d.year, d.month), []).append(e)
    for (y, m), es in by_month.items():
        tf = third_friday(y, m)
        keep.add(min(es, key=lambda e: abs(pd.Timestamp(e, unit="s") - tf)))
    return sorted(keep)


def is_leap(expiry_epoch):
    return pd.Timestamp(expiry_epoch, unit="s").month == 1


def filter_contracts(contracts, spot, leap, moneyness=MONEYNESS,
                     moneyness_leap=MONEYNESS_LEAP, top_oi=TOP_OI_PER_SIDE):
    """Contracts worth persisting for one side of one expiry: alive (a bid
    or OI) within the moneyness window, plus the top-`top_oi` OI strikes
    regardless of moneyness."""
    if not contracts or not spot:
        return []
    window = moneyness_leap if leap else moneyness
    lo, hi = spot * (1 - window), spot * (1 + window)
    keep = {}
    for c in contracts:
        k = c.get("strike")
        alive = (c.get("bid") or 0) > 0 or (c.get("openInterest") or 0) > 0
        if k and alive and lo <= k <= hi:
            keep[k] = c
    ranked = sorted((c for c in contracts if (c.get("openInterest") or 0) > 0),
                    key=lambda c: -(c.get("openInterest") or 0))
    for c in ranked[:top_oi]:
        keep.setdefault(c.get("strike"), c)
    return [keep[k] for k in sorted(keep)]


def contract_rows(chain, symbol, spot, expiry_epoch, snap_date):
    """Flat persistence rows for one expiry's kept calls+puts."""
    leap = is_leap(expiry_epoch)
    exp = str(pd.Timestamp(expiry_epoch, unit="s").date())
    rows = []
    for right, side in (("C", "calls"), ("P", "puts")):
        for c in filter_contracts(chain.get(side) or [], spot, leap):
            rows.append({
                "date": snap_date, "symbol": symbol, "expiry": exp,
                "right": right, "strike": c.get("strike"),
                "iv": c.get("impliedVolatility"), "oi": c.get("openInterest"),
                "volume": c.get("volume"), "bid": c.get("bid"),
                "ask": c.get("ask"), "last": c.get("lastPrice"), "spot": spot,
            })
    return rows


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------
def snapshot(symbols, out_dir, request_gap=REQUEST_GAP_S):
    if N.requests is None:
        raise SystemExit("requests not available")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap_date = str(pd.Timestamp.today().date())
    sess = N.make_session(4)
    sess.headers.update(UA)
    try:
        sess.get(CRUMB_SEED_URL, timeout=15)
    except Exception:  # noqa: BLE001 -- 404 still sets the cookie
        pass
    crumb = sess.get(CRUMB_URL, timeout=15).text.strip()

    def get_chain(sym, expiry=None):
        params = {"crumb": crumb}
        if expiry is not None:
            params["date"] = expiry
        for attempt in (1, 2):
            try:
                d = sess.get(CHAIN_URL.format(sym=sym), params=params,
                             timeout=25).json()
                return d["optionChain"]["result"][0]
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(2.0)
            finally:
                time.sleep(request_gap)

    all_rows, status = [], {}
    for sym in symbols:
        try:
            base = get_chain(sym)
            spot = base["quote"]["regularMarketPrice"]
            exps = select_expiries(base.get("expirationDates") or [],
                                   pd.Timestamp.today())
            n0 = len(all_rows)
            for e in exps:
                res = get_chain(sym, expiry=e)
                opts = res.get("options") or []
                if opts:
                    all_rows.extend(contract_rows(opts[0], sym, spot, e, snap_date))
            status[sym] = {"ok": True, "expiries": len(exps),
                           "rows": len(all_rows) - n0}
        except Exception as exc:  # noqa: BLE001
            status[sym] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{sym}] FAILED: {status[sym]['error']}")

    df = pd.DataFrame(all_rows)
    day_path = out / f"{snap_date}.csv.gz"
    df.to_csv(day_path, index=False, compression="gzip")
    (out / f"_status-{snap_date}.json").write_text(json.dumps({
        "date": snap_date,
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbols": status,
    }, indent=1))
    ok = sum(1 for s in status.values() if s["ok"])
    print(f"wrote {day_path}  ({len(df)} rows, {day_path.stat().st_size // 1024} KB, "
          f"{ok}/{len(status)} symbols ok)")
    return df, status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="optsnap", help="snapshot output directory")
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE),
                    help="CSV with a `symbol` column")
    ap.add_argument("--symbols", default=None,
                    help="comma list overriding the universe file (debugging)")
    args = ap.parse_args()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(pd.read_csv(args.universe)["symbol"].str.upper())
    snapshot(symbols, args.out)


if __name__ == "__main__":
    main()
