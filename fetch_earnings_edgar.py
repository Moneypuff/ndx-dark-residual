#!/usr/bin/env python3
"""
Fetch earnings report dates for a universe from SEC EDGAR.

Authoritative, uniform, one-format: for each company we pull its submissions
JSON, keep 8-K filings carrying Item 2.02 ("Results of Operations", i.e. the
earnings release), and read the filing's acceptance timestamp. Converting that
timestamp to US/Eastern classifies each report as:

    bmo       accepted before 09:30 ET   (news out before the open)
    amc       accepted at/after 16:00 ET (news out after the close)
    intraday  accepted during the session (rarer; flagged)

Output CSV columns: ticker, report_date (filing date), accept_et, timing.
This is a drop-in replacement for the hand-curated earnings_dates.csv.

    python fetch_earnings_edgar.py --out earnings_dates_edgar.csv
    python fetch_earnings_edgar.py --tickers AAPL,MSFT,PEP --start 2018-01-01
"""
import argparse
import sys
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import ndx_dark_residual as N

SEC_UA = "ndx-dark-residual research 91345777quebecinc@gmail.com"
HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
ET = ZoneInfo("America/New_York")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBS_URL = "https://data.sec.gov/submissions/{name}"
OPEN_T, CLOSE_T = dtime(9, 30), dtime(16, 0)


def ticker_cik_map(session):
    r = session.get(TICKERS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = f"CIK{int(row['cik_str']):010d}"
    return out


def classify(accept_iso):
    """acceptanceDateTime (UTC ISO) -> (announce_date, et_string, timing).

    announce_date is the ET calendar date the 8-K hit EDGAR -- the day the news
    became public -- which is what the market reacts to. It can differ from the
    SEC-assigned filingDate: documents accepted after ~17:30 ET are stamped with
    the next business day as filingDate, so filingDate lags an after-hours
    release by a day. We key off acceptance instead.
    """
    dt = datetime.strptime(accept_iso, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(ET)
    t = dt.timetz().replace(tzinfo=None)
    if t < OPEN_T:
        timing = "bmo"
    elif t >= CLOSE_T:
        timing = "amc"
    else:
        timing = "intraday"
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d %H:%M ET"), timing


def _iter_filing_blocks(session, cik):
    """Yield the 'recent' block plus any older paginated blocks for a CIK."""
    r = session.get(SUBS_URL.format(name=f"{cik}.json"), headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return
    doc = r.json()
    yield doc["filings"]["recent"]
    for extra in doc["filings"].get("files", []):
        time.sleep(0.12)
        rr = session.get(SUBS_URL.format(name=extra["name"]), headers=HEADERS, timeout=30)
        if rr.status_code == 200:
            yield rr.json()


def earnings_for_cik(session, cik, start=None, match_window=35):
    """Return the quarterly earnings releases for a CIK.

    Candidates = 8-K filings carrying Item 2.02. But some companies furnish 2.02
    for non-earnings news (Tesla's delivery numbers, monthly sales, guidance
    pre-announcements), which inflates the count. The real quarterly earnings
    release clusters tightly around each periodic report (10-Q / 10-K), while
    operational 2.02s do not -- so we keep, for each 10-Q/10-K, the single 2.02
    whose announce date is nearest (within `match_window` days). This isolates
    ~4 earnings/year without looking at prices (no circularity).
    """
    candidates, periodic = [], []
    for blk in _iter_filing_blocks(session, cik):
        forms = blk.get("form", [])
        for i in range(len(forms)):
            form = forms[i]
            fdate = blk["filingDate"][i]
            if form == "8-K":
                if "2.02" not in (blk["items"][i] or ""):
                    continue
                accept = blk["acceptanceDateTime"][i]
                try:
                    adate, et, timing = classify(accept)
                except Exception:
                    adate, et, timing = fdate, "", ""
                candidates.append({"adate": adate, "fdate": fdate, "et": et, "timing": timing})
            elif form in ("10-Q", "10-K"):
                periodic.append(fdate)
    if not candidates:
        return []
    cand = sorted(candidates, key=lambda c: c["adate"])
    cad = [pd.Timestamp(c["adate"]) for c in cand]
    keep = {}
    for p in periodic:
        pt = pd.Timestamp(p)
        # nearest candidate to this periodic report
        best, bestd = None, None
        for j, ct in enumerate(cad):
            dd = abs((ct - pt).days)
            if bestd is None or dd < bestd:
                bestd, best = dd, j
        if best is not None and bestd <= match_window:
            keep[best] = cand[best]
    rows = [(c["adate"], c["fdate"], c["et"], c["timing"])
            for c in sorted(keep.values(), key=lambda c: c["adate"])
            if not (start and c["adate"] < start)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="earnings_dates_edgar.csv")
    ap.add_argument("--tickers", default="", help="comma list; default = NDX100 universe")
    ap.add_argument("--start", default="2018-01-01", help="drop reports before this date")
    ap.add_argument("--pause", type=float, default=0.12, help="seconds between SEC requests")
    args = ap.parse_args()

    universe = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
                if args.tickers else list(N.NDX100))

    session = requests.Session()
    cikmap = ticker_cik_map(session)

    out_rows = []
    missing_cik, no_earn = [], []
    for tk in universe:
        cik = cikmap.get(tk)
        if not cik:
            missing_cik.append(tk)
            continue
        time.sleep(args.pause)
        evs = earnings_for_cik(session, cik, start=args.start)
        if not evs:
            no_earn.append(tk)
            continue
        for adate, fdate, et, timing in evs:
            out_rows.append({"ticker": tk, "report_date": adate, "filing_date": fdate,
                             "accept_et": et, "timing": timing})
        print(f"  {tk:6s} {cik}  {len(evs):3d} earnings 8-Ks", file=sys.stderr)

    df = pd.DataFrame(out_rows).sort_values(["ticker", "report_date"])
    # de-dupe: occasionally an 8-K/A amendment repeats a date
    df = df.drop_duplicates(subset=["ticker", "report_date"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(df)} events, {df['ticker'].nunique()} names",
          file=sys.stderr)
    if missing_cik:
        print(f"no CIK match ({len(missing_cik)}): {', '.join(missing_cik)}", file=sys.stderr)
    if no_earn:
        print(f"no 8-K/2.02 found ({len(no_earn)}): {', '.join(no_earn)}", file=sys.stderr)
    print("timing mix:", df["timing"].value_counts().to_dict(), file=sys.stderr)
    return df


if __name__ == "__main__":
    main()
