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

Output CSV columns: ticker, report_date (filing date), accept_et, timing, source.
This is a drop-in replacement for the hand-curated earnings_dates.csv.

Foreign private issuers (ASML, ARM, PDD, ...) never file 8-K/2.02 -- their
results go out as 6-K filings mixed in with every other press release. For any
ticker with no 8-K earnings events we fall back to a 6-K heuristic: within each
calendar quarter's expected reporting window (5-80 days after quarter end),
candidate 6-Ks are scored by their filing-index contents (an EX-99 exhibit,
earnings-keyword document names) and the best-scoring one becomes that
quarter's event (`source=6k`). Heuristic by construction -- flagged in the
study's caveats -- but it recovers the quarterly cadence for the six foreign
NDX names that were previously dropped.

    python fetch_earnings_edgar.py --out earnings_dates_edgar.csv
    python fetch_earnings_edgar.py --tickers AAPL,MSFT,PEP --start 2018-01-01
"""
import argparse
import re
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


FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"
# document-name / exhibit-type hints that a 6-K carries quarterly results
EARN_NAME_RE = re.compile(
    r"(press|results|earning|interim|quarter|[._-]q[1-4]|fy\d{2})", re.IGNORECASE)


def _six_k_score(session, cik, accession):
    """Score a 6-K filing by its index contents: +2 if any document name looks
    like an earnings release, +1 if it carries an EX-99 exhibit. Returns 0 on
    any fetch problem (the filing is then only used as a last resort)."""
    acc = accession.replace("-", "")
    try:
        r = session.get(FILING_INDEX_URL.format(cik=int(cik[3:]), acc=acc),
                        headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return 0
        items = r.json().get("directory", {}).get("item", [])
    except Exception:  # noqa: BLE001
        return 0
    score = 0
    names = " ".join(str(it.get("name", "")) for it in items)
    types = " ".join(str(it.get("type", "")) for it in items).upper()
    if EARN_NAME_RE.search(names):
        score += 2
    if "EX-99" in types:
        score += 1
    return score


def foreign_6k_events(session, cik, start=None, pause=0.12):
    """Quarterly earnings events for a foreign private issuer via its 6-K trail.

    For each calendar quarter, candidate 6-Ks accepted 5-80 days after the
    quarter end are scored (`_six_k_score`); the best scorer wins the quarter
    (earliest on ties). A lone candidate in the window is accepted even when
    unscored -- many foreign filers name their documents blandly. Returns rows
    shaped like `earnings_for_cik`'s.
    """
    cands = []
    for blk in _iter_filing_blocks(session, cik):
        forms = blk.get("form", [])
        for i in range(len(forms)):
            if forms[i] != "6-K":
                continue
            accept = blk["acceptanceDateTime"][i]
            try:
                adate, et, timing = classify(accept)
            except Exception:  # noqa: BLE001
                adate, et, timing = blk["filingDate"][i], "", ""
            cands.append({"adate": adate, "fdate": blk["filingDate"][i], "et": et,
                          "timing": timing, "acc": blk["accessionNumber"][i]})
    if not cands:
        return []
    cands.sort(key=lambda c: c["adate"])
    lo = pd.Timestamp(start) if start else pd.Timestamp("2018-01-01")
    qends = pd.date_range(lo.normalize() - pd.offsets.QuarterEnd(),
                          pd.Timestamp.today() + pd.offsets.QuarterEnd(), freq="QE")
    score_cache = {}
    picked = {}
    for qe in qends:
        win = [c for c in cands
               if qe + pd.Timedelta(days=5) <= pd.Timestamp(c["adate"])
               <= qe + pd.Timedelta(days=80)]
        best, best_score = None, -1
        for c in win:
            if c["acc"] not in score_cache:
                time.sleep(pause)
                score_cache[c["acc"]] = _six_k_score(session, cik, c["acc"])
            s = score_cache[c["acc"]]
            if s > best_score:
                best, best_score = c, s
        if best is None:
            continue
        if best_score >= 1 or len(win) == 1:
            picked[best["acc"]] = best
    return [(c["adate"], c["fdate"], c["et"], c["timing"])
            for c in sorted(picked.values(), key=lambda c: c["adate"])
            if not (start and c["adate"] < start)]


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
        source = "8k"
        if not evs:
            # foreign private issuers (ASML, ARM, PDD, ...) publish results as
            # 6-Ks instead of 8-K/2.02 -- recover them heuristically
            evs = foreign_6k_events(session, cik, start=args.start, pause=args.pause)
            source = "6k"
        if not evs:
            no_earn.append(tk)
            continue
        for adate, fdate, et, timing in evs:
            out_rows.append({"ticker": tk, "report_date": adate, "filing_date": fdate,
                             "accept_et": et, "timing": timing, "source": source})
        print(f"  {tk:6s} {cik}  {len(evs):3d} earnings "
              f"{'8-Ks' if source == '8k' else '6-Ks (heuristic)'}", file=sys.stderr)

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
