#!/usr/bin/env python3
"""
Build a point-in-time Nasdaq-100 membership dataset (2018-08-01 .. today).

Sources (Wikipedia):
  1. https://en.wikipedia.org/wiki/Nasdaq-100  (fetched to discover the split-out pages)
  2. https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies         (current components)
  3. https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100 (change log table:
     Date | Added(Ticker,Security) | Removed(Ticker,Security) | Reason)

Method: take the CURRENT component set, then replay the change log BACKWARD to the
window start (2018-08-01), opening/closing membership stints as we go. Ticker-symbol
renames that happened while a company was a member are injected as synthetic
remove(old)+add(new) pairs so that every stint's ticker is valid for its own window.

Output: ndx_membership.csv with columns ticker,added,removed (one row per stint;
`removed` empty = still a member). Dates are effective index-change dates; a ticker is
considered a member on dates d with added <= d < removed.

Dependencies: numpy, pandas, requests (HTML tables are parsed with the stdlib
html.parser so no lxml/html5lib is needed).
"""

import os
import re
import sys
import datetime as dt
from html.parser import HTMLParser

import pandas as pd
import requests

WINDOW_START = dt.date(2018, 8, 1)
TODAY = dt.date.today()

MAIN_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
LIST_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
HIST_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100"

HEADERS = {"User-Agent": "ndx-membership-research/1.0 (contact: local analysis script)"}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "data", "ndx_historical_membership.csv")

# Check dates for the reconciliation report (must land in [98, 104] securities).
CHECK_DATES = [
    dt.date(2018, 8, 1),
    dt.date(2019, 6, 3),
    dt.date(2020, 6, 1),
    dt.date(2021, 6, 1),
    dt.date(2022, 6, 1),
    dt.date(2023, 6, 1),
    dt.date(2024, 6, 3),
    dt.date(2025, 6, 2),
    TODAY,
]
COUNT_LO, COUNT_HI = 98, 104

# ---------------------------------------------------------------------------
# Manual corrections layered on top of Wikipedia
# ---------------------------------------------------------------------------
# Wikipedia's change log sometimes shows a company's *current* ticker on
# historical rows. Map those back to the symbol that was actually trading on
# the event date (window-relevant rows only).
EVENT_TICKER_ALIASES = {
    "FI": "FISV",    # Fiserv traded as FISV for its whole membership; the FI
                     # rename coincided with the NYSE transfer/removal 2023-06-07.
    "WTW": "WLTW",   # Willis Towers Watson traded as WLTW during its
                     # 2018-12-24 .. 2020-04-30 stint (WTW rename: 2022-01).
}

# In-window ticker renames of sitting members: (date, old_ticker, new_ticker).
# Encoded as end-of-stint(old) + start-of-stint(new) on the rename date.
RENAMES = [
    (dt.date(2019, 11, 5), "SYMC", "NLOK"),  # Symantec -> NortonLifeLock (approx. early Nov 2019)
    (dt.date(2019, 11, 5), "CTRP", "TCOM"),  # Ctrip -> Trip.com Group (approx. early Nov 2019)
    (dt.date(2022, 6, 9), "FB", "META"),     # Facebook -> Meta Platforms ticker change
]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


class TableExtractor(HTMLParser):
    """Extract all <table> elements as lists of rows of cell strings (stdlib only)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._tstack = []   # stack of tables (rows)
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._tstack.append([])
        elif tag == "tr" and self._tstack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = "".join(self._cell)
            text = re.sub(r"\[\d+\]", "", text)          # strip footnote markers
            text = re.sub(r"\s+", " ", text).strip()
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._tstack and self._row:
                self._tstack[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._tstack:
            self.tables.append(self._tstack.pop())

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def parse_tables(html):
    p = TableExtractor()
    p.feed(html)
    return p.tables


def norm_ticker(t):
    """Yahoo convention: uppercase, class shares with '-'."""
    t = (t or "").strip().upper()
    t = re.sub(r"\[\d+\]", "", t)
    t = t.replace(".", "-")
    return t


def parse_date(s):
    m = re.match(r"\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if not m or m.group(1) not in MONTHS:
        return None
    return dt.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))


def get_current_components():
    html = fetch(LIST_URL)
    for tbl in parse_tables(html):
        if not tbl:
            continue
        header = [c.lower() for c in tbl[0]]
        if "ticker" in header and any("company" in h for h in header):
            i = header.index("ticker")
            tickers = sorted({norm_ticker(r[i]) for r in tbl[1:] if len(r) > i and r[i].strip()})
            if len(tickers) >= 90:
                return tickers
    raise RuntimeError("could not find the current components table on %s" % LIST_URL)


def get_change_events():
    """Return list of (date, kind, ticker) with kind in {'add','remove'}, newest first."""
    html = fetch(HIST_URL)
    best = None
    for tbl in parse_tables(html):
        # Header row is "Date | Added | Removed | Reason" (4 cells because of
        # colspans); data rows have 6 cells: date, add_t, add_name, rem_t, rem_name, reason.
        if len(tbl) > 50 and tbl[0] and "date" in tbl[0][0].lower():
            best = tbl
            break
    if best is None:
        raise RuntimeError("could not find the change-log table on %s" % HIST_URL)

    # The table has a two-row header (Date | Added | Removed | Reason, then
    # Ticker/Security sub-headers). Data rows: date, add_t, add_name, rem_t, rem_name, reason.
    events = []
    unparsed = []
    for row in best:
        if len(row) < 5:
            continue
        d = parse_date(row[0])
        if d is None:
            if row[0].strip().lower() not in ("date", "ticker", ""):
                unparsed.append(row[0])
            continue
        add_t = norm_ticker(row[1])
        rem_t = norm_ticker(row[3]) if len(row) > 3 else ""
        if add_t:
            events.append((d, "add", EVENT_TICKER_ALIASES.get(add_t, add_t)))
        if rem_t:
            events.append((d, "remove", EVENT_TICKER_ALIASES.get(rem_t, rem_t)))
        if not add_t and not rem_t:
            unparsed.append(" | ".join(row))
    return events, unparsed


def build_stints(current, events):
    """Replay events backward from the current membership to WINDOW_START."""
    # Keep only events inside the study window (events on/before WINDOW_START are
    # already reflected in the membership state we reconstruct at WINDOW_START).
    events = [e for e in events if WINDOW_START < e[0] <= TODAY]

    # Inject rename events: remove(old) + add(new) on the rename date.
    for d, old, new in RENAMES:
        if WINDOW_START < d <= TODAY:
            events.append((d, "add", new))
            events.append((d, "remove", old))

    # Backward order: newest date first; within a date process ADDs before
    # REMOVEs (handles same-ticker same-date swaps like the FOX/FOXA 2019 rows).
    kind_rank = {"add": 0, "remove": 1}
    events.sort(key=lambda e: (-e[0].toordinal(), kind_rank[e[1]], e[2]))

    active = {}       # ticker -> removed date (None = still member today)
    stints = []       # (ticker, added or None, removed or None)
    unplaced = []
    for t in current:
        active[t] = None

    for d, kind, t in events:
        if kind == "add":
            if t in active:
                stints.append((t, d, active.pop(t)))
            else:
                unplaced.append("%s add %s (ticker not in membership set at that point)" % (d, t))
        else:  # remove
            if t in active:
                unplaced.append("%s remove %s (ticker already in membership set)" % (d, t))
            else:
                active[t] = d

    # Whatever is still active was a member at the window start.
    for t, rem in active.items():
        stints.append((t, WINDOW_START, rem))

    # Drop stints that end on/before the window start (fully out of scope).
    stints = [s for s in stints if s[2] is None or s[2] > WINDOW_START]

    # Merge adjacent same-ticker stints (removed == next added), e.g. the
    # 21st-Century-Fox -> Fox Corporation rows reused the FOXA/FOX tickers.
    merged = []
    for t, a, r in sorted(stints, key=lambda s: (s[0], s[1])):
        if merged and merged[-1][0] == t and merged[-1][2] == a:
            merged[-1][2] = r
        else:
            merged.append([t, a, r])
    return merged, unplaced


def members_on(stints, d):
    return sorted(t for t, a, r in stints if a <= d and (r is None or r > d))


def main():
    # Fetch the main article first (primary source; also confirms the two
    # split-out pages are still what it links to).
    try:
        main_html = fetch(MAIN_URL)
        for name, url in (("List_of_NASDAQ-100_companies", LIST_URL),
                          ("Historical_components_of_the_Nasdaq-100", HIST_URL)):
            if name not in main_html:
                print("WARNING: main article no longer links %s; using %s anyway" % (name, url))
    except Exception as e:
        print("WARNING: could not fetch main article (%s); continuing with subpages" % e)

    current = get_current_components()
    print("Current components: %d tickers" % len(current))

    events, unparsed = get_change_events()
    in_window = [e for e in events if WINDOW_START < e[0] <= TODAY]
    print("Change events parsed: %d total, %d inside %s..%s"
          % (len(events), len(in_window), WINDOW_START, TODAY))

    stints, unplaced = build_stints(current, events)

    # ------------------------------------------------------------------ checks
    problems = []
    for t, a, r in stints:
        if r is not None and r <= a:
            problems.append("stint with removed<=added: %s %s %s" % (t, a, r))
    by_ticker = {}
    for t, a, r in stints:
        by_ticker.setdefault(t, []).append((a, r or dt.date(9999, 1, 1)))
    for t, ivs in by_ticker.items():
        ivs.sort()
        for (a1, r1), (a2, r2) in zip(ivs, ivs[1:]):
            if a2 < r1:
                problems.append("overlapping stints for %s: %s-%s and %s-%s" % (t, a1, r1, a2, r2))

    print("\nReconciliation report")
    print("---------------------")
    for d in CHECK_DATES:
        n = len(members_on(stints, d))
        ok = COUNT_LO <= n <= COUNT_HI
        print("  members on %s: %3d %s" % (d, n, "OK" if ok else "OUT OF RANGE [98,104]"))
        if not ok:
            problems.append("membership count %d on %s outside [98,104]" % (n, d))

    print("  unplaced change events: %d" % len(unplaced))
    for u in unplaced:
        print("    ", u)
    if unparsed:
        print("  unparsed change-log rows: %d" % len(unparsed))
        for u in unparsed[:10]:
            print("    ", u)

    final = set(members_on(stints, TODAY))
    if final != set(current):
        problems.append("final membership does not equal current components table: %s"
                        % sorted(final.symmetric_difference(set(current))))

    if problems:
        print("\nSANITY-CHECK FAILURES:")
        for p in problems:
            print("  ", p)
        sys.exit(1)

    # ------------------------------------------------------------------ output
    df = pd.DataFrame(
        [(t, a.isoformat(), "" if r is None else r.isoformat()) for t, a, r in stints],
        columns=["ticker", "added", "removed"],
    ).sort_values(["ticker", "added"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print("\nWrote %s (%d stints, %d distinct tickers)" % (OUT_CSV, len(df), df["ticker"].nunique()))


if __name__ == "__main__":
    main()
