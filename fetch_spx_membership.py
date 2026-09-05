#!/usr/bin/env python3
"""
Build a point-in-time S&P 500 membership dataset (2018-08-01 .. today).

Sources (Wikipedia):
  1. https://en.wikipedia.org/wiki/List_of_S%26P_500_companies  (current components
     table AND the primary article; as of this writing the "Selected changes to the
     list of S&P 500 components" table has been split OUT of this page into its own
     article -- confirmed by fetching the live page's raw wikitext, which links
     "Historical components of the S&P 500" for the change log. This mirrors the
     Nasdaq-100 page's own split-page layout, just under a different pair of titles.)
  2. https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500 (change log
     table: Effective Date | Added(Ticker,Security) | Removed(Ticker,Security) |
     Reason | Refs)

Method: take the CURRENT component set, then replay the change log BACKWARD to the
window start (2018-08-01), opening/closing membership stints as we go. Ticker-symbol
renames of a *sitting* member (no add/remove row logged because the company never
actually left the index) are injected as synthetic remove(old)+add(new) pairs so
that every stint's ticker is valid for its own window -- same technique the Nasdaq-100
script uses for FB->META etc.

Output: spx_membership.csv with columns ticker,added,removed (one row per stint;
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

import numpy as np  # noqa: F401  (present per spec; pandas covers our numeric needs)
import pandas as pd
import requests

WINDOW_START = dt.date(2018, 8, 1)
TODAY = dt.date.today()

MAIN_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HIST_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"

HEADERS = {"User-Agent": "spx-membership-research/1.0 (contact: local analysis script)"}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "spx_membership.csv")

# Check dates for the reconciliation report (must land in [495, 508] securities --
# the S&P 500 can briefly exceed 500 names because of overlapping add/remove timing
# or dual share classes, e.g. GOOGL/GOOG, FOX/FOXA, NWS/NWSA).
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
COUNT_LO, COUNT_HI = 495, 508

# ---------------------------------------------------------------------------
# Manual corrections layered on top of Wikipedia
# ---------------------------------------------------------------------------
# The live "current components" table still shows a stale ticker for at least one
# name (Fiserv trades as FI since its June 2023 NYSE move / ticker change, but the
# Wikipedia row was never updated). Patch known-stale current-table tickers here.
CURRENT_TICKER_OVERRIDES = {
    "FISV": "FI",
}

# The live "current components" table can also lag the change log outright: as of
# this writing it is missing EchoStar (SATS), which the change log says replaced
# Paycom (PAYC) on 2026-03-23 -- and PAYC is indeed absent from the current table,
# confirming the swap happened and only the addition side of the table row is
# stale/missing. Patch such gaps here (ticker -> reason), reviewed each run via the
# "unplaced change events" report.
CURRENT_TABLE_ADDITIONS = {
    "SATS": "EchoStar; change log shows it replacing PAYC on 2026-03-23, and PAYC "
            "is indeed absent from the current table, but SATS itself is missing "
            "from that table -- treated as a stale gap, not a real absence.",
}

# Wikipedia's change log sometimes shows a company's *current* ticker on
# historical rows. Map those back to the symbol that was actually trading on
# the event date (window-relevant rows only). None identified for SPX so far,
# but the hook is kept for parity with the Nasdaq-100 script / future-proofing.
EVENT_TICKER_ALIASES = {}

# In-window ticker renames of *sitting* members: (date, old_ticker, new_ticker).
# These are cases where the underlying company never left the index -- it just
# changed its trading symbol -- so the Historical-components change log has no
# add/remove row for the event at all. Encoded as end-of-stint(old) +
# start-of-stint(new) on the rename date. Dates marked "approx." were not found
# verbatim in the Wikipedia change log and are sourced from general market-history
# knowledge; see VERIFICATION.md for details and confidence notes.
RENAMES = [
    # CBS Corp -> ViacomCBS -> Paramount Global -> Paramount Skydance: one
    # continuously-held S&P 500 slot renamed three times. Only the *other* side
    # of the 2019 merger (Viacom's own VIAB share class) has an explicit
    # add/remove row in the change log; CBS's own class silently carried the
    # slot through every rename. Dates for the 2022 and 2025 renames are approx.
    (dt.date(2019, 12, 5), "CBS", "VIAC"),    # CBS+Viacom merger closed (table date)
    (dt.date(2022, 2, 16), "VIAC", "PARA"),   # ViacomCBS -> Paramount Global (approx.)
    (dt.date(2025, 8, 7), "PARA", "PSKY"),    # Paramount+Skydance merger closed (approx.)
    (dt.date(2019, 12, 9), "BBT", "TFC"),     # BB&T -> Truist Financial (table date, paired with STI removal)
    (dt.date(2020, 3, 2), "IR", "TT"),        # Ingersoll-Rand plc -> Trane Technologies (table date, paired with "new" IR add)
    (dt.date(2020, 4, 3), "UTX", "RTX"),      # United Technologies -> Raytheon Technologies (table date, paired with OTIS/CARR spin-offs)
    (dt.date(2021, 8, 3), "LB", "BBWI"),      # L Brands -> Bath & Body Works, Victoria's Secret spin-off (approx.)
    (dt.date(2022, 6, 9), "FB", "META"),      # Facebook -> Meta Platforms ticker change
    (dt.date(2022, 6, 28), "ANTM", "ELV"),    # Anthem -> Elevance Health (approx.)
    (dt.date(2023, 6, 7), "FISV", "FI"),      # Fiserv ticker change / NYSE move (approx.; matches current-table override date)
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


def get_current_components(html):
    for tbl in parse_tables(html):
        if not tbl:
            continue
        header = [c.lower() for c in tbl[0]]
        if any(h in ("symbol", "ticker") for h in header) and any("security" in h or "company" in h for h in header):
            i = 0
            for j, h in enumerate(header):
                if h in ("symbol", "ticker"):
                    i = j
                    break
            tickers = set()
            for r in tbl[1:]:
                if len(r) > i and r[i].strip():
                    t = norm_ticker(r[i])
                    t = CURRENT_TICKER_OVERRIDES.get(t, t)
                    tickers.add(t)
            if len(tickers) >= 400:
                tickers |= set(CURRENT_TABLE_ADDITIONS)
                return sorted(tickers)
    raise RuntimeError("could not find the current components table on %s" % MAIN_URL)


def get_change_events():
    """Return list of (date, kind, ticker) with kind in {'add','remove'}, newest first."""
    html = fetch(HIST_URL)
    best = None
    for tbl in parse_tables(html):
        # Header row is "Effective Date | Added | Removed | Reason | Refs" (5 cells
        # thanks to colspans); data rows have 7 cells: date, add_t, add_name, rem_t,
        # rem_name, reason, refs.
        if len(tbl) > 50 and tbl[0] and "date" in tbl[0][0].lower():
            best = tbl
            break
    if best is None:
        raise RuntimeError("could not find the change-log table on %s" % HIST_URL)

    events = []
    unparsed = []
    for row in best:
        if len(row) < 5:
            continue
        d = parse_date(row[0])
        if d is None:
            if row[0].strip().lower() not in ("date", "effective date", "ticker", ""):
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
    # REMOVEs (handles same-ticker same-date swaps like FOX/FOXA in 2019, and
    # ticker-reuse handoffs like the 2020-03-02 Ingersoll Rand / Trane split).
    kind_rank = {"add": 0, "remove": 1}
    events.sort(key=lambda e: (-e[0].toordinal(), kind_rank[e[1]], e[2]))

    active = {}       # ticker -> removed date (None = still member today)
    stints = []        # (ticker, added or None, removed or None)
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

    # Merge adjacent same-ticker stints (removed == next added), e.g. ticker-reuse
    # handoffs (Ingersoll Rand's "IR" symbol) or multi-part corporate actions
    # recorded as separate rows on the same date.
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
    # Fetch the main article first (primary source; also confirms it still links
    # the change-log page the same way the raw wikitext did when this script was
    # written -- Wikipedia's page layout has moved this content around before).
    main_html = fetch(MAIN_URL)
    if "Historical_components_of_the_S" not in main_html and "Historical components of the S&P 500" not in main_html:
        print("WARNING: main article no longer visibly links the historical-components "
              "page; using %s anyway" % HIST_URL)

    current = get_current_components(main_html)
    print("Current components: %d tickers" % len(current))
    for probe in ("AAPL", "MSFT", "BRK-B", "JPM", "XOM"):
        print("  sanity check: %-6s %s" % (probe, "present" if probe in current else "MISSING"))
    if CURRENT_TABLE_ADDITIONS:
        print("  manual additions to current table (see CURRENT_TABLE_ADDITIONS):")
        for t, why in CURRENT_TABLE_ADDITIONS.items():
            print("    %s: %s" % (t, why))

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
        print("  members on %s: %3d %s" % (d, n, "OK" if ok else "OUT OF RANGE [%d,%d]" % (COUNT_LO, COUNT_HI)))
        if not ok:
            problems.append("membership count %d on %s outside [%d,%d]" % (n, d, COUNT_LO, COUNT_HI))

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
