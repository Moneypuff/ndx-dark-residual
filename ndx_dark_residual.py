#!/usr/bin/env python3
"""
NDX-100 Dark-Ratio (D) Residual Dashboard
==========================================

Builds the dark-pool indicator (`D`) for every Nasdaq-100 constituent plus QQQ
entirely from free public data -- NO paid API, NO key:

  * dark signal : FINRA's daily consolidated off-exchange (CNMSshvol) files.
                  Per-name D = 5-day MA of ShortVolume / off-exchange TotalVolume
                  -- the dark-pool indicator (DPI), the same construction
                  SqueezeMetrics uses, computed here directly.
  * prices      : Yahoo Finance daily bars (raw close for as-traded dollar
                  weighting, adjusted close for split-safe forward returns).

Each name's *name-specific* dark-flow is isolated by residualizing its D against
a reconstructed NDX-100 dollar-DIX benchmark (sum(price*short)/sum(price*total)
across the constituents) two ways:

  1. Simple difference     :  resid_i,t = D_i,t - NDX_DIX_t
  2. Regression residual   :  rolling OLS  D_i ~ a + b * NDX_DIX  ->  epsilon
                              (removes both the common level AND each name's
                              beta to the market dark-flow)

Also includes a "D vs forward return" tab (1mo / 2mo / 3mo horizon toggle), and
SPX / IWM tabs that show reconstructed dollar-DIX built the same way over the
S&P 500 (iShares IVV holdings) and Russell 2000 (iShares IWM holdings) universes.

Output is a single self-contained static HTML file, safe to host on GitHub Pages
(no credentials are embedded because none are used).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  # real data -- no key needed
  python ndx_dark_residual.py --dark-start 2018-08-01 --out docs/index.html

  # preview the layout with synthetic data, no network needed
  python ndx_dark_residual.py --demo --out demo.html

  # skip the SPX tab (avoids the ~500 S&P 500 constituent fetch)
  python ndx_dark_residual.py --no-spx --out docs/index.html

Design notes
------------
* Rolling regression beta/alpha are computed in closed form from rolling
  covariance / variance (fast, no per-window OLS loop).
* FINRA's consolidated (CNMSshvol) off-exchange files cover ADF + both TRFs
  (all off-exchange trading) and begin 2018-08-01, which is the history floor.
* Data is cached on disk (--cache-dir): FINRA day-files accumulate incrementally,
  and Yahoo prices reuse a same-day cache, so re-runs only fetch what's new.
"""

import argparse
import concurrent.futures
import io
import json
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

FINRA_TMPL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
FINRA_MIN_DATE = pd.Timestamp("2018-08-01")  # earliest date FINRA's consolidated NMS short-volume file covers
# A recent trading day whose FINRA file was not posted yet when we last ran returns "no file"
# -- identical to a holiday -- and gets cached as an all-NaN row. Within this many days of
# today we re-check such empty rows each run, so a delayed post (FINRA usually publishes after
# ~5pm ET, occasionally later) is picked up instead of being frozen as a permanent gap. Wide
# enough to span a long holiday weekend; a genuine holiday just stays empty and ages out.
FINRA_RECENT_REFETCH_DAYS = 6
# Cache lives under the user's home directory by default: a short, stable path that
# (a) is found no matter which working directory the script runs from, and (b) avoids
# Windows' 260-char MAX_PATH limit that a script-relative deep path would blow past.
DEFAULT_CACHE_DIR = str(Path.home() / ".ndx_dark_cache")

# Nasdaq-100 constituents in index-weight order (descending) with approximate index
# weights (%). Source: https://www.slickcharts.com/nasdaq100 (which mirrors the QQQ /
# Nasdaq-100 weighting), retrieved 2026-07-18. Weights drift daily and membership
# changes at the quarterly rebalance -- refresh from the same source as needed.
# 2026-07-18 refresh added two new members vs the 2026-07-05 snapshot: SPCX (Space
# Exploration Technologies / SpaceX, ~4% -- a top-10 name) and HONA (Honeywell
# Aerospace, the Honeywell breakup spinco, trading alongside the remaining HON).
NDX100_WEIGHTS = [
    ("NVDA", 12.24), ("AAPL", 12.21), ("MSFT", 7.29), ("AMZN", 6.63), ("GOOGL", 5.44),
    ("GOOG", 5.09), ("AVGO", 4.40), ("META", 4.09), ("SPCX", 4.07), ("TSLA", 3.56),
    ("MU", 2.39), ("WMT", 2.27), ("AMD", 2.01), ("ASML", 1.67), ("INTC", 1.19),
    ("CSCO", 1.10), ("AMAT", 1.05), ("COST", 1.04), ("LRCX", 0.98), ("PLTR", 0.79),
    ("PANW", 0.73), ("NFLX", 0.72), ("ARM", 0.71), ("KLAC", 0.69), ("TXN", 0.64),
    ("LIN", 0.59), ("TMUS", 0.52), ("CRWD", 0.52), ("SNDK", 0.50), ("AMGN", 0.49),
    ("PEP", 0.47), ("ADI", 0.46), ("QCOM", 0.45), ("STX", 0.44), ("MRVL", 0.42),
    ("GILD", 0.42), ("WDC", 0.41), ("SHOP", 0.40), ("APP", 0.36), ("BKNG", 0.35),
    ("VRTX", 0.31), ("ISRG", 0.30), ("SBUX", 0.30), ("PDD", 0.30), ("FTNT", 0.29),
    ("ADP", 0.25), ("MAR", 0.24), ("MNST", 0.24), ("ADBE", 0.23), ("CSX", 0.23),
    ("DDOG", 0.23), ("MELI", 0.23), ("CDNS", 0.23), ("CEG", 0.22), ("ABNB", 0.22),
    ("CMCSA", 0.21), ("CTAS", 0.20), ("DASH", 0.20), ("INTU", 0.20), ("MDLZ", 0.20),
    ("ROST", 0.19), ("SNPS", 0.18), ("AEP", 0.18), ("ORLY", 0.18), ("HON", 0.18),
    ("REGN", 0.18), ("WBD", 0.17), ("NXPI", 0.17), ("HONA", 0.17), ("PCAR", 0.17),
    ("MPWR", 0.16), ("LITE", 0.14), ("BKR", 0.14), ("FANG", 0.14), ("EA", 0.13),
    ("FAST", 0.13), ("ALAB", 0.13), ("TER", 0.13), ("PYPL", 0.12), ("XEL", 0.12),
    ("ODFL", 0.12), ("EXC", 0.12), ("CCEP", 0.12), ("ADSK", 0.11), ("FER", 0.11),
    ("NBIS", 0.11), ("IDXX", 0.11), ("MCHP", 0.11), ("TTWO", 0.11), ("RKLB", 0.11),
    ("KDP", 0.10), ("TRI", 0.10), ("AXON", 0.10), ("PAYX", 0.10), ("CRWV", 0.10),
    ("ROP", 0.09), ("WDAY", 0.09), ("ALNY", 0.09), ("MSTR", 0.09), ("KHC", 0.08),
    ("DXCM", 0.07), ("GEHC", 0.07), ("CPRT", 0.06),
]
NDX100 = [t for t, _ in NDX100_WEIGHTS]
NDX100_WEIGHT = dict(NDX100_WEIGHTS)  # ticker -> index weight (%)
BENCH = "QQQ"

# GICS sector per name, shown in the small-multiple cell footer (between the two dates).
# Labels match the SECTOR_ETFS display names so the NDX-100 (static) and S&P 500
# (supplemented from the broad SPDR sector-fund holdings) grids read consistently.
TICKER_SECTOR = {
    "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology",
    "AMZN": "Cons. Discretionary", "GOOGL": "Comm. Services", "GOOG": "Comm. Services",
    "AVGO": "Technology", "META": "Comm. Services", "TSLA": "Cons. Discretionary",
    "MU": "Technology", "WMT": "Cons. Staples", "AMD": "Technology", "ASML": "Technology",
    "INTC": "Technology", "AMAT": "Technology", "CSCO": "Technology", "LRCX": "Technology",
    "COST": "Cons. Staples", "ARM": "Technology", "NFLX": "Comm. Services",
    "PLTR": "Technology", "KLAC": "Technology", "PANW": "Technology", "TXN": "Technology",
    "SNDK": "Technology", "LIN": "Materials", "MRVL": "Technology", "AMGN": "Health Care",
    "PEP": "Cons. Staples", "TMUS": "Comm. Services", "WDC": "Technology",
    "QCOM": "Technology", "STX": "Technology", "ADI": "Technology", "APP": "Technology",
    "GILD": "Health Care", "SHOP": "Technology", "ISRG": "Health Care",
    "BKNG": "Cons. Discretionary", "VRTX": "Health Care", "SBUX": "Cons. Discretionary",
    "PDD": "Cons. Discretionary", "FTNT": "Technology", "CDNS": "Technology",
    "MAR": "Cons. Discretionary", "ADP": "Industrials", "MNST": "Cons. Staples",
    "DDOG": "Technology", "CSX": "Industrials", "MELI": "Cons. Discretionary",
    "ABNB": "Cons. Discretionary", "ADBE": "Technology", "CEG": "Utilities",
    "CMCSA": "Comm. Services", "SNPS": "Technology", "DASH": "Cons. Discretionary",
    "MDLZ": "Cons. Staples", "AEP": "Utilities", "INTU": "Technology",
    "ORLY": "Cons. Discretionary", "HON": "Industrials", "CTAS": "Industrials",
    "ALAB": "Technology", "NXPI": "Technology", "REGN": "Health Care",
    "ROST": "Cons. Discretionary", "WBD": "Comm. Services", "MPWR": "Technology",
    "PCAR": "Industrials", "RKLB": "Industrials", "TER": "Technology", "LITE": "Technology",
    "FAST": "Industrials", "NBIS": "Technology", "BKR": "Energy", "EA": "Comm. Services",
    "XEL": "Utilities", "CRWD": "Technology", "EXC": "Utilities", "FER": "Industrials",
    "FANG": "Energy", "AXON": "Industrials", "TTWO": "Comm. Services",
    "CCEP": "Cons. Staples", "MCHP": "Technology", "KDP": "Cons. Staples",
    "ODFL": "Industrials", "CRWV": "Technology", "IDXX": "Health Care",
    "ADSK": "Technology", "ALNY": "Health Care", "PYPL": "Financials",
    "TRI": "Industrials", "PAYX": "Industrials", "ROP": "Technology", "MSTR": "Technology",
    "WDAY": "Technology", "KHC": "Cons. Staples", "GEHC": "Health Care",
    "CPRT": "Industrials", "DXCM": "Health Care",
    "SPCX": "Industrials", "HONA": "Industrials",
}


# Recycled tickers: a symbol whose FINRA volume history covers a PRIOR, unrelated
# security before the current company started trading under it. FINRA's daily files are
# keyed purely by ticker string, so a naive fetch splices the defunct predecessor's
# off-exchange volume onto the new name and contaminates the aggregate. Map ticker ->
# first valid trading day for the CURRENT security; everything before it is dropped to
# NaN at read time (see fetch_finra_dark_volume_panel._slice). Read-time masking is
# deliberate -- it is non-destructive and survives a full --refresh, unlike purging the
# cache in place (which a later refetch would silently undo).
TICKER_VALID_FROM = {
    # SPCX now = SpaceX (Space Exploration Technologies Cl A), first traded 2026-06-12.
    # The pre-cutover SPCX FINRA column is the defunct Tuttle Capital SPAC & New-Issue
    # ETF (2020-12-16 -> 2026-04-06), an unrelated fund -- must not blend into NDX DIX.
    "SPCX": pd.Timestamp("2026-06-12"),
    # ECHO now = EchoStar (renamed its ticker from SATS on 2026-06-24). The ticker ECHO
    # previously belonged to Echo Global Logistics until it went private 2021-11-24 -- an
    # unrelated company. Drop that native pre-2021-11-24 ECHO history; the EchoStar era is
    # reconstructed from the SATS alias below + native ECHO from 2026-06-24 on.
    "ECHO": pd.Timestamp("2021-11-24"),
}


# Ticker aliases for FINRA off-exchange volume. FINRA's daily files are keyed purely by
# the ticker string that was LIVE that day, so a security that later renamed its ticker has
# its earlier volume filed under the OLD string. Map current-ticker -> list of
# (old_ticker, start, end) spans. SEMANTICS (see fetch_finra_dark_volume_panel._row): on a
# trading day in [start, end) the current name's value is taken from the OLD ticker if that
# old ticker is present in FINRA's daily file; if none of the day's applicable old tickers
# is present, the day is DROPPED (never fall back to the current string in-span). This is a
# prefer-predecessor OVERRIDE, not a fill-if-absent -- necessary because many current
# tickers were RECYCLED: an unrelated company traded under the same string before the
# rename, and that predecessor often traded SIMULTANEOUSLY with the real security under its
# old ticker (e.g. FB and an unrelated META both traded pre-2022-06). Overriding to the old
# ticker recovers the real history and discards the impostor. Outside every span the native
# current ticker is used. Because it acts during (re)fetch, a full --refresh RECONSTRUCTS
# the correct series -- unlike a manual cache splice, which any refetch silently NaNs out.
# Ranges were traced empirically from monthly FINRA samples (scratchpad/predecessor_trace).
TICKER_ALIASES = {
    # EchoStar traded as SATS until it renamed to ECHO on 2026-06-24. ECHO was dormant
    # (Echo Global went private) 2021-11-24 -> 2026-06-24, so read SATS in between. (The
    # still-earlier Echo Global era is masked by TICKER_VALID_FROM["ECHO"].)
    "ECHO": [("SATS", pd.Timestamp("2021-11-24"), pd.Timestamp("2026-06-24"))],
    # Fiserv round-tripped: FISV -> FI (~2023-06-06) -> back to FISV (2025-11-11). Fill the
    # FI era from the FI string; native FISV covers both ends.
    "FISV": [("FI", pd.Timestamp("2023-06-07"), pd.Timestamp("2025-11-11"))],
    # Marsh & McLennan renamed MMC -> MRSH on 2026-01-14; backfill everything before it.
    "MRSH": [("MMC", FINRA_MIN_DATE, pd.Timestamp("2026-01-14"))],
    # Facebook traded as FB until it renamed to META on 2022-06-09; an UNRELATED "META" held
    # the ticker before that, so override to FB for the whole pre-rename era.
    "META": [("FB", FINRA_MIN_DATE, pd.Timestamp("2022-06-09"))],
    # Axon Enterprise traded as AAXN until ~2021-01-26 (unrelated pre-gap AXON before that).
    "AXON": [("AAXN", FINRA_MIN_DATE, pd.Timestamp("2021-01-26"))],
    # Willis Towers Watson traded as WLTW until 2022-01-10 (pre-gap WTW = old Weight Watchers).
    "WTW":  [("WLTW", FINRA_MIN_DATE, pd.Timestamp("2022-01-10"))],
    # AmerisourceBergen (ABC) renamed to Cencora (COR) on 2023-08-30 (pre-gap COR = CoreSite).
    "COR":  [("ABC", FINRA_MIN_DATE, pd.Timestamp("2023-08-30"))],
    # Gen Digital chain: Symantec (SYMC) -> NortonLifeLock (NLOK, ~2019-11) -> GEN (2022-11-08).
    # Overlapping spans so the SYMC->NLOK handoff picks whichever is present near the boundary.
    "GEN":  [("SYMC", FINRA_MIN_DATE, pd.Timestamp("2019-12-01")),
             ("NLOK", pd.Timestamp("2019-10-01"), pd.Timestamp("2022-11-08"))],
    # Bank of New York Mellon renamed BK -> BNY on ~2026-05-21 (pre-gap BNY = unrelated).
    "BNY":  [("BK", FINRA_MIN_DATE, pd.Timestamp("2026-05-21"))],
    # II-VI (IIVI) renamed to Coherent (COHR) ~2022-09-08 (pre-gap COHR = the acquired,
    # unrelated old Coherent Inc).
    "COHR": [("IIVI", FINRA_MIN_DATE, pd.Timestamp("2022-09-08"))],
}

# Bump whenever TICKER_ALIASES or TICKER_VALID_FROM change. The FINRA cache stamps the
# version it was built under (per namespace); when a cache predates the current version,
# fetch_finra_dark_volume_panel drops the affected target columns so the backfill rebuilds
# them under the new maps. This is what propagates a transition fix to ANY existing cache --
# crucially the CI runner's persistent cache, which otherwise keeps serving stale columns
# forever (the aliases only rebuild a column when its days are actually (re)fetched, and an
# already-cached column is never refetched). Deterministic: fires exactly once per bump.
TICKER_TRANSITION_VERSION = 1


# --------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------
def make_session(pool_size=8):
    """A requests.Session whose connection pool is sized for concurrent use."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size,
                                            pool_maxsize=pool_size)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def parallel_map(fn, items, workers):
    """Map `fn` over `items` concurrently, preserving input order (like the builtin
    `map`). Falls back to a serial loop when workers <= 1. `fn` must swallow its own
    errors -- an exception here would abort the whole batch.
    """
    items = list(items)
    if workers and workers > 1 and len(items) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            yield from ex.map(fn, items)
    else:
        yield from map(fn, items)










def compute_forward_return(close_panel, horizon):
    """Forward `horizon`-trading-day % return from a wide CLOSE panel."""
    return (close_panel.shift(-horizon) / close_panel - 1) * 100


# --------------------------------------------------------------------------
# SPX-wide Dark Index (DIX) via SqueezeMetrics' "yacht club" GEX+ endpoint
# --------------------------------------------------------------------------




def build_index_payload(df, col, out_key, start=None):
    """Pack a single index-level series (`col` of `df`, which also carries CLOSE) plus
    its own 1mo/2mo/3mo forward returns for an index tab. Returns None if no data,
    so the front end can render an empty state instead of a broken tab.

    `start` (a Timestamp) restricts the analysis window; forward returns are computed
    on the full series first so returns just after the cutoff still see their (present)
    future closes, then the series is sliced to [start, ...].
    """
    if df is None or df.empty:
        return None
    df = df.dropna(subset=[col])
    if df.empty:
        return None
    r21 = compute_forward_return(df[["CLOSE"]], 21)["CLOSE"]  # 1 month
    r42 = compute_forward_return(df[["CLOSE"]], 42)["CLOSE"]  # 2 months
    r63 = compute_forward_return(df[["CLOSE"]], 63)["CLOSE"]  # 3 months

    if start is not None:
        keep = df.index >= start
        df, r21, r42, r63 = df[keep], r21[keep], r42[keep], r63[keep]
        if df.empty:
            return None

    def rnd(s):
        return [None if pd.isna(x) else round(float(x), 4) for x in s.values]

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    return {
        "dates": dates,
        out_key: rnd(df[col]),
        "r21": rnd(r21.reindex(df.index)),
        "r42": rnd(r42.reindex(df.index)),
        "r63": rnd(r63.reindex(df.index)),
        "range": [dates[0], dates[-1]] if dates else None,
    }








def demo_russell_dix(start="2022-01-01", seed=29):
    """Synthetic Russell 2000 reconstructed dollar-DIX + IWM CLOSE for
    `--demo --iwm-reconstruct`: an AR(1) DIX (0-1) plus a CLOSE random walk with a mild
    negative DIX effect, so the reconstructed IWM tab is non-trivial without a key/net.
    Returns (dix_series, close_series)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    n = len(dates)
    dix = np.empty(n); dix[0] = 0.46
    for t in range(1, n):
        dix[t] = 0.46 + 0.94 * (dix[t - 1] - 0.46) + rng.normal(0, 0.008)
    dix = np.clip(dix, 0.30, 0.62)
    z = (dix - dix.mean()) / (dix.std(ddof=0) + 1e-9)
    daily = -0.01 * z + rng.normal(0, 1.4, n)
    close = 180 * np.cumprod(1 + daily / 100)
    return pd.Series(dix, index=dates), pd.Series(close, index=dates)


# --------------------------------------------------------------------------
# FINRA daily short-sale volume files -> raw (unsmoothed) 1-day dark ratio
# --------------------------------------------------------------------------
def _finra_missing(status, text):
    """True if this response means 'no file exists for this date'. FINRA's CDN
    returns 404, or 403 with an S3 'AccessDenied'/'NoSuchKey' body, for dates with
    no published file (weekends/holidays). Other non-200s (throttling, outages) are
    treated as transient and retried instead of being cached as 'no data'."""
    if status == 404:
        return True
    if status == 403 and ("AccessDenied" in text or "NoSuchKey" in text):
        return True
    return False


def _parse_finra_volumes(text):
    """Parse a FINRA short-volume file body into {symbol: (ShortVolume, TotalVolume)}
    -- shares sold short and total reported shares on the off-exchange venues. Both
    are needed: TotalVolume drives the per-name dark-share ratio, ShortVolume is the
    numerator of the DPI/DIX construction."""
    out = {}
    for line in text.splitlines()[1:]:  # skip header
        parts = line.split("|")
        if len(parts) < 5:
            continue
        try:
            out[parts[1]] = (float(parts[2]), float(parts[4]))
        except ValueError:  # a trailing summary/footer line, or a blank field
            continue
    return out


def fetch_finra_offexchange_volume(date_str, session=None, retries=2, pause=0.3):
    """Fetch {symbol: (ShortVolume, TotalVolume)} for one date (YYYYMMDD) from FINRA's
    consolidated daily short-sale volume file (all ADF + TRF venues -- every trade
    *not* executed on a lit exchange).

    Returns (volumes, resolved): `resolved` is True when we got a definitive answer --
    either data (a 200) or a confirmed 'no file' for a holiday (404 / 403 AccessDenied,
    -> empty dict). `resolved` is False when the fetch failed transiently (network
    error / other non-200) so the caller can leave the date for a later run instead of
    recording a false 'no data'.
    """
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    get = (session or requests).get
    url = FINRA_TMPL.format(date=date_str)
    for attempt in range(retries):
        try:
            r = get(url, timeout=20)
            if _finra_missing(r.status_code, r.text):
                return {}, True                      # confirmed: no file this day (holiday)
            if r.status_code != 200:
                time.sleep(pause * (attempt + 1))
                continue
            return _parse_finra_volumes(r.text), True
        except Exception:  # noqa: BLE001
            time.sleep(pause * (attempt + 1))
    return {}, False                                 # transient failure -> don't record


# --------------------------------------------------------------------------
# Consolidated on-disk documents (one CSV each, dates x symbols) that accumulate the
# FINRA volumes we have ever fetched and grow a little each run. Two kinds: "offexch"
# (TotalVolume; legacy filename so pre-existing caches keep working) and "short"
# (ShortVolume, needed for the DIX construction). This replaces the old
# one-file-per-day cache, which broke on Windows' 260-char path limit.
# --------------------------------------------------------------------------
FINRA_DOC_NAMES = {"offexch": "finra_offexch_volume.csv", "short": "finra_short_volume.csv"}


def finra_doc_path(cache_dir, kind="offexch", ns=""):
    # `ns` namespaces the cache file so a different symbol universe (e.g. the ~2,000
    # Russell 2000 names) accumulates in its OWN document instead of colliding with the
    # NDX-100 one (whose rows exist for the same dates but hold only NDX-100 columns).
    name = FINRA_DOC_NAMES[kind]
    if ns:
        name = name.replace(".csv", f"_{ns}.csv")
    return Path(cache_dir) / name


def load_finra_document(cache_dir, kind="offexch", ns=""):
    """Load a consolidated volume document (dates x symbols); empty DataFrame if it
    does not exist or cannot be read."""
    if not cache_dir:
        return pd.DataFrame()
    p = finra_doc_path(cache_dir, kind, ns)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read cache document {p.name} ({e}); rebuilding it", file=sys.stderr)
        return pd.DataFrame()


def save_finra_document(df, cache_dir, kind="offexch", ns=""):
    if not cache_dir or df is None or df.empty:
        return
    p = finra_doc_path(cache_dir, kind, ns)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # write to a temp file then replace, so an interrupted run can't corrupt it
        tmp = p.with_suffix(".csv.tmp")
        df.sort_index().to_csv(tmp)
        tmp.replace(p)
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not write cache document {p.name} ({e})", file=sys.stderr)


def _txn_version_path(cache_dir, ns=""):
    name = "finra_txnver.json"
    if ns:
        name = name.replace(".json", f"_{ns}.json")
    return Path(cache_dir) / name


def _read_txn_version(cache_dir, ns=""):
    """Transition-config version this namespace's cache was last built under (0 = legacy)."""
    if not cache_dir:
        return TICKER_TRANSITION_VERSION      # caching off -> nothing to migrate
    p = _txn_version_path(cache_dir, ns)
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text()).get("version", 0))
    except Exception:  # noqa: BLE001
        return 0


def _write_txn_version(cache_dir, ns=""):
    if not cache_dir:
        return
    try:
        _txn_version_path(cache_dir, ns).write_text(
            json.dumps({"version": TICKER_TRANSITION_VERSION}))
    except Exception:  # noqa: BLE001
        pass


def fetch_finra_dark_volume_panel(dates, symbols, workers=8, cache_dir=None, ns=""):
    """(ShortVolume panel, TotalVolume panel) for `symbols` over `dates`.

    Reads both consolidated documents, fetches only the dates not already covered by
    BOTH (so a cache built before short volumes were recorded refetches once to fill
    them in), records holidays as all-NaN rows so they are never re-requested, then
    saves the grown documents back. One HTTP request per genuinely-new trading day.

    `ns` selects a cache namespace (its own document files) so a distinct symbol
    universe -- e.g. the Russell 2000 -- does not collide with the NDX-100 cache, whose
    rows cover the same dates but store only NDX-100 columns.
    """
    wanted = list(dict.fromkeys(symbols))  # de-dup, keep order
    dates = [d for d in dates if d >= FINRA_MIN_DATE]
    if not dates:
        return pd.DataFrame(), pd.DataFrame()

    doc_t = load_finra_document(cache_dir, "offexch", ns)
    doc_s = load_finra_document(cache_dir, "short", ns)
    # Transition-config migration: if this cache was built under an older
    # TICKER_TRANSITION_VERSION, drop the aliased/masked target columns so the backfill
    # below refetches and rebuilds them under the current maps (the alias override only runs
    # while fetching, so an already-cached stale column would otherwise never be corrected).
    if cache_dir and _read_txn_version(cache_dir, ns) < TICKER_TRANSITION_VERSION:
        stale = (set(TICKER_ALIASES) | set(TICKER_VALID_FROM)) & set(wanted)
        drop_t = [c for c in stale if c in doc_t.columns]
        drop_s = [c for c in stale if c in doc_s.columns]
        if drop_t or drop_s:
            doc_t = doc_t.drop(columns=drop_t)
            doc_s = doc_s.drop(columns=drop_s)
            print(f"FINRA cache [{ns or 'ndx'}]: transition config v{TICKER_TRANSITION_VERSION}"
                  f" -> dropping {len(set(drop_t) | set(drop_s))} target column(s) to rebuild: "
                  f"{', '.join(sorted(set(drop_t) | set(drop_s)))}", file=sys.stderr)
    # Self-heal delayed FINRA posts: an all-NaN cached row within the recency window is either a
    # genuine holiday or a day whose file simply was not up yet when we last ran. Drop those so
    # they re-enter `missing` and are re-fetched -- once FINRA publishes, the real row replaces
    # the empty one; a true holiday just comes back empty and eventually ages out of the window.
    # (Rows are all-NaN in both docs for an empty day, so dropping per-doc stays consistent via
    # the `have` intersection below.) Skipped on a full --refresh, which re-fetches everything.
    if cache_dir and (not doc_t.empty or not doc_s.empty):
        recent_cut = pd.Timestamp.today().normalize() - pd.Timedelta(days=FINRA_RECENT_REFETCH_DAYS)
        def _drop_recent_empty(doc):
            if doc.empty:
                return doc, 0
            recent = doc.index[doc.index >= recent_cut]
            empty = recent[doc.loc[recent].isna().all(axis=1)] if len(recent) else recent[:0]
            return (doc.drop(index=empty), len(empty)) if len(empty) else (doc, 0)
        doc_t, n_t = _drop_recent_empty(doc_t)
        doc_s, n_s = _drop_recent_empty(doc_s)
        if n_t or n_s:
            print(f"FINRA cache [{ns or 'ndx'}]: re-checking {max(n_t, n_s)} recent empty "
                  f"day-row(s) in case FINRA has since posted them", file=sys.stderr)

    have_cols = (set(doc_t.columns) if not doc_t.empty else set()) & \
                (set(doc_s.columns) if not doc_s.empty else set())
    new_syms = [s for s in wanted if s not in have_cols]
    have = (set(doc_t.index) if not doc_t.empty else set()) & \
           (set(doc_s.index) if not doc_s.empty else set())
    if new_syms:
        # The cached document is missing one or more requested columns entirely (e.g. the
        # symbol universe grew since this ns was last built) -- a date already present in
        # `have` still needs re-fetching to backfill those columns, since FINRA's daily file
        # covers every symbol and a per-date refetch fills in whatever the cache lacks.
        missing = list(dates)
        print(f"FINRA cache: {len(new_syms)} new symbol(s) not yet in cached document "
              f"(e.g. {', '.join(new_syms[:5])}{'...' if len(new_syms) > 5 else ''}) -- "
              f"refetching all {len(dates)} day(s) to backfill.", file=sys.stderr)
    else:
        missing = [d for d in dates if d not in have]
    print(f"FINRA cache: {len(dates)-len(missing)} of {len(dates)} day(s) already in documents"
          + (f" [{finra_doc_path(cache_dir, ns=ns)}]" if cache_dir else " (caching disabled)")
          + f"; fetching {len(missing)} new day(s)...", file=sys.stderr)

    if missing:
        session = make_session(workers) if requests else None
        total = len(missing)
        counter = {"n": 0}
        lock = threading.Lock()

        def _one(d):
            vols, resolved = fetch_finra_offexchange_volume(d.strftime("%Y%m%d"), session=session)
            with lock:
                counter["n"] += 1
                if counter["n"] % 50 == 0 or counter["n"] == total:
                    print(f"[{counter['n']:>4}/{total}] FINRA short-volume fetched", file=sys.stderr)
            return d, (vols if resolved else None)

        # Pre-resolve which wanted symbols have a ticker alias (renamed security), so the
        # per-day loop can fill their old-ticker rows without re-scanning the whole map.
        aliased = {t: TICKER_ALIASES[t] for t in wanted if t in TICKER_ALIASES}

        def _row(vols, take):
            row = {s: vols[s][take] for s in wanted if s in vols}
            for tgt, spans in aliased.items():
                applicable = [old for old, a0, a1 in spans if a0 <= d < a1]
                if not applicable:
                    continue            # native-ticker era -> keep the current string as-is
                chosen = next((o for o in applicable if o in vols), None)
                if chosen is not None:
                    row[tgt] = vols[chosen][take]   # override to the predecessor's real flow
                else:
                    row.pop(tgt, None)  # in-span but no predecessor today -> drop (never the impostor)
            return row

        rows_s, rows_t, recorded = {}, {}, []
        for d, vols in parallel_map(_one, missing, workers):
            if vols is None:            # transient failure -> leave for a future run
                continue
            recorded.append(d)          # includes holidays (empty vols) as NaN rows below
            if vols:
                rows_s[d] = _row(vols, 0)
                rows_t[d] = _row(vols, 1)
        if recorded:
            idx = pd.DatetimeIndex(sorted(recorded))
            for rows, doc, kind in ((rows_s, doc_s, "short"), (rows_t, doc_t, "offexch")):
                new_df = pd.DataFrame.from_dict(rows, orient="index").reindex(idx)
                doc2 = pd.concat([doc, new_df]) if not doc.empty else new_df
                doc2 = doc2[~doc2.index.duplicated(keep="last")].sort_index()
                save_finra_document(doc2, cache_dir, kind, ns)
                if kind == "short": doc_s = doc2
                else: doc_t = doc2
            print(f"FINRA cache: documents now cover {len(doc_t)} day(s)", file=sys.stderr)

    # Stamp the transition-config version this cache now reflects, so the migration drop
    # above fires exactly once per version bump rather than on every run.
    if cache_dir:
        _write_txn_version(cache_dir, ns)

    def _slice(doc):
        if doc.empty:
            return pd.DataFrame()
        cols = [s for s in wanted if s in doc.columns]
        out = doc.reindex(pd.DatetimeIndex(dates))[cols]
        # Drop recycled-ticker history that predates the current security (see
        # TICKER_VALID_FROM). Applied to both short and total panels so the dark
        # ratio and dollar weighting only ever see the current company's flow.
        for sym, valid_from in TICKER_VALID_FROM.items():
            if sym in out.columns:
                out.loc[out.index < valid_from, sym] = np.nan
        return out
    return _slice(doc_s), _slice(doc_t)


# --------------------------------------------------------------------------
# Split adjustment
# --------------------------------------------------------------------------
# FINRA off-exchange volume is reported in *as-traded* shares (unadjusted), while
# SqueezeMetrics' VOLUME denominator is *split-adjusted*. So across a stock split the
# ratio FINRA/VOLUME breaks by exactly the split factor: FINRA jumps (share count
# multiplies overnight) but VOLUME is continuous -> pre-split dark ratios come out too
# low by the split factor, dumping a whole cluster of points to near-zero on the
# scatter's x-axis. We detect the split from that step and rescale the pre-split FINRA
# volume back onto the adjusted basis so the ratio is continuous and correct.
SPLIT_ALLOWED = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]  # plausible split ratios
SPLIT_JUMP_THR = float(np.log(2.2))                    # min multiplicative step to flag (catches >=3:1 robustly)


def detect_split_factors(ratio, window=12, gap=6, thr=SPLIT_JUMP_THR):
    """Detect stock splits in a FINRA/VOLUME ratio series. Returns {date: factor}.

    A split shows up as a sustained upward multiplicative step (going forward in time)
    in the ratio. The factor is estimated from robust medians of clean windows either
    side of the step and snapped to the nearest plausible split ratio; ambiguous steps
    that don't land near a ratio are ignored (so genuine dark-flow regime shifts are
    not mistaken for splits).
    """
    s = ratio.dropna()
    if len(s) < 3 * window:
        return {}
    L = np.log(s.clip(lower=1e-9)).values
    idx = list(s.index)
    n = len(L)
    jump = np.full(n, -np.inf)
    for i in range(window, n - window):
        jump[i] = np.median(L[i:i + window]) - np.median(L[i - window:i])
    out = {}
    i = window
    while i < n - window:
        if jump[i] > thr:
            j = i
            while j + 1 < n - window and jump[j + 1] > thr:
                j += 1
            k = i + int(np.argmax(jump[i:j + 1]))
            before = np.median(np.exp(L[max(0, k - 31):max(1, k - gap)]))
            after = np.median(np.exp(L[k + gap:k + 31]))
            f = after / before if before > 0 else 0.0
            best = min(SPLIT_ALLOWED, key=lambda a: abs(a - f))
            if best and abs(best - f) / best < 0.22:
                # localize the actual effective date: the median-jump estimate lands ~a
                # half-window early, so walk to the first day the ratio crosses the
                # geometric midpoint of the two plateaus (robust to a lone spike via a
                # short forward median). Everything before it is treated as pre-split.
                mid = np.sqrt(max(before, 1e-12) * max(after, 1e-12))
                k_eff = k
                for t in range(max(0, k - window), min(n - 1, k + window)):
                    # first genuinely post-split day: two consecutive days above the
                    # plateau midpoint (so a lone pre-split spike can't trigger early).
                    if np.exp(L[t]) >= mid and np.exp(L[t + 1]) >= mid:
                        k_eff = t
                        break
                out[idx[k_eff]] = best
            i = j + 1
        else:
            i += 1
    return out


def _cumulative_split_factor(index, splits):
    """Series over `index`: product of the factors of splits strictly *after* each date
    (i.e. how much to scale that day's as-traded volume up onto the adjusted basis)."""
    fac = pd.Series(1.0, index=index)
    for d, f in splits.items():
        fac.loc[index < d] *= f
    return fac


def panel_split_factors(finra_panel, volume_panel):
    """Detect per-symbol split factors from the FINRA-total / consolidated-volume
    ratio. Returns {symbol: {date: factor}}; detected splits are logged to stderr.
    Detect once (on the TOTAL panel, the more liquid series) and apply the same
    factors to both the total and short panels so they stay on one basis."""
    facs = {}
    for sym in finra_panel.columns:
        if sym not in volume_panel.columns:
            continue
        vol = volume_panel[sym].reindex(finra_panel.index)
        ratio = (finra_panel[sym] / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        splits = detect_split_factors(ratio)
        if not splits:
            continue
        facs[sym] = splits
        for d, f in sorted(splits.items()):
            print(f"  split-adjusted {sym}: {f}:1 at {d.date()} "
                  f"(rescaled pre-split off-exchange volume)", file=sys.stderr)
    return facs


def apply_split_factors(panel, factors):
    """Rescale a volume panel's pre-split rows onto the adjusted basis."""
    if panel.empty or not factors:
        return panel
    adj = panel.copy()
    for sym, splits in factors.items():
        if sym in adj.columns:
            adj[sym] = adj[sym] * _cumulative_split_factor(adj.index, splits)
    return adj


def split_adjust_finra_panel(finra_panel, volume_panel):
    """Rescale each symbol's FINRA off-exchange volume onto SqueezeMetrics' split-
    adjusted basis, so the dark ratio is continuous across splits."""
    if finra_panel.empty:
        return finra_panel
    return apply_split_factors(finra_panel, panel_split_factors(finra_panel, volume_panel))


def compute_aggregate_dark_ratio(finra_offexchange_panel, volume_panel, exclude=(), min_names=20):
    """INDEX-level dark ratio: sum of off-exchange volume across all constituents
    divided by the sum of their total consolidated volume, per day -- i.e. the share
    of ALL constituent volume that traded off-exchange, volume-weighted by
    construction (a high-volume name moves it more, and both Alphabet share classes
    simply add into the sums). A name enters a day's sums only when both its FINRA
    and consolidated volumes exist; days with fewer than `min_names` contributors are
    NaN. `exclude` drops the ETF bench (its own volume is not constituent flow).
    """
    cols = [c for c in finra_offexchange_panel.columns
            if c in volume_panel.columns and c not in exclude]
    fx = finra_offexchange_panel[cols]
    vol = volume_panel.reindex(fx.index)[cols]
    mask = fx.notna() & vol.notna()
    num = fx.where(mask).sum(axis=1)
    den = vol.where(mask).sum(axis=1)
    agg = (num / den.replace(0, np.nan)).where(mask.sum(axis=1) >= min_names)
    return agg.clip(lower=0, upper=1)


def compute_dollar_dix(short_panel, offexch_panel, close_panel, exclude=(), min_names=20,
                       min_coverage=0.6):
    """Dollar-weighted DIX over a component universe, replicating SqueezeMetrics'
    construction: sum(price x FINRA ShortVolume) / sum(price x FINRA off-exchange
    TotalVolume) per day. This equals the dollar-volume-weighted average of the
    per-name DPI (short/total within the off-exchange venues) because the weighting
    volume and the DPI denominator are the same series. Volumes must already be on
    the split-adjusted basis so that adjusted-volume x adjusted-price = true dollars.
    A name enters a day's sums only when short, total AND price all exist.

    Coverage guard: a day is emitted only when its contributor count clears BOTH an
    absolute floor (`min_names`) AND `min_coverage` of the universe's own recent typical
    contributor count (trailing-63d median). This is what defuses the "partial cache"
    artifact: if the price/volume cache is only fractionally populated for the newest days
    (e.g. a throttled Yahoo pull leaves most names without a recent close), the day would
    otherwise be computed over a small, non-representative rump and read wildly off -- so
    instead it is dropped to NaN and the caller falls back to the last complete day. The
    trailing-median basis means genuinely-thin early history (names not yet listed) is not
    penalised: only a drop RELATIVE to the recent norm trips the guard.
    """
    cols = [c for c in short_panel.columns
            if c in offexch_panel.columns and c in close_panel.columns and c not in exclude]
    sh = short_panel[cols]
    tv = offexch_panel[cols]
    px = close_panel.reindex(sh.index)[cols]
    mask = sh.notna() & tv.notna() & px.notna()
    num = (sh * px).where(mask).sum(axis=1)
    den = (tv * px).where(mask).sum(axis=1)
    contrib = mask.sum(axis=1)
    typical = contrib.rolling(63, min_periods=10).median()
    ok = (contrib >= min_names) & (contrib >= min_coverage * typical.fillna(0))
    dix = (num / den.replace(0, np.nan)).where(ok)
    return dix.clip(lower=0, upper=1)




# --------------------------------------------------------------------------
# Russell 2000 (IWM) reconstruction: rebuild an index-level dollar-DIX from IWM's
# ~2,000 equity constituents (the same construction as the NDX ndx_dix), rather than
# using the IWM ETF's own SqueezeMetrics D -- which, like SPY vs SPX DIX, is nearly
# uncorrelated with the constituents' dark flow. Membership comes from iShares'
# official IWM holdings file (the canonical definition of what IWM holds).
# --------------------------------------------------------------------------
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")   # skip cash / derivative / blank rows


def _iwm_tickers_from_spreadsheetml(text):
    """Extract equity tickers from iShares' SpreadsheetML (Excel-XML) holdings document
    -- the format BlackRock's fund-document endpoint returns. Locates the 'Ticker' header
    row, then keeps rows whose 'Asset Class' is 'Equity'. Returns [] if the body is not
    parseable SpreadsheetML (e.g. an HTML gate or a plain CSV was served instead)."""
    ns = "{urn:schemas-microsoft-com:office:spreadsheet}"
    # BlackRock leaves a few raw '&' in disclaimer hyperlinks, which is invalid XML; escape
    # any ampersand not already part of an entity before handing it to the parser.
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", text)
    try:
        root = ET.fromstring(text)
    except Exception:  # noqa: BLE001
        return []

    def row_cells(row):
        out, col = {}, 0
        for c in row.findall(ns + "Cell"):
            idx = c.get(ns + "Index")           # SpreadsheetML skips empty cells via ss:Index
            col = int(idx) if idx else col + 1
            d = c.find(ns + "Data")
            out[col] = d.text if d is not None else None
        return out

    rows = [row_cells(rw) for rw in root.iter(ns + "Row")]
    hdr_i = next((i for i, rw in enumerate(rows) if "Ticker" in rw.values()), None)
    if hdr_i is None:
        return []
    hdr = rows[hdr_i]
    tcol = next((k for k, v in hdr.items() if v == "Ticker"), None)
    acol = next((k for k, v in hdr.items() if v == "Asset Class"), None)
    if tcol is None:
        return []
    tickers, seen = [], set()
    for rw in rows[hdr_i + 1:]:
        t = (rw.get(tcol) or "").strip().upper()
        ac = (rw.get(acol) or "Equity").strip().lower() if acol else "equity"
        if _TICKER_RE.match(t) and ac == "equity" and t not in seen:
            seen.add(t); tickers.append(t)
    return tickers


def _ishares_weights_from_spreadsheetml(text):
    """Extract {ticker: index weight %} from iShares SpreadsheetML holdings (the 'Weight (%)'
    column). Alphabet's GOOG (class C) weight is folded into GOOGL to match the grid's
    share-class merge. Returns {} if the weight column is absent or the body isn't parseable."""
    ns = "{urn:schemas-microsoft-com:office:spreadsheet}"
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", text)
    try:
        root = ET.fromstring(text)
    except Exception:  # noqa: BLE001
        return {}

    def row_cells(row):
        out, col = {}, 0
        for c in row.findall(ns + "Cell"):
            idx = c.get(ns + "Index")
            col = int(idx) if idx else col + 1
            d = c.find(ns + "Data")
            out[col] = d.text if d is not None else None
        return out

    rows = [row_cells(rw) for rw in root.iter(ns + "Row")]
    hdr_i = next((i for i, rw in enumerate(rows) if "Ticker" in rw.values()), None)
    if hdr_i is None:
        return {}
    hdr = rows[hdr_i]
    tcol = next((k for k, v in hdr.items() if v == "Ticker"), None)
    wcol = next((k for k, v in hdr.items() if v and v.strip().lower().startswith("weight")), None)
    acol = next((k for k, v in hdr.items() if v == "Asset Class"), None)
    if tcol is None or wcol is None:
        return {}
    wmap = {}
    for rw in rows[hdr_i + 1:]:
        t = (rw.get(tcol) or "").strip().upper()
        ac = (rw.get(acol) or "Equity").strip().lower() if acol else "equity"
        if not _TICKER_RE.match(t) or ac != "equity":
            continue
        try:
            w = float((rw.get(wcol) or "").replace(",", "").strip())
        except ValueError:
            continue
        key = "GOOGL" if t == "GOOG" else t
        wmap[key] = wmap.get(key, 0.0) + w
    return wmap


def _iwm_tickers_from_csv_text(text):
    """Extract equity tickers from an iShares IWM holdings CSV body (a multi-line
    fund-info preamble followed by a holdings table whose header row starts with
    'Ticker,'). Keeps rows whose Asset Class is 'Equity' and normalizes to plain
    symbols. Returns [] if no holdings table is present (e.g. an HTML page was served)."""
    lines = text.splitlines()
    hdr = next((i for i, ln in enumerate(lines)
                if ln.replace('"', '').startswith("Ticker,")), None)
    if hdr is None:
        return []
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    except Exception:  # noqa: BLE001
        return []
    df.columns = [str(c).strip() for c in df.columns]
    if "Ticker" not in df.columns:
        return []
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip().str.lower() == "equity"]
    tickers, seen = [], set()
    for t in df["Ticker"].astype(str):
        s = t.strip().upper()
        if not re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", s):  # skip cash/derivative/blank rows
            continue
        if s in seen:
            continue
        seen.add(s); tickers.append(s)
    return tickers




def load_iwm_holdings(path):
    """Load the Russell 2000 universe from a local file: an iShares IWM holdings document
    (SpreadsheetML/Excel-XML or CSV) or a plain ticker list (one per line, or comma/space
    separated). Returns a de-duplicated ticker list, or [] if nothing parseable."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read --iwm-holdings file {path} ({e})", file=sys.stderr)
        return []
    tickers = _iwm_tickers_from_spreadsheetml(text) or _iwm_tickers_from_csv_text(text)
    if tickers:
        print(f"IWM holdings file: {len(tickers)} equity constituents [{path}]", file=sys.stderr)
        return tickers
    seen, out = set(), []                       # fall back: treat the file as a plain list
    for tok in re.split(r"[\s,]+", text.strip()):
        s = tok.strip().upper()
        if re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", s) and s not in seen:
            seen.add(s); out.append(s)
    if out:
        print(f"IWM holdings file: {len(out)} tickers (plain list) [{path}]", file=sys.stderr)
    else:
        print(f"  ! no tickers parsed from --iwm-holdings file {path}", file=sys.stderr)
    return out








# --------------------------------------------------------------------------
# Synthetic data (for --demo / testing)
# --------------------------------------------------------------------------
def demo_panel(tickers, bench, start="2020-01-01", seed=7):
    """
    Build a plausible synthetic D panel: a common market dark-flow factor
    (the QQQ series) plus per-name beta and idiosyncratic AR(1) noise, so the
    two residual definitions differ in a meaningful, checkable way. Also
    synthesizes daily returns (and CLOSE paths, from which the 1mo/2mo/3mo forward
    returns derive) with a mild, heterogeneous (mostly negative) dependence on D,
    plus a noisier
    "raw_dark" series (D + high-frequency noise, standing in for the FINRA-
    derived unsmoothed 1-day ratio used in live mode) -- purely to make the
    "D vs forward return" tab non-trivial in demo mode. None of this is a
    claim about the real-world relationship, which the live API/FINRA data
    determines.

    History spans `start` to today (business days), matching live mode where
    SqueezeMetrics' `D` history runs far deeper than the default plot window.

    Returns a dict: {"d", "r21", "r42", "r63", "raw_dark"}.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    n_days = len(dates)

    def ar1(mu, phi, sig, n):
        x = np.empty(n); x[0] = mu
        for t in range(1, n):
            x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, sig)
        return x

    market = np.clip(ar1(0.42, 0.96, 0.010, n_days), 0.15, 0.75)  # QQQ dark-ratio
    cols = {bench: pd.Series(market, index=dates, name=bench)}

    for k, t in enumerate(tickers):
        beta = rng.uniform(0.4, 1.4)
        base = rng.uniform(0.33, 0.52)
        idio = ar1(0.0, 0.90, 0.012, n_days)
        # a couple of names get an injected idiosyncratic accumulation bump
        if k % 17 == 0:
            bump = np.linspace(0, 0.06, n_days) * np.sin(np.linspace(0, 3, n_days))
            idio = idio + bump
        series = np.clip(base + beta * (market - market.mean()) + idio, 0.05, 0.95)
        # simulate short history for a few "recent IPO" names
        if t in {"ARM", "CRWV", "NBIS", "RKLB", "ALAB", "SNDK"}:
            cut = rng.integers(120, 260)
            series[:-cut] = np.nan
        cols[t] = pd.Series(series, index=dates, name=t)

    d_panel = pd.DataFrame(cols).sort_index()

    # Synthesize DAILY returns with a mild (mostly negative) dependence on each name's
    # D z-score, build CLOSE paths from them, then derive the 1/2/3-month forward
    # returns from those closes exactly as live mode does. This keeps every view --
    # histograms, the event-time drift curve, tables -- internally consistent: the
    # drift curve at day 21 matches the 1-month histogram by construction.
    close_cols = {}
    for sym in d_panel.columns:
        s = d_panel[sym]
        z = ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).fillna(0.0)
        effect = 0.0 if sym == bench else rng.uniform(-0.15, 0.06)   # % per day
        daily = effect * z.values + rng.normal(0, 1.3, len(s))       # daily % return
        close = pd.Series(100.0 * np.cumprod(1 + daily / 100.0), index=s.index)
        close[s.isna()] = np.nan       # short-history names have no early prices
        close_cols[sym] = close
    close_panel = pd.DataFrame(close_cols).sort_index()
    r21_panel = compute_forward_return(close_panel, 21)   # 1 month
    r42_panel = compute_forward_return(close_panel, 42)   # 2 months
    r63_panel = compute_forward_return(close_panel, 63)   # 3 months

    raw_dark_cols = {}
    for sym in d_panel.columns:
        base = d_panel[sym]
        noise = ar1(0.0, 0.25, 0.09, n_days)  # low autocorrelation, high variance: single-day print
        raw = pd.Series(np.clip(base.values + noise, 0.01, 0.99), index=base.index)
        raw[base.isna()] = np.nan
        raw_dark_cols[sym] = raw
    raw_dark_panel = pd.DataFrame(raw_dark_cols).sort_index()

    # synthetic share volumes (per-name scale, lognormal daily) so the index-level
    # aggregate dark ratio (sum dark vol / sum total vol) exists in demo mode too;
    # implied off-exchange volume = ratio * volume, same identity as live data
    vol_cols = {}
    for sym in d_panel.columns:
        scale = rng.uniform(2e6, 8e7)
        v = pd.Series(scale * np.exp(rng.normal(0, 0.35, n_days)), index=dates)
        v[d_panel[sym].isna()] = np.nan
        vol_cols[sym] = v
    volume_panel = pd.DataFrame(vol_cols).sort_index()
    offexch_panel = raw_dark_panel * volume_panel      # implied off-exchange volume
    ndx_agg = compute_aggregate_dark_ratio(offexch_panel, volume_panel, exclude=(bench,))
    # implied short volume: per-name DPI (short share of off-exchange volume) uses the
    # smoothed D as its level, same identity as live data (short = DPI x off-exch vol)
    short_panel = d_panel.clip(0.05, 0.95) * offexch_panel
    ndx_dix = compute_dollar_dix(short_panel, offexch_panel, close_panel, exclude=(bench,))

    return {"d": d_panel, "r21": r21_panel, "r42": r42_panel, "r63": r63_panel,
            "close": close_panel, "raw_dark": raw_dark_panel, "ndx_agg": ndx_agg,
            "ndx_dix": ndx_dix}


# --------------------------------------------------------------------------
# Residual computation
# --------------------------------------------------------------------------
def compute_residuals(panel, bench, window=126, min_periods=40, smooth=0, bench_series=None):
    """
    Returns dict with two wide DataFrames of residuals aligned to panel.index:
      'diff'  : D_i - D_bench
      'reg'   : rolling-OLS residual of D_i on D_bench
    plus 'raw' (the aligned D levels) and metadata.

    By default the benchmark is the `bench` ticker's own D (panel[bench]). If
    `bench_series` is supplied (e.g. the reconstructed dollar-DIX), each name is
    residualized against THAT series instead; `bench` is then used only to drop that
    ticker from the residualized names (if present) and to key the benchmark column.
    `bench` need not be a column when `bench_series` is given -- e.g. the S&P 500 grid
    has no index ticker in its constituent panel. Because the residuals are only defined
    where the benchmark is, a FINRA-window reconstruction limits the residual history.
    """
    if bench_series is None and bench not in panel.columns:
        raise ValueError(f"benchmark {bench} not present in panel")
    b = (bench_series.reindex(panel.index) if bench_series is not None else panel[bench])
    names = [c for c in panel.columns if c != bench]

    # rolling stats of benchmark (shared across names)
    b_mean = b.rolling(window, min_periods=min_periods).mean()
    b_var = b.rolling(window, min_periods=min_periods).var()

    diff_cols, reg_cols = {}, {}
    for c in names:
        x = panel[c]
        if x.dropna().empty:
            continue
        # simple difference
        diff_cols[c] = x - b
        # rolling regression residual: beta = cov(x,b)/var(b); alpha = E[x]-beta*E[b]
        cov = x.rolling(window, min_periods=min_periods).cov(b)
        beta = cov / b_var
        x_mean = x.rolling(window, min_periods=min_periods).mean()
        alpha = x_mean - beta * b_mean
        reg_cols[c] = x - (alpha + beta * b)

    diff = pd.DataFrame(diff_cols, index=panel.index)
    reg = pd.DataFrame(reg_cols, index=panel.index)

    if smooth and smooth > 1:
        diff = diff.rolling(smooth, min_periods=1).mean()
        reg = reg.rolling(smooth, min_periods=1).mean()

    return {"diff": diff, "reg": reg, "raw": panel[names], "bench": b}


# --------------------------------------------------------------------------
# HTML dashboard
# --------------------------------------------------------------------------
def signed_divergence_order(cols, latest):
    """Sort columns by sign group first (all accumulating/positive names, then all
    distributing/negative names), and by magnitude descending within each group --
    rather than by |value| alone, which interleaves positive and negative names purely
    by magnitude and obscures which side a name is diverging on. Names with no latest
    value (`latest.get(c)` is None) sort last."""
    def key(c):
        v = latest.get(c)
        if v is None:
            return (2, 0.0)
        return (0, -v) if v >= 0 else (1, v)
    return sorted(cols, key=key)


def build_grid_payload(res, bench_key, bench_label, keep, weight_map=None, weight_order=None,
                       sector_map=None):
    """Build one universe's Small-multiples grid payload (diff/reg/raw data over `keep`,
    plus the four sort orders and the weight table). `res` is a compute_residuals() result;
    the benchmark series (res['bench']) is shown as a cell keyed under `bench_key`. Alphabet's
    two share classes are merged into GOOGL. `weight_map`/`weight_order` drive the optional
    index-weight ordering; without them, weight-sort falls back to the divergence order."""
    diff, reg = res["diff"], res["reg"]
    raw = res["raw"].copy()
    raw[bench_key] = res["bench"]

    def _merge_alphabet(df):
        if "GOOG" in df.columns and "GOOGL" in df.columns:
            df = df.copy()
            df["GOOGL"] = df[["GOOG", "GOOGL"]].mean(axis=1)
            df = df.drop(columns=["GOOG"])
        return df
    diff, reg, raw = _merge_alphabet(diff), _merge_alphabet(reg), _merge_alphabet(raw)

    def pack(df):
        out = {}
        for c in df.columns:
            v = df[c].loc[keep]
            if v.notna().sum() == 0:
                continue
            out[c] = [None if pd.isna(x) else round(float(x), 4) for x in v.values]
        return out

    data = {"diff": pack(diff), "reg": pack(reg), "raw": pack(raw)}

    def _latest(df):
        out = {}
        for c in df.columns:
            s = df[c].dropna()
            out[c] = float(s.iloc[-1]) if len(s) else None
        return out
    order = signed_divergence_order(list(data["reg"]), _latest(reg))
    order_diff = signed_divergence_order(list(data["diff"]), _latest(diff))
    latest_raw = {c: (v if v is not None else -1.0) for c, v in _latest(raw).items()}
    order_raw = sorted(list(data["raw"]), key=lambda c: latest_raw.get(c, -1), reverse=True)

    present = set(data["raw"])
    wmap = weight_map or {}
    if weight_order:
        order_weight = ([bench_key] if bench_key in present else []) + \
            [t for t in weight_order if t in present and t != bench_key] + \
            [t for t in present if t != bench_key and t not in wmap]
    else:  # no weight table -> fall back to divergence order (bench cell first)
        order_weight = ([bench_key] if bench_key in present else []) + \
            [c for c in order if c != bench_key]
    weights = {t: wmap[t] for t in present if t in wmap}
    smap = sector_map or {}
    sectors = {t: smap[t] for t in present if smap.get(t)}
    return {"data": data, "order": order, "order_diff": order_diff, "order_raw": order_raw,
            "order_weight": order_weight, "weights": weights, "sector_map": sectors,
            "bench": bench_key, "bench_label": bench_label,
            "dates": [d.strftime("%Y-%m-%d") for d in keep]}


def _weekly_anchored(idx, step=5):
    """Every `step`-th entry of a DatetimeIndex, anchored to the LAST element so the most
    recent session is always kept. A plain `idx[::step]` starts at position 0 and drops the
    final `step-1` sessions when the length isn't a multiple of `step` -- which left the
    downsampled S&P 500 grid a few days behind the other tabs (e.g. ending 07-15 while
    everything else showed 07-20)."""
    n = len(idx)
    if n == 0:
        return idx
    return idx[sorted(range(n - 1, -1, -step))]


def pack_name_rel(dpi_panel, adjclose_panel, keep_days=252, plot_start=None, weekly_over=378):
    """Per-name raw 1-day dark ratio ('d') + 1/2/3-month forward returns for the cell-modal
    decile bars. Bounded to everything since `plot_start` when given (to match the grid's
    full-history window), else to the most recent `keep_days` sessions. A long window
    (> `weekly_over` sessions) is downsampled to weekly (every 5th session) so a large
    universe like the S&P 500 stays light in the payload. Alphabet's two share classes are
    merged into GOOGL."""
    r21 = compute_forward_return(adjclose_panel, 21)
    r42 = compute_forward_return(adjclose_panel, 42)
    r63 = compute_forward_return(adjclose_panel, 63)

    def _mg(df):
        if "GOOG" in df.columns and "GOOGL" in df.columns:
            df = df.copy()
            df["GOOGL"] = df[["GOOG", "GOOGL"]].mean(axis=1)
            df = df.drop(columns=["GOOG"])
        return df
    dpi, r21, r42, r63 = _mg(dpi_panel), _mg(r21), _mg(r42), _mg(r63)
    idx = dpi.index
    if plot_start is not None:
        keep = idx[idx >= plot_start]
    else:
        keep = idx[-keep_days:] if len(idx) > keep_days else idx
    if len(keep) > weekly_over:
        keep = _weekly_anchored(keep)

    def pk(df):
        out = {}
        for c in df.columns:
            v = df[c].reindex(keep)
            if v.notna().sum() == 0:
                continue
            out[c] = [None if pd.isna(x) else round(float(x), 4) for x in v.values]
        return out
    return {"d": pk(dpi), "r21": pk(r21), "r42": pk(r42), "r63": pk(r63)}


def build_html(res, bench, r21_panel, r42_panel, r63_panel, close_panel, raw_dark_panel,
               ndx_agg=None, ndx_dix=None, spx=None, iwm=None, bench_label=None,
               spx_res=None, spx_rel=None, spx_weight_map=None, spx_weight_order=None,
               breadth_px=None, sector_data=None, spx_keep_days=378,
               plot_days=378, plot_start=None,
               title=None, window=126, demo=False):
    # `bench_label` is what the residuals are actually taken against (e.g. the
    # reconstructed "NDX-DIX"); `bench` remains the ticker used for forward returns.
    bench_label = bench_label or bench
    if title is None:
        title = f"NDX-100 Dark-Ratio (D) Residual vs {bench_label}"
    idx = res["diff"].index
    if plot_start is not None:
        keep = idx[idx >= plot_start]
    else:
        keep = idx[-plot_days:] if len(idx) > plot_days else idx
    dates = [d.strftime("%Y-%m-%d") for d in keep]

    def pack(df, rows=keep):
        """Serialize a wide DataFrame to {col: [values]} (JSON-friendly, 4 dp, None
        for NaN), restricted to `rows` (None -> the full index, for the relationship tab)."""
        sub = df.loc[rows] if rows is not None else df
        out = {}
        for c in sub.columns:
            v = sub[c]
            if v.notna().sum() == 0:
                continue
            out[c] = [None if pd.isna(x) else round(float(x), 4) for x in v.values]
        return out

    # NDX-100 Small-multiples grid (residuals vs the reconstructed NDX-DIX)
    ndx_grid = build_grid_payload(res, bench, bench_label, keep, NDX100_WEIGHT, NDX100,
                                  sector_map=TICKER_SECTOR)
    data = ndx_grid["data"]
    order, order_diff = ndx_grid["order"], ndx_grid["order_diff"]
    order_raw, order_weight, weights = ndx_grid["order_raw"], ndx_grid["order_weight"], ndx_grid["weights"]
    ndx_sector_map = ndx_grid["sector_map"]

    # Per-name sector labels for the S&P 500 grid: start from the static GICS map (covers the
    # NDX-100 overlap) and supplement from the broad SPDR sector-fund holdings when a live
    # sector build is present. Only the eight broad funds are used (specialty funds like
    # SOXX/XBI are skipped) so each name resolves to its broad GICS sector.
    spx_sector_map = dict(TICKER_SECTOR)
    if sector_data and sector_data.get("members"):
        # `members` is a list of (etf, sector_name, [tickers], level, parent) tuples.
        broad = {"XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU"}
        for etf, sec_name, syms, *_ in sector_data["members"]:
            if etf not in broad:
                continue
            for t in syms:
                for key in (str(t).strip().upper(), to_yahoo_symbol(str(t))):
                    spx_sector_map.setdefault(key, sec_name)

    # S&P 500 Small-multiples grid: same construction over the IVV constituents, residualized
    # vs the S&P 500 dollar-DIX. Capped to the most recent `spx_keep_days` sessions to bound
    # the payload size (500 names x full history would bloat the HTML).
    spx_grid = None
    if spx_res is not None and not spx_res["reg"].dropna(how="all").empty:
        sidx = spx_res["reg"].index
        spx_keep = sidx[sidx >= plot_start] if plot_start is not None else sidx
        # Match NDX's full-history window (since plot_start). Rather than truncating to the
        # most recent spx_keep_days sessions (which made the S&P panels start ~1.5y ago while
        # NDX ran from 2020), downsample the 500-name grid sparklines to weekly (every 5th
        # session) when the window is long, so full history stays light in the payload.
        if len(spx_keep) > spx_keep_days:
            spx_keep = _weekly_anchored(spx_keep)
        spx_grid = build_grid_payload(spx_res, "SPX-DIX", "SPX-DIX", spx_keep,
                                      spx_weight_map, spx_weight_order,
                                      sector_map=spx_sector_map)

    # Relationship tab: x-axis is the raw (unsmoothed) 1-day dark ratio derived from
    # FINRA's daily off-exchange volume, NOT the 5-day-MA `D` used everywhere else --
    # so it has its own (shorter, FINRA-bounded) date index. r21/r42/r63 are reindexed to
    # that same index so positional pairing in the JS lines up to the same dates.
    # Restrict the raw-D relationship panel to the grid's names, but re-include GOOG (the
    # grid already merged Alphabet and dropped it) so the merge just below can still average
    # the two share classes here.
    rel_src = list(data["raw"])
    if "GOOGL" in rel_src and "GOOG" in raw_dark_panel.columns and "GOOG" not in rel_src:
        rel_src.append("GOOG")
    dark = raw_dark_panel.reindex(columns=rel_src)
    # Alphabet trades as two near-identical share classes whose dark ratios move in
    # tandem, so the raw-D studies would double-count the same flow signal (and its
    # events would fire twice). Merge them: GOOGL becomes the simple average of the
    # two ratios (they track within noise) and GOOG is dropped from the relationship
    # payload entirely (returns/closes kept under GOOGL). The Small-multiples grid now
    # merges Alphabet the same way.
    if "GOOG" in dark.columns and "GOOGL" in dark.columns:
        dark["GOOGL"] = dark[["GOOG", "GOOGL"]].mean(axis=1)
        dark = dark.drop(columns=["GOOG"])
    rel_cols = dark.columns
    rel = {
        "d": pack(dark, None),
        "r21": pack(r21_panel.reindex(index=dark.index, columns=rel_cols), None),
        "r42": pack(r42_panel.reindex(index=dark.index, columns=rel_cols), None),
        "r63": pack(r63_panel.reindex(index=dark.index, columns=rel_cols), None),
        "close": pack(close_panel.reindex(index=dark.index, columns=rel_cols), None),
        # index-level aggregate dark ratio: sum(off-exch vol)/sum(total vol) per day
        "ndx_agg": ([None if pd.isna(x) else round(float(x), 4)
                     for x in ndx_agg.reindex(dark.index).values]
                    if ndx_agg is not None else None),
        # dollar-weighted DIX replica: sum($ short vol)/sum($ off-exch vol) per day
        "ndx_dix": ([None if pd.isna(x) else round(float(x), 4)
                     for x in ndx_dix.reindex(dark.index).values]
                    if ndx_dix is not None else None),
        "dates": [d.strftime("%Y-%m-%d") for d in dark.index],  # for regime splits
        "range": [dark.index.min().strftime("%Y-%m-%d"), dark.index.max().strftime("%Y-%m-%d")]
                 if len(dark.index) else None,
    }

    sectors_payload = (build_sector_payload(sector_data["members"], sector_data["short"],
                                            sector_data["total"], sector_data["close"],
                                            sector_data["d"], keep)
                       if sector_data else None)
    # Decile source for the sector drill-down: pack raw-D + forward returns for just the names
    # actually shown in the sector modals (top constituents by dark-dollar share), from the
    # sector universe's own panels -- so every clickable constituent has the same decile view
    # as the grid, without packing the full ~1,500-name sector union into the payload.
    sector_rel = None
    if sectors_payload and sector_data and sector_data.get("dpi") is not None:
        shown = {n["t"] for it in sectors_payload["items"] for n in it["names"]}
        dpi_s, adj_s = sector_data["dpi"], sector_data["adjclose"]
        cols = [c for c in shown if c in dpi_s.columns and c in adj_s.columns]
        if cols:
            sector_rel = pack_name_rel(dpi_s[cols], adj_s[cols], plot_start=plot_start)

    payload = {
        "dates": dates,
        "data": data,
        "order": order,
        "order_diff": order_diff,
        "order_raw": order_raw,
        "order_weight": order_weight,
        "weights": weights,
        "sector_map": ndx_sector_map,
        "spx_grid": spx_grid,
        "spx_rel": spx_rel,
        "sectors": sectors_payload,
        # Per-name raw-D -> forward-return deciles for the constituents shown in the sector
        # drill-down modals, so an individual stock there opens the same 1/2/3-month decile
        # view as the small-multiples grid. Restricted to the displayed names to bound size.
        "sector_rel": sector_rel,
        "rel": rel,
        "spx": spx,
        "iwm": iwm,
        "bench": bench,
        "bench_label": bench_label,
        "window": window,
        "demo": demo,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "title": title,
    }
    blob = json.dumps(payload, separators=(",", ":"))

    return HTML_TEMPLATE.replace("/*__DATA__*/", blob)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NDX-100 Dark-Ratio Residual</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#0f141b; --grid:#232b36;
    --ink:#e6edf3; --mut:#8b949e; --pos:#3fb950; --neg:#f85149; --zero:#30363d;
    --accent:#58a6ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{padding:18px 22px 10px;border-bottom:1px solid var(--grid);
         position:sticky;top:0;background:linear-gradient(180deg,#0d1117,#0d1117f2);z-index:5;backdrop-filter:blur(4px)}
  h1{margin:0 0 3px;font-size:17px;font-weight:650;letter-spacing:.2px}
  .sub{color:var(--mut);font-size:12px}
  .controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-top:12px}
  .seg{display:inline-flex;border:1px solid var(--grid);border-radius:8px;overflow:hidden}
  .seg button{background:var(--panel);color:var(--mut);border:0;padding:6px 12px;
              font-size:12px;cursor:pointer;font-weight:600}
  .seg button.on{background:var(--accent);color:#0d1117}
  .seg button+button{border-left:1px solid var(--grid)}
  label.chk{color:var(--mut);display:inline-flex;gap:6px;align-items:center;cursor:pointer;user-select:none}
  input[type=search]{background:var(--panel2);border:1px solid var(--grid);color:var(--ink);
                     border-radius:8px;padding:6px 10px;font-size:12px;width:150px}
  .legend{color:var(--mut);font-size:11px;display:flex;gap:14px;align-items:center}
  .dot{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:middle;margin-right:4px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;padding:16px 22px 60px}
  .cell{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:8px 9px 6px;cursor:pointer}
  .cell.hot{border-color:#3d2b2f}
  .cell:hover,.cell.hot:hover{border-color:var(--accent)}
  .crow:hover .hit{fill:rgba(88,166,255,0.10)}
  .chead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
  .tkr{font-weight:700;font-size:13px;letter-spacing:.3px}
  .wt{color:var(--mut);font-weight:500;font-size:10px;margin-left:5px;font-variant-numeric:tabular-nums}
  .val{font-variant-numeric:tabular-nums;font-weight:650}
  .val.p{color:var(--pos)} .val.n{color:var(--neg)}
  svg{display:block;width:100%;height:64px}
  .meta{color:var(--mut);font-size:10.5px;display:flex;justify-content:space-between;align-items:baseline;gap:4px;margin-top:2px}
  .meta>span:first-child,.meta>span:last-child{flex:none;white-space:nowrap}
  .meta .sec{flex:1;min-width:0;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;color:var(--ink);opacity:.65}
  footer{position:fixed;bottom:0;left:0;right:0;background:#0d1117e8;border-top:1px solid var(--grid);
         padding:6px 22px;color:var(--mut);font-size:11px;display:flex;justify-content:space-between}
  a{color:var(--accent);text-decoration:none}
  .tabs{display:flex;gap:6px;margin-bottom:12px;overflow-x:auto;-webkit-overflow-scrolling:touch;
        scrollbar-width:none;-ms-overflow-style:none}
  .tabs::-webkit-scrollbar{display:none}
  .tabs button{background:transparent;color:var(--mut);border:0;border-bottom:2px solid transparent;
               padding:4px 2px;margin-right:16px;font-size:13px;font-weight:650;cursor:pointer;
               white-space:nowrap;flex:0 0 auto}
  .tabs button:last-child{margin-right:22px}
  .tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
  select{background:var(--panel2);border:1px solid var(--grid);color:var(--ink);
         border-radius:8px;padding:6px 10px;font-size:12px}
  .stats{display:flex;gap:18px;color:var(--mut);font-size:11.5px;margin:4px 22px 0}
  .stats b{color:var(--ink);font-variant-numeric:tabular-nums}
  .rel-wrap{padding:16px 22px 60px;display:flex;flex-direction:column;gap:22px}
  .rel-card{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:14px 16px}
  .rel-card h2{margin:0 0 10px;font-size:12.5px;color:var(--mut);font-weight:650;letter-spacing:.3px;text-transform:uppercase}
  .bars{display:block;width:100%;height:220px}
  .scatter{display:block;width:100%;height:320px}
  .bar{cursor:pointer}
  .bar:hover{filter:brightness(1.35)}
  .dot{opacity:0.5;transition:opacity .12s}
  .dot.dim{opacity:0.06}
  .dot.hi{opacity:1;transform-box:fill-box;transform-origin:center;transform:scale(2.4);stroke:var(--ink);stroke-width:0.6}
  .vanilla-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
  .ev-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:22px;padding:0 22px}
  .ev-table{border-collapse:collapse;font-size:11.5px;width:100%;min-width:480px}
  .ev-table th,.ev-table td{padding:5px 12px;text-align:right;border-bottom:1px solid var(--grid);
                            font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.25}
  .ev-table th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--panel);z-index:1}
  .ev-table th.sortable{cursor:pointer;user-select:none}
  .ev-table th.sortable:hover{color:var(--ink)}
  .ev-table th:first-child,.ev-table td:first-child{text-align:left;font-weight:600}
  .ev-table td .mut{color:var(--mut);font-size:9.5px}
  .ev-table td .p{color:var(--pos)} .ev-table td .n{color:var(--neg)}
  .ev-table tbody tr:hover td{background:var(--panel2)}
  .ev-table tr.baserow td{border-bottom:2px solid var(--grid);color:var(--mut)}
  .ev-table tr.baserow td:first-child{font-style:italic}
  .panel-stats{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:11.5px;margin-top:8px}
  .panel-stats b{color:var(--ink);font-variant-numeric:tabular-nums}
  @media (max-width:900px){ .vanilla-grid{grid-template-columns:1fr} }
  .dtbl{border-collapse:collapse;font-size:10.5px;font-variant-numeric:tabular-nums}
  .dtbl th,.dtbl td{padding:3px 7px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
  .dtbl thead th{position:sticky;top:0;background:var(--panel);z-index:1;color:var(--mut);font-weight:600}
  .dtbl th:first-child,.dtbl td.tk{text-align:left;position:sticky;left:0;background:var(--panel);font-weight:700}
  .dtbl th:first-child{z-index:2}
  .dtbl td.now{color:var(--mut);font-weight:600}
  .dtbl td .m{display:block}
  .dtbl td .s{display:block;color:var(--mut);font-size:9px}
  .dtbl td.pos .m{color:var(--pos)}
  .dtbl td.neg .m{color:var(--neg)}
  .dtbl td.cur{outline:1.5px solid var(--accent);outline-offset:-2px;border-radius:4px}
  .dtbl td.na{color:var(--mut)}
  .dtbl tbody tr:hover td{background:var(--grid)}
  .dtbl tbody tr:hover td.tk{background:var(--grid)}
  .overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;
           align-items:flex-start;justify-content:center;z-index:50;padding:5vh 20px;overflow:auto}
  .overlay.on{display:flex}
  /* the per-name detail modal must sit ABOVE the sector drill-down it can open over,
     so the performance plane is on top for viewing and exiting (both are .overlay = z50,
     and #sectorOverlay is later in the DOM, so it would otherwise paint on top) */
  #overlay{z-index:60}
  .modal{background:var(--panel);border:1px solid var(--grid);border-radius:12px;
         max-width:1280px;width:100%;padding:18px 22px 22px}
  .modal-head{display:flex;justify-content:space-between;align-items:baseline}
  .modal-head .tkr{font-size:18px;font-weight:700;letter-spacing:.3px}
  .modal-head .val{font-size:15px;margin-left:8px}
  .modal-close{background:transparent;border:0;color:var(--mut);font-size:22px;line-height:1;
               cursor:pointer;padding:2px 4px}
  .modal-close:hover{color:var(--ink)}
  .modal-sub{color:var(--mut);font-size:11px;display:flex;justify-content:space-between;margin:2px 0 10px}
  #mSpark svg{height:170px}
  .modal-rel{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-top:18px}
  .modal-rel h3{margin:0 0 6px;font-size:11px;color:var(--mut);font-weight:650;
                text-transform:uppercase;letter-spacing:.3px}
  .modal-rel .stat{color:var(--mut);font-size:11px;margin-top:4px}
  .modal-rel .stat b{color:var(--ink);font-variant-numeric:tabular-nums}
  @media (max-width:900px){ .modal-rel{grid-template-columns:1fr} }
  .modal-empty{color:var(--mut);font-size:12px;padding:30px 0;text-align:center}
</style>
</head>
<body>
<header>
  <h1 id="ttl">NDX-100 Dark-Ratio (D) Residual vs QQQ</h1>
  <div class="sub" id="sub"></div>
  <div class="tabs" id="tabs">
    <button data-t="grid" class="on">Small multiples</button>
    <button data-t="rel">D vs forward return</button>
    <button data-t="idx">DIX vs Return</button>
    <button data-t="xs">Cross-sectional L/S</button>
    <button data-t="ev">D-streak events</button>
    <button data-t="sectors">Sector DIX</button>
    <button data-t="spxtbl">SP500 decile table</button>
  </div>
  <div class="controls" id="ctl-grid">
    <div class="seg" id="univ" title="which index's constituents to show as small multiples">
      <button data-u="ndx" class="on">NDX-100</button>
      <button data-u="spx">S&amp;P 500</button>
    </div>
    <div class="seg" id="mode">
      <button data-m="reg" class="on">Regression residual</button>
      <button data-m="raw">Raw D</button>
    </div>
    <div class="seg" id="sort" title="order of the small-multiple panels">
      <button data-s="weight" class="on">Weight</button>
      <button data-s="div">Divergence</button>
    </div>
    <label class="chk"><input type="checkbox" id="zero" checked/> shared y-scale</label>
    <input type="search" id="q" placeholder="filter ticker..."/>
    <div class="legend" id="legend"></div>
  </div>
  <div class="controls" id="ctl-rel" style="display:none">
    <div class="seg" id="horizon">
      <button data-h="21" class="on">1mo</button>
      <button data-h="42">2mo</button>
      <button data-h="63">3mo</button>
    </div>
    <label class="chk">ticker
      <select id="relTkr"><option value="__ALL__">All names (pooled)</option></select>
    </label>
    <label class="chk" title="z-score each name's dark ratio over the window before pooling, so buckets reflect 'unusually dark for this name' rather than structurally-dark names"><input type="checkbox" id="zscore"/> z-score D per name</label>
    <label class="chk" title="subtract QQQ's forward return from each name's, stripping the market-beta component"><input type="checkbox" id="excess"/> excess vs QQQ</label>
    <label class="chk" title="use the median forward return per decile instead of the mean (robust to outliers)"><input type="checkbox" id="median"/> median</label>
    <div class="legend"><span id="relLegend"></span></div>
  </div>
  <div class="controls" id="ctl-idx" style="display:none">
    <div class="seg" id="idxSel" title="which index DIX-vs-return views to show -- toggle any combination (one or several)">
      <button data-i="ndx" class="on">NDX-100</button>
      <button data-i="spx">S&amp;P 500</button>
      <button data-i="iwm">IWM</button>
    </div>
    <span class="sub" style="align-self:center">toggle one or several indices to compare</span>
  </div>
  <div class="controls" id="ctl-spx" style="display:none">
    <label class="chk" title="force all three horizon panels onto one shared y-axis (and scatter x/y range) so the magnitude of the effect is directly comparable across 1mo / 2mo / 3mo"><input type="checkbox" id="spxShared"/> shared axis · SPX panels</label>
  </div>
  <div class="controls" id="ctl-ndx" style="display:none">
    <div class="seg" id="ndxModeSeg" title="how the index-level dark ratio is built each day">
      <button data-m="dix" class="on" title="dollar-weighted DIX replica: sum(price x FINRA short volume) / sum(price x FINRA off-exchange total volume) -- SqueezeMetrics' DIX construction applied to the NDX-100 universe">NDX DIX (&Sigma;$ short &divide; &Sigma;$ dark)</button>
      <button data-m="agg" title="sum of all constituents' off-exchange volume divided by the sum of their total consolidated volume -- volume-weighted dark SHARE (different numerator than DIX)">&Sigma; dark vol &divide; &Sigma; total vol</button>
      <button data-m="mean" title="equal-weight average of the ~100 per-name ratios -- every name counts the same regardless of its volume">Equal-weight mean of ratios</button>
    </div>
    <label class="chk" title="force all three horizon panels onto one shared y-axis (and scatter x/y range) so the magnitude of the effect is directly comparable across 1mo / 2mo / 3mo"><input type="checkbox" id="ndxShared"/> shared axis · NDX panels</label>
  </div>
  <div class="controls" id="ctl-iwm" style="display:none">
    <label class="chk" title="force all three horizon panels onto one shared y-axis (and scatter x/y range) so the magnitude of the effect is directly comparable across 1mo / 2mo / 3mo"><input type="checkbox" id="iwmShared"/> shared axis · IWM panels</label>
  </div>
  <div class="controls" id="ctl-xs" style="display:none">
    <div class="seg" id="xsHorizonSeg" title="forward-return horizon">
      <button data-h="21" class="on">1mo</button>
      <button data-h="42">2mo</button>
      <button data-h="63">3mo</button>
    </div>
    <div class="seg" id="xsSignalSeg" title="how names are ranked cross-sectionally each day">
      <button data-sig="rel" class="on">Name-specific (vs own avg)</button>
      <button data-sig="raw">Raw D (structural)</button>
    </div>
  </div>
  <div class="controls" id="ctl-ev" style="display:none">
    <label class="chk" title="restrict the study to a single name (events, baseline and placebos all scoped to it)">ticker
      <select id="evTkr" style="margin-left:4px"><option value="__ALL__">All names (pooled)</option></select>
    </label>
    <label class="chk" title="a streak of this many consecutive days with D in the decile band triggers one event on the final day">streak length
      <input type="number" id="evRun" min="2" max="30" value="5" style="width:52px;background:var(--panel2);border:1px solid var(--grid);color:var(--ink);border-radius:6px;padding:5px 6px;font-size:12px;margin-left:4px"/>
    </label>
    <label class="chk">decile band
      <select id="evLo" style="margin-left:4px"></select>
      <span style="color:var(--mut)">to</span>
      <select id="evHi"></select>
    </label>
    <div class="seg" id="evBasisSeg" title="which distribution defines the decile cutoffs">
      <button data-b="trail" class="on" title="each day's D ranked against that name's previous 252 observations (min 120) -- cutoffs knowable in real time">Per-name trailing (no look-ahead)</button>
      <button data-b="name" title="each name's full-sample distribution -- mild look-ahead, kept for comparison">Per-name full-sample</button>
      <button data-b="pool" title="one global distribution pooled across names -- absolute dark levels">Pooled (global)</button>
    </div>
    <label class="chk" title="measure forward return in excess of QQQ over the same window"><input type="checkbox" id="evExcess"/> excess vs QQQ</label>
    <label class="chk" title="one-way transaction cost per trade side, in basis points (backtest only)">cost bps/side
      <input type="number" id="evCost" min="0" max="100" value="5" style="width:52px;background:var(--panel2);border:1px solid var(--grid);color:var(--ink);border-radius:6px;padding:5px 6px;font-size:12px;margin-left:4px"/>
    </label>
  </div>
  <div class="stats" id="relStats" style="display:none"></div>
</header>
<div class="grid" id="grid"></div>
<div class="rel-wrap" id="relWrap" style="display:none">
  <div class="rel-card">
    <h2 id="barsTitle">Average 1mo forward return by D decile</h2>
    <svg id="barsSvg" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
  </div>
  <div class="rel-card" id="scatterCard" style="display:none">
    <h2 id="scatterTitle">D vs 1mo forward return (daily observations)</h2>
    <svg id="scatterSvg" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
  </div>
</div>
<div class="rel-wrap" id="spxWrap" style="display:none">
  <div class="sub" id="spxSub" style="margin:0 22px 4px">S&amp;P 500 reconstructed dollar-DIX (IVV constituents, FINRA) vs SPX forward return</div>
  <div class="vanilla-grid" id="spxGrid">
    <div class="rel-card">
      <h2>DIX vs 1-month SPX forward return <span style="color:var(--mut);font-weight:400">(21 trading days)</span></h2>
      <svg id="sBars21" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="sScatter21" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="sStats21"></div>
    </div>
    <div class="rel-card">
      <h2>DIX vs 2-month SPX forward return <span style="color:var(--mut);font-weight:400">(42 trading days)</span></h2>
      <svg id="sBars42" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="sScatter42" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="sStats42"></div>
    </div>
    <div class="rel-card">
      <h2>DIX vs 3-month SPX forward return <span style="color:var(--mut);font-weight:400">(63 trading days)</span></h2>
      <svg id="sBars63" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="sScatter63" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="sStats63"></div>
    </div>
  </div>
  <div class="rel-card" style="margin-top:22px">
    <h2>Cross-universe check &middot; NDX-100 vs S&amp;P 500 dollar-DIX</h2>
    <svg id="spxCmp" class="scatter" viewBox="0 0 800 300" preserveAspectRatio="none" style="height:300px"></svg>
    <div class="panel-stats" id="spxCmpStats"></div>
    <div class="sub" style="margin-top:8px;font-size:11px;line-height:1.5">
      Both series are the dollar-weighted DPI = &Sigma;(price &times; FINRA short volume) &divide;
      &Sigma;(price &times; FINRA off-exchange total volume) across each index's components each day
      -- the NDX-100 (this dashboard's benchmark) vs the S&amp;P 500 (the SPX tab). Expect a level
      offset and tech-led divergences: judge by correlation and day-to-day co-movement. Tight
      co-movement confirms the construction behaves consistently across universes.
    </div>
  </div>
</div>
<div class="rel-wrap" id="ndxWrap" style="display:none">
  <div class="sub" id="ndxSub" style="margin:0 22px 4px"></div>
  <div class="vanilla-grid">
    <div class="rel-card">
      <h2>NDX avg dark ratio vs 1-month QQQ forward return <span style="color:var(--mut);font-weight:400">(21 trading days)</span></h2>
      <svg id="nBars21" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="nScatter21" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="nStats21"></div>
    </div>
    <div class="rel-card">
      <h2>NDX avg dark ratio vs 2-month QQQ forward return <span style="color:var(--mut);font-weight:400">(42 trading days)</span></h2>
      <svg id="nBars42" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="nScatter42" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="nStats42"></div>
    </div>
    <div class="rel-card">
      <h2>NDX avg dark ratio vs 3-month QQQ forward return <span style="color:var(--mut);font-weight:400">(63 trading days)</span></h2>
      <svg id="nBars63" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="nScatter63" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="nStats63"></div>
    </div>
  </div>
</div>
<div class="rel-wrap" id="iwmWrap" style="display:none">
  <div class="sub" id="iwmSub" style="margin:0 22px 4px"></div>
  <div class="vanilla-grid" id="iwmGrid">
    <div class="rel-card">
      <h2>IWM D vs 1-month IWM forward return <span style="color:var(--mut);font-weight:400">(21 trading days)</span></h2>
      <svg id="wBars21" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="wScatter21" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="wStats21"></div>
    </div>
    <div class="rel-card">
      <h2>IWM D vs 2-month IWM forward return <span style="color:var(--mut);font-weight:400">(42 trading days)</span></h2>
      <svg id="wBars42" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="wScatter42" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="wStats42"></div>
    </div>
    <div class="rel-card">
      <h2>IWM D vs 3-month IWM forward return <span style="color:var(--mut);font-weight:400">(63 trading days)</span></h2>
      <svg id="wBars63" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <svg id="wScatter63" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="wStats63"></div>
    </div>
  </div>
</div>
<div class="rel-wrap" id="xsWrap" style="display:none">
  <div class="sub" id="xsSub" style="margin:0 22px 4px"></div>
  <div class="rel-card" style="margin:0 22px">
    <h2>Forward excess return by cross-sectional dark decile <span id="xsHdr" style="color:var(--mut);font-weight:400"></span></h2>
    <svg id="xsBars" class="bars" viewBox="0 0 800 240" preserveAspectRatio="none" style="height:260px"></svg>
    <div class="panel-stats" id="xsStats"></div>
    <div class="sub" style="margin-top:10px;font-size:11px;line-height:1.55">
      Each day the ~100 names are ranked by <b id="xsSigLbl"></b> and split into deciles
      (D1 = least dark, D10 = most dark); each bar is the mean forward return <i>in excess of
      QQQ</i> for names in that decile, pooled across all days. Long-short (D10&minus;D1) is
      market-neutral by construction. Whiskers = &plusmn;1 SE using an overlap-adjusted
      effective N (daily observations of an h-day return are not independent). Ranking against
      each name's own expanding average (not QQQ) is what makes the signal name-specific
      cross-sectionally &mdash; subtracting QQQ's dark ratio is a per-day constant and would
      not change the ranking.
    </div>
  </div>
</div>
<div class="rel-wrap" id="evWrap" style="display:none">
  <div class="sub" id="evSub" style="margin:0 22px 4px"></div>
  <div id="evNow" style="margin:0 22px 10px;padding:8px 12px;border:1px solid var(--grid);border-radius:8px;font-size:12px;line-height:1.9;color:var(--ink)"></div>
  <div class="ev-grid">
    <div class="rel-card">
      <h2>1-month forward return after event</h2>
      <svg id="evHist21" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="evStats21"></div>
    </div>
    <div class="rel-card">
      <h2>2-month forward return after event</h2>
      <svg id="evHist42" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="evStats42"></div>
    </div>
    <div class="rel-card">
      <h2>3-month forward return after event</h2>
      <svg id="evHist63" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      <div class="panel-stats" id="evStats63"></div>
    </div>
  </div>
  <div class="rel-card" style="margin:20px 22px 0">
    <h2>Event-time drift &middot; average cumulative return, day 0 &rarr; +63</h2>
    <svg id="evDrift" class="scatter" viewBox="0 0 800 300" preserveAspectRatio="none" style="height:300px"></svg>
    <div class="panel-stats" id="evDriftStats"></div>
  </div>
  <div class="rel-card" style="margin:20px 22px 0">
    <h2>Regime robustness &middot; event returns by sub-period</h2>
    <div style="overflow-x:auto"><table id="evRegime" class="ev-table"></table></div>
  </div>
  <div class="rel-card" style="margin:20px 22px 0">
    <h2>T+1 strategy backtest &middot; enter at the close AFTER the streak completes, hold 1/2/3 months</h2>
    <svg id="evBt" class="scatter" viewBox="0 0 800 320" preserveAspectRatio="none"></svg>
    <div style="overflow-x:auto;margin-top:8px"><table id="evBtStats" class="ev-table"></table></div>
    <div class="sub" style="margin-top:8px;font-size:11px;line-height:1.5">
      FINRA's daily file publishes after the close, so a streak completing on day T is tradeable
      at T+1's close at the earliest -- that lag is baked in. Equal weight across all open
      positions, rebalanced daily; cash (0%) when no position is open. Costs are charged per
      side at entry and exit. The <b>T+0 total</b> column shows the same strategy executed at
      the (unattainable) completion-day close -- the gap between the two columns is the price
      of the one-day lag. With "excess vs QQQ" on, positions are long stock / short QQQ
      (market-neutral); otherwise long-only.
    </div>
  </div>
  <div style="padding:0 22px;margin-top:20px">
    <h2 style="font-size:12.5px;color:var(--mut);font-weight:650;text-transform:uppercase;letter-spacing:.3px;margin:0 0 8px">
      All 55 decile bands &middot; mean forward return &amp; MAD</h2>
    <div style="overflow-x:auto;max-height:560px;overflow-y:auto"><table id="evTable" class="ev-table"></table></div>
  </div>
  <div class="sub" id="evNote" style="margin:12px 22px 0;font-size:11px;line-height:1.55"></div>
</div>
<div class="rel-wrap" id="sectorsWrap" style="display:none">
  <div style="margin:0 22px 8px">
    <span class="seg" id="sectorLevelSeg">
      <button data-lvl="sector" class="on">Sectors</button>
      <button data-lvl="subsector">Subsectors</button>
    </span>
  </div>
  <div class="sub" id="sectorsSub" style="margin:0 22px 4px"></div>
  <div class="rel-card" style="margin:0 22px 14px">
    <h2 id="sectorRankTitle">Dark accumulation by sector &middot; reconstructed dollar-DIX, ranked by 1-year percentile</h2>
    <svg id="sectorRank" viewBox="0 0 900 300" preserveAspectRatio="none" style="width:100%;height:300px"></svg>
  </div>
  <div id="sectorGrid" class="grid" style="padding-top:0"></div>
  <div class="sub" style="margin:2px 22px 40px;font-size:11px;line-height:1.55">
    Each sector's DIX = &Sigma;(price &times; FINRA off-exchange short volume) &divide;
    &Sigma;(price &times; off-exchange total volume) over that ETF's constituents (5-day MA) &mdash;
    the same dollar-weighted construction as the SPX / IWM tabs, applied per sector. The ranking bar
    is each sector's latest DIX percentile within its own trailing year: high (green) = dark
    accumulation elevated vs its own history, low (red) = distribution. Sparklines show the DIX level
    over the plot window (&#9650;/&#9660; = 20-day change). The dashed green/red lines are the
    trailing-year 80th / 20th percentile band; a &#9650; marks where the DIX crossed <em>into</em>
    the top of that band (accumulation entering the top of its own 1-year range) and a &#9660; where
    it crossed <em>into</em> the bottom &mdash; with the most recent such crossing (&#9650;P80 /
    &#9660;P20 + date) shown in the cell footer. Constituents: SPDR Select Sector funds
    (State Street) plus SOXX (iShares). Use the <b>Sectors / Subsectors</b> toggle to switch
    between the broad GICS sectors and finer SPDR S&amp;P industry funds (e.g. Homebuilders,
    Retail, Oil&nbsp;&amp;&nbsp;Gas&nbsp;E&amp;P, Regional Banks), which use the identical
    construction. Cross-sector correlation is moderate (~0.5) &mdash; sectors
    share a common dark-flow component but carry distinct signals.
  </div>
</div>
<div class="rel-wrap" id="spxtblWrap" style="display:none">
  <div class="sub" id="spxtblSub" style="margin:0 22px 4px"></div>
  <div style="margin:0 22px 8px">
    <input id="spxtblFilter" type="search" placeholder="filter ticker..." autocomplete="off"
           style="background:var(--panel);border:1px solid var(--grid);color:var(--ink);border-radius:7px;padding:5px 10px;font-size:12px;width:180px">
  </div>
  <div id="spxtblBody" style="margin:0 22px 20px;max-height:72vh;overflow:auto;border:1px solid var(--grid);border-radius:8px"></div>
  <div class="sub" style="margin:2px 22px 40px;font-size:11px;line-height:1.55">
    For each S&amp;P 500 name, its history is bucketed into deciles of its own dark ratio D (D1 = least dark,
    D10 = most dark), and each cell shows the <b>mean forward 1-month (21-day) return</b> when D sat in that
    decile, with <b>&plusmn;1 standard error</b> below it &mdash; the whisker level, using the overlap-adjusted
    effective sample (n&divide;21) so autocorrelated monthly returns aren't over-counted. The <b>now</b> column
    is the decile D sits in today; that cell is outlined. Green/red = positive/negative mean return. This is a
    within-name, time-series relationship (each stock vs its own history), not a cross-sectional signal.
  </div>
</div>
<footer>
  <span id="foot"></span>
  <span id="footHint">ordered by NDX index weight &middot; toggle Weight/Divergence to reorder &middot; hover a panel for date/value</span>
</footer>

<div class="overlay" id="overlay">
  <div class="modal" id="modal">
    <div class="modal-head">
      <div><span class="tkr" id="mTkr"></span><span class="val" id="mVal"></span></div>
      <button class="modal-close" id="mClose" aria-label="close">&times;</button>
    </div>
    <div class="modal-sub" id="mSub"></div>
    <div id="mSpark"></div>
    <div id="mRel" class="modal-rel">
      <div>
        <h3>Forward return by decile of this name's raw D &middot; 1mo</h3>
        <svg id="mBars21" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
        <div class="stat" id="mStat21"></div>
      </div>
      <div>
        <h3>2mo</h3>
        <svg id="mBars42" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
        <div class="stat" id="mStat42"></div>
      </div>
      <div>
        <h3>3mo</h3>
        <svg id="mBars63" class="bars" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
        <div class="stat" id="mStat63"></div>
      </div>
    </div>
  </div>
</div>

<div class="overlay" id="sectorOverlay">
  <div class="modal" id="sectorModal">
    <div class="modal-head">
      <div><span class="tkr" id="secmTkr"></span><span class="val" id="secmVal"></span></div>
      <button class="modal-close" id="secmClose" aria-label="close">&times;</button>
    </div>
    <div class="modal-sub" id="secmSub"></div>
    <svg id="secmBars" viewBox="0 0 900 640" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto"></svg>
    <div class="sub" id="secmNote" style="font-size:11px;line-height:1.5;margin-top:6px"></div>
  </div>
</div>

<script>
const P = /*__DATA__*/;
const grid = document.getElementById('grid');
let mode = 'reg', shared = true, filter = '', sortMode = 'weight';
let gridUniv = 'ndx';                                    // 'ndx' | 'spx' small-multiples grid
function GS(){ return (gridUniv === 'spx' && P.spx_grid) ? P.spx_grid : P; }  // active grid source

function fmt(x, m){ return m==='raw' ? x.toFixed(3) : (x>=0?'+':'') + x.toFixed(3); }

// global y-extent for shared scale (per mode)
function extent(m){
  const d = GS().data[m];
  if(m === 'raw'){
    let lo=Infinity, hi=-Infinity;
    for(const k in d){ for(const v of d[k]){ if(v==null) continue; if(v<lo)lo=v; if(v>hi)hi=v; } }
    if(lo===Infinity){ lo=0; hi=1; }
    const pad = (hi-lo)*0.08 || 0.05;
    return [Math.max(0,lo-pad), hi+pad];
  }
  let lo=0, hi=0;
  for(const k in d){ for(const v of d[k]){ if(v==null) continue; if(v<lo)lo=v; if(v>hi)hi=v; } }
  const a = Math.max(Math.abs(lo),Math.abs(hi))||0.1;
  return [-a, a];
}

function spark(vals, ylo, yhi, m, w=204, h=64, pad=4, overlay=null){
  const n = vals.length;
  const xs = i => pad + (w-2*pad) * (n<=1?0:i/(n-1));
  const ys = v => { const t=(v-ylo)/(yhi-ylo); return h-pad-(h-2*pad)*t; };
  let d='', started=false;
  for(let i=0;i<n;i++){
    const v=vals[i];
    if(v==null){ started=false; continue; }
    d += (started?'L':'M') + xs(i).toFixed(1)+','+ys(v).toFixed(1)+' ';
    started=true;
  }
  const last = [...vals].reverse().find(v=>v!=null);
  const lc = m==='raw' ? 'var(--accent)' : (last>=0 ? 'var(--pos)' : 'var(--neg)');
  const zeroLine = (m==='raw' || ylo>0 || yhi<0) ? '' :
    `<line x1="${pad}" y1="${ys(0).toFixed(1)}" x2="${w-pad}" y2="${ys(0).toFixed(1)}" stroke="var(--zero)" stroke-width="1"/>`;
  // Optional overlay: dashed trailing-percentile band lines (drawn behind the series) plus
  // triangle markers where the series crossed into the top (▲, up) or bottom (▼, down) band.
  let bands='', marks='';
  if(overlay){
    const bandPath = arr => {
      let dd='', st=false;
      for(let i=0;i<arr.length;i++){ const v=arr[i]; if(v==null){ st=false; continue; }
        dd += (st?'L':'M') + xs(i).toFixed(1)+','+ys(v).toFixed(1)+' '; st=true; }
      return dd;
    };
    if(overlay.p80) bands += `<path d="${bandPath(overlay.p80)}" fill="none" stroke="var(--pos)" stroke-width="0.8" stroke-dasharray="3,2" opacity="0.5"/>`;
    if(overlay.p20) bands += `<path d="${bandPath(overlay.p20)}" fill="none" stroke="var(--neg)" stroke-width="0.8" stroke-dasharray="3,2" opacity="0.5"/>`;
    for(const c of (overlay.cross||[])){
      const v=vals[c.i]; if(v==null) continue;
      const cx=xs(c.i), cy=ys(v), up=c.dir==='up', col=up?'var(--pos)':'var(--neg)', s=3;
      const tri = up
        ? `${cx.toFixed(1)},${(cy-s).toFixed(1)} ${(cx-s).toFixed(1)},${(cy+s).toFixed(1)} ${(cx+s).toFixed(1)},${(cy+s).toFixed(1)}`
        : `${cx.toFixed(1)},${(cy+s).toFixed(1)} ${(cx-s).toFixed(1)},${(cy-s).toFixed(1)} ${(cx+s).toFixed(1)},${(cy-s).toFixed(1)}`;
      marks += `<polygon points="${tri}" fill="${col}" opacity="0.95"/>`;
    }
  }
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    ${zeroLine}
    ${bands}
    <path d="${d}" fill="none" stroke="${lc}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
    ${marks}
    ${last!=null?`<circle cx="${(w-pad).toFixed(1)}" cy="${ys(last).toFixed(1)}" r="2.4" fill="${lc}"/>`:''}
  </svg>`;
}

function updateChrome(){
  const S = GS();
  const uni = gridUniv === 'spx' ? 'S&P 500' : 'NDX-100';
  const hasW = S.weights && Object.keys(S.weights).length;
  const sortTxt = sortMode==='weight' ? (hasW ? 'ordered by index weight' : 'ordered by latest divergence')
    : mode==='raw' ? 'ordered by highest raw D'
    : 'ordered by latest divergence (accumulating, then distributing)';
  const benchDisp = S.bench_label || S.bench;
  document.getElementById('sub').textContent = (mode === 'raw'
    ? `${uni} · ${S.order_raw.length} names (incl. ${benchDisp} bench) · raw dark-ratio D (5-day MA) · 0-1 scale`
    : `${uni} · ${S.order.length} names · rolling window ${P.window}d · residual of each name's dark-ratio D against ${benchDisp}`)
    + ` · ${sortTxt}`;
  document.getElementById('legend').innerHTML = mode === 'raw'
    ? `<span><i class="dot" style="background:var(--accent)"></i>raw D (share of volume that is dark)</span>`
    : `<span><i class="dot" style="background:var(--pos)"></i>above ${benchDisp} (accumulation)</span>
       <span><i class="dot" style="background:var(--neg)"></i>below ${benchDisp}</span>`;
}

function render(){
  if(gridUniv === 'spx' && !P.spx_grid){
    grid.innerHTML = '<div class="modal-empty" style="padding:40px 22px">S&amp;P 500 small multiples require a live build (run without --demo, and without --no-spx).</div>';
    return;
  }
  const S = GS();
  // Iterate the mode's own sort order, then append any constituent it omits (drawn from
  // the full weight-ordered membership) so a name is never dropped from the grid purely
  // because the CURRENT view has no series for it -- e.g. a newly-listed name with too
  // little history for the rolling regression (SPCX, HONA). Those fall back to their
  // simple-difference residual below so they stay visible and findable via search.
  const primary = sortMode === 'weight' ? S.order_weight :
    (mode === 'raw' ? S.order_raw : mode === 'diff' ? S.order_diff : S.order);
  const seen = new Set(primary);
  const ord = primary.concat((S.order_weight||[]).filter(t => !seen.has(t)));
  const [ylo,yhi] = shared ? extent(mode) : [null,null];
  const d = S.data[mode];
  const gdates = S.dates;
  const html = [];
  for(const tkr of ord){
    if(filter && !tkr.includes(filter)) continue;
    // Series for this view. If the name has none here (short history -> no regression
    // residual) fall back to the simple-difference residual so the cell still renders.
    // `raw` already covers every name, so a fallback is only ever needed in a residual view;
    // the benchmark has no residual by construction and is skipped when it has no series.
    let vals = d[tkr], vmode = mode, fellBack = false;
    if(vals === undefined){
      if(mode !== 'raw' && S.data.diff && (tkr in S.data.diff)){
        vals = S.data.diff[tkr]; vmode = 'diff'; fellBack = true;
      } else continue;
    }
    let lo=ylo, hi=yhi;
    if(!shared || fellBack){   // fallback cells always scale to themselves (mixed scales otherwise)
      if(vmode==='raw'){
        let mn=Infinity, mx=-Infinity;
        for(const v of vals){ if(v!=null){ mn=Math.min(mn,v); mx=Math.max(mx,v); } }
        if(mn===Infinity){ mn=0; mx=1; }
        const pad=(mx-mn)*0.1 || 0.05;
        lo=Math.max(0,mn-pad); hi=mx+pad;
      } else {
        let a=0.05; for(const v of vals){ if(v!=null) a=Math.max(a,Math.abs(v)); }
        lo=-a; hi=a;
      }
    }
    const last = [...vals].reverse().find(v=>v!=null);
    const cls = vmode==='raw' ? '' : (last>=0?'p':'n');
    const hot = vmode!=='raw' && Math.abs(last) >= 0.12 ? ' hot':'';
    const isBench = tkr === S.bench;
    const dispTkr = isBench ? (S.bench_label || S.bench) : tkr;
    const wt = S.weights[tkr];
    const wtBadge = wt!=null ? `<span class="wt">${wt.toFixed(2)}%</span>` : '';
    const sec = (S.sector_map && S.sector_map[tkr]) || '';
    const fbBadge = fellBack ? ` <span class="wt" style="opacity:.6" title="too little history for the rolling regression -- showing the simple D − benchmark difference">Δ</span>` : '';
    const fbTitle = fellBack ? ' · simple difference (insufficient history for the regression residual)' : '';
    html.push(
      `<div class="cell${hot}" data-tkr="${tkr}" title="${dispTkr}${wt!=null?' · index weight '+wt.toFixed(2)+'%':''}${fbTitle} · click for detail">
        <div class="chead">
          <span class="tkr">${dispTkr}${isBench?' <span style="color:var(--mut);font-weight:500">(bench)</span>':wtBadge}${fbBadge}</span>
          <span class="val ${cls}">${last==null?'--':fmt(last, vmode)}</span>
        </div>
        ${spark(vals, lo, hi, vmode)}
        <div class="meta"><span>${gdates[0]}</span><span class="sec" title="${sec}">${sec}</span><span>${gdates[gdates.length-1]}</span></div>
      </div>`);
  }
  grid.innerHTML = html.join('');
}

// forward-return horizons used throughout: 1mo / 2mo / 3mo (21 / 42 / 63 trading days)
const HMONTH = {'21':'1mo', '42':'2mo', '63':'3mo'};

// -------------------------------------------------------------------------
// Cell detail modal: click a small-multiple to see its residual enlarged, plus this
// name's forward return bucketed by decile of its own raw 1-day dark ratio, with the
// decile today's D lands in spotlighted (same highlight as the index DIX tabs).
// -------------------------------------------------------------------------
const overlay = document.getElementById('overlay');

// forward-return-by-decile bars for one name at one horizon (uses the raw-D relationship
// panel, which exists for the NDX-100 universe; S&P-only names have no per-name history).
// the per-name raw-D source for the modal: S&P grid cells use P.spx_rel, NDX cells P.rel
function modalRel(){ return (gridUniv === 'spx' && P.spx_rel) ? P.spx_rel : P.rel; }

function renderModalDeciles(h, tkr, R){
  const bars = document.getElementById('mBars'+h);
  const stat = document.getElementById('mStat'+h);
  R = R || modalRel();
  const ds = (R.d||{})[tkr], rs = ((R['r'+h])||{})[tkr];
  if(!ds || !rs){ bars.innerHTML=''; stat.textContent='no per-name dark-ratio history for this name'; return; }
  const pts=[]; const n=Math.min(ds.length, rs.length);
  for(let i=0;i<n;i++){ const x=ds[i], y=rs[i]; if(x==null||y==null) continue; pts.push([x,y]); }
  if(pts.length<20){ bars.innerHTML=''; stat.textContent='too few overlapping observations'; return; }
  const hn=parseInt(h,10), todayD=lastNonNull(ds), r=pearson(pts);
  bars.innerHTML = renderBars(deciles(pts,10), hn, null, todayD);
  stat.innerHTML = `n = <b>${pts.length.toLocaleString()}</b> · Pearson r = <b>${r==null?'--':r.toFixed(3)}</b> · today's D &rarr; highlighted decile`;
}

function openCellModal(tkr){
  const S = GS();
  const vals = S.data[mode][tkr];
  if(!vals) return;
  let lo, hi;
  if(mode === 'raw'){
    let mn=Infinity, mx=-Infinity;
    for(const v of vals){ if(v!=null){ mn=Math.min(mn,v); mx=Math.max(mx,v); } }
    if(mn===Infinity){ mn=0; mx=1; }
    const pad=(mx-mn)*0.1 || 0.05;
    lo=Math.max(0,mn-pad); hi=mx+pad;
  } else {
    let a=0.05; for(const v of vals){ if(v!=null) a=Math.max(a,Math.abs(v)); }
    lo=-a; hi=a;
  }
  const last = [...vals].reverse().find(v=>v!=null);
  const wt = S.weights[tkr];

  document.getElementById('mTkr').textContent = tkr===S.bench ? (S.bench_label || S.bench) : tkr;
  const mVal = document.getElementById('mVal');
  mVal.textContent = last==null ? '--' : fmt(last, mode);
  mVal.className = 'val' + (mode==='raw' ? '' : (last>=0?' p':' n'));
  document.getElementById('mSub').innerHTML =
    `<span>${tkr===S.bench ? 'benchmark (reconstructed constituent DIX)' : (wt!=null ? 'index weight '+wt.toFixed(2)+'%' : '')}</span>` +
    `<span>${S.dates[0]} &rarr; ${S.dates[S.dates.length-1]}</span>`;
  document.getElementById('mSpark').innerHTML = spark(vals, lo, hi, mode, 860, 170, 10);

  // forward-return-by-decile for this name (NDX cells use P.rel; S&P cells use P.spx_rel)
  const hasRel = !!((modalRel().d||{})[tkr]);
  document.getElementById('mRel').style.display = hasRel ? '' : 'none';
  if(hasRel){ renderModalDeciles('21', tkr); renderModalDeciles('42', tkr); renderModalDeciles('63', tkr); }

  overlay.classList.add('on');
}

function closeModal(){ overlay.classList.remove('on'); }
document.getElementById('mClose').addEventListener('click', closeModal);
overlay.addEventListener('click', e=>{ if(e.target === overlay) closeModal(); });
// When this modal is open, Escape closes it and stops here, so a modal stacked under it
// (the sector drill-down) is not also closed by the same keypress -- one level per Escape.
document.addEventListener('keydown', e=>{
  if(e.key === 'Escape' && overlay.classList.contains('on')){ closeModal(); e.stopImmediatePropagation(); }
});
grid.addEventListener('click', e=>{
  const cell = e.target.closest('.cell');
  if(!cell) return;
  openCellModal(cell.dataset.tkr);
});

document.getElementById('univ').addEventListener('click', e=>{
  const b=e.target.closest('button'); if(!b) return;
  gridUniv = b.dataset.u;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));
  updateChrome();
  render();
});
document.getElementById('mode').addEventListener('click', e=>{
  const b=e.target.closest('button'); if(!b) return;
  mode=b.dataset.m;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));
  updateChrome();
  render();
});
document.getElementById('sort').addEventListener('click', e=>{
  const b=e.target.closest('button'); if(!b) return;
  sortMode = b.dataset.s === 'weight' ? 'weight' : 'div';
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));
  updateChrome();
  render();
});
document.getElementById('zero').addEventListener('change', e=>{ shared=e.target.checked; render(); });
document.getElementById('q').addEventListener('input', e=>{ filter=e.target.value.toUpperCase().trim(); render(); });

document.getElementById('ttl').textContent = P.title;
document.getElementById('foot').textContent =
  `generated ${P.generated} · data: ` + (P.demo
    ? 'SYNTHETIC DEMO DATA (not real -- generated with --demo)'
    : 'FINRA off-exchange volumes + Yahoo prices (D = 5-day dark-pool indicator)');
if(P.demo){
  const banner = document.createElement('div');
  banner.textContent = '⚠ SYNTHETIC DEMO DATA -- randomly generated, not real market data';
  banner.style.cssText = 'background:#5a1e1e;color:#ffd9d9;font-weight:700;font-size:12px;'
    + 'text-align:center;padding:5px;letter-spacing:.3px';
  document.body.insertBefore(banner, document.body.firstChild);
  document.title = '[DEMO] ' + document.title;
}
updateChrome();
render();

// -------------------------------------------------------------------------
// Tab: D vs forward return (1mo / 2mo / 3mo)
// -------------------------------------------------------------------------
let horizon = '21', zscore = false, excess = false, useMedian = false;
let sharedSpxAxis = false;  // SPX tab: force all horizon panels onto one axis
const relTkrSel = document.getElementById('relTkr');
// bench pinned first, then constituents alphabetically
{
  const names = Object.keys(P.rel.d).filter(n => n !== P.bench).sort();
  if(P.bench in P.rel.d) names.unshift(P.bench);
  for(const tkr of names){
    const opt = document.createElement('option');
    opt.value = tkr; opt.textContent = tkr === P.bench ? `${tkr} (bench)` : tkr;
    relTkrSel.appendChild(opt);
  }
}

// mean/std of the non-null entries of an array (for the per-name z-score toggle)
function seriesStats(arr){
  let s=0, c=0; for(const v of arr){ if(v!=null){ s+=v; c++; } }
  if(c < 2) return null;
  const m=s/c; let ss=0; for(const v of arr){ if(v!=null) ss+=(v-m)*(v-m); }
  const sd=Math.sqrt(ss/c);
  return {mean:m, std: sd>1e-9 ? sd : 1};
}

// most recent non-null value of a date-ordered array (today's reading), or null
function lastNonNull(arr){
  if(!arr) return null;
  for(let i=arr.length-1;i>=0;i--) if(arr[i]!=null) return arr[i];
  return null;
}

// Build (dark-ratio, forward-return) pairs, applying the active toggles:
//  - excess : subtract QQQ's same-day forward return (strip market beta)
//  - zscore : standardize each name's dark ratio over the window before pooling
function pairs(tkr){
  const out = [];
  const rmap = P.rel['r' + horizon];
  const bench = P.bench;
  const brs = excess ? (rmap[bench] || null) : null;
  const names = tkr === '__ALL__' ? Object.keys(P.rel.d) : [tkr];
  for(const k of names){
    if(excess && k === bench) continue;                 // QQQ-minus-QQQ is identically 0
    const ds = P.rel.d[k], rs = rmap[k];
    if(!ds || !rs) continue;
    const xstat = zscore ? seriesStats(ds) : null;
    const n = Math.min(ds.length, rs.length, brs ? brs.length : Infinity);
    for(let i=0;i<n;i++){
      let x = ds[i], y = rs[i];
      if(x==null || y==null) continue;
      if(brs){ if(brs[i]==null) continue; y -= brs[i]; }
      if(xstat){ x = (x - xstat.mean) / xstat.std; }
      out.push([x, y]);
    }
  }
  return out;
}

function pearson(pts){
  const n = pts.length; if(n < 2) return null;
  let sx=0, sy=0; for(const [x,y] of pts){ sx+=x; sy+=y; }
  const mx=sx/n, my=sy/n;
  let cov=0, vx=0, vy=0;
  for(const [x,y] of pts){ cov += (x-mx)*(y-my); vx += (x-mx)**2; vy += (y-my)**2; }
  if(vx===0 || vy===0) return null;
  return cov / Math.sqrt(vx*vy);
}

// average-rank transform (ties get the mean rank), for Spearman
function ranks(a){
  const idx = a.map((v,i)=>[v,i]).sort((p,q)=>p[0]-q[0]);
  const r = new Array(a.length);
  let i=0;
  while(i < idx.length){
    let j=i; while(j+1 < idx.length && idx[j+1][0]===idx[i][0]) j++;
    const avg=(i+j)/2;
    for(let t=i;t<=j;t++) r[idx[t][1]] = avg;
    i = j+1;
  }
  return r;
}
function spearman(pts){
  if(pts.length < 3) return null;
  const rx = ranks(pts.map(p=>p[0])), ry = ranks(pts.map(p=>p[1]));
  return pearson(rx.map((v,i)=>[v, ry[i]]));
}

// 95% CI for Pearson r via a moving-block bootstrap. Block length = the return
// horizon, so each resample preserves the autocorrelation that overlapping
// forward returns induce -- otherwise the CI would be far too tight.
function blockBootstrapCI(pts, blk, B=300){
  const n = pts.length;
  if(n < blk*3) return null;
  if(n > 40000) B = Math.min(B, 100);   // big pooled sets: fewer resamples keeps it snappy
  const nb = Math.ceil(n/blk), rs = [];
  for(let b=0;b<B;b++){
    const samp = [];
    for(let q=0;q<nb;q++){
      const start = Math.floor(Math.random()*(n-blk+1));
      for(let k=0;k<blk;k++) samp.push(pts[start+k]);
    }
    const r = pearson(samp);
    if(r!=null) rs.push(r);
  }
  if(rs.length < 30) return null;
  rs.sort((a,b)=>a-b);
  return [rs[Math.floor(0.025*rs.length)], rs[Math.floor(0.975*rs.length)]];
}

function median(sortedYs){
  const m = sortedYs.length;
  return m%2 ? sortedYs[(m-1)/2] : (sortedYs[m/2-1] + sortedYs[m/2]) / 2;
}

function deciles(pts, k=10){
  const sorted = [...pts].sort((a,b)=>a[0]-b[0]);
  const n = sorted.length;
  const buckets = [];
  for(let b=0; b<k; b++){
    const lo = Math.floor(n*b/k), hi = Math.floor(n*(b+1)/k);
    const slice = sorted.slice(lo, hi);
    if(!slice.length){ buckets.push(null); continue; }
    let sx=0, sy=0; for(const [x,y] of slice){ sx+=x; sy+=y; }
    const rMean = sy/slice.length;
    let ss=0; for(const [,y] of slice) ss += (y-rMean)*(y-rMean);
    const ys = slice.map(p=>p[1]).sort((a,b)=>a-b);
    // slice is a contiguous run of the x-sorted points, so its first/last x are the
    // decile's domain edges (the range of D values that fall into this bucket)
    buckets.push({n: slice.length, dAvg: sx/slice.length, rMean,
                  rMedian: median(ys), rStd: Math.sqrt(ss/slice.length),
                  dLo: slice[0][0], dHi: slice[slice.length-1][0]});
  }
  return buckets;
}

// "nice" round tick values spanning [lo,hi] (1/2/5 x 10^k spacing)
function niceTicks(lo, hi, count){
  const span = hi - lo;
  if(!(span > 0)) return [lo];
  const raw = span/count, mag = Math.pow(10, Math.floor(Math.log10(raw))), nn = raw/mag;
  const step = (nn<1.5?1:nn<3?2:nn<7?5:10) * mag;
  const out = [];
  for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(Math.abs(v)<step*1e-6 ? 0 : v);
  return out;
}
function fmtPct(v){ return (v>0?'+':'') + (Math.abs(v)>=10 ? v.toFixed(0) : v.toFixed(1)) + '%'; }
// compact fixed-precision for a D-domain value; precision adapts to scale so it
// reads well for raw D (0-1), z-scored D (~-3..3), and DIX (percent, ~20-60)
function fmtDom(v){
  const a = Math.abs(v);
  return a>=100 ? v.toFixed(0) : a>=10 ? v.toFixed(1) : v.toFixed(2);
}

// the symmetric y-half-extent renderBars would pick for a bucket set at horizon h.
// Factored out so a caller can take the max across several panels and force a shared
// axis (pass the result as renderBars' forcedA).
function barsAxisExtent(buckets, h){
  const barVal = b => useMedian ? b.rMedian : b.rMean;
  const se = b => useMedian ? 0 : b.rStd / Math.sqrt(Math.max(1, b.n / h));
  let aBar=0.1, aWhisk=0.1;
  for(const b of buckets){ if(!b) continue;
    aBar = Math.max(aBar, Math.abs(barVal(b)));
    aWhisk = Math.max(aWhisk, Math.abs(barVal(b)) + se(b));
  }
  // keep the bars readable: let whiskers stretch the axis up to 2.5x the tallest
  // bar, then clip -- otherwise a single huge SE flattens every bar (as it did).
  return Math.max(aBar*1.15, Math.min(aWhisk, aBar*2.5));
}

// h = return horizon (trading days), used to shrink each bucket's standard error
// to an *effective* independent count (overlapping daily obs are not independent).
// forcedA (optional): override the auto y-half-extent, e.g. to share an axis across
// several panels for direct visual comparison.
// todayVal (optional): the latest DIX/D print -- its decile bucket gets spotlighted so
// you can see where today's dark reading sits in its historical distribution.
function renderBars(buckets, h, forcedA, todayVal){
  const w=800, H=220, padL=50, padR=14, padT=12, padB=40;  // padB fits Dn + domain range
  const barVal = b => useMedian ? b.rMedian : b.rMean;
  const se = b => useMedian ? 0 : b.rStd / Math.sqrt(Math.max(1, b.n / h));
  const a = forcedA != null ? forcedA : barsAxisExtent(buckets, h);
  const mid = padT + (H-padT-padB)/2, half = (H-padT-padB)/2;
  const ys = v => mid - (Math.max(-a, Math.min(a, v))/a) * half;
  const y0 = ys(0), bw = (w-padL-padR)/buckets.length;
  // which decile does today's print fall into? (first bucket whose upper edge it clears)
  let todayDec = -1;
  if(todayVal != null && isFinite(todayVal)){
    for(let i=0;i<buckets.length;i++){ if(buckets[i] && todayVal <= buckets[i].dHi){ todayDec=i; break; } }
    if(todayDec === -1) for(let i=buckets.length-1;i>=0;i--){ if(buckets[i]){ todayDec=i; break; } }
  }
  let grid='';
  for(const t of niceTicks(-a, a, 4)){
    const y=ys(t), zero=Math.abs(t)<1e-9;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w-padR}" y2="${y.toFixed(1)}" stroke="${zero?'var(--zero)':'var(--grid)'}" stroke-width="1"${zero?'':' stroke-dasharray="2,3"'}/>`
          + `<text x="${padL-6}" y="${(y+3).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="end">${fmtPct(t)}</text>`;
  }
  let bg='', bars='', whisk='', mark='', labels='';
  if(todayDec >= 0){                                  // faint accent band spotlighting today's column
    const x = padL + todayDec*bw;
    bg += `<rect x="${x.toFixed(1)}" y="${padT.toFixed(1)}" width="${bw.toFixed(1)}" height="${(H-padT-padB).toFixed(1)}" fill="var(--accent)" opacity="0.12"/>`;
  }
  buckets.forEach((b,i)=>{
    const x = padL + i*bw;
    if(!b){ return; }
    const v = barVal(b), yv = ys(v);
    const top = Math.min(y0, yv), hgt = Math.abs(yv-y0);
    const col = v>=0 ? 'var(--pos)' : 'var(--neg)';
    const isToday = i===todayDec;
    bars += `<rect class="bar" data-dec="${i}" x="${(x+bw*0.16).toFixed(1)}" y="${top.toFixed(1)}" width="${(bw*0.68).toFixed(1)}" height="${Math.max(hgt,1).toFixed(1)}" fill="${col}" rx="2"/>`;
    if(isToday){                                       // accent outline + a downward marker at the top
      mark += `<rect x="${(x+bw*0.16).toFixed(1)}" y="${top.toFixed(1)}" width="${(bw*0.68).toFixed(1)}" height="${Math.max(hgt,1).toFixed(1)}" fill="none" stroke="var(--accent)" stroke-width="2" rx="2"/>`
            + `<path d="M${(x+bw/2-4.5).toFixed(1)},${(padT+1).toFixed(1)} L${(x+bw/2+4.5).toFixed(1)},${(padT+1).toFixed(1)} L${(x+bw/2).toFixed(1)},${(padT+8).toFixed(1)} Z" fill="var(--accent)"/>`;
    }
    if(!useMedian){
      const s=se(b), cx=x+bw/2, yhiW=ys(v+s), yloW=ys(v-s);
      whisk += `<line x1="${cx.toFixed(1)}" y1="${yhiW.toFixed(1)}" x2="${cx.toFixed(1)}" y2="${yloW.toFixed(1)}" stroke="var(--ink)" stroke-width="1" opacity="0.7"/>`
             + `<line x1="${(cx-3).toFixed(1)}" y1="${yhiW.toFixed(1)}" x2="${(cx+3).toFixed(1)}" y2="${yhiW.toFixed(1)}" stroke="var(--ink)" stroke-width="1" opacity="0.7"/>`
             + `<line x1="${(cx-3).toFixed(1)}" y1="${yloW.toFixed(1)}" x2="${(cx+3).toFixed(1)}" y2="${yloW.toFixed(1)}" stroke="var(--ink)" stroke-width="1" opacity="0.7"/>`;
    }
    const cxL = x+bw/2;
    const lc = isToday ? 'var(--accent)' : 'var(--mut)', lw = isToday ? '700' : '400';
    labels += `<text x="${cxL.toFixed(1)}" y="${(H-24).toFixed(1)}" font-size="9.5" font-weight="${lw}" fill="${lc}" text-anchor="middle">${isToday?'D'+(i+1)+' · today':'D'+(i+1)}</text>`
            + `<text x="${cxL.toFixed(1)}" y="${(H-13).toFixed(1)}" font-size="8" fill="var(--mut)" text-anchor="middle">${fmtDom(b.dLo)}</text>`
            + `<text x="${cxL.toFixed(1)}" y="${(H-4).toFixed(1)}" font-size="8" fill="var(--mut)" text-anchor="middle">&#8594; ${fmtDom(b.dHi)}</text>`;
  });
  return `${bg}${grid}${bars}${whisk}${mark}${labels}`;
}

// per-point decile index (0..k-1), using the identical rank-based partition as
// deciles() above, so a dot's data-dec always matches the bar it fell into.
function decileIndexOf(pts, k=10){
  const n = pts.length;
  const order = pts.map((_,i)=>i).sort((a,b)=>pts[a][0]-pts[b][0]);
  const dec = new Array(n);
  for(let b=0;b<k;b++){
    const lo=Math.floor(n*b/k), hi=Math.floor(n*(b+1)/k);
    for(let j=lo;j<hi;j++) dec[order[j]] = b;
  }
  return dec;
}

// union raw (unpadded) [xlo,xhi,ylo,yhi] over several point sets -- for a shared
// scatter axis across panels. Ignores empty sets; returns null if all are empty.
function ptsBounds(ptsList){
  let xlo=Infinity,xhi=-Infinity,ylo=Infinity,yhi=-Infinity;
  for(const pts of ptsList) for(const [x,y] of pts){
    if(x<xlo)xlo=x; if(x>xhi)xhi=x; if(y<ylo)ylo=y; if(y>yhi)yhi=y;
  }
  return xlo===Infinity ? null : [xlo,xhi,ylo,yhi];
}

// forcedRaw (optional): [xlo,xhi,ylo,yhi] raw bounds to use instead of this set's own
// min/max, so several scatters can share one axis. The same fractional padding is
// applied either way, so forced panels stay directly comparable.
// todayX (optional): draw a vertical marker at this x (today's dark-ratio reading,
// whose forward return isn't realized yet so it's not a scatter dot) -- only meaningful
// for single-series scatters, NOT the pooled ones where dates interleave across names.
// maxDots: cap the number of *plotted* dots (systematically sampled) so a pooled cloud
// of ~100k points doesn't create 100k SVG nodes and stall the page. All statistics
// (trend, and the caller's r / rho / CI / deciles) still use the full point set --
// only the visual density is thinned. Single-series scatters (~1k pts) are well under
// the cap and unaffected.
function renderScatter(pts, k=10, forcedRaw, todayX=null, maxDots=7000){
  const w=800, H=320, padL=50, padR=16, padT=12, padB=28;
  if(!pts.length) return '';
  const decOf = decileIndexOf(pts, k);
  const xs = pts.map(p=>p[0]), yv_ = pts.map(p=>p[1]);
  let xlo, xhi, ylo, yhi;
  if(forcedRaw){ [xlo,xhi,ylo,yhi] = forcedRaw; }
  else { xlo=Math.min(...xs); xhi=Math.max(...xs); ylo=Math.min(...yv_); yhi=Math.max(...yv_); }
  if(todayX!=null){ xlo=Math.min(xlo,todayX); xhi=Math.max(xhi,todayX); }  // keep marker in view
  const xr=(xhi-xlo)||1, yr=(yhi-ylo)||1;
  xlo-=xr*0.03; xhi+=xr*0.03; ylo-=yr*0.05; yhi+=yr*0.05;
  const X = v => padL + (w-padL-padR) * ((v-xlo)/(xhi-xlo));
  const Y = v => (H-padB) - (H-padT-padB) * ((v-ylo)/(yhi-ylo));
  let grid='';
  for(const t of niceTicks(ylo, yhi, 5)){
    const y=Y(t), zero=Math.abs(t)<1e-9;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w-padR}" y2="${y.toFixed(1)}" stroke="${zero?'var(--zero)':'var(--grid)'}" stroke-width="1"${zero?'':' stroke-dasharray="2,3"'}/>`
          + `<text x="${padL-6}" y="${(y+3).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="end">${fmtPct(t)}</text>`;
  }
  for(const t of niceTicks(xlo, xhi, 6)){
    const x=X(t);
    grid += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="var(--grid)" stroke-width="0.5" stroke-dasharray="2,3"/>`
          + `<text x="${x.toFixed(1)}" y="${(H-padB+13).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="middle">${t.toFixed(2)}</text>`;
  }
  // plot every `stride`-th point when the cloud exceeds maxDots (thins the visual only)
  const stride = (maxDots > 0 && pts.length > maxDots) ? Math.ceil(pts.length / maxDots) : 1;
  let dots='', shown=0;
  for(let i=0;i<pts.length;i+=stride){
    const [x,y]=pts[i];
    dots += `<circle class="dot" data-dec="${decOf[i]}" cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="2" fill="var(--accent)"/>`;
    shown++;
  }
  const sampleNote = stride>1
    ? `<text x="${(w-padR).toFixed(1)}" y="${(padT+8).toFixed(1)}" font-size="9" fill="var(--mut)" text-anchor="end">plotting ${shown.toLocaleString()} of ${pts.length.toLocaleString()} pts (sampled; stats use all)</text>`
    : '';
  // simple OLS trend line
  const n=pts.length; let sx=0,sy=0; for(const[x,y] of pts){sx+=x;sy+=y;}
  const mx=sx/n, my=sy/n;
  let num=0, den=0; for(const[x,y] of pts){ num+=(x-mx)*(y-my); den+=(x-mx)**2; }
  let trend = '';
  if(den>0){
    const slope = num/den, icpt = my - slope*mx;
    trend = `<line x1="${X(xlo).toFixed(1)}" y1="${Y(slope*xlo+icpt).toFixed(1)}" x2="${X(xhi).toFixed(1)}" y2="${Y(slope*xhi+icpt).toFixed(1)}" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="4,3"/>`;
  }
  // vertical marker at today's D reading (its forward return isn't realized yet, so it's
  // not a scatter dot) -- shows where the current dark-flow regime sits on the x-axis.
  let todayMark='';
  if(todayX!=null){
    const tx=X(todayX), right = todayX > (xlo+xhi)/2;
    todayMark = `<line x1="${tx.toFixed(1)}" y1="${padT}" x2="${tx.toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="#e3b341" stroke-width="1.5"/>`
      + `<text x="${(tx + (right?-5:5)).toFixed(1)}" y="${(padT+9).toFixed(1)}" font-size="9.5" font-weight="700" fill="#e3b341" text-anchor="${right?'end':'start'}">D today ${fmtDom(todayX)}</text>`;
  }
  return `${grid}${dots}${trend}${todayMark}${sampleNote}`;
}

function renderRel(){
  const tkr = relTkrSel.value;
  const h = parseInt(horizon, 10);
  // z-scoring is a per-name linear rescale -> inert for a single name (deciles,
  // Pearson/Spearman and the auto-scaled scatter are all unchanged). It only bites
  // in the pooled view, so disable it otherwise to avoid looking broken.
  const zc = document.getElementById('zscore');
  zc.disabled = tkr !== '__ALL__';
  zc.parentElement.style.opacity = zc.disabled ? '0.4' : '';
  const pts = pairs(tkr);
  const r = pearson(pts), rho = spearman(pts);
  const ci = pts.length > 200 ? blockBootstrapCI(pts, h) : null;
  const nEff = Math.round(pts.length / h);
  const xlab = zscore ? 'z-scored 1-day dark ratio (per name)'
                      : 'raw 1-day dark ratio (FINRA off-exch vol / total vol)';
  const ylab = excess ? `${HMONTH[horizon]} forward return in excess of ${P.bench} (%)`
                      : `${HMONTH[horizon]} forward return (%)`;
  document.getElementById('barsTitle').textContent =
    `${useMedian ? 'Median' : 'Mean'} ${ylab} by dark-ratio decile`;
  document.getElementById('scatterTitle').textContent = `${xlab} vs ${ylab} (daily obs)`;
  document.getElementById('relLegend').textContent = `x: ${xlab}  ·  y: ${ylab}`;
  const ciTxt = ci ? ` [95% CI ${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : '';
  document.getElementById('relStats').innerHTML =
    `<span>n = <b>${pts.length.toLocaleString()}</b> (≈<b>${nEff.toLocaleString()}</b> indep.)</span>` +
    `<span>Pearson r = <b>${r==null?'--':r.toFixed(3)}</b>${ciTxt}</span>` +
    `<span>Spearman ρ = <b>${rho==null?'--':rho.toFixed(3)}</b></span>` +
    `<span>${tkr==='__ALL__' ? `${P.order_raw.length} names pooled` : tkr}` +
    `${P.rel.range ? `, ${P.rel.range[0]}→${P.rel.range[1]}` : ''}</span>`;
  document.getElementById('barsSvg').innerHTML = renderBars(deciles(pts, 10), h);
  const showScatter = tkr !== '__ALL__';
  document.getElementById('scatterCard').style.display = showScatter ? '' : 'none';
  if(showScatter){
    // today's D for this name, z-scored to match the scatter's x if that toggle is on
    const ds = P.rel.d[tkr];
    let todayX = lastNonNull(ds);
    if(zscore && todayX!=null){ const st=seriesStats(ds); if(st) todayX=(todayX-st.mean)/st.std; }
    document.getElementById('scatterSvg').innerHTML = renderScatter(pts, 10, undefined, todayX);
  }
}
relTkrSel.addEventListener('change', renderRel);
document.getElementById('zscore').addEventListener('change', e=>{ zscore=e.target.checked; renderRel(); });
document.getElementById('excess').addEventListener('change', e=>{ excess=e.target.checked; renderRel(); });
document.getElementById('median').addEventListener('change', e=>{ useMedian=e.target.checked; renderRel(); });
document.getElementById('horizon').addEventListener('click', e=>{
  const b=e.target.closest('button'); if(!b) return;
  horizon=b.dataset.h;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));
  renderRel();
});

// -------------------------------------------------------------------------
// Tab: SPX D vs Return -- SqueezeMetrics' own market-wide Dark Index (DIX, in
// percent) against the S&P 500's own 1mo/2mo/3mo forward return. A single time
// series (not pooled across names), so no ticker selector is needed.
// -------------------------------------------------------------------------
function spxPairs(h){
  if(!P.spx) return [];
  const dix = P.spx.dix, rs = P.spx['r' + h];
  const out = [];
  const n = Math.min(dix.length, rs.length);
  for(let i=0;i<n;i++){
    const x = dix[i], y = rs[i];
    if(x==null || y==null) continue;
    out.push([x, y]);
  }
  return out;
}

const SPX_HORIZONS = ['21', '42', '63'];  // 1mo / 2mo / 3mo (21 / 42 / 63 trading days)
function renderSpx(){
  if(!P.spx){
    document.getElementById('spxSub').textContent =
      'no SPX DIX data (run without --no-spx and with a valid --key, or check --demo)';
    document.getElementById('spxGrid').innerHTML = '<div class="modal-empty">No SPX DIX data available.</div>';
    return;
  }
  document.getElementById('spxSub').textContent =
    `S&P 500 reconstructed dollar-DIX (IVV constituents, FINRA) vs SPX forward return · ${P.spx.dates.length.toLocaleString()} dates` +
    (P.spx.range ? ` · ${P.spx.range[0]} → ${P.spx.range[1]}` : '') +
    (sharedSpxAxis ? ' · shared axis' : '');

  // precompute so a shared axis can span all panels; deciles() is reused for the bars.
  const panels = SPX_HORIZONS.map(h => {
    const pts = spxPairs(h);
    return { h, hn: parseInt(h, 10), pts, buckets: deciles(pts, 10) };
  });
  // when sharing: the tallest bar-axis across horizons, and the union scatter bounds.
  const forcedA = sharedSpxAxis
    ? Math.max(...panels.map(p => barsAxisExtent(p.buckets, p.hn))) : null;
  const forcedRaw = sharedSpxAxis ? ptsBounds(panels.map(p => p.pts)) : null;
  const dixToday = lastNonNull(P.spx.dix);  // same latest DIX for all three horizons

  for(const {h, hn, pts, buckets} of panels){
    const r = pearson(pts), rho = spearman(pts);
    const ci = pts.length > 200 ? blockBootstrapCI(pts, hn) : null;
    const nEff = Math.round(pts.length / hn);
    const ciTxt = ci ? ` [95% CI ${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : '';
    document.getElementById('sBars' + h).innerHTML = renderBars(buckets, hn, forcedA, dixToday);
    document.getElementById('sScatter' + h).innerHTML = renderScatter(pts, 10, forcedRaw, dixToday);  // mark today's DIX
    document.getElementById('sStats' + h).innerHTML =
      `<span>n = <b>${pts.length.toLocaleString()}</b> (&asymp;<b>${nEff.toLocaleString()}</b> indep.)</span>` +
      `<span>Pearson r = <b>${r==null?'--':r.toFixed(3)}</b>${ciTxt}</span>` +
      `<span>Spearman &rho; = <b>${rho==null?'--':rho.toFixed(3)}</b></span>`;
  }
  renderDixCompare();
}

// two aligned series (percent) over labelled dates -- for the DIX formula check
function renderTwoLineChart(dates, a, b, labA, labB){
  const w=800, H=300, padL=46, padR=16, padT=14, padB=28, T=dates.length;
  let lo=Infinity, hi=-Infinity;
  for(const s of [a,b]) for(const v of s) if(v!=null){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const pad=(hi-lo)*0.08||1; lo-=pad; hi+=pad;
  const X = t => padL + (w-padL-padR)*t/Math.max(1,T-1);
  const Y = v => (H-padB) - (H-padT-padB)*((v-lo)/(hi-lo));
  let grid='';
  for(const tk of niceTicks(lo, hi, 5)){
    const y=Y(tk);
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w-padR}" y2="${y.toFixed(1)}" stroke="var(--grid)" stroke-width="1" stroke-dasharray="2,3"/>`
          + `<text x="${padL-6}" y="${(y+3).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="end">${tk.toFixed(0)}%</text>`;
  }
  for(let t=1;t<T;t++){
    if(dates[t].slice(0,4) !== dates[t-1].slice(0,4)){
      const x=X(t);
      grid += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="var(--grid)" stroke-width="0.5" stroke-dasharray="2,3"/>`
            + `<text x="${x.toFixed(1)}" y="${(H-padB+13).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="middle">${dates[t].slice(0,4)}</text>`;
    }
  }
  const line = (s, color, dash) => {
    let d='', started=false;
    for(let t=0;t<T;t++){
      const v=s[t]; if(v==null){ started=false; continue; }
      d += (started?'L':'M') + X(t).toFixed(1) + ',' + Y(v).toFixed(1) + ' ';
      started=true;
    }
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.4"${dash?` stroke-dasharray="${dash}"`:''} stroke-linejoin="round"/>`;
  };
  const legend = `<text x="${w-padR}" y="${(padT+9).toFixed(1)}" font-size="9.5" text-anchor="end">`
    + `<tspan fill="var(--mut)">${labA}</tspan><tspan fill="var(--mut)"> &#183; </tspan>`
    + `<tspan fill="#e3b341" font-weight="700">${labB}</tspan></text>`;
  return `${grid}${line(a,'var(--mut)','4,3')}${line(b,'#e3b341')}${legend}`;
}

function renderDixCompare(){
  const el = document.getElementById('spxCmp'), st = document.getElementById('spxCmpStats');
  if(!el) return;
  if(!P.rel.ndx_dix || !P.spx){
    el.innerHTML = `<text x="400" y="150" font-size="12" fill="var(--mut)" text-anchor="middle">regenerate the dashboard (with FINRA short volumes) to enable the replica comparison</text>`;
    st.innerHTML = ''; return;
  }
  const dixByDate = new Map();
  P.spx.dates.forEach((d,i)=>{ if(P.spx.dix[i]!=null) dixByDate.set(d, P.spx.dix[i]); });
  const dates=[], dixS=[], repS=[];
  P.rel.dates.forEach((d,i)=>{
    const rep = P.rel.ndx_dix[i], dv = dixByDate.get(d);
    if(rep!=null && dv!=null){ dates.push(d); dixS.push(dv); repS.push(rep*100); }
  });
  if(dates.length < 30){
    el.innerHTML = `<text x="400" y="150" font-size="12" fill="var(--mut)" text-anchor="middle">not enough overlapping dates</text>`;
    st.innerHTML = ''; return;
  }
  const lvl = pearson(dixS.map((v,i)=>[v, repS[i]]));
  const dChg = [], rChg = [];
  for(let i=1;i<dates.length;i++){ dChg.push(dixS[i]-dixS[i-1]); rChg.push(repS[i]-repS[i-1]); }
  const chg = pearson(dChg.map((v,i)=>[v, rChg[i]]));
  const diff = repS.map((v,i)=>v-dixS[i]);
  el.innerHTML = renderTwoLineChart(dates, dixS, repS, 'S&P 500 DIX', 'NDX-100 DIX');
  st.innerHTML =
    `<span>overlap = <b>${dates.length.toLocaleString()}</b> days</span>` +
    `<span>level corr = <b>${lvl==null?'--':lvl.toFixed(3)}</b></span>` +
    `<span title="correlation of day-to-day CHANGES -- the sharper test of whether the construction is right">&Delta; corr = <b>${chg==null?'--':chg.toFixed(3)}</b></span>` +
    `<span>mean offset (replica &minus; DIX) = <b>${mean(diff).toFixed(2)}pp</b></span>` +
    `<span>mean |diff| = <b>${mean(diff.map(Math.abs)).toFixed(2)}pp</b></span>`;
}
document.getElementById('spxShared').addEventListener('change', e=>{
  sharedSpxAxis = e.target.checked;
  renderSpx();
});

// -------------------------------------------------------------------------
// Tab: NDX-100 D vs Return -- an index-level dark gauge built from the panel
// itself: the equal-weight mean of every constituent's raw 1-day FINRA dark
// ratio, against QQQ's forward return. The home-grown analogue of the SPX
// tab's SqueezeMetrics DIX.
// -------------------------------------------------------------------------
let ndxDbarCache = null;
function ndxDbar(){
  if(ndxDbarCache) return ndxDbarCache;
  const dser = P.rel.d, bench = P.bench;
  const names = Object.keys(dser).filter(n => n !== bench);
  const T = P.rel.dates.length;
  const out = new Array(T).fill(null);
  for(let i=0;i<T;i++){
    let s=0, n=0;
    for(const nm of names){ const v = dser[nm][i]; if(v!=null){ s+=v; n++; } }
    if(n >= 20) out[i] = s/n;      // require a real cross-section, not a thin day
  }
  ndxDbarCache = out;
  return out;
}
let ndxMode = 'dix', ndxShared = false;
// 'dix' = dollar-weighted Σ$short/Σ$dark (SqueezeMetrics construction);
// 'agg' = Σ dark vol / Σ total vol (dark share); 'mean' = equal-weight mean of ratios
function ndxSeries(){
  if(ndxMode === 'dix' && P.rel.ndx_dix) return P.rel.ndx_dix;
  if(ndxMode !== 'mean' && P.rel.ndx_agg) return P.rel.ndx_agg;
  return ndxDbar();
}
function ndxPairs(h, series){
  const br = (P.rel['r'+h]||{})[P.bench];
  const out = [];
  if(!br) return out;
  const n = Math.min(series.length, br.length);
  for(let i=0;i<n;i++){
    if(series[i]==null || br[i]==null) continue;
    out.push([series[i], br[i]]);
  }
  return out;
}
const NDX_MODE_SUB = {
  dix: 'NDX DIX replica: Σ(price × FINRA short volume) ÷ Σ(price × FINRA off-exchange volume) across all constituents each day (dollar-weighted DPI, the SqueezeMetrics construction)',
  agg: 'index-level dark SHARE: Σ off-exchange volume ÷ Σ consolidated volume across all constituents each day (volume-weighted; note the different numerator vs DIX)',
  mean: 'equal-weight mean of the raw 1-day FINRA dark ratio across constituents (GOOG/GOOGL merged)',
};
function renderNdx(){
  const series = ndxSeries();
  const wanted = ndxMode==='dix' ? P.rel.ndx_dix : ndxMode==='agg' ? P.rel.ndx_agg : series;
  const fellBack = !wanted && ndxMode !== 'mean';
  document.getElementById('ndxSub').textContent =
    NDX_MODE_SUB[fellBack ? 'mean' : ndxMode] +
    (fellBack ? ' -- selected series unavailable in this payload, regenerate to enable it' : '') +
    ` vs ${P.bench} forward return` +
    (ndxShared ? ' · shared axis' : '') +
    (P.rel.range ? ` · ${P.rel.range[0]} → ${P.rel.range[1]}` : '');
  const today = lastNonNull(series);
  // precompute so a shared axis can span all three horizons (same pattern as SPX)
  const panels = ['21','42','63'].map(h => {
    const pts = ndxPairs(h, series);
    return { h, hn: parseInt(h,10), pts, buckets: deciles(pts, 10) };
  });
  const forcedA = ndxShared
    ? Math.max(...panels.map(p => barsAxisExtent(p.buckets, p.hn))) : null;
  const forcedRaw = ndxShared ? ptsBounds(panels.map(p => p.pts)) : null;
  for(const {h, hn, pts, buckets} of panels){
    const r = pearson(pts), rho = spearman(pts);
    const ci = pts.length > 200 ? blockBootstrapCI(pts, hn) : null;
    const nEff = Math.round(pts.length/hn);
    const ciTxt = ci ? ` [95% CI ${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : '';
    document.getElementById('nBars'+h).innerHTML = renderBars(buckets, hn, forcedA, today);
    document.getElementById('nScatter'+h).innerHTML = renderScatter(pts, 10, forcedRaw, today);
    document.getElementById('nStats'+h).innerHTML =
      `<span>n = <b>${pts.length.toLocaleString()}</b> (&asymp;<b>${nEff.toLocaleString()}</b> indep.)</span>` +
      `<span>Pearson r = <b>${r==null?'--':r.toFixed(3)}</b>${ciTxt}</span>` +
      `<span>Spearman &rho; = <b>${rho==null?'--':rho.toFixed(3)}</b></span>`;
  }
}
document.getElementById('ndxModeSeg').addEventListener('click', e=>{
  const b = e.target.closest('button'); if(!b) return;
  ndxMode = b.dataset.m;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on', x===b));
  renderNdx();
});
document.getElementById('ndxShared').addEventListener('change', e=>{
  ndxShared = e.target.checked;
  renderNdx();
});

// -------------------------------------------------------------------------
// Tab: IWM D vs Return -- the Russell 2000 ETF's own SqueezeMetrics D (5-day
// dark short-sale ratio, 0-1) against IWM's forward return. Same layout and
// add-ons as the SPX/NDX tabs (deciles, scatter, today marker, shared axis).
// -------------------------------------------------------------------------
let iwmShared = false;
function iwmPairs(h){
  if(!P.iwm) return [];
  const d = P.iwm.d, rs = P.iwm['r' + h];
  const out = [];
  const n = Math.min(d.length, rs.length);
  for(let i=0;i<n;i++){
    if(d[i]==null || rs[i]==null) continue;
    out.push([d[i], rs[i]]);
  }
  return out;
}
function renderIwm(){
  if(!P.iwm){
    document.getElementById('iwmSub').textContent =
      'no IWM data (run without --no-spx and with a valid --key, or check --demo)';
    document.getElementById('iwmGrid').innerHTML = '<div class="modal-empty">No IWM data available.</div>';
    return;
  }
  document.getElementById('iwmSub').textContent =
    (P.iwm.reconstructed
      ? `Russell 2000 reconstructed dollar-DIX (Σ$short/Σ$off-exch across ${P.iwm.constituents ? P.iwm.constituents.toLocaleString() : '~2,000'} IWM constituents) vs IWM forward return`
      : `iShares Russell 2000 ETF · SqueezeMetrics D (5-day dark short-sale ratio of the ETF itself) vs IWM forward return`) +
    ` · ${P.iwm.dates.length.toLocaleString()} dates` +
    (P.iwm.range ? ` · ${P.iwm.range[0]} → ${P.iwm.range[1]}` : '') +
    (iwmShared ? ' · shared axis' : '');
  const panels = ['21','42','63'].map(h => {
    const pts = iwmPairs(h);
    return { h, hn: parseInt(h,10), pts, buckets: deciles(pts, 10) };
  });
  const forcedA = iwmShared
    ? Math.max(...panels.map(p => barsAxisExtent(p.buckets, p.hn))) : null;
  const forcedRaw = iwmShared ? ptsBounds(panels.map(p => p.pts)) : null;
  const dToday = lastNonNull(P.iwm.d);
  for(const {h, hn, pts, buckets} of panels){
    const r = pearson(pts), rho = spearman(pts);
    const ci = pts.length > 200 ? blockBootstrapCI(pts, hn) : null;
    const nEff = Math.round(pts.length/hn);
    const ciTxt = ci ? ` [95% CI ${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : '';
    document.getElementById('wBars'+h).innerHTML = renderBars(buckets, hn, forcedA, dToday);
    document.getElementById('wScatter'+h).innerHTML = renderScatter(pts, 10, forcedRaw, dToday);
    document.getElementById('wStats'+h).innerHTML =
      `<span>n = <b>${pts.length.toLocaleString()}</b> (&asymp;<b>${nEff.toLocaleString()}</b> indep.)</span>` +
      `<span>Pearson r = <b>${r==null?'--':r.toFixed(3)}</b>${ciTxt}</span>` +
      `<span>Spearman &rho; = <b>${rho==null?'--':rho.toFixed(3)}</b></span>`;
  }
}
document.getElementById('iwmShared').addEventListener('change', e=>{
  iwmShared = e.target.checked;
  renderIwm();
});

// -------------------------------------------------------------------------
// Tab: Cross-sectional L/S -- each day, rank the ~100 names by a cross-sectional
// dark signal, split into deciles, and measure forward EXCESS return (vs QQQ)
// per decile pooled across days. The long-short D10-D1 is the factor return.
// This is the powered, orthogonalized test the pooled scatter only approximates.
// -------------------------------------------------------------------------
let xsHorizon = '21', xsSignal = 'rel';

// per-position expanding mean of a date-ordered array (mean of all non-null values
// up to and including each index; carried forward on nulls). No look-ahead: only
// past+present values are used, so it's a fair real-time baseline.
function expandingMeanCount(arr){
  const mean = new Array(arr.length).fill(null), cnt = new Array(arr.length).fill(0);
  let s = 0, c = 0;
  for(let i=0;i<arr.length;i++){
    const v = arr[i];
    if(v!=null){ s+=v; c++; }
    if(c){ mean[i] = s/c; cnt[i] = c; }
  }
  return {mean, cnt};
}

function crossSectionalDeciles(h, signalMode){
  const K = 10, MIN_NAMES = 20, MIN_HIST = 20;
  const bench = P.bench, dser = P.rel.d, rser = P.rel['r' + h];
  if(!rser) return null;
  const names = Object.keys(dser).filter(n => n !== bench && rser[n]);
  const br = rser[bench];                         // QQQ forward return (for excess)
  if(!br) return null;
  let nT = 0; for(const nm of names) if(dser[nm]) nT = Math.max(nT, dser[nm].length);

  // expanding per-name mean, only needed for the name-specific ('rel') signal
  const em = {};
  if(signalMode === 'rel') for(const nm of names) if(dser[nm]) em[nm] = expandingMeanCount(dser[nm]);

  const buckets = Array.from({length:K}, () => ({rets:[], sigs:[]}));
  const spreads = [];
  let daysUsed = 0;
  for(let i=0;i<nT;i++){
    if(br[i]==null) continue;                     // no realized QQQ return yet -> no excess
    const row = [];
    for(const nm of names){
      const dv = dser[nm]?.[i], rv = rser[nm]?.[i];
      if(dv==null || rv==null) continue;
      let signal;
      if(signalMode === 'rel'){
        const e = em[nm];
        if(!e || e.mean[i]==null || e.cnt[i] < MIN_HIST) continue;   // need a stable baseline
        signal = dv - e.mean[i];                  // unusually dark vs this name's own norm
      } else {
        signal = dv;                              // raw D: a bet on structurally-dark names
      }
      row.push({signal, y: rv - br[i]});          // y = forward return in excess of QQQ
    }
    if(row.length < MIN_NAMES) continue;
    row.sort((a,b)=>a.signal-b.signal);
    const m = row.length, decMean = new Array(K).fill(0);
    for(let d=0; d<K; d++){
      const lo = Math.floor(m*d/K), hi = Math.floor(m*(d+1)/K);
      let sy=0, c=0;
      for(let j=lo;j<hi;j++){ const p=row[j]; buckets[d].rets.push(p.y); buckets[d].sigs.push(p.signal); sy+=p.y; c++; }
      if(c) decMean[d] = sy/c;
    }
    spreads.push(decMean[K-1] - decMean[0]);      // that day's D10-D1 excess spread
    daysUsed++;
  }

  const agg = buckets.map(b => {
    const n = b.rets.length; if(!n) return null;
    const mean = b.rets.reduce((s,c)=>s+c,0)/n;
    let ss=0; for(const y of b.rets) ss += (y-mean)*(y-mean);
    const sig = b.sigs.reduce((s,c)=>s+c,0)/b.sigs.length;
    return {n, mean, std: Math.sqrt(ss/n), sig};
  });
  const N = spreads.length;
  const lsMean = N ? spreads.reduce((s,c)=>s+c,0)/N : null;
  let lsStd = 0; if(N>1){ for(const s of spreads) lsStd += (s-lsMean)*(s-lsMean); lsStd = Math.sqrt(lsStd/N); }
  const nEff = Math.max(1, N / parseInt(h,10));   // forward returns overlap by h days
  const lsSE = (N>1) ? lsStd/Math.sqrt(nEff) : null;
  const t = (lsSE && lsSE>0) ? lsMean/lsSE : null;
  return {buckets: agg, lsMean, lsSE, t, N, nEff};
}

// dedicated decile-bar renderer for the cross-sectional tab: x is a decile RANK
// (D1..D10), labelled with each decile's average signal, not a fixed D range.
function renderXsBars(buckets, h){
  const w=800, H=240, padL=52, padR=14, padT=14, padB=46;
  const bv = b => b.mean, se = b => b.std/Math.sqrt(Math.max(1, b.n/h));
  let aBar=0.1, aWhisk=0.1;
  for(const b of buckets){ if(!b) continue;
    aBar = Math.max(aBar, Math.abs(bv(b)));
    aWhisk = Math.max(aWhisk, Math.abs(bv(b)) + se(b));
  }
  const a = Math.max(aBar*1.15, Math.min(aWhisk, aBar*2.5));
  const mid = padT+(H-padT-padB)/2, half = (H-padT-padB)/2;
  const ys = v => mid - (Math.max(-a, Math.min(a, v))/a)*half;
  const y0 = ys(0), bw = (w-padL-padR)/buckets.length;
  let grid='';
  for(const tk of niceTicks(-a, a, 4)){
    const y=ys(tk), zero=Math.abs(tk)<1e-9;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w-padR}" y2="${y.toFixed(1)}" stroke="${zero?'var(--zero)':'var(--grid)'}" stroke-width="1"${zero?'':' stroke-dasharray="2,3"'}/>`
          + `<text x="${padL-6}" y="${(y+3).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="end">${fmtPct(tk)}</text>`;
  }
  let bars='', whisk='', labels='';
  buckets.forEach((b,i)=>{
    const x = padL + i*bw; if(!b) return;
    const v = bv(b), yv = ys(v), top = Math.min(y0, yv), hgt = Math.abs(yv-y0);
    const col = i===0 ? 'var(--neg)' : i===buckets.length-1 ? 'var(--pos)' : (v>=0?'var(--pos)':'var(--neg)');
    const op = (i===0 || i===buckets.length-1) ? '1' : '0.55';   // emphasise the L/S legs
    bars += `<rect x="${(x+bw*0.16).toFixed(1)}" y="${top.toFixed(1)}" width="${(bw*0.68).toFixed(1)}" height="${Math.max(hgt,1).toFixed(1)}" fill="${col}" opacity="${op}" rx="2"/>`;
    const s=se(b), cx=x+bw/2, yh=ys(v+s), yl=ys(v-s);
    whisk += `<line x1="${cx.toFixed(1)}" y1="${yh.toFixed(1)}" x2="${cx.toFixed(1)}" y2="${yl.toFixed(1)}" stroke="var(--ink)" stroke-width="1" opacity="0.7"/>`
           + `<line x1="${(cx-3).toFixed(1)}" y1="${yh.toFixed(1)}" x2="${(cx+3).toFixed(1)}" y2="${yh.toFixed(1)}" stroke="var(--ink)" stroke-width="1" opacity="0.7"/>`
           + `<line x1="${(cx-3).toFixed(1)}" y1="${yl.toFixed(1)}" x2="${(cx+3).toFixed(1)}" y2="${yl.toFixed(1)}" stroke="var(--ink)" stroke-width="1" opacity="0.7"/>`;
    const cxL = x+bw/2, nlab = b.n>=1000 ? (b.n/1000).toFixed(1)+'k' : ''+b.n;
    labels += `<text x="${cxL.toFixed(1)}" y="${(H-30).toFixed(1)}" font-size="9.5" font-weight="600" fill="var(--mut)" text-anchor="middle">D${i+1}</text>`
            + `<text x="${cxL.toFixed(1)}" y="${(H-18).toFixed(1)}" font-size="8" fill="var(--mut)" text-anchor="middle">${fmtDom(b.sig)}</text>`
            + `<text x="${cxL.toFixed(1)}" y="${(H-7).toFixed(1)}" font-size="8" fill="var(--mut)" text-anchor="middle">n ${nlab}</text>`;
  });
  return `${grid}${bars}${whisk}${labels}`;
}

function renderXs(){
  const h = xsHorizon;
  const res = crossSectionalDeciles(h, xsSignal);
  const sigLbl = xsSignal === 'rel'
    ? "how unusually dark each name is vs its own expanding average"
    : "raw dark ratio (a bet on structurally-dark names)";
  document.getElementById('xsSigLbl').innerHTML = sigLbl;
  document.getElementById('xsHdr').textContent =
    `(${HMONTH[h]} horizon · ${xsSignal==='rel' ? 'name-specific signal' : 'raw-D signal'})`;
  if(!res || !res.buckets.some(Boolean)){
    document.getElementById('xsSub').textContent = 'not enough data to form cross-sectional deciles';
    document.getElementById('xsBars').innerHTML = '';
    document.getElementById('xsStats').innerHTML = '';
    return;
  }
  document.getElementById('xsBars').innerHTML = renderXsBars(res.buckets, parseInt(h,10));
  const monoPts = res.buckets.map((b,i)=> b ? [i, b.mean] : null).filter(Boolean);
  const mono = spearman(monoPts);
  const perYr = 252/parseInt(h,10);
  const lsTxt = res.lsMean==null ? '--' : `${res.lsMean>=0?'+':''}${res.lsMean.toFixed(2)}%`;
  const annTxt = res.lsMean==null ? '--' : `${res.lsMean*perYr>=0?'+':''}${(res.lsMean*perYr).toFixed(1)}%`;
  document.getElementById('xsSub').textContent =
    `Cross-sectional decile long-short · outcome: forward excess return vs ${P.bench}` +
    (P.rel.range ? ` · ${P.rel.range[0]} → ${P.rel.range[1]}` : '');
  document.getElementById('xsStats').innerHTML =
    `<span>Long-short D10&minus;D1 = <b>${lsTxt}</b> per ${HMONTH[h]} (t = <b>${res.t==null?'--':res.t.toFixed(1)}</b>)</span>` +
    `<span>&asymp; <b>${annTxt}</b> annualized</span>` +
    `<span>monotonicity &rho; = <b>${mono==null?'--':mono.toFixed(2)}</b></span>` +
    `<span><b>${res.N.toLocaleString()}</b> days (&asymp;<b>${Math.round(res.nEff).toLocaleString()}</b> indep.)</span>`;
}
document.getElementById('xsHorizonSeg').addEventListener('click', e=>{
  const b=e.target.closest('button'); if(!b) return;
  xsHorizon = b.dataset.h;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));
  renderXs();
});
document.getElementById('xsSignalSeg').addEventListener('click', e=>{
  const b=e.target.closest('button'); if(!b) return;
  xsSignal = b.dataset.sig;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));
  renderXs();
});

// -------------------------------------------------------------------------
// Tab: D-streak events -- define discrete EVENTS as runs of >= N consecutive days
// where a name's dark ratio sits inside a decile band, then study the forward
// return from each event's trigger day (the Nth day of the run -- no look-ahead).
// One event per run. Deciles are per-name (vs the name's own history) or pooled.
// -------------------------------------------------------------------------
let evRun = 5, evLo = 1, evHi = 2, evBasis = 'trail', evExcess = false, evTicker = '__ALL__';

// 9 internal decile boundaries from a value array (nulls ignored); null if too few
function decileBoundaries(vals){
  const s = vals.filter(v => v!=null).sort((a,b)=>a-b);
  if(s.length < 20) return null;
  const q = [];
  for(let j=1;j<10;j++) q.push(s[Math.floor(s.length*j/10)]);
  return q;
}
// decile (1..10) of v given 9 boundaries
function decileOf(v, bounds){
  let d = 1; for(const b of bounds){ if(v > b) d++; }
  return d > 10 ? 10 : d;
}

// entry indices of every qualifying streak for one name's decile series
function streakEntries(decs, lo, hi, run){
  const out = []; let c = 0;
  for(let i=0;i<decs.length;i++){
    const d = decs[i];
    if(d!=null && d>=lo && d<=hi){ c++; if(c === run) out.push(i); }   // fire once, at the Nth day
    else c = 0;
  }
  return out;
}

// trailing (no-look-ahead) decile per day: each day's D is ranked against the name's
// PREVIOUS `win` non-null observations (strictly prior -- today's value is excluded from
// its own window), so the cutoffs were knowable in real time. Needs `minObs` history
// before producing a decile, so the first ~6 months yield no events.
const EV_TRAIL_WIN = 252, EV_TRAIL_MIN = 120;
function trailingDecileSeries(d, win=EV_TRAIL_WIN, minObs=EV_TRAIL_MIN){
  const out = new Array(d.length).fill(null);
  const buf = [];                                   // last `win` non-null values, oldest first
  for(let i=0;i<d.length;i++){
    const v = d[i];
    if(v!=null){
      if(buf.length >= minObs){
        let le = 0; for(const x of buf) if(x <= v) le++;
        out[i] = Math.max(1, Math.ceil(le/buf.length * 10));   // percentile -> decile 1..10
      }
      buf.push(v); if(buf.length > win) buf.shift();
    }
  }
  return out;
}

// decile (1..10) per day for every name, under the chosen basis. Computed once per
// basis (cached -- the data is static) and shared by histograms, table and placebos.
const decsCacheEv = {};
function decsByName(basis){
  if(decsCacheEv[basis]) return decsCacheEv[basis];
  const bench = P.bench, dser = P.rel.d;
  const names = Object.keys(dser).filter(n => n !== bench && P.rel.r21[n]);
  let poolB = null;
  if(basis === 'pool'){
    const all = [];
    for(const nm of names){ const a=dser[nm]; if(a) for(const v of a) if(v!=null) all.push(v); }
    poolB = decileBoundaries(all);
  }
  const out = {};
  for(const nm of names){
    const d = dser[nm]; if(!d) continue;
    if(basis === 'trail'){
      out[nm] = trailingDecileSeries(d);
    } else {
      const b = basis === 'pool' ? poolB : decileBoundaries(d);
      if(!b) continue;
      out[nm] = d.map(v => v==null ? null : decileOf(v, b));
    }
  }
  decsCacheEv[basis] = out;
  return out;
}

// forward returns of every event in a decile band, for one horizon. nStreaks = total
// streaks detected (before dropping those without a realized forward return yet);
// days[j] = the trigger day-index of ret[j] (used for matched baselines/regime splits).
function bandReturns(decs, lo, hi, run, h, excess){
  const rser = P.rel['r' + h], br = rser[P.bench], ret = [], days = [];
  let nStreaks = 0;
  for(const nm in decs){
    const r = rser[nm]; if(!r) continue;
    const ents = streakEntries(decs[nm], lo, hi, run); nStreaks += ents.length;
    for(const i of ents){
      let y = r[i]; if(y==null) continue;
      if(excess){ if(br[i]==null) continue; y -= br[i]; }
      ret.push(y); days.push(i);
    }
  }
  return {ret, days, nStreaks};
}

// cross-sectional mean forward return per day (all names, one horizon) -- the
// DATE-MATCHED baseline: an event is compared against how the whole universe did
// from that same day, so "events happened in good months" can't masquerade as edge.
const xsMeanCache = {};
function xsMeanByDay(h, excess){
  const key = h + '|' + excess;
  if(xsMeanCache[key]) return xsMeanCache[key];
  const rser = P.rel['r' + h], br = rser[P.bench];
  let T = 0; for(const nm in rser){ if(nm !== P.bench && rser[nm]) T = Math.max(T, rser[nm].length); }
  const sums = new Array(T).fill(0), ns = new Array(T).fill(0);
  for(const nm in rser){
    if(nm === P.bench) continue;
    const r = rser[nm]; if(!r) continue;
    for(let i=0;i<r.length;i++){
      let y = r[i]; if(y==null) continue;
      if(excess){ if(br[i]==null) continue; y -= br[i]; }
      sums[i] += y; ns[i]++;
    }
  }
  const out = sums.map((s,i)=> ns[i] ? s/ns[i] : null);
  xsMeanCache[key] = out;
  return out;
}
// mean of the matched baseline over a set of event days
function matchedBase(days, h, excess){
  const xs = xsMeanByDay(h, excess);
  let s=0, c=0; for(const i of days){ const v=xs[i]; if(v!=null){ s+=v; c++; } }
  return c ? s/c : null;
}

// unconditional forward returns over every (name, day) -- the "always invested"
// baseline. Pass `tkr` to scope it to one name (matching a single-name event study).
function baselineReturns(h, excess, tkr=null){
  const rser = P.rel['r' + h], br = rser[P.bench], out = [];
  for(const nm in rser){
    if(nm === P.bench) continue;
    if(tkr && nm !== tkr) continue;
    const r = rser[nm]; if(!r) continue;
    for(let i=0;i<r.length;i++){
      let y = r[i]; if(y==null) continue;
      if(excess){ if(br[i]==null) continue; y -= br[i]; }
      out.push(y);
    }
  }
  return out;
}

function mean(a){ return a.length ? a.reduce((s,c)=>s+c,0)/a.length : null; }
function stdev(a, m){ if(a.length<2) return null; let ss=0; for(const v of a) ss+=(v-m)*(v-m); return Math.sqrt(ss/a.length); }
function medianOf(a){ if(!a.length) return null; const s=[...a].sort((x,y)=>x-y); const m=s.length; return m%2?s[(m-1)/2]:(s[m/2-1]+s[m/2])/2; }
// median absolute deviation: median(|x - median(x)|), a robust dispersion measure
function madOf(a){ if(a.length<2) return null; const med=medianOf(a); return medianOf(a.map(v=>Math.abs(v-med))); }

// Permutation (placebo) test for an event mean. Each name's real event dates are mapped
// to positions in that name's list of ELIGIBLE days (days with a realized forward
// return), then the whole pattern is CIRCULARLY SHIFTED by one random offset per name
// per draw. This preserves each name's event count AND the clustering/spacing of its
// events -- so the null distribution inherits the overlapping-window and clustering
// structure that a naive t-test ignores. Returns a two-sided empirical p-value and the
// percentile of the real mean among the placebo means.
function permPValue(decs, lo, hi, run, h, excess, B=500){
  const rser = P.rel['r' + h], br = rser[P.bench];
  const perName = [], realVals = [];
  for(const nm in decs){
    const r = rser[nm]; if(!r) continue;
    const elig = [], posOf = new Map();             // eligible-day returns + day-index -> position
    for(let i=0;i<r.length;i++){
      let y = r[i]; if(y==null) continue;
      if(excess){ if(br[i]==null) continue; y -= br[i]; }
      posOf.set(i, elig.length); elig.push(y);
    }
    if(!elig.length) continue;
    const pos = [];
    for(const i of streakEntries(decs[nm], lo, hi, run)){
      const p = posOf.get(i);
      if(p !== undefined){ pos.push(p); realVals.push(elig[p]); }
    }
    if(pos.length) perName.push({elig, pos});
  }
  if(realVals.length < 3) return null;
  const realMean = realVals.reduce((s,c)=>s+c,0)/realVals.length;
  const nullMeans = [];
  for(let b=0;b<B;b++){
    let s=0, c=0;
    for(const {elig, pos} of perName){
      const off = Math.floor(Math.random()*elig.length);
      for(const p of pos){ s += elig[(p+off)%elig.length]; c++; }
    }
    nullMeans.push(s/c);
  }
  const nullMean = nullMeans.reduce((s,c)=>s+c,0)/nullMeans.length;
  const dev = Math.abs(realMean - nullMean);
  let ge = 0, below = 0;
  for(const m of nullMeans){ if(Math.abs(m - nullMean) >= dev) ge++; if(m < realMean) below++; }
  return {p: (ge + 1)/(B + 1), pct: below/B*100, B};
}

// Event-time drift: mean cumulative return at each offset 0..maxK days after the
// trigger, across all events in the band, with a per-offset +/-1 SE. Events near the
// end of the sample contribute only their available prefix (n shrinks with k).
function eventDrift(decs, lo, hi, run, excess, maxK=63){
  const close = P.rel.close, bclose = close[P.bench];
  const sums = new Array(maxK+1).fill(0), sqs = new Array(maxK+1).fill(0), ns = new Array(maxK+1).fill(0);
  let events = 0; const days = [];
  for(const nm in decs){
    const c = close[nm]; if(!c) continue;
    for(const i of streakEntries(decs[nm], lo, hi, run)){
      if(c[i]==null || (excess && bclose[i]==null)) continue;
      events++; days.push(i);
      for(let k=0; k<=maxK && i+k<c.length; k++){
        const ck = c[i+k]; if(ck==null) continue;
        let v = (ck/c[i] - 1) * 100;
        if(excess){ const bk = bclose[i+k]; if(bk==null) continue; v -= (bk/bclose[i] - 1) * 100; }
        sums[k] += v; sqs[k] += v*v; ns[k]++;
      }
    }
  }
  const meanP = [], seP = [];
  for(let k=0;k<=maxK;k++){
    if(!ns[k]){ meanP.push(null); seP.push(null); continue; }
    const m = sums[k]/ns[k]; meanP.push(m);
    seP.push(ns[k] > 1 ? Math.sqrt(Math.max(0, sqs[k]/ns[k] - m*m))/Math.sqrt(ns[k]) : 0);
  }
  return {mean: meanP, se: seP, n: ns, events, days};
}

// DATE-MATCHED drift baseline: from each event's trigger day, the cross-sectional
// mean path of ALL names over the same window, averaged across events (with
// multiplicity). This is the fair control -- it starts on the exact same dates as
// the events, so a rally that happens to follow event-heavy months shows up in the
// baseline too, not just in the event curve.
function matchedDrift(days, excess, maxK=63){
  const close = P.rel.close, bclose = close[P.bench];
  const sums = new Array(maxK+1).fill(0), ns = new Array(maxK+1).fill(0);
  for(const i of days){
    const s = new Array(maxK+1).fill(0), c = new Array(maxK+1).fill(0);
    if(excess && bclose[i]==null) continue;
    for(const nm in close){
      if(nm === P.bench) continue;
      const cl = close[nm]; if(!cl || cl[i]==null) continue;
      for(let k=0; k<=maxK && i+k<cl.length; k++){
        const ck = cl[i+k]; if(ck==null) continue;
        let v = (ck/cl[i] - 1) * 100;
        if(excess){ const bk = bclose[i+k]; if(bk==null) continue; v -= (bk/bclose[i] - 1) * 100; }
        s[k] += v; c[k]++;
      }
    }
    for(let k=0;k<=maxK;k++) if(c[k]){ sums[k] += s[k]/c[k]; ns[k]++; }
  }
  return sums.map((s,k)=> ns[k] ? s/ns[k] : null);
}

function renderDriftChart(dr, base, maxK=63){
  const w=800, H=300, padL=52, padR=16, padT=14, padB=30;
  let lo=0, hi=0;
  dr.mean.forEach((m,k)=>{ if(m==null) return; const s=dr.se[k]||0; lo=Math.min(lo,m-s); hi=Math.max(hi,m+s); });
  for(const b of base) if(b!=null){ lo=Math.min(lo,b); hi=Math.max(hi,b); }
  const pad=(hi-lo)*0.12||1; lo-=pad; hi+=pad;
  const X = k => padL + (w-padL-padR)*k/maxK;
  const Y = v => (H-padB) - (H-padT-padB)*((v-lo)/(hi-lo));
  let grid='';
  for(const t of niceTicks(lo, hi, 5)){
    const y=Y(t), zero=Math.abs(t)<1e-9;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w-padR}" y2="${y.toFixed(1)}" stroke="${zero?'var(--zero)':'var(--grid)'}" stroke-width="1"${zero?'':' stroke-dasharray="2,3"'}/>`
          + `<text x="${padL-6}" y="${(y+3).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="end">${fmtPct(t)}</text>`;
  }
  for(const k of [0,21,42,63]){
    const x=X(k);
    grid += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="var(--grid)" stroke-width="0.5" stroke-dasharray="2,3"/>`
          + `<text x="${x.toFixed(1)}" y="${(H-padB+13).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="middle">${k===0?'day 0':'+'+k}</text>`;
  }
  // ±1 SE band around the event mean (upper edge forward, lower edge back)
  let up='', dn='';
  for(let k=0;k<=maxK;k++){
    const m=dr.mean[k]; if(m==null) continue;
    const s=dr.se[k]||0;
    up += (up?'L':'M') + X(k).toFixed(1) + ',' + Y(m+s).toFixed(1) + ' ';
  }
  for(let k=maxK;k>=0;k--){
    const m=dr.mean[k]; if(m==null) continue;
    const s=dr.se[k]||0;
    dn += 'L' + X(k).toFixed(1) + ',' + Y(m-s).toFixed(1) + ' ';
  }
  const band = up ? `<path d="${up}${dn}Z" fill="#e3b341" opacity="0.14"/>` : '';
  const line = pts => {
    let d='', started=false;
    pts.forEach((v,k)=>{ if(v==null){ started=false; return; } d += (started?'L':'M') + X(k).toFixed(1) + ',' + Y(v).toFixed(1) + ' '; started=true; });
    return d;
  };
  const baseLine = `<path d="${line(base)}" fill="none" stroke="var(--mut)" stroke-width="1.2" stroke-dasharray="4,3"/>`;
  const meanLine = `<path d="${line(dr.mean)}" fill="none" stroke="#e3b341" stroke-width="2" stroke-linejoin="round"/>`;
  const legend = `<text x="${(w-padR).toFixed(1)}" y="${(padT+9).toFixed(1)}" font-size="9.5" text-anchor="end">`
    + `<tspan fill="#e3b341" font-weight="700">event mean &plusmn;1 SE</tspan>`
    + `<tspan fill="var(--mut)">&nbsp;&nbsp;&#183;&nbsp;&nbsp;matched baseline (all names, same dates)</tspan></text>`;
  return `${grid}${band}${baseLine}${meanLine}${legend}`;
}

function renderHistogram(values, meanV, baseV){
  const w=800, H=220, padL=44, padR=14, padT=16, padB=26;
  if(values.length < 2)
    return `<text x="400" y="110" font-size="12" fill="var(--mut)" text-anchor="middle">no events at this horizon</text>`;
  const s = [...values].sort((a,b)=>a-b);
  const q = p => s[Math.min(s.length-1, Math.max(0, Math.floor(p*(s.length-1))))];
  let lo = q(0.01), hi = q(0.99);
  lo = Math.min(lo, 0, meanV, baseV); hi = Math.max(hi, 0, meanV, baseV);
  if(hi <= lo) hi = lo + 1;
  const NB = 26, bw = (hi-lo)/NB, bins = new Array(NB).fill(0);
  for(const v of values){ let b = Math.floor((v-lo)/bw); b = Math.max(0, Math.min(NB-1, b)); bins[b]++; }
  const maxc = Math.max(...bins, 1);
  const X = v => padL + (w-padL-padR)*((v-lo)/(hi-lo));
  const Yc = c => (H-padB) - (H-padT-padB)*(c/maxc);
  let grid='';
  for(const t of niceTicks(lo, hi, 6)){
    const x=X(t), zero=Math.abs(t)<1e-9;
    grid += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="${zero?'var(--zero)':'var(--grid)'}" stroke-width="${zero?1:0.5}"${zero?'':' stroke-dasharray="2,3"'}/>`
          + `<text x="${x.toFixed(1)}" y="${(H-padB+13).toFixed(1)}" font-size="9" fill="var(--mut)" text-anchor="middle">${fmtPct(t)}</text>`;
  }
  let bars='';
  bins.forEach((c,i)=>{
    if(!c) return;
    const x0=X(lo+i*bw), x1=X(lo+(i+1)*bw), y=Yc(c);
    bars += `<rect x="${(x0+0.5).toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1,x1-x0-1).toFixed(1)}" height="${(H-padB-y).toFixed(1)}" fill="var(--accent)" opacity="0.55"/>`;
  });
  const vline = (v,color,dash,label) =>
    `<line x1="${X(v).toFixed(1)}" y1="${padT}" x2="${X(v).toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="${color}" stroke-width="1.5"${dash?` stroke-dasharray="${dash}"`:''}/>`
    + `<text x="${(X(v)+ (X(v)>w-70?-4:4)).toFixed(1)}" y="${(padT+9).toFixed(1)}" font-size="9" font-weight="700" fill="${color}" text-anchor="${X(v)>w-70?'end':'start'}">${label}</text>`;
  const marks = vline(baseV, 'var(--mut)', '3,3', `matched ${fmtPct(baseV)}`)
              + vline(meanV, '#e3b341', '', `event ${fmtPct(meanV)}`);
  return `${grid}${bars}${marks}`;
}

const fmtSigned = v => (v==null ? '--' : (v>=0?'+':'') + v.toFixed(2) + '%');
let evSort = {key: null, dir: 1};   // table sort: key in {21,42,63} (by mean), dir 1=desc

const EV_BASIS_LBL = {
  trail: `per-name trailing deciles (last ${EV_TRAIL_WIN} obs, min ${EV_TRAIL_MIN} -- no look-ahead)`,
  name: "per-name full-sample deciles (mild look-ahead)",
  pool: "pooled (global) deciles",
};

// Live streak status: which names are in a qualifying streak RIGHT NOW (trailing run
// ending at their latest print, >= run days), which are one day away, and -- when
// nothing is active -- the most recent completed event. "Current" requires the name's
// latest non-null print to be within the last 3 trading days of the sample, so stale
// series can't show as live.
function currentStreaks(decs, lo, hi, run){
  const L = P.rel.dates.length - 1;
  const active = [], building = [];
  let lastEvent = null;
  for(const nm in decs){
    const d = decs[nm]; if(!d) continue;
    let li = d.length - 1; while(li >= 0 && d[li]==null) li--;
    if(li < 0) continue;
    if(li >= L - 2){
      let c = 0, j = li;
      while(j >= 0 && d[j]!=null && d[j] >= lo && d[j] <= hi){ c++; j--; }
      if(c >= run) active.push({nm, c, dec: d[li]});
      else if(c === run - 1) building.push({nm, c, dec: d[li]});
    }
    const es = streakEntries(d, lo, hi, run);
    if(es.length){
      const i = es[es.length - 1];
      if(!lastEvent || i > lastEvent.i) lastEvent = {nm, i};
    }
  }
  active.sort((a,b)=>b.c-a.c);
  return {active, building, lastEvent};
}

function renderCurrentStreak(decs){
  const el = document.getElementById('evNow');
  const L = P.rel.dates.length - 1, asOf = P.rel.dates[L];
  const {active, building, lastEvent} = currentStreaks(decs, evLo, evHi, evRun);
  const bandLbl = evLo===evHi ? `D${evLo}` : `D${evLo}–D${evHi}`;
  const chip = a =>
    `<span style="border:1px solid #e3b341;border-radius:6px;padding:2px 8px;margin-right:4px;background:rgba(227,179,65,.09);white-space:nowrap">` +
    `<b style="color:#e3b341">${a.nm}</b> ${a.c} days &middot; today D${a.dec}</span>`;
  let html;
  if(active.length){
    html = `<b style="color:#e3b341">&#9889; ${active.length} active streak${active.length>1?'s':''} right now</b> ` +
      `<span class="mut">(&ge;${evRun} consecutive days in ${bandLbl}, as of ${asOf})</span><br>` +
      active.slice(0,12).map(chip).join(' ') +
      (active.length > 12 ? ` <span class="mut">+${active.length-12} more</span>` : '');
  } else {
    html = `No active ${evRun}-day streak in ${bandLbl} as of <b>${asOf}</b>.`;
    if(lastEvent){
      const ago = L - lastEvent.i;
      html += ` Last completed event: <b>${lastEvent.nm}</b> on <b>${P.rel.dates[lastEvent.i]}</b> ` +
        `<span class="mut">(${ago} trading day${ago===1?'':'s'} ago)</span>.`;
    } else {
      html += ` No completed events anywhere in the sample for this configuration.`;
    }
  }
  if(building.length){
    html += `<br><span class="mut">One day away (${evRun-1}/${evRun}): ` +
      building.slice(0,8).map(x=>`${x.nm} (today D${x.dec})`).join(', ') +
      (building.length > 8 ? ` +${building.length-8} more` : '') + `</span>`;
  }
  el.innerHTML = html;
  el.style.borderColor = active.length ? '#e3b341' : 'var(--grid)';
}

function renderEvents(){
  evRun = Math.max(2, Math.min(30, parseInt(document.getElementById('evRun').value||'5', 10)));
  const oneTkr = evTicker === '__ALL__' ? null : evTicker;
  document.getElementById('evSub').textContent =
    `Event = ${evRun} consecutive days with the RAW 1-day FINRA dark ratio in deciles ${evLo}–${evHi} ` +
    `(${EV_BASIS_LBL[evBasis]}); forward return measured from the streak's final day` +
    (evExcess ? ', in excess of '+P.bench : '') +
    (oneTkr ? ` · ${oneTkr} only` : ' · all names pooled') +
    (P.rel.range ? ` · ${P.rel.range[0]} → ${P.rel.range[1]}` : '');

  let decs = decsByName(evBasis);                      // computed once per basis (cached)
  if(oneTkr) decs = decs[oneTkr] ? {[oneTkr]: decs[oneTkr]} : {};
  renderCurrentStreak(decs);
  let evN = 0;
  for(const h of ['21','42','63']){
    const {ret, days, nStreaks} = bandReturns(decs, evLo, evHi, evRun, h, evExcess);
    const m = mean(ret), md = medianOf(ret), mad = madOf(ret);
    const bm = matchedBase(days, h, evExcess);         // all names, same trigger dates
    const hit = ret.length ? ret.filter(v=>v>0).length/ret.length*100 : null;
    const perm = permPValue(decs, evLo, evHi, evRun, h, evExcess);
    evN = nStreaks;
    document.getElementById('evHist'+h).innerHTML = renderHistogram(ret, m==null?0:m, bm==null?0:bm);
    document.getElementById('evStats'+h).innerHTML =
      `<span>events = <b>${ret.length.toLocaleString()}</b></span>` +
      `<span title="matched base = mean forward return of ALL names from the events' exact trigger dates">mean = <b>${fmtSigned(m)}</b> (matched base <b>${fmtSigned(bm)}</b>)</span>` +
      `<span>median = <b>${fmtSigned(md)}</b></span>` +
      `<span>MAD = <b>${mad==null?'--':mad.toFixed(2)+'%'}</b></span>` +
      `<span>hit rate = <b>${hit==null?'--':hit.toFixed(0)+'%'}</b></span>` +
      `<span title="two-sided empirical p from ${perm?perm.B:500} circular-shift placebos; pctl = where the real mean sits among placebo means">perm p = <b>${perm==null?'--':perm.p.toFixed(3)}</b>${perm==null?'':' (pctl '+perm.pct.toFixed(0)+')'}</span>`;
  }
  // event-time drift curve (needs the packed CLOSE series; absent in older payloads)
  if(P.rel.close){
    const dr = eventDrift(decs, evLo, evHi, evRun, evExcess);
    const bp = matchedDrift(dr.days, evExcess);        // control path from the same dates
    if(dr.events < 1){
      document.getElementById('evDrift').innerHTML =
        `<text x="400" y="150" font-size="12" fill="var(--mut)" text-anchor="middle">no events for this band / streak length</text>`;
      document.getElementById('evDriftStats').innerHTML = '';
    } else {
      document.getElementById('evDrift').innerHTML = renderDriftChart(dr, bp);
      const at = k => `${fmtSigned(dr.mean[k])} <span style="opacity:.6">(matched ${fmtSigned(bp[k])})</span>`;
      document.getElementById('evDriftStats').innerHTML =
        `<span>events = <b>${dr.events.toLocaleString()}</b></span>` +
        `<span>day +21: <b>${at(21)}</b></span>` +
        `<span>day +42: <b>${at(42)}</b></span>` +
        `<span>day +63: <b>${at(63)}</b></span>` +
        `<span>n at day 63 = <b>${dr.n[63].toLocaleString()}</b></span>`;
    }
  } else {
    document.getElementById('evDrift').innerHTML =
      `<text x="400" y="150" font-size="12" fill="var(--mut)" text-anchor="middle">regenerate the dashboard to add CLOSE data for the drift curve</text>`;
  }
  renderRegimeSplit(decs);
  renderBacktest(decs);
  renderEventTable(decs);
  document.getElementById('evNote').innerHTML =
    `<b>${evN.toLocaleString()}</b> streaks in the selected band ` +
    (oneTkr ? `for <b>${oneTkr}</b> (single-name mode: expect few events -- means and p-values are correspondingly noisy). `
            : `across ${P.order_raw.length} names. `) +
    `<b>Streaks are computed on the raw (unsmoothed) 1-day FINRA dark ratio</b> -- off-exchange volume / total ` +
    `volume for that single day -- NOT SqueezeMetrics' 5-day-MA D; the footer's "D = 5-day..." wording refers to ` +
    `the Small-multiples tab only. <b>Matched base</b> compares each event against the mean forward return of ` +
    `ALL names from the event's exact trigger date, so events clustering in months that happened to rally can't ` +
    `masquerade as edge (the old always-invested baseline had that flaw); the drift curve's dashed control is ` +
    `built the same way. The <b>regime table</b> splits events by sub-period -- an effect that only exists in one ` +
    `half is a regime artifact, not a signal. ` +
    `The 55-band table sweeps every D<sub>lo</sub>&rarr;D<sub>hi</sub> band at the current settings; each cell ` +
    `shows the mean forward return, <b>&plusmn;MAD</b> (robust dispersion), and a <b>&plusmn;1&sigma; volatility ` +
    `band</b> (mean&minus;&sigma; &hellip; mean+&sigma;, the range ~68% of events land in if roughly normal -- fat ` +
    `tails make it optimistic); hover for median, mean/MAD and &sigma;. Sort by any horizon's mean or by ` +
    `<b>Streaks</b> (click again to flip; click Band to restore order); the baseline row is the plain all-days average. ` +
    `The <b>drift curve</b> traces the average cumulative return day-by-day after the trigger (n shrinks toward ` +
    `day +63 as late events run out of realized data) -- read its shape to pick a holding period. ` +
    `<b>perm p</b> is a two-sided empirical p-value from 500 <i>circular-shift placebos</i>: each name's real ` +
    `event-date pattern is slid by a random offset (event count and clustering preserved) and the pooled mean ` +
    `recomputed. <b>Trailing deciles</b> rank each day's ratio against the name's previous ` +
    `${EV_TRAIL_WIN} observations (min ${EV_TRAIL_MIN}, strictly prior), so cutoffs are knowable in real time; ` +
    `the first ~6 months of history produce no events while the window fills. Beware the 55-band sweep is a ` +
    `multiple-testing machine: judge the band you chose ex-ante, not the best cell in the table.`;
}

// Regime robustness: the selected band's events re-measured inside sub-periods.
// Split at 2024-01-01 when that date falls inside the sample; otherwise at the
// sample midpoint. Each cell: event mean vs the matched (same-dates) baseline.
function renderRegimeSplit(decs){
  const el = document.getElementById('evRegime');
  const dates = P.rel.dates;
  if(!dates || !dates.length){ el.innerHTML = '<tbody><tr><td>regenerate the dashboard to add per-day dates for the regime split</td></tr></tbody>'; return; }
  const BOUNDARY = '2024-01-01';
  let split = dates.findIndex(d => d >= BOUNDARY);
  let lblA, lblB;
  if(split <= 0 || split >= dates.length - 1){
    split = Math.floor(dates.length/2);
    lblA = `first half (${dates[0]} → ${dates[split-1]})`;
    lblB = `second half (${dates[split]} → ${dates[dates.length-1]})`;
  } else {
    lblA = `${dates[0].slice(0,4)}–${dates[split-1].slice(0,4)}`;
    lblB = `${dates[split].slice(0,4)} → present`;
  }
  const H = ['21','42','63'];
  const periods = [
    {lbl: 'full sample', lo: 0, hi: Infinity},
    {lbl: lblA, lo: 0, hi: split},
    {lbl: lblB, lo: split, hi: Infinity},
  ];
  let html = `<thead><tr><th>Period</th>` + H.map(h=>`<th>${HMONTH[h]}</th>`).join('') + `</tr></thead><tbody>`;
  for(const per of periods){
    let cells = '';
    for(const h of H){
      const {ret, days} = bandReturns(decs, evLo, evHi, evRun, h, evExcess);
      const fRet = [], fDays = [];
      for(let j=0;j<ret.length;j++) if(days[j] >= per.lo && days[j] < per.hi){ fRet.push(ret[j]); fDays.push(days[j]); }
      const m = mean(fRet), bm = matchedBase(fDays, h, evExcess);
      if(m==null){ cells += `<td>--</td>`; continue; }
      const d = (bm!=null) ? m - bm : null;
      cells += `<td title="event mean ${fmtSigned(m)}, matched base ${fmtSigned(bm)}, edge ${fmtSigned(d)}, n=${fRet.length}">`
             + `<span class="${m>=0?'p':'n'}">${fmtSigned(m)}</span>`
             + (d==null ? '' : `<br><span class="mut">edge </span><span class="${d>=0?'p':'n'}">${fmtSigned(d)}</span>`)
             + `<br><span class="mut">vs ${fmtSigned(bm)} &middot; n ${fRet.length}</span></td>`;
    }
    html += `<tr><td>${per.lbl}</td>${cells}</tr>`;
  }
  html += `</tbody>`;
  el.innerHTML = html;
}

// -------------------------------------------------------------------------
// T+1 strategy backtest. A streak completing on day i is knowable only after
// that day's close (FINRA publishes in the evening), so entry is at close[i+lag]
// with lag=1; exit at close[entry+H]. Equal weight across all open positions,
// rebalanced daily; 100% cash when no position is open. Costs charged per side.
// With `excess`, each position is long stock / short QQQ (daily-differenced).
// -------------------------------------------------------------------------
let evCost = 5;
function strategyBacktest(decs, lo, hi, run, H, excess, costBps, lag){
  const close = P.rel.close, bclose = close[P.bench];
  const T = P.rel.dates.length, cost = costBps/10000;
  const trades = [];
  for(const nm in decs){
    const c = close[nm]; if(!c) continue;
    for(const i of streakEntries(decs[nm], lo, hi, run)){
      const e = i + lag;
      if(e >= T || c[e]==null) continue;
      let x = Math.min(e + H, T - 1);
      while(x > e && c[x]==null) x--;
      if(x <= e) continue;
      trades.push({nm, e, x});
    }
  }
  const opens = Array.from({length: T}, () => []);
  for(const tr of trades) opens[tr.e].push(tr);
  const active = new Set();
  const daily = new Array(T).fill(0), npos = new Array(T).fill(0);
  for(let t=0;t<T;t++){
    for(const tr of opens[t]) active.add(tr);
    let s=0, n=0;
    for(const tr of active){
      if(t > tr.e && t <= tr.x){
        const c = close[tr.nm];
        if(c[t]!=null && c[t-1]!=null){
          let r = c[t]/c[t-1] - 1;
          if(excess && bclose[t]!=null && bclose[t-1]!=null) r -= bclose[t]/bclose[t-1] - 1;
          if(t === tr.e+1) r -= cost;          // entry side
          if(t === tr.x)   r -= cost;          // exit side
          s += r; n++;
        }
      }
      if(t >= tr.x) active.delete(tr);
    }
    daily[t] = n ? s/n : 0;
    npos[t] = n;
  }
  const eq = new Array(T); eq[0] = 1;
  for(let t=1;t<T;t++) eq[t] = eq[t-1]*(1+daily[t]);
  // stats
  const total = eq[T-1] - 1, years = T/252;
  const cagr = Math.pow(eq[T-1], 1/years) - 1;
  let mu=0; for(const r of daily) mu += r; mu /= T;
  let sg=0; for(const r of daily) sg += (r-mu)*(r-mu); sg = Math.sqrt(sg/T);
  const sharpe = sg > 0 ? mu/sg*Math.sqrt(252) : null;
  let peak=1, mdd=0;
  for(const v of eq){ if(v>peak) peak=v; const dd=v/peak-1; if(dd<mdd) mdd=dd; }
  let wins=0, judged=0;
  for(const tr of trades){
    const c = close[tr.nm];
    if(c[tr.e]==null || c[tr.x]==null) continue;
    let r = c[tr.x]/c[tr.e] - 1 - 2*cost;
    if(excess && bclose[tr.e]!=null && bclose[tr.x]!=null) r -= bclose[tr.x]/bclose[tr.e] - 1;
    judged++; if(r > 0) wins++;
  }
  let inv=0, posSum=0;
  for(const n of npos){ if(n>0){ inv++; posSum += n; } }
  return {eq, nTrades: trades.length, total, cagr, sharpe, maxDD: mdd,
          winRate: judged ? wins/judged*100 : null,
          avgPos: inv ? posSum/inv : 0, pctIn: inv/T*100};
}

function renderEquityChart(runs, cols){
  const w=800, H=320, padL=52, padR=16, padT=14, padB=30;
  const T = P.rel.dates.length, dates = P.rel.dates;
  const bclose = P.rel.close[P.bench];
  // bench buy&hold reference, normalized at its first valid close
  let bEq = null;
  if(bclose){
    let f=-1; for(let i=0;i<T;i++) if(bclose[i]!=null){ f=i; break; }
    if(f>=0) bEq = bclose.map(v => v==null ? null : v/bclose[f]);
  }
  let lo=1, hi=1;
  for(const h in runs) for(const v of runs[h].eq){ if(v<lo)lo=v; if(v>hi)hi=v; }
  if(bEq) for(const v of bEq) if(v!=null){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const pad=(hi-lo)*0.08||0.1; lo-=pad; hi+=pad;
  const X = t => padL + (w-padL-padR)*t/(T-1);
  const Y = v => (H-padB) - (H-padT-padB)*((v-lo)/(hi-lo));
  let grid='';
  for(const tk of niceTicks(lo, hi, 5)){
    const y=Y(tk), one=Math.abs(tk-1)<1e-9;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w-padR}" y2="${y.toFixed(1)}" stroke="${one?'var(--zero)':'var(--grid)'}" stroke-width="1"${one?'':' stroke-dasharray="2,3"'}/>`
          + `<text x="${padL-6}" y="${(y+3).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="end">${tk.toFixed(2)}&#215;</text>`;
  }
  for(let t=1;t<T;t++){
    if(dates[t].slice(0,4) !== dates[t-1].slice(0,4)){
      const x=X(t);
      grid += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${(H-padB).toFixed(1)}" stroke="var(--grid)" stroke-width="0.5" stroke-dasharray="2,3"/>`
            + `<text x="${x.toFixed(1)}" y="${(H-padB+13).toFixed(1)}" font-size="9.5" fill="var(--mut)" text-anchor="middle">${dates[t].slice(0,4)}</text>`;
    }
  }
  const line = (arr, color, dash) => {
    let d='', started=false;
    for(let t=0;t<T;t++){
      const v=arr[t]; if(v==null){ started=false; continue; }
      d += (started?'L':'M') + X(t).toFixed(1) + ',' + Y(v).toFixed(1) + ' ';
      started=true;
    }
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.6"${dash?` stroke-dasharray="${dash}"`:''} stroke-linejoin="round"/>`;
  };
  let lines = bEq ? line(bEq, 'var(--mut)', '4,3') : '';
  for(const h in runs) lines += line(runs[h].eq, cols[h]);
  let lx = w-padR, legend = `<text x="${lx}" y="${(padT+9).toFixed(1)}" font-size="9.5" text-anchor="end">`
    + Object.keys(runs).map(h=>`<tspan fill="${cols[h]}" font-weight="700">hold ${HMONTH[h]}</tspan>`).join(`<tspan fill="var(--mut)"> &#183; </tspan>`)
    + `<tspan fill="var(--mut)"> &#183; ${P.bench} buy&amp;hold</tspan></text>`;
  return `${grid}${lines}${legend}`;
}

function renderBacktest(decs){
  const el = document.getElementById('evBt'), st = document.getElementById('evBtStats');
  evCost = Math.max(0, Math.min(100, parseInt(document.getElementById('evCost').value||'5', 10)));
  if(!P.rel.close || !P.rel.dates){
    el.innerHTML = `<text x="400" y="160" font-size="12" fill="var(--mut)" text-anchor="middle">regenerate the dashboard to add CLOSE data for the backtest</text>`;
    st.innerHTML = ''; return;
  }
  const HS = ['21','42','63'], cols = {'21':'#58a6ff','42':'#3fb950','63':'#e3b341'};
  const runs = {}, runs0 = {};
  let any = false;
  for(const h of HS){
    runs[h]  = strategyBacktest(decs, evLo, evHi, evRun, +h, evExcess, evCost, 1);
    runs0[h] = strategyBacktest(decs, evLo, evHi, evRun, +h, evExcess, evCost, 0);
    if(runs[h].nTrades > 0) any = true;
  }
  if(!any){
    el.innerHTML = `<text x="400" y="160" font-size="12" fill="var(--mut)" text-anchor="middle">no trades for this band / streak length</text>`;
    st.innerHTML = ''; return;
  }
  el.innerHTML = renderEquityChart(runs, cols);
  const pc = v => v==null ? '--' : (v>=0?'+':'') + (v*100).toFixed(1) + '%';
  let html = `<thead><tr><th>Hold</th><th>Trades</th><th>Total</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Win %</th><th>Avg pos</th><th>% invested</th><th title="same strategy executed at the unattainable completion-day close; the gap vs Total is the cost of the T+1 lag">T+0 total</th></tr></thead><tbody>`;
  for(const h of HS){
    const r = runs[h], r0 = runs0[h];
    html += `<tr><td style="color:${cols[h]};font-weight:700">${HMONTH[h]}</td>`
      + `<td>${r.nTrades.toLocaleString()}</td>`
      + `<td><span class="${r.total>=0?'p':'n'}">${pc(r.total)}</span></td>`
      + `<td>${pc(r.cagr)}</td>`
      + `<td>${r.sharpe==null?'--':r.sharpe.toFixed(2)}</td>`
      + `<td><span class="n">${pc(r.maxDD)}</span></td>`
      + `<td>${r.winRate==null?'--':r.winRate.toFixed(0)+'%'}</td>`
      + `<td>${r.avgPos.toFixed(1)}</td>`
      + `<td>${r.pctIn.toFixed(0)}%</td>`
      + `<td>${pc(r0.total)} <span class="mut">(lag ${pc(r.total - r0.total)})</span></td></tr>`;
  }
  html += `</tbody>`;
  st.innerHTML = html;
}
document.getElementById('evCost').addEventListener('change', renderEvents);

// summary table: every decile band (lo<=hi) x 1mo/2mo/3mo. Each cell: mean forward
// return, ±MAD (robust dispersion), and a ±1σ volatility band (mean−σ … mean+σ).
// Sortable by any horizon's mean or by streak count; click Band to restore order.
function renderEventTable(decs){
  const H = ['21','42','63'];
  const rows = [];
  for(let lo=1;lo<=10;lo++) for(let hi=lo;hi<=10;hi++){
    const cells = {}; let nStreaks = 0;
    for(const h of H){
      const {ret, nStreaks: ns} = bandReturns(decs, lo, hi, evRun, h, evExcess);
      nStreaks = ns;   // horizon-independent count
      const m = mean(ret);
      cells[h] = {n: ret.length, m, md: medianOf(ret), mad: madOf(ret), sd: stdev(ret, m)};
    }
    rows.push({lo, hi, nStreaks, cells});
  }
  // baseline row for reference
  const baseCells = {};
  for(const h of H){
    const b = baselineReturns(h, evExcess, evTicker==='__ALL__'?null:evTicker);
    const m = mean(b);
    baseCells[h] = {n:b.length, m, md:medianOf(b), mad:madOf(b), sd:stdev(b, m)};
  }

  if(evSort.key === 'streaks'){
    rows.sort((a,b)=> (b.nStreaks - a.nStreaks) * evSort.dir);
  } else if(evSort.key){
    rows.sort((a,b)=>{
      const av=a.cells[evSort.key].m, bv=b.cells[evSort.key].m;
      if(av==null) return 1; if(bv==null) return -1;
      return (bv-av) * evSort.dir;
    });
  }
  const cell = c => {
    if(c.m==null) return `<td>--</td>`;
    const cls = c.m>=0 ? 'p' : 'n';
    const ratio = (c.mad && c.mad>0) ? (c.m/c.mad).toFixed(2) : '--';
    const band = c.sd==null ? '--' : `${fmtSigned(c.m-c.sd)} &hellip; ${fmtSigned(c.m+c.sd)}`;
    return `<td title="median ${fmtSigned(c.md)}, MAD ${c.mad==null?'--':c.mad.toFixed(2)+'%'}, mean/MAD ${ratio}, sigma ${c.sd==null?'--':c.sd.toFixed(2)+'%'}, n=${c.n.toLocaleString()}">`
         + `<span class="${cls}">${fmtSigned(c.m)}</span>`
         + `<br><span class="mut">&plusmn;${c.mad==null?'--':c.mad.toFixed(1)} MAD</span>`
         + `<br><span class="mut">&sigma; ${band}</span></td>`;
  };
  const arrow = k => evSort.key===k ? (evSort.dir>0 ? ' ▾' : ' ▴') : '';
  let html = `<thead><tr><th data-k="band" class="sortable" title="click to restore natural band order">Band</th>`
    + `<th data-k="streaks" class="sortable" title="click to sort by streak count">Streaks${arrow('streaks')}</th>`
    + H.map(h=>`<th data-k="${h}" class="sortable" title="click to sort by ${HMONTH[h]} mean">${HMONTH[h]}${arrow(h)}</th>`).join('')
    + `</tr></thead><tbody>`;
  html += `<tr class="baserow"><td>baseline</td><td>--</td>` + H.map(h=>cell(baseCells[h])).join('') + `</tr>`;
  for(const r of rows){
    const label = r.lo===r.hi ? `D${r.lo}` : `D${r.lo}–D${r.hi}`;
    html += `<tr><td>${label}</td><td>${r.nStreaks.toLocaleString()}</td>` + H.map(h=>cell(r.cells[h])).join('') + `</tr>`;
  }
  html += `</tbody>`;
  document.getElementById('evTable').innerHTML = html;
}
document.getElementById('evTable').addEventListener('click', e=>{
  const th = e.target.closest('th.sortable'); if(!th) return;
  const k = th.dataset.k;
  if(k === 'band'){ evSort = {key: null, dir: 1}; }
  else if(evSort.key === k) evSort.dir *= -1;
  else { evSort.key = k; evSort.dir = 1; }
  renderEvents();
});

// populate decile selectors 1..10 and the ticker dropdown (alphabetical; bench excluded
// since the event study never includes it)
(function(){
  const lo=document.getElementById('evLo'), hi=document.getElementById('evHi');
  for(let d=1;d<=10;d++){
    lo.insertAdjacentHTML('beforeend', `<option value="${d}"${d===evLo?' selected':''}>D${d}</option>`);
    hi.insertAdjacentHTML('beforeend', `<option value="${d}"${d===evHi?' selected':''}>D${d}</option>`);
  }
  const sel=document.getElementById('evTkr');
  const names = Object.keys(P.rel.d).filter(n => n !== P.bench).sort();
  for(const tkr of names){
    const opt=document.createElement('option');
    opt.value=tkr; opt.textContent=tkr;
    sel.appendChild(opt);
  }
})();
document.getElementById('evTkr').addEventListener('change', e=>{ evTicker=e.target.value; renderEvents(); });
document.getElementById('evRun').addEventListener('change', renderEvents);
document.getElementById('evLo').addEventListener('change', e=>{ evLo=+e.target.value; if(evHi<evLo){ evHi=evLo; document.getElementById('evHi').value=evHi; } renderEvents(); });
document.getElementById('evHi').addEventListener('change', e=>{ evHi=+e.target.value; if(evHi<evLo){ evLo=evHi; document.getElementById('evLo').value=evLo; } renderEvents(); });
document.getElementById('evBasisSeg').addEventListener('click', e=>{ const b=e.target.closest('button'); if(!b) return; evBasis=b.dataset.b; [...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b)); renderEvents(); });
document.getElementById('evExcess').addEventListener('change', e=>{ evExcess=e.target.checked; renderEvents(); });


// -------------------------------------------------------------------------
// Tab: Sector DIX -- reconstructed dollar-DIX per sector ETF, a ranking bar (1y percentile
// of dark accumulation) plus a small-multiple DIX sparkline per sector.
// -------------------------------------------------------------------------
let sectorLevel = 'sector';   // 'sector' (broad GICS) | 'subsector' (SPDR industry funds)
function renderSectors(){
  const S = P.sectors;
  const sub = document.getElementById('sectorsSub');
  const rank = document.getElementById('sectorRank');
  const grid = document.getElementById('sectorGrid');
  const title = document.getElementById('sectorRankTitle');
  if(!S || !S.items || !S.items.length){
    sub.textContent = 'sector DIX unavailable (live builds only)';
    rank.innerHTML = '<text x="450" y="150" font-size="12" fill="var(--mut)" text-anchor="middle">No sector data (run a live build).</text>';
    grid.innerHTML = ''; return;
  }
  const lvl = sectorLevel;
  const noun = lvl === 'subsector' ? 'subsector' : 'sector';
  const pool = S.items.filter(it => (it.level || 'sector') === lvl);
  if(title) title.innerHTML = `Dark accumulation by ${noun} &middot; reconstructed dollar-DIX, ranked by 1-year percentile`;
  if(!pool.length){
    sub.textContent = `no ${noun} data in this build`;
    rank.innerHTML = `<text x="450" y="150" font-size="12" fill="var(--mut)" text-anchor="middle">No ${noun} data (run a live build).</text>`;
    grid.innerHTML = ''; return;
  }
  const items = [...pool].sort((a,b)=> b.pct - a.pct);
  const dates = S.dates;
  sub.textContent = `${items.length} ${noun}s · reconstructed dollar-DIX · ${dates.length.toLocaleString()} days · ${dates[0]} → ${dates[dates.length-1]}`;

  // horizontal ranking bars: x = latest DIX percentile within its own trailing year
  const W=900,H=300,padL=150,padR=96,padT=8,padB=26,n=items.length;
  const rowH=(H-padT-padB)/n;
  const xp = p => padL + (W-padL-padR)*(p/100);
  let g='';
  for(const p of [0,25,50,75,100]){
    g += `<line x1="${xp(p).toFixed(1)}" y1="${padT}" x2="${xp(p).toFixed(1)}" y2="${H-padB}" stroke="var(--grid)" stroke-width="1" stroke-dasharray="2,3"/>`
       + `<text x="${xp(p).toFixed(1)}" y="${H-padB+13}" font-size="9.5" fill="var(--mut)" text-anchor="middle">${p}</text>`;
  }
  g += `<text x="${((padL+W-padR)/2).toFixed(1)}" y="${H-3}" font-size="9.5" fill="var(--mut)" text-anchor="middle">1-year percentile of DIX &middot; higher = dark accumulation elevated vs own history</text>`;
  items.forEach((it,i)=>{
    const y=padT+i*rowH+rowH/2, acc=it.pct>=50, col=acc?'var(--pos)':'var(--neg)';
    const bw=Math.max(0.5, xp(it.pct)-padL);
    g += `<rect x="${padL}" y="${(y-rowH*0.28).toFixed(1)}" width="${bw.toFixed(1)}" height="${(rowH*0.56).toFixed(1)}" fill="${col}" opacity="0.78" rx="2"/>`
       + `<text x="${(padL-8).toFixed(1)}" y="${(y+3.5).toFixed(1)}" font-size="11" fill="var(--ink)" text-anchor="end"><tspan font-weight="700">${it.etf}</tspan> <tspan fill="var(--mut)">${it.name}</tspan></text>`
       + `<text x="${(xp(it.pct)+6).toFixed(1)}" y="${(y+3.5).toFixed(1)}" font-size="10" fill="var(--mut)"><tspan fill="${col}" font-weight="700">${it.pct}%</tspan> · ${it.cur.toFixed(3)}</text>`;
  });
  rank.innerHTML = g;

  // small-multiple DIX sparklines, shared y-range for cross-sector comparability. The trailing
  // percentile band lines are included in the extent so they never clip outside the sparkline.
  let lo=Infinity,hi=-Infinity;
  for(const it of items){
    for(const v of it.series){ if(v==null)continue; if(v<lo)lo=v; if(v>hi)hi=v; }
    for(const arr of [it.p80, it.p20]){ if(!arr)continue; for(const v of arr){ if(v==null)continue; if(v<lo)lo=v; if(v>hi)hi=v; } }
  }
  if(lo===Infinity){ lo=0.35; hi=0.55; }
  const pad=(hi-lo)*0.08||0.02; lo-=pad; hi+=pad;
  grid.innerHTML = items.map(it=>{
    const arrow = it.d20==null ? '' : (it.d20>=0
      ? ` <span style="color:var(--pos)">▲${it.d20.toFixed(3)}</span>`
      : ` <span style="color:var(--neg)">▼${Math.abs(it.d20).toFixed(3)}</span>`);
    const lcr = it.last_cross;
    const cross = lcr ? ` · <span style="color:${lcr.dir==='up'?'var(--pos)':'var(--neg)'}" title="most recent crossing into the trailing-year 80th/20th percentile band">${lcr.dir==='up'?'▲P80':'▼P20'} ${lcr.date}</span>` : '';
    return `<div class="cell" data-etf="${it.etf}" title="click for constituent dark flow">
      <div class="chead">
        <span class="tkr">${it.etf} <span style="color:var(--mut);font-weight:500;font-size:10px">${it.name}</span></span>
        <span class="val">${it.cur.toFixed(3)}</span>
      </div>
      ${spark(it.series, lo, hi, 'raw', 204, 64, 4, {p80: it.p80, p20: it.p20, cross: it.cross})}
      <div class="meta"><span>${it.pct}%ile 1y${arrow}${cross}</span><span>${it.n} names</span></div>
    </div>`;
  }).join('');
}

// Sector drill-down: click a sector panel to see which constituents are receiving the dark flow.
const sectorOverlay = document.getElementById('sectorOverlay');
function closeSectorModal(){ sectorOverlay.classList.remove('on'); }
document.getElementById('secmClose').addEventListener('click', closeSectorModal);
sectorOverlay.addEventListener('click', e=>{ if(e.target === sectorOverlay) closeSectorModal(); });
// Escape closes the constituent performance plane FIRST (it sits on top); a second Escape
// then closes the sector modal underneath -- so nested modals unwind one level at a time.
document.addEventListener('keydown', e=>{
  if(e.key === 'Escape' && !overlay.classList.contains('on')) closeSectorModal();
});
document.getElementById('sectorGrid').addEventListener('click', e=>{
  const cell = e.target.closest('.cell'); if(cell && cell.dataset.etf) openSectorModal(cell.dataset.etf);
});
document.getElementById('secmBars').addEventListener('click', e=>{
  const row = e.target.closest('.crow'); if(row && row.dataset.t) openSectorConstituent(row.dataset.t);
});
document.getElementById('sectorLevelSeg').addEventListener('click', e=>{
  const b = e.target.closest('button'); if(!b || b.dataset.lvl === sectorLevel) return;
  sectorLevel = b.dataset.lvl;
  [...e.currentTarget.children].forEach(x => x.classList.toggle('on', x === b));
  renderSectors();
});

function openSectorModal(etf){
  const S = P.sectors; if(!S) return;
  const it = S.items.find(x=>x.etf===etf); if(!it || !it.names || !it.names.length) return;
  document.getElementById('secmTkr').textContent =
    `${it.etf} · ${it.name}` + (it.parent ? ` (${it.parent})` : '');
  document.getElementById('secmVal').textContent = `DIX ${it.cur.toFixed(3)} · ${it.pct}%ile 1y`;
  document.getElementById('secmSub').innerHTML =
    `<span>who is receiving the dark flow — constituents by share of the sector's off-exchange short $ volume</span>`
    + `<span>top ${it.names.length} of ${it.n} names</span>`;
  const names = it.names, rowH=20, padT=8, padL=66, padR=168, W=900;
  const maxW = Math.max(...names.map(n=>n.w), 0.01);
  const H = padT*2 + names.length*rowH;
  const svg = document.getElementById('secmBars');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const xw = w => (W-padL-padR) * (w/maxW);
  let g='';
  names.forEach((nm,i)=>{
    const y = padT + i*rowH + rowH/2;
    const acc = nm.dd > 0.003, dist = nm.dd < -0.003;
    const col = acc ? 'var(--pos)' : dist ? 'var(--neg)' : 'var(--mut)';
    const bw = Math.max(1, xw(nm.w));
    const arrow = acc ? `▲${nm.dd.toFixed(3)}` : dist ? `▼${Math.abs(nm.dd).toFixed(3)}` : `·${nm.dd.toFixed(3)}`;
    // Each row is a clickable group -> opens the name's forward-return-by-decile view. A
    // transparent full-width rect makes the whole row a hit target (see .crow CSS + handler).
    g += `<g class="crow" data-t="${nm.t}" style="cursor:pointer">`
       + `<rect class="hit" x="0" y="${(y-rowH/2).toFixed(1)}" width="${W}" height="${rowH}" fill="transparent"/>`
       + `<text x="${(padL-8).toFixed(1)}" y="${(y+3.5).toFixed(1)}" font-size="11" fill="var(--ink)" text-anchor="end" font-weight="600">${nm.t}</text>`
       + `<rect x="${padL}" y="${(y-rowH*0.34).toFixed(1)}" width="${bw.toFixed(1)}" height="${(rowH*0.68).toFixed(1)}" fill="${col}" opacity="0.8" rx="2"/>`
       + `<text x="${(padL+bw+6).toFixed(1)}" y="${(y+3.5).toFixed(1)}" font-size="10" fill="var(--mut)">${nm.w.toFixed(1)}% · D ${nm.d.toFixed(2)} <tspan fill="${col}">${arrow}</tspan></text>`
       + `</g>`;
  });
  svg.innerHTML = g;
  document.getElementById('secmNote').innerHTML =
    '<b>Click a name</b> for its forward return by dark-ratio decile (1 / 2 / 3-month), the same view as the small-multiples grid. '
    + 'Bar length = the name\'s share of the sector\'s off-exchange short <b>dollar</b> volume (the dark-accumulation flow, recent avg). '
    + 'Colour / arrow = 20-session change in that name\'s dark ratio D: <span style="color:var(--pos)">▲ accumulating</span>, '
    + '<span style="color:var(--neg)">▼ distributing</span>. D = current 5-day-MA dark ratio (share of the name\'s volume that trades dark).';
  sectorOverlay.classList.add('on');
}

// Sector constituent -> per-name decile view. Reuses the cell-detail modal (#overlay) to show
// the clicked stock's forward return by decile of its raw D (1/2/3-month) -- the same view the
// small-multiples grid gives -- sourced from P.sector_rel (packed for the displayed names).
function openSectorConstituent(tkr){
  const R = P.sector_rel || null;
  let key = tkr;
  if(R && !((R.d||{})[key]) && key === 'GOOG' && (R.d||{}).GOOGL) key = 'GOOGL';
  const dser = R ? (R.d||{})[key] : null;
  document.getElementById('mTkr').textContent = tkr;
  const mVal = document.getElementById('mVal');
  const todayD = dser ? lastNonNull(dser) : null;
  mVal.textContent = todayD == null ? '' : ('D ' + todayD.toFixed(3));
  mVal.className = 'val';
  const spEl = document.getElementById('mSpark'), relBox = document.getElementById('mRel');
  if(!dser){
    document.getElementById('mSub').innerHTML =
      `<span>no per-name dark-ratio history for ${tkr}</span><span></span>`;
    spEl.innerHTML = ''; relBox.style.display = 'none';
    overlay.classList.add('on'); return;
  }
  document.getElementById('mSub').innerHTML =
    `<span>forward return by decile of this name's raw dark ratio D</span><span></span>`;
  let mn=Infinity, mx=-Infinity;
  for(const v of dser){ if(v!=null){ if(v<mn)mn=v; if(v>mx)mx=v; } }
  if(mn===Infinity){ mn=0; mx=1; }
  const pad=(mx-mn)*0.1||0.02;
  spEl.innerHTML = spark(dser, Math.max(0,mn-pad), mx+pad, 'raw', 860, 140, 10);
  relBox.style.display = '';
  renderModalDeciles('21', key, R); renderModalDeciles('42', key, R); renderModalDeciles('63', key, R);
  overlay.classList.add('on');
}

// -------------------------------------------------------------------------
// Tab: SP500 decile table -- for every S&P 500 name, mean 1-month (21d) forward return by
// decile of its own dark ratio D, with the +/-1 SE whisker level (overlap-adjusted, n/21)
// shown under each mean -- the same bar+whisker the cell modal draws, as a numeric table.
// -------------------------------------------------------------------------
const SPXTBL_H = 21;
let spxTblRows = null;
function computeSpxTableRows(){
  const R = P.spx_rel, out = [];
  if(!R || !R.d) return out;
  for(const t of Object.keys(R.d).sort()){
    const ds = R.d[t], rs = (R.r21||{})[t];
    if(!ds || !rs) continue;
    const pts=[], n=Math.min(ds.length, rs.length);
    for(let i=0;i<n;i++){ const x=ds[i], y=rs[i]; if(x==null||y==null) continue; pts.push([x,y]); }
    if(pts.length < 30) continue;
    const bk = deciles(pts, 10), todayD = lastNonNull(ds);
    let curDec = -1;
    if(todayD!=null){
      for(let b=0;b<bk.length;b++){ if(bk[b] && todayD>=bk[b].dLo && todayD<=bk[b].dHi){ curDec=b; break; } }
      if(curDec<0) curDec = (bk[9] && todayD>bk[9].dHi) ? 9 : 0;
    }
    out.push({t, bk, curDec});
  }
  return out;
}
function renderSpxTable(){
  const sub = document.getElementById('spxtblSub'), body = document.getElementById('spxtblBody');
  if(!P.spx_rel || !P.spx_rel.d || !Object.keys(P.spx_rel.d).length){
    sub.textContent = 'S&P 500 per-name data unavailable (live builds only, with the S&P grid enabled)';
    body.innerHTML = ''; return;
  }
  if(!spxTblRows) spxTblRows = computeSpxTableRows();
  const f = (document.getElementById('spxtblFilter').value || '').toUpperCase().trim();
  const shown = f ? spxTblRows.filter(r=>r.t.includes(f)) : spxTblRows;
  sub.innerHTML = `${shown.length} of ${spxTblRows.length} S&P 500 names · mean 1-month (21d) forward return by decile of each name's dark ratio D · ±1 SE (whisker level) below each mean`;
  let h = '<table class="dtbl"><thead><tr><th>Ticker</th><th>now</th>';
  for(let b=1;b<=10;b++) h += `<th>D${b}</th>`;
  h += '</tr></thead><tbody>';
  for(const r of shown){
    h += `<tr><td class="tk">${r.t}</td><td class="now">${r.curDec>=0?('D'+(r.curDec+1)):'--'}</td>`;
    for(let b=0;b<10;b++){
      const bk = r.bk[b];
      if(!bk){ h += '<td class="na">·</td>'; continue; }
      const se = bk.rStd/Math.sqrt(Math.max(1, bk.n/SPXTBL_H));
      const cls = (bk.rMean>=0?'pos':'neg') + (b===r.curDec?' cur':'');
      h += `<td class="${cls}"><span class="m">${fmtPct(bk.rMean)}</span><span class="s">±${se.toFixed(1)}</span></td>`;
    }
    h += '</tr>';
  }
  body.innerHTML = h + '</tbody></table>';
}
document.getElementById('spxtblFilter').addEventListener('input', renderSpxTable);

// -------------------------------------------------------------------------
// Top-level tab switching
// -------------------------------------------------------------------------
let spxRendered = false, ndxRendered = false, iwmRendered = false, sectorsRendered = false;
// Unified "DIX vs Return" tab: a multi-toggle selects any combination of the
// NDX / SPX / IWM reconstructed-DIX views, each keeping its own wrap, controls
// and render function untouched.
function updateIdx(){
  const active = [...document.querySelectorAll('#idxSel button')]
    .filter(x => x.classList.contains('on')).map(x => x.dataset.i);
  [['ndx','ndxWrap','ctl-ndx'],['spx','spxWrap','ctl-spx'],['iwm','iwmWrap','ctl-iwm']].forEach(([i,w,c])=>{
    const on = active.includes(i);
    document.getElementById(w).style.display = on ? '' : 'none';
    document.getElementById(c).style.display = on ? '' : 'none';
    if(on){
      if(i==='ndx' && !ndxRendered){ renderNdx(); ndxRendered = true; }
      if(i==='spx' && !spxRendered){ renderSpx(); spxRendered = true; }
      if(i==='iwm' && !iwmRendered){ renderIwm(); iwmRendered = true; }
    }
  });
}
document.getElementById('idxSel').addEventListener('click', e=>{
  const b = e.target.closest('button'); if(!b) return;
  const on = document.querySelectorAll('#idxSel button.on');
  if(b.classList.contains('on') && on.length === 1) return;   // keep at least one
  b.classList.toggle('on');
  updateIdx();
});
document.getElementById('tabs').addEventListener('click', e=>{
  const b = e.target.closest('button'); if(!b) return;
  const t = b.dataset.t;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle('on', x===b));
  document.getElementById('ctl-grid').style.display = t==='grid' ? '' : 'none';
  document.getElementById('ctl-rel').style.display = t==='rel' ? '' : 'none';
  document.getElementById('ctl-idx').style.display = t==='idx' ? '' : 'none';
  document.getElementById('ctl-xs').style.display = t==='xs' ? '' : 'none';
  document.getElementById('ctl-ev').style.display = t==='ev' ? '' : 'none';
  document.getElementById('relStats').style.display = t==='rel' ? '' : 'none';
  document.getElementById('grid').style.display = t==='grid' ? '' : 'none';
  document.getElementById('relWrap').style.display = t==='rel' ? '' : 'none';
  // index wraps + their controls are governed by the idx multi-toggle; hide them off-tab
  if(t!=='idx'){ ['spxWrap','ndxWrap','iwmWrap','ctl-spx','ctl-ndx','ctl-iwm']
    .forEach(id=>document.getElementById(id).style.display='none'); }
  document.getElementById('xsWrap').style.display = t==='xs' ? '' : 'none';
  document.getElementById('evWrap').style.display = t==='ev' ? '' : 'none';
  document.getElementById('sectorsWrap').style.display = t==='sectors' ? '' : 'none';
  document.getElementById('spxtblWrap').style.display = t==='spxtbl' ? '' : 'none';
  document.getElementById('footHint').textContent = t==='rel'
    ? 'whiskers = ±1 SE on the overlap-adjusted (effective-N) mean; r CI via block bootstrap; overlapping daily returns → n far exceeds independent obs'
    : t==='idx'
    ? 'reconstructed dollar-weighted DIX per index (NDX-100 / S&P 500 / Russell 2000) vs that index\'s own forward return -- toggle any combination of indices to compare'
    : t==='xs'
    ? 'cross-sectional rank of every name each day → deciles → forward excess return; long-short D10-D1 is market-neutral'
    : t==='ev'
    ? 'streak events: N days in a row inside a decile band → one event on the Nth day → forward return vs an always-invested baseline'
    : t==='sectors'
    ? 'reconstructed dollar-DIX per sector ETF, ranked by 1-year percentile of dark accumulation. Constituents from SPDR Select Sector funds + iShares SOXX'
    : t==='spxtbl'
    ? 'per-S&P-500-name mean 1-month forward return by decile of its own dark ratio D, with the ±1 SE whisker level; the outlined cell is the decile D sits in today'
    : (sortMode==='weight' ? 'ordered by NDX index weight'
       : mode==='raw' ? 'ordered by highest raw D'
       : 'ordered by latest divergence (accumulating, then distributing)') + ' · toggle Weight/Divergence to reorder · hover a panel for date/value';
  if(t==='rel') renderRel();
  if(t==='idx') updateIdx();
  if(t==='xs') renderXs();
  if(t==='ev') renderEvents();
  if(t==='sectors' && !sectorsRendered){ renderSectors(); sectorsRendered = true; }
  if(t==='spxtbl') renderSpxTable();
});

// -------------------------------------------------------------------------
// Cross-highlight: hovering a decile bar highlights the scatter dots that fed
// it (and dims the rest). Wired once on the persistent <svg> containers via
// event delegation, since their contents are replaced wholesale on every
// re-render -- listeners on the container survive that; listeners on
// individual bars/dots wouldn't.
// -------------------------------------------------------------------------
function wireDecileHover(barsId, scatterId){
  const barsEl = document.getElementById(barsId);
  const scatterEl = document.getElementById(scatterId);
  if(!barsEl || !scatterEl) return;
  barsEl.addEventListener('mouseover', e=>{
    const bar = e.target.closest('.bar');
    if(!bar) return;
    const dec = bar.dataset.dec;
    scatterEl.querySelectorAll('.dot').forEach(dot=>{
      const match = dot.dataset.dec === dec;
      dot.classList.toggle('hi', match);
      dot.classList.toggle('dim', !match);
    });
  });
  barsEl.addEventListener('mouseout', e=>{
    if(!e.target.closest('.bar')) return;
    scatterEl.querySelectorAll('.dot').forEach(dot=>{ dot.classList.remove('hi', 'dim'); });
  });
}
[['barsSvg','scatterSvg'],
 ['sBars21','sScatter21'], ['sBars42','sScatter42'], ['sBars63','sScatter63'],
 ['nBars21','nScatter21'], ['nBars42','nScatter42'], ['nBars63','nScatter63'],
 ['wBars21','wScatter21'], ['wBars42','wScatter42'], ['wBars63','wScatter63']].forEach(([b, s]) => wireDecileHover(b, s));
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# ==========================================================================
# FINRA + Yahoo data layer -- replaces the SqueezeMetrics API entirely.
#   dark signal : FINRA off-exchange short/total volumes -> per-name DPI (= D)
#   prices      : Yahoo Finance daily raw close (for as-traded dollar weighting),
#                 adjusted close (split-safe forward returns) and volume. Free, no key.
# ==========================================================================
YAHOO_CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
               "?period1={p1}&period2={p2}&interval=1d")
_YF_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
YAHOO_CACHE = "yahoo_prices.pkl"
ISHARES_DOC_TMPL = ("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/"
                    "api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES"
                    "&targetSite=us-ishares&locale=en_US&portfolioId={pid}"
                    "&component=fundDownload&userType=individual")
IVV_PORTFOLIO_ID = "239726"   # iShares Core S&P 500 ETF
IWM_PORTFOLIO_ID = "239710"   # iShares Russell 2000 ETF
SOXX_PORTFOLIO_ID = "239705"  # iShares Semiconductor ETF
IGV_PORTFOLIO_ID = "239771"   # iShares Expanded Tech-Software Sector ETF
# iShares sector funds resolve holdings by BlackRock portfolioId (SSGA funds fetch by ticker).
ISHARES_SECTOR_PIDS = {"SOXX": SOXX_PORTFOLIO_ID, "IGV": IGV_PORTFOLIO_ID}

# SSGA SPDR Select Sector daily-holdings .xlsx (State Street); {tkr} is the lowercase ETF symbol
SSGA_HOLDINGS_TMPL = ("https://www.ssga.com/us/en/intermediary/library-content/products/"
                      "fund-data/etfs/us/holdings-daily-us-en-{tkr}.xlsx")
# Sector universe for the reconstructed sector-DIX tab: (ETF, display name, holdings source).
# The eight SPDR Select Sector funds come from SSGA; SOXX (semiconductors) is iShares.
SECTOR_ETFS = [
    ("XLK", "Technology", "ssga"), ("XLF", "Financials", "ssga"),
    ("XLV", "Health Care", "ssga"), ("XLI", "Industrials", "ssga"),
    ("XLY", "Cons. Discretionary", "ssga"), ("XLP", "Cons. Staples", "ssga"),
    ("XLE", "Energy", "ssga"), ("XLU", "Utilities", "ssga"),
    ("SOXX", "Semiconductors", "ishares"), ("XBI", "Biotech", "ssga"),
    ("XLB", "Materials", "ssga"), ("XLC", "Comm. Services", "ssga"),
    ("IGV", "Software", "ishares"),
]

# SPDR S&P *industry* funds -- one granularity finer than the broad Select-Sector funds
# above, and fetched from the identical SSGA daily-holdings source, so they plug straight
# into the same reconstructed-DIX path. Shown under a Sectors/Subsectors toggle in the
# Sector DIX tab. `parent` is the broad SECTOR_ETFS display name each rolls up to (used only
# for grouping/labeling). Curated to span the main sectors without duplicating the specialty
# funds already in SECTOR_ETFS (SOXX semis, XBI biotech).
SUBSECTOR_ETFS = [
    ("XHB", "Homebuilders",        "ssga", "Cons. Discretionary"),
    ("XRT", "Retail",              "ssga", "Cons. Discretionary"),
    ("XSW", "Software & IT Svcs",  "ssga", "Technology"),
    ("XTN", "Transportation",      "ssga", "Industrials"),
    ("XAR", "Aerospace & Defense", "ssga", "Industrials"),
    ("XME", "Metals & Mining",     "ssga", "Materials"),
    ("XOP", "Oil & Gas E&P",       "ssga", "Energy"),
    ("XES", "Oil & Gas Equip.",    "ssga", "Energy"),
    ("XPH", "Pharmaceuticals",     "ssga", "Health Care"),
    ("KRE", "Regional Banks",      "ssga", "Financials"),
]


def to_yahoo_symbol(sym):
    """Holdings/FINRA ticker -> Yahoo convention (class shares use '-' not '.')."""
    return sym.strip().upper().replace(".", "-")


def _unix(ts):
    return int(pd.Timestamp(ts).timestamp())


def fetch_yahoo_one(sym, start, end, session=None, retries=3, pause=0.5):
    """(close, adjclose, volume) DataFrame indexed by date for one symbol, or empty."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    get = (session or requests).get
    # Yahoo's period2 is exclusive of bars at/after that instant, and `end` is a normalized
    # midnight -- so passing it verbatim drops the `end` day's own daily bar (the pipeline then
    # lags a full session behind, e.g. stuck on Friday all Monday evening even after the close).
    # Push period2 to the end of the `end` day so that session's bar is included.
    p2 = _unix(pd.Timestamp(end).normalize() + pd.Timedelta(days=1))
    url = YAHOO_CHART.format(sym=to_yahoo_symbol(sym), p1=_unix(start), p2=p2)
    last = None
    for a in range(retries):
        try:
            r = get(url, timeout=30, headers={"User-Agent": _YF_UA})
            if r.status_code == 429:
                last = "429"; time.sleep(pause * (a + 2) + 0.5); continue
            if r.status_code not in (200, 404):
                last = f"HTTP {r.status_code}"; time.sleep(pause * (a + 1)); continue
            j = r.json()
            res = j.get("chart", {}).get("result")
            if not res or j.get("chart", {}).get("error"):
                return pd.DataFrame(columns=["close", "adjclose", "volume"])
            res = res[0]
            ts = res.get("timestamp")
            if not ts:
                return pd.DataFrame(columns=["close", "adjclose", "volume"])
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            idx = pd.to_datetime(ts, unit="s").normalize()
            df = pd.DataFrame({"close": q.get("close"),
                               "adjclose": adj if adj is not None else q.get("close"),
                               "volume": q.get("volume")}, index=idx)
            return df[~df.index.duplicated(keep="last")].dropna(how="all")
        except Exception as e:  # noqa: BLE001
            last = str(e); time.sleep(pause * (a + 1))
    print(f"  ! yahoo {sym}: failed ({last})", file=sys.stderr)
    return pd.DataFrame(columns=["close", "adjclose", "volume"])


def load_yahoo_panels(symbols, start, end, workers=8, cache_dir=None, refresh=False,
                      label="symbol"):
    """{'close','adjclose','volume'} wide panels (dates x symbols) over [start, end].

    Incrementally cached: a symbol synced today is skipped; one behind is re-fetched only
    from its last cached date; new symbols fetch the full window. No-data symbols are
    remembered for the day so they aren't retried on same-day re-runs.
    """
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols))
    cache = (Path(cache_dir) / YAHOO_CACHE) if cache_dir else None
    cached = {}
    if cache is not None and cache.exists() and not refresh:
        try:
            cached = pd.read_pickle(cache)
        except Exception:  # noqa: BLE001
            cached = {}
    fields = ("close", "adjclose", "volume")
    base = {f: cached.get(f, pd.DataFrame()) for f in fields}
    end_n = pd.Timestamp(end).normalize()
    # The session we should already hold once the day's bars are published: today if a weekday,
    # else the prior business day. A same-day cache is trusted only when its freshest close
    # actually reaches that session -- otherwise a cache stamped "synced today" while still a
    # session behind (e.g. written before the close, or under the old exclusive-end bug) would
    # freeze the whole pipeline a day back, since Yahoo's calendar also drives the FINRA dates.
    target_session = end_n if end_n.weekday() < 5 else (end_n - pd.offsets.BDay(1)).normalize()
    base_latest = base["close"].dropna(how="all").index.max() if not base["close"].empty else None
    synced_today = (cached.get("_synced") == str(end_n.date()) and not refresh
                    and base_latest is not None and base_latest >= target_session)
    nodata = set(cached.get("_nodata", [])) if synced_today else set()

    def _is_current(sym):
        # Up to date when the symbol's freshest cached close reaches the target session (or it
        # produced no data today). Judged per-symbol against the target rather than a blanket
        # "synced today" flag: the Yahoo cache is shared across universes (NDX/SPX/IWM), so once
        # one universe advances it to today the flag flips True for all -- which must NOT cause
        # another universe's still-behind symbols to be skipped and served a session stale.
        if sym in nodata:
            return True
        c = base["close"]
        if sym not in c.columns:
            return False
        s = c[sym].dropna()
        return (not s.empty) and s.index.max() >= target_session

    def _fetch_start(sym):
        if _is_current(sym):
            return None
        c = base["close"]
        if refresh or sym not in c.columns:
            return pd.Timestamp(start)
        s = c[sym].dropna()
        return pd.Timestamp(start) if s.empty else s.index.max()

    def _window(panels):
        win = [d for d in panels["close"].index if pd.Timestamp(start) <= d <= end_n]
        return {f: panels[f].reindex(index=win, columns=symbols) for f in fields}

    if synced_today and all(_is_current(s) for s in symbols):
        print(f"Yahoo prices: reusing today's cache (all {len(symbols)} {label}s current); "
              f"pass --refresh to re-poll.", file=sys.stderr)
        return _window(base)

    todo = [(s, st) for s in symbols for st in [_fetch_start(s)] if st is not None]
    print(f"Yahoo prices: {len(symbols)-len(todo)}/{len(symbols)} {label}s current in cache; "
          f"fetching {len(todo)}...", file=sys.stderr)

    fetched = {}
    if todo:
        session = make_session(workers) if requests else None
        counter = {"n": 0}
        lock = threading.Lock()

        def _one(item):
            sym, st = item
            df = fetch_yahoo_one(sym, st, end, session=session)
            with lock:
                counter["n"] += 1
                if counter["n"] % 200 == 0 or counter["n"] == len(todo):
                    print(f"[{counter['n']:>4}/{len(todo)}] yahoo fetched", file=sys.stderr)
            return sym, df

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for sym, df in ex.map(_one, todo):
                if len(df):
                    fetched[sym] = df
                    nodata.discard(sym)
                elif sym not in base["close"].columns:
                    nodata.add(sym)

    out = {}
    for f in fields:
        new = pd.DataFrame({s: d[f] for s, d in fetched.items() if f in d})
        out[f] = (new.combine_first(base[f]) if not base[f].empty else new).sort_index()

    # Record a same-day sync ONLY when (nearly) every requested symbol actually reached the
    # freshest session now available. Otherwise a partial fetch -- e.g. Yahoo throttling most
    # of a 500-name pull, leaving the majority stuck at an older close -- would still stamp
    # `_synced` = today and make every later same-day run short-circuit on that stale panel
    # (the "partial cache" artifact). When too many names lag, leave `_synced` unset so the
    # next run re-polls instead of trusting the incomplete panel. A small tail (halted/late
    # names, ~5%) is tolerated so ordinary gaps don't force perpetual re-polling.
    sub = out["close"].reindex(columns=symbols)
    latest = sub.dropna(how="all").index.max() if not sub.empty else None
    if latest is not None:
        last_seen = sub.apply(lambda c: c.last_valid_index())
        behind = [s for s in symbols
                  if s not in nodata and (last_seen.get(s) is None or last_seen[s] < latest)]
    else:
        behind = list(symbols)
    tol = max(3, int(0.05 * len(symbols)))
    synced = str(end_n.date()) if len(behind) <= tol else ""
    if not synced:
        print(f"Yahoo prices: {len(behind)}/{len(symbols)} {label}s behind "
              f"{latest.date() if latest is not None else 'n/a'} -- NOT marking cache synced "
              f"(next run will re-poll; lower --workers if this persists).", file=sys.stderr)

    if cache is not None and not out["close"].empty:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".pkl.tmp")
            pd.to_pickle({**out, "_synced": synced, "_nodata": sorted(nodata)}, tmp)
            tmp.replace(cache)
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not write yahoo cache ({e})", file=sys.stderr)
    return _window(out)


def finra_dpi_to_d(short_panel, total_panel, smooth=5):
    """Per-name D = `smooth`-day MA of the off-exchange DPI (short / total), clipped 0..1.
    This is SqueezeMetrics' D construction, computed from FINRA directly."""
    dpi = (short_panel / total_panel.replace(0, np.nan)).clip(lower=0, upper=1)
    return dpi, dpi.rolling(smooth, min_periods=1).mean()


def fetch_ishares_holdings(portfolio_id, label="fund", session=None, retries=3, pause=0.5,
                           return_weights=False):
    """Equity constituent tickers for an iShares fund via BlackRock's fund-document API.
    With return_weights=True, returns (tickers, {ticker: index weight %}); the weight map
    is {} when the document has no parseable weight column (e.g. a CSV fallback body)."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    get = (session or requests).get
    url = ISHARES_DOC_TMPL.format(pid=portfolio_id)
    last_err = None
    for attempt in range(retries):
        try:
            r = get(url, timeout=60, headers={"User-Agent": _YF_UA})
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"; time.sleep(pause * (attempt + 1)); continue
            head = r.text.lstrip()[:600].lower()
            if head.startswith("<!doctype") or "<html" in head:
                last_err = "endpoint served HTML (bot/consent gate)"; break
            tickers = _iwm_tickers_from_spreadsheetml(r.text) or _iwm_tickers_from_csv_text(r.text)
            if tickers:
                print(f"iShares {label} holdings: {len(tickers)} equity constituents", file=sys.stderr)
                if return_weights:
                    return tickers, _ishares_weights_from_spreadsheetml(r.text)
                return tickers
            last_err = "no holdings table found"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(pause * (attempt + 1))
    print(f"  ! {label} holdings fetch failed ({last_err}).", file=sys.stderr)
    return ([], {}) if return_weights else []


def _ssga_tickers_from_xlsx(raw):
    """Constituent tickers from an SSGA Select-Sector holdings .xlsx: a few preamble rows
    (Fund Name / Ticker Symbol / Holdings date), then a 'Name, Ticker, Identifier, SEDOL,
    Weight, ...' table, then a disclaimer trailer. Returns [] if unreadable (e.g. no openpyxl)."""
    try:
        df0 = pd.read_excel(io.BytesIO(raw), header=None, engine="openpyxl")
    except Exception:  # noqa: BLE001  (missing openpyxl, corrupt book, HTML gate, ...)
        return []
    hdr = None
    for i in range(len(df0)):
        row_vals = [str(v).strip().lower() for v in df0.iloc[i].tolist()]
        if "ticker" in row_vals:
            hdr = i
            break
    if hdr is None:
        return []
    try:
        df = pd.read_excel(io.BytesIO(raw), header=hdr, engine="openpyxl")
    except Exception:  # noqa: BLE001
        return []
    df.columns = [str(c).strip() for c in df.columns]
    tcol = next((c for c in df.columns if c.lower() == "ticker"), None)
    if tcol is None:
        return []
    col = df[tcol]
    if isinstance(col, pd.DataFrame):  # duplicate "Ticker" columns -- take the first
        col = col.iloc[:, 0]
    out, seen = [], set()
    for t in col.tolist():
        u = str(t).strip().upper()
        if _TICKER_RE.match(u) and u not in seen:
            seen.add(u); out.append(u)
    return out


def fetch_ssga_holdings(etf, label=None, session=None, retries=5, pause=1.5):
    """Constituent tickers for a State Street SPDR Select Sector ETF via SSGA's daily
    holdings .xlsx. Returns [] on HTTP error, a non-xlsx body (bot/consent gate), or an
    unparseable book.

    SSGA rate-limits / bot-gates rapid back-to-back requests: the first hit of a run
    tends to succeed, then follow-ups get served an HTML consent page in place of the
    workbook. So callers should pass a shared session (to carry any consent cookie
    forward across sectors), requests send browser-like headers, and -- crucially -- a
    consent gate is retried with exponential backoff rather than abandoning the sector
    for the whole build. Giving up on the first gate is what left the Sector DIX tab
    with only the one sector fetched before the gate kicked in."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    label = label or etf
    get = (session or requests).get
    url = SSGA_HOLDINGS_TMPL.format(tkr=etf.lower())
    headers = {
        "User-Agent": _YF_UA,
        "Accept": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
                   "application/vnd.ms-excel,application/octet-stream,*/*"),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.ssga.com/us/en/intermediary/etfs",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = get(url, timeout=60, headers=headers)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(20.0, pause * (2 ** attempt))); continue
            if r.content[:2] != b"PK":                      # .xlsx is a zip; anything else is a gate
                # The consent/bot gate is rate-based and clears on its own; back off
                # (exponentially) and retry rather than dropping the sector for the run.
                last_err = "endpoint served non-xlsx (bot/consent gate)"
                time.sleep(min(20.0, pause * (2 ** attempt))); continue
            tickers = _ssga_tickers_from_xlsx(r.content)
            if tickers:
                print(f"SSGA {label} holdings: {len(tickers)} constituents", file=sys.stderr)
                return tickers
            last_err = "no holdings table found (openpyxl missing?)"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(min(20.0, pause * (2 ** attempt)))
    print(f"  ! {label} holdings fetch failed ({last_err}).", file=sys.stderr)
    return []


def build_universe_panels(symbols, start, end, workers=8, cache_dir=None, ns="", refresh=False,
                          label="symbol"):
    """FINRA (short/total off-exchange) + Yahoo (raw close, adj close, volume) for `symbols`.

    Yahoo's trading-day calendar drives the FINRA date set, so both align to real sessions.
    Returns dict of wide panels incl. per-name 1-day DPI ('dpi') and 5-day-MA D ('d').
    """
    ypan = load_yahoo_panels(symbols, start, end, workers=workers, cache_dir=cache_dir,
                             refresh=refresh, label=label)
    dates = [pd.Timestamp(d) for d in ypan["close"].index]
    short, total = fetch_finra_dark_volume_panel(dates, symbols, workers=workers,
                                                 cache_dir=cache_dir, ns=ns)
    dpi, d = finra_dpi_to_d(short, total)
    return {"short": short, "total": total, "dpi": dpi, "d": d,
            "close": ypan["close"], "adjclose": ypan["adjclose"], "volume": ypan["volume"]}


def build_reconstructed_index_payload(dix_series, etf_close, out_key="d", start=None,
                                      reconstructed=True, n_constituents=None):
    """Pack a reconstructed index DIX (x-axis) vs an ETF's forward returns (from its close).
    `out_key` is 'd' for the IWM tab, 'dix' for the SPX tab."""
    if dix_series is None or etf_close is None or dix_series.dropna().empty:
        return None
    df = pd.DataFrame({"D": dix_series, "CLOSE": etf_close}).sort_index().dropna(subset=["D"])
    if df.empty or df["CLOSE"].notna().sum() == 0:
        return None
    payload = build_index_payload(df, "D", out_key, start=start)
    if payload is not None:
        payload["reconstructed"] = reconstructed
        if n_constituents:
            payload["constituents"] = int(n_constituents)
    return payload


def build_breadth_payload(etf_px, keep):
    """Cumulative performance of RSP / IWM / SPY *relative to QQQ* (mega-cap), rebased to 100
    over `keep`, plus the (RSP+IWM)/2 'average-stock vs mega-cap' breadth gauge. Below 100 =
    underperforming the mega-caps = narrowing breadth (RSP and IWM sink together here)."""
    if etf_px is None or etf_px.empty or "QQQ" not in etf_px.columns:
        return None
    want = [c for c in ["RSP", "IWM", "SPY", "QQQ"] if c in etf_px.columns]
    b = etf_px[want].reindex(keep)
    b = b[b["QQQ"].notna()]
    have = [c for c in want if b[c].notna().sum() > 5]
    if "QQQ" not in have or len(b) < 5:
        return None
    b = b[have].dropna(how="any")
    if len(b) < 5:
        return None
    norm = b.div(b.iloc[0])                            # each rebased to 1 at the window start
    rel = norm.div(norm["QQQ"], axis=0) * 100.0        # cumulative return relative to QQQ, x100
    series = {c: [round(float(x), 2) for x in rel[c].values] for c in have}
    if "RSP" in rel and "IWM" in rel:
        series["AVG"] = [round(float(x), 2) for x in ((rel["RSP"] + rel["IWM"]) / 2).values]
    return {"dates": [d.strftime("%Y-%m-%d") for d in b.index], "series": series,
            "range": [b.index[0].strftime("%Y-%m-%d"), b.index[-1].strftime("%Y-%m-%d")]}


def build_sector_payload(members, short, total, close, d, keep, hist_win=252, min_names=8,
                         top_names=30):
    """One reconstructed dollar-DIX per sector over `keep`, plus the latest level, its percentile
    within the trailing `hist_win` sessions (dark accumulation vs its own recent history), its
    20-session change, the trailing-year 80th/20th percentile band (`p80`/`p20`) with the dates it
    crossed into the top/bottom of that band (`cross`, `last_cross`), and a per-constituent
    breakdown of which names are receiving the dark flow.
    `members` is [(etf, name, [tickers]), ...]; short/total/close/`d` are the shared union panels
    (`d` = per-name 5d-MA dark ratio). Same Sum($ short)/Sum($ off-exch) DIX construction as the
    SPX/IWM tabs, computed over each sector's own constituents."""
    if not members:
        return None
    items = []
    for etf, name, syms, *rest in members:
        # rest = (level, parent) for the current 5-tuple members; default to a broad sector
        # so older 3-tuple callers still work.
        level = rest[0] if len(rest) > 0 else "sector"
        parent = rest[1] if len(rest) > 1 else None
        cols = [c for c in syms if c in short.columns]
        if len(cols) < min_names:
            continue
        dix = compute_dollar_dix(short[cols], total[cols], close[cols],
                                 min_names=min_names).rolling(5, min_periods=1).mean()
        full = dix.dropna()
        if full.empty:
            continue
        cur = float(full.iloc[-1])
        hist = full.iloc[-hist_win:]
        pct = round(float((hist < cur).mean()) * 100)
        d20 = round(float(cur - full.iloc[-21]), 4) if len(full) > 21 else None
        series = [round(float(x), 4) if pd.notna(x) else None for x in dix.reindex(keep).values]
        if all(v is None for v in series):
            continue

        # Rolling trailing-year 80th/20th percentile band. Crossing ABOVE the 80th flags
        # the sector's dark accumulation entering the top of its own 1-year range; crossing
        # BELOW the 20th flags the bottom. Computed on the full history so the band is a true
        # trailing-`hist_win` percentile even at the left edge of the (shorter) plot window.
        roll = full.rolling(hist_win, min_periods=min(hist_win, 63))
        hi_band = roll.quantile(0.80)
        lo_band = roll.quantile(0.20)
        hi_ok, lo_ok = hi_band.notna(), lo_band.notna()
        above = (full >= hi_band) & hi_ok
        below = (full <= lo_band) & lo_ok
        # A crossing needs the band to have been defined the day BEFORE too, so the first day
        # the trailing window fills doesn't read as a phantom crossing (the prior "not above"
        # would just be an undefined band, not a genuine transition).
        cross_up = above & ~above.shift(1, fill_value=False) & hi_ok.shift(1, fill_value=False)
        cross_dn = below & ~below.shift(1, fill_value=False) & lo_ok.shift(1, fill_value=False)
        events = cross_up | cross_dn
        keep_pos = {dt: k for k, dt in enumerate(keep)}
        crosses = [{"i": keep_pos[dt], "dir": "up" if bool(cross_up[dt]) else "dn"}
                   for dt in full.index[events.values] if dt in keep_pos]
        ev_idx = full.index[events.values]
        last_cross = ({"date": ev_idx[-1].strftime("%Y-%m-%d"),
                       "dir": "up" if bool(cross_up[ev_idx[-1]]) else "dn"}
                      if len(ev_idx) else None)

        def _band(s):
            return [round(float(x), 4) if pd.notna(x) else None for x in s.reindex(keep).values]
        p80, p20 = _band(hi_band), _band(lo_band)
        # per-constituent dark flow: each name's share of the sector's off-exchange short DOLLAR
        # volume (recent mean, to dodge single-day FINRA reporting sparsity) = "who is receiving
        # the flow", plus its own dark ratio D and 20-session change (the direction of that flow).
        dark_dollar = (close[cols] * short[cols]).tail(25).mean()
        tot = float(dark_dollar.sum())
        names = []
        for c in cols:
            s = d[c].dropna()
            if s.empty or c not in dark_dollar.index or pd.isna(dark_dollar[c]) or not tot:
                continue
            dcur = float(s.iloc[-1])
            dchg = float(dcur - s.iloc[-21]) if len(s) > 21 else 0.0
            names.append({"t": c, "d": round(dcur, 3), "dd": round(dchg, 3),
                          "w": round(100.0 * float(dark_dollar[c]) / tot, 2)})
        names.sort(key=lambda x: x["w"], reverse=True)
        items.append({"etf": etf, "name": name, "n": len(cols), "cur": round(cur, 4),
                      "pct": pct, "d20": d20, "series": series, "names": names[:top_names],
                      "p80": p80, "p20": p20, "cross": crosses, "last_cross": last_cross,
                      "level": level, "parent": parent})
    if not items:
        return None
    return {"dates": [dt.strftime("%Y-%m-%d") for dt in keep], "items": items}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ndx_dark_residual.html", help="output HTML path")
    ap.add_argument("--window", type=int, default=126,
                    help="rolling window (trading days) for regression residual")
    ap.add_argument("--min-periods", type=int, default=40)
    ap.add_argument("--smooth", type=int, default=0,
                    help="optional MA (days) applied to residuals")
    ap.add_argument("--plot-days", type=int, default=378,
                    help="how many recent trading days to plot per panel (ignored "
                         "when --plot-start is set)")
    ap.add_argument("--plot-start", default="2020-01-01",
                    help="start date (YYYY-MM-DD) for the Small multiples plot window "
                         "-- takes precedence over --plot-days (empty string to fall "
                         "back to --plot-days' trailing window instead)")
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic data (no network needed)")
    ap.add_argument("--dump-csv", default="",
                    help="optional path to also write the residual panels as CSV")
    ap.add_argument("--dark-start", default="2018-08-01",
                    help="earliest date (YYYY-MM-DD) for the FINRA dark data, which is also "
                         "the history/residual start. FINRA's consolidated off-exchange files "
                         "begin 2018-08-01, so that is the practical floor.")
    ap.add_argument("--workers", type=int, default=10,
                    help="concurrent HTTP workers for the FINRA + Yahoo fetches")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                    help="cache directory for the FINRA off-exchange documents and the Yahoo "
                         f"price cache (default: {DEFAULT_CACHE_DIR}; empty string to disable)")
    ap.add_argument("--refresh", action="store_true", default=False,
                    help="ignore the same-day Yahoo price cache and re-poll prices")
    ap.add_argument("--no-spx", dest="spx", action="store_false", default=True,
                    help="skip the 'SPX D vs Return' tab (skips the ~500 S&P 500 / IVV "
                         "constituent fetch)")
    ap.add_argument("--iwm-holdings", dest="iwm_holdings", default="",
                    help="OPTIONAL: local IWM holdings file (SpreadsheetML/Excel-XML, CSV or a "
                         "plain ticker list) for the Russell 2000 universe. Auto-fetched from "
                         "BlackRock when omitted.")
    args = ap.parse_args()

    cache_dir = args.cache_dir or None
    plot_start = pd.Timestamp(args.plot_start) if args.plot_start else None
    end = pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.dark_start) if args.dark_start else pd.Timestamp("2018-08-01")
    ndx_syms = [BENCH] + [t for t in NDX100 if t != BENCH]

    if args.demo:
        print("DEMO mode: synthetic data (no network)...", file=sys.stderr)
        data = demo_panel(NDX100, BENCH, start=args.plot_start or "2020-01-01")
        panel, r21_panel, r42_panel, r63_panel = data["d"], data["r21"], data["r42"], data["r63"]
        close_panel = data["close"]
        raw_dark_panel, ndx_agg, ndx_dix = data["raw_dark"], data["ndx_agg"], data["ndx_dix"]
        sdix, sclose = demo_russell_dix(start=args.plot_start or "2020-01-01", seed=11)
        spx_payload = (build_reconstructed_index_payload(sdix, sclose, out_key="dix",
                       start=plot_start, n_constituents=500) if args.spx else None)
        idix, iclose = demo_russell_dix(start=args.plot_start or "2020-01-01", seed=29)
        iwm_payload = build_reconstructed_index_payload(idix, iclose, out_key="d",
                                                        start=plot_start, n_constituents=1900)
        spx_res = None   # S&P 500 grid is a live-data feature (empty in --demo)
        spx_rel = None
        spx_weight_map = {}
        spx_weight_order = None
        breadth_px = None
        sector_data = None   # sector-DIX tab is a live-data feature (empty in --demo)
    else:
        # ---- NDX-100: per-name D (FINRA DPI, 5d-MA) + prices (Yahoo) ----
        print(f"Building NDX-100 panel from FINRA + Yahoo ({start.date()} -> {end.date()})...",
              file=sys.stderr)
        NDX = build_universe_panels(ndx_syms, start, end, workers=args.workers,
                                    cache_dir=cache_dir, ns="", refresh=args.refresh, label="NDX")
        panel = NDX["d"]
        close_panel, raw_dark_panel = NDX["close"], NDX["dpi"]
        r21_panel = compute_forward_return(NDX["adjclose"], 21)
        r42_panel = compute_forward_return(NDX["adjclose"], 42)
        r63_panel = compute_forward_return(NDX["adjclose"], 63)
        ndx_dix = compute_dollar_dix(NDX["short"], NDX["total"], NDX["close"], exclude=(BENCH,))
        # FINRA off-exchange volume is as-traded; Yahoo's volume is split-adjusted. Rescale
        # the pre-split FINRA total onto the adjusted basis so the aggregate dark ratio stays
        # continuous across splits. (The dollar-DIX above needs no such fix: it pairs RAW
        # close with as-traded volume, which is already true dollars -- adjusting there would
        # double-count. Split adjustment belongs only where FINRA meets Yahoo's adj. volume.)
        ndx_agg = compute_aggregate_dark_ratio(
            split_adjust_finra_panel(NDX["total"], NDX["volume"]), NDX["volume"], exclude=(BENCH,))

        # ---- SPX tab + S&P 500 small-multiples: dollar-DIX (IVV constituents) ----
        spx_payload = None
        spx_res = None
        spx_rel = None
        spx_weight_map = {}
        spx_weight_order = None
        if args.spx:
            print("Building SPX DIX from IVV (S&P 500) constituents...", file=sys.stderr)
            sp_syms, spx_weight_map = fetch_ishares_holdings(
                IVV_PORTFOLIO_ID, label="IVV S&P 500", return_weights=True)
            spx_weight_order = sorted(spx_weight_map, key=spx_weight_map.get, reverse=True)
            if sp_syms:
                SP = build_universe_panels(sp_syms, start, end, workers=args.workers,
                                           cache_dir=cache_dir, ns="sp500", refresh=args.refresh,
                                           label="S&P 500")
                spx_dix = compute_dollar_dix(SP["short"], SP["total"], SP["close"])
                spy = load_yahoo_panels(["SPY"], start, end, workers=2, cache_dir=cache_dir,
                                        refresh=args.refresh, label="SPY")
                n_sp = int((SP["short"].notna() & SP["total"].notna()).any().sum())
                spx_payload = build_reconstructed_index_payload(
                    spx_dix, spy["adjclose"].get("SPY"), out_key="dix",
                    start=plot_start, n_constituents=n_sp)
                # residualize each S&P name's D against the S&P 500 DIX (5d MA) for its grid
                spx_bench = spx_dix.rolling(5, min_periods=1).mean().reindex(SP["d"].index)
                spx_res = compute_residuals(SP["d"], "SPX-DIX", window=args.window,
                                            min_periods=args.min_periods, bench_series=spx_bench)
                # per-name raw-D + forward returns for the S&P cell-modal decile bars
                spx_rel = pack_name_rel(SP["dpi"], SP["adjclose"], plot_start=plot_start)
                print(f"S&P 500 grid: residualized {spx_res['reg'].notna().any().sum()} names "
                      f"vs the S&P 500 DIX", file=sys.stderr)

        # ---- IWM tab: Russell 2000 dollar-DIX (IWM constituents) vs IWM forward return ----
        print("Building IWM DIX from Russell 2000 (IWM) constituents...", file=sys.stderr)
        iwm_syms = (load_iwm_holdings(args.iwm_holdings) if args.iwm_holdings
                    else fetch_ishares_holdings(IWM_PORTFOLIO_ID, label="IWM Russell 2000"))
        iwm_payload = None
        if iwm_syms:
            RU = build_universe_panels(iwm_syms, start, end, workers=args.workers,
                                       cache_dir=cache_dir, ns="russell", refresh=args.refresh,
                                       label="Russell")
            iwm_dix = compute_dollar_dix(RU["short"], RU["total"], RU["close"], exclude=("IWM",))
            iwmp = load_yahoo_panels(["IWM"], start, end, workers=2, cache_dir=cache_dir,
                                     refresh=args.refresh, label="IWM")
            n_ru = int((RU["short"].notna() & RU["total"].notna()).any().sum())
            iwm_payload = build_reconstructed_index_payload(
                iwm_dix, iwmp["adjclose"].get("IWM"), out_key="d",
                start=plot_start, n_constituents=n_ru)
        if iwm_payload is None:
            print("  ! IWM reconstruction unavailable (holdings / FINRA / price); "
                  "IWM tab will be empty.", file=sys.stderr)

        # ---- Breadth panel: RSP (equal-weight) + IWM (small-cap) vs QQQ (mega-cap) ----
        breadth_px = load_yahoo_panels(["RSP", "IWM", "SPY", "QQQ"], start, end, workers=4,
                                       cache_dir=cache_dir, refresh=args.refresh,
                                       label="breadth ETF")["adjclose"]

        # ---- Sector DIX: reconstructed dollar-DIX per SPDR / iShares sector ETF ----
        sector_data = None
        print("Building sector DIX (SPDR + iShares sector constituents)...", file=sys.stderr)
        hcache_path = (Path(cache_dir) / "holdings_cache.json") if cache_dir else None
        hcache = {}
        if hcache_path is not None and hcache_path.exists():
            try:
                hcache = json.loads(hcache_path.read_text())
            except Exception:  # noqa: BLE001
                hcache = {}
        today_str = str(end.date())
        sec_members = []
        # Broad Select-Sector funds plus the finer SPDR industry ("subsector") funds, tagged
        # with their granularity level (and, for subsectors, the parent sector they roll up to)
        # so the Sector DIX tab can switch between the two views. Both come from the same SSGA
        # holdings source and share the reconstructed-DIX path below.
        sector_funds = [(etf, name, src, "sector", None) for etf, name, src in SECTOR_ETFS] + \
                       [(etf, name, src, "subsector", parent)
                        for etf, name, src, parent in SUBSECTOR_ETFS]
        # One session across all sectors so any SSGA consent cookie earned on an early
        # fetch carries forward and helps later sectors clear the bot gate.
        sec_session = requests.Session() if requests is not None else None
        for i, (etf, sec_name, source, level, parent) in enumerate(sector_funds):
            key = f"sector:{etf}"
            cached_entry = hcache.get(key)
            if cached_entry and cached_entry.get("date") == today_str and not args.refresh:
                syms = cached_entry["tickers"]
            else:
                if i > 0:
                    time.sleep(3.0)  # stagger requests -- SSGA bot-gates fast back-to-back hits
                syms = (fetch_ssga_holdings(etf, label=etf, session=sec_session)
                        if source == "ssga"
                        else fetch_ishares_holdings(ISHARES_SECTOR_PIDS[etf], label=etf))
                if syms:
                    hcache[key] = {"date": today_str, "tickers": syms}
                elif cached_entry:
                    print(f"  ! {etf} fetch failed; reusing holdings cached "
                          f"{cached_entry['date']}", file=sys.stderr)
                    syms = cached_entry["tickers"]
            if syms:
                sec_members.append((etf, sec_name, syms, level, parent))
        if hcache_path is not None:
            try:
                hcache_path.parent.mkdir(parents=True, exist_ok=True)
                hcache_path.write_text(json.dumps(hcache))
            except Exception as e:  # noqa: BLE001
                print(f"  ! could not write holdings cache ({e})", file=sys.stderr)
        if sec_members:
            sec_union = sorted({s for _, _, syms, *_ in sec_members for s in syms})
            SEC = build_universe_panels(sec_union, start, end, workers=args.workers,
                                        cache_dir=cache_dir, ns="sector", refresh=args.refresh,
                                        label="sector")
            sector_data = {"members": sec_members, "short": SEC["short"],
                           "total": SEC["total"], "close": SEC["close"], "d": SEC["d"],
                           "dpi": SEC["dpi"], "adjclose": SEC["adjclose"]}

    if BENCH not in panel.columns or panel.shape[1] < 3:
        sys.exit(f"Insufficient data (got {panel.shape[1]} names incl. bench).")
    print(f"Panel: {panel.shape[0]} dates x {panel.shape[1]} names "
          f"({panel.index.min().date()} -> {panel.index.max().date()})", file=sys.stderr)

    # Residual-grid benchmark: the reconstructed NDX-100 dollar-DIX (5-day MA), on the same
    # DPI scale as the per-name D it is subtracted from.
    bench_series = None
    bench_label = BENCH
    if ndx_dix is not None and ndx_dix.notna().any():
        bench_series = ndx_dix.rolling(5, min_periods=1).mean().reindex(panel.index)
        bench_label = "NDX-DIX"
        print(f"Residual benchmark: reconstructed NDX-100 dollar-DIX (5d MA) over "
              f"{int(bench_series.notna().sum())} days "
              f"[{bench_series.dropna().index.min().date()} -> "
              f"{bench_series.dropna().index.max().date()}]", file=sys.stderr)

    res = compute_residuals(panel, BENCH, window=args.window,
                            min_periods=args.min_periods, smooth=args.smooth,
                            bench_series=bench_series)

    if spx_payload:
        print(f"SPX DIX: {len(spx_payload['dates'])} dates "
              f"({spx_payload['range'][0]} -> {spx_payload['range'][1]})", file=sys.stderr)
    if iwm_payload:
        print(f"IWM DIX: {len(iwm_payload['dates'])} dates "
              f"({iwm_payload['range'][0]} -> {iwm_payload['range'][1]})", file=sys.stderr)

    if args.dump_csv:
        res["reg"].to_csv(args.dump_csv.replace(".csv", "_regresid.csv"))
        res["diff"].to_csv(args.dump_csv.replace(".csv", "_diff.csv"))
        print(f"Wrote residual CSVs alongside {args.dump_csv}", file=sys.stderr)

    html = build_html(res, BENCH, r21_panel, r42_panel, r63_panel, close_panel,
                       raw_dark_panel, ndx_agg=ndx_agg, ndx_dix=ndx_dix, spx=spx_payload,
                       iwm=iwm_payload, bench_label=bench_label, spx_res=spx_res, spx_rel=spx_rel,
                       spx_weight_map=spx_weight_map, spx_weight_order=spx_weight_order,
                       breadth_px=breadth_px, sector_data=sector_data,
                       plot_days=args.plot_days, plot_start=plot_start, window=args.window,
                       demo=args.demo)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.out} ({len(html)/1024:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
