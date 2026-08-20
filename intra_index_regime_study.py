#!/usr/bin/env python3
"""
Intra-index comovement regimes x DIX -> 1-month forward return study.
=====================================================================

Question (per request): an index behaves differently depending on the regime
INSIDE it. At times all constituents rally together and correlation is high;
in lower-correlation periods only a certain group rallies while the rest
sells off. Divide the data into such regimes within each index -- NDX-100,
S&P 500 and Russell 2000 / IWM -- and dig into how each index's DIX reads
inside each one:

    1. Does the index's own dark-flow gauge (DIX5) predict differently in
       high- vs low-comovement regimes?
    2. In the dispersed (low-correlation) tape, does PER-NAME dark flow
       identify which group rallies and which sells off?

Regime construction (per day t, trailing windows only -- no look-ahead in
the signal)
-----------------------------------------------------------------------
From per-name daily prices we build daily % returns and three comovement
gauges over a trailing 21-session window:

    AVG_CORR  -- equal-weight average pairwise correlation of daily returns
                 across all names with a full window (the "do the stocks
                 move together" gauge; the same quantity Cboe's implied-
                 correlation indices proxy, here realized and index-internal).
    DISP21    -- 21-session mean of the daily cross-sectional std of returns
                 (how far apart the names land on a given day).
    BREADTH   -- fraction of names with a positive trailing 21-session
                 return (participation).

AVG_CORR is the primary regime axis, split Low/Mid/High on a 30/40/30 basis
(matching the comovement study's Low/Mid/High convention). Two bases are
reported: full-sample cutoffs (mild look-ahead) and EXPANDING cutoffs (each
day ranked only against its own past, min 250 obs -- what a live trader
could have known). The DIX side is each index's dollar-DIX smoothed to a
5-day MA (DIX5), zoned Low/Mid/High the same two ways.

Per-index inputs
----------------
    NDX  -- everything from the dashboard payload: per-name split-adjusted
            closes (`P.rel.close`, the NDX-100 grid names), raw dark ratios
            (`P.rel.d`), the NDX dollar-DIX and QQQ forward returns. The
            packed closes are validated per name against the payload's own
            adjusted r21 (names below --min-valid-corr are dropped;
            dividends leave the rest ~0.995+).
    SPX  -- the payload carries only index-level series for the S&P 500
            reconstruction, so the comovement gauges come from a fetched
            basket: the top --basket-size IVV names by weight, daily
            adjusted closes via Yahoo (same incremental cache as the other
            builds). DIX/outcome: `P.spx.dix` and `P.spx.r21` (SPY).
    IWM  -- same construction over the top --basket-size iShares IWM names;
            DIX/outcome from `P.iwm.d` / `P.iwm.r21`. NOTE: 100 of ~2000
            names is a behavioral proxy for the small-cap tape, not a
            replication of the index's weight (coverage is printed).

Per-name dark-flow tilt (for the "which group rallies" test): each name's
5-day-MA raw dark ratio minus its own expanding mean (min 60 obs) -- the
dashboard's "name-specific vs own average" signal. Each day the names are
split at the 20th/80th tilt percentiles and the equal-weight Q5-minus-Q1
forward-return spread is recorded. Daily panels exist only for NDX; for the
S&P 500 the payload's `spx_rel` block (501 names, weekly-sampled to the
`spx_grid` dates) supports a WEEKLY variant with the raw single-day print
in place of the 5d MA (min 12 weekly obs); IWM has no per-name panel.

Beyond the regime tables the study reports, per index:
  * an INTERACTION regression -- r1m on zDIX, zCORR, zDIX*zCORR with
    Newey-West (HAC, 21-lag) errors, with and without a realized-vol
    control (AVG_CORR and index vol are highly correlated; the control
    asks which one carries the signal);
  * a TAPE taxonomy -- trailing 21d index return sign x breadth tercile
    ("broad rally", "narrow rally", "broad selloff", ...) with each tape's
    forward return and DIX split;
  * a REGIME-ENTRY event study (first day the comovement regime forms,
    21-session cool-down) on the expanding (live-knowable) basis;
  * BLOCK-BOOTSTRAP 95% CIs (21-day moving blocks) on regime means;
  * an OUT-OF-SAMPLE split (cutoffs and loadings fitted < 2024, evaluated
    on 2024+);
and across indices:
  * the pairwise correlation of the three AVG_CORR gauges, the share of
    days the three comovement regimes agree, and each index's forward
    return by how many of the three sit in LowCorr;
  * an optional cross-check of NDX AVG_CORR against the GEX/dispersion
    barometer's realized top-50 SPX correlation and Cboe's COR1M implied
    correlation (pass --barometer docs/gex_dispersion.html).

Data source
-----------
The payload is read from the built dashboard (`docs/index.html`), plain
`const P = {...}` or compressed `const PZ = "<base64 deflate>"`. SPX/IWM
constituent lists come from iShares' fund documents (cached to JSON so an
offline re-run keeps working) and their prices from Yahoo via
`load_yahoo_panels` (incrementally cached). `--indices ndx` runs fully
offline against the payload alone.

Usage
-----
    python intra_index_regime_study.py                        # all three
    python intra_index_regime_study.py --indices ndx          # offline
    python intra_index_regime_study.py --csv intra_index_regimes.csv
    python intra_index_regime_study.py --barometer docs/gex_dispersion.html
"""
import argparse
import base64
import json
import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 21            # trailing window for the comovement gauges (= 1 month)
MIN_NAMES = 30         # min names with a full window before a gauge is defined
TILT_MIN_NAMES = 50    # min names before the daily tilt quantile split is scored
TILT_MIN_OBS = 60      # min history before a name's expanding tilt mean is used
TILT_MIN_OBS_WEEKLY = 12   # same guard in weekly rows (the spx_rel cadence)
BOOT_B = 2000          # bootstrap replications
BOOT_L = 21            # moving-block length (= the forward-return horizon)
EXP_MIN = 250          # min history before an expanding-window zone is defined
ROLL_WIN = 504         # rolling-percentile window for the zone bases (~2 years)
GATE_DAYS = 42         # print gate: min scored days before a conditional mean prints
GATE_EPISODES = 5      # print gate: min distinct episodes behind the mean
GATE_EVENTS = 10       # print gate: min entries before an event-study mean prints
OOS_SPLIT = "2024-01-01"
BASKET_N = 100         # default basket size for the fetched SPX/IWM universes
FETCH_START = "2019-10-01"   # SPX/IWM DIX starts 2020-01; buffer for the 21d warm-up
ZONES = ("LowCorr", "MidCorr", "HighCorr")
DZONES = ("DIXLow", "DIXMid", "DIXHigh")
VZONES = ("VolLow", "VolMid", "VolHigh")
TAIL_COMPLETE = 0.95   # min share of basket names with a close before the
                       # frame's trailing rows count as a settled session
IDX_PROXY = {"NDX": "QQQ", "SPX": "SPY", "IWM": "IWM"}


def load_payload(html_path):
    """Pull the dashboard payload out of a built HTML. Handles both the plain
    `const P = {...};` blob and the compressed `const PZ = "...";` form
    (base64-encoded zlib/deflate of the same JSON)."""
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    m = re.search(r"const P = (\{.*?\});", html, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r'const PZ = "([A-Za-z0-9+/=]+)";', html)
    if m:
        return json.loads(zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8"))
    sys.exit(f"Could not find the embedded payload in {html_path}")


# ----------------------------------------------------------------------------
# Small stats helpers (no scipy): normal sf, Newey-West OLS, block bootstrap
# ----------------------------------------------------------------------------
def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ols_nw(y, X, lags=21):
    """OLS with Newey-West (Bartlett) HAC standard errors.
    y: (n,), X: (n,k) including the constant. Returns (beta, se, t, p, r2)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t = np.where(se > 0, beta / se, np.nan)
    p = np.array([2 * _norm_sf(abs(tt)) if np.isfinite(tt) else np.nan for tt in t])
    ss_res = float(u @ u)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, p, r2


def block_boot_ci(r, B=BOOT_B, L=BOOT_L, seed=0, levels=(2.5, 97.5)):
    """Moving-block bootstrap 95% CI for the mean of `r` (a 1-D array of the
    regime's forward returns in date order). Blocks of L consecutive
    observations respect the ~21-day overlap autocorrelation."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    # below ~2 blocks the cyclic resample degenerates -> report no CI
    if n < 2 * L:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    means = r[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, levels))


def run_ids(mask):
    """Integer id per contiguous True run of `mask` (positional contiguity:
    consecutive rows of the frame), -1 on False rows. The cluster unit for
    episode-level inference."""
    m = np.asarray(mask, dtype=bool)
    ids = np.full(len(m), -1, dtype=int)
    cur, prev = -1, False
    for i, v in enumerate(m):
        if v and not prev:
            cur += 1
        if v:
            ids[i] = cur
        prev = v
    return ids


def cluster_boot_ci(r, ids, B=BOOT_B, seed=0, levels=(2.5, 97.5),
                    min_clusters=GATE_EPISODES):
    """Episode-cluster bootstrap 95% CI for the mean of `r`: resample whole
    contiguous episodes (cluster ids from `run_ids`) with replacement.
    Regime days arrive in a handful of multi-week episodes, so this is the
    honest uncertainty; the 21-day moving-block CI understates it whenever
    episodes run longer than a month. NaN below `min_clusters` episodes."""
    r = np.asarray(r, dtype=float)
    ids = np.asarray(ids)
    ok = np.isfinite(r) & (ids >= 0)
    r, ids = r[ok], ids[ok]
    uniq = np.unique(ids)
    if len(uniq) < min_clusters:
        return (np.nan, np.nan)
    groups = [r[ids == u] for u in uniq]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(B, len(groups)))
    means = np.array([np.concatenate([groups[j] for j in row]).mean()
                      for row in draws])
    return tuple(float(x) for x in np.percentile(means, levels))


def expanding_pctile(s, min_obs=EXP_MIN):
    """Percentile (0..1, mid-rank on ties) of each value within the series'
    OWN history up to and including that day. NaN until `min_obs`
    observations have accrued -- the cutoffs a live user could have known."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(min_obs - 1, len(v)):
        hist = v[: i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_obs or not np.isfinite(v[i]):
            continue
        out[i] = (hist < v[i]).mean() + 0.5 * (hist == v[i]).mean()
    return pd.Series(out, index=s.index)


def rolling_pctile(s, window=ROLL_WIN, min_obs=EXP_MIN):
    """Percentile (0..1, mid-rank on ties) of each value within its own
    TRAILING `window` sessions only. Unlike the expanding variant this
    re-normalizes as old history rolls off, so a non-stationary series (the
    dollar-DIX drifted 0.41 -> 0.49 -> 0.43 across 2018-2026) cannot turn
    zone membership into a calendar label. NaN until `min_obs` observations."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(min_obs - 1, len(v)):
        hist = v[max(0, i - window + 1): i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_obs or not np.isfinite(v[i]):
            continue
        out[i] = (hist < v[i]).mean() + 0.5 * (hist == v[i]).mean()
    return pd.Series(out, index=s.index)


def zones_30_40_30(s, basis="full", labels=("Low", "Mid", "High"), min_obs=EXP_MIN):
    """Low/Mid/High on a 30/40/30 split of `s` (the comovement study's zone
    convention). basis='full' uses full-sample 30th/70th pct cutoffs (mild
    look-ahead); basis='expanding' ranks each day against its own past only
    (NaN -> 'NA' until min_obs); basis='rolling' ranks against the trailing
    ROLL_WIN sessions only (live-knowable AND drift-proof -- the headline
    basis)."""
    if basis in ("expanding", "rolling"):
        pct = (expanding_pctile if basis == "expanding" else rolling_pctile)(
            s, min_obs=min_obs)
        out = np.where(pct.isna(), "NA",
                       np.where(pct <= 0.30, labels[0],
                                np.where(pct >= 0.70, labels[2], labels[1])))
        return pd.Series(out, index=s.index)
    lo, hi = s.quantile([0.30, 0.70])
    out = np.where(s.isna(), "NA",
                   np.where(s <= lo, labels[0],
                            np.where(s >= hi, labels[2], labels[1])))
    return pd.Series(out, index=s.index)


# ----------------------------------------------------------------------------
# Panels and comovement gauges
# ----------------------------------------------------------------------------
def panel_from(rel, key, dates):
    """One `rel` sub-dict ({name: [values]}) as a float DataFrame on `dates`."""
    return pd.DataFrame(
        {t: pd.array([np.nan if v is None else float(v) for v in s], dtype="float64")
         for t, s in rel[key].items()}, index=dates)


def series_from(values, dates):
    return pd.Series([np.nan if v is None else float(v) for v in values],
                     index=dates, dtype="float64")


def daily_returns(close):
    """Per-name daily % returns. A return needs BOTH endpoints present, so a
    gap in the close series never manufactures a multi-day 'daily' move."""
    ret = close.pct_change() * 100.0
    ret[close.isna() | close.shift().isna()] = np.nan
    return ret


def validate_names(close, r21, min_corr=0.98, horizon=WINDOW):
    """Compare each name's compounded `horizon`-session close return against
    the payload's adjusted forward return. Returns (ok_names, dropped) where
    dropped is {name: corr}. Guards against any name whose packed closes are
    NOT on the same (split-adjusted) basis as the return panel."""
    comp = (close.shift(-horizon) / close - 1.0) * 100.0
    ok, dropped = [], {}
    for c in close.columns:
        if c not in r21.columns:
            dropped[c] = np.nan
            continue
        d = pd.concat([comp[c], r21[c]], axis=1).dropna()
        if len(d) < 100:          # too short to validate -> keep (recent IPOs)
            ok.append(c)
            continue
        corr = float(d.corr().iloc[0, 1])
        if corr >= min_corr:
            ok.append(c)
        else:
            dropped[c] = round(corr, 4)
    return ok, dropped


def avg_pairwise_corr(ret, window=WINDOW, min_names=MIN_NAMES):
    """Equal-weight average pairwise correlation of daily returns over a
    trailing `window`, per day. Only names with a complete window (and
    non-degenerate variance) enter; NaN when fewer than `min_names` qualify."""
    R = ret.to_numpy(dtype=float)
    out = np.full(len(ret), np.nan)
    for i in range(window - 1, len(ret)):
        block = R[i - window + 1: i + 1]
        full = ~np.isnan(block).any(axis=0)
        if full.sum() < min_names:
            continue
        b = block[:, full]
        sd = b.std(axis=0, ddof=0)
        live = sd > 1e-9
        n = int(live.sum())
        if n < min_names:
            continue
        C = np.corrcoef(b[:, live].T)
        out[i] = (C.sum() - n) / (n * (n - 1))
    return pd.Series(out, index=ret.index)


def cross_sectional_dispersion(ret, window=WINDOW, min_names=MIN_NAMES):
    """21-session mean of the daily cross-sectional std of returns (%).
    Days with fewer than `min_names` live names contribute NaN."""
    counts = ret.notna().sum(axis=1)
    daily = ret.std(axis=1, ddof=0).where(counts >= min_names)
    return daily.rolling(window, min_periods=max(2, window // 2)).mean()


def breadth_positive(close, window=WINDOW, min_names=MIN_NAMES):
    """Fraction of names with a positive trailing `window`-session return."""
    tr = close / close.shift(window) - 1.0
    n = tr.notna().sum(axis=1)
    return (tr > 0).sum(axis=1).where(n >= min_names) / n


def tilt_spread(dpan, r21, names, q=(0.20, 0.80), min_names=TILT_MIN_NAMES,
                min_obs=TILT_MIN_OBS, ma_window=5):
    """Per-row equal-weight Q5-minus-Q1 forward-return spread on the per-name
    dark-flow TILT (`ma_window`-row MA of the raw dark ratio minus the name's
    own expanding mean, min `min_obs` rows). Rows are usually days;
    ma_window=1 gives the raw-print variant for weekly-sampled panels.
    Returns a DataFrame with q5/q1/spread (%, 21-session forward)."""
    d5 = dpan[names].rolling(ma_window, min_periods=max(1, ma_window - 2)).mean()
    tilt = d5 - dpan[names].expanding(min_periods=min_obs).mean()
    T = tilt.to_numpy(dtype=float)
    F = r21[names].to_numpy(dtype=float)
    q5 = np.full(len(dpan), np.nan)
    q1 = np.full(len(dpan), np.nan)
    for i in range(len(dpan)):
        ok = np.isfinite(T[i]) & np.isfinite(F[i])
        if ok.sum() < min_names:
            continue
        t, f = T[i][ok], F[i][ok]
        lo, hi = np.quantile(t, q)
        q1[i] = f[t <= lo].mean()
        q5[i] = f[t >= hi].mean()
    out = pd.DataFrame({"q5": q5, "q1": q1}, index=dpan.index)
    out["spread"] = out["q5"] - out["q1"]
    return out


def tape_label(tr_index, breadth, blo, bhi):
    """Six-way tape taxonomy from the trailing index return sign x breadth
    zone: 'broad rally' (up, breadth >= bhi), 'narrow rally' (up, <= blo),
    'mid rally', and the mirrored selloffs ('broad selloff' = down + LOW
    breadth, i.e. most names falling; 'selective selloff' = down + HIGH
    breadth: the index dragged down while most names still hold up)."""
    if not np.isfinite(tr_index) or not np.isfinite(breadth):
        return "NA"
    if tr_index > 0:
        if breadth >= bhi:
            return "broad rally"
        if breadth <= blo:
            return "narrow rally"
        return "mid rally"
    if breadth <= blo:
        return "broad selloff"
    if breadth >= bhi:
        return "selective selloff"
    return "mid selloff"


# ----------------------------------------------------------------------------
# Frame construction
# ----------------------------------------------------------------------------
def assemble_frame(px, proxy_close, dix, r1m):
    """Per-day frame of the comovement gauges (from the per-name price panel
    `px`), DIX5, tape label and the index's forward return. `proxy_close` is
    the index proxy's own price series (QQQ/SPY/IWM) for the vol control and
    the trailing-return leg of the tape."""
    # latest-day completeness guard: drop TRAILING rows where fewer than
    # TAIL_COMPLETE of the basket names have a close -- a partial/unsettled
    # last session must not drive the gauges' "current" reads
    comp = px.notna().mean(axis=1)
    live = np.where(comp.to_numpy() >= TAIL_COMPLETE)[0]
    if len(live) and live[-1] < len(px) - 1:
        px = px.iloc[: live[-1] + 1]

    ret = daily_returns(px)
    M = pd.DataFrame(index=px.index)
    M["avg_corr"] = avg_pairwise_corr(ret)
    M["disp21"] = cross_sectional_dispersion(ret)
    M["breadth"] = breadth_positive(px)
    M["dix5"] = dix.reindex(px.index).rolling(5, min_periods=3).mean()
    M["r1m"] = r1m.reindex(px.index)
    pret = daily_returns(proxy_close.to_frame("p"))["p"]
    M["rv"] = pret.rolling(WINDOW).std(ddof=0) * np.sqrt(252)
    M["tr21"] = (proxy_close / proxy_close.shift(WINDOW) - 1.0) * 100.0

    M = M[M["avg_corr"].notna() & M["dix5"].notna()].copy()

    # zones on all three bases (rolling = headline: live-knowable and
    # drift-proof; expanding kept for comparison; full-sample = descriptive)
    for col, prefix, labels in (("avg_corr", "c", ZONES), ("dix5", "d", DZONES)):
        M[prefix + "z_full"] = zones_30_40_30(M[col], "full", labels)
        M[prefix + "z_exp"] = zones_30_40_30(M[col], "expanding", labels)
        M[prefix + "z_roll"] = zones_30_40_30(M[col], "rolling", labels)
    # realized-vol zones (the competing conditioning variable; corr and vol
    # are ~0.8 correlated, so every corr claim owes the reader this parallel)
    M["vz_roll"] = zones_30_40_30(M["rv"], "rolling", VZONES)
    # DIX zone on a one-session-lagged signal: FINRA publishes day t's file
    # after the close, so dz_roll_l1 is what a live trader could act on at
    # the close of day t (the conservative timing convention)
    M["dz_roll_l1"] = zones_30_40_30(M["dix5"].shift(1), "rolling", DZONES)

    # era-adjusted forward return (diagnostic ONLY -- demeaning uses the full
    # calendar year, i.e. within-year future information; it answers "was this
    # cell anything beyond its era?", it is not a live signal)
    M["r1m_ex"] = M["r1m"] - M["r1m"].groupby(M.index.year).transform("mean")

    # z-scores for the regressions (full-sample; the OOS section refits its own)
    for col, z in (("dix5", "zDIX"), ("avg_corr", "zCORR"), ("rv", "zRV")):
        M[z] = (M[col] - M[col].mean()) / M[col].std(ddof=0)

    # tape taxonomy (breadth terciles are full-sample cutoffs; descriptive)
    blo, bhi = M["breadth"].quantile([1 / 3, 2 / 3])
    M["tape"] = [tape_label(t, b, blo, bhi) for t, b in zip(M["tr21"], M["breadth"])]
    if "tilt_spread" not in M.columns:
        M[["tilt_q5", "tilt_q1", "tilt_spread"]] = np.nan
    return M


def build_ndx_frame(P, min_valid_corr=0.98):
    """NDX frame straight from the payload (plus the daily tilt spread).
    Returns (frame, meta-dict)."""
    rel = P["rel"]
    bench = P["bench"]
    dates = pd.to_datetime(rel["dates"])
    close = panel_from(rel, "close", dates)
    dpan = panel_from(rel, "d", dates)
    r21 = panel_from(rel, "r21", dates)
    dix = series_from(rel["ndx_dix"], dates)

    all_names = [c for c in close.columns if c != bench]
    names, dropped = validate_names(close[all_names], r21, min_corr=min_valid_corr)
    M = assemble_frame(close[names], close[bench], dix, r21[bench])
    sp = tilt_spread(dpan, r21, [n for n in names if n in dpan.columns])
    M[["tilt_q5", "tilt_q1", "tilt_spread"]] = sp[["q5", "q1", "spread"]].reindex(M.index)
    meta = {"names": len(names), "dropped": dropped, "proxy": bench,
            "note": f"{len(names)} payload grid names"}
    return M, meta


def load_basket_prices(index_key, basket_n, cache_dir, refresh=False):
    """(per-name adjclose panel, proxy adjclose series, note) for SPX/IWM:
    top `basket_n` iShares holdings by weight (holdings cached to JSON so an
    offline re-run keeps working), prices via the shared Yahoo cache."""
    import ndx_dark_residual as N
    pid, fund = ((N.IVV_PORTFOLIO_ID, "IVV S&P 500") if index_key == "SPX"
                 else (N.IWM_PORTFOLIO_ID, "IWM Russell 2000"))
    proxy = IDX_PROXY[index_key]
    wcache = (Path(cache_dir) / f"intra_regime_{index_key.lower()}_weights.json"
              if cache_dir else None)
    tickers, weights = [], {}
    try:
        tickers, weights = N.fetch_ishares_holdings(pid, label=fund, return_weights=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"  ! {fund} holdings fetch failed ({e})", file=sys.stderr)
    if tickers and weights:
        if wcache:
            wcache.parent.mkdir(parents=True, exist_ok=True)
            wcache.write_text(json.dumps({"tickers": tickers, "weights": weights}))
    elif wcache and wcache.exists():
        print(f"  ! using cached {fund} holdings", file=sys.stderr)
        d = json.loads(wcache.read_text())
        tickers, weights = d["tickers"], d["weights"]
    if not tickers or not weights:
        raise RuntimeError(f"no {fund} holdings/weights available (fetch failed, no cache)")

    w = pd.Series({t: weights[t] for t in tickers if weights.get(t)},
                  dtype="float64").sort_values(ascending=False).head(basket_n)
    coverage = float(w.sum())
    basket = [N.to_yahoo_symbol(t) for t in w.index]
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    panels = N.load_yahoo_panels(sorted(set(basket + [proxy])), FETCH_START, end,
                                 cache_dir=cache_dir or None, refresh=refresh,
                                 label=f"REGIME-{index_key}")
    adj = panels["adjclose"].dropna(how="all")
    have = [s for s in basket if s in adj.columns and adj[s].notna().sum() > WINDOW]
    note = (f"top {len(have)} {fund} names by weight "
            f"({coverage:.1f}% of the index pre-normalization)")
    return adj[have], adj[proxy], note


def build_basket_frame(P, index_key, basket_n, cache_dir, refresh=False):
    """SPX/IWM frame: fetched-basket gauges + payload DIX/forward return.
    Returns (frame, meta-dict). Raises RuntimeError when the basket cannot
    be built (offline with a cold cache)."""
    node = P["spx"] if index_key == "SPX" else P["iwm"]
    dates = pd.to_datetime(node["dates"])
    dix = series_from(node["dix" if index_key == "SPX" else "d"], dates)
    r1m = series_from(node["r21"], dates)
    px, proxy_close, note = load_basket_prices(index_key, basket_n, cache_dir, refresh)
    # restrict the price panel to the DIX's own span (assemble_frame trims the
    # rest); reindex DIX/outcome onto the price calendar
    px = px[px.index >= dates.min() - pd.Timedelta(days=60)]
    M = assemble_frame(px, proxy_close, dix, r1m)
    return M, {"names": px.shape[1], "dropped": {}, "proxy": IDX_PROXY[index_key],
               "note": note}


def spx_weekly_tilt(P):
    """Weekly-cadence tilt spread for the S&P 500 from the payload's
    `spx_rel` block (per-name raw dark ratio + adjusted r21, sampled to the
    `spx_grid` dates). Returns a spread DataFrame on those dates, or None
    when the payload lacks the block."""
    sr = P.get("spx_rel")
    sg = P.get("spx_grid")
    if not sr or not sg or not sg.get("dates"):
        return None
    dates = pd.to_datetime(sg["dates"])
    if any(len(v) != len(dates) for v in sr.get("d", {}).values()):
        return None
    dpan = panel_from(sr, "d", dates)
    r21 = panel_from(sr, "r21", dates)
    names = [c for c in dpan.columns if c in r21.columns]
    return tilt_spread(dpan, r21, names, min_names=TILT_MIN_NAMES,
                       min_obs=TILT_MIN_OBS_WEEKLY, ma_window=1)


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def mask_stats(M, mask, col="r1m", with_ci=True, seed=0):
    """Standard cell statistics for the rows of `M` where `mask` holds:
    day AND episode counts, mean/median/hit of `col`, the era-adjusted mean
    (for r1m), and the episode-cluster bootstrap CI. `gate` is the global
    print gate -- a conditional mean may only be quoted when the cell has
    >= GATE_DAYS scored days and >= GATE_EPISODES distinct episodes."""
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    ids_all = run_ids(m)
    vals = M[col].to_numpy(dtype=float)
    fin = m & np.isfinite(vals)
    n_days = int(fin.sum())
    st = {"n_days": n_days, "n_eps": int(len(np.unique(ids_all[fin])))}
    st["gate"] = n_days >= GATE_DAYS and st["n_eps"] >= GATE_EPISODES
    if n_days:
        r = vals[fin]
        st.update(mean=float(r.mean()), med=float(np.median(r)),
                  hit=float((r > 0).mean() * 100))
        if col == "r1m" and "r1m_ex" in M.columns:
            ex = M["r1m_ex"].to_numpy(dtype=float)[fin]
            st["ex"] = float(np.nanmean(ex)) if np.isfinite(ex).any() else np.nan
        st["ci"] = (cluster_boot_ci(vals[fin], ids_all[fin], seed=seed)
                    if with_ci and st["gate"] else (np.nan, np.nan))
    return st


def fmt_cell(st):
    """One-line rendering of a mask_stats cell; below the print gate only
    the counts appear (no mean -- Rule: never quote an ungated mean)."""
    n = f"n={st['n_days']}d/{st['n_eps']}ep"
    if not st["n_days"]:
        return f"--  ({n})"
    if not st["gate"]:
        return f"--  (below gate; {n})"
    s = f"{st['mean']:+.2f}% (med {st['med']:+.2f}, hit {st['hit']:3.0f}%)"
    if np.isfinite(st.get("ex", np.nan)):
        s += f"  ex {st['ex']:+.2f}"
    lo, hi = st.get("ci", (np.nan, np.nan))
    if np.isfinite(lo):
        s += f"  [epCI {lo:+.1f},{hi:+.1f}]"
    return s + f"  ({n})"


def year_mix(M, mask, col="r1m", top=3):
    """Top-`top` calendar-year shares of a cell's scored days, e.g.
    '23:44% 20:31% 25:12%' -- the direct diagnostic for era-confounded
    zones (a healthy cell is not dominated by one year)."""
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    fin = m & np.isfinite(M[col].to_numpy(dtype=float))
    if not fin.any():
        return ""
    yrs = pd.Series(M.index.year[fin]).value_counts(normalize=True)
    return " ".join(f"{y % 100:02d}:{100 * s:.0f}%" for y, s in yrs.head(top).items())


def per_year_line(M, mask, col="r1m", min_days=10):
    """Per-year means for a cell ('20:+3.2 21:+1.0 25:-0.4'; years with
    fewer than `min_days` scored days print '.')."""
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    v = M[col].where(pd.Series(m, index=M.index))
    parts = []
    for y, g in v.groupby(M.index.year):
        g = g.dropna()
        if len(g) == 0:
            continue
        parts.append(f"{y % 100:02d}:{g.mean():+.1f}" if len(g) >= min_days
                     else f"{y % 100:02d}:.")
    return " ".join(parts)


def loyo_range(M, mask, col="r1m"):
    """Leave-one-year-out range of a cell's mean -- how much any single
    calendar year moves the number."""
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    v = M[col].where(pd.Series(m, index=M.index)).dropna()
    if len(v) < GATE_DAYS:
        return (np.nan, np.nan)
    means = []
    for y in v.index.year.unique():
        rest = v[v.index.year != y]
        if len(rest) >= GATE_DAYS // 2:
            means.append(rest.mean())
    return (float(min(means)), float(max(means))) if means else (np.nan, np.nan)


def describe_regimes(M, zcol, proxy, with_ci=True):
    lines = [f"=== COMOVEMENT REGIMES ({zcol}; 30/40/30 split of AVG_CORR) ===",
             "  ex = era-adjusted mean (minus same-calendar-year baseline; "
             "diagnostic only)   epCI = episode-cluster bootstrap 95%"]
    seed = 0
    for z in ZONES:
        mask = M[zcol] == z
        sub = M[mask]
        if not len(sub):
            continue
        seed += 1
        st = mask_stats(M, mask, with_ci=with_ci, seed=seed)
        lines.append(
            f"  {z:9s} avg_corr {sub['avg_corr'].mean():.2f}  "
            f"disp {sub['disp21'].mean():.2f}%  breadth {sub['breadth'].mean():.2f}  "
            f"{proxy}vol {sub['rv'].mean():4.1f}  ->  {proxy} 1m " + fmt_cell(st))
    return "\n".join(lines)


def episodes(M, zcol, zone, top=8):
    """Longest contiguous runs of `zone` (calendar-contiguous rows of M)."""
    mask = (M[zcol] == zone).to_numpy()
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    runs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    return [(M.index[a].date(), M.index[b].date(), b - a + 1) for a, b in runs[:top]]


def episode_report(M, zcol):
    lines = [f"=== LONGEST EPISODES ({zcol}) ==="]
    for zone in ("HighCorr", "LowCorr"):
        runs = episodes(M, zcol, zone)
        lines.append(f"  {zone}: " + "; ".join(
            f"{a}->{b} ({n}d)" for a, b, n in runs[:6]))
    return "\n".join(lines)


def dix_by_corr_table(M, czone_col, dzone_col, with_ci=True):
    """corr-regime x DIX-regime table. Every row carries day AND episode
    counts, the era-adjusted mean, an episode-cluster CI (the honest one)
    plus the legacy 21d block CI, and the cell's top-3 year mix. Rows below
    the print gate keep their counts but blank the means."""
    rows = []
    seed = 100
    czones = VZONES if czone_col.startswith("vz") else ZONES
    for cz in czones:
        for dz in DZONES:
            mask = (M[czone_col] == cz) & (M[dzone_col] == dz)
            if not mask.any():
                continue
            seed += 1
            st = mask_stats(M, mask, with_ci=with_ci, seed=seed)
            r = M.loc[mask, "r1m"].dropna()
            blo, bhi = (block_boot_ci(r.to_numpy(), seed=seed)
                        if with_ci and st["gate"] else (np.nan, np.nan))
            sp = M.loc[mask, "tilt_spread"].dropna()
            gate = st["gate"]
            rows.append({
                "corr_regime": cz, "dix_regime": dz,
                "n_days": st["n_days"], "n_eps": st["n_eps"],
                "mean": round(st["mean"], 2) if gate else np.nan,
                "ex": round(st.get("ex", np.nan), 2) if gate else np.nan,
                "med": round(st["med"], 2) if gate else np.nan,
                "hit": round(st["hit"]) if gate else np.nan,
                "ep_ci_lo": round(st["ci"][0], 2) if np.isfinite(st["ci"][0]) else np.nan,
                "ep_ci_hi": round(st["ci"][1], 2) if np.isfinite(st["ci"][1]) else np.nan,
                "blk_ci_lo": round(blo, 2) if np.isfinite(blo) else np.nan,
                "blk_ci_hi": round(bhi, 2) if np.isfinite(bhi) else np.nan,
                "spread_mean": (round(float(sp.mean()), 2)
                                if len(sp) >= GATE_DAYS else np.nan),
                "years": year_mix(M, mask),
                "gate": "ok" if gate else "low-n",
            })
    return pd.DataFrame(rows)


def flagship_report(M, czone_col, dzone_col):
    """Era diagnostics for the two PRIMARY (pre-specified) cells: per-year
    means and the leave-one-year-out range. Everything else in this report
    is descriptive; these two are the study's registered claims."""
    lines = ["=== FLAGSHIP CELL DIAGNOSTICS (PRIMARY, pre-specified; all other "
             "cells are descriptive) ==="]
    cells = [
        ("LowCorr & DIXHigh -> r1m", (M[czone_col] == "LowCorr")
         & (M[dzone_col] == "DIXHigh"), "r1m"),
        ("LowCorr tilt spread", (M[czone_col] == "LowCorr")
         & M["tilt_spread"].notna(), "tilt_spread"),
    ]
    for label, mask, col in cells:
        st = mask_stats(M, mask, col=col, with_ci=True, seed=17)
        lo, hi = loyo_range(M, mask, col=col)
        lines.append(f"  {label:26s} " + fmt_cell(st))
        yl = per_year_line(M, mask, col=col)
        if yl:
            lines.append(f"    per-year: {yl}")
        if np.isfinite(lo):
            lines.append(f"    leave-one-year-out mean range: [{lo:+.2f}, {hi:+.2f}]")
    return "\n".join(lines)


def vol_parallel_report(M, proxy, with_ci=False):
    """The competing conditioning variable, run in parallel: (a) the DIX
    grid inside realized-VOL terciles instead of correlation terciles;
    (b) nested NW regressions -- does zCORR add anything over zRV?;
    (c) the DIX Low->High gradient double-sorted vol x corr (gate-limited)."""
    lines = [f"=== COMOVEMENT vs VOLATILITY ({proxy}; corr(zCORR,zRV)="
             f"{M['zCORR'].corr(M['zRV']):.2f}) ==="]
    Mb = M[(M["vz_roll"] != "NA") & (M["dz_roll"] != "NA")]
    tab = dix_by_corr_table(Mb, "vz_roll", "dz_roll", with_ci=with_ci)
    lines.append("  -- DIX grid inside realized-VOL terciles (rolling basis) --")
    lines.append(tab[["corr_regime", "dix_regime", "n_days", "n_eps", "mean",
                      "ex", "hit", "gate"]].rename(
        columns={"corr_regime": "vol_regime"}).to_string(index=False))
    # nested models
    d = M[["r1m", "zDIX", "zCORR", "zRV"]].dropna()
    y = d["r1m"].to_numpy()
    specs = [("r1m ~ zRV", ["zRV"]),
             ("r1m ~ zRV + zCORR", ["zRV", "zCORR"])]
    for label, cols in specs:
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
        beta, se, t, p, r2 = ols_nw(y, X)
        terms = "   ".join(f"{c} {b:+.2f} (t={tt:+.2f})"
                           for c, b, tt in zip(cols, beta[1:], t[1:]))
        lines.append(f"  {label:22s} {terms}   R2={100 * r2:.1f}%")
    # double sort: DIX gradient within vol x corr cells (mostly gated out --
    # which is itself the honest answer on this sample size)
    parts = []
    for vz in VZONES:
        for cz in ZONES:
            cell = []
            for dz in ("DIXLow", "DIXHigh"):
                mask = ((M["vz_roll"] == vz) & (M["cz_roll"] == cz)
                        & (M["dz_roll"] == dz))
                st = mask_stats(M, mask, with_ci=False)
                cell.append(st)
            if all(s["gate"] for s in cell):
                parts.append(f"  {vz}/{cz}: DIXLow {cell[0]['mean']:+.2f}% "
                             f"(n={cell[0]['n_days']}) -> DIXHigh "
                             f"{cell[1]['mean']:+.2f}% (n={cell[1]['n_days']})")
    lines.append("  -- DIX Low->High gradient, double-sorted vol x corr "
                 "(only cells passing the gate) --")
    lines += parts if parts else ["  (no vol x corr cell passes the print gate)"]
    return "\n".join(lines)


def interaction_report(M, proxy):
    lines = [f"=== INTERACTION REGRESSION ({proxy} r1m; Newey-West 21-lag t-stats) ===",
             "  Does the DIX read differently by comovement regime? zRV controls for",
             "  the corr<->vol overlap (corr(zCORR,zRV)="
             f"{M['zCORR'].corr(M['zRV']):.2f})."]
    d = M[["r1m", "zDIX", "zCORR", "zRV"]].dropna()
    y = d["r1m"].to_numpy()
    inter = (d["zDIX"] * d["zCORR"]).to_numpy()
    specs = [
        ("r1m ~ zDIX + zCORR + zDIX*zCORR",
         np.column_stack([np.ones(len(d)), d["zDIX"], d["zCORR"], inter]),
         ["const", "zDIX", "zCORR", "zDIX*zCORR"]),
        ("r1m ~ zDIX + zCORR + zDIX*zCORR + zRV",
         np.column_stack([np.ones(len(d)), d["zDIX"], d["zCORR"], inter, d["zRV"]]),
         ["const", "zDIX", "zCORR", "zDIX*zCORR", "zRV"]),
    ]
    for label, X, cols in specs:
        beta, se, t, p, r2 = ols_nw(y, X)
        terms = "   ".join(f"{c} {b:+.2f} (t={tt:+.2f})"
                           for c, b, tt in zip(cols[1:], beta[1:], t[1:]))
        lines.append(f"  {label}:  n={len(d)}  {terms}   R2={100 * r2:.1f}%")
    return "\n".join(lines)


def spread_report(M, czone_col, cadence="daily"):
    lines = [f"=== PER-NAME DARK-FLOW TILT ({cadence}): Q5-minus-Q1 1-month spread "
             "by regime ===",
             "  tilt = MA raw dark ratio minus the name's own expanding mean;",
             "  spread = mean fwd r21 of top-20% tilt names minus bottom-20%."]
    if M["tilt_spread"].notna().sum() == 0:
        return lines[0] + "\n  (no per-name dark-flow panel for this index)"
    seed = 300
    for z in ZONES:
        mask = (M[czone_col] == z) & M["tilt_spread"].notna()
        sub = M[mask]
        seed += 1
        st = mask_stats(M, mask, col="tilt_spread", seed=seed)
        lines.append(f"  {z:9s} spread " + fmt_cell(st)
                     + f"   (Q5 {sub['tilt_q5'].mean():+.2f}%, Q1 {sub['tilt_q1'].mean():+.2f}%)")
        if z == "LowCorr" and st["n_days"]:
            lines.append(f"    LowCorr per-year: "
                         f"{per_year_line(M, mask, col='tilt_spread')}")
    d = M[["tilt_spread", "zCORR"]].dropna()
    if len(d) > 100:
        X = np.column_stack([np.ones(len(d)), d["zCORR"].to_numpy()])
        beta, se, t, p, _ = ols_nw(d["tilt_spread"].to_numpy(), X)
        lines.append(f"  continuous: spread ~ zCORR slope {beta[1]:+.2f}pp/z "
                     f"(NW t={t[1]:+.2f}, p={p[1]:.3f}; const {beta[0]:+.2f}pp, "
                     f"t={t[0]:+.2f})")
    return "\n".join(lines)


def tape_report(M, proxy, with_ci=True):
    lines = [f"=== TAPE TAXONOMY (trailing 21d {proxy} sign x breadth tercile) ===",
             "  'all stocks rally' vs 'only a certain group rallies', counted directly."]
    order = ["broad rally", "mid rally", "narrow rally",
             "selective selloff", "mid selloff", "broad selloff"]
    seed = 400
    for tape in order:
        mask = M["tape"] == tape
        if not mask.any():
            continue
        seed += 1
        sub = M[mask]
        st = mask_stats(M, mask, with_ci=with_ci, seed=seed)
        lines.append(f"  {tape:18s} avg_corr {sub['avg_corr'].mean():.2f}  "
                     f"->  {proxy} 1m " + fmt_cell(st))
    # DIX inside the tapes (rolling-basis zones), print-gated
    for tape in ("broad rally", "mid rally", "narrow rally", "broad selloff"):
        if (M["tape"] == tape).sum() < 120:
            continue
        parts = []
        for dz in DZONES:
            mask = (M["tape"] == tape) & (M["dz_roll"] == dz)
            st = mask_stats(M, mask, with_ci=False)
            parts.append(f"{dz} {st['mean']:+.2f}% (n={st['n_days']}d/{st['n_eps']}ep)"
                         if st["gate"] else f"{dz} -- (n={st['n_days']})")
        lines.append(f"    {tape} by DIX: " + "   ".join(parts))
    return "\n".join(lines)


def entry_events(M, mask, min_gap=WINDOW):
    """First-day-of-condition events with a `min_gap`-session cool-down."""
    m = mask.to_numpy()
    entries, last = [], None
    for pos in range(len(m)):
        if m[pos] and (pos == 0 or not m[pos - 1]):
            if last is None or pos - last >= min_gap:
                entries.append(pos)
                last = pos
    return entries


def entry_report(M, czone_col, dzone_col, proxy):
    lines = [f"=== REGIME-ENTRY EVENT STUDY ({czone_col} basis; first day the "
             "condition forms, 21-session cool-down) ==="]
    conds = [
        ("enter HighCorr", M[czone_col] == "HighCorr"),
        ("enter LowCorr", M[czone_col] == "LowCorr"),
        ("enter LowCorr & DIXHigh", (M[czone_col] == "LowCorr") & (M[dzone_col] == "DIXHigh")),
        ("enter LowCorr & DIXLow", (M[czone_col] == "LowCorr") & (M[dzone_col] == "DIXLow")),
        ("enter HighCorr & DIXLow", (M[czone_col] == "HighCorr") & (M[dzone_col] == "DIXLow")),
    ]
    if "dz_roll_l1" in M.columns:
        conds.append(("enter LowCorr & DIXHigh (lag-1)",
                      (M[czone_col] == "LowCorr") & (M["dz_roll_l1"] == "DIXHigh")))
    for label, mask in conds:
        idx = entry_events(M, mask)
        if not idx:
            lines.append(f"  {label:26s} entries=  0")
            continue
        r = M["r1m"].iloc[idx].dropna()
        last = M.index[idx[-1]].date()
        if len(r) < GATE_EVENTS:
            lines.append(f"  {label:26s} entries={len(idx):3d}  -- (below "
                         f"{GATE_EVENTS}-event gate)   [last entry {last}]")
            continue
        lines.append(f"  {label:26s} entries={len(idx):3d}  {proxy} "
                     f"{r.mean():+.2f}% (med {r.median():+.2f}, "
                     f"hit {100 * (r > 0).mean():3.0f}%)   [last entry {last}]")
    return "\n".join(lines)


def oos_report(M, proxy):
    """Cutoffs and loadings fitted < OOS_SPLIT, evaluated on the rest."""
    tr = M[M.index < OOS_SPLIT]
    te = M[M.index >= OOS_SPLIT].copy()
    lines = [f"=== OUT-OF-SAMPLE SPLIT (fit < {OOS_SPLIT}, evaluate >= {OOS_SPLIT}) ===",
             f"  train n={len(tr)}   test n={len(te)}"]
    if len(tr) < 300 or len(te) < 100:
        return "\n".join(lines + ["  insufficient data for the split"])

    def zone_by_train(col, labels):
        lo, hi = tr[col].quantile([0.30, 0.70])
        return np.where(te[col] <= lo, labels[0],
                        np.where(te[col] >= hi, labels[2], labels[1]))

    te["cz_oos"] = zone_by_train("avg_corr", ZONES)
    te["dz_oos"] = zone_by_train("dix5", DZONES)
    base = te["r1m"].dropna()
    lines.append(f"  test baseline: {proxy} {base.mean():+.2f}% "
                 f"(hit {100 * (base > 0).mean():.0f}%, n={len(base)})")

    def grid(frame, czl, dzl, tag):
        out = [f"  -- {tag} --"]
        for cz in ZONES:
            parts = []
            for dz in DZONES:
                mask = (frame[czl] == cz) & (frame[dzl] == dz)
                st = mask_stats(frame, mask, with_ci=False)
                parts.append(f"{dz} {st['mean']:+.2f}% (n={st['n_days']}d/"
                             f"{st['n_eps']}ep)" if st["gate"]
                             else f"{dz} -- (n={st['n_days']})")
            spm = (frame[czl] == cz) & frame["tilt_spread"].notna()
            sps = mask_stats(frame, spm, col="tilt_spread", with_ci=False)
            parts.append(f"tilt-spread {sps['mean']:+.2f}pp" if sps["gate"]
                         else "tilt-spread --")
            out.append(f"  {cz:9s} " + "   ".join(parts))
        return out

    # (a) live rolling-basis zones scored on the test window only -- what the
    # headline construction actually delivered post-split
    lines += grid(te, "cz_roll", "dz_roll", "live rolling-basis zones, test window")
    # (b) the stricter classic variant: cutoff LEVELS frozen on train
    lines += grid(te, "cz_oos", "dz_oos", "train-frozen cutoff levels")

    # interaction loadings from train applied to test
    mu = {c: tr[c].mean() for c in ("dix5", "avg_corr")}
    sd = {c: tr[c].std(ddof=0) for c in ("dix5", "avg_corr")}

    def zc(df, c):
        return (df[c] - mu[c]) / sd[c]

    dtr = pd.DataFrame({"D": zc(tr, "dix5"), "C": zc(tr, "avg_corr"),
                        "r": tr["r1m"]}).dropna()
    dte = pd.DataFrame({"D": zc(te, "dix5"), "C": zc(te, "avg_corr"),
                        "r": te["r1m"]}).dropna()
    if len(dtr) >= 200 and len(dte) >= 60:
        Xtr = np.column_stack([np.ones(len(dtr)), dtr["D"], dtr["C"], dtr["D"] * dtr["C"]])
        beta, se, t, p, _ = ols_nw(dtr["r"].to_numpy(), Xtr)
        pred = (beta[0] + beta[1] * dte["D"] + beta[2] * dte["C"]
                + beta[3] * dte["D"] * dte["C"]).to_numpy()
        real = dte["r"].to_numpy()
        oc = float(np.corrcoef(pred, real)[0, 1]) if len(dte) > 2 else np.nan
        # overlap-honest significance: realized ~ pred with NW(21) errors
        # (a plain day-count corr overstates n by ~21x on overlapping windows)
        Xte = np.column_stack([np.ones(len(dte)), pred])
        _, _, t_te, _, _ = ols_nw(real, Xte)
        edge = real[pred > dtr["r"].mean()].mean() - real.mean()
        lines.append(f"  interaction fit on train (zDIX*zCORR b={beta[3]:+.2f}, "
                     f"t={t[3]:+.2f})   OOS corr(pred, realized)={oc:+.3f} "
                     f"(NW t={t_te[1]:+.2f}, n={len(dte)})   "
                     f"OOS mean when pred>train-avg: {edge:+.2f}pp vs all-test")
    return "\n".join(lines)


def cross_index_report(frames):
    """Pairwise correlation of the AVG_CORR gauges, regime agreement, and
    each index's forward return by how many of the three sit in LowCorr
    (rolling basis, common dates). Cells carry era-adjusted means, episode
    counts, and (for the contested all-dispersed row) episode-cluster CIs."""
    lines = ["=== CROSS-INDEX COMOVEMENT-REGIME AGREEMENT (rolling basis) ===",
             "  ex = era-adjusted mean (diagnostic)   epCI = episode-cluster 95%"]
    if len(frames) < 2:
        return lines[0] + "\n  (needs at least two indices)"
    keys = list(frames)
    ac = pd.DataFrame({k: frames[k]["avg_corr"] for k in keys}).dropna()
    cz = pd.DataFrame({k: frames[k]["cz_roll"] for k in keys}).dropna()
    cz = cz[(cz != "NA").all(axis=1)]
    lines.append(f"  common days: {len(cz)}  "
                 f"[{cz.index.min().date()} -> {cz.index.max().date()}]")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            lines.append(f"  corr(AVG_CORR {a}, {b}) = {ac[a].corr(ac[b]):+.2f}   "
                         f"same regime {100 * (cz[a] == cz[b]).mean():.0f}% of days")
    if len(keys) == 3:
        all_same = (cz.nunique(axis=1) == 1).mean()
        lines.append(f"  all three in the same regime: {100 * all_same:.0f}% of days")
    nlow = (cz == "LowCorr").sum(axis=1)
    lines.append("  forward 1m by number of indices in LowCorr:")
    for k in range(len(keys) + 1):
        mask_dates = nlow[nlow == k].index
        if not len(mask_dates):
            continue
        parts = []
        for key in keys:
            Mi = frames[key][["r1m", "r1m_ex"]].reindex(cz.index)
            mask = pd.Series(False, index=cz.index)
            mask.loc[mask_dates] = True
            st = mask_stats(Mi, mask, with_ci=(k == len(keys)))
            if st["gate"]:
                s = f"{key} {st['mean']:+.2f}%"
                if np.isfinite(st.get("ex", np.nan)):
                    s += f" (ex {st['ex']:+.2f})"
                lo, hi = st.get("ci", (np.nan, np.nan))
                if np.isfinite(lo):
                    s += f" [epCI {lo:+.1f},{hi:+.1f}]"
            else:
                s = f"{key} --"
            parts.append(s)
        eps = int(run_ids(nlow.eq(k).to_numpy()).max() + 1)
        lines.append(f"    {k} of {len(keys)} dispersed: n={len(mask_dates):4d}d/"
                     f"{eps}ep   " + "   ".join(parts))
    return "\n".join(lines)


def barometer_crosscheck(M, barometer_path):
    """Correlate the NDX-internal AVG_CORR gauge against the GEX/dispersion
    barometer's realized top-50 SPX correlation and Cboe's COR1M (both packed
    as `const SER = {...}` in the built barometer page)."""
    try:
        with open(barometer_path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return ("=== EXTERNAL GAUGE CROSS-CHECK ===\n"
                f"  (no barometer page at {barometer_path} -- skipped)")
    m = re.search(r"const SER = (\{.*?\});\n", html, re.S)
    if not m:
        return ("=== EXTERNAL GAUGE CROSS-CHECK ===\n"
                "  (no SER series in the barometer page -- skipped)")
    ser = json.loads(m.group(1))
    dates = pd.to_datetime(ser["dates"])
    lines = ["=== EXTERNAL GAUGE CROSS-CHECK (NDX AVG_CORR vs GEX/dispersion barometer) ==="]
    ours = M["avg_corr"]
    for key, label in (("cor", "realized top-50 SPX corr"),
                       ("cor1m", "Cboe COR1M implied corr")):
        if key not in ser:
            continue
        s = series_from(ser[key], dates)
        d = pd.concat([ours, s], axis=1, join="inner").dropna()
        if len(d) < 100:
            lines.append(f"  {label}: insufficient overlap")
            continue
        lvl = float(d.corr().iloc[0, 1])
        dd = d.diff(WINDOW).dropna()
        chg = float(dd.corr().iloc[0, 1]) if len(dd) > 50 else np.nan
        lines.append(f"  {label:26s} corr(level)={lvl:+.2f}  "
                     f"corr(21d change)={chg:+.2f}  (n={len(d)})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def report_index(name, M, meta, args):
    proxy = meta["proxy"]
    with_ci = not args.no_ci
    print(f"##### {name}: {meta['note']}  "
          f"({M.index.min().date()} -> {M.index.max().date()}, {len(M)} days) #####")
    if meta["dropped"]:
        print(f"Dropped by close-vs-r21 validation: {meta['dropped']}")
    print(f"AVG_CORR: mean {M['avg_corr'].mean():.2f}  "
          f"p10 {M['avg_corr'].quantile(0.10):.2f}  p90 {M['avg_corr'].quantile(0.90):.2f}  "
          f"latest {M['avg_corr'].iloc[-1]:.2f}\n")
    tables = {}
    for basis, czl, dzl in (
            ("ROLLING (headline: live-knowable, drift-proof)", "cz_roll", "dz_roll"),
            ("expanding (no look-ahead, drift-EXPOSED)", "cz_exp", "dz_exp"),
            ("full-sample (descriptive, look-ahead)", "cz_full", "dz_full")):
        Mb = M if czl == "cz_full" else M[(M[czl] != "NA") & (M[dzl] != "NA")]
        print(describe_regimes(Mb, czl, proxy, with_ci))
        print()
        tab = dix_by_corr_table(Mb, czl, dzl, with_ci)
        print(f"=== {proxy} 1m FORWARD BY corr-regime x DIX-regime ({basis}) ===")
        print(tab.to_string(index=False))
        print()
        if czl == "cz_roll":
            tables[name] = tab
            print(flagship_report(Mb, czl, dzl))
            print()
            # realistic timing: same grid on the one-session-lagged DIX zone
            Ml = M[(M["cz_roll"] != "NA") & (M["dz_roll_l1"] != "NA")]
            lag = dix_by_corr_table(Ml, "cz_roll", "dz_roll_l1", with_ci=False)
            print(f"=== {proxy} 1m FORWARD, DIX signal LAGGED 1 SESSION "
                  "(tradeable timing: FINRA prints after the close) ===")
            print(lag[["corr_regime", "dix_regime", "n_days", "n_eps",
                       "mean", "ex", "hit", "gate"]].to_string(index=False))
            print()
    print(episode_report(M, "cz_roll"))
    print()
    print(interaction_report(M, proxy))
    print()
    print(vol_parallel_report(M, proxy))
    print()
    print(spread_report(M, "cz_roll",
                        cadence="weekly, raw print" if name == "SPX" else "daily"))
    print()
    print(tape_report(M, proxy, with_ci))
    print()
    print(entry_report(M[(M["cz_roll"] != "NA") & (M["dz_roll"] != "NA")],
                       "cz_roll", "dz_roll", proxy))
    print()
    print(oos_report(M, proxy))
    print()
    return tables[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="docs/index.html",
                    help="built dashboard HTML carrying the payload (default docs/index.html)")
    ap.add_argument("--indices", default="ndx,spx,iwm",
                    help="comma list of ndx/spx/iwm (default all three; "
                         "spx and iwm need network or a warm cache)")
    ap.add_argument("--basket-size", type=int, default=BASKET_N,
                    help=f"constituent basket size for the SPX/IWM gauges (default {BASKET_N})")
    ap.add_argument("--cache-dir", default=".ndx_dark_cache",
                    help="holdings/price cache directory (default .ndx_dark_cache)")
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download of the SPX/IWM basket prices")
    ap.add_argument("--csv", default=None,
                    help="write the per-index corr-regime x DIX tables to this CSV")
    ap.add_argument("--aligned-csv", default=None,
                    help="write the aligned per-day frames here (one file per index, "
                         "suffixed _ndx/_spx/_iwm before the extension)")
    ap.add_argument("--barometer", default="docs/gex_dispersion.html",
                    help="built GEX/dispersion page for the external gauge cross-check "
                         "(skipped when absent)")
    ap.add_argument("--min-valid-corr", type=float, default=0.98,
                    help="min corr between packed-close 21d returns and the payload's "
                         "adjusted r21 before an NDX name is dropped (default 0.98)")
    ap.add_argument("--no-ci", action="store_true",
                    help="skip the block-bootstrap CIs (faster)")
    args = ap.parse_args()

    P = load_payload(args.html)
    print(f"Payload generated: {P.get('generated')}")
    print("NOTE: this report prints many conditional cells; only the two cells "
          "marked PRIMARY in the flagship section are pre-specified claims -- "
          "every other cell is descriptive. Means are suppressed below the "
          f"print gate ({GATE_DAYS} days AND {GATE_EPISODES} episodes; "
          f"{GATE_EVENTS} events for entry studies).\n")
    wanted = [w.strip().upper() for w in args.indices.split(",") if w.strip()]

    frames, metas, tabs = {}, {}, []
    for name in ("NDX", "SPX", "IWM"):
        if name not in wanted:
            continue
        try:
            if name == "NDX":
                M, meta = build_ndx_frame(P, min_valid_corr=args.min_valid_corr)
            else:
                M, meta = build_basket_frame(P, name, args.basket_size,
                                             args.cache_dir, refresh=args.refresh)
                if name == "SPX":
                    sp = spx_weekly_tilt(P)
                    if sp is not None:
                        M[["tilt_q5", "tilt_q1", "tilt_spread"]] = (
                            sp[["q5", "q1", "spread"]].reindex(M.index))
        except RuntimeError as e:
            print(f"##### {name}: SKIPPED ({e}) #####\n", file=sys.stderr)
            continue
        frames[name], metas[name] = M, meta
        if args.aligned_csv:
            stem, dot, ext = args.aligned_csv.rpartition(".")
            M.to_csv(f"{stem}_{name.lower()}{dot}{ext}" if dot else
                     f"{args.aligned_csv}_{name.lower()}")
        tab = report_index(name, M, meta, args)
        tab.insert(0, "index", name)
        tabs.append(tab)

    if args.csv and tabs:
        pd.concat(tabs, ignore_index=True).to_csv(args.csv, index=False)

    print(cross_index_report(frames))
    print()
    if "NDX" in frames:
        print(barometer_crosscheck(frames["NDX"], args.barometer))


if __name__ == "__main__":
    main()
