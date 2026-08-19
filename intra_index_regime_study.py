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
OOS_SPLIT = "2024-01-01"
BASKET_N = 100         # default basket size for the fetched SPX/IWM universes
FETCH_START = "2019-10-01"   # SPX/IWM DIX starts 2020-01; buffer for the 21d warm-up
ZONES = ("LowCorr", "MidCorr", "HighCorr")
DZONES = ("DIXLow", "DIXMid", "DIXHigh")
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


def zones_30_40_30(s, basis="full", labels=("Low", "Mid", "High"), min_obs=EXP_MIN):
    """Low/Mid/High on a 30/40/30 split of `s` (the comovement study's zone
    convention). basis='full' uses full-sample 30th/70th pct cutoffs (mild
    look-ahead); basis='expanding' ranks each day against its own past only
    (NaN -> 'NA' until min_obs)."""
    if basis == "expanding":
        pct = expanding_pctile(s, min_obs=min_obs)
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

    # zones on both bases
    for col, prefix, labels in (("avg_corr", "c", ZONES), ("dix5", "d", DZONES)):
        M[prefix + "z_full"] = zones_30_40_30(M[col], "full", labels)
        M[prefix + "z_exp"] = zones_30_40_30(M[col], "expanding", labels)

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
def fmt_stats(r, with_ci=True, seed=0):
    r = pd.Series(r).dropna()
    if not len(r):
        return "--"
    s = f"{r.mean():+.2f}% (med {r.median():+.2f}, hit {100 * (r > 0).mean():3.0f}%)"
    if with_ci:
        lo, hi = block_boot_ci(r.to_numpy(), seed=seed)
        if np.isfinite(lo):
            s += f" [CI {lo:+.1f},{hi:+.1f}]"
    return s


def describe_regimes(M, zcol, proxy, with_ci=True):
    lines = [f"=== COMOVEMENT REGIMES ({zcol}; 30/40/30 split of AVG_CORR) ==="]
    seed = 0
    for z in ZONES:
        sub = M[M[zcol] == z]
        if not len(sub):
            continue
        seed += 1
        lines.append(
            f"  {z:9s} n={len(sub):4d}  avg_corr {sub['avg_corr'].mean():.2f}  "
            f"disp {sub['disp21'].mean():.2f}%  breadth {sub['breadth'].mean():.2f}  "
            f"{proxy}vol {sub['rv'].mean():4.1f}  ->  {proxy} 1m "
            + fmt_stats(sub["r1m"], with_ci, seed=seed))
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
    rows = []
    seed = 100
    for cz in ZONES:
        for dz in DZONES:
            sub = M[(M[czone_col] == cz) & (M[dzone_col] == dz)]
            if not len(sub):
                continue
            seed += 1
            r = sub["r1m"].dropna()
            lo, hi = block_boot_ci(r.to_numpy(), seed=seed) if with_ci else (np.nan, np.nan)
            sp = sub["tilt_spread"].dropna()
            rows.append({
                "corr_regime": cz, "dix_regime": dz, "n_days": len(sub),
                "mean": round(float(r.mean()), 2) if len(r) else np.nan,
                "med": round(float(r.median()), 2) if len(r) else np.nan,
                "hit": round(float((r > 0).mean() * 100)) if len(r) else np.nan,
                "ci_lo": round(lo, 2) if np.isfinite(lo) else np.nan,
                "ci_hi": round(hi, 2) if np.isfinite(hi) else np.nan,
                "spread_mean": round(float(sp.mean()), 2) if len(sp) else np.nan,
            })
    return pd.DataFrame(rows)


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
        sub = M[M[czone_col] == z]
        seed += 1
        lines.append(f"  {z:9s} n={sub['tilt_spread'].notna().sum():4d}   spread "
                     + fmt_stats(sub["tilt_spread"], seed=seed)
                     + f"   (Q5 {sub['tilt_q5'].mean():+.2f}%, Q1 {sub['tilt_q1'].mean():+.2f}%)")
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
        sub = M[M["tape"] == tape]
        if not len(sub):
            continue
        seed += 1
        lines.append(f"  {tape:18s} n={len(sub):4d}  avg_corr {sub['avg_corr'].mean():.2f}  "
                     f"->  {proxy} 1m " + fmt_stats(sub["r1m"], with_ci, seed=seed))
    # DIX inside the tapes with enough days
    for tape in ("broad rally", "mid rally", "narrow rally", "broad selloff"):
        sub = M[M["tape"] == tape]
        if len(sub) < 120:
            continue
        parts = []
        for dz in DZONES:
            r = sub[sub["dz_full"] == dz]["r1m"].dropna()
            parts.append(f"{dz} {r.mean():+.2f}% (n={len(r)})" if len(r) else f"{dz} --")
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
    for label, mask in conds:
        idx = entry_events(M, mask)
        if not idx:
            lines.append(f"  {label:26s} entries=  0")
            continue
        r = M["r1m"].iloc[idx].dropna()
        last = M.index[idx[-1]].date()
        lines.append(f"  {label:26s} entries={len(idx):3d}  {proxy} "
                     + fmt_stats(r, with_ci=False) + f"   [last entry {last}]")
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
    for cz in ZONES:
        parts = []
        for dz in DZONES:
            r = te[(te["cz_oos"] == cz) & (te["dz_oos"] == dz)]["r1m"].dropna()
            parts.append(f"{dz} {r.mean():+.2f}% (n={len(r)})" if len(r) else f"{dz} --")
        sp = te[te["cz_oos"] == cz]["tilt_spread"].dropna()
        parts.append(f"tilt-spread {sp.mean():+.2f}pp" if len(sp) else "tilt-spread --")
        lines.append(f"  {cz:9s} " + "   ".join(parts))

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
        edge = real[pred > dtr["r"].mean()].mean() - real.mean()
        lines.append(f"  interaction fit on train (zDIX*zCORR b={beta[3]:+.2f}, "
                     f"t={t[3]:+.2f})   OOS corr(pred, realized)={oc:+.3f} (n={len(dte)})   "
                     f"OOS mean when pred>train-avg: {edge:+.2f}pp vs all-test")
    return "\n".join(lines)


def cross_index_report(frames):
    """Pairwise correlation of the AVG_CORR gauges, regime agreement, and
    each index's forward return by how many of the three sit in LowCorr
    (full-sample basis, common dates)."""
    lines = ["=== CROSS-INDEX COMOVEMENT-REGIME AGREEMENT (full-sample basis) ==="]
    if len(frames) < 2:
        return lines[0] + "\n  (needs at least two indices)"
    keys = list(frames)
    ac = pd.DataFrame({k: frames[k]["avg_corr"] for k in keys}).dropna()
    cz = pd.DataFrame({k: frames[k]["cz_full"] for k in keys}).dropna()
    r1 = pd.DataFrame({f"{k}_r1m": frames[k]["r1m"] for k in keys})
    lines.append(f"  common days: {len(ac)}  "
                 f"[{ac.index.min().date()} -> {ac.index.max().date()}]")
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
        sub = r1.reindex(nlow[nlow == k].index)
        if not len(sub):
            continue
        parts = [f"{c[:-4]} {sub[c].dropna().mean():+.2f}%" for c in r1.columns
                 if sub[c].notna().sum()]
        lines.append(f"    {k} of {len(keys)} dispersed: n={len(sub):4d}   "
                     + "   ".join(parts))
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
    for basis, czl, dzl in (("full-sample", "cz_full", "dz_full"),
                            ("EXPANDING (no look-ahead)", "cz_exp", "dz_exp")):
        Mb = M if basis == "full-sample" else M[(M[czl] != "NA") & (M[dzl] != "NA")]
        print(describe_regimes(Mb, czl, proxy, with_ci))
        print()
        tab = dix_by_corr_table(Mb, czl, dzl, with_ci)
        print(f"=== {proxy} 1m FORWARD BY corr-regime x DIX-regime ({basis}) ===")
        print(tab.to_string(index=False))
        print()
        if basis == "full-sample":
            tables[name] = tab
    print(episode_report(M, "cz_full"))
    print()
    print(interaction_report(M, proxy))
    print()
    print(spread_report(M, "cz_full",
                        cadence="weekly, raw print" if name == "SPX" else "daily"))
    print()
    print(tape_report(M, proxy, with_ci))
    print()
    print(entry_report(M[(M["cz_exp"] != "NA") & (M["dz_exp"] != "NA")],
                       "cz_exp", "dz_exp", proxy))
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
    print(f"Payload generated: {P.get('generated')}\n")
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
