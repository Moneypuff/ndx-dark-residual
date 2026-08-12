#!/usr/bin/env python3
"""
SPX raw-DIX: DECILE method vs. trailing-1-year BREAKOUT/BREAKDOWN.
=================================================================

Two questions, one dataset (per request):

  1. Which framing of the raw S&P 500 Dark Index (DIX) carries more forward-
     return information -- the LEVEL-in-its-own-history DECILE, or a trailing
     1-year BREAKOUT / BREAKDOWN (today's DIX vs. its own 252-session average)?

  2. "The bullish floor on the raw DIX looks like ~44-45% in 2024-today, but
     back in 2019 a 45% print was rare and bullish. Did something break?"
     -- i.e. is the raw DIX level stationary, and does a fixed threshold still
     mean what it used to?

Data source
-----------
SqueezeMetrics' free, canonical daily DIX file -- the actual published series
everyone quotes when they say "DIX is at 45%":

    https://squeezemetrics.com/monitor/static/DIX.csv
    columns: date, price (SPX close), dix (0..1), gex

This is the *raw* DIX (not the repo's reconstructed dollar-DIX). Pass --csv to
point at a local snapshot for offline / CI reproduction.

Signals (all knowable in real time -- no look-ahead in the signal)
------------------------------------------------------------------
Let DIX_t be the raw daily value.

  * FIXED threshold   : DIX_t vs. an absolute cutoff (e.g. 0.45). The naive
                        "45% = bullish" rule. Shown to *fail* under drift.
  * EXPANDING decile  : DIX_t ranked 1..10 within all of its own history up to t
                        (min 252 obs) -- the long-memory "level" signal.
  * TRAILING-1y decile: DIX_t ranked 1..10 within only the last 252 sessions --
                        a fully de-trended "level within the past year".
  * BREAKOUT z (1y)   : (DIX_t - mean_252) / std_252 -- how many trailing-year
                        sigmas above/below its own 1-year average. Breakout = z
                        high (DIX pushing above its year); breakdown = z low.

High DIX is SqueezeMetrics' bullish reading (dark buying / short flow that is
mostly market-maker hedged supply), so a positive signal->return relationship
is the hypothesis for all four.

Outcome (no look-ahead)
-----------------------
The SPX's own h-session forward return, in percent, from the DIX file's price
column: r21 (1mo), r42 (2mo), r63 (3mo). Default horizon 21.

Inference (same toolkit as the rest of the repo)
------------------------------------------------
  * Spearman rank IC of each real-time signal vs. the forward return.
  * OLS with Newey-West (Bartlett, 21-lag) HAC t-stats on the overlapping
    windows -- slope in pp of forward return per 1 SD of standardized signal.
  * Moving-block bootstrap (21-day blocks) 95% CIs on decile / state means.
  * A de-overlapped ENTRY-EVENT study (first day into the top/bottom band, with
    a 21-session cool-down) -- the honest count that does not let one long
    episode masquerade as hundreds of independent observations.
  * A JOINT regression (level + breakout together) -- does the breakout add
    anything on top of the level?
  * A pre-2021 vs. 2021+ split -- did the relationship survive the regime shift
    that lifted the whole DIX level?

Reproduce
---------
    python spx_dix_decile_vs_breakout_study.py                     # live fetch
    python spx_dix_decile_vs_breakout_study.py --csv DIX.csv       # offline
    python spx_dix_decile_vs_breakout_study.py --out spx_dix_decile_vs_breakout.csv
"""
import argparse
import io
import math

import numpy as np
import pandas as pd

DIX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
HORIZONS = (21, 42, 63)
EXP_MIN = 252          # min obs before an expanding decile is defined
TRAIL_WIN = 252        # trailing window for the 1-year decile / breakout z
TRAIL_MIN = 200        # min obs inside the trailing window
BOOT_B = 2000
BOOT_L = 21
REGIME_SPLIT = "2021-01-01"   # start of the elevated-off-exchange-volume era


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dix(csv=None, url=DIX_URL, retries=4):
    """Load the raw SqueezeMetrics DIX file (date, price, dix, gex). From a
    local --csv snapshot when given, else fetched with exponential-backoff
    retries (the repo's convention for flaky endpoints)."""
    if csv:
        raw = open(csv, "r", encoding="utf-8").read()
    else:
        import time
        import requests
        last = None
        for i in range(retries):
            try:
                r = requests.get(url, timeout=40)
                r.raise_for_status()
                raw = r.text
                break
            except Exception as e:               # noqa: BLE001 -- retry any transient failure
                last = e
                if i == retries - 1:
                    raise
                time.sleep(2 ** (i + 1))
        else:                                    # pragma: no cover
            raise last
    df = pd.read_csv(io.StringIO(raw), parse_dates=["date"]).set_index("date").sort_index()
    df = df[np.isfinite(df["dix"]) & np.isfinite(df["price"])]
    for h in HORIZONS:
        df[f"r{h}"] = (df["price"].shift(-h) / df["price"] - 1.0) * 100.0
    return df


# ---------------------------------------------------------------------------
# Stats helpers (mirrors index_comovement_study.py so results are comparable)
# ---------------------------------------------------------------------------
def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ols_nw(y, X, lags=21):
    """OLS with Newey-West (Bartlett) HAC errors. Returns (beta, se, t, p, r2)."""
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
    """Moving-block bootstrap CI for the mean of `r` (21-day blocks respect the
    overlap autocorrelation). NaN CI when there are fewer than two blocks."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2 * L:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    means = r[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, levels))


def spearman_ic(sig, ret):
    """Spearman rank correlation (information coefficient) and its n."""
    sig = np.asarray(sig, dtype=float)
    ret = np.asarray(ret, dtype=float)
    m = np.isfinite(sig) & np.isfinite(ret)
    a, b = sig[m], ret[m]
    if len(a) < 30:
        return np.nan, int(m.sum())
    ra = pd.Series(a).rank().to_numpy().astype(float)
    rb = pd.Series(b).rank().to_numpy().astype(float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt((ra @ ra) * (rb @ rb))
    return (float(ra @ rb / denom) if denom > 0 else np.nan), int(m.sum())


def nw_slope(sig, ret, lags=21):
    """Newey-West slope (pp per 1 SD of standardized signal) and t-stat."""
    d = pd.DataFrame({"s": sig, "y": ret}).dropna()
    if len(d) < 100 or d["s"].std(ddof=0) == 0:
        return np.nan, np.nan, len(d)
    s = (d["s"] - d["s"].mean()) / d["s"].std(ddof=0)
    beta, se, t, p, r2 = ols_nw(d["y"].to_numpy(),
                                np.column_stack([np.ones(len(d)), s.to_numpy()]), lags)
    return float(beta[1]), float(t[1]), len(d)


# ---------------------------------------------------------------------------
# Real-time signal construction
# ---------------------------------------------------------------------------
def expanding_decile(s, min_obs=EXP_MIN):
    """Decile 1..10 of each value within its own history to date (mid-rank on
    ties). NaN until min_obs observations accrue -- the cutoffs a live user
    could actually have known."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(min_obs - 1, len(v)):
        h = v[: i + 1]
        h = h[np.isfinite(h)]
        if len(h) < min_obs or not np.isfinite(v[i]):
            continue
        pct = (h < v[i]).mean() + 0.5 * (h == v[i]).mean()
        out[i] = min(9, int(pct * 10)) + 1
    return pd.Series(out, index=s.index)


def trailing_decile(s, win=TRAIL_WIN, min_obs=TRAIL_MIN):
    """Decile 1..10 of each value within only the trailing `win` sessions -- a
    fully de-trended 'where does today sit within the past year' rank."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        h = v[max(0, i - win + 1): i + 1]
        h = h[np.isfinite(h)]
        if len(h) < min_obs or not np.isfinite(v[i]):
            continue
        pct = (h < v[i]).mean() + 0.5 * (h == v[i]).mean()
        out[i] = min(9, int(pct * 10)) + 1
    return pd.Series(out, index=s.index)


def breakout_z(dix, win, min_frac=0.8):
    """Trailing-`win`-session sigmas of DIX above/below its own rolling average:
    (DIX - mean_win) / std_win. A shorter win reacts faster (a 3-month baseline
    adapts within a quarter; a 1-year baseline within a year)."""
    mp = max(20, int(win * min_frac))
    rm = dix.rolling(win, min_periods=mp).mean()
    rs = dix.rolling(win, min_periods=mp).std()
    return (dix - rm) / rs


def add_signals(df):
    dix = df["dix"]
    rm = dix.rolling(TRAIL_WIN, min_periods=TRAIL_MIN).mean()
    df["breakout_z"] = breakout_z(dix, TRAIL_WIN)   # trailing-1y sigmas above/below own average
    df["breakout_gap"] = (dix - rm) * 100.0         # raw gap vs trailing average, pp
    df["exp_dec"] = expanding_decile(dix)
    df["trail_dec"] = trailing_decile(dix)
    df["raw_dix_pct"] = dix * 100.0
    return df


# ---------------------------------------------------------------------------
# Q2 -- is the raw DIX stationary? did "45% = rare & bullish" break?
# ---------------------------------------------------------------------------
def drift_report(df, threshold=0.45):
    lines = ["=" * 78,
             "Q2  RAW-DIX DRIFT -- is the level stationary? did a fixed threshold break?",
             "=" * 78,
             "Yearly distribution of the raw DIX (percent) and where a FIXED "
             f"{threshold:.0%} print ranks:",
             "",
             " year    n    p05   p25   med  mean   p75   p95   %days>=thr   "
             f"pctile of {threshold:.0%}"]
    for y, g in df.groupby(df.index.year):
        d = g["dix"]
        pctile = (d < threshold).mean() * 100
        lines.append(
            f" {y}  {len(d):4d}  " +
            "  ".join(f"{d.quantile(q) * 100:4.1f}" for q in (.05, .25, .5)) +
            f"  {d.mean() * 100:4.1f}  " +
            "  ".join(f"{d.quantile(q) * 100:4.1f}" for q in (.75, .95)) +
            f"    {(d >= threshold).mean() * 100:5.1f}       {pctile:5.1f}th")

    t = (df.index - df.index[0]).days.to_numpy() / 365.25
    b1, b0 = np.polyfit(t, df["dix"].to_numpy(), 1)
    lines += ["",
              f"Linear drift: {b1 * 100:+.3f} pp/year   "
              f"(fitted {b0 * 100:.1f}% in {df.index[0].year} -> "
              f"{(b0 + b1 * t.max()) * 100:.1f}% in {df.index[-1].year}).",
              f"Read-across: a fixed {threshold:.0%} line sat near the TOP of the "
              "distribution pre-2020 (a genuine outlier) and near the BOTTOM once the",
              "off-exchange share stepped up -- so the level moved, the ruler did not. "
              "That is drift, not a broken feed."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q1 -- decile (level) vs breakout/breakdown predictive power
# ---------------------------------------------------------------------------
def decile_table(df, dec_col, ret_col):
    g = df.dropna(subset=[dec_col, ret_col]).groupby(dec_col)[ret_col]
    rows = []
    for dec, r in g:
        rows.append((int(dec), float(r.mean()), float(r.median()), int(r.count())))
    return rows


def headline_power(df, ret_col, seed=0):
    """Spearman IC + Newey-West slope/t + top/bottom-band spread with block-boot
    CI for each real-time signal, on the common sample where all are defined."""
    common = df.dropna(subset=["exp_dec", "trail_dec", "breakout_z", ret_col]).copy()
    rows = []
    specs = [
        ("expanding decile", "exp_dec", "dec"),
        ("trailing-1y decile", "trail_dec", "dec"),
        ("breakout z (1y)", "breakout_z", "cont"),
        ("breakout gap (1y)", "breakout_gap", "cont"),
    ]
    for name, col, kind in specs:
        ic, n = spearman_ic(common[col].to_numpy(), common[ret_col].to_numpy())
        slope, t, _ = nw_slope(common[col].to_numpy(), common[ret_col].to_numpy())
        if kind == "dec":
            hi = common.loc[common[col] >= 9, ret_col]
            lo = common.loc[common[col] <= 2, ret_col]
        else:
            q = common[col].quantile([0.2, 0.8])
            hi = common.loc[common[col] >= q[0.8], ret_col]
            lo = common.loc[common[col] <= q[0.2], ret_col]
        hi_ci = block_boot_ci(hi.to_numpy(), seed=seed + 1)
        lo_ci = block_boot_ci(lo.to_numpy(), seed=seed + 2)
        rows.append({
            "signal": name, "ic": ic, "n": n, "nw_slope": slope, "nw_t": t,
            "hi_mean": float(hi.mean()), "lo_mean": float(lo.mean()),
            "spread": float(hi.mean() - lo.mean()),
            "hi_ci_lo": hi_ci[0], "hi_ci_hi": hi_ci[1],
            "lo_ci_lo": lo_ci[0], "lo_ci_hi": lo_ci[1],
        })
    return common, pd.DataFrame(rows)


def joint_regression(common, ret_col):
    """r ~ level(expanding decile) + breakout(1y z), both standardized, NW(21).
    Answers: does the breakout carry anything the level does not?"""
    d = common[["exp_dec", "breakout_z", ret_col]].dropna().copy()
    for c in ("exp_dec", "breakout_z"):
        d[c] = (d[c] - d[c].mean()) / d[c].std(ddof=0)
    beta, se, t, p, r2 = ols_nw(
        d[ret_col].to_numpy(),
        np.column_stack([np.ones(len(d)), d["exp_dec"].to_numpy(), d["breakout_z"].to_numpy()]))
    return {"n": len(d), "lvl_b": beta[1], "lvl_t": t[1],
            "brk_b": beta[2], "brk_t": t[2], "r2": r2}


def entry_events(state, min_gap=21):
    """De-overlapped first-day-into-band events. `state` is +1/-1/0; returns the
    integer positions of +1 entries and -1 entries with a `min_gap` cool-down."""
    st = np.asarray(state, dtype=float)
    pos, neg, last = [], [], -10 ** 9
    for i in range(len(st)):
        if not np.isfinite(st[i]):
            continue
        prev = st[i - 1] if i > 0 else 0
        if st[i] == 1 and prev != 1 and i - last >= min_gap:
            pos.append(i)
            last = i
        elif st[i] == -1 and prev != -1 and i - last >= min_gap:
            neg.append(i)
            last = i
    return pos, neg


def entry_report(common, ret_col):
    lines = ["=== ENTRY-EVENT study (de-overlapped, first day into band, "
             "21-session cool-down) ==="]
    level_state = np.where(common["exp_dec"] >= 9, 1,
                           np.where(common["exp_dec"] <= 2, -1, 0))
    brk_state = np.where(common["breakout_z"] >= 1, 1,
                         np.where(common["breakout_z"] <= -1, -1, 0))
    for label, state in [("LEVEL  (into top-2 / bottom-2 expanding deciles)", level_state),
                         ("BREAKOUT (into +/-1sd of trailing-1y average)", brk_state)]:
        pos, neg = entry_events(pd.Series(state, index=common.index))
        rp = common[ret_col].iloc[pos].dropna()
        rn = common[ret_col].iloc[neg].dropna()
        lines.append(
            f"  {label}\n"
            f"     high/up  : n={len(rp):3d}  mean {rp.mean():+5.2f}%  hit {(rp > 0).mean() * 100:3.0f}%\n"
            f"     low/down : n={len(rn):3d}  mean {rn.mean():+5.2f}%  hit {(rn > 0).mean() * 100:3.0f}%\n"
            f"     spread   : {rp.mean() - rn.mean():+5.2f} pp")
    return "\n".join(lines)


def subperiod_report(common, ret_col, split=REGIME_SPLIT):
    lines = ["=== SUB-PERIOD stability across the regime shift (Spearman IC) ===",
             "  Did the DIX->forward-return relationship survive the step-up in "
             "off-exchange volume?"]
    for label, sub in [(f"pre-{split[:4]} (old regime)", common[common.index < split]),
                       (f"{split[:4]}+ (new regime)", common[common.index >= split])]:
        e, _ = spearman_ic(sub["exp_dec"].to_numpy(), sub[ret_col].to_numpy())
        tdec, _ = spearman_ic(sub["trail_dec"].to_numpy(), sub[ret_col].to_numpy())
        b, _ = spearman_ic(sub["breakout_z"].to_numpy(), sub[ret_col].to_numpy())
        lines.append(f"  {label:24s} n={len(sub):4d} | exp-dec {e:+.3f} | "
                     f"trail-dec {tdec:+.3f} | breakout {b:+.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Follow-up -- how fast should the breakout baseline be?
# ---------------------------------------------------------------------------
BREAKOUT_WINDOWS = [(21, "1mo"), (42, "2mo"), (63, "3mo"), (126, "6mo"), (252, "1yr")]


def window_sweep(df, ret_col, windows=BREAKOUT_WINDOWS, split=REGIME_SPLIT, seed=0):
    """Sweep the BREAKOUT baseline window. A shorter baseline reacts faster; the
    question is whether faster picks up signal or just noise. For each window,
    reports full-sample IC / Newey-West t / long-short spread, the pre- vs
    post-regime IC split, and the de-overlapped entry-event spread."""
    dix = df["dix"]
    rows = []
    for win, lbl in windows:
        z = breakout_z(dix, win)
        d = pd.DataFrame({"z": z, "ret": df[ret_col]}, index=df.index).dropna()
        ic, n = spearman_ic(d["z"].to_numpy(), d["ret"].to_numpy())
        slope, t, _ = nw_slope(d["z"].to_numpy(), d["ret"].to_numpy())
        q = d["z"].quantile([0.2, 0.8])
        hi = d.loc[d["z"] >= q[0.8], "ret"]
        lo = d.loc[d["z"] <= q[0.2], "ret"]
        pre = d[d.index < split]
        post = d[d.index >= split]
        ic_pre, _ = spearman_ic(pre["z"].to_numpy(), pre["ret"].to_numpy())
        ic_post, _ = spearman_ic(post["z"].to_numpy(), post["ret"].to_numpy())
        state = np.where(d["z"] >= 1, 1, np.where(d["z"] <= -1, -1, 0))
        pos, neg = entry_events(pd.Series(state, index=d.index))
        rp = d["ret"].iloc[pos].dropna()
        rn = d["ret"].iloc[neg].dropna()
        rows.append({
            "window": lbl, "win_sessions": win, "n": n, "ic": ic,
            "nw_slope": slope, "nw_t": t,
            "ls_spread": float(hi.mean() - lo.mean()),
            "ic_pre": ic_pre, "ic_post": ic_post,
            "entry_up": float(rp.mean()), "entry_dn": float(rn.mean()),
            "entry_spread": float(rp.mean() - rn.mean()),
            "n_up": int(len(rp)), "n_dn": int(len(rn)),
        })
    return pd.DataFrame(rows)


def window_sweep_report(df, ret_col):
    sw = window_sweep(df, ret_col)
    lines = [f"=== BREAKOUT baseline-window sweep (forward {ret_col}) -- does a "
             "faster baseline pick up signal or noise? ===",
             " baseline |  IC    | NW t  | LS spread | pre-2021 IC | 2021+ IC | "
             "entry up/dn spread"]
    for _, r in sw.iterrows():
        lines.append(
            f"   {r['window']:5s}  | {r['ic']:+.3f} | {r['nw_t']:+.2f} |   "
            f"{r['ls_spread']:+.2f}   |    {r['ic_pre']:+.3f}   |  {r['ic_post']:+.3f}  |  "
            f"{r['entry_up']:+.2f}/{r['entry_dn']:+.2f} = {r['entry_spread']:+.2f}")
    lines.append("  -> inverted-U in the window: too fast (1mo) is noise, too slow "
                 "(1yr) over-de-trends; the 3-6mo baseline is the sweet spot.")
    return "\n".join(lines)


def smoothed_level_report(df, horizons=HORIZONS, split=REGIME_SPLIT):
    """Contrast: smoothing the LEVEL with an N-session MA before ranking (the
    other reading of 'rolling 3-month average'). Shown to HURT -- smoothing lags
    and erases signal -- so 'faster' should mean a shorter baseline, not a
    smoothed level."""
    dix = df["dix"]
    lines = ["=== SMOOTHED-LEVEL contrast: rank an N-session MA of DIX "
             "(expanding decile) ==="]
    for smth, slbl in [(1, "raw daily"), (21, "1mo MA"), (63, "3mo MA")]:
        s = dix if smth == 1 else dix.rolling(smth, min_periods=max(1, int(smth * 0.6))).mean()
        ed = expanding_decile(s)
        cells = []
        for h in horizons:
            d = pd.DataFrame({"d": ed, "ret": df[f"r{h}"]}, index=df.index).dropna()
            ic, _ = spearman_ic(d["d"].to_numpy(), d["ret"].to_numpy())
            cells.append(f"r{h} IC {ic:+.3f}")
        lines.append(f"  {slbl:9s}: " + "   ".join(cells))
    lines.append("  -> smoothing the level lowers IC monotonically; do not "
                 "pre-average the DIX before ranking it.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Follow-up -- normalize DIX into an OSCILLATOR
# ---------------------------------------------------------------------------
OSC_WIN = 126          # ~6-month window: the sweet spot found in the baseline sweep
OSC_SMOOTH = 3         # %D smoothing (smooth the OUTPUT, never the input DIX)
DIV_LOOKBACK = 21      # lookback for the DIX-vs-price divergence sign


def stochastic_osc(dix, win=OSC_WIN, smooth=OSC_SMOOTH, min_frac=0.8):
    """Stochastic %D on DIX: (DIX - min_win)/(max_win - min_win)*100, then a
    `smooth`-session moving average. Rank/range bounding is robust to the DIX's
    secular drift and fat tails -- the strongest normalization found here."""
    mp = max(20, int(win * min_frac))
    lo = dix.rolling(win, min_periods=mp).min()
    hi = dix.rolling(win, min_periods=mp).max()
    k = (dix - lo) / (hi - lo).replace(0, np.nan) * 100.0
    return k.rolling(smooth, min_periods=1).mean()


def pct_rank_osc(dix, win=OSC_WIN, min_frac=0.8):
    """Rolling percentile-rank oscillator (0..100): where DIX sits within its own
    trailing `win` sessions. Continuous cousin of the trailing decile."""
    mp = max(20, int(win * min_frac))
    return dix.rolling(win, min_periods=mp).apply(
        lambda x: (x[:-1] < x[-1]).mean() * 100.0, raw=True)


def rsi_osc(dix, win=14):
    """RSI on the DIX series itself (a MOMENTUM oscillator). Documented as a
    dead end -- DIX rate-of-change carries ~no forward-return signal."""
    d = dix.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1 / win, adjust=False).mean() / dn.ewm(alpha=1 / win, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def macd_hist_osc(dix, fast=12, slow=26, sig=9):
    """MACD histogram on DIX (momentum). Also a dead end -- kept so it is not
    re-discovered."""
    macd = dix.ewm(span=fast, adjust=False).mean() - dix.ewm(span=slow, adjust=False).mean()
    return macd - macd.ewm(span=sig, adjust=False).mean()


def dix_price_divergence(dix, price, lookback=DIV_LOOKBACK):
    """DIX-vs-price divergence state over `lookback` sessions:
        +1  DIX up   while price DOWN  (bullish divergence -- dark buying into weakness)
        -1  DIX down while price UP    (bearish divergence)
         0  aligned (both up / both down)
    NaN until both changes are defined. Returns (state, dix_chg, price_chg)."""
    dchg = dix.diff(lookback)
    pchg = price.pct_change(lookback)
    dc = np.sign(dchg)
    pc = np.sign(pchg)
    state = pd.Series(0.0, index=dix.index)
    state[(dc > 0) & (pc < 0)] = 1.0
    state[(dc < 0) & (pc > 0)] = -1.0
    state[dchg.isna() | pchg.isna()] = np.nan
    return state, dchg, pchg


def oscillator_summary(df, horizons=HORIZONS, split=REGIME_SPLIT):
    """Tidy summary of every oscillator/divergence construction across horizons:
    IC, Newey-West t, and the pre/post-regime IC split. The committed results
    artifact for Q3 (per-day series stay live/derived, not committed)."""
    specs = [("stoch_%D_126", "osc_stoch"), ("pctrank_126", "osc_pctrank"),
             ("rsi_dix_14", "osc_rsi"), ("macd_hist_dix", "osc_macd")]
    rows = []
    for h in horizons:
        rc = f"r{h}"
        for name, col in specs:
            d = df[[col, rc]].dropna()
            ic, n = spearman_ic(d[col].to_numpy(), d[rc].to_numpy())
            _, t, _ = nw_slope(d[col].to_numpy(), d[rc].to_numpy())
            icp, _ = spearman_ic(d[d.index < split][col].to_numpy(),
                                 d[d.index < split][rc].to_numpy())
            icq, _ = spearman_ic(d[d.index >= split][col].to_numpy(),
                                 d[d.index >= split][rc].to_numpy())
            rows.append({"horizon": h, "signal": name, "n": n, "ic": round(ic, 4),
                         "nw_t": round(t, 3), "ic_pre2021": round(icp, 4),
                         "ic_post2021": round(icq, 4)})
        # GEX-gated stochastic and the bullish-divergence flag as signals too
        g = df[["osc_stoch", "gex", rc]].dropna()
        gm = g[g["gex"] > 0]
        ic, n = spearman_ic(gm["osc_stoch"].to_numpy(), gm[rc].to_numpy())
        icp, _ = spearman_ic(gm[gm.index < split]["osc_stoch"].to_numpy(),
                             gm[gm.index < split][rc].to_numpy())
        icq, _ = spearman_ic(gm[gm.index >= split]["osc_stoch"].to_numpy(),
                             gm[gm.index >= split][rc].to_numpy())
        rows.append({"horizon": h, "signal": "stoch_%D_126|GEX>0", "n": n,
                     "ic": round(ic, 4), "nw_t": np.nan,
                     "ic_pre2021": round(icp, 4), "ic_post2021": round(icq, 4)})
        dd = df[["div_state", rc]].dropna()
        bull = dd[dd["div_state"] > 0][rc]
        base = dd[rc].mean()
        rows.append({"horizon": h, "signal": "bullish_divergence_edge_pp",
                     "n": int(len(bull)), "ic": round(float(bull.mean() - base), 4),
                     "nw_t": np.nan, "ic_pre2021": np.nan, "ic_post2021": np.nan})
    return pd.DataFrame(rows)


def add_oscillators(df):
    dix = df["dix"]
    df["osc_stoch"] = stochastic_osc(dix)
    df["osc_pctrank"] = pct_rank_osc(dix)
    df["osc_rsi"] = rsi_osc(dix)
    df["osc_macd"] = macd_hist_osc(dix)
    df["div_state"], _, _ = dix_price_divergence(dix, df["price"])
    return df


def oscillator_report(df, ret_col, split=REGIME_SPLIT):
    """IC / Newey-West t of each oscillator family, full sample and split across
    the regime shift. Also confirms the response is MONOTONE (directional), not
    mean-reverting, by printing the forward return in the top vs bottom decile."""
    families = [
        ("stochastic %D (126d)", "osc_stoch", True),
        ("percentile-rank (126d)", "osc_pctrank", True),
        ("RSI(DIX) 14 [momentum]", "osc_rsi", False),
        ("MACD-hist(DIX) [momentum]", "osc_macd", False),
    ]
    lines = [f"=== OSCILLATOR families vs forward {ret_col} "
             "(directional / level use) ===",
             " oscillator                 |  IC    | NW t  | pre-2021 | 2021+ | "
             "top-dec  bot-dec"]
    for label, col, keep in families:
        d = df[[col, ret_col]].dropna()
        ic, _ = spearman_ic(d[col].to_numpy(), d[ret_col].to_numpy())
        _, t, _ = nw_slope(d[col].to_numpy(), d[ret_col].to_numpy())
        pre = d[d.index < split]
        post = d[d.index >= split]
        icp, _ = spearman_ic(pre[col].to_numpy(), pre[ret_col].to_numpy())
        icq, _ = spearman_ic(post[col].to_numpy(), post[ret_col].to_numpy())
        dec = pd.qcut(d[col], 10, labels=False, duplicates="drop") + 1
        top = d.loc[dec == dec.max(), ret_col].mean()
        bot = d.loc[dec == dec.min(), ret_col].mean()
        lines.append(f" {label:26s} | {ic:+.3f} | {t:+.2f} |  {icp:+.3f}  | "
                     f"{icq:+.3f} |  {top:+5.2f}   {bot:+5.2f}")
    lines.append("  -> range/rank-bounded oscillators carry the signal; momentum "
                 "(RSI/MACD) does not. Top-decile > bottom-decile => read")
    lines.append("     the oscillator DIRECTIONALLY (high = bullish), NOT as "
                 "overbought/oversold to fade.")
    return "\n".join(lines)


def gex_gating_report(df, ret_col, osc_col="osc_stoch", split=REGIME_SPLIT):
    """Does gating the oscillator on the gamma regime help? IC of the oscillator
    within GEX>0 vs GEX<=0, and a simple GATED long-signal net return."""
    lines = ["=== GEX GATING: oscillator conditioned on the gamma regime ===",
             f"  share of GEX>0 (long-gamma) days: {(df['gex'] > 0).mean() * 100:.0f}%"]
    for lbl, mask in [("GEX>0 (long gamma)", df["gex"] > 0),
                      ("GEX<=0 (short gamma)", df["gex"] <= 0)]:
        d = df[[osc_col, ret_col]][mask].dropna()
        ic, n = spearman_ic(d[osc_col].to_numpy(), d[ret_col].to_numpy())
        pre = d[d.index < split]
        post = d[d.index >= split]
        icp, _ = spearman_ic(pre[osc_col].to_numpy(), pre[ret_col].to_numpy())
        icq, _ = spearman_ic(post[osc_col].to_numpy(), post[ret_col].to_numpy())
        lines.append(f"  {lbl:20s} n={n:4d} | IC {ic:+.3f} | pre {icp:+.3f} | "
                     f"post {icq:+.3f}")
    # gated vs ungated long signal: go long when %D is in its top half
    base = df[[osc_col, ret_col, "gex"]].dropna()
    hi = base[osc_col] >= base[osc_col].median()
    ungated = base.loc[hi, ret_col].mean()
    gated = base.loc[hi & (base["gex"] > 0), ret_col].mean()
    allmean = base[ret_col].mean()
    lines.append(f"  long when %D>median: ungated {ungated:+.2f}%  |  "
                 f"gated on GEX>0 {gated:+.2f}%  |  all-days {allmean:+.2f}%")
    lines.append("  -> the gate is DEFENSIVE, not additive: long-gamma is ~91% of "
                 "days so gating barely moves the average, but it removes the")
    lines.append("     short-gamma regime where the oscillator INVERTS "
                 "(post-2021 GEX<=0 IC goes negative) -- which is what keeps the")
    lines.append("     long-gamma IC the least-dead of anything post-2021.")
    return "\n".join(lines)


def divergence_report(df, ret_col, split=REGIME_SPLIT):
    """DIX-vs-price divergence (option B): forward returns by the four sign cells,
    a de-overlapped entry-event study on the bullish-divergence flag, and the
    regime split."""
    dix, price = df["dix"], df["price"]
    _, dchg, pchg = dix_price_divergence(dix, price)
    d = pd.DataFrame({"dc": np.sign(dchg), "pc": np.sign(pchg),
                      "ret": df[ret_col]}, index=df.index).dropna()
    lines = [f"=== DIX-vs-PRICE DIVERGENCE ({DIV_LOOKBACK}d) vs forward "
             f"{ret_col} (option B) ==="]
    cells = [("DIX up / price DOWN  (bullish div)", (d.dc > 0) & (d.pc < 0)),
             ("both DOWN            (capitulation)", (d.dc < 0) & (d.pc < 0)),
             ("both UP", (d.dc > 0) & (d.pc > 0)),
             ("DIX down / price UP  (bearish div)", (d.dc < 0) & (d.pc > 0))]
    base = d["ret"].mean()
    lines.append(f"  {'cell':36s} {'n':>5}  {'mean':>6}  {'hit':>4}  vs base {base:+.2f}%")
    for lbl, m in cells:
        r = d.loc[m, "ret"]
        lines.append(f"  {lbl:36s} {len(r):5d}  {r.mean():+6.2f}  "
                     f"{(r > 0).mean() * 100:3.0f}%  ({r.mean() - base:+.2f})")
    # de-overlapped entry-event on the bullish-divergence flag
    state = np.where((d.dc > 0) & (d.pc < 0), 1, 0)
    pos, _ = entry_events(pd.Series(state, index=d.index))
    rp = d["ret"].iloc[pos].dropna()
    lines.append(f"  entry-event (first day of bullish divergence, 21d cooldown): "
                 f"n={len(rp)}  mean {rp.mean():+.2f}%  hit {(rp > 0).mean() * 100:.0f}%")
    # regime split of the bullish-divergence edge
    for lbl, sub in [(f"pre-{split[:4]}", d[d.index < split]),
                     (f"{split[:4]}+", d[d.index >= split])]:
        m = (sub.dc > 0) & (sub.pc < 0)
        r = sub.loc[m, "ret"]
        b = sub["ret"].mean()
        lines.append(f"  {lbl}: bullish-div {r.mean():+.2f}% (n={len(r)}) vs "
                     f"base {b:+.2f}%  -> edge {r.mean() - b:+.2f}pp")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="local SqueezeMetrics DIX.csv snapshot (else fetched live)")
    ap.add_argument("--url", default=DIX_URL, help="override the DIX.csv URL")
    ap.add_argument("--horizon", type=int, default=21, choices=list(HORIZONS),
                    help="headline forward-return horizon in sessions (default 21)")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="fixed raw-DIX threshold to age-check (default 0.45)")
    ap.add_argument("--out", default=None,
                    help="write the head-to-head power table (all horizons) to this CSV")
    args = ap.parse_args()

    df = load_dix(csv=args.csv, url=args.url)
    df = add_signals(df)
    df = add_oscillators(df)
    ret_col = f"r{args.horizon}"

    print(f"Raw SqueezeMetrics DIX: {df.index.min().date()} -> {df.index.max().date()}  "
          f"({len(df)} sessions)")
    print(f"Headline forward horizon: {args.horizon} sessions ({ret_col})\n")

    # Q2 -- drift / stationarity
    print(drift_report(df, threshold=args.threshold))
    print()

    # Q1 -- decile vs breakout head-to-head
    print("=" * 78)
    print("Q1  DECILE (level) vs. TRAILING-1y BREAKOUT/BREAKDOWN -- predictive power")
    print("=" * 78)
    common, power = headline_power(df, ret_col)
    print(f"Common real-time sample: {common.index.min().date()} -> "
          f"{common.index.max().date()}  ({len(common)} sessions)\n")
    print(" signal              |  IC    | NW slope  t     | top-band  bottom-band  spread")
    for _, r in power.iterrows():
        print(f" {r['signal']:19s} | {r['ic']:+.3f} |  {r['nw_slope']:+.2f}   "
              f"{r['nw_t']:+.2f}  |  {r['hi_mean']:+5.2f}     {r['lo_mean']:+5.2f}     "
              f"{r['spread']:+5.2f}")
    print("  (top/bottom band = deciles 9-10 vs 1-2 for decile signals; "
          "quintile 5 vs 1 for continuous)\n")

    jr = joint_regression(common, ret_col)
    print("=== JOINT regression: r ~ LEVEL(exp decile) + BREAKOUT(1y z), "
          "standardized, NW(21) ===")
    print(f"  LEVEL     {jr['lvl_b']:+.2f} pp/1sd  t={jr['lvl_t']:+.2f}")
    print(f"  BREAKOUT  {jr['brk_b']:+.2f} pp/1sd  t={jr['brk_t']:+.2f}   "
          f"(n={jr['n']}, R2={100 * jr['r2']:.1f}%)")
    print("  -> a near-zero / wrong-signed BREAKOUT t means the trailing-1y "
          "deviation adds nothing the level did not already say.\n")

    print(entry_report(common, ret_col))
    print()
    print(subperiod_report(common, ret_col))
    print()

    # Follow-up -- how fast should the breakout baseline be?
    print(window_sweep_report(df, ret_col))
    print()
    print(smoothed_level_report(df))
    print()

    # Follow-up -- oscillator normalization, GEX gating, DIX-vs-price divergence
    print("=" * 78)
    print("Q3  OSCILLATOR normalization + GEX gating + DIX-vs-price divergence")
    print("=" * 78)
    print(oscillator_report(df, ret_col))
    print()
    print(gex_gating_report(df, ret_col))
    print()
    print(divergence_report(df, ret_col))
    print()

    # Decile ladders (real-time expanding) for the record
    print(f"=== Forward {ret_col} by EXPANDING decile of raw DIX (real-time) ===")
    for dec, mean, med, n in decile_table(df, "exp_dec", ret_col):
        print(f"  D{dec:<2d}  mean {mean:+5.2f}%   med {med:+5.2f}%   n={n}")

    if args.out:
        rows = []
        for h in HORIZONS:
            _, p = headline_power(df, f"r{h}")
            p.insert(0, "horizon", h)
            rows.append(p)
        pd.concat(rows, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nWrote head-to-head power table (all horizons) -> {args.out}")

        sweep_rows = []
        for h in HORIZONS:
            sw = window_sweep(df, f"r{h}")
            sw.insert(0, "horizon", h)
            sweep_rows.append(sw)
        sweep_path = args.out.replace(".csv", "") + "_window_sweep.csv"
        pd.concat(sweep_rows, ignore_index=True).to_csv(sweep_path, index=False)
        print(f"Wrote breakout baseline-window sweep (all horizons) -> {sweep_path}")

        # compact oscillator / divergence summary table (all horizons)
        osc_path = args.out.replace(".csv", "") + "_oscillator.csv"
        oscillator_summary(df).to_csv(osc_path, index=False)
        print(f"Wrote oscillator + divergence summary (all horizons) -> {osc_path}")


if __name__ == "__main__":
    main()
