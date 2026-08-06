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


def add_signals(df):
    dix = df["dix"]
    rm = dix.rolling(TRAIL_WIN, min_periods=TRAIL_MIN).mean()
    rs = dix.rolling(TRAIL_WIN, min_periods=TRAIL_MIN).std()
    df["breakout_z"] = (dix - rm) / rs            # trailing-1y sigmas above/below own average
    df["breakout_gap"] = (dix - rm) * 100.0       # raw gap vs trailing average, pp
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
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(args.out, index=False)
        print(f"\nWrote head-to-head power table (all horizons) -> {args.out}")


if __name__ == "__main__":
    main()
