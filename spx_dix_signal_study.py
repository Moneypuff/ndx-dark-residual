#!/usr/bin/env python3
"""SPX DIX: absolute-decile ("level") signal vs trailing-1yr breakout/breakdown.

Question (from the dashboard user): the raw SPX DIX level is non-stationary -- its
"bullish floor" drifted up from ~0.44-0.45 being rare in 2019-2021 to being the
ordinary floor in 2024+. So does the classic ABSOLUTE-LEVEL decile signal (the
dashboard's "forward return by DIX decile" bars) actually predict SPX forward
returns, or does a signal measured RELATIVE to the trailing 1-year average --
a breakout above / breakdown below its own recent range -- carry more, and more
STABLE, predictive power?

Data: the reconstructed SPX dollar-DIX + SPY forward returns (1/2/3-month),
pulled straight from the built dashboard payload (docs/index.html) -- no network.

Signals compared (all daily), each scored against 21/42/63-day forward returns:
  A. abs        : raw DIX level (the absolute signal the decile bars encode)
  A'. abs_dec_LA: full-history decile of DIX  -- LOOK-AHEAD (dashboard bar method)
  A''.abs_dec_XP: expanding decile of DIX     -- no look-ahead (real-time abs level)
  B. trail_pct  : percentile of DIX within its trailing 252d  (no look-ahead)
  C. dev        : DIX - trailing 252d mean    (breakout distance, no look-ahead)
  D. z          : (DIX - trailing mean)/trailing std (standardized breakout)

Predictive power is judged by, per horizon:
  * Spearman IC (rank corr, sign of the edge)
  * Newey-West OLS slope t-stat  (return per +1 SD of signal, overlap-adjusted)
  * D10-D1 decile spread (mean fwd return, top signal-decile minus bottom)
  * breakout vs breakdown state means (DIX above vs below its 1yr average)
  * REGIME STABILITY: the yearly IC of each signal -- the crux of the question.
"""
import sys
import numpy as np
import pandas as pd

from index_comovement_study import load_payload, ols_nw, block_boot_ci

WIN = 252          # trailing window (~1 year of sessions)
MINP = 120         # min prior obs before a trailing signal is defined
HORIZONS = [21, 42, 63]
HLAB = {21: "1mo", 42: "2mo", 63: "3mo"}


# ---------------------------------------------------------------------------
# signal construction (every trailing signal uses PRIOR obs only -> no look-ahead)
# ---------------------------------------------------------------------------
def trailing_percentile(s, win=WIN, minp=MINP):
    """Percentile (0-1) of today's value within its PREVIOUS `win` observations."""
    v = s.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        lo = max(0, i - win)
        hist = v[lo:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < minp:
            continue
        out[i] = (hist < v[i]).mean() + 0.5 * (hist == v[i]).mean()
    return pd.Series(out, index=s.index)


def expanding_decile_series(s, minp=250):
    """Decile 1..10 of each value within its OWN full past (no look-ahead)."""
    v = s.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        hist = v[: i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < minp:
            continue
        pct = (hist < v[i]).mean() + 0.5 * (hist == v[i]).mean()
        out[i] = min(9, int(pct * 10)) + 1
    return pd.Series(out, index=s.index)


def full_decile(s):
    """Full-sample decile 1..10 (LOOK-AHEAD: cutpoints use the whole series)."""
    valid = s.dropna()
    q = pd.qcut(valid.rank(method="first"), 10, labels=False) + 1
    return q.reindex(s.index).astype(float)


def trailing_stats(s, win=WIN, minp=MINP):
    """Trailing mean/std over the PRIOR `win` obs (shifted to exclude today)."""
    prior = s.shift(1)
    m = prior.rolling(win, min_periods=minp).mean()
    sd = prior.rolling(win, min_periods=minp).std()
    return m, sd


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------
def spearman(x, y):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 30:
        return np.nan, 0
    return d["x"].corr(d["y"], method="spearman"), len(d)


def nw_slope(x, y, h):
    """OLS y ~ const + z(x) with Newey-West(lags=h). x standardized so the slope
    is 'mean fwd return (pp) per +1 SD of signal'. Returns (slope, t, p, r2, n)."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 60:
        return (np.nan,) * 4 + (len(d),)
    xz = (d["x"] - d["x"].mean()) / d["x"].std(ddof=0)
    X = np.column_stack([np.ones(len(d)), xz.to_numpy()])
    beta, se, t, p, r2 = ols_nw(d["y"].to_numpy(float), X, lags=h)
    return beta[1], t[1], p[1], r2, len(d)


def decile_spread(x, y, h, dec=None):
    """D10-D1 mean forward return using deciles of `x` (or a supplied decile
    series `dec`). Returns (spread, d1_mean, d10_mean, d1_ci, d10_ci, n)."""
    if dec is None:
        d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(d) < 100:
            return (np.nan,) * 5 + (len(d),)
        q = pd.qcut(d["x"].rank(method="first"), 10, labels=False) + 1
        d["dec"] = q.values
    else:
        d = pd.concat([dec.rename("dec"), y.rename("y")], axis=1).dropna()
        if len(d) < 100:
            return (np.nan,) * 5 + (len(d),)
    lo = d.loc[d["dec"] == d["dec"].min(), "y"]
    hi = d.loc[d["dec"] == d["dec"].max(), "y"]
    return (hi.mean() - lo.mean(), lo.mean(), hi.mean(),
            block_boot_ci(lo.to_numpy(), L=h), block_boot_ci(hi.to_numpy(), L=h), len(d))


def state_means(state, y, h):
    """Mean fwd return, hit rate, n and block-CI for each discrete state value."""
    d = pd.concat([state.rename("s"), y.rename("y")], axis=1).dropna()
    out = {}
    for val, g in d.groupby("s"):
        out[val] = dict(mean=g["y"].mean(), hit=(g["y"] > 0).mean(),
                        n=len(g), ci=block_boot_ci(g["y"].to_numpy(), L=h))
    return out


def diff_t(a, b, h):
    """Newey-West t for (mean(a) - mean(b)) via a pooled 0/1 group regression."""
    ya, yb = a.dropna().to_numpy(float), b.dropna().to_numpy(float)
    if len(ya) < 30 or len(yb) < 30:
        return np.nan
    y = np.concatenate([ya, yb])
    g = np.concatenate([np.ones(len(ya)), np.zeros(len(yb))])
    X = np.column_stack([np.ones(len(y)), g])
    _, _, t, _, _ = ols_nw(y, X, lags=h)
    return t[1]


# ---------------------------------------------------------------------------
def _ser(vals, idx):
    return pd.Series([np.nan if v is None else v for v in vals], index=idx)


def extract_universe(P, universe):
    """Return (dix, {h: fwd_return}) for 'SPX' | 'NDX' | 'IWM' from a payload."""
    if universe == "SPX":
        w = P["spx"]; idx = pd.to_datetime(w["dates"])
        return _ser(w["dix"], idx), {h: _ser(w[f"r{h}"], idx) for h in HORIZONS}
    if universe == "IWM":
        w = P["iwm"]; idx = pd.to_datetime(w["dates"])
        return _ser(w["d"], idx), {h: _ser(w[f"r{h}"], idx) for h in HORIZONS}
    if universe == "NDX":
        rel = P["rel"]; idx = pd.to_datetime(rel["dates"]); bench = P["bench"]
        return _ser(rel["ndx_dix"], idx), {h: _ser(rel[f"r{h}"][bench], idx) for h in HORIZONS}
    raise ValueError(universe)


def robustness(P):
    """Cross-index + subsample check: 3-month IC & NW-t for the absolute level vs
    the breakout deviation, full sample and post-2021 (drops the 2020-21 recovery)."""
    print("=" * 78)
    print("5) ROBUSTNESS -- 3-month horizon across indices & subsamples")
    print("=" * 78)
    print(f"{'universe / sample':<26}{'abs IC':>9}{'abs t':>8}{'dev IC':>9}{'dev t':>8}   winner")
    for u in ("SPX", "NDX", "IWM"):
        dix, rets = extract_universe(P, u)
        tmean, _ = trailing_stats(dix)
        dev = dix - tmean
        for lbl, mask in (("full", dix.notna()),
                          ("2022+", pd.Series(dix.index >= "2022-01-01", index=dix.index))):
            di, de, r = dix[mask], dev[mask], rets[63][mask]
            ic_a, _ = spearman(di, r); _, t_a, _, _, _ = nw_slope(di, r, 63)
            ic_d, _ = spearman(de, r); _, t_d, _, _, _ = nw_slope(de, r, 63)
            win = "breakout" if abs(t_d) > abs(t_a) else "abs-level"
            print(f"{u+' '+lbl:<26}{ic_a:>9.3f}{t_a:>8.1f}{ic_d:>9.3f}{t_d:>8.1f}   {win}")
    print("  abs = raw DIX level ; dev = DIX - trailing-1yr avg (breakout). t = NW t (lags=63).")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/index.html"
    P = load_payload(path)
    dix, rets = extract_universe(P, "SPX")
    idx = dix.index

    print(f"# SPX DIX signal study  ({path})")
    print(f"generated payload: {P.get('generated')}   demo={P.get('demo')}")
    print(f"DIX: {idx[0].date()} -> {idx[-1].date()}  n={dix.notna().sum()}  "
          f"level min/mean/max = {dix.min():.3f}/{dix.mean():.3f}/{dix.max():.3f}")
    yr = dix.groupby(dix.index.year).mean()
    print("mean DIX by year:", {int(k): round(v, 3) for k, v in yr.items()})
    print()

    # ---- build the signals -------------------------------------------------
    tmean, tstd = trailing_stats(dix)
    sig = {
        "abs (raw level)":            dix,
        "trail_pct (vs 1yr range)":   trailing_percentile(dix),
        "dev (DIX - 1yr avg)":        dix - tmean,
        "z (breakout, standardized)": (dix - tmean) / tstd,
    }
    # absolute-decile variants scored separately (they ARE the "decile method")
    abs_dec_LA = full_decile(dix)                     # look-ahead (dashboard bars)
    abs_dec_XP = expanding_decile_series(dix)          # real-time absolute decile

    # ---- 1) continuous predictive power per horizon -----------------------
    print("=" * 78)
    print("1) CONTINUOUS SIGNAL vs FORWARD RETURN  (Spearman IC ; NW slope per +1 SD)")
    print("=" * 78)
    hdr = f"{'signal':<28}" + "".join(f"{HLAB[h]:>22}" for h in HORIZONS)
    print(hdr)
    print(f"{'':<28}" + "".join(f"{'IC   slope(t)':>22}" for _ in HORIZONS))
    for name, s in sig.items():
        row = f"{name:<28}"
        for h in HORIZONS:
            ic, n = spearman(s, rets[h])
            slope, t, p, r2, _ = nw_slope(s, rets[h], h)
            star = "*" if (np.isfinite(p) and p < 0.05) else " "
            row += f"{ic:>6.3f} {slope:>+5.2f}({t:>+4.1f}){star:>1}"
        print(row)
    print("  IC = Spearman rank corr (sign of edge).  slope = pp fwd return per +1 SD "
          "of signal,\n  (t) = Newey-West t (lags=h, overlap-adjusted).  * = p<0.05.")
    print()

    # ---- 2) decile spreads: the 'decile method' three ways -----------------
    print("=" * 78)
    print("2) DECILE SPREAD  D10-D1 mean forward return (pp)  [top signal-decile - bottom]")
    print("=" * 78)
    print(f"{'method':<34}" + "".join(f"{HLAB[h]:>14}" for h in HORIZONS))
    dec_methods = [
        ("abs decile  LOOK-AHEAD (bars)", abs_dec_LA),
        ("abs decile  expanding (RT)",     abs_dec_XP),
        ("trail_pct decile (RT)",          None),   # deciles of the trailing-pct signal
        ("breakout dev decile (RT)",       None),
    ]
    dec_sig = {"trail_pct decile (RT)": sig["trail_pct (vs 1yr range)"],
               "breakout dev decile (RT)": sig["dev (DIX - 1yr avg)"]}
    for label, dser in dec_methods:
        row = f"{label:<34}"
        for h in HORIZONS:
            if dser is not None:
                sp, d1, d10, c1, c10, n = decile_spread(None, rets[h], h, dec=dser)
            else:
                sp, d1, d10, c1, c10, n = decile_spread(dec_sig[label], rets[h], h)
            row += f"{sp:>+13.2f} "
        print(row)
    print("  (absolute decile bars are the dashboard view; LOOK-AHEAD uses whole-sample "
          "cutpoints.)")
    print()

    # ---- 3) breakout vs breakdown states -----------------------------------
    print("=" * 78)
    print("3) BREAKOUT vs BREAKDOWN STATE  (DIX above / below its trailing 1yr average)")
    print("=" * 78)
    above = pd.Series(np.where(dix > tmean, "breakout(>1yr avg)", "breakdown(<1yr avg)"),
                      index=idx).where(tmean.notna())
    for h in HORIZONS:
        sm = state_means(above, rets[h], h)
        bo, bd = sm.get("breakout(>1yr avg)"), sm.get("breakdown(<1yr avg)")
        if not bo or not bd:
            continue
        t = diff_t(rets[h][above == "breakout(>1yr avg)"],
                   rets[h][above == "breakdown(<1yr avg)"], h)
        print(f"  {HLAB[h]}:  breakout mean {bo['mean']:+.2f}pp (hit {bo['hit']:.0%}, "
              f"n={bo['n']})   breakdown mean {bd['mean']:+.2f}pp (hit {bd['hit']:.0%}, "
              f"n={bd['n']})")
        print(f"        spread {bo['mean'] - bd['mean']:+.2f}pp   NW t={t:+.2f}   "
              f"uncond mean {rets[h].mean():+.2f}pp")
    print()

    # ---- 4) REGIME STABILITY: yearly IC ------------------------------------
    print("=" * 78)
    print("4) REGIME STABILITY -- Spearman IC by year (1mo horizon)  [the crux]")
    print("=" * 78)
    years = sorted({d.year for d in idx})
    stab = {"abs (raw level)": dix, "trail_pct (vs 1yr range)": sig["trail_pct (vs 1yr range)"],
            "dev (DIX - 1yr avg)": sig["dev (DIX - 1yr avg)"]}
    print(f"{'signal':<28}" + "".join(f"{y:>7}" for y in years) + f"{'|std|':>8}{'sign+':>7}")
    for name, s in stab.items():
        ics = []
        row = f"{name:<28}"
        for y in years:
            mask = pd.Series(idx.year == y, index=idx)
            ic, n = spearman(s[mask], rets[21][mask])
            ics.append(ic)
            row += f"{ic:>7.2f}" if np.isfinite(ic) else f"{'--':>7}"
        arr = np.array([v for v in ics if np.isfinite(v)])
        signpos = np.mean(arr > 0) if len(arr) else np.nan
        row += f"{arr.std():>8.2f}{signpos:>7.0%}"
        print(row)
    print("  |std| = dispersion of yearly IC (lower = more stable).  sign+ = share of "
          "years with IC>0\n  (consistency of the edge's direction).")
    # how much of the SPX edge is just 2020-2021?
    sub = pd.Series(idx >= "2022-01-01", index=idx)
    for name, s in [("abs (raw level)", dix), ("dev (DIX - 1yr avg)", sig["dev (DIX - 1yr avg)"])]:
        ic_f, _ = spearman(s, rets[63]); ic_s, _ = spearman(s[sub], rets[63][sub])
        print(f"  {name:<28} 3mo IC  full={ic_f:+.3f}   2022+={ic_s:+.3f}")
    print()

    robustness(P)

    print("Legend: pp = percentage points of the index ETF forward return.  RT = real-time "
          "(no look-ahead).")


if __name__ == "__main__":
    main()
