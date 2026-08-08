#!/usr/bin/env python3
"""
GEX-regime study: bucket GEX into quintiles and read DIX within each regime,
on volatility-adjusted SPX forward returns
==============================================================================

Premise (per request): in a downtrend GEX tends to be low/negative anyway, so
instead of conditioning the DIX indicator on a PRICE trend, condition it on the
GEX REGIME -- lowest quintile = the negative-gamma regime, top quintile = ultra
high positive gamma. SqueezeMetrics publishes a DIX x GEX heatmap; this rebuilds
the exploration from raw data with the repo's no-look-ahead discipline.

Construction
------------
* GEX quintile: trailing-252-session percentile rank of raw GEX (handles the
  series' secular growth; no look-ahead), cut at 20/40/60/80 -> Q1..Q5. Q1 is
  verified to be the negative-gamma regime (share of GEX<0 reported per
  bucket), and a raw sign split (GEX<0 vs GEX>=0) is reported as robustness.
* DIX reading: stochastic %D(126) of DIX (the surviving normalization), cut
  into bands D1..D5 at 20/40/60/80 (already 0-100, self-normalized).
* Outcome: forward 21/42-session SPX return, VOL-ADJUSTED two ways --
  z_rv  = fwd / (trailing-21d realized daily vol * sqrt(h))   [sigma units]
  z_vix = fwd / (VIX/100 * sqrt(h/252))                        [sigma units]
  (raw % also reported). Vol is known at t; no look-ahead.

Tests
-----
1. GEX-quintile main effect (raw and sigma units) + premise check: share of
   3-month price downtrends per GEX bucket.
2. The 5x5 heatmap: mean sigma-unit forward return by GEX quintile x DIX band.
3. DIX usefulness BY regime: Spearman IC of %D vs vol-adjusted forward return
   within each GEX quintile, with moving-block-bootstrap CIs -- the cleanest
   "does DIX fit better in a given GEX regime" statistic.
4. The optimized entry overlay (%D(126) crossing >=70, hold 21) grouped by the
   GEX quintile on the entry day.

Reproduce
---------
    python spx_gex_regime_study.py --out spx_gex_regime.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import spx_dix_decile_vs_breakout_study as B   # load_dix, stochastic_osc, spearman_ic, block_boot_ci
import spx_xs_dip_dix_study as XS              # trend_slope

PCT_WIN, PCT_MIN = 252, 200
HORIZONS = (21, 42)
BANDS = (20, 40, 60, 80)
ENTRY_THR, ENTRY_HOLD = 70, 21


def trailing_pct(s, win=PCT_WIN, min_obs=PCT_MIN):
    """Trailing percentile rank (0..100) of today's value within the prior
    `win` sessions (today excluded from the comparison set)."""
    return s.rolling(win, min_periods=min_obs).apply(
        lambda x: (x[:-1] < x[-1]).mean() * 100.0, raw=True)


def to_band(pct):
    """0..100 -> 1..5 at cuts 20/40/60/80."""
    return np.clip(np.floor(np.asarray(pct, dtype=float) / 20.0), 0, 4) + 1


def build_frame(df, cache_dir=None, vix=None):
    out = df.copy()
    out["gexpct"] = trailing_pct(out["gex"])
    out["gexq"] = pd.Series(to_band(out["gexpct"]), index=out.index).where(out["gexpct"].notna())
    out["pd126"] = B.stochastic_osc(out["dix"], win=126)
    out["dband"] = pd.Series(to_band(out["pd126"]), index=out.index).where(out["pd126"].notna())
    out["rv21"] = out["price"].pct_change().rolling(21, min_periods=15).std()
    out["trend63"] = XS.trend_slope(pd.DataFrame({"SPX": out["price"]}), 63)["SPX"]
    if vix is not None:
        out["vix"] = vix.reindex(out.index)
    for h in HORIZONS:
        fwd = out["price"].shift(-h) / out["price"] - 1.0
        out[f"fwd{h}"] = fwd * 100.0
        out[f"zrv{h}"] = fwd / (out["rv21"] * np.sqrt(h))
        if vix is not None:
            out[f"zvix{h}"] = fwd / (out["vix"] / 100.0 * np.sqrt(h / 252.0))
    return out


def _stat(a, seed=0):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 30:
        return {"n": int(len(a)), "mean": np.nan, "ci": None}
    ci = B.block_boot_ci(a, seed=seed)
    return {"n": int(len(a)), "mean": round(float(a.mean()), 3),
            "hit": round(float((a > 0).mean() * 100), 1),
            "ci": (round(ci[0], 3), round(ci[1], 3)) if np.isfinite(ci[0]) else None}


def gex_profile(d, h):
    rows = []
    for q in range(1, 6):
        m = d["gexq"] == q
        rows.append({
            "q": q, "n": int(m.sum()),
            "gex_mean": round(float(d.loc[m, "gex"].mean()), 2),
            "neg_pct": round(float((d.loc[m, "gex"] < 0).mean() * 100), 1),
            "down_pct": round(float((d.loc[m, "trend63"] < 0).mean() * 100), 1),
            "raw": _stat(d.loc[m, f"fwd{h}"], seed=q),
            "zrv": _stat(d.loc[m, f"zrv{h}"], seed=10 + q),
            "zvix": _stat(d.loc[m, f"zvix{h}"], seed=20 + q) if f"zvix{h}" in d else None,
        })
    return rows


def sign_split(d, h):
    rows = []
    for lab, m in [("GEX<0", d["gex"] < 0), ("GEX>=0", d["gex"] >= 0)]:
        rows.append({"lab": lab, "n": int(m.sum()),
                     "raw": _stat(d.loc[m, f"fwd{h}"], seed=31),
                     "zrv": _stat(d.loc[m, f"zrv{h}"], seed=32)})
    return rows


def heatmap(d, h, out_col=None):
    """5x5 grid: mean sigma-unit forward return (and n) by GEX quintile (rows)
    x DIX %D band (cols)."""
    col = out_col or f"zrv{h}"
    grid_mean = np.full((5, 5), np.nan)
    grid_n = np.zeros((5, 5), dtype=int)
    for qi in range(1, 6):
        for di in range(1, 6):
            v = d.loc[(d["gexq"] == qi) & (d["dband"] == di), col].dropna()
            grid_n[qi - 1, di - 1] = len(v)
            if len(v) >= 30:
                grid_mean[qi - 1, di - 1] = float(v.mean())
    return grid_mean, grid_n


def _ic_boot_ci(sig, ret, B_=1000, L=21, seed=0):
    """Moving-block bootstrap CI for the Spearman IC of paired daily series."""
    s = np.asarray(sig, dtype=float)
    r = np.asarray(ret, dtype=float)
    m = np.isfinite(s) & np.isfinite(r)
    s, r = s[m], r[m]
    n = len(s)
    if n < 60:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    vals = []
    for _ in range(B_):
        starts = rng.integers(0, n - L + 1, nb)
        idx = np.concatenate([np.arange(st, st + L) for st in starts])[:n]
        ic, _ = B.spearman_ic(s[idx], r[idx])
        if np.isfinite(ic):
            vals.append(ic)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def ic_by_regime(d, h, out_col=None):
    col = out_col or f"zrv{h}"
    rows = []
    for q in range(1, 6):
        m = d["gexq"] == q
        ic, n = B.spearman_ic(d.loc[m, "pd126"], d.loc[m, col])
        lo, hi = _ic_boot_ci(d.loc[m, "pd126"], d.loc[m, col], seed=40 + q)
        rows.append({"q": q, "n": n, "ic": round(ic, 3) if np.isfinite(ic) else np.nan,
                     "ci": (round(lo, 3), round(hi, 3)) if np.isfinite(lo) else None})
    ic, n = B.spearman_ic(d["pd126"], d[col])
    lo, hi = _ic_boot_ci(d["pd126"], d[col], seed=49)
    rows.append({"q": "all", "n": n, "ic": round(ic, 3),
                 "ci": (round(lo, 3), round(hi, 3)) if np.isfinite(lo) else None})
    return rows


def entry_by_regime(d, h):
    """The optimized entry (%D crossing >= ENTRY_THR): forward outcomes grouped
    by the GEX quintile on the entry day."""
    act = (d["pd126"] >= ENTRY_THR).fillna(False)
    ent = act & ~act.shift(1, fill_value=False)
    rows = []
    for grp, m in [("Q1-2 (low/neg)", d["gexq"] <= 2), ("Q3", d["gexq"] == 3),
                   ("Q4-5 (high)", d["gexq"] >= 4)]:
        e = d[ent & m]
        rows.append({"grp": grp, "n": int(len(e)),
                     "raw": _stat(e[f"fwd{h}"], seed=51),
                     "zrv": _stat(e[f"zrv{h}"], seed=52)})
    return rows


def _ci(ci):
    return f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "--"


def report(d, has_vix):
    out = []
    for h in HORIZONS:
        out.append("\n" + "=" * 96)
        out.append(f"HORIZON {h}d")
        out.append("=" * 96)
        out.append("1) GEX QUINTILE main effect (trailing-1yr percentile buckets; "
                   "zrv = sigma-units vs 21d realized vol)")
        hdr = f"   {'Q':>2} {'n':>5} {'meanGEX':>8} {'%neg':>5} {'%down':>6}  " \
              f"{'raw %':>14} {'zrv (sigma)':>16}"
        if has_vix:
            hdr += f" {'zvix (sigma)':>16}"
        out.append(hdr)
        for r in gex_profile(d, h):
            line = (f"   {r['q']:>2} {r['n']:>5} {r['gex_mean']:>8} {r['neg_pct']:>5} "
                    f"{r['down_pct']:>6}  {r['raw']['mean']:+6.2f} {_ci(r['raw']['ci'])} "
                    f"{r['zrv']['mean']:+6.3f} {_ci(r['zrv']['ci'])}")
            if has_vix and r["zvix"]:
                line += f" {r['zvix']['mean']:+6.3f} {_ci(r['zvix']['ci'])}"
            out.append(line)
        out.append("   sign split (robustness): " + " | ".join(
            f"{r['lab']} n={r['n']} raw {r['raw']['mean']:+.2f}% zrv {r['zrv']['mean']:+.3f}"
            for r in sign_split(d, h)))
        out.append(f"\n2) HEATMAP -- mean sigma-unit fwd {h}d (rows = GEX Q1 low/neg .. Q5 ultra-high; "
                   "cols = %D(126) band D1 dark-light .. D5 dark-heavy)")
        gm, gn = heatmap(d, h)
        out.append("        " + "".join(f"{f'D{i}':>9}" for i in range(1, 6)))
        for qi in range(5):
            cells = "".join(f"{gm[qi, di]:+9.3f}" if np.isfinite(gm[qi, di]) else f"{'--':>9}"
                            for di in range(5))
            out.append(f"   Q{qi + 1}  {cells}    n=" +
                       ",".join(str(gn[qi, di]) for di in range(5)))
        out.append(f"\n3) DIX IC BY REGIME -- Spearman(%D(126), zrv{h}) within each GEX quintile")
        for r in ic_by_regime(d, h):
            out.append(f"   Q{r['q']}: IC {r['ic']:+.3f} {_ci(r['ci'])}  (n={r['n']})")
        out.append(f"\n4) OPTIMIZED ENTRY (%D crossing >={ENTRY_THR}) by GEX regime on entry day")
        for r in entry_by_regime(d, h):
            out.append(f"   {r['grp']:14s} n={r['n']:>3}  raw {r['raw']['mean']:+.2f}% "
                       f"{_ci(r['raw']['ci'])}  hit {r['raw'].get('hit', float('nan'))}%  "
                       f"zrv {r['zrv']['mean']:+.3f} {_ci(r['zrv']['ci'])}")
    return "\n".join(out)


def summary_rows(d, has_vix):
    rows = []
    for h in HORIZONS:
        for r in gex_profile(d, h):
            rows.append({"horizon": h, "test": "gex_quintile", "gexq": r["q"], "dband": None,
                         "n": r["n"], "neg_pct": r["neg_pct"], "down_pct": r["down_pct"],
                         "raw": r["raw"]["mean"], "zrv": r["zrv"]["mean"],
                         "zvix": r["zvix"]["mean"] if (has_vix and r["zvix"]) else None,
                         "ci_lo": (r["zrv"]["ci"] or [None, None])[0],
                         "ci_hi": (r["zrv"]["ci"] or [None, None])[1]})
        gm, gn = heatmap(d, h)
        for qi in range(5):
            for di in range(5):
                rows.append({"horizon": h, "test": "heatmap", "gexq": qi + 1, "dband": di + 1,
                             "n": int(gn[qi, di]), "neg_pct": None, "down_pct": None,
                             "raw": None, "zrv": round(gm[qi, di], 3) if np.isfinite(gm[qi, di]) else None,
                             "zvix": None, "ci_lo": None, "ci_hi": None})
        for r in ic_by_regime(d, h):
            rows.append({"horizon": h, "test": "ic_by_regime", "gexq": r["q"], "dband": None,
                         "n": r["n"], "neg_pct": None, "down_pct": None, "raw": None,
                         "zrv": r["ic"], "zvix": None,
                         "ci_lo": (r["ci"] or [None, None])[0],
                         "ci_hi": (r["ci"] or [None, None])[1]})
        for r in entry_by_regime(d, h):
            rows.append({"horizon": h, "test": "entry_by_regime", "gexq": r["grp"], "dband": None,
                         "n": r["n"], "neg_pct": None, "down_pct": None,
                         "raw": r["raw"]["mean"], "zrv": r["zrv"]["mean"], "zvix": None,
                         "ci_lo": (r["zrv"]["ci"] or [None, None])[0],
                         "ci_hi": (r["zrv"]["ci"] or [None, None])[1]})
    return pd.DataFrame(rows)


def load_vix(start, end, cache_dir):
    try:
        pan = N.load_yahoo_panels(["^VIX"], start, end, workers=1,
                                  cache_dir=cache_dir, label="VIX")
        return pan["close"]["^VIX"]
    except Exception as e:                       # noqa: BLE001 -- VIX is optional
        print(f"(VIX fetch failed: {e} -- proceeding with realized vol only)")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="local DIX.csv snapshot")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--no-vix", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = B.load_dix(csv=args.csv)
    vix = None if args.no_vix else load_vix(df.index[0], df.index[-1] + pd.Timedelta(days=1),
                                            args.cache_dir)
    d = build_frame(df, vix=vix)
    d = d[d["gexq"].notna() & d["dband"].notna()]
    has_vix = vix is not None
    print(f"DIX/GEX {d.index[0].date()} .. {d.index[-1].date()}  ({len(d)} sessions with "
          f"both buckets defined)  VIX: {'yes' if has_vix else 'no'}")
    print(f"GEX<0 share overall: {(d['gex'] < 0).mean() * 100:.1f}%")
    print(report(d, has_vix))
    if args.out:
        summary_rows(d, has_vix).to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
