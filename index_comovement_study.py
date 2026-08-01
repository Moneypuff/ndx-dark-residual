#!/usr/bin/env python3
"""
Cross-index DIX comovement -> 1-month forward return study.
===========================================================

Question (per request): using a 5-day moving average of the dollar-DIX for all
three index gauges (NDX-100, S&P 500, Russell 2000 / IWM), how do the indices'
own 1-month forward returns behave when the three DIX gauges COMOVE vs. DIVERGE?
For example: when SPX-DIX and NDX-DIX sit in their lower deciles while IWM-DIX
sits in its upper deciles, what are the subsequent 1-month returns of each index?

Signal (per index, no look-ahead in the DIX itself)
---------------------------------------------------
DIX_t = dollar-weighted Sum($ short vol) / Sum($ off-exchange vol) across the
index constituents each day (SqueezeMetrics' construction; see
`compute_dollar_dix` in ndx_dark_residual.py). We smooth it with a 5-day moving
average (the same 5d MA the dashboard uses as its residual benchmark):

    DIX5_t = mean(DIX over the 5 trading days ending t)      # min 3 obs

Each index's DIX5 is then bucketed into deciles and the deciles collapsed to
three regimes per index:

    Low  = deciles 1-3   (DIX5 in the bottom 30% of its own history)
    Mid  = deciles 4-7
    High = deciles 8-10  (DIX5 in the top 30% of its own history)

Two decile bases are reported side by side:
  * full-sample (the original construction; mild look-ahead), and
  * EXPANDING-WINDOW (each day ranked only against its own past, min 250 obs)
    -- the cutoffs a live trader could actually have known.

Outcome (per index)
-------------------
The 1-month (21 trading day) forward return of each index's own price proxy:
QQQ for NDX, SPY for SPX, IWM for the Russell 2000 -- exactly the forward
returns already packed into the dashboard payload (`compute_forward_return`,
in percent). A regime observed on date t is scored by the return realised over
the following 21 sessions, so there is no look-ahead in the outcome either.

Beyond the 27-regime table the study now also reports:
  * a TWO-FACTOR regression -- joint dark-flow LEVEL (mean of the three DIX5
    z-scores) and the large-vs-small SPREAD ((zNDX+zSPX)/2 - zIWM) -- with
    Newey-West (HAC, 21-lag) standard errors. This uses every observation,
    drops the arbitrary Low/Mid/High binning, and directly tests the study's
    own conclusion that divergence (not level) carries the signal;
  * a REGIME-ENTRY event study (first day a regime appears, 21-session
    cool-down) -- overlap-free-ish counts of what happens after the setup
    forms, rather than every day sitting inside it;
  * BLOCK-BOOTSTRAP 95% CIs (21-day moving blocks) on each regime's mean
    forward return, in the regime CSV and printed for the headline regimes;
  * a SECTOR dark-flow gauge: defensive (XLP/XLU/XLV) minus cyclical
    (XLI/XLF/XLE/XLB) DIX z-spread vs the indices' forward returns (needs a
    dashboard payload that carries the sector series, i.e. any current build);
  * an OUT-OF-SAMPLE split: regime cutoffs and factor loadings fitted on
    pre-2024 data only, evaluated on 2024+.

Data source
-----------
The DIX series and their 1-month forward returns are read straight from the
JSON payload embedded in the built dashboard (`docs/index.html`) so the study
is reproducible offline against the latest refresh, with no live FINRA
re-fetch. Both payload encodings are supported: the legacy plain `const P =
{...}` blob and the current compressed `const PZ = "<base64 deflate>"` form.

    NDX:  P.rel.ndx_dix              vs  P.rel.r21[P.bench]   (QQQ)
    SPX:  P.spx.dix                  vs  P.spx.r21            (SPY)
    IWM:  P.iwm.d                    vs  P.iwm.r21            (IWM)

Usage
-----
    python index_comovement_study.py                       # text report
    python index_comovement_study.py --html docs/index.html --csv out.csv
    python index_comovement_study.py --basis expanding     # live-tradable cutoffs
"""
import argparse
import base64
import json
import math
import re
import sys
import zlib

import numpy as np
import pandas as pd

IDX = ["NDX", "SPX", "IWM"]
BOOT_B = 2000          # bootstrap replications
BOOT_L = 21            # moving-block length (= the forward-return horizon)
EXP_MIN = 250          # min history before an expanding-window decile is defined
OOS_SPLIT = "2024-01-01"
DEFENSIVE = ["XLP", "XLU", "XLV"]
CYCLICAL = ["XLI", "XLF", "XLE", "XLB"]


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
    # below ~2 blocks the cyclic resample degenerates (every draw is a rotation
    # of the whole series -> zero-width CI), so report no CI instead of a fake one
    if n < 2 * L:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    means = r[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, levels))


def expanding_decile(s, min_obs=EXP_MIN):
    """Decile (1..10) of each value within the series' OWN history up to and
    including that day (mid-rank on ties). NaN until `min_obs` observations
    have accrued -- these are the cutoffs a live user could have known."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(min_obs - 1, len(v)):
        hist = v[: i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_obs or not np.isfinite(v[i]):
            continue
        pct = (hist < v[i]).mean() + 0.5 * (hist == v[i]).mean()
        out[i] = min(9, int(pct * 10)) + 1
    return pd.Series(out, index=s.index)


def zone_from_dec(dec):
    return np.where(dec <= 3, "Low", np.where(dec >= 8, "High", "Mid"))


# ----------------------------------------------------------------------------
# Frame construction
# ----------------------------------------------------------------------------
def build_aligned(P, ma_window=5, min_periods=3, basis="full"):
    """Aligned frame of each index's DIX 5d MA + its 1-month forward return (%),
    restricted to the dates on which all three DIX gauges are available.
    `basis` picks the decile construction: 'full' (full-sample qcut, mild
    look-ahead) or 'expanding' (each day vs its own past only)."""
    def series(dates, dix, ret):
        return pd.DataFrame(
            {"date": pd.to_datetime(dates), "dix": dix, "r1m": ret}
        ).set_index("date")

    raw = {
        "NDX": series(P["rel"]["dates"], P["rel"]["ndx_dix"], P["rel"]["r21"][P["bench"]]),
        "SPX": series(P["spx"]["dates"], P["spx"]["dix"], P["spx"]["r21"]),
        "IWM": series(P["iwm"]["dates"], P["iwm"]["d"], P["iwm"]["r21"]),
    }
    cols = {}
    for name, df in raw.items():
        cols[name + "_dix5"] = df["dix"].rolling(ma_window, min_periods=min_periods).mean()
        cols[name + "_r1m"] = df["r1m"]  # already in percent
    A = pd.DataFrame(cols).sort_index()
    A = A[A[[f"{i}_dix5" for i in IDX]].notna().all(axis=1)].copy()

    for i in IDX:
        if basis == "expanding":
            A[i + "_dec"] = expanding_decile(A[i + "_dix5"])
        else:
            A[i + "_dec"] = pd.qcut(A[i + "_dix5"], 10, labels=False, duplicates="drop") + 1
        A[i + "_z"] = np.where(A[i + "_dec"].isna(), "NA", zone_from_dec(A[i + "_dec"]))
        zz = (A[i + "_dix5"] - A[i + "_dix5"].mean()) / A[i + "_dix5"].std(ddof=0)
        A[i + "_zs"] = zz          # full-sample z-score (for the factor regression)
    if basis == "expanding":
        A = A[(A[[f"{i}_z" for i in IDX]] != "NA").all(axis=1)].copy()
    A["code"] = A["NDX_z"].str[0] + A["SPX_z"].str[0] + A["IWM_z"].str[0]
    A["LEVEL"] = A[[f"{i}_zs" for i in IDX]].mean(axis=1)
    A["SPREAD"] = (A["NDX_zs"] + A["SPX_zs"]) / 2.0 - A["IWM_zs"]
    return A


def regime_stats(A, mask, label, with_ci=False, seed=0):
    sub = A[mask]
    out = {"regime": label, "n_days": int(len(sub))}
    for i in IDX:
        r = sub[i + "_r1m"].dropna()
        out[i + "_mean"] = round(float(r.mean()), 2) if len(r) else np.nan
        out[i + "_med"] = round(float(r.median()), 2) if len(r) else np.nan
        out[i + "_hit"] = round(float((r > 0).mean() * 100), 0) if len(r) else np.nan
        if with_ci:
            lo, hi = block_boot_ci(r.to_numpy(), seed=seed)
            out[i + "_ci_lo"] = round(lo, 2) if np.isfinite(lo) else np.nan
            out[i + "_ci_hi"] = round(hi, 2) if np.isfinite(hi) else np.nan
    out["ret_n"] = int(sub[[f"{i}_r1m" for i in IDX]].notna().all(axis=1).sum())
    return out


def all_regimes(A, with_ci=True):
    rows = []
    seed = 0
    for sn in ("Low", "Mid", "High"):
        for ss in ("Low", "Mid", "High"):
            for si in ("Low", "Mid", "High"):
                m = (A["NDX_z"] == sn) & (A["SPX_z"] == ss) & (A["IWM_z"] == si)
                if m.sum() == 0:
                    continue
                seed += 1
                rows.append(regime_stats(A, m, f"N={sn},S={ss},I={si}",
                                         with_ci=with_ci, seed=seed))
    return pd.DataFrame(rows).sort_values("n_days", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# New analyses
# ----------------------------------------------------------------------------
def two_factor_report(A, label="full sample"):
    """r1m_i ~ b0 + b1*LEVEL + b2*SPREAD with Newey-West(21) errors, per index."""
    lines = [f"=== TWO-FACTOR REGRESSION ({label}) ===",
             "  r1m = b0 + b1*LEVEL + b2*SPREAD   "
             "(LEVEL = mean DIX5 z of the 3 gauges; SPREAD = (zNDX+zSPX)/2 - zIWM;",
             "   Newey-West HAC t-stats, 21 lags -- the honest test of "
             "'divergence, not level, carries the signal')"]
    for i in IDX:
        d = A[["LEVEL", "SPREAD", i + "_r1m"]].dropna()
        if len(d) < 100:
            lines.append(f"  {i}: insufficient data (n={len(d)})")
            continue
        y = d[i + "_r1m"].to_numpy()
        X = np.column_stack([np.ones(len(d)), d["LEVEL"].to_numpy(), d["SPREAD"].to_numpy()])
        beta, se, t, p, r2 = ols_nw(y, X)
        lines.append(
            f"  {i}:  n={len(d):4d}   const {beta[0]:+.2f}   "
            f"LEVEL {beta[1]:+.2f} (t={t[1]:+.2f}, p={p[1]:.3f})   "
            f"SPREAD {beta[2]:+.2f} (t={t[2]:+.2f}, p={p[2]:.3f})   R2={100*r2:.1f}%")
    return "\n".join(lines)


def entry_events(A, code, min_gap=21):
    """First-day-of-regime events with a `min_gap`-session cool-down between
    counted entries. Returns (entry_dates, stats-dict per index)."""
    mask = (A["code"] == code).to_numpy()
    entries = []
    last_pos = None
    for pos in range(len(mask)):
        if mask[pos] and (pos == 0 or not mask[pos - 1]):
            if last_pos is None or pos - last_pos >= min_gap:
                entries.append(pos)
                last_pos = pos
    dates = A.index[entries]
    stats = {}
    for i in IDX:
        r = A[i + "_r1m"].iloc[entries].dropna()
        stats[i] = {
            "n": int(len(r)),
            "mean": round(float(r.mean()), 2) if len(r) else np.nan,
            "med": round(float(r.median()), 2) if len(r) else np.nan,
            "hit": round(float((r > 0).mean() * 100)) if len(r) else np.nan,
        }
    return dates, stats


def entry_report(A, codes=("LLH", "HML", "MML", "HHL", "LLL", "HHH")):
    name = {"L": "Low", "M": "Mid", "H": "High"}
    lines = ["=== REGIME-ENTRY EVENT STUDY (first day a regime appears; "
             "21-session cool-down) ===",
             "  Every-day regime stats overlap heavily; entries are the "
             "closest thing to independent setups."]
    for code in codes:
        dates, st = entry_events(A, code)
        if not len(dates):
            continue
        lbl = f"N={name[code[0]]},S={name[code[1]]},I={name[code[2]]}"
        lines.append(f"  {lbl:22s} entries={len(dates):3d}  " + "   ".join(
            f"{i} {st[i]['mean']:+.2f}% (med {st[i]['med']:+.2f}, hit {st[i]['hit']}%)"
            if st[i]["n"] else f"{i} --"
            for i in IDX) + f"   [last entry {dates[-1].date()}]")
    return "\n".join(lines)


def sector_gauge_report(P):
    """Defensive-minus-cyclical dark-flow z-spread vs the indices' 1-month
    forward returns. Sector DIX series come from the dashboard's sector payload
    (5d-MA reconstructed dollar-DIX per SPDR fund)."""
    S = P.get("sectors")
    if not S or not S.get("items"):
        return "=== SECTOR DARK-FLOW GAUGE ===\n  (payload has no sector series -- rebuild the dashboard)"
    dates = pd.to_datetime(S["dates"])
    ser = {}
    for it in S["items"]:
        if it["etf"] in DEFENSIVE + CYCLICAL:
            ser[it["etf"]] = pd.Series(
                [np.nan if v is None else v for v in it["series"]],
                index=dates, dtype="float64")
    have_d = [e for e in DEFENSIVE if e in ser]
    have_c = [e for e in CYCLICAL if e in ser]
    if len(have_d) < 2 or len(have_c) < 2:
        return ("=== SECTOR DARK-FLOW GAUGE ===\n"
                f"  insufficient sector coverage (defensive {have_d}, cyclical {have_c})")

    def zmean(etfs):
        return pd.DataFrame({e: (ser[e] - ser[e].mean()) / ser[e].std(ddof=0)
                             for e in etfs}).mean(axis=1)

    spread = (zmean(have_d) - zmean(have_c)).rename("DEFCYC")
    rets = {
        "NDX": pd.Series(P["rel"]["r21"][P["bench"]],
                         index=pd.to_datetime(P["rel"]["dates"]), dtype="float64"),
        "SPX": pd.Series(P["spx"]["r21"], index=pd.to_datetime(P["spx"]["dates"]),
                         dtype="float64"),
        "IWM": pd.Series(P["iwm"]["r21"], index=pd.to_datetime(P["iwm"]["dates"]),
                         dtype="float64"),
    }
    lines = ["=== SECTOR DARK-FLOW GAUGE: defensive minus cyclical DIX z-spread ===",
             f"  defensive = mean z of {'/'.join(have_d)}   "
             f"cyclical = mean z of {'/'.join(have_c)}   (5d-MA sector DIX, "
             f"{len(spread.dropna())} days)",
             "  positive spread = dark accumulation rotating into defensives "
             "(classic risk-off tell if it leads weakness)"]
    for i in IDX:
        d = spread.to_frame("DEFCYC").join(rets[i].rename("r"), how="inner").dropna()
        if len(d) < 100:
            lines.append(f"  {i}: insufficient overlap")
            continue
        y = d["r"].to_numpy()
        X = np.column_stack([np.ones(len(d)), d["DEFCYC"].to_numpy()])
        beta, se, t, p, r2 = ols_nw(y, X)
        # tercile cut of the spread for a non-parametric read
        q1, q2 = d["DEFCYC"].quantile([1 / 3, 2 / 3])
        lo = d[d["DEFCYC"] <= q1]["r"]
        hi = d[d["DEFCYC"] >= q2]["r"]
        lines.append(
            f"  {i}:  slope {beta[1]:+.2f}%/z (NW t={t[1]:+.2f}, p={p[1]:.3f})   "
            f"defensive-tilt tercile {hi.mean():+.2f}% vs cyclical-tilt {lo.mean():+.2f}% "
            f"(spread {hi.mean()-lo.mean():+.2f}pp)")
    return "\n".join(lines)


def oos_report(A_full):
    """Fit regime cutoffs (30th/70th pct of DIX5) and the two-factor loadings on
    pre-OOS_SPLIT data only; evaluate on the rest."""
    tr = A_full[A_full.index < OOS_SPLIT]
    te = A_full[A_full.index >= OOS_SPLIT].copy()
    lines = [f"=== OUT-OF-SAMPLE SPLIT (fit < {OOS_SPLIT}, evaluate >= {OOS_SPLIT}) ===",
             f"  train n={len(tr)}   test n={len(te)}"]
    if len(tr) < 300 or len(te) < 100:
        return "\n".join(lines + ["  insufficient data for the split"])

    # regimes in the test window using TRAIN cutoffs only
    for i in IDX:
        lo_c, hi_c = tr[i + "_dix5"].quantile([0.30, 0.70])
        te[i + "_z_oos"] = np.where(te[i + "_dix5"] <= lo_c, "Low",
                                    np.where(te[i + "_dix5"] >= hi_c, "High", "Mid"))
    te["code_oos"] = (te["NDX_z_oos"].str[0] + te["SPX_z_oos"].str[0]
                      + te["IWM_z_oos"].str[0])

    def stats_line(sub, lbl):
        parts = []
        for i in IDX:
            r = sub[i + "_r1m"].dropna()
            parts.append(f"{i} {r.mean():+.2f}% (hit {100*(r>0).mean():.0f}%)"
                         if len(r) else f"{i} --")
        return f"  {lbl:34s} n={len(sub):4d}   " + "   ".join(parts)

    lines.append(stats_line(te, "test baseline (all days)"))
    feat = te[te["code_oos"] == "LLH"]
    lines.append(stats_line(feat, "requested divergence (train cutoffs)")
                 if len(feat) else
                 "  requested divergence (N,S Low + I High): 0 test days under train cutoffs")

    # two-factor: z-params and betas from train; evaluate on test
    mu = {i: tr[i + "_dix5"].mean() for i in IDX}
    sd = {i: tr[i + "_dix5"].std(ddof=0) for i in IDX}
    zt = {i: (te[i + "_dix5"] - mu[i]) / sd[i] for i in IDX}
    lvl_t = sum(zt.values()) / 3.0
    spr_t = (zt["NDX"] + zt["SPX"]) / 2.0 - zt["IWM"]
    ztr = {i: (tr[i + "_dix5"] - mu[i]) / sd[i] for i in IDX}
    lvl_r = sum(ztr.values()) / 3.0
    spr_r = (ztr["NDX"] + ztr["SPX"]) / 2.0 - ztr["IWM"]
    lines.append("  two-factor fit on train -> applied to test "
                 "(pred = b0 + b1*LEVEL + b2*SPREAD):")
    for i in IDX:
        dtr = pd.DataFrame({"L": lvl_r, "S": spr_r, "r": tr[i + "_r1m"]}).dropna()
        dte = pd.DataFrame({"L": lvl_t, "S": spr_t, "r": te[i + "_r1m"]}).dropna()
        if len(dtr) < 200 or len(dte) < 60:
            lines.append(f"    {i}: insufficient data")
            continue
        Xtr = np.column_stack([np.ones(len(dtr)), dtr["L"], dtr["S"]])
        beta, se, t, p, _ = ols_nw(dtr["r"].to_numpy(), Xtr)
        pred = beta[0] + beta[1] * dte["L"].to_numpy() + beta[2] * dte["S"].to_numpy()
        real = dte["r"].to_numpy()
        oc = float(np.corrcoef(pred, real)[0, 1]) if len(dte) > 2 else np.nan
        # long when predicted above the train-period unconditional mean
        edge = real[pred > dtr["r"].mean()].mean() - real.mean() if len(real) else np.nan
        lines.append(
            f"    {i}: train SPREAD b={beta[2]:+.2f} (t={t[2]:+.2f})   "
            f"OOS corr(pred, realized)={oc:+.3f} (n={len(dte)})   "
            f"OOS mean when pred>train-avg: {edge:+.2f}pp vs all-test")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="docs/index.html",
                    help="built dashboard HTML carrying the payload (default docs/index.html)")
    ap.add_argument("--csv", default=None, help="write the 27-regime table to this CSV")
    ap.add_argument("--aligned-csv", default=None, help="write the aligned per-day frame here")
    ap.add_argument("--basis", choices=["full", "expanding", "both"], default="both",
                    help="decile basis for the regime tables (default: both)")
    ap.add_argument("--no-ci", action="store_true",
                    help="skip the block-bootstrap CIs (faster)")
    args = ap.parse_args()

    P = load_payload(args.html)
    bases = ["full", "expanding"] if args.basis == "both" else [args.basis]
    A_by = {b: build_aligned(P, basis=b) for b in bases}
    A = A_by[bases[0]]
    if args.aligned_csv:
        A.to_csv(args.aligned_csv)

    print(f"Payload generated: {P.get('generated')}")
    print(f"Common DIX dates : {len(A)}  "
          f"[{A.index.min().date()} -> {A.index.max().date()}]")
    print("Low=dec1-3, Mid=dec4-7, High=dec8-10 of each index's DIX5.\n")

    def line(s):
        def ci(i):
            if f"{i}_ci_lo" in s and np.isfinite(s.get(f"{i}_ci_lo", np.nan)):
                return f" [CI {s[i+'_ci_lo']:+.1f},{s[i+'_ci_hi']:+.1f}]"
            return ""
        return (f"  n_days={s['n_days']:>4} (ret n={s['ret_n']:>4})   "
                + "   ".join(f"{i} {s[i+'_mean']:>+6.2f}% (med {s[i+'_med']:>+5.2f}, "
                             f"hit {s[i+'_hit']:>3.0f}%){ci(i)}" for i in IDX))

    for b in bases:
        Ab = A_by[b]
        tag = ("full-sample deciles" if b == "full"
               else f"EXPANDING deciles (no look-ahead, min {EXP_MIN} obs; "
                    f"{len(Ab)} scored days)")
        print(f"=== BASELINE ({tag}) ===")
        print(line(regime_stats(Ab, pd.Series(True, index=Ab.index),
                                "BASELINE", with_ci=not args.no_ci)))
        print(f"\n=== REQUESTED DIVERGENCE: SPX & NDX DIX Low, IWM DIX High ({tag}) ===")
        print(line(regime_stats(Ab, (Ab["SPX_z"] == "Low") & (Ab["NDX_z"] == "Low")
                                & (Ab["IWM_z"] == "High"), "LLH", with_ci=not args.no_ci)))
        print()

    print(two_factor_report(A))
    print()
    print(entry_report(A))
    print()
    print(sector_gauge_report(P))
    print()
    print(oos_report(A))
    print()

    reg = all_regimes(A, with_ci=not args.no_ci)
    if args.csv:
        reg.to_csv(args.csv, index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print("=== ALL 27 COMOVEMENT REGIMES (1-month forward return %, by frequency; "
          "full-sample deciles; CI = 95% block bootstrap) ===")
    show = [c for c in reg.columns if not c.endswith("_ci_lo") and not c.endswith("_ci_hi")]
    if not args.no_ci:
        for i in IDX:
            reg[i + "_CI"] = reg.apply(
                lambda r: (f"[{r[i+'_ci_lo']:+.1f},{r[i+'_ci_hi']:+.1f}]"
                           if np.isfinite(r.get(i + "_ci_lo", np.nan)) else "--"), axis=1)
        show = ["regime", "n_days"] + [c for i in IDX for c in
                                       (f"{i}_mean", f"{i}_med", f"{i}_hit", f"{i}_CI")] + ["ret_n"]
    print(reg[show].to_string(index=False))


if __name__ == "__main__":
    main()
