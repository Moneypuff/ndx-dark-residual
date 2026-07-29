#!/usr/bin/env python3
"""
Verifier for the post-event forward-return calculations.
========================================================

Two pipelines in this repo turn prices into "forward return after an event":

  1. The dashboard's "D vs forward return" tab / S&P-500 decile table
     (`ndx_dark_residual.py`). Each name's history is bucketed into 10 deciles
     of its own dark ratio D, and every **decile candle** shows the mean (or
     median) forward return when D sat in that decile, with a **+/-1 standard
     error whisker** under/over it -- the SE uses an *overlap-adjusted* effective
     sample n/h because h-day forward returns overlap and are autocorrelated.
     The candle math lives in JavaScript (`deciles()` / `renderBars()`), so it
     never gets exercised by Python tests. This file re-implements it in Python
     and checks it.

  2. The earnings-event study (`earnings_dpi_study.py`) -> `earnings_dpi_events.csv`:
     forward returns adjClose(T+h)/adjClose(T)-1 at h = 1/5/10/21 sessions after a
     timing-aware base close T, plus realized vol per window.

WHAT A "DECILE CANDLE" IS (and what it is NOT)
----------------------------------------------
The whisker is a **standard error of the estimated mean return** for that
decile -- a precision band on the average -- NOT a price wick and NOT the range
of up/down outcomes. A "small downside tail" on a name like Alphabet means its
per-decile mean is estimated *precisely* (many observations, low dispersion of
the mean), NOT that the stock rarely falls. The whisker half-width is

    SE = popstd(returns in the decile) / sqrt(max(1, n / h))

so it shrinks like 1/sqrt(n) as a name accumulates history; the *spread of
actual returns* (e.g. p5..p95) is many times wider and does not shrink with n.
`check_candle_semantics` demonstrates the gap numerically.

USAGE
-----
    python verify_forward_returns.py                 # offline checks (default)
    python verify_forward_returns.py --rebuild       # + live price cross-check
    python verify_forward_returns.py --candles GOOGL --horizon 21   # print a
                                                     # name's real decile candles

Exit status is non-zero if any check fails, so it drops into CI as-is.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N

TOL = 1e-9


# ---------------------------------------------------------------------------
# Independent re-implementations (deliberately NOT importing the originals, so
# a bug in the original can't hide behind the same code verifying it).
# ---------------------------------------------------------------------------
def fwd_return_pct(close, h):
    """Forward h-session % return, re-derived from scratch: at each row t it is
    100*(close[t+h]/close[t] - 1) and NaN for the final h rows (their future
    close doesn't exist yet -- the no-look-ahead guarantee)."""
    close = np.asarray(close, dtype=float)
    out = np.full(close.shape[0], np.nan)
    for t in range(close.shape[0] - h):
        c0, c1 = close[t], close[t + h]
        if np.isfinite(c0) and np.isfinite(c1) and c0 != 0:
            out[t] = 100.0 * (c1 / c0 - 1.0)
    return out


def decile_buckets(pairs, k=10):
    """Port of the dashboard's `deciles()` (ndx_dark_residual.py JS).

    `pairs` is a list/array of (D, forward_return). Sort by D, cut into k
    equal-*count* buckets by position (lo=floor(n*b/k), hi=floor(n*(b+1)/k)),
    and per bucket return mean/median/POPULATION-std of the returns plus the
    D-domain edges. Population std (divide by n, not n-1) matches the JS."""
    pts = sorted((float(x), float(y)) for x, y in pairs)
    n = len(pts)
    out = []
    for b in range(k):
        lo, hi = (n * b) // k, (n * (b + 1)) // k
        sl = pts[lo:hi]
        if not sl:
            out.append(None)
            continue
        ys = np.array([p[1] for p in sl])
        xs = np.array([p[0] for p in sl])
        rmean = ys.mean()
        out.append({
            "n": len(sl), "dAvg": xs.mean(), "rMean": rmean,
            "rMedian": float(np.median(ys)),
            "rStd": float(np.sqrt(np.mean((ys - rmean) ** 2))),  # population
            "dLo": sl[0][0], "dHi": sl[-1][0],
        })
    return out


def candle(bucket, h, use_median=False):
    """The (value, whisker_lo, whisker_hi) a decile candle draws -- port of
    `renderBars`. value = median if use_median else mean; whisker half-width is
    the overlap-adjusted standard error SE = rStd / sqrt(max(1, n/h)) (zero in
    median mode, which draws no whisker)."""
    v = bucket["rMedian"] if use_median else bucket["rMean"]
    se = 0.0 if use_median else bucket["rStd"] / np.sqrt(max(1.0, bucket["n"] / h))
    return v, v - se, v + se


# ---------------------------------------------------------------------------
# Check harness
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.rows = []  # (status, name, detail)

    def ok(self, name, detail=""):
        self.rows.append(("PASS", name, detail))

    def fail(self, name, detail=""):
        self.rows.append(("FAIL", name, detail))

    def skip(self, name, detail=""):
        self.rows.append(("SKIP", name, detail))

    def check(self, cond, name, detail=""):
        (self.ok if cond else self.fail)(name, detail)
        return bool(cond)

    def report(self):
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}
        for status, name, detail in self.rows:
            line = f"  {icon[status]} [{status}] {name}"
            if detail:
                line += f"  --  {detail}"
            print(line)
        n_fail = sum(r[0] == "FAIL" for r in self.rows)
        n_pass = sum(r[0] == "PASS" for r in self.rows)
        n_skip = sum(r[0] == "SKIP" for r in self.rows)
        print("-" * 72)
        print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
        return n_fail


# ---------------------------------------------------------------------------
# 1. Forward-return primitive
# ---------------------------------------------------------------------------
def check_forward_primitive(c):
    print("\n[1] Forward-return primitive (compute_forward_return)")

    # (a) exactness on a geometric series: close = c0 * g^t  =>  r_h == g^h - 1
    g, h, T = 1.01, 21, 400
    close = 100.0 * g ** np.arange(T)
    panel = pd.DataFrame({"X": close})
    got = N.compute_forward_return(panel, h)["X"].to_numpy()
    want = np.full(T, np.nan)
    want[:T - h] = 100.0 * (g ** h - 1.0)
    finite = np.isfinite(want)
    c.check(np.allclose(got[finite], want[finite], atol=1e-6),
            "exact on geometric series",
            f"g^{h}-1 = {100*(g**h-1):.4f}%  matched")

    # (b) no look-ahead: exactly the last h rows are NaN, the rest finite.
    nan_mask = ~np.isfinite(got)
    c.check(nan_mask.sum() == h and nan_mask[-h:].all() and not nan_mask[:-h].any(),
            "no look-ahead (last h rows NaN, earlier rows finite)",
            f"{nan_mask.sum()} trailing NaNs, expected {h}")

    # (c) independent re-derivation agrees with the shipped vectorized version.
    rng = np.random.default_rng(7)
    rand = pd.DataFrame({"A": 50 * np.cumprod(1 + rng.normal(0, 0.02, 300)),
                         "B": 80 * np.cumprod(1 + rng.normal(0, 0.03, 300))})
    for col in ("A", "B"):
        a = N.compute_forward_return(rand[[col]], h)[col].to_numpy()
        b = fwd_return_pct(rand[col].to_numpy(), h)
        both = np.isfinite(a) & np.isfinite(b)
        agree = np.allclose(a[both], b[both], atol=1e-9) and \
            (np.isfinite(a) == np.isfinite(b)).all()
        c.check(agree, f"independent re-derivation matches ({col})")

    # (d) split-safety: a raw 2:1 split halves raw close, but the ADJUSTED close
    #     is continuous -- forward returns off adjusted close must show no -50%
    #     artifact at the split. (Confirms the pipeline uses adjusted close.)
    adj = 100.0 * g ** np.arange(200)          # smooth, already split-adjusted
    r_adj = N.compute_forward_return(pd.DataFrame({"X": adj}), 1)["X"].to_numpy()
    raw = adj.copy(); raw[100:] /= 2.0          # a raw 2:1 split at t=100
    r_raw = N.compute_forward_return(pd.DataFrame({"X": raw}), 1)["X"].to_numpy()
    c.check(np.nanmin(r_adj) > -1.0,
            "adjusted-close returns have no split cliff",
            f"min 1d fwd on adj = {np.nanmin(r_adj):+.3f}%")
    c.check(r_raw[99] < -40.0 and r_adj[99] > 0,
            "raw (unadjusted) close WOULD show a -50% cliff (control)",
            f"raw t=99 = {r_raw[99]:+.1f}%  vs adj {r_adj[99]:+.3f}%")


# ---------------------------------------------------------------------------
# 2. Decile-candle math
# ---------------------------------------------------------------------------
def check_decile_candles(c):
    print("\n[2] Decile-candle bucketing & whisker (deciles / renderBars)")

    # Build a clean monotone relationship: return increases with D.
    rng = np.random.default_rng(1)
    D = rng.uniform(0, 1, 5000)
    ret = 20.0 * (D - 0.5) + rng.normal(0, 5, D.size)   # slope +20pp across D
    buckets = decile_buckets(list(zip(D, ret)), k=10)

    # (a) equal-count partition: bucket sizes differ by at most 1, sum == n.
    sizes = [b["n"] for b in buckets]
    c.check(sum(sizes) == D.size and max(sizes) - min(sizes) <= 1,
            "10 equal-count deciles partition the sample",
            f"sizes {min(sizes)}..{max(sizes)}, sum {sum(sizes)}")

    # (b) domain edges are sorted and non-overlapping (deciles of D are ordered).
    edges_ok = all(buckets[i]["dHi"] <= buckets[i + 1]["dLo"] + TOL
                   for i in range(9))
    c.check(edges_ok, "decile D-domain edges are monotone & non-overlapping")

    # (c) means recover the injected monotone gradient.
    means = [b["rMean"] for b in buckets]
    c.check(all(means[i] < means[i + 1] for i in range(9)),
            "decile means are monotonically increasing on a monotone signal",
            f"D1 {means[0]:+.2f}%  ->  D10 {means[-1]:+.2f}%")

    # (d) whisker = mean +/- SE, SE = popstd / sqrt(n/h): recompute one bucket by
    #     hand and match candle(); median mode draws no whisker.
    h = 21
    b0 = buckets[0]
    se_hand = b0["rStd"] / np.sqrt(b0["n"] / h)
    v, lo, hi = candle(b0, h, use_median=False)
    c.check(abs((hi - lo) / 2 - se_hand) < 1e-9 and abs((hi + lo) / 2 - v) < 1e-9,
            "whisker half-width == overlap-adjusted SE, centered on the mean",
            f"SE(D1) = {se_hand:.4f}pp")
    vm, lom, him = candle(b0, h, use_median=True)
    c.check(lom == vm == him,
            "median mode draws no whisker (SE forced to 0)")

    # (e) effective-N: dividing n by h widens SE by exactly sqrt(h) vs naive n.
    se_naive = b0["rStd"] / np.sqrt(b0["n"])
    c.check(abs(se_hand / se_naive - np.sqrt(h)) < 1e-9,
            "overlap adjustment widens SE by sqrt(h)",
            f"{se_hand:.4f} = {np.sqrt(h):.3f} x {se_naive:.4f}")


def check_candle_semantics(c):
    """The user's actual question: a 'small downside tail' is a *precision* band,
    not a distribution of outcomes. Show the whisker half-width is far smaller
    than the p5..p95 spread of the very same returns, and that it -- not the
    spread -- is what shrinks as a name accumulates history."""
    print("\n[3] Candle semantics: whisker (SE of the mean) vs outcome spread")
    rng = np.random.default_rng(3)
    h = 21

    def one_decile(n, vol=6.0):
        ret = rng.normal(1.0, vol, n)           # ~1% mean, `vol`% dispersion
        b = {"n": n, "rMean": ret.mean(), "rMedian": float(np.median(ret)),
             "rStd": float(np.std(ret)), "dAvg": 0.5, "dLo": 0.4, "dHi": 0.6}
        _, lo, hi = candle(b, h)
        se_half = (hi - lo) / 2
        p5, p95 = np.percentile(ret, [5, 95])
        return se_half, (p95 - p5) / 2, ret

    # A mega-cap-sized decile (long history -> big n) has a tight whisker...
    se_big, spread_big, _ = one_decile(n=400)
    # ...the outcome spread is many multiples wider than the whisker.
    c.check(spread_big > 5 * se_big,
            "downside 'tail' (whisker) << spread of actual returns",
            f"whisker +/-{se_big:.2f}pp  vs  p5..p95 +/-{spread_big:.1f}pp "
            f"({spread_big/se_big:.0f}x wider)")

    # The whisker shrinks ~1/sqrt(n); the return spread does not shrink with n.
    se_small, spread_small, _ = one_decile(n=40)
    c.check(se_small > 2 * se_big and abs(spread_small - spread_big) < 0.35 * spread_big,
            "more history -> smaller whisker, ~unchanged outcome spread",
            f"n=40 whisker +/-{se_small:.2f}pp  ->  n=400 whisker +/-{se_big:.2f}pp")


# ---------------------------------------------------------------------------
# 3b. Alphabet share-class merge (dashboard candles vs earnings study)
# ---------------------------------------------------------------------------
def check_alphabet_merge(c):
    """The decile candles fold GOOG (class C) into GOOGL. Verify the dashboard's
    merge is VOLUME-WEIGHTED and identical to the earnings study's construction
    -- (short_G+short_GL)/(total_G+total_GL) -- so the two pipelines feed the
    same Alphabet D, not a plain mean (which over-weights the thinner class C)."""
    print("\n[3b] Alphabet GOOG->GOOGL merge is volume-weighted & consistent")
    rng = np.random.default_rng(11)
    idx = pd.date_range("2021-01-01", periods=40, freq="B")
    total = pd.DataFrame({"GOOG": rng.uniform(80, 120, 40),
                          "GOOGL": rng.uniform(300, 500, 40),
                          "AAPL": rng.uniform(900, 1100, 40)}, index=idx)
    ratio = pd.DataFrame({"GOOG": rng.uniform(0.35, 0.5, 40),
                          "GOOGL": rng.uniform(0.25, 0.4, 40),
                          "AAPL": rng.uniform(0.1, 0.15, 40)}, index=idx)
    short = ratio * total                                  # short = dark-ratio * total

    merged = N.vw_merge_alphabet(ratio, total)
    # earnings-study construction, computed independently here:
    study = (short["GOOG"] + short["GOOGL"]) / (total["GOOG"] + total["GOOGL"])
    c.check(np.allclose(merged["GOOGL"].to_numpy(), study.to_numpy(), atol=1e-12),
            "dashboard merge == (sumShort)/(sumTotal) (earnings-study construction)")
    c.check("GOOG" not in merged.columns and "AAPL" in merged.columns,
            "class C dropped, unrelated names untouched")

    # It must NOT be the plain mean (unless the classes happened to tie).
    plain = ratio[["GOOG", "GOOGL"]].mean(axis=1)
    c.check(not np.allclose(merged["GOOGL"].to_numpy(), plain.to_numpy(), atol=1e-6),
            "volume weighting differs from the old plain mean",
            f"vw {merged['GOOGL'].iloc[0]:.4f} vs mean {plain.iloc[0]:.4f}")

    # With weights absent (e.g. --demo) it falls back to the plain mean, safely.
    fb = N.vw_merge_alphabet(ratio, None)
    c.check(np.allclose(fb["GOOGL"].to_numpy(), plain.to_numpy(), atol=1e-12),
            "falls back to a simple mean when weights are unavailable")


# ---------------------------------------------------------------------------
# 4. Earnings-event forward returns (committed CSV invariants)
# ---------------------------------------------------------------------------
def check_earnings_events(c, path):
    print("\n[4] Earnings-event forward returns (earnings_dpi_events.csv)")
    p = Path(path)
    if not p.exists():
        c.skip("events CSV present", f"{path} not found")
        return
    ev = pd.read_csv(p)
    HZ = {"next_day": 1, "w1": 5, "w2": 10, "m1": 21}

    # (a) has_data flag is exactly "next-day return exists".
    hd = ev["next_day_ret"].notna().astype(int)
    c.check((hd == ev["has_data"]).all(),
            "has_data == isfinite(next_day_ret)")

    # (b) horizon nesting: a longer-horizon return can only exist if every
    #     shorter one does (T+21 present => T+10,T+5,T+1 present). Catches
    #     off-by-one / windowing bugs without needing the price panels.
    order = ["next_day_ret", "w1_ret", "w2_ret", "m1_ret"]
    nest_ok = True
    for i in range(len(order) - 1, 0, -1):
        longer = ev[order[i]].notna()
        shorter = ev[order[i - 1]].notna()
        if (longer & ~shorter).any():
            nest_ok = False
    c.check(nest_ok, "forward-return horizons nest (T+21 => T+10 => T+5 => T+1)")

    # (c) realized vol: non-negative where present, and present exactly when the
    #     full daily-return window is present (its horizon return exists).
    rv_ok = True
    for k, col in [("next_day", "next_day_rvol"), ("w1", "w1_rvol"),
                   ("w2", "w2_rvol"), ("m1", "m1_rvol")]:
        s = ev[col]
        if (s.dropna() < 0).any():
            rv_ok = False
        # rvol needs T+1..T+h; if it's finite the h-horizon return must be too.
        if (s.notna() & ev[f"{k}_ret"].isna()).any():
            rv_ok = False
    c.check(rv_ok, "realized vol >= 0 and present only with a full return window")

    # (d) looks_reaction flag implies a >=2% (or >=2 sigma) next-day move; the 2%
    #     floor is a necessary condition we can check directly.
    lr = ev[ev["looks_reaction"] == 1]["next_day_ret"].abs()
    c.check((lr.dropna() >= 0.02 - 1e-9).all(),
            "looks_reaction => |next-day move| >= 2%",
            f"min flagged |move| = {lr.min()*100:.2f}%")

    # (e) timing-aware base close T: base_T is never after the report; for
    #     before-open / intraday reporters it is strictly earlier.
    rd = pd.to_datetime(ev["report_date"]); bt = pd.to_datetime(ev["base_T"])
    c.check((bt <= rd).all(), "base_T <= report_date for every event")
    non_amc = ev["timing"].isin(["bmo", "intraday"])
    c.check((bt[non_amc] < rd[non_amc]).all(),
            "before-open / intraday reporters use an earlier base close",
            f"n={int(non_amc.sum())} non-amc events")

    # (f) within-name DPI percentile ranks recomputed from the raw DPI columns
    #     must match the stored *_pct columns (pct rank within each ticker).
    pct_ok = True
    for w in (5, 10):
        recomputed = ev.groupby("ticker")[f"dpi{w}"].rank(pct=True)
        a, b = recomputed, ev[f"dpi{w}_pct"]
        both = a.notna() & b.notna()
        if not np.allclose(a[both], b[both], atol=1e-9):
            pct_ok = False
        if (a.isna() != b.isna()).any():
            pct_ok = False
    c.check(pct_ok, "within-name DPI percentile columns reproduce from raw DPI")

    # (g) percentile columns are bounded to [0,1]; anchor mode is off (as shipped).
    for w in (5, 10):
        s = ev[f"dpi{w}_pct"].dropna()
        c.check(((s >= 0) & (s <= 1)).all(), f"dpi{w}_pct in [0,1]")
    c.check((ev["anchored"] == 0).all(),
            "no events anchored (anchor mode off by default)")


# ---------------------------------------------------------------------------
# 5. Optional live cross-check + per-name candle dump
# ---------------------------------------------------------------------------
def _build_panels(args):
    """Rebuild the split-adjusted price + DPI panels the way build_report.py does.
    Returns (panels, earnings) or (None, None) if data can't be reached."""
    import earnings_dpi_study as E
    try:
        earn = E.load_earnings(args.earnings)
        syms = sorted(earn.ticker.unique())
        start = (earn.report_date.min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        end = (earn.report_date.max() + pd.Timedelta(days=70)).strftime("%Y-%m-%d")
        panels = N.build_universe_panels(syms, start, end, workers=args.workers,
                                         cache_dir=args.cache_dir or None, ns="earn",
                                         label="VERIFY")
        panels, earn = E.merge_share_classes(panels, earn)
        if panels["adjclose"].dropna(how="all").empty:
            return None, None
        return panels, earn
    except Exception as e:  # noqa: BLE001
        print(f"  (panel rebuild unavailable: {e})", file=sys.stderr)
        return None, None


def check_live_recompute(c, args):
    print("\n[5] Live price cross-check (--rebuild)")
    import earnings_dpi_study as E
    panels, earn = _build_panels(args)
    if panels is None:
        c.skip("price panels reachable",
               "no FINRA/Yahoo data or cache (offline) -- run where data is reachable")
        return

    horizons = dict(E.HORIZONS); horizons["m1"] = args.month_sessions
    fresh = E.build_events(earn, panels, horizons=horizons)
    adj = panels["adjclose"]
    idx = adj.index

    # (a) ARITHMETIC (strict): re-derive each event's forward returns straight
    #     from the SAME freshly built adjusted-close panel at the event's own
    #     base close T -- adjClose(T+h)/adjClose(T)-1, computed independently of
    #     build_events. Same data in, so the math must agree to float precision;
    #     any gap here is a real bug in the forward-return arithmetic.
    worst = 0.0
    for _, r in fresh.iterrows():
        tk = r["ticker"]
        if tk not in adj.columns:
            continue
        pos = idx.searchsorted(pd.Timestamp(r["base_T"]))
        if pos >= len(idx) or idx[pos] != pd.Timestamp(r["base_T"]):
            continue
        a = adj[tk].to_numpy(); base = a[pos]
        for col, h in horizons.items():
            want = r[f"{col}_ret"]
            got = (a[pos + h] / base - 1) if (pos + h < len(a) and np.isfinite(base)
                                              and np.isfinite(a[pos + h])) else np.nan
            if np.isfinite(want) and np.isfinite(got):
                worst = max(worst, abs(want - got))
    c.check(worst < 1e-9,
            "event forward returns == adjClose(T+h)/adjClose(T)-1 (independent)",
            f"max abs diff = {worst:.2e}")

    # (b) REPRODUCIBILITY (drift-tolerant): committed CSV vs today's rebuild. The
    #     arithmetic is fixed, but Yahoo revises historical adjusted closes as
    #     dividends accrue and FINRA posts late corrections, so a handful of cells
    #     move. Pass when the typical event is unchanged (tiny median) and only a
    #     negligible fraction drift -- a systematic code/regression would move the
    #     median or a large share, not 2 cells in 2,884.
    disk = pd.read_csv(args.events)
    m = disk.merge(fresh, on=["ticker", "report_date"], suffixes=("_disk", "_fresh"))
    alld = []
    for col in ["next_day_ret", "w1_ret", "w2_ret", "m1_ret"]:
        a, b = m[f"{col}_disk"], m[f"{col}_fresh"]
        both = a.notna() & b.notna()
        alld.append((a[both] - b[both]).abs())
    d = pd.concat(alld)
    med, mx = float(d.median()), float(d.max())
    frac_drift = float((d > 1e-3).mean())
    c.check(med < 1e-6 and frac_drift < 0.01,
            "committed CSV reproduces (median exact; only isolated data revisions)",
            f"median {med:.1e}, max {mx:.2e}, {frac_drift*100:.2f}% of cells >0.1pp "
            f"(n={len(d)})")


def dump_candles(name, horizon, args):
    """Print the actual decile candles for one name -- the exact numbers the
    dashboard draws -- using the same packed (d, forward-return) series the
    front end receives, then this file's independent candle math."""
    panels, _ = _build_panels(args)
    if panels is None:
        print(f"\nCannot draw candles for {name}: price panels unavailable "
              f"(offline / no cache). Re-run with data reachable.", file=sys.stderr)
        return 1
    hcol = {21: "r21", 42: "r42", 63: "r63"}.get(horizon)
    if hcol is None:
        print(f"horizon must be 21, 42 or 63 (got {horizon})", file=sys.stderr)
        return 2
    # The dashboard packs these bars over the analysis window (plot_start). Default
    # to FULL history -- the small whiskers you notice on mega-caps come from the
    # long window (many obs per decile), not the trailing year. pack_name_rel folds
    # GOOG into GOOGL and rounds to 4dp, so the candles below match the on-screen ones.
    plot_start = (pd.Timestamp(args.plot_start) if args.plot_start
                  else panels["d"].index.min())
    rel = N.pack_name_rel(panels["d"], panels["adjclose"], plot_start=plot_start)
    if name not in rel["d"]:
        print(f"{name} not in universe. Names: {', '.join(sorted(rel['d']))[:400]}...",
              file=sys.stderr)
        return 2
    d = rel["d"][name]; r = rel[hcol][name]
    pairs = [(dv, rv) for dv, rv in zip(d, r) if dv is not None and rv is not None]
    buckets = decile_buckets(pairs, k=10)
    print(f"\nDecile candles for {name}  (forward horizon {horizon} sessions, "
          f"n={len(pairs)} paired obs)")
    print(f"{'decile':>7} {'D range':>15} {'n':>5} {'mean%':>8} "
          f"{'median%':>8} {'SE(pp)':>7} {'whisker (mean +/- SE)':>24}")
    for i, b in enumerate(buckets, 1):
        if b is None:
            continue
        v, lo, hi = candle(b, horizon)
        print(f"D{i:<6} {b['dLo']:.3f}..{b['dHi']:.3f} {b['n']:>5} "
              f"{b['rMean']:>+8.2f} {b['rMedian']:>+8.2f} {(hi-lo)/2:>7.3f} "
              f"[{lo:>+7.2f}, {hi:>+7.2f}]")
    print("\nThe SE(pp) column is each candle's whisker half-width -- the "
          "'downside tail' you see is +/-1 standard error of that decile's MEAN "
          "forward return (overlap-adjusted n/h), not a range of price outcomes. "
          "It is small because n is large and shrinks like 1/sqrt(n).")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="earnings_dpi_events.csv")
    ap.add_argument("--earnings", default="earnings_dates_edgar.csv")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--month-sessions", type=int, default=21)
    ap.add_argument("--rebuild", action="store_true",
                    help="also rebuild price panels and cross-check the CSV (needs data)")
    ap.add_argument("--candles", metavar="TICKER", default=None,
                    help="print the real decile candles for one name and exit")
    ap.add_argument("--horizon", type=int, default=21,
                    help="candle forward horizon in sessions: 21, 42 or 63")
    ap.add_argument("--plot-start", default=None,
                    help="candle window start (YYYY-MM-DD); default is full history")
    args = ap.parse_args()

    if args.candles:
        return dump_candles(args.candles.upper(), args.horizon, args)

    print("=" * 72)
    print("FORWARD-RETURN VERIFIER")
    print("=" * 72)
    c = Checks()
    check_forward_primitive(c)
    check_decile_candles(c)
    check_candle_semantics(c)
    check_alphabet_merge(c)
    check_earnings_events(c, args.events)
    if args.rebuild:
        check_live_recompute(c, args)
    else:
        c.skip("live price cross-check", "pass --rebuild to enable (needs FINRA/Yahoo)")

    print()
    n_fail = c.report()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
