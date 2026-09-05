#!/usr/bin/env python3
"""
DPI responder segregation: which names respond to a high dark ratio, which don't.

Question (per request): bucket the names into those whose forward return responds POSITIVELY
to having their DPI in the top deciles (>D7), those that don't respond, and those that respond
NEGATIVELY.

Response is measured WITHIN each name and in excess of QQQ, so it isolates the DPI effect and
is immune both to survivorship (it is the name's own returns) and to the market-timing of when
high-DPI clusters occur:

    high_{i,t}     = trailing (no-look-ahead) DPI decile of name i at t >= DEC_HI  (default 8, i.e. >D7)
    x_{i,t}(h)     = name i's h-day forward return minus QQQ's over the same window
    response_i(h)  = mean(x | high) - mean(x | all days)

Significance per name: a moving-block bootstrap (block = h) that preserves the overlapping-window
autocorrelation. Buckets at |t| >= TCUT.

The point of the study is NOT the in-sample roster (with ~100 names a t-test manufactures a few
"responders" by chance) but three honesty checks:
  1. chance test  -- do the bucket counts exceed what multiple testing yields under the null?
  2. persistence  -- does first-half response predict second-half response? (name-stability)
  3. structure    -- does any ex-ante feature (size, dark-share level, DPI vol, sector) separate them?

Reads the dashboard payload (the `rel` block: per-name 5-day-MA dark ratio `d`, close, QQQ).
Writes data/dpi_responder_buckets.csv. No scipy dependency (rank corr and the normal tail are
hand-rolled) so it runs in the plain pipeline environment.

Usage:
  python dpi_responder_study.py --payload docs/index.html
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

DEC_HI = 8          # "top deciles above 7"
MIN_HI = 30         # minimum high-DPI days to classify a name
TRAIL_WIN, TRAIL_MIN = 252, 120
TCUT = 1.5
HORIZONS = (21, 42, 63)


# ---------------------------------------------------------------- payload
def load_payload(path):
    import base64
    import json
    import re
    import zlib
    html = open(path, encoding="utf-8").read()
    m = re.search(r"const P = (\{.*?\});", html, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r'const PZ = "([A-Za-z0-9+/=]+)";', html)
    if m:
        return json.loads(zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8"))
    raise SystemExit(f"no embedded payload in {path}")


def _arr(a):
    return np.array([np.nan if v is None else v for v in a], float)


# ---------------------------------------------------------------- primitives
def trailing_deciles(d, win=TRAIL_WIN, minobs=TRAIL_MIN):
    """Each day's value ranked against the name's previous `win` non-null obs (strictly prior)."""
    out = np.full(len(d), np.nan)
    buf = []
    for i, v in enumerate(d):
        if np.isnan(v):
            continue
        if len(buf) >= minobs:
            le = sum(1 for x in buf if x <= v)
            out[i] = max(1, math.ceil(le / len(buf) * 10))
        buf.append(v)
        if len(buf) > win:
            buf.pop(0)
    return out


def fwd_excess(logc, logb, T, h):
    out = np.full(T, np.nan)
    for i in range(T - h):
        if np.isfinite(logc[i]) and np.isfinite(logc[i + h]) and np.isfinite(logb[i]) and np.isfinite(logb[i + h]):
            out[i] = (math.exp(logc[i + h] - logc[i]) - 1) * 100 - (math.exp(logb[i + h] - logb[i]) - 1) * 100
    return out


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def _norm_tail(z):
    """One-sided P(Z >= z) via erfc -- expected fraction of names past a |t| threshold under null."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------------------------------------------------------- response
class Study:
    def __init__(self, payload):
        rel = payload["rel"]
        self.bench = payload["bench"]
        self.dates = rel["dates"]
        self.T = len(self.dates)
        self.names = [n for n in rel["d"] if n != self.bench and n in rel.get("r21", rel["close"])]
        close = rel["close"]
        self.logb = np.log(_arr(close[self.bench]))
        self.logc = {n: np.log(_arr(close[n])) for n in self.names if n in close}
        self.names = [n for n in self.names if n in self.logc]
        self.dpi = {n: _arr(rel["d"][n]) for n in self.names}
        self.dec = {n: trailing_deciles(self.dpi[n]) for n in self.names}
        self._fwd = {}

    def fwd(self, nm, h):
        key = (nm, h)
        if key not in self._fwd:
            self._fwd[key] = fwd_excess(self.logc[nm], self.logb, self.T, h)
        return self._fwd[key]

    def response(self, nm, h, lo=0, hi=None, boot=0, rng=None):
        hi = self.T if hi is None else hi
        x = self.fwd(nm, h); dec = self.dec[nm]
        idx = np.array([i for i in range(lo, min(hi, self.T))
                        if np.isfinite(x[i]) and not np.isnan(dec[i])])
        if len(idx) < 50:
            return None
        himask = dec[idx] >= DEC_HI
        nh = int(himask.sum())
        if nh < MIN_HI:
            return None
        xv = x[idx]
        resp = float(xv[himask].mean() - xv.mean())
        out = dict(nm=nm, resp=resp, nh=nh, n=len(idx),
                   hit=float((xv[himask] > 0).mean() * 100),
                   cond=float(xv[himask].mean()), uncond=float(xv.mean()), se=None, t=None)
        if boot:
            rng = rng or np.random.default_rng(7)
            nb = int(np.ceil(len(idx) / h)); smax = len(idx) - h
            stats = []
            for _ in range(boot):
                take = []
                for _ in range(nb):
                    s = rng.integers(0, max(1, smax + 1)); take.extend(range(s, s + h))
                take = np.array(take[:len(idx)])
                xb, hb = xv[take], himask[take]
                if hb.sum() >= 5 and (~hb).sum() >= 5:
                    stats.append(xb[hb].mean() - xb.mean())
            if len(stats) > 30:
                out["se"] = float(np.std(stats))
                out["t"] = resp / out["se"] if out["se"] > 0 else None
        return out

    def bucket(self, r):
        if r["t"] is None:
            return "neutral"
        if r["resp"] > 0 and r["t"] >= TCUT:
            return "responder"
        if r["resp"] < 0 and r["t"] <= -TCUT:
            return "negative"
        return "neutral"


def run(payload, boot=400, out_csv="data/dpi_responder_buckets.csv"):
    st = Study(payload)
    rng = np.random.default_rng(7)
    print(f"Universe: {len(st.names)} names with DPI+price | high-DPI = trailing decile >= {DEC_HI} "
          f"(>D7) | {st.dates[0]}..{st.dates[-1]}")
    rows_by_h = {}
    for h in HORIZONS:
        rows = [r for r in (st.response(nm, h, boot=boot, rng=rng) for nm in st.names) if r]
        for r in rows:
            r["bucket"] = st.bucket(r)
        rows.sort(key=lambda r: r["resp"], reverse=True)
        rows_by_h[h] = rows
        nb = {b: sum(1 for r in rows if r["bucket"] == b) for b in ("responder", "neutral", "negative")}
        ts = np.array([r["t"] for r in rows if r["t"] is not None])
        exp = len(ts) * _norm_tail(2.0)
        print(f"\n==== {h}d ====  classified {len(rows)}  |  responder {nb['responder']} / "
              f"neutral {nb['neutral']} / negative {nb['negative']}")
        print(f"  cross-name response: mean {np.mean([r['resp'] for r in rows]):+.2f}%  "
              f"median {np.median([r['resp'] for r in rows]):+.2f}%  "
              f"%negative {np.mean([r['resp'] < 0 for r in rows]) * 100:.0f}%")
        print(f"  |t|>=2:  positive {int((ts >= 2).sum())}  negative {int((ts <= -2).sum())}  "
              f"(chance ~{exp:.1f} each side)  <- positive bucket vs chance is the test")
        # persistence
        mid = st.T // 2
        pers = [(nm, a["resp"], b["resp"]) for nm in st.names
                for a in [st.response(nm, h, 0, mid)] for b in [st.response(nm, h, mid, st.T)]
                if a and b]
        if len(pers) >= 10:
            sr = _spearman([p[1] for p in pers], [p[2] for p in pers])
            print(f"  persistence: Spearman(H1 resp, H2 resp) = {sr:+.2f}  over {len(pers)} names "
                  f"(split {st.dates[mid]})")
    # structural characterization at 42d
    rows = rows_by_h[42]
    try:
        import ndx_dark_residual as NM
        sec_map, wt_map = NM.TICKER_SECTOR, NM.NDX100_WEIGHT
    except Exception:  # noqa: BLE001
        sec_map, wt_map = {}, {}
    resp = np.array([r["resp"] for r in rows])
    wt = np.array([wt_map.get(r["nm"], np.nan) for r in rows])
    mdpi = np.array([np.nanmean(st.dpi[r["nm"]]) for r in rows])
    vdpi = np.array([np.nanstd(st.dpi[r["nm"]]) for r in rows])
    print("\n  STRUCTURAL (42d) corr(response, feature):")
    for lbl, f in [("index weight (size)", wt), ("mean DPI level", mdpi), ("DPI volatility", vdpi)]:
        print(f"    {lbl:<22} Spearman {_spearman(resp, f):+.2f}")
    # write per-name buckets CSV (42d = the reference horizon)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "sector", "bucket_42d", "response_42d_pct", "t_42d", "hit_42d_pct",
                    "n_high_days", "response_63d_pct", "t_63d"])
        r63 = {r["nm"]: r for r in rows_by_h[63]}
        for r in rows:
            x = r63.get(r["nm"], {})
            w.writerow([r["nm"], sec_map.get(r["nm"], ""), r["bucket"], f"{r['resp']:.2f}",
                        f"{r['t']:.2f}" if r["t"] is not None else "",
                        f"{r['hit']:.0f}", r["nh"],
                        f"{x.get('resp', float('nan')):.2f}" if x else "",
                        f"{x['t']:.2f}" if x.get("t") is not None else ""])
    print(f"\n  wrote {out_csv} ({len(rows)} names)")
    return rows_by_h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", default="docs/index.html", help="built dashboard HTML with the rel payload")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default="data/dpi_responder_buckets.csv")
    args = ap.parse_args()
    run(load_payload(args.payload), boot=args.boot, out_csv=args.out)


if __name__ == "__main__":
    main()
