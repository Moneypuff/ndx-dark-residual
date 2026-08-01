#!/usr/bin/env python3
"""
Build the cross-index DIX comovement tab (docs/comovement.html).

Reads the DIX + 1-month forward-return series straight from the already-built
dashboard payload (the `const P = {...}` blob in docs/index.html), joins them to
daily QQQ / SPY / IWM closes (the index price proxies, fetched via the repo's own
Yahoo loader), and computes:

  * a 5-day moving average of each index's dollar-DIX, bucketed into deciles
    (Low = 1-3, Mid = 4-7, High = 8-10) -> the 27 comovement regimes  ->  `DATA`
  * per-day rebased price paths + regime code + forward returns -> `PX`
    (powers the filterable price chart)
  * baseline / requested-regime / headline figures -> `META`
    (so the prose figures stay correct as the dashboard refreshes)

The HTML shell lives in comovement_template.html with three placeholders
(/*__DATA__*/, /*__PX__*/ and /*__META__*/). Mirrors build_report.py.

    python build_comovement.py --docs-out docs/comovement.html --cache-dir .ndx_dark_cache
"""
import argparse
import base64
import json
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N

IDX = ["NDX", "SPX", "IWM"]
PROXY = {"NDX": "QQQ", "SPX": "SPY", "IWM": "IWM"}   # index -> ETF price proxy
LVLW = {"L": "Low", "M": "Mid", "H": "High"}
FEATURE = "LLH"   # the requested regime: NDX Low, SPX Low, IWM High


def load_payload(html_path):
    """Pull the payload out of the built dashboard. Handles both the legacy
    plain `const P = {...};` blob and the current compressed
    `const PZ = "<base64 deflate>";` form."""
    html = Path(html_path).read_text(encoding="utf-8")
    m = re.search(r"const P = (\{.*?\});", html, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r'const PZ = "([A-Za-z0-9+/=]+)";', html)
    if m:
        return json.loads(zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8"))
    raise SystemExit(f"No embedded payload found in {html_path} -- build docs/index.html first.")


def dix_and_returns(P):
    """(dix, ret) dicts of per-index daily Series, from the dashboard payload."""
    def s(dates, vals):
        return pd.Series(vals, index=pd.to_datetime(dates), dtype="float64")
    dix = {"NDX": s(P["rel"]["dates"], P["rel"]["ndx_dix"]),
           "SPX": s(P["spx"]["dates"], P["spx"]["dix"]),
           "IWM": s(P["iwm"]["dates"], P["iwm"]["d"])}
    ret = {"NDX": s(P["rel"]["dates"], P["rel"]["r21"][P["bench"]]),
           "SPX": s(P["spx"]["dates"], P["spx"]["r21"]),
           "IWM": s(P["iwm"]["dates"], P["iwm"]["r21"])}
    return dix, ret


def build_aligned(dix, ret, adjclose):
    """Aligned per-day frame: DIX5 deciles -> regime code, forward returns, rebased prices.
    Restricted to the dates where all three DIX gauges *and* all three prices exist."""
    cols = {}
    for k in IDX:
        cols[k + "_dix5"] = dix[k].rolling(5, min_periods=3).mean()
        cols[k + "_ret"] = ret[k]
    A = pd.DataFrame(cols).sort_index()
    A = A[A[[f"{k}_dix5" for k in IDX]].notna().all(axis=1)]
    spine = A.index.intersection(adjclose.index)
    A = A.loc[spine].copy()
    for k in IDX:
        dec = pd.qcut(A[k + "_dix5"], 10, labels=False, duplicates="drop") + 1
        A[k + "_z"] = np.where(dec <= 3, "L", np.where(dec >= 8, "H", "M"))
    A["code"] = A["NDX_z"] + A["SPX_z"] + A["IWM_z"]
    reb = adjclose.loc[spine].div(adjclose.loc[spine].iloc[0]).mul(100)
    for k in IDX:
        A[k + "_px"] = reb[PROXY[k]]
    return A


def regime_row(g):
    """(mean, median, hit%) triple per index for a group of days."""
    out = []
    for k in IDX:
        r = g[k + "_ret"].dropna()
        out.append([round(float(r.mean()), 2), round(float(r.median()), 2),
                    round(float((r > 0).mean() * 100)) if len(r) else 0] if len(r)
                   else [None, None, 0])
    return out


def build_data(A):
    """The 27-regime table, sorted by frequency (matches the study's ordering)."""
    rows = []
    for code, g in A.groupby("code"):
        n, s, i = regime_row(g)
        reg = f"N={LVLW[code[0]]},S={LVLW[code[1]]},I={LVLW[code[2]]}"
        rows.append([reg, int(len(g)), n, s, i])
    rows.sort(key=lambda r: -r[1])
    return rows


def build_px(A):
    def col(name):
        return [None if pd.isna(x) else round(float(x), 4) for x in A[name].values]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in A.index],
        "px": {k: [round(float(x), 2) for x in A[k + "_px"].values] for k in IDX},
        "ret": {k: col(k + "_ret") for k in IDX},
        "code": list(A["code"]),
        "start": A.index[0].strftime("%Y-%m-%d"),
        "end": A.index[-1].strftime("%Y-%m-%d"),
    }


def build_meta(A, data, P):
    n = len(A)
    base = {}
    for k in IDX:
        r = A[k + "_ret"].dropna()
        base[k] = [round(float(r.mean()), 2), round(float(r.median()), 2),
                   round(float((r > 0).mean() * 100))]
    # requested-regime stats (+ delta vs baseline mean)
    fg = A[A["code"] == FEATURE]
    feat = {}
    for k in IDX:
        r = fg[k + "_ret"].dropna()
        m = round(float(r.mean()), 2) if len(r) else 0.0
        feat[k] = [m, round(float(r.median()), 2) if len(r) else 0.0,
                   round(float((r > 0).mean() * 100)) if len(r) else 0,
                   round(m - base[k][0], 2)]
    # findings ranges, all derived from the table so prose never drifts
    by = {r[0]: r for r in data}

    def means_for(codes):
        vals = []
        for c in codes:
            reg = f"N={LVLW[c[0]]},S={LVLW[c[1]]},I={LVLW[c[2]]}"
            if reg in by:
                r = by[reg]
                vals += [r[2][0], r[3][0], r[4][0]]
        return [v for v in vals if v is not None]

    def rng(vals):
        return f"{'+' if min(vals) >= 0 else ''}{min(vals):.1f}% to {'+' if max(vals) >= 0 else ''}{max(vals):.1f}%"

    comov = means_for(["HHH", "LLL"])
    feat_means = [feat[k][0] for k in IDX]
    # high-IWM-alone weakness: the purest case -- IWM DIX High while both large caps
    # sit only Mid (N=Mid, S=Mid, I=High) -- a single, definitionally-stable regime.
    pure = by.get("N=Mid,S=Mid,I=High")
    iwm_val = f"{pure[4][0]:+.2f}%" if pure and pure[4][0] is not None else "—"
    iwm_hit = f"{pure[4][2]:.0f}%" if pure else "—"
    # best large-cap tailwind: max NDX mean across all regimes
    best = max(data, key=lambda r: (r[2][0] if r[2][0] is not None else -99))
    best_parts = dict(p.split("=") for p in best[0].split(","))

    return {
        "range": f"{A.index[0].strftime('%Y-%m-%d')} → {A.index[-1].strftime('%Y-%m-%d')}",
        "n_days": n,
        "n_indep": round(n / 21),
        "n_reg": f"{len(data)} of 27 regimes observed",
        "built": "Data " + P.get("generated", "").split()[0] + " · built "
                 + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "baseline": base,
        "feature": {"stats": feat},
        "find": {
            "comov": rng(comov) if comov else "—",
            "iwmVal": iwm_val,
            "iwmHit": iwm_hit,
            "feat": rng(feat_means),
            "bestVal": f"+{best[2][0]:.2f}%",
            "bestHit": f"{best[2][2]:.0f}%",
            "bestReg": f"NDX {best_parts['N']} / SPX {best_parts['S']} / IWM {best_parts['I']}",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", default="docs/index.html",
                    help="built dashboard carrying the DIX payload (default docs/index.html)")
    ap.add_argument("--template", default="comovement_template.html")
    ap.add_argument("--docs-out", default="docs/comovement.html")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    args = ap.parse_args()

    P = load_payload(args.payload)
    dix, ret = dix_and_returns(P)
    lo = min(s.index.min() for s in dix.values()) - pd.Timedelta(days=10)
    hi = max(s.index.max() for s in dix.values()) + pd.Timedelta(days=2)
    panels = N.load_yahoo_panels(list(PROXY.values()), lo, hi,
                                 cache_dir=args.cache_dir or None, label="COMOVE")
    adjclose = panels["adjclose"][list(PROXY.values())].dropna(how="all")

    A = build_aligned(dix, ret, adjclose)
    if A.empty:
        raise SystemExit("No overlapping DIX + price dates -- nothing to build.")
    data = build_data(A)
    px = build_px(A)
    meta = build_meta(A, data, P)

    body = (Path(args.template).read_text(encoding="utf-8")
            .replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
            .replace("/*__PX__*/", json.dumps(px, separators=(",", ":")))
            .replace("/*__META__*/", json.dumps(meta, separators=(",", ":"))))
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>DIX Comovement → 1-Month Forward Returns</title></head>'
           '<body>\n' + body + '\n</body></html>\n')
    out = Path(args.docs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc)//1024} KB, {len(A)} days, {len(data)} regimes)")


if __name__ == "__main__":
    main()
