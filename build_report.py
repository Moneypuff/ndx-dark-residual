#!/usr/bin/env python3
"""
Build the self-contained earnings-DPI HTML report from the event data.

Reads earnings_dpi_events.csv (from earnings_dpi_study.py) plus the split-
adjusted price panels (rebuilt from cache) to compute:
  * pooled correlations, tercile/quintile buckets, yearly and cohort robustness
    cuts  -> injected as `DATA`
  * per-name post-earnings price paths (cumulative return over the 21 sessions
    after each report, plus median / mean / high-DPI-mean / low-DPI-mean paths)
    -> injected as `PATHS`, powering the click-through fan chart.

The HTML shell lives in report_template.html with two placeholders
(/*__PAYLOAD__*/ and /*__PATHS__*/). Output: earnings_dpi_report.html.

    python build_report.py --events earnings_dpi_events.csv \
        --earnings earnings_dates_edgar.csv --cache-dir ~/.ndx_dark_cache
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import earnings_dpi_study as E

MEGA = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "AMD",
        "QCOM", "ADBE", "AMAT", "INTC", "MU", "CSCO", "COST", "TXN", "PANW", "GILD"]
H = 21  # path horizon in sessions


def _terc(ev, pcol, rcol):
    hi = ev[ev[pcol] >= 2/3][rcol]; mid = ev[(ev[pcol] > 1/3) & (ev[pcol] < 2/3)][rcol]
    lo = ev[ev[pcol] <= 1/3][rcol]
    diff, t, p, _, _ = E._welch(hi.to_numpy(), lo.to_numpy())
    return dict(hi=float(hi.mean()), mid=float(mid.mean()), lo=float(lo.mean()),
                hi_up=float((hi > 0).mean()), lo_up=float((lo > 0).mean()),
                nh=int(hi.notna().sum()), nm=int(mid.notna().sum()), nl=int(lo.notna().sum()),
                diff=float(diff), t=float(t), p=float(p))


def build_payload(ev):
    ev = ev.copy()
    ev["yr"] = pd.to_datetime(ev.report_date).dt.year
    ev["mega"] = ev.ticker.isin(MEGA)
    pay = {"n_events": int(ev.dpi10.notna().sum()), "n_names": int(ev.ticker.nunique()),
           "window": ["2018-08", "2026-07"],
           "next_day": {"mean": float(ev.next_day_ret.mean()), "median": float(ev.next_day_ret.median()),
                        "pos": float((ev.next_day_ret > 0).mean())},
           "m1": {"mean": float(ev.m1_ret.mean()), "median": float(ev.m1_ret.median()),
                  "pos": float((ev.m1_ret > 0).mean())},
           "corr": {}, "tercile": {}, "quintile": {}, "period": [], "subgroup": {},
           "pername": [], "scatter": []}
    HZS = [("next_day", "Next day", "next_day_ret"), ("w1", "1 week", "w1_ret"),
           ("w2", "2 weeks", "w2_ret"), ("m1", "1 month", "m1_ret")]
    for w in (5, 10):
        for hz, _, col in HZS:
            pr, pp, pn = E._pearson(ev[f"dpi{w}"], ev[col]); sr, sp, _ = E._spearman(ev[f"dpi{w}"], ev[col])
            wr, wp, _ = E._pearson(ev[f"dpi{w}_pct"], ev[col])
            pay["corr"][f"dpi{w}_{hz}"] = dict(pearson=pr, pearson_p=pp, spearman=sr,
                                               spearman_p=sp, within=wr, within_p=wp, n=pn)
    for hz, _, col in HZS:
        pay["tercile"][hz] = _terc(ev, "dpi10_pct", col)
    pay["tercile"]["next"] = pay["tercile"]["next_day"]   # back-compat alias for template
    pay["byhorizon"] = []
    for hz, label, col in HZS:
        t = pay["tercile"][hz]; c = pay["corr"][f"dpi10_{hz}"]
        pay["byhorizon"].append(dict(key=hz, label=label, r=c["pearson"], p=c["pearson_p"],
                                     hi=t["hi"], lo=t["lo"], spread=t["diff"], spread_t=t["t"],
                                     spread_p=t["p"], hi_up=t["hi_up"], lo_up=t["lo_up"]))
    ev["q"] = pd.qcut(ev.dpi10_pct, 5, labels=[1, 2, 3, 4, 5])
    for hz, col in [("next_day", "next_day_ret"), ("m1", "m1_ret")]:
        gg = ev.groupby("q", observed=True)[col].mean()
        pay["quintile"][hz] = [float(gg.loc[i]) for i in [1, 2, 3, 4, 5]]
    for y in range(2018, 2027):
        d = ev[ev.yr == y]
        rm, pm, _ = E._pearson(d.dpi10, d.m1_ret); rn, pn, _ = E._pearson(d.dpi10, d.next_day_ret)
        pay["period"].append(dict(year=y, n=int(d.dpi10.notna().sum()),
                                  m1_r=rm, m1_p=pm, next_r=rn, next_p=pn))
    for name, mask in [("mega", ev.mega), ("nonmega", ~ev.mega),
                       ("amc", ev.timing == "amc"), ("bmo", ev.timing == "bmo")]:
        d = ev[mask]
        rn, pn, _ = E._pearson(d.dpi10, d.next_day_ret); rm, pm, _ = E._pearson(d.dpi10, d.m1_ret)
        pay["subgroup"][name] = dict(n=int(d.dpi10.notna().sum()), next_r=rn, next_p=pn, m1_r=rm, m1_p=pm)
    for tk, d in ev.groupby("ticker"):
        if d.dpi10.notna().sum() >= 8:
            r, p, n = E._pearson(d.dpi10, d.m1_ret)
            pay["pername"].append(dict(ticker=tk, r=float(r), n=int(n)))
    pay["pername"].sort(key=lambda x: x["r"])
    s = ev.dropna(subset=["dpi10", "m1_ret"])
    for _, r in s.iterrows():
        pay["scatter"].append([round(float(r.dpi10)*100, 2), round(float(r.m1_ret)*100, 2), r.ticker])
    x = s.dpi10.to_numpy()*100; y = s.m1_ret.to_numpy()*100
    b1 = np.cov(x, y, ddof=0)[0, 1] / np.var(x); b0 = y.mean() - b1*x.mean()
    pay["ols"] = dict(b0=float(b0), b1=float(b1), xmin=float(x.min()), xmax=float(x.max()))
    return pay


def build_paths(ev, adj):
    idx = adj.index
    ev = ev.copy(); ev["base_T"] = pd.to_datetime(ev.base_T)
    out = {}
    for tk, g in ev.groupby("ticker"):
        if tk not in adj.columns:
            continue
        a = adj[tk]; events = []; mat = []
        d10 = g.dpi10.dropna()
        lo_th, hi_th = (d10.quantile(1/3), d10.quantile(2/3)) if len(d10) >= 6 else (np.nan, np.nan)
        for _, r in g.iterrows():
            if pd.isna(r.base_T):
                continue
            pos = idx.searchsorted(r.base_T)
            if pos >= len(idx) or idx[pos] != r.base_T or pos + H >= len(a):
                continue
            base = a.iloc[pos]; seg = a.iloc[pos:pos + H + 1].to_numpy()
            if not np.isfinite(base) or np.isnan(seg).any():
                continue
            p = [int(round((v/base - 1)*10000)) for v in seg]
            cls = "mid"
            if np.isfinite(hi_th):
                cls = "hi" if r.dpi10 >= hi_th else ("lo" if r.dpi10 <= lo_th else "mid")
            events.append({"d": r.report_date, "cls": cls, "p": p}); mat.append(p)
        if len(mat) < 5:
            continue
        M = np.array(mat)
        med = [int(round(np.median(M[:, h]))) for h in range(H + 1)]
        mean = [int(round(M[:, h].mean())) for h in range(H + 1)]

        def cmean(cl):
            sub = np.array([e["p"] for e in events if e["cls"] == cl])
            return [int(round(sub[:, h].mean())) for h in range(H + 1)] if len(sub) >= 3 else None
        out[tk] = {"n": len(mat), "events": events, "median": med, "mean": mean,
                   "hi_mean": cmean("hi"), "lo_mean": cmean("lo"),
                   "final_med": med[-1], "final_mean": mean[-1],
                   "pos": int((M[:, -1] > 0).sum()), "tot": len(mat)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="earnings_dpi_events.csv")
    ap.add_argument("--earnings", default="earnings_dates_edgar.csv")
    ap.add_argument("--template", default="report_template.html")
    ap.add_argument("--out", default="earnings_dpi_report.html")
    ap.add_argument("--docs-out", default="",
                    help="also write a standalone, full-document copy for GitHub Pages")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    args = ap.parse_args()

    ev = pd.read_csv(args.events)
    earn = E.load_earnings(args.earnings)
    syms = sorted(earn.ticker.unique())
    start = (earn.report_date.min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (earn.report_date.max() + pd.Timedelta(days=70)).strftime("%Y-%m-%d")
    panels = N.build_universe_panels(syms, start, end, workers=10,
                                     cache_dir=args.cache_dir or None, ns="earn", label="EARN")
    panels, earn = E.merge_share_classes(panels, earn)  # match the study's universe

    payload = build_payload(ev)
    paths = build_paths(ev, panels["adjclose"])
    html = (Path(args.template).read_text()
            .replace("/*__PAYLOAD__*/", json.dumps(payload))
            .replace("/*__PATHS__*/", json.dumps(paths)))
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({len(html)//1024} KB, {len(paths)} names with paths)")

    # The template is body content designed for the Artifact host (no <html>/<head>
    # wrapper). For GitHub Pages we emit a standalone document around it.
    if args.docs_out:
        doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>DPI into Earnings vs Post-Earnings Performance</title></head>'
               '<body>\n' + html + '\n</body></html>\n')
        Path(args.docs_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.docs_out).write_text(doc)
        print(f"wrote {args.docs_out} (standalone for Pages)")


if __name__ == "__main__":
    main()
