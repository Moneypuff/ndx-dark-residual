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
        rm, pm, nm = E._pearson(d.dpi10, d.m1_ret); rn, pn, _ = E._pearson(d.dpi10, d.next_day_ret)
        if nm < 5 or not np.isfinite(rm):
            continue   # skip years with no usable DPI (e.g. a year still missing FINRA data)
        pay["period"].append(dict(year=y, n=int(nm),
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


PRE_PATH = 10   # sessions of DPI shown before the base close T in the event path


def build_dpichange(ev, dpi):
    """Payload for the post-earnings DPI-change section.

    Everything is expressed in percentage points of the DPI ratio, measured against
    each event's own pre-earnings DPI10, so the numbers read as "dark-pool short
    share ran N pp above/below where it was going into the print".
    """
    ev = ev.copy()
    lv = {k: float(ev[c].mean()) for k, c in
          [("pre", "dpi10"), ("base60", "dpi_base60"), ("d1", "dpi_post_next_day"),
           ("w1", "dpi_post_w1"), ("w2", "dpi_post_w2"), ("m1", "dpi_post_m1")]}
    HZ = [("next_day", "Reaction day"), ("w1", "1 week"), ("w2", "2 weeks"), ("m1", "1 month")]
    horizons = []
    for k, label in HZ:
        c = ev[f"d_dpi_{k}"]
        m, t, p, n = E._ttest1(c)
        horizons.append(dict(key=k, label=label, chg=float(m), med=float(c.median()),
                             up=float((c > 0).mean()), t=float(t), p=float(p), n=int(n),
                             post=float(ev[f"dpi_post_{k}"].mean())))
    slices = []
    for k, label in [("sw1", "Week 1"), ("sw2", "Week 2"), ("sw34", "Weeks 3–4")]:
        m, t, p, n = E._ttest1(ev[f"d_dpi_{k}"])
        slices.append(dict(key=k, label=label, chg=float(m), t=float(t), p=float(p), n=int(n)))

    revert, reaction, predict = [], [], []
    for k, label in HZ[1:]:
        c = ev[f"d_dpi_{k}"]
        rn, pn, n = E._pearson(ev["dpi10"], c)
        ri, pi, _ = E._pearson(ev["dpi_prior"], c)
        pp = ev["dpi_prior_pct"]
        hh, ll = ev[pp >= 2/3][f"d_dpi_{k}"].to_numpy(), ev[pp <= 1/3][f"d_dpi_{k}"].to_numpy()
        diff, t, pv, _, _ = E._welch(hh, ll)
        revert.append(dict(key=k, label=label, r_naive=float(rn), r_prior=float(ri),
                           p_prior=float(pi), hi=float(np.nanmean(hh)), lo=float(np.nanmean(ll)),
                           spread=float(diff), t=float(t), p=float(pv), n=int(n)))
        up = ev[ev.next_day_ret > 0][f"d_dpi_{k}"].to_numpy()
        dn = ev[ev.next_day_ret <= 0][f"d_dpi_{k}"].to_numpy()
        diff, t, pv, nu, nd = E._welch(up, dn)
        reaction.append(dict(key=k, label=label, up=float(np.nanmean(up)), dn=float(np.nanmean(dn)),
                             diff=float(diff), t=float(t), p=float(pv), nu=int(nu), nd=int(nd)))
    for k, label, rcol, rlab in [("w1", "1 week", "fwd_w1_m1", "T+5 → T+21"),
                                 ("w2", "2 weeks", "fwd_w2_m1", "T+10 → T+21"),
                                 ("m1", "1 month", "fwd_m1_m2", "T+21 → T+42")]:
        r, p, n = E._pearson(ev[f"d_dpi_{k}"], ev[rcol])
        lr, lp, _ = E._pearson(ev["dpi10"], ev[rcol])
        pc = ev[f"d_dpi_{k}_pct"]
        hh, ll = ev[pc >= 2/3][rcol].to_numpy(), ev[pc <= 1/3][rcol].to_numpy()
        diff, t, pv, _, _ = E._welch(hh, ll)
        predict.append(dict(key=k, label=label, window=rlab, r=float(r), p=float(p), n=int(n),
                            level_r=float(lr), level_p=float(lp), hi=float(np.nanmean(hh)),
                            lo=float(np.nanmean(ll)), spread=float(diff), t=float(t), p_spread=float(pv)))

    # --- average DPI path around the report, T-10 .. T+21, vs each event's DPI10 ---
    idx = dpi.index
    ev["base_T"] = pd.to_datetime(ev.base_T)
    span = PRE_PATH + H + 1
    rows, tags = [], []
    for _, r in ev.iterrows():
        if r.ticker not in dpi.columns or pd.isna(r.base_T) or not np.isfinite(r.dpi10):
            continue
        pos = idx.searchsorted(r.base_T)
        if pos >= len(idx) or idx[pos] != r.base_T or pos - PRE_PATH < 0 or pos + H >= len(idx):
            continue
        seg = dpi[r.ticker].iloc[pos - PRE_PATH:pos + H + 1].to_numpy(dtype=float)
        if len(seg) != span or np.isnan(seg).sum() > span // 4:
            continue
        rows.append(seg - float(r.dpi10))
        tags.append((bool(r.next_day_ret > 0),
                     "hi" if r.dpi_prior_pct >= 2/3 else ("lo" if r.dpi_prior_pct <= 1/3 else "mid")))
    M = np.array(rows)

    def curve(sel):
        sub = M[np.array(sel, dtype=bool)] if len(M) else M
        if len(sub) < 20:
            return None
        return [round(float(v) * 100, 3) for v in np.nanmean(sub, axis=0)]
    path = {"x": list(range(-PRE_PATH, H + 1)), "n": int(len(M)),
            "all": curve([True] * len(M)),
            "up": curve([t[0] for t in tags]), "dn": curve([not t[0] for t in tags]),
            "hi": curve([t[1] == "hi" for t in tags]), "lo": curve([t[1] == "lo" for t in tags])}
    return dict(levels=lv, horizons=horizons, slices=slices, revert=revert,
                reaction=reaction, predict=predict, path=path)


def build_paths(ev, adj):
    idx = adj.index
    ev = ev.copy(); ev["base_T"] = pd.to_datetime(ev.base_T)
    out = {}
    for tk, g in ev.groupby("ticker"):
        if tk not in adj.columns:
            continue
        a = adj[tk]; events = []; mat = []; rv_list = []
        rvcols = ["next_day_rvol", "w1_rvol", "w2_rvol", "m1_rvol"]
        have_rv = all(c in g.columns for c in rvcols)
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
            if have_rv:
                rv_list.append((cls, [float(r[c]) if pd.notna(r[c]) else np.nan for c in rvcols]))
        if len(mat) < 5:
            continue
        M = np.array(mat)
        med = [int(round(np.median(M[:, h]))) for h in range(H + 1)]
        mean = [int(round(M[:, h].mean())) for h in range(H + 1)]

        def cmean(cl):
            sub = np.array([e["p"] for e in events if e["cls"] == cl])
            return [int(round(sub[:, h].mean())) for h in range(H + 1)] if len(sub) >= 3 else None

        # mean annualized realized vol per horizon [1d, 1wk, 2wk, 1mo], overall & by DPI tercile
        def rvmean(sel):
            arr = np.array([rv for cl, rv in rv_list if sel(cl)], dtype=float)
            if arr.size == 0:
                return None
            m = np.nanmean(arr, axis=0)
            return [round(float(x), 1) if np.isfinite(x) else None for x in m]
        out[tk] = {"n": len(mat), "events": events, "median": med, "mean": mean,
                   "hi_mean": cmean("hi"), "lo_mean": cmean("lo"),
                   "final_med": med[-1], "final_mean": mean[-1],
                   "pos": int((M[:, -1] > 0).sum()), "tot": len(mat),
                   "rvol": rvmean(lambda c: True) if have_rv else None,
                   "rvol_hi": rvmean(lambda c: c == "hi") if have_rv else None,
                   "rvol_lo": rvmean(lambda c: c == "lo") if have_rv else None}
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
    # same window as earnings_dpi_study.py so both read the identical cached panels
    start = (earn.report_date.min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (earn.report_date.max() + pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    panels = N.build_universe_panels(syms, start, end, workers=10,
                                     cache_dir=args.cache_dir or None, ns="earn", label="EARN")
    panels, earn = E.merge_share_classes(panels, earn)  # match the study's universe

    payload = build_payload(ev)
    payload["dpichg"] = build_dpichange(ev, panels["dpi"].reindex(panels["adjclose"].index))
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
