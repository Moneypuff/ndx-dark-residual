#!/usr/bin/env python3
"""
Optimizing the index-level DIX signal -- honestly (walk-forward, not curve-fit)
==============================================================================

Request: "how can I optimize the tradeable signal at the index level?" The
index-level signals that survived the earlier studies are the DIX-vs-price
divergence (option B, on the dashboard) and the stochastic %D oscillator read
directionally. "Optimization" here is dangerous -- a parameter sweep on the full
sample WILL find something that looks better and IS overfit. So this study is
built the other way round:

  1. A CONFIG GRID over the defensible knobs: divergence lookback (10..63d),
     %D window (63/126/252) x threshold (50..80), OR/AND combinations, and a
     GEX>0 gate. Each config is a daily LONG/FLAT position in SPX (the bearish
     side was weak in the base study, so short exposure is out of scope), with
     position taken at the close AFTER the signal day (no look-ahead) and a
     per-switch transaction cost.
  2. The HEADLINE test is WALK-FORWARD: each January, pick the config with the
     best Sharpe on all data up to that date, trade it for the following year.
     The walk-forward line is out of sample BY CONSTRUCTION -- it is what
     "optimizing" would actually have delivered, picks and mistakes included.
  3. Benchmarks: buy-and-hold, and the SHIPPED default (21d divergence as on
     the dashboard). If walk-forward cannot beat those, tuning is not worth it.
  4. Full-sample landscape + IS(<2023)/OOS(>=2023) split are reported for
     context, with block-bootstrap CIs on the timing edge (in-market daily mean
     minus unconditional daily mean).

Reproduce
---------
    python spx_index_signal_opt_study.py --out spx_index_signal_opt.csv
    (add --csv DIX.csv to use a local snapshot)
"""
import argparse

import numpy as np
import pandas as pd

import spx_dix_decile_vs_breakout_study as B   # load_dix, oscillators, divergence, block_boot_ci

TC_BPS = 2.0                 # one-way cost per position switch, bps
WF_START = 2021              # first walk-forward trading year
SPLIT_DATE = "2023-01-01"
DIV_LOOKBACKS = (10, 15, 21, 31, 42, 63)
OSC_WINDOWS = (63, 126, 252)
OSC_THRESHOLDS = (50, 60, 70, 80)
DEFAULT_CONFIG = "DIV21"     # what the dashboard ships today


# ---------------------------------------------------------------------------
# Config grid: each entry maps the raw frame -> a 0/1 daily position series
# ---------------------------------------------------------------------------
def entry_hold_pos(active, hold):
    """1 for `hold` sessions starting at each FIRST day `active` turns true
    (entries inside an open hold just refresh it). With strat_returns' one-day
    lag this is 'buy at the entry day's close, hold `hold` sessions' -- the
    shape the earlier event studies actually measured, unlike holding only
    while the state stays true."""
    a = active.fillna(False)
    ent = a & ~a.shift(1, fill_value=False)
    return ent.astype(float).rolling(hold, min_periods=1).max().fillna(0.0)


HOLD_HORIZONS = (10, 21, 42)


def build_configs(df):
    dix, price, gex = df["dix"], df["price"], df["gex"]
    pos = {}
    div = {}
    for L in DIV_LOOKBACKS:
        st, _, _ = B.dix_price_divergence(dix, price, lookback=L)
        div[L] = st
        pos[f"DIV{L}"] = (st == 1).astype(float)
        for H in HOLD_HORIZONS:
            pos[f"DIV{L}H{H}"] = entry_hold_pos(st == 1, H)
    osc = {}
    for w in OSC_WINDOWS:
        osc[w] = B.stochastic_osc(dix, win=w)
        for t in OSC_THRESHOLDS:
            pos[f"OSC{w}>{t}"] = (osc[w] >= t).astype(float)
    # oscillator threshold-crossing entries with a hold (the event-study shape)
    for w, t in ((126, 70), (126, 80)):
        pos[f"OSC{w}x{t}H21"] = entry_hold_pos(osc[w] >= t, 21)
    # combinations around the shipped pieces
    pos["DIV21H21+GEXentry"] = entry_hold_pos((div[21] == 1) & (gex > 0), 21)
    pos["DIV21|OSC126>70"] = ((div[21] == 1) | (osc[126] >= 70)).astype(float)
    pos["DIV21&OSC126>50"] = ((div[21] == 1) & (osc[126] >= 50)).astype(float)
    gate = (gex > 0).astype(float)
    for base in ("DIV21", "OSC126>70", "DIV21|OSC126>70"):
        pos[f"{base}+GEX"] = pos[base] * gate
    pos["HOLD"] = pd.Series(1.0, index=df.index)          # buy-and-hold benchmark
    return pos


# ---------------------------------------------------------------------------
# Strategy accounting
# ---------------------------------------------------------------------------
def strat_returns(pos, mkt_ret, cost_bps=TC_BPS):
    """Daily net strategy returns. The position is decided at the close of day
    t and earns day t+1's market return (shift(1) -- no look-ahead); each
    position CHANGE pays `cost_bps` one way."""
    p = pos.shift(1).fillna(0.0)
    turns = p.diff().abs().fillna(p.abs())
    return p * mkt_ret - turns * cost_bps / 1e4, p


def perf(net, held):
    """Annualized performance of a daily net-return series."""
    net, held = net.dropna(), held.reindex(net.index)
    if len(net) < 60:
        return {"days": int(len(net)), "sharpe": np.nan}
    eq = (1.0 + net).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min() * 100)
    vol = float(net.std() * np.sqrt(252))
    inmkt = net[held > 0]
    return {"days": int(len(net)), "exposure": round(float(held.mean()) * 100, 1),
            "cagr": round((float(eq.iloc[-1]) ** (252 / len(net)) - 1.0) * 100, 2),
            "vol": round(vol * 100, 2),
            "sharpe": round(float(net.mean() * 252 / vol), 2) if vol > 0 else np.nan,
            "maxdd": round(dd, 1),
            "hit": round(float((inmkt > 0).mean()) * 100, 1) if len(inmkt) else np.nan}


def timing_edge(pos, mkt_ret, seed=0):
    """In-market daily mean minus unconditional daily mean (bps/day), with a
    block-bootstrap CI -- the exposure-agnostic 'was being in on those days
    better than being in on average days' test."""
    p = pos.shift(1).reindex(mkt_ret.index)
    r = mkt_ret[p.notna() & mkt_ret.notna()]
    p = p[r.index]
    inm = r[p > 0]
    if len(inm) < 60:
        return {"edge_bps": np.nan, "ci": None}
    diff = (r.where(p > 0) - r.mean()).dropna() * 1e4
    ci = B.block_boot_ci(diff.to_numpy(), seed=seed)
    return {"edge_bps": round(float(inm.mean() - r.mean()) * 1e4, 2),
            "ci": (round(ci[0], 2), round(ci[1], 2))}


def _slice(s, lo=None, hi=None):
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index < pd.Timestamp(hi)]
    return s


def sharpe_on(pos, mkt_ret, lo=None, hi=None):
    net, held = strat_returns(pos, mkt_ret)
    net = _slice(net, lo, hi)
    if len(net) < 60 or net.std() == 0:
        return np.nan
    return float(net.mean() * 252 / (net.std() * np.sqrt(252)))


# ---------------------------------------------------------------------------
# Walk-forward: re-pick the best config each January on all data so far
# ---------------------------------------------------------------------------
def walk_forward(df, configs, mkt_ret, start_year=WF_START, exclude=("HOLD",)):
    """Each Jan 1 from `start_year`, rank candidate configs by Sharpe on ALL
    data before that date and trade the winner for the calendar year. Returns
    the stitched position series and the pick per year."""
    years = range(start_year, int(df.index[-1].year) + 1)
    stitched = pd.Series(np.nan, index=df.index)
    picks = {}
    cands = [k for k in configs if k not in exclude]
    for y in years:
        cut = f"{y}-01-01"
        scored = {k: sharpe_on(configs[k], mkt_ret, hi=cut) for k in cands}
        best = max(scored, key=lambda k: (np.nan_to_num(scored[k], nan=-9e9)))
        picks[y] = (best, round(scored[best], 2))
        sel = (df.index >= cut) & (df.index < f"{y + 1}-01-01")
        stitched[sel] = configs[best][sel]
    return stitched.dropna(), picks


# ---------------------------------------------------------------------------
def landscape(df, configs, mkt_ret, split=SPLIT_DATE):
    rows = []
    for i, (name, pos) in enumerate(configs.items()):
        net, held = strat_returns(pos, mkt_ret)
        row = {"config": name, **perf(net, held)}
        row["edge"] = timing_edge(pos, mkt_ret, seed=i)
        row["sharpe_is"] = round(sharpe_on(pos, mkt_ret, hi=split), 2)
        row["sharpe_oos"] = round(sharpe_on(pos, mkt_ret, lo=split), 2)
        rows.append(row)
    return rows


def _ci(ci):
    return f"[{ci[0]:+.1f},{ci[1]:+.1f}]" if ci else "--"


def report(df, configs, mkt_ret, split=SPLIT_DATE):
    out = []
    out.append("=" * 100)
    out.append("1) CONFIG LANDSCAPE (full sample, net of costs)  --  context only; "
               "the walk-forward below is the honest test")
    out.append("=" * 100)
    out.append(f"   {'config':18s} {'expo%':>5} {'CAGR%':>6} {'Sharpe':>6} {'maxDD%':>6} "
               f"{'hit%':>5}  {'edge bps/d':>10} {'95% CI':>14}  {'Shp IS':>6} {'Shp OOS':>7}")
    rows = landscape(df, configs, mkt_ret, split=split)
    for r in sorted(rows, key=lambda x: -(x.get("sharpe") or -9)):
        e = r["edge"]
        out.append(f"   {r['config']:18s} {r.get('exposure', float('nan')):>5} "
                   f"{r.get('cagr', float('nan')):>6} {r.get('sharpe', float('nan')):>6} "
                   f"{r.get('maxdd', float('nan')):>6} {r.get('hit', float('nan')):>5}  "
                   f"{e['edge_bps']:>10} {_ci(e['ci']):>14}  {r['sharpe_is']:>6} {r['sharpe_oos']:>7}")
    out.append("\n" + "=" * 100)
    out.append(f"2) WALK-FORWARD (re-pick best-Sharpe config each Jan, trade it the "
               f"next year; trading {WF_START}+)")
    out.append("=" * 100)
    wf_pos, picks = walk_forward(df, configs, mkt_ret)
    for y, (name, s) in picks.items():
        out.append(f"   {y}: picked {name:18s} (trailing Sharpe {s:+.2f})")
    lo = f"{WF_START}-01-01"
    comps = {"WALK-FORWARD": wf_pos,
             f"fixed {DEFAULT_CONFIG} (shipped)": configs[DEFAULT_CONFIG],
             "buy-and-hold": configs["HOLD"]}
    out.append(f"   {'strategy':28s} {'expo%':>5} {'CAGR%':>6} {'Sharpe':>6} "
               f"{'maxDD%':>6} {'hit%':>5}  {'edge bps/d':>10} {'95% CI':>14}")
    for name, pos in comps.items():
        pos_w = _slice(pos, lo=lo)
        net, held = strat_returns(pos_w, _slice(mkt_ret, lo=lo))
        p = perf(net, held)
        e = timing_edge(pos_w, _slice(mkt_ret, lo=lo), seed=99)
        out.append(f"   {name:28s} {p.get('exposure', float('nan')):>5} "
                   f"{p.get('cagr', float('nan')):>6} {p.get('sharpe', float('nan')):>6} "
                   f"{p.get('maxdd', float('nan')):>6} {p.get('hit', float('nan')):>5}  "
                   f"{e['edge_bps']:>10} {_ci(e['ci']):>14}")
    return "\n".join(out), rows, picks


def summary_rows(rows, picks):
    recs = []
    for r in rows:
        e = r.pop("edge")
        recs.append({**r, "edge_bps": e["edge_bps"],
                     "edge_ci_lo": (e["ci"] or [None, None])[0],
                     "edge_ci_hi": (e["ci"] or [None, None])[1]})
    for y, (name, s) in picks.items():
        recs.append({"config": f"WF_pick_{y}", "sharpe": s, "days": None,
                     "exposure": None, "cagr": None, "vol": None, "maxdd": None,
                     "hit": None, "sharpe_is": None, "sharpe_oos": None,
                     "edge_bps": None, "edge_ci_lo": None, "edge_ci_hi": None,
                     "note": name})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="local DIX.csv snapshot")
    ap.add_argument("--split-date", default=SPLIT_DATE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = B.load_dix(csv=args.csv)
    mkt_ret = df["price"].pct_change()
    configs = build_configs(df)
    print(f"DIX {df.index[0].date()} .. {df.index[-1].date()}  ({len(df)} sessions), "
          f"{len(configs)} configs, cost {TC_BPS:.0f}bps/switch")
    text, rows, picks = report(df, configs, mkt_ret, split=args.split_date)
    print(text)
    if args.out:
        summary_rows(rows, picks).to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
