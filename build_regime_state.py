#!/usr/bin/env python3
"""
Nightly regime-state strip: docs/regime_state.html + docs/regime_state.json.
============================================================================

Computes each index's CURRENT comovement/DIX regime state (rolling-basis
zones from `intra_index_regime_study`) and evaluates the pre-registered
rules in `frozen_rules.json` against it, so the research has a live,
dated surface and an auditable forward record:

  * docs/regime_state.html -- one-line-per-index strip (zone, episode age,
    gauge percentile, lag-1 DIX zone, N-of-3 dispersed) plus the rule
    readings, linked from the regime log page;
  * docs/regime_state.json -- the same as a machine row. The scoring
    workflow (.github/workflows/regime_score.yml) appends it nightly to
    the `regime-scoring-data` branch and resolves 21-session outcomes
    from the log's own closes (scripts/score_regime_log.py).

Degrades gracefully: if the SPX/IWM basket build fails (offline, cold
cache) the page still ships with NDX-only state and says so. All reads
respect the study's trailing-completeness guard, so the state is never
computed off a partial session.

Optionally reads a committed operational envelope
(`regime_path_study.py --envelopes regime_path_envelopes.json`) and, when
the CURRENT cell (this index's zone / lag-1 DIX zone) has one and clears
its own print gate, adds an "expected path" line to the strip: the
committed p10/p50/p90 at a few checkpoints, the q25 MAE and the playbook
sizing weight, dated by the envelope's own `generated`/`payload_generated`
stamps (which are almost certainly OLDER than tonight's build -- the
envelope is regenerated on demand, not nightly, by design). Absent,
missing, or malformed envelope file -> no line, no failure; this build
never regenerates the envelope itself.

Usage (mirrors the other page builds):
    python build_regime_state.py --html docs/index.html \
        --docs-out docs/regime_state.html --json-out docs/regime_state.json \
        --cache-dir .ndx_dark_cache --envelopes regime_path_envelopes.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import intra_index_regime_study as R

RULES_PATH = Path(__file__).resolve().parent / "frozen_rules.json"


def zone_age(zone_series):
    """Consecutive sessions the series has spent in its CURRENT zone."""
    z = zone_series.dropna()
    if not len(z):
        return 0
    cur = z.iloc[-1]
    age = 0
    for v in z.iloc[::-1]:
        if v != cur:
            break
        age += 1
    return age


def index_state(M, proxy_close=None):
    """State dict for one index frame (already completeness-guarded)."""
    M = M[M["avg_corr"].notna() & M["dix5"].notna()]
    last = M.iloc[-1]
    pct = R.rolling_pctile(M["avg_corr"]).iloc[-1]
    st = {
        "asof": M.index[-1].strftime("%Y-%m-%d"),
        "avg_corr": round(float(last["avg_corr"]), 3),
        "corr_pct": round(float(pct), 2) if np.isfinite(pct) else None,
        "zone": str(last["cz_roll"]),
        "zone_age": zone_age(M["cz_roll"]),
        "dix5": round(float(last["dix5"]), 4),
        "dz_roll": str(last["dz_roll"]),
        "dz_roll_l1": str(last["dz_roll_l1"]),
        "breadth": round(float(last["breadth"]), 2),
    }
    if proxy_close is not None:
        s = proxy_close.dropna()
        s = s[s.index <= M.index[-1]]
        if len(s):
            st["proxy_close"] = round(float(s.iloc[-1]), 4)
            st["proxy_close_date"] = s.index[-1].strftime("%Y-%m-%d")
    return st


def ndx_tilt_screen(P, asof, q=0.80):
    """Top-tilt NDX names on the as-of date (payload panel, current
    members): the rule-A avoid list. Empty when tilt is undefined."""
    rel = P["rel"]
    dates = pd.to_datetime(rel["dates"])
    dpan = R.panel_from(rel, "d", dates)
    names = [c for c in dpan.columns if c != P["bench"]]
    d5 = dpan[names].rolling(5, min_periods=3).mean()
    tilt = d5 - dpan[names].expanding(min_periods=R.TILT_MIN_OBS).mean()
    tilt = tilt[tilt.index <= pd.Timestamp(asof)]
    if not len(tilt):
        return []
    row = tilt.iloc[-1].dropna()
    if len(row) < R.TILT_MIN_NAMES:
        return []
    cut = row.quantile(q)
    return sorted(row[row >= cut].index)


def load_envelopes(path):
    """The committed operational envelope JSON, or None (never raises) if
    `path` is falsy, the file is absent, or it fails to parse."""
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                   # noqa: BLE001
        print(f"regime_state: envelope file unreadable ({e})", file=sys.stderr)
        return None


def envelope_for_state(envelopes, index_name, state):
    """The current cell's committed path envelope for `index_name`, keyed
    off `state`'s own zone/dz_roll_l1 ('LowCorr' or 'LowCorr|DIXLow',
    matching `regime_path_study.build_envelopes`'s key convention). None
    (no line, no failure) when there is no envelope file, it doesn't cover
    this index, or the current cell is below its own print gate."""
    if not envelopes:
        return None
    idx_env = envelopes.get("indices", {}).get(index_name)
    if not idx_env:
        return None
    zone, dz = state.get("zone"), state.get("dz_roll_l1")
    if not zone:
        return None
    key = f"{zone}|{dz}" if dz and f"{zone}|{dz}" in idx_env else zone
    cell = idx_env.get(key)
    if not cell or not cell.get("gate"):
        return None
    out = dict(cell)
    out["cell"] = key
    out["envelope_generated"] = envelopes.get("generated")
    out["envelope_payload_generated"] = envelopes.get("payload_generated")
    return out


def eval_rules(states, screen):
    """Evaluate frozen_rules.json against the state dicts."""
    ndx = states.get("NDX") or {}
    low = {k: (v or {}).get("zone") == "LowCorr" for k, v in states.items()}
    return {
        "ndx_tilt_screen_v1": {
            "active": bool(low.get("NDX")),
            "names": screen if low.get("NDX") else [],
        },
        "ndx_dixlow_caution_v1": {
            "active": bool(low.get("NDX")) and ndx.get("dz_roll_l1") == "DIXLow",
        },
        "all_dispersed_derisk_v1": {
            "active": all(low.get(k, False) for k in ("NDX", "SPX", "IWM")),
            "evaluable": all(k in states for k in ("NDX", "SPX", "IWM")),
        },
    }


def render_envelope_line(name, env):
    return (f"<li><b>{name}</b> expected path ({env['cell']}, 21d): "
           f"p10 {env['p10'].get('21', '--')}%  p50 {env['p50'].get('21', '--')}%  "
           f"p90 {env['p90'].get('21', '--')}%   MAE q25 {env['mae_q25']}%   "
           f"size/1% risk {env['size_q25']}%   "
           f"<span class=dim>(envelope as of {env.get('envelope_generated', '?')}, "
           f"payload {env.get('envelope_payload_generated', '?')})</span></li>")


def render_html(state):
    dot = {"LowCorr": "#e0a458", "MidCorr": "#8a8f98", "HighCorr": "#c05d5d"}
    rows = []
    for k in ("NDX", "SPX", "IWM"):
        s = state["indices"].get(k)
        if not s:
            rows.append(f"<tr><td>{k}</td><td colspan=7 class=dim>"
                        "unavailable this build</td></tr>")
            continue
        rows.append(
            f"<tr><td><b>{k}</b></td>"
            f"<td><span class=dot style='background:{dot.get(s['zone'], '#888')}'>"
            f"</span>{s['zone']}</td>"
            f"<td>{s['zone_age']}d</td>"
            f"<td>{s['avg_corr']:.2f} (p{'' if s['corr_pct'] is None else int(100 * s['corr_pct'])})</td>"
            f"<td>{s['breadth']:.2f}</td>"
            f"<td>{s['dix5']:.3f}</td>"
            f"<td>{s['dz_roll_l1']}</td>"
            f"<td class=dim>{s['asof']}</td></tr>")
    ru = state["rules"]
    screen = ru["ndx_tilt_screen_v1"]["names"]
    flags = []
    flags.append("<li><b>ndx_tilt_screen_v1</b>: "
                 + ("ACTIVE -- avoid: " + ", ".join(screen)
                    if ru["ndx_tilt_screen_v1"]["active"] and screen
                    else ("ACTIVE (no names above cut)"
                          if ru["ndx_tilt_screen_v1"]["active"] else "inactive"))
                 + "</li>")
    flags.append("<li><b>ndx_dixlow_caution_v1</b>: "
                 + ("ACTIVE" if ru["ndx_dixlow_caution_v1"]["active"] else "inactive")
                 + "</li>")
    a3 = ru["all_dispersed_derisk_v1"]
    flags.append("<li><b>all_dispersed_derisk_v1</b>: "
                 + ("ACTIVE" if a3["active"] else
                    ("inactive" if a3.get("evaluable", True) else
                     "not evaluable (missing an index this build)")) + "</li>")
    env_lines = [render_envelope_line(k, s["envelope"])
                for k in ("NDX", "SPX", "IWM")
                for s in [state["indices"].get(k)] if s and s.get("envelope")]
    envelope_section = (f"<h1>Expected path (committed envelope)</h1><ul>{''.join(env_lines)}"
                        "</ul><p class=note>From regime_path_study.py's committed "
                        "regime_path_envelopes.json -- an IN-SAMPLE estimate, regenerated on "
                        "demand rather than nightly on purpose; the dates above may lag "
                        "tonight's build by design. See REGIME_PATH_FINDINGS.md.</p>"
                        if env_lines else "")
    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regime state</title>
<style>
body{{background:#14161a;color:#cfd3da;font:14px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif;
     max-width:880px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.15rem;color:#e8eaee}} a{{color:#7aa2c9;text-decoration:none}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
td,th{{padding:.45rem .6rem;border-bottom:1px solid #262a31;text-align:left}}
th{{color:#8a8f98;font-weight:500}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.45rem}}
.dim{{color:#8a8f98}} ul{{padding-left:1.2rem}} li{{margin:.3rem 0}}
.note{{color:#8a8f98;font-size:.85rem;margin-top:1.5rem}}
</style>
<h1>Intra-index regime state</h1>
<p class=dim>Rolling-basis (504-session) comovement and DIX zones; DIX zone on the
one-session-lagged signal. Generated {state["generated"]} &middot;
<a href="regime_log.html">&larr; regime log</a></p>
<table>
<tr><th>index</th><th>comovement zone</th><th>age</th><th>avg corr (pct)</th>
<th>breadth</th><th>DIX5</th><th>DIX zone (lag-1)</th><th>as of</th></tr>
{''.join(rows)}
</table>
<h1>Pre-registered rule readings</h1>
<ul>{''.join(flags)}</ul>
{envelope_section}
<p class=note>Rules are frozen in <code>frozen_rules.json</code>
(sha256 {state["rules_hash"][:12]}&hellip;, frozen {state["rules_frozen"]}); readings
and realized 21-session outcomes accumulate on the <code>regime-scoring-data</code>
branch. See INTRA_INDEX_REGIME_FINDINGS.md for what these rules are testing --
the in-sample estimates behind them are one-cycle evidence, and this forward
log is the arbiter.</p>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="docs/index.html")
    ap.add_argument("--docs-out", default="docs/regime_state.html")
    ap.add_argument("--json-out", default="docs/regime_state.json")
    ap.add_argument("--cache-dir", default=".ndx_dark_cache")
    ap.add_argument("--basket-size", type=int, default=R.BASKET_N)
    ap.add_argument("--envelopes", default="regime_path_envelopes.json",
                    help="committed operational envelope from regime_path_study.py "
                         "--envelopes (optional; absent/unreadable file -> no envelope "
                         "line, no failure)")
    args = ap.parse_args()

    P = R.load_payload(args.html)
    rules_text = RULES_PATH.read_text(encoding="utf-8")
    rules = json.loads(rules_text)

    # proxy closes for the scoring log (tiny cached fetch; NDX proxy comes
    # from the payload itself so the NDX row works fully offline)
    rel = P["rel"]
    dates = pd.to_datetime(rel["dates"])
    qqq_close = R.panel_from(rel, "close", dates)[P["bench"]]
    proxies = {"NDX": qqq_close, "SPX": None, "IWM": None}
    try:
        import ndx_dark_residual as N
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        adj = N.load_yahoo_panels(["SPY", "IWM"], R.FETCH_START, end,
                                  cache_dir=args.cache_dir or None,
                                  label="REGIME-STATE")["adjclose"]
        proxies["SPX"], proxies["IWM"] = adj.get("SPY"), adj.get("IWM")
    except Exception as e:                                       # noqa: BLE001
        print(f"regime_state: proxy close fetch failed ({e})", file=sys.stderr)

    states = {}
    M_ndx, _ = R.build_ndx_frame(P)
    states["NDX"] = index_state(M_ndx, qqq_close)
    for name in ("SPX", "IWM"):
        try:
            M, _ = R.build_basket_frame(P, name, args.basket_size,
                                        args.cache_dir)
            states[name] = index_state(M, proxies[name])
        except Exception as e:                                   # noqa: BLE001
            print(f"regime_state: {name} unavailable ({e})", file=sys.stderr)

    envelopes = load_envelopes(args.envelopes)
    for name, st in states.items():
        env = envelope_for_state(envelopes, name, st)
        if env:
            st["envelope"] = env

    screen = ndx_tilt_screen(P, states["NDX"]["asof"])
    state = {
        "schema": 1,
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "rules_hash": hashlib.sha256(rules_text.encode()).hexdigest(),
        "rules_frozen": rules.get("frozen_utc"),
        "indices": states,
        "rules": eval_rules(states, screen),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(state, indent=1), encoding="utf-8")
    Path(args.docs_out).write_text(render_html(state), encoding="utf-8")
    print(f"regime_state: {len(states)}/3 indices; "
          f"rules active: "
          + ", ".join(k for k, v in state["rules"].items() if v.get("active"))
          or "none", file=sys.stderr)


if __name__ == "__main__":
    main()
