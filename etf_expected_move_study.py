#!/usr/bin/env python3
"""
Expected-move study -- the option-pricing lens on the post-signal path map.

ETF_PATH_PLAYBOOK.md says what each leaderboard ETF does after its own
signals (conviction up/down breaks, rel-strength turns vs SPY). To express
those trades in options you need the signal-conditional **expected move**:
how big is |forward return| over the next quarter / half-year given the
event, versus the ETF's unconditional move over the same horizon, versus
what the option market currently implies.

For each ETF x family (n >= 8) x horizon (63 / 126 sessions) this prints,
from the empirical forward-return distribution (no model):

  * location -- mean, median, P(>0)
  * move size -- E[|r|] (the straddle-payoff equivalent), sigma, q10-q90
  * move ratio -- conditional E[|r|] / unconditional E[|r|] over all
    overlapping windows: does the signal predict a bigger move, or only
    tilt the direction of a normal-sized one?
  * empirical fair premia, % of spot, for expiry-style payoffs held to the
    horizon: ATM straddle (=E[|r|]), ATM call/put, +5% / +10% OTM calls,
    and the short-put loss odds P(r < -5%), P(r < -10%).

Best-effort live overlay (skipped gracefully when the endpoint is not
reachable, e.g. in CI): Yahoo option chains via the cookie+crumb handshake,
cached per symbol per day in --cache-dir. For expirations nearest ~91 and
~182 calendar days it computes the implied expected move (ATM straddle mid
/ spot; 0.8 * ATM-IV * sqrt(T) when quotes are unusable) and prints implied
vs conditional-historical per family. A final section details any ETF whose
latest event fired within the last 10 sessions -- the trades that are live
right now.

    python etf_expected_move_study.py --cache-dir .ndx_dark_cache
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N
from build_regime_log import (SECTORS, THEMES, conviction_frame,
                              conviction_events, detect_breaks, spread_frames)

HORIZONS = (63, 126)
MIN_N = 8
LIVE_SESSIONS = 10           # "live now" = last event within this many sessions
TARGET_DTE = (91, 182)       # ~3M / ~6M expirations
OTM_STRIKES = (5.0, 10.0)    # % OTM call strikes priced from the distribution

CRUMB_SEED_URL = "https://fc.yahoo.com"
CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
CHAIN_URL = "https://query2.finance.yahoo.com/v7/finance/options/{sym}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def fwd_returns(close, dates, horizon):
    """Forward `horizon`-session %-returns from each event date (skips events
    whose horizon runs past the data)."""
    out = []
    for d in dates:
        i = close.index.get_loc(d)
        if i + horizon < len(close):
            out.append(float(close.iloc[i + horizon] / close.iloc[i] - 1) * 100)
    return np.asarray(out)


def uncond_returns(close, horizon):
    """All overlapping `horizon`-session %-returns (the unconditional base)."""
    r = (close.shift(-horizon) / close - 1).dropna() * 100
    return r.to_numpy()


def dist_stats(rets):
    """Location and move-size stats of a %-return sample."""
    r = np.asarray(rets, float)
    return {"n": len(r), "mean": float(np.mean(r)), "med": float(np.median(r)),
            "pos": float(np.mean(r > 0) * 100), "sigma": float(np.std(r)),
            "eabs": float(np.mean(np.abs(r))),
            "q10": float(np.percentile(r, 10)), "q25": float(np.percentile(r, 25)),
            "q75": float(np.percentile(r, 75)), "q90": float(np.percentile(r, 90))}


def fair_premia(rets, otm=OTM_STRIKES):
    """Expiry-style payoff values (% of spot) under the empirical
    distribution: ATM straddle/call/put, +k% OTM calls, and short-put loss
    odds P(r < -k)."""
    r = np.asarray(rets, float)
    out = {"straddle": float(np.mean(np.abs(r))),
           "call": float(np.mean(np.maximum(r, 0.0))),
           "put": float(np.mean(np.maximum(-r, 0.0)))}
    for k in otm:
        out[f"call{k:g}"] = float(np.mean(np.maximum(r - k, 0.0)))
        out[f"pbelow{k:g}"] = float(np.mean(r < -k) * 100)
    return out


def pick_expiry(expirations, today, target_days):
    """Expiration (epoch seconds) nearest `target_days` calendar days out."""
    if not expirations:
        return None
    t0 = pd.Timestamp(today).timestamp()
    return min(expirations, key=lambda e: abs((e - t0) / 86400 - target_days))


def implied_move(chain, spot, expiry_epoch, today):
    """Implied expected move (% of spot) for one expiration's chain: ATM
    straddle mid when both sides have live bids, else 0.8*IV*sqrt(T)."""
    calls = chain.get("calls") or []
    puts = chain.get("puts") or []
    if not calls or not puts or not spot:
        return None

    def atm(side):
        return min(side, key=lambda c: abs(c.get("strike", 1e9) - spot))

    c, p = atm(calls), atm(puts)
    cb, ca = c.get("bid") or 0, c.get("ask") or 0
    pb, pa = p.get("bid") or 0, p.get("ask") or 0
    if min(cb, ca, pb, pa) > 0:
        return float(((cb + ca) / 2 + (pb + pa) / 2) / spot * 100)
    ivs = [v for v in (c.get("impliedVolatility"), p.get("impliedVolatility"))
           if v and v > 1e-3]
    if not ivs:
        return None
    t_years = max((expiry_epoch - pd.Timestamp(today).timestamp()) / 86400, 1) / 365.25
    return float(0.8 * np.mean(ivs) * math.sqrt(t_years) * 100)


# ----------------------------------------------------------------------------
# Live option chains (best-effort)
# ----------------------------------------------------------------------------
def fetch_chains(symbols, cache_dir, refresh=False, targets=TARGET_DTE):
    """{sym: {"spot": float, "implied": {dte: move%}}} from Yahoo option
    chains, cached one JSON per symbol per day. Returns {} when the
    endpoint is unreachable (the study degrades to historical-only)."""
    if N.requests is None:
        return {}
    today = str(pd.Timestamp.today().date())
    cache = Path(cache_dir) if cache_dir else None
    sess = N.make_session(4)
    sess.headers.update(UA)
    crumb = None

    def get_crumb():
        nonlocal crumb
        if crumb is None:
            try:
                sess.get(CRUMB_SEED_URL, timeout=15)
            except Exception:  # noqa: BLE001 -- 404 still sets the cookie
                pass
            crumb = sess.get(CRUMB_URL, timeout=15).text.strip()
        return crumb

    out = {}
    for sym in symbols:
        cpath = cache / f"optchain_{sym}.json" if cache else None
        payload = None
        if cpath and cpath.exists() and not refresh:
            try:
                cached = json.loads(cpath.read_text())
                if cached.get("_synced") == today:
                    payload = cached
            except Exception:  # noqa: BLE001
                payload = None
        if payload is None:
            try:
                base = sess.get(CHAIN_URL.format(sym=sym),
                                params={"crumb": get_crumb()}, timeout=20).json()
                res = base["optionChain"]["result"][0]
                spot = res["quote"]["regularMarketPrice"]
                exps = res.get("expirationDates") or []
                chains = {}
                for dte in targets:
                    e = pick_expiry(exps, pd.Timestamp.today(), dte)
                    if e is None:
                        continue
                    full = sess.get(CHAIN_URL.format(sym=sym),
                                    params={"date": e, "crumb": get_crumb()},
                                    timeout=20).json()
                    opts = full["optionChain"]["result"][0]["options"]
                    if opts:
                        chains[str(dte)] = {"expiry": e, "calls": opts[0].get("calls"),
                                            "puts": opts[0].get("puts")}
                payload = {"_synced": today, "spot": spot, "chains": chains}
                if cpath:
                    cpath.write_text(json.dumps(payload))
            except Exception as exc:  # noqa: BLE001
                print(f"  [options] {sym}: {type(exc).__name__} -- skipping live overlay")
                continue
        imp = {}
        for dte, ch in payload.get("chains", {}).items():
            m = implied_move(ch, payload.get("spot"), ch.get("expiry"),
                             pd.Timestamp.today())
            if m is not None:
                imp[int(dte)] = m
        if imp:
            out[sym] = {"spot": payload.get("spot"), "implied": imp}
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--no-options", action="store_true",
                    help="skip the live option-chain overlay")
    args = ap.parse_args()

    univ = {**SECTORS, **THEMES}
    syms = sorted(set(univ) | {"SPY"})
    panels = N.load_yahoo_panels(syms, "2004-01-01", pd.Timestamp.today().normalize(),
                                 cache_dir=args.cache_dir or None,
                                 refresh=args.refresh, label="EXPMOVE")
    adj = panels["adjclose"].dropna(how="all")
    volp = panels["volume"]
    spy = adj["SPY"].dropna()

    rows, live = [], []
    for t in sorted(univ):
        if t not in adj:
            continue
        c = adj[t].dropna()
        conv = conviction_frame(c, volp[t])
        ev = conviction_events(c, conv, min_score=3, cooldown=21, horizon=63)
        s21, s63, z, _ = spread_frames(c, spy)
        fams = {"up": list(ev[ev["dir"] == 1]["date"]),
                "dn": list(ev[ev["dir"] == -1]["date"]),
                "turn": [d for d, s in detect_breaks(s63, z) if s == -1]}
        base = {h: dist_stats(uncond_returns(c, h)) for h in HORIZONS}
        for fam, dates in fams.items():
            if dates and (len(c) - 1 - c.index.get_loc(dates[-1])) <= LIVE_SESSIONS:
                live.append((t, fam, dates[-1]))
            for h in HORIZONS:
                r = fwd_returns(c, dates, h)
                if len(r) < MIN_N:
                    continue
                st = dist_stats(r)
                st.update(fair_premia(r))
                st.update(etf=t, family=fam, horizon=h,
                          base_eabs=base[h]["eabs"],
                          ratio=st["eabs"] / base[h]["eabs"])
                rows.append(st)

    R = pd.DataFrame(rows).set_index(["etf", "family", "horizon"])

    for fam, label in (("up", "UP-BREAKS (chase)"), ("dn", "DOWN-BREAKS (capitulation)"),
                       ("turn", "TURN SIGNALS")):
        try:
            F = R.xs(fam, level="family")
        except KeyError:
            continue
        print(f"\n== {label}: conditional expected move vs unconditional ==")
        print(f"{'etf':<5}{'h':>4}{'n':>5}{'med':>7}{'mean':>7}{'P>0':>6}"
              f"{'E|r|':>7}{'base':>7}{'ratio':>7}{'q10':>7}{'q90':>7}"
              f"{'straddle':>9}{'call':>7}{'put':>7}{'call5':>7}{'P<-10':>7}")
        for (t, h), r in F.sort_values(["etf", "horizon"]).iterrows():
            print(f"{t:<5}{h:>4}{int(r['n']):>5}{r['med']:>+7.1f}{r['mean']:>+7.1f}"
                  f"{r['pos']:>6.0f}{r['eabs']:>7.1f}{r['base_eabs']:>7.1f}"
                  f"{r['ratio']:>7.2f}{r['q10']:>+7.1f}{r['q90']:>+7.1f}"
                  f"{r['straddle']:>9.1f}{r['call']:>7.1f}{r['put']:>7.1f}"
                  f"{r['call5']:>7.1f}{r['pbelow10']:>7.0f}")

    # ---- live implied overlay ----------------------------------------------
    chains = {} if args.no_options else fetch_chains(
        sorted(univ), args.cache_dir, refresh=args.refresh)
    if chains:
        print("\n== implied expected move (ATM straddle, % of spot) vs conditional E|r| ==")
        print(f"{'etf':<5}{'spot':>9}{'imp~3M':>8}{'imp~6M':>8}"
              f"{'up63':>7}{'dn63':>7}{'up126':>8}{'dn126':>8}   imp3M/up63")
        for t in sorted(chains):
            imp = chains[t]["implied"]

            def eabs(fam, h):
                try:
                    return R.loc[(t, fam, h), "eabs"]
                except KeyError:
                    return np.nan
            i3, i6 = imp.get(TARGET_DTE[0]), imp.get(TARGET_DTE[1])
            u63, d63 = eabs("up", 63), eabs("dn", 63)
            u126, d126 = eabs("up", 126), eabs("dn", 126)
            ratio = (i3 / u63) if (i3 and np.isfinite(u63)) else np.nan
            fmt = lambda v, w=7, p=1: f"{v:>{w}.{p}f}" if v is not None and np.isfinite(v) else " " * (w - 3) + "---"
            print(f"{t:<5}{chains[t]['spot']:>9.2f}{fmt(i3, 8)}{fmt(i6, 8)}"
                  f"{fmt(u63)}{fmt(d63)}{fmt(u126, 8)}{fmt(d126, 8)}   {fmt(ratio, 7, 2)}")
    else:
        print("\n[options] no live chains available -- historical tables only")

    # ---- live signals ------------------------------------------------------
    if live:
        print(f"\n== live signals (event within {LIVE_SESSIONS} sessions) ==")
        for t, fam, d in live:
            print(f"\n{t} {fam}-event {pd.Timestamp(d).date()}:")
            for h in HORIZONS:
                try:
                    r = R.loc[(t, fam, h)]
                except KeyError:
                    continue
                print(f"  h={h}: med {r['med']:+.1f}%  mean {r['mean']:+.1f}%  "
                      f"P>0 {r['pos']:.0f}%  E|r| {r['eabs']:.1f}% "
                      f"(uncond {r['base_eabs']:.1f}, ratio {r['ratio']:.2f})")
                print(f"        fair: straddle {r['straddle']:.1f}  call {r['call']:.1f}  "
                      f"put {r['put']:.1f}  call+5% {r['call5']:.1f}  "
                      f"call+10% {r['call10']:.1f}  P(r<-5) {r['pbelow5']:.0f}%  "
                      f"P(r<-10) {r['pbelow10']:.0f}%")
            if t in chains:
                imp = chains[t]["implied"]
                for dte, m in sorted(imp.items()):
                    print(f"        implied ~{dte}d move: {m:.1f}% of spot")

    if args.csv_out:
        R.round(2).to_csv(args.csv_out)
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
