#!/usr/bin/env python3
"""
Fixed-strike vol tracker -- phases 2 and 3 of VOL_TRACKER.md.

Consumes the daily chain snapshots captured by snapshot_option_chains.py
(the optsnap-data branch) and derives, per contract keyed
(symbol, expiry, strike, right) for its whole life:

  * fixed-strike IV series and per-(symbol, expiry) ATM IV series,
  * the local re-pricing  dIV_local = dIV(strike) - dIV(ATM, same expiry),
  * the dOI x dIV aggressor read. Yahoo OI is T+1, so the flow of session
    t-1 is scored by pairing dOI(t) = OI_t - OI_{t-1} with dIV_local(t-1):
        dOI up   + IV up   -> buyers opening (paying up)
        dOI up   + IV down -> SELLERS opening (overwriters/note desks)
        dOI down + IV down -> longs closing
        dOI down + IV up   -> shorts covering (squeeze)
  * per-symbol/side daily pressure index: the dOI-weighted sum of local
    IV changes over classified strikes (positive = buyer-aggressed flow),
  * the big-OI strike map (the standing structured-note positions).

Phase 3, signal linkage: for each live regime-log event (<= 63 sessions)
on an option-universe ETF, the playbook structure is resolved to actual
listed contracts on the first snapshot at/after the event date and each
leg is tracked daily -- entry vs current fixed-strike IV, local
re-pricing, OI drift, and the aggressor read at that strike.

Output degrades honestly with history: day 1 gives entry marks and the
OI map; IV deltas need 2 snapshots; the lagged aggressor read needs 3;
the pressure index is worth reading after ~5.

    python build_vol_tracker.py --snap-dir optsnap --out-dir vol_tracker_out
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import trade_structures as T
from build_regime_log import (SECTORS, THEMES, conviction_frame,
                              conviction_events, detect_breaks, spread_frames)
from etf_expected_move_study import fwd_returns
from etf_path_study import family_stats

# Delta-1 fallback for illiquid chains: long stock sized so the stop-loss
# costs RISK_NAV of the portfolio. The stop sits at the family's q25 max
# adverse excursion (thesis-broken territory -- only the worst quartile of
# historical winners ever traded there; the playbook showed tighter stops
# amputate the recovery), the take-profit at the family's median max
# favorable excursion. Defaults cover the rare family without n >= 8.
RISK_NAV = 1.0             # % of NAV lost if the stop trades
D1_STOP_DEFAULT = -12.0    # % from entry, when no family stats exist
D1_TP_DEFAULT = 10.0

DOI_MIN = 100          # contracts of OI change below this are noise, unclassified
TOP_OI = 15            # strikes per symbol in the big-OI map
LIVE_SESSIONS = 63     # regime-log events younger than this get a monitor
ROUND_TRIP = {"GDX", "KRE", "XLE", "XLRE", "XLP"}   # playbook's round-trippers

# Liquidity gate: option TRADING layers (suggestions, trade log) only run on
# chains that clear it -- recomputed from each day's snapshot, so a chain
# that dries up drops out by itself. Capture is NOT gated (history is
# irreplaceable and cheap). A chain is liquid when its near-ATM market is
# tight, or its OI is so deep that %-of-mid spreads on cheap options
# overstate the cost of trading it.
LIQ_MIN_OI = 100_000       # total OI across captured expiries
LIQ_MAX_SPREAD = 20.0      # median near-ATM spread, % of mid
LIQ_DEEP_OI = 2_000_000    # OI this deep passes on depth alone


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def recompute_iv(df):
    """Overwrite the snapshot's vendor-supplied iv with our own Black-
    Scholes (r=0) implied vol (trade_structures.implied_vol), inverted
    from the quoted price -- mid of bid/ask when the market is two-sided,
    the last trade otherwise. One consistent methodology across every
    source (Yahoo's live capture, ORATS' backfill) instead of trusting
    each vendor's own IV convention, so there's no seam where sources
    meet. A row the model can't invert (no usable price, price outside
    the no-arbitrage band) gets NaN, same as a vendor-missing iv did."""
    if df.empty:
        return df
    bid, ask, last = (df["bid"].to_numpy(dtype=float), df["ask"].to_numpy(dtype=float),
                      df["last"].to_numpy(dtype=float))
    two_sided = (bid > 0) & (ask >= bid) & (ask > 0)
    mid = np.where(two_sided, (bid + ask) / 2.0, np.nan)
    price = np.where(np.isfinite(mid), mid, np.where(last > 0, last, np.nan))
    t_years = ((pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"]))
              .dt.days.to_numpy(dtype=float)) / 365.25
    df = df.copy()
    df["iv"] = T.implied_vol(price, df["spot"].to_numpy(dtype=float),
                             df["strike"].to_numpy(dtype=float),
                             t_years, df["right"].to_numpy())
    return df


def load_snapshots(snap_dir):
    """All daily snapshots concatenated, sorted by date, iv recomputed by
    our own Black-Scholes solver (recompute_iv). Empty frame when no
    capture exists yet (the docs page renders a stub in that case)."""
    files = sorted(Path(snap_dir).glob("????-??-??.csv.gz"))
    if not files:
        return pd.DataFrame(columns=["date", "symbol", "expiry", "right",
                                     "strike", "iv", "oi", "volume", "bid",
                                     "ask", "last", "spot"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["date", "symbol", "expiry", "right", "strike"])
    return recompute_iv(df)


def chain_liquidity(day_df, min_oi=LIQ_MIN_OI, max_spread=LIQ_MAX_SPREAD,
                    deep_oi=LIQ_DEEP_OI):
    """Per-symbol liquidity read off one day's snapshot: total OI, count of
    live near-ATM contracts (within +/-10% moneyness), their median spread
    (% of mid), and the gate verdict."""
    rows = []
    for sym, g in day_df.groupby("symbol"):
        near = g[(g["strike"] / g["spot"]).between(0.9, 1.1) &
                 (g["bid"].fillna(0) > 0) & (g["ask"] >= g["bid"])]
        mid = (near["bid"] + near["ask"]) / 2
        spr = ((near["ask"] - near["bid"]) / mid * 100)
        tot = int(g["oi"].fillna(0).sum())
        med = float(spr.median()) if len(near) else np.nan
        liquid = bool((np.isfinite(med) and med <= max_spread and tot >= min_oi)
                      or tot >= deep_oi)
        rows.append({"symbol": sym, "tot_oi": tot, "live_near": len(near),
                     "med_spread": med, "liquid": liquid})
    return pd.DataFrame(rows).sort_values("med_spread")


def atm_iv(day_rows):
    """ATM IV for one (date, symbol, expiry) group: interpolate the live,
    despiked OTM smile (trade_structures.smile_points) at spot. NaN when
    the smile is too thin or one-sided."""
    spot = day_rows["spot"].iloc[0]
    ks, vs = T.smile_points(day_rows, spot)
    if len(ks) < 4 or not (ks[0] < spot < ks[-1]):
        return np.nan
    return float(np.interp(spot, ks, vs))


# --- delta-pillar risk reversals (VOL_TRACKER.md "Risk reversals & skew") ---
RR_TENORS = (30, 90)       # constant-maturity tenors, calendar days
RR_PILLARS = ("iv10p", "iv25p", "iv50", "iv25c", "iv10c")
RR_MIN_SIDE = 3            # live OTM points a side needs after despiking
RR_MIN_DTE = 5             # expiration-week pillars are pin noise -- skip
RR_MAX_DTE = 400           # nothing past this can bracket a 90d tenor usefully
RR_DAILY_YEARS = 2         # payload: daily resolution this far back, W-FRI before


def side_smile(expiry_rows, spot, right):
    """(strikes, ivs) of ONE live, despiked OTM side of an expiry group --
    puts strictly below spot or calls at/above (smile_points' convention,
    single-sided so the two sides can be pillared in delta space
    separately)."""
    r = expiry_rows
    if right == "P":
        otm = r[(r["right"] == "P") & (r["strike"] < spot)]
    else:
        otm = r[(r["right"] == "C") & (r["strike"] >= spot)]
    otm = otm[(otm["bid"].fillna(0) > 0) | (otm["oi"].fillna(0) > 0)]
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    return T.despike_smile(otm["strike"].to_numpy(dtype=float),
                           otm["iv"].to_numpy(dtype=float))


def delta_pillars(expiry_rows, spot, t_years, min_side=RR_MIN_SIDE):
    """The five delta-pillar IVs for one (date, symbol, expiry) group:
    each row's own iv -> our own BS delta (trade_structures.bs_delta) ->
    linear interpolation in delta space. The wings are single-sided (call
    side at delta 0.25/0.10 -> iv25c/iv10c, put side |delta| 0.25/0.10 ->
    iv25p/iv10p); the ATM pillar (iv50) interpolates the parity-COMBINED
    OTM smile on a call-delta axis (a put's call-equivalent delta is
    1 + delta_put), because both OTM sides individually approach 0.50
    only from below -- whenever spot sits between two strikes neither
    side alone spans it. This is the delta-space analogue of what atm_iv
    does in strike space. A pillar is NaN when its side is thin
    (< min_side points) or the target falls outside the observed span --
    no extrapolation."""
    out = dict.fromkeys(RR_PILLARS, np.nan)
    sides = {}
    for right, targets in (("C", {"iv25c": 0.25, "iv10c": 0.10}),
                           ("P", {"iv25p": 0.25, "iv10p": 0.10})):
        ks, vs = side_smile(expiry_rows, spot, right)
        d = np.abs(T.bs_delta(spot, ks, t_years, vs, right))
        keep = np.isfinite(d)
        d, vs_k = d[keep], vs[keep]
        sides[right] = (d, vs_k)
        if len(d) < min_side:
            continue
        order = np.argsort(d)
        d, vs_k = d[order], vs_k[order]
        for name, tgt in targets.items():
            if d[0] <= tgt <= d[-1]:
                out[name] = float(np.interp(tgt, d, vs_k))
    dc, vc = sides["C"]
    dp, vp = sides["P"]
    comb_d = np.concatenate([dc, 1.0 - dp])     # put -> call-equivalent delta
    comb_v = np.concatenate([vc, vp])
    if len(comb_d) >= min_side:
        order = np.argsort(comb_d)
        comb_d, comb_v = comb_d[order], comb_v[order]
        if comb_d[0] <= 0.50 <= comb_d[-1]:
            out["iv50"] = float(np.interp(0.50, comb_d, comb_v))
    return out


def cm_pillars(by_expiry, tenor_days):
    """Constant-maturity pillars: linear-in-IV interpolation, per pillar,
    across the two DTEs bracketing `tenor_days`. (Linear in IV vs linear
    in total variance differ by well under 0.1 vol pt at monthly expiry
    spacing -- the simple form matches the repo's np.interp-everywhere
    convention.) `by_expiry`: [(dte, pillar-dict)] sorted or not. NaN per
    pillar when no bracket exists or a bracket side is NaN."""
    if not by_expiry:
        return dict.fromkeys(RR_PILLARS, np.nan)
    by_expiry = sorted(by_expiry, key=lambda p: p[0])
    out = {}
    for name in RR_PILLARS:
        pts = [(dte, p[name]) for dte, p in by_expiry if np.isfinite(p[name])]
        lo = [(d, v) for d, v in pts if d <= tenor_days]
        hi = [(d, v) for d, v in pts if d >= tenor_days]
        if not lo or not hi:
            out[name] = np.nan
            continue
        d0, v0 = lo[-1]
        d1, v1 = hi[0]
        out[name] = v0 if d0 == d1 else \
            v0 + (v1 - v0) * (tenor_days - d0) / (d1 - d0)
    return out


def rr_day_rows(day_df, tenors=RR_TENORS):
    """Long RR rows [{date, symbol, tenor, iv10p..iv10c}] for every
    (date, symbol) in the frame, via delta_pillars per captured expiry
    (RR_MIN_DTE <= dte <= RR_MAX_DTE) then cm_pillars per tenor."""
    rows = []
    for (date, sym), g in day_df.groupby(["date", "symbol"]):
        spot = float(g["spot"].iloc[0])
        if not spot > 0:
            continue
        by_expiry = []
        for exp, ge in g.groupby("expiry"):
            dte = (pd.Timestamp(exp) - pd.Timestamp(date)).days
            if not RR_MIN_DTE <= dte <= RR_MAX_DTE:
                continue
            by_expiry.append((dte, delta_pillars(ge, spot, dte / 365.25)))
        for tenor in tenors:
            cm = cm_pillars(by_expiry, tenor)
            if any(np.isfinite(v) for v in cm.values()):
                rows.append({"date": date, "symbol": sym, "tenor": tenor, **cm})
    return rows


def rr_frame(df, tenors=RR_TENORS):
    """rr_day_rows over the whole snapshot frame -> long frame [date,
    symbol, tenor, iv10p, iv25p, iv50, iv25c, iv10c], date-sorted."""
    cols = ["date", "symbol", "tenor", *RR_PILLARS]
    if df.empty:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rr_day_rows(df, tenors), columns=cols)
              .sort_values(["date", "symbol", "tenor"], ignore_index=True))


def load_rr_backfill(snap_dir):
    """optsnap/rr_history.csv.gz -- the one-time ORATS-derived deep
    history (build_rr_history.py), pushed to the optsnap-data branch and
    fetched alongside the snapshots. Empty frame when absent."""
    p = Path(snap_dir) / "rr_history.csv.gz"
    if not p.exists():
        return pd.DataFrame(columns=["date", "symbol", "tenor", *RR_PILLARS])
    return pd.read_csv(p)


def merge_rr(backfill, live):
    """Backfill rows strictly BEFORE the first live (snapshot-derived)
    date, live rows from there on -- at the seam the live pipeline wins."""
    if live.empty:
        return backfill.copy()
    first_live = live["date"].min()
    old = backfill[backfill["date"] < first_live]
    return (pd.concat([old, live], ignore_index=True)
              .sort_values(["symbol", "tenor", "date"], ignore_index=True))


def rr_payload(rr, daily_years=RR_DAILY_YEARS):
    """The /*__RR__*/ template payload. One shared date axis -- W-FRI
    weekly before (last date - daily_years), daily after -- keeps the
    full 2007+ history chartable at ~1.4k points instead of ~5k. The
    history chart ships only the derived series (rr10/rr25/atm, 2dp vol
    points, NaN -> None); the five raw pillars ship only in the skew
    slice (latest day, ~1 week back, ~1 month back), taken at full
    resolution before the downsampling."""
    empty = {"tenors": list(RR_TENORS), "dates": [], "cut": None,
             "symbols": {}, "skew": {}}
    if rr.empty:
        return empty
    rr = rr.copy()
    rr["rr10"] = (rr["iv10c"] - rr["iv10p"]) * 100
    rr["rr25"] = (rr["iv25c"] - rr["iv25p"]) * 100
    rr["atm"] = rr["iv50"] * 100

    all_dates = pd.to_datetime(sorted(rr["date"].unique()))
    cut = all_dates[-1] - pd.DateOffset(years=daily_years)
    old = all_dates[all_dates < cut]
    weekly = set(pd.Series(old, index=old).resample("W-FRI").last().dropna())
    axis = [d for d in all_dates if d >= cut or d in weekly]
    axis_str = [str(d.date()) for d in axis]

    def rnd(v):
        return None if not np.isfinite(v) else round(float(v), 2)

    tenors = sorted(rr["tenor"].unique())
    symbols, skew = {}, {}
    for sym, gs in rr.groupby("symbol"):
        symbols[sym], skew[sym] = {}, {}
        for tenor, gt in gs.groupby("tenor"):
            by_date = gt.set_index("date")
            aligned = by_date.reindex(axis_str)
            symbols[sym][f"t{tenor}"] = {
                k: [rnd(v) for v in aligned[k]] for k in ("rr10", "rr25", "atm")}
            # skew slice: latest finite day, then nearest to -7 and -30
            finite = by_date[by_date[list(RR_PILLARS)].notna().any(axis=1)]
            if finite.empty:
                continue
            idx = pd.to_datetime(finite.index)
            last = idx[-1]
            picks = [last]
            for back in (7, 30):
                near = idx[np.argmin(np.abs(idx - (last - pd.Timedelta(days=back))))]
                picks.append(near)
            skew[sym][f"t{tenor}"] = {
                "dates": [str(p.date()) for p in picks],
                **{name: [rnd(finite.loc[str(p.date()), pil] * 100)
                          for p in picks]
                   for name, pil in (("p10", "iv10p"), ("p25", "iv25p"),
                                     ("atm", "iv50"), ("c25", "iv25c"),
                                     ("c10", "iv10c"))}}
    return {"tenors": [int(t) for t in tenors], "dates": axis_str,
            "cut": str(cut.date()), "symbols": symbols, "skew": skew}


def contract_panel(df):
    """(iv, oi, atm) wide frames -- rows keyed (symbol, expiry, strike,
    right) for iv/oi and (symbol, expiry) for atm; columns = snapshot
    dates. The fixed-strike identity: a contract keeps its row for life."""
    key = ["symbol", "expiry", "strike", "right"]
    iv = df.pivot_table(index=key, columns="date", values="iv", aggfunc="last")
    oi = df.pivot_table(index=key, columns="date", values="oi", aggfunc="last")
    atm = (df.groupby(["date", "symbol", "expiry"])
             .apply(atm_iv, include_groups=False)
             .unstack("date"))
    return iv, oi, atm


def local_repricing(iv, atm):
    """Fixed-strike dIV minus same-(symbol, expiry) ATM dIV, per date."""
    div = iv.diff(axis=1)
    datm = atm.diff(axis=1)
    aligned = datm.reindex(div.index.droplevel(["strike", "right"])).to_numpy()
    return div - aligned


def classify_flow(doi, dlocal, doi_min=DOI_MIN):
    """Aggressor label for one contract-flow observation (dOI of the T+1
    print vs the prior session's local dIV)."""
    if not np.isfinite(doi) or not np.isfinite(dlocal) or abs(doi) < doi_min:
        return None
    if doi > 0:
        return "buyers_opening" if dlocal > 0 else "sellers_opening"
    return "longs_closing" if dlocal < 0 else "shorts_covering"


def flow_table(iv, oi, atm, doi_min=DOI_MIN):
    """Per-contract flow observations: for each date t (>= 3rd snapshot),
    dOI(t) paired with dlocal(t-1). Long frame with the classification."""
    dloc = local_repricing(iv, atm)
    doi = oi.diff(axis=1)
    dates = list(iv.columns)
    rows = []
    for i in range(2, len(dates)):
        t, t1 = dates[i], dates[i - 1]
        d_oi, d_loc = doi[t], dloc[t1]
        mask = d_oi.abs() >= doi_min
        for key in d_oi.index[mask.fillna(False)]:
            lab = classify_flow(float(d_oi[key]), float(d_loc.get(key, np.nan)),
                                doi_min)
            if lab:
                rows.append({"date": t, "symbol": key[0], "expiry": key[1],
                             "strike": key[2], "right": key[3],
                             "doi": float(d_oi[key]),
                             "dlocal": float(d_loc[key]) * 100,
                             "flow": lab})
    return pd.DataFrame(rows)


def pressure_index(flows, oi):
    """Per (date, symbol, right): dOI-weighted sum of local dIV (vol pts x
    contracts, in thousands) over the classified strikes -- positive =
    net buyer-aggressed flow at that side's big strikes."""
    if flows.empty:
        return pd.DataFrame()
    f = flows.copy()
    f["w"] = f["doi"] * f["dlocal"] / 1000.0
    g = (f.groupby(["date", "symbol", "right"])
          .agg(pressure=("w", "sum"), n_strikes=("w", "size"),
               net_doi=("doi", "sum")).reset_index())
    return g


def big_oi_map(df, top=TOP_OI, asof=None):
    """Latest snapshot's top-OI strikes per symbol with their IV history:
    entry (first-seen) IV vs latest, and the local drift if computable.
    Strikes whose expiry has already passed `asof` (real calendar time,
    default now -- NOT just the last snapshot date) are dropped: a stale
    build (a skipped nightly run, a weekend between builds) would
    otherwise still show a since-expired contract's last-alive OI as if
    it were a live position."""
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today()
    last_date = df["date"].max()
    latest = df[df["date"] == last_date].dropna(subset=["oi"])
    latest = latest[pd.to_datetime(latest["expiry"]) >= asof.normalize()]
    out = []
    for sym, g in latest.groupby("symbol"):
        for _, r in g.nlargest(top, "oi").iterrows():
            hist = df[(df["symbol"] == sym) & (df["expiry"] == r["expiry"]) &
                      (df["strike"] == r["strike"]) & (df["right"] == r["right"])]
            first = hist.iloc[0]
            out.append({"symbol": sym, "expiry": r["expiry"], "right": r["right"],
                        "strike": r["strike"], "spot": r["spot"],
                        "moneyness": r["strike"] / r["spot"] * 100,
                        "oi": r["oi"], "oi_first": first["oi"],
                        "iv": r["iv"], "iv_first": first["iv"],
                        "days_seen": hist["date"].nunique()})
    return pd.DataFrame(out)


def resolve_leg(day_df, symbol, right, tenor_days, moneyness, asof):
    """Nearest listed contract to a structure leg inside one day's
    snapshot: closest captured expiry to `tenor_days` out, then closest
    strike to spot*(1+moneyness). Returns the row or None."""
    g = day_df[(day_df["symbol"] == symbol) & (day_df["right"] == right)]
    if g.empty:
        return None
    tgt = pd.Timestamp(asof) + pd.Timedelta(days=tenor_days)
    exp = min(g["expiry"].unique(), key=lambda e: abs(pd.Timestamp(e) - tgt))
    ge = g[g["expiry"] == exp]
    spot = ge["spot"].iloc[0]
    k = ge.iloc[(ge["strike"] - spot * (1 + moneyness)).abs().argsort()].iloc[0]
    return k


# ----------------------------------------------------------------------------
# Phase 3: structure suggestions + the replayed trade log
# ----------------------------------------------------------------------------
def signal_context(cache_dir, refresh=False):
    """Live regime-log signals with the context the suggester needs:
    conditional E|move| 63d over the family's full event history, the
    playbook class, and the rule-based exit date (event + 63 sessions)."""
    univ = {**SECTORS, **THEMES}
    syms = sorted(set(univ) | {"SPY"})
    panels = N.load_yahoo_panels(syms, "2004-01-01", pd.Timestamp.today().normalize(),
                                 cache_dir=cache_dir or None, refresh=refresh,
                                 label="VOLTRK")
    adj = panels["adjclose"].dropna(how="all")
    spy = adj["SPY"].dropna()
    out = []
    for t in sorted(univ):
        if t not in adj:
            continue
        c = adj[t].dropna()
        conv = conviction_frame(c, panels["volume"][t])
        ev = conviction_events(c, conv, min_score=3, cooldown=21, horizon=63)
        s21, s63, z, _ = spread_frames(c, spy)
        fams = {"up": list(ev[ev["dir"] == 1]["date"]),
                "dn": list(ev[ev["dir"] == -1]["date"]),
                "turn": [d for d, s in detect_breaks(s63, z) if s == -1]}
        for fam, dates in fams.items():
            if not dates:
                continue
            i = c.index.get_loc(dates[-1])
            if len(c) - 1 - i > LIVE_SESSIONS:
                continue
            r = fwd_returns(c, dates, 63)
            st = family_stats(c, dates)
            out.append({"symbol": t, "family": fam,
                        "event": pd.Timestamp(dates[-1]),
                        "roundtrip": t in ROUND_TRIP,
                        "cond_eabs63": float(np.mean(np.abs(r))) if len(r) >= 8 else np.nan,
                        "mae_q25": st[0]["mae_q25"] if st else np.nan,
                        "mfe_med": st[0]["mfe_med"] if st else np.nan,
                        "exit_date": c.index[i + 63] if i + 63 < len(c) else None})
    return out


def expiry_stats(day_df, symbol, target_days, asof):
    """For the captured expiry nearest `target_days` out: its ATM IV,
    implied move (0.8*sigma), and +/-1-sigma risk reversal, from one day's
    snapshot rows."""
    g = day_df[day_df["symbol"] == symbol]
    if g.empty:
        return None
    tgt = pd.Timestamp(asof) + pd.Timedelta(days=target_days)
    exp = min(g["expiry"].unique(), key=lambda e: abs(pd.Timestamp(e) - tgt))
    rows = g[g["expiry"] == exp]
    spot = float(rows["spot"].iloc[0])
    dte = max((pd.Timestamp(exp) - pd.Timestamp(asof)).days, 1)
    a = atm_iv(rows)
    st = {"expiry": exp, "dte": dte, "spot": spot, "atm_iv": a,
          "imp_move": np.nan, "rr": np.nan}
    if not np.isfinite(a):
        return st
    sig = a * np.sqrt(dte / 365.25)
    st["imp_move"] = 0.8 * sig * 100
    ks, vs = T.smile_points(rows, spot)
    if len(ks) >= 4:
        def wing(k):
            return float(np.interp(k, ks, vs)) if ks[0] <= k <= ks[-1] else np.nan
        st["rr"] = ((wing(spot * (1 + sig)) - a) - (wing(spot * (1 - sig)) - a)) * 100
    return st


def resolve_suggestion(day_df, symbol, legs, asof):
    """Suggested sigma-unit legs -> listed contracts on one day's
    snapshot: expiry nearest each leg's tenor, strike nearest the
    sigma-implied moneyness. [] when anything is unresolvable."""
    resolved = []
    for spec in legs:
        st = expiry_stats(day_df, symbol, spec["tenor"], asof)
        if st is None or not np.isfinite(st["atm_iv"]):
            return []
        m = T.sigma_to_moneyness(spec["sigma"], st["atm_iv"], st["dte"])
        rows = day_df[(day_df["symbol"] == symbol) &
                      (day_df["expiry"] == st["expiry"]) &
                      (day_df["right"] == spec["right"])]
        if rows.empty:
            return []
        k = rows.iloc[(rows["strike"] - st["spot"] * (1 + m)).abs().argsort()].iloc[0]
        resolved.append({**spec, "expiry": st["expiry"],
                         "strike": float(k["strike"]), "moneyness": m * 100})
    return resolved


def delta1_trade(spots, s, dates):
    """Delta-1 (stock) trade for a signal whose chain fails the liquidity
    gate: long at the entry snapshot's spot, stop at the family's q25 max
    adverse excursion, take-profit at its median max favorable excursion,
    weight sized so the stop costs RISK_NAV% of the portfolio. Replayed on
    snapshot closes (close-through, no intraday fills): stop, then target,
    then the 63-session time exit."""
    entry_snap = next((d for d in dates if pd.Timestamp(d) >= s["event"]), None)
    entry = spots.get((entry_snap, s["symbol"])) if entry_snap else None
    if entry is None or not np.isfinite(entry):
        return None
    stop_pct = s["mae_q25"] if np.isfinite(s["mae_q25"]) else D1_STOP_DEFAULT
    tp_pct = s["mfe_med"] if np.isfinite(s["mfe_med"]) else D1_TP_DEFAULT
    stop_px, tp_px = entry * (1 + stop_pct / 100), entry * (1 + tp_pct / 100)
    weight = RISK_NAV / abs(stop_pct) * 100
    exit_snap = (next((d for d in dates if pd.Timestamp(d) >= s["exit_date"]), None)
                 if s["exit_date"] is not None else None)
    status, last_px, mark_snap = "open", entry, entry_snap
    for d in dates:
        if d <= entry_snap:
            continue
        px = spots.get((d, s["symbol"]))
        if px is None or not np.isfinite(px):
            continue
        last_px, mark_snap = px, d
        if px <= stop_px:
            status, last_px = "stopped", stop_px
            break
        if px >= tp_px:
            status, last_px = "target", tp_px
            break
        if exit_snap is not None and d >= exit_snap:
            status = "closed"
            break
    move = (last_px / entry - 1) * 100
    return {"structure": "delta-1 stock", "entry_snap": entry_snap,
            "proxied": pd.Timestamp(entry_snap) > s["event"],
            "legs": (f"long stock @ {entry:.2f} · stop {stop_px:.2f} "
                     f"({stop_pct:+.1f}%) · tp {tp_px:.2f} ({tp_pct:+.1f}%) "
                     f"· wt {weight:.1f}% NAV"),
            "entry_px": round(entry, 2), "stop_px": round(stop_px, 2),
            "tp_px": round(tp_px, 2), "weight_pct": round(weight, 1),
            "mark_snap": mark_snap, "value_pct": move, "pnl_pct": move,
            "nav_pnl_pct": move * weight / 100, "status": status,
            "rationale": (f"illiquid chain -- delta-1 with the stop at the "
                          f"family's q25 excursion; stop costs {RISK_NAV:.0f}% NAV")}


def trade_log(df, signals, liquid=None):
    """Deterministic replay of the established rules over the snapshot
    history: enter each live signal's suggested structure on the first
    snapshot at/after the event (flagged when proxied from a later first
    capture), exit at event + 63 sessions (the playbook horizon). Marks
    are smile-based (trade_structures.leg_mark), never raw closing mids.
    Symbols outside `liquid` (a set; None = no gate) get the delta-1
    stock trade instead -- entry/stop/take-profit prices sized to RISK_NAV
    -- wide chains don't get option structures."""
    dates = sorted(df["date"].unique())
    spots = df.groupby(["date", "symbol"])["spot"].first().to_dict()
    rows = []
    for s in signals:
        if liquid is not None and s["symbol"] not in liquid:
            base = {"symbol": s["symbol"], "family": s["family"],
                    "event": str(s["event"].date())}
            d1 = delta1_trade(spots, s, dates)
            rows.append({**base, **d1} if d1 else
                        {**base, "structure": None, "status": "unresolvable",
                         "rationale": "illiquid chain and no snapshot spot to "
                         "anchor a delta-1 entry"})
            continue
        entry_snap = next((d for d in dates if pd.Timestamp(d) >= s["event"]), None)
        if entry_snap is None:
            continue
        day = df[df["date"] == entry_snap]
        st3 = expiry_stats(day, s["symbol"], 91, entry_snap) or {}
        rich = (st3.get("imp_move", np.nan) / s["cond_eabs63"]
                if np.isfinite(st3.get("imp_move", np.nan)) and
                np.isfinite(s["cond_eabs63"]) and s["cond_eabs63"] > 0 else np.nan)
        name, legs, why = T.suggest_structure(s["family"], s["roundtrip"],
                                              s["symbol"], rich, st3.get("rr", np.nan))
        base = {"symbol": s["symbol"], "family": s["family"],
                "event": str(s["event"].date()), "entry_snap": entry_snap,
                "proxied": pd.Timestamp(entry_snap) > s["event"],
                "imp3m": st3.get("imp_move", np.nan),
                "cond63": s["cond_eabs63"], "rich": rich,
                "rr3m": st3.get("rr", np.nan), "structure": name,
                "rationale": why}
        if name is None:
            rows.append({**base, "status": "no_trade"})
            continue
        resolved = resolve_suggestion(day, s["symbol"], legs, entry_snap)
        if not resolved:
            rows.append({**base, "status": "unresolvable"})
            continue
        cost, _ = T.structure_mark(day, s["symbol"], resolved, entry_snap)
        exit_snap = (next((d for d in dates if pd.Timestamp(d) >= s["exit_date"]), None)
                     if s["exit_date"] is not None else None)
        mark_snap = exit_snap or dates[-1]
        value, _ = T.structure_mark(df[df["date"] == mark_snap], s["symbol"],
                                    resolved, mark_snap)
        rows.append({**base,
                     "legs": " / ".join(
                         f"{'+' if L['qty'] > 0 else '-'}{L['right']} "
                         f"{L['expiry']} {L['strike']:g} ({L['moneyness']:+.0f}%)"
                         for L in resolved),
                     "entry_cost_pct": cost, "mark_snap": mark_snap,
                     "value_pct": value,
                     "pnl_pct": (value - cost) if cost is not None and
                                value is not None else np.nan,
                     "status": "closed" if exit_snap else "open"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def render_page(template, docs_out, ctx, log, bigmap, liq, rr):
    """docs/vol_tracker.html -- payloads injected into the template the
    same way the other pages do it."""
    def records(frame):
        return (json.loads(frame.replace({np.nan: None}).to_json(orient="records"))
                if len(frame) else [])
    body = (Path(template).read_text(encoding="utf-8")
            .replace("/*__CTX__*/", json.dumps(ctx, separators=(",", ":")))
            .replace("/*__LOG__*/", json.dumps(records(log), separators=(",", ":")))
            .replace("/*__BIG__*/", json.dumps(records(bigmap), separators=(",", ":")))
            .replace("/*__LIQ__*/", json.dumps(records(liq), separators=(",", ":")))
            .replace("/*__RR__*/", json.dumps(rr, separators=(",", ":"))))
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>Vol Tracker</title></head>'
           '<body>\n' + body + '\n</body></html>\n')
    out = Path(docs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc) // 1024} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap-dir", default="optsnap")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="write pressure.csv / big_oi.csv / flows.csv here")
    ap.add_argument("--docs-out", default=None,
                    help="also render the dashboard page (docs/vol_tracker.html)")
    ap.add_argument("--template", default="vol_tracker_template.html")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    df = load_snapshots(args.snap_dir)
    days = sorted(df["date"].unique())
    if not days:
        # the RR backfill can chart on its own even before any capture
        rr = merge_rr(load_rr_backfill(args.snap_dir), rr_frame(df))
        print(f"no snapshots under {args.snap_dir}")
        if args.docs_out:
            render_page(args.template, args.docs_out, {
                "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "days": 0, "note": "No chain capture yet -- the optsnap "
                "workflow appends the first snapshot after the next close."},
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), rr_payload(rr))
        return
    print(f"{len(days)} snapshot day(s): {days[0]} .. {days[-1]}  "
          f"({len(df)} rows, {df['symbol'].nunique()} symbols)")

    liq = chain_liquidity(df[df["date"] == days[-1]])
    liquid_set = set(liq[liq["liquid"]]["symbol"])
    print(f"\n== liquidity gate ({len(liquid_set)}/{len(liq)} chains pass) ==")
    print("excluded: " + ", ".join(sorted(set(liq["symbol"]) - liquid_set)))

    iv, oi, atm = contract_panel(df)
    flows = flow_table(iv, oi, atm) if len(days) >= 3 else pd.DataFrame()
    press = pressure_index(flows, oi)
    bigmap = big_oi_map(df)

    rr_bf = load_rr_backfill(args.snap_dir)
    rr = merge_rr(rr_bf, rr_frame(df))
    if len(rr):
        print(f"\n== rr history: {len(rr)} sym-tenor-days, "
              f"{rr['date'].min()} .. {rr['date'].max()} "
              f"(backfill {len(rr) - len(rr[rr['date'] >= days[0]])}, "
              f"live {len(rr[rr['date'] >= days[0]])}) ==")

    if len(days) < 2:
        print("\n[deltas] single snapshot -- IV/OI changes start with day 2, "
              "aggressor reads with day 3.")
    elif flows.empty:
        print("\n[flows] fewer than 3 snapshots (or no dOI over threshold) -- "
              "no aggressor reads yet.")
    else:
        print(f"\n== aggressor flows ({len(flows)} classified strike-days) ==")
        print(flows.groupby("flow").size().to_string())
        latest = press[press["date"] == press["date"].max()]
        print("\n== pressure index, latest session (dOI-weighted local dIV) ==")
        print(latest.sort_values("pressure").to_string(index=False))

    print(f"\n== big-OI map, latest snapshot (top strikes; {bigmap['symbol'].nunique()} symbols) ==")
    show = bigmap.sort_values("oi", ascending=False).head(15).copy()
    show["moneyness"] = show["moneyness"].round(0).astype(int)
    print(show[["symbol", "expiry", "right", "strike", "moneyness", "oi",
                "iv", "days_seen"]].to_string(index=False))

    signals = signal_context(args.cache_dir, refresh=args.refresh)
    log = trade_log(df, signals, liquid=liquid_set)
    print(f"\n== suggested structures & trade log ({len(log)} signals; "
          f"smile-marked, % of spot, debit positive) ==")
    for _, r in log.iterrows():
        head = (f"\n{r['symbol']} {r['family']} {r['event']}  "
                f"imp3M {r['imp3m']:.1f}% vs cond {r['cond63']:.1f}% "
                f"(x{r['rich']:.2f})  rr {r['rr3m']:+.1f}"
                if np.isfinite(r.get("rich", np.nan)) else
                f"\n{r['symbol']} {r['family']} {r['event']}")
        print(head)
        if r["status"] in ("no_trade", "unresolvable"):
            print(f"  {r['status'].upper()}: {r['rationale']}")
            continue
        tag = " (entry proxied)" if r["proxied"] else ""
        if r["structure"] == "delta-1 stock":
            print(f"  delta-1: {r['legs']}{tag}")
            print(f"  move {r['pnl_pct']:+.2f}% -> NAV P&L "
                  f"{r['nav_pnl_pct']:+.2f}% ({r['mark_snap']})  "
                  f"[{r['status'].upper()}]")
        else:
            print(f"  {r['structure']}: {r['legs']}{tag}")
            print(f"  entry {r['entry_cost_pct']:+.2f}% -> {r['value_pct']:+.2f}% "
                  f"({r['mark_snap']})  P&L {r['pnl_pct']:+.2f}% of spot  "
                  f"[{r['status'].upper()}]")
        print(f"  why: {r['rationale']}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        liq.round(2).to_csv(out / "liquidity.csv", index=False)
        bigmap.round(4).to_csv(out / "big_oi.csv", index=False)
        log.round(4).to_csv(out / "trade_log.csv", index=False)
        if not flows.empty:
            flows.round(4).to_csv(out / "flows.csv", index=False)
            press.round(4).to_csv(out / "pressure.csv", index=False)
        if not rr.empty:
            rr.round(4).to_csv(out / "rr.csv", index=False)
        print(f"\nwrote {out}/")

    if args.docs_out:
        bigmap2 = bigmap.merge(liq[["symbol", "liquid"]], on="symbol", how="left")
        render_page(args.template, args.docs_out, {
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "days": len(days), "first": days[0], "last": days[-1],
            "nSymbols": int(df["symbol"].nunique()),
            "nLiquid": len(liquid_set),
            "note": None if len(days) >= 3 else (
                "IV deltas need 2 snapshots and aggressor reads 3 -- "
                f"{len(days)} captured so far; columns fill in as history accrues."),
        }, log, bigmap2, liq, rr_payload(rr))


if __name__ == "__main__":
    main()
