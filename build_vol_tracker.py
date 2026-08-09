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
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N
from build_regime_log import (SECTORS, THEMES, conviction_frame,
                              conviction_events, detect_breaks, spread_frames)

DOI_MIN = 100          # contracts of OI change below this are noise, unclassified
TOP_OI = 15            # strikes per symbol in the big-OI map
LIVE_SESSIONS = 63     # regime-log events younger than this get a monitor
PRESSURE_TOP = 25      # strikes per symbol/side feeding the pressure index

# Playbook structures (ETF_PATH_PLAYBOOK / EXPECTED_MOVE_FINDINGS): tenor is
# target calendar days, moneyness is strike/spot - 1. qty +1 = long, -1 = short.
ROUND_TRIP = {"GDX", "KRE", "XLE", "XLRE", "XLP"}   # playbook's round-trippers
STRUCTURES = {
    "up_chaser":  [{"right": "C", "tenor": 183, "moneyness": 0.00, "qty": 1}],
    "up_roundtrip": [{"right": "C", "tenor": 183, "moneyness": 0.00, "qty": 1},
                     {"right": "C", "tenor": 183, "moneyness": 0.10, "qty": -1},
                     {"right": "P", "tenor": 91, "moneyness": -0.08, "qty": -1}],
    "dn":         [{"right": "P", "tenor": 91, "moneyness": -0.05, "qty": -1}],
    "turn":       [{"right": "C", "tenor": 183, "moneyness": 0.00, "qty": 1}],
}


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def load_snapshots(snap_dir):
    """All daily snapshots concatenated, sorted by date."""
    files = sorted(Path(snap_dir).glob("????-??-??.csv.gz"))
    if not files:
        raise SystemExit(f"no snapshots under {snap_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values(["date", "symbol", "expiry", "right", "strike"])


def atm_iv(day_rows):
    """ATM IV for one (date, symbol, expiry) group: interpolate the OTM
    smile (puts below spot, calls at/above) at spot. NaN when the smile is
    too thin or one-sided."""
    spot = day_rows["spot"].iloc[0]
    otm = day_rows[((day_rows["right"] == "P") & (day_rows["strike"] < spot)) |
                   ((day_rows["right"] == "C") & (day_rows["strike"] >= spot))]
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    ks, vs = otm["strike"].to_numpy(), otm["iv"].to_numpy()
    if len(ks) < 4 or not (ks[0] < spot < ks[-1]):
        return np.nan
    return float(np.interp(spot, ks, vs))


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


def big_oi_map(df, top=TOP_OI):
    """Latest snapshot's top-OI strikes per symbol with their IV history:
    entry (first-seen) IV vs latest, and the local drift if computable."""
    last_date = df["date"].max()
    latest = df[df["date"] == last_date].dropna(subset=["oi"])
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
# Phase 3: live-signal monitors
# ----------------------------------------------------------------------------
def live_signals(cache_dir, refresh=False):
    """(symbol, family, event_date) for regime-log events younger than
    LIVE_SESSIONS on the leaderboard ETFs."""
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
            if dates and (len(c) - 1 - c.index.get_loc(dates[-1])) <= LIVE_SESSIONS:
                out.append((t, fam, pd.Timestamp(dates[-1])))
    return out


def monitor_blocks(df, signals):
    """Resolved + tracked structure legs for each live signal. Entry marks
    come from the first snapshot at/after the event date (proxied from the
    first available capture when the event predates capture history)."""
    dates = sorted(df["date"].unique())
    blocks = []
    for sym, fam, ev_date in signals:
        entry_date = next((d for d in dates if pd.Timestamp(d) >= ev_date), dates[0])
        proxied = pd.Timestamp(entry_date) > ev_date or ev_date < pd.Timestamp(dates[0])
        skey = "up_roundtrip" if (fam == "up" and sym in ROUND_TRIP) else \
               ("up_chaser" if fam == "up" else fam)
        legs = []
        day = df[df["date"] == entry_date]
        for spec in STRUCTURES[skey]:
            c = resolve_leg(day, sym, spec["right"], spec["tenor"],
                            spec["moneyness"], entry_date)
            if c is None:
                continue
            hist = df[(df["symbol"] == sym) & (df["expiry"] == c["expiry"]) &
                      (df["strike"] == c["strike"]) & (df["right"] == c["right"])]
            latest = hist.iloc[-1]
            legs.append({
                "qty": spec["qty"], "right": spec["right"], "expiry": c["expiry"],
                "strike": float(c["strike"]),
                "entry_iv": float(c["iv"]) if pd.notna(c["iv"]) else np.nan,
                "iv": float(latest["iv"]) if pd.notna(latest["iv"]) else np.nan,
                "entry_oi": c["oi"], "oi": latest["oi"],
                "days": hist["date"].nunique(),
            })
        blocks.append({"symbol": sym, "family": fam, "structure": skey,
                       "event": str(ev_date.date()), "entry_snap": entry_date,
                       "entry_proxied": bool(proxied), "legs": legs})
    return blocks


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap-dir", default="optsnap")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="write pressure.csv / big_oi.csv / flows.csv here")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    df = load_snapshots(args.snap_dir)
    days = sorted(df["date"].unique())
    print(f"{len(days)} snapshot day(s): {days[0]} .. {days[-1]}  "
          f"({len(df)} rows, {df['symbol'].nunique()} symbols)")

    iv, oi, atm = contract_panel(df)
    flows = flow_table(iv, oi, atm) if len(days) >= 3 else pd.DataFrame()
    press = pressure_index(flows, oi)
    bigmap = big_oi_map(df)

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

    signals = live_signals(args.cache_dir, refresh=args.refresh)
    print(f"\n== live-signal monitors ({len(signals)} signals) ==")
    for b in monitor_blocks(df, signals):
        tag = " (entry proxied from first capture)" if b["entry_proxied"] else ""
        print(f"\n{b['symbol']} {b['family']} {b['event']} -> {b['structure']}{tag}")
        for L in b["legs"]:
            side = "long" if L["qty"] > 0 else "short"
            div = (L["iv"] - L["entry_iv"]) * 100 if np.isfinite(L["iv"]) and \
                np.isfinite(L["entry_iv"]) else np.nan
            print(f"  {side:>5} {L['right']} {L['expiry']} K={L['strike']:g}  "
                  f"entry IV {L['entry_iv']*100:.1f} -> {L['iv']*100:.1f} "
                  f"({div:+.1f} vol pts)  OI {L['entry_oi']:.0f} -> {L['oi']:.0f}  "
                  f"[{L['days']}d tracked]")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bigmap.round(4).to_csv(out / "big_oi.csv", index=False)
        if not flows.empty:
            flows.round(4).to_csv(out / "flows.csv", index=False)
            press.round(4).to_csv(out / "pressure.csv", index=False)
        print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
