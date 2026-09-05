#!/usr/bin/env python3
"""
Forward-path study across the intra-index regime cells.
=========================================================

Companion to `intra_index_regime_study.py`: that study answers "what is the
mean 1-month forward return inside each comovement x DIX regime cell". A
mean cannot say how volatile the path to that return is -- how deep it goes
before it pays, how often a stop gets hit before a target, whether realized
vol expands or contracts while you hold, or how often the regime itself
ends mid-hold. This study answers those questions for the SAME cells the
corrected regime study already defines (rolling-basis comovement zones,
DIX zone on the one-session-lagged signal, plus the vol-parallel zones),
reusing its frames, zones, episode ids and print-gate discipline.

Design record: REGIME_PATH_STUDY_PLAN.md (phased spec; read it before
extending this file -- every open decision is written down there).

Per cell (family = ENV: every day the cell's condition holds is an anchor)
this reports:
  * a FAN of the terminal-return distribution at nine checkpoints (day 1
    through day 63), each checkpoint scored over its own complete-to-h
    population;
  * EXCURSION stats over the primary 21-session hold (max adverse/favorable
    excursion, trough/peak day, give-back, dip-depth shares);
  * BARRIER touch probabilities and a stop x target first-passage BRACKET
    matrix (close-only -- a floor on true intraday touch rates);
  * a VOLATILITY block: forward realized vol vs. the frame's own trailing
    realized vol (expansion ratio), a variance ratio (trending vs.
    mean-reverting inside the hold), and vol-scaled tail shares against the
    Gaussian benchmarks (|z|>1 in 31.7%, |z|>2 in 4.6%);
  * a SIZING translation (ETF_PATH_PLAYBOOK.md's convention: position
    weight that loses 1% NAV at the q25/p10 max adverse excursion);
  * episode-cluster bootstrap CIs (resampling whole contiguous episodes,
    same estimator as the regime study) for the mean terminal return, the
    q25 MAE, the -5% touch probability and the median vol-expansion ratio.

Cells below the print gate (42 anchors AND 5 distinct episodes, matching
the regime study's Rule 5) print only their counts.

Data source: same as `intra_index_regime_study.py` -- the built dashboard
payload (`docs/index.html`) for NDX (fully offline), plus a fetched iShares
basket for SPX/IWM (needs network or a warm cache).

Usage
-----
    python regime_path_study.py                        # all three indices
    python regime_path_study.py --indices ndx           # offline
    python regime_path_study.py --csv regime_paths.csv --risk-csv regime_path_risk.csv
"""
import argparse
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

import intra_index_regime_study as R

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
H = 63                              # path horizon, sessions (secondary window)
HOLD = R.WINDOW                     # 21 -- the primary hold window
CHECKPOINTS = (1, 2, 3, 5, 10, 15, 21, 42, 63)
FAN_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BARRIER_LEVELS = (3, 5, 8, 12)
BRACKET_LEVELS = (3, 5, 8)
GATE_DAYS = R.GATE_DAYS
GATE_EPISODES = R.GATE_EPISODES
GATE_EVENTS = R.GATE_EVENTS


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def forward_path_panel(close, horizon=H):
    """(len(close) x horizon+1) DataFrame of forward %-return paths: value
    at row t, column h = close[t+h]/close[t] - 1, in percent. Column 0 is
    identically 0 (or NaN if the anchor's own close is missing). NaN once
    the horizon runs past the end of `close`. Vectorized via `shift(-h)`."""
    close = close.astype(float)
    cols = {h: (close.shift(-h) / close - 1.0) * 100.0 for h in range(horizon + 1)}
    return pd.DataFrame(cols, index=close.index)


def daily_rets_from_path(sub, h):
    """Per-anchor daily %-returns for sessions 1..h, derived from the path
    panel's cumulative-from-anchor values (column c = close[t+c]/close[t]-1
    in percent) via consecutive-column ratios -- this reconstructs each
    day's actual return without re-touching the underlying close series, so
    every downstream volatility statistic stays a pure function of the path
    panel. Returns an (anchors x h) array; rows with any NaN are NOT
    dropped here (callers filter via their own completeness mask so row
    order/count stays aligned with `excursion_stats`/`barrier_touch`).
    None if columns 0..h are not all present."""
    cols = list(range(h + 1))
    if any(c not in sub.columns for c in cols):
        return None
    growth = sub[cols].to_numpy(dtype=float) / 100.0 + 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = (growth[:, 1:] / growth[:, :-1] - 1.0) * 100.0
    return rets


def excursion_stats(sub, h):
    """Block B (adverse/favorable excursion) over columns 1..h of a
    path-panel subset (one row per anchor). A row counts only when
    COMPLETE to h (all of columns 1..h finite) -- "a path is scored at
    checkpoint h only when complete to h". Underscore-prefixed keys carry
    the raw per-anchor arrays (for episode-cluster CIs downstream); they
    are not meant for direct printing."""
    cols = list(range(1, h + 1))
    if any(c not in sub.columns for c in cols) or h not in sub.columns:
        return {"n": 0}
    M = sub[cols].to_numpy(dtype=float)
    term = sub[h].to_numpy(dtype=float)
    ok = np.all(np.isfinite(M), axis=1) & np.isfinite(term)
    out = {"n": int(ok.sum()), "_ok": ok}
    if not ok.any():
        return out
    Mok, termok = M[ok], term[ok]
    mins = Mok.min(axis=1)
    argmin = Mok.argmin(axis=1) + 1
    maxs = Mok.max(axis=1)
    argmax = Mok.argmax(axis=1) + 1
    out.update(
        mae_med=float(np.median(mins)), mae_q25=float(np.percentile(mins, 25)),
        mae_p10=float(np.percentile(mins, 10)), mae_worst=float(np.min(mins)),
        trough_d=float(np.median(argmin)),
        mfe_med=float(np.median(maxs)), mfe_q75=float(np.percentile(maxs, 75)),
        peak_d=float(np.median(argmax)),
        giveback_med=float(np.median(maxs - termok)),
        dip3=float(np.mean(mins < -3) * 100), dip5=float(np.mean(mins < -5) * 100),
        dip8=float(np.mean(mins < -8) * 100), dip12=float(np.mean(mins < -12) * 100),
        _mins=mins, _maxs=maxs, _term=termok,
    )
    return out


def barrier_touch(sub, h, levels=BARRIER_LEVELS):
    """P(touch -level%) / P(touch +level%) at any point within columns
    1..h, over the complete-to-h population (same criterion as
    `excursion_stats`). Close-only -- a floor on true intraday touch
    rates."""
    cols = list(range(1, h + 1))
    if any(c not in sub.columns for c in cols):
        return {"n": 0}
    M = sub[cols].to_numpy(dtype=float)
    ok = np.all(np.isfinite(M), axis=1)
    out = {"n": int(ok.sum())}
    if not ok.any():
        return out
    Mok = M[ok]
    mins, maxs = Mok.min(axis=1), Mok.max(axis=1)
    for lv in levels:
        out[f"touch_m{lv}"] = float(np.mean(mins <= -lv) * 100)
        out[f"touch_p{lv}"] = float(np.mean(maxs >= lv) * 100)
    return out


def bracket_outcomes(sub, h, stops=BRACKET_LEVELS, targets=BRACKET_LEVELS):
    """First-passage stop/target matrix over columns 1..h (close-only -- a
    same-day double-touch cannot occur for x, y > 0 since one day's return
    cannot be both <= -x and >= +y). For each (stop x%, target y%): P(target
    first), P(stop first), P(neither), and E[bracket return] (+y / -x / the
    terminal return at h, respectively -- a fixed-horizon exit when
    neither level is touched)."""
    cols = list(range(1, h + 1))
    if any(c not in sub.columns for c in cols) or h not in sub.columns:
        return {"n": 0, "brackets": {}}
    M = sub[cols].to_numpy(dtype=float)
    term = sub[h].to_numpy(dtype=float)
    ok = np.all(np.isfinite(M), axis=1) & np.isfinite(term)
    out = {"n": int(ok.sum()), "brackets": {}}
    if not ok.any():
        return out
    Mok, termok = M[ok], term[ok]
    for x in stops:
        stop_hit = Mok <= -x
        stop_any = stop_hit.any(axis=1)
        stop_day = np.where(stop_any, stop_hit.argmax(axis=1), h)
        for y in targets:
            tgt_hit = Mok >= y
            tgt_any = tgt_hit.any(axis=1)
            tgt_day = np.where(tgt_any, tgt_hit.argmax(axis=1), h)
            target_first = tgt_day < stop_day
            stop_first = stop_day < tgt_day
            neither = ~(target_first | stop_first)
            ret = np.where(target_first, float(y), np.where(stop_first, float(-x), termok))
            out["brackets"][(x, y)] = {
                "p_target": float(np.mean(target_first) * 100),
                "p_stop": float(np.mean(stop_first) * 100),
                "p_neither": float(np.mean(neither) * 100),
                "e_ret": float(np.mean(ret)),
            }
    return out


def fan_quantiles(sub, checkpoints=CHECKPOINTS, qs=FAN_QUANTILES):
    """Long frame (h, q, value, n) of the terminal-return distribution at
    each checkpoint -- the chartable fan. Each checkpoint scores over its
    OWN complete-to-h population (a later checkpoint has fewer complete
    anchors near the end of history, since its window runs off the end of
    the sample sooner); `q` is a float quantile in (0, 1) or the strings
    'mean' / 'hit'."""
    rows = []
    for h in checkpoints:
        if h not in sub.columns:
            continue
        v = sub[h].dropna().to_numpy(dtype=float)
        n = len(v)
        if n == 0:
            continue
        for q in qs:
            rows.append({"h": h, "q": q, "value": float(np.percentile(v, q * 100)), "n": n})
        rows.append({"h": h, "q": "mean", "value": float(np.mean(v)), "n": n})
        rows.append({"h": h, "q": "hit", "value": float(np.mean(v > 0) * 100), "n": n})
    return pd.DataFrame(rows)


def sizing(mae_q25, mae_p10, med_term):
    """Position weight (% NAV) that loses 1% of NAV at the q25 / p10 max
    adverse excursion, and reward/risk = median terminal return / |q25
    MAE| -- ETF_PATH_PLAYBOOK.md's sizing convention, same units, so a
    reader can compare a regime cell against that playbook's ETF rows."""
    def weight(mae):
        return (float(100.0 / abs(mae))
                if mae is not None and np.isfinite(mae) and mae != 0 else np.nan)
    size_q25, size_p10 = weight(mae_q25), weight(mae_p10)
    rr = (float(med_term / abs(mae_q25))
          if (mae_q25 is not None and np.isfinite(mae_q25) and mae_q25 != 0
              and med_term is not None and np.isfinite(med_term)) else np.nan)
    return size_q25, size_p10, rr


def cell_masks(M):
    """Ordered {label: boolean mask} for the rolling-basis comovement-zone
    marginals, the 3x3 comovement x DIX(lag-1) grid, and (when the frame
    carries realized-vol zones) the vol-parallel 3x3 -- the tradeable-
    timing cells the corrected regime study already defines. No new regime
    definitions are introduced here."""
    masks = OrderedDict()
    for cz in R.ZONES:
        masks[cz] = (M["cz_roll"] == cz)
    for cz in R.ZONES:
        for dz in R.DZONES:
            masks[f"{cz}x{dz}(l1)"] = (M["cz_roll"] == cz) & (M["dz_roll_l1"] == dz)
    if "vz_roll" in M.columns:
        for vz in R.VZONES:
            for dz in R.DZONES:
                masks[f"{vz}x{dz}(l1)"] = (M["vz_roll"] == vz) & (M["dz_roll_l1"] == dz)
    return masks


def cell_stats(P, rv, mask, family="ENV", h=HOLD, h2=H, seed=0):
    """One flat stats dict (fan, excursion at h and h2, barriers, brackets,
    volatility, sizing, episode-cluster CIs) for one cell's ENV anchors:
    every day `mask` holds. `P` is the forward-path panel on the PROXY's
    own calendar (from `forward_path_panel`); `mask` and `rv` (the frame's
    trailing realized vol, `M['rv']`) are on the FRAME's calendar. Episodes
    are contiguous runs of `mask` (positional contiguity on the frame's own
    index -- a gap in the frame does not merge two episodes, matching
    `intra_index_regime_study.run_ids`)."""
    m = mask.to_numpy(dtype=bool)
    ids_all = R.run_ids(m)
    dates = mask.index[m]
    ids = ids_all[m]
    n_days = len(dates)
    n_eps = int(len(np.unique(ids))) if n_days else 0
    gate = n_days >= GATE_DAYS and n_eps >= GATE_EPISODES
    out = {"family": family, "n_days": n_days, "n_eps": n_eps, "gate": gate}
    if not n_days:
        return out
    sub = P.reindex(dates)

    out["fan"] = fan_quantiles(sub)
    exc = excursion_stats(sub, h)
    out["exc"] = {k: v for k, v in exc.items() if not k.startswith("_")}
    exc2 = excursion_stats(sub, h2)
    out["exc2"] = {k: v for k, v in exc2.items() if not k.startswith("_")}
    out["bar"] = barrier_touch(sub, h)
    out["brk"] = bracket_outcomes(sub, h)

    vol = {"rv21_med": np.nan, "rv21_p90": np.nan, "vratio_med": np.nan,
          "vratio_gt1": np.nan, "vr21": np.nan, "z_gt1": np.nan, "z_gt2": np.nan}
    ci = {}
    ok = exc.get("_ok")
    if ok is not None and ok.any():
        rets_full = daily_rets_from_path(sub, h)
        rets = rets_full[ok]
        fv = rets.std(axis=1, ddof=0) * np.sqrt(252)
        vol["rv21_med"], vol["rv21_p90"] = float(np.median(fv)), float(np.percentile(fv, 90))
        rv_ok = rv.reindex(dates).to_numpy(dtype=float)[ok]
        rok = np.isfinite(rv_ok) & (rv_ok > 0)
        if rok.any():
            ratio = fv[rok] / rv_ok[rok]
            vol["vratio_med"] = float(np.median(ratio))
            vol["vratio_gt1"] = float(np.mean(ratio > 1) * 100)
        term = rets.sum(axis=1)
        dvar = float(np.var(rets.reshape(-1), ddof=0))
        if dvar > 0:
            vol["vr21"] = float(np.var(term, ddof=0) / (h * dvar))
        term21 = exc["_term"]
        with np.errstate(invalid="ignore", divide="ignore"):
            zdenom = rv_ok * np.sqrt(h / 252.0)
        zok = np.isfinite(zdenom) & (zdenom > 0)
        if zok.any():
            z = term21[zok] / zdenom[zok]
            vol["z_gt1"] = float(np.mean(np.abs(z) > 1) * 100)
            vol["z_gt2"] = float(np.mean(np.abs(z) > 2) * 100)

        if gate:
            ids_ok = ids[ok]
            ci["mean_r"] = R.cluster_boot_ci(term21, ids_ok, seed=seed)
            ci["mae_q25"] = R.cluster_boot_ci(
                exc["_mins"], ids_ok, seed=seed + 1,
                stat=lambda x: float(np.percentile(x, 25)))
            touch5 = np.where(exc["_mins"] <= -5, 100.0, 0.0)
            ci["touch_m5"] = R.cluster_boot_ci(touch5, ids_ok, seed=seed + 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_full = np.where(rok, fv / np.where(rv_ok == 0, np.nan, rv_ok), np.nan)
            ci["vratio"] = R.cluster_boot_ci(
                ratio_full, ids_ok, seed=seed + 3, stat=lambda x: float(np.median(x)))
    out["vol"] = vol
    out["ci"] = ci

    if exc.get("n"):
        med_term = float(np.median(exc["_term"]))
        s25, s10, rr = sizing(exc.get("mae_q25"), exc.get("mae_p10"), med_term)
        out["size"] = {"size_q25": s25, "size_p10": s10, "rr": rr, "med_term": med_term}
    return out


# ----------------------------------------------------------------------------
# ENTRY family: first day the cell's condition forms, cool-down apart
# ----------------------------------------------------------------------------
def entry_stats(M, P, mask, min_gap=HOLD, h=HOLD, seed=0):
    """ENTRY family: first-day-of-condition events (`min_gap`-session
    cool-down -- unchanged semantics from `intra_index_regime_study.
    entry_events`, so ENTRY numbers here are directly comparable to the
    regime study's own entry section), scored on the same excursion/
    barrier blocks as ENV. Entries are >= `min_gap` sessions apart by
    construction, so the events are treated as independent draws (a plain
    percentile bootstrap over events, not an episode-cluster resample).
    Gated at GATE_EVENTS (10)."""
    idx = R.entry_events(M, mask, min_gap=min_gap)
    n_events = len(idx)
    out = {"family": "ENTRY", "n_events": n_events, "gate": n_events >= GATE_EVENTS}
    if not n_events:
        return out
    dates = M.index[idx]
    sub = P.reindex(dates)
    exc = excursion_stats(sub, h)
    out["exc"] = {k: v for k, v in exc.items() if not k.startswith("_")}
    out["bar"] = barrier_touch(sub, h)
    if exc.get("n"):
        term = exc["_term"]
        out["mean_r"] = float(np.mean(term))
        out["hit"] = float(np.mean(term > 0) * 100)
        if out["gate"]:
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(term), size=(R.BOOT_B, len(term)))
            means = term[draws].mean(axis=1)
            out["ci_mean_r"] = tuple(float(x) for x in np.percentile(means, (2.5, 97.5)))
    return out


# ----------------------------------------------------------------------------
# EPISODE-HOLD family: a confirmed, variable-duration position
# ----------------------------------------------------------------------------
def episode_hold_paths(P, mask, confirm=3, cap=H):
    """One row per EPISODE-HOLD episode: enter at the close of the
    CONFIRM-th consecutive session the cell condition holds (DIX-zone
    episodes flicker daily -- see the design plan's "regime flicker" open
    risk -- so a naive first-day entry would churn), exit at the close of
    the first session the condition fails, capped at `cap` sessions. Skips
    episodes shorter than `confirm` (never confirm) and episodes still
    open at the end of the sample (no realized exit yet). Columns:
    start/exit (dates), duration, terminal/mae/mfe (%), trough_d/peak_d,
    hold_rv (annualized, from the hold's own daily returns), post_exit_r21
    (%, the 21-session return AFTER exit -- what the regime's ending costs
    or pays, pricing the jump risk the regime study's transition section
    left unpriced). `P` is the forward-path panel on the proxy's own
    calendar."""
    m = mask.to_numpy(dtype=bool)
    ids = R.run_ids(m)
    dates = mask.index
    n = len(m)
    rows = []
    for ep in np.unique(ids[ids >= 0]):
        pos = np.where(ids == ep)[0]
        start_pos, end_pos = pos[0], pos[-1]
        if end_pos - start_pos + 1 < confirm or end_pos + 1 >= n:
            continue    # never confirms, or the episode is still open (no realized exit)
        entry_pos = start_pos + confirm - 1
        exit_pos = min(end_pos + 1, entry_pos + cap)
        h = exit_pos - entry_pos
        if h <= 0:
            continue
        entry_date, exit_date = dates[entry_pos], dates[exit_pos]
        if entry_date not in P.index or h not in P.columns:
            continue
        path = P.loc[entry_date, list(range(h + 1))].to_numpy(dtype=float)
        if not np.isfinite(path).all():
            continue
        seg = path[1:]
        growth = path / 100.0 + 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = (growth[1:] / growth[:-1] - 1.0) * 100.0
        post_exit = (float(P.loc[exit_date, HOLD])
                    if exit_date in P.index and HOLD in P.columns
                    and np.isfinite(P.loc[exit_date, HOLD]) else np.nan)
        rows.append({
            "episode": int(ep), "start": entry_date, "exit": exit_date,
            "duration": int(h), "terminal": float(path[h]),
            "mae": float(seg.min()), "trough_d": int(seg.argmin()) + 1,
            "mfe": float(seg.max()), "peak_d": int(seg.argmax()) + 1,
            "hold_rv": float(np.std(rets, ddof=0) * np.sqrt(252)) if len(rets) else np.nan,
            "post_exit_r21": post_exit,
        })
    return pd.DataFrame(rows)


def hold_stats(P, mask, confirm=3, cap=H, seed=0):
    """EPISODE-HOLD family summary: confirmed, variable-duration holds
    (`episode_hold_paths`). Each row IS its own episode, so the
    episode-cluster CI on the terminal return degenerates to a standard
    per-observation bootstrap here (one observation per cluster) -- the
    same estimator, applied to a family whose anchors already are
    episodes. Gated at GATE_EPISODES (5)."""
    holds = episode_hold_paths(P, mask, confirm=confirm, cap=cap)
    n_eps = len(holds)
    out = {"family": "EPISODE-HOLD", "n_eps": n_eps, "gate": n_eps >= GATE_EPISODES}
    if not n_eps:
        return out
    dur = holds["duration"].to_numpy(dtype=float)
    term = holds["terminal"].to_numpy(dtype=float)
    mae = holds["mae"].to_numpy(dtype=float)
    post = holds["post_exit_r21"].dropna().to_numpy(dtype=float)
    out.update(
        dur_med=float(np.median(dur)), dur_q75=float(np.percentile(dur, 75)),
        dur_max=float(np.max(dur)),
        term_med=float(np.median(term)), term_hit=float(np.mean(term > 0) * 100),
        mae_med=float(np.median(mae)), mae_q25=float(np.percentile(mae, 25)),
        post_exit_med=float(np.median(post)) if len(post) else np.nan,
        post_exit_hit=float(np.mean(post > 0) * 100) if len(post) else np.nan,
    )
    if out["gate"]:
        ids = holds["episode"].to_numpy()
        out["ci_term"] = R.cluster_boot_ci(term, ids, seed=seed)
    return out


# ----------------------------------------------------------------------------
# Survival and the transition-within-the-hold split
# ----------------------------------------------------------------------------
def survival(mask, checkpoints=(5, 10, 21, 42, 63)):
    """P(the mask condition still holds at t+h) for h in `checkpoints`,
    scored over every ENV anchor (day mask holds) with a defined t+h row
    in the frame. NaN for a checkpoint with no scoreable anchor."""
    m = mask.to_numpy(dtype=bool)
    n = len(m)
    anchor_idx = np.where(m)[0]
    out = {}
    for h in checkpoints:
        if len(anchor_idx) == 0:
            out[h] = np.nan
            continue
        future = anchor_idx + h
        valid = future < n
        out[h] = float(np.mean(m[future[valid]]) * 100) if valid.any() else np.nan
    return out


def transition_within(M, mask, h=HOLD, zone_col="cz_roll", to="HighCorr"):
    """Per-anchor boolean: does `zone_col` reach `to` at any point within
    t+1..t+h (a regime-ending jump landing inside the hold)? Returns
    (hit, valid) Series indexed by the ENV anchor dates -- `valid` marks
    anchors whose t+h window is complete in the frame; `hit` is only
    meaningful where `valid` is True (it is False, not NaN, elsewhere, so
    it composes safely with `excursion_stats`' own completeness filter)."""
    m = mask.to_numpy(dtype=bool)
    zone = M[zone_col].to_numpy()
    n = len(m)
    anchor_idx = np.where(m)[0]
    hit = np.zeros(len(anchor_idx), dtype=bool)
    valid = np.zeros(len(anchor_idx), dtype=bool)
    for i, t in enumerate(anchor_idx):
        if t + h < n:
            valid[i] = True
            hit[i] = bool(np.any(zone[t + 1: t + h + 1] == to))
    dates = M.index[anchor_idx]
    return pd.Series(hit, index=dates), pd.Series(valid, index=dates)


# ----------------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------------
def fmt_gate(st):
    tag = "" if st.get("gate") else "  (below gate)"
    return f"n={st['n_days']}d/{st['n_eps']}ep{tag}"


def render_cell(label, st, proxy):
    """One cell's report block. Below the print gate only the header with
    counts prints -- no conditional statistic is printed below the gate
    (Rule 5 of the design plan, matching the regime study's own gate)."""
    header = f"--- {label}  {fmt_gate(st)} ---"
    if not st.get("n_days") or not st.get("gate"):
        return header
    lines = [header]
    fan = st.get("fan")
    if fan is not None and len(fan):
        hs = [h for h in CHECKPOINTS if h in set(fan["h"])]
        lines.append(f"  fan ({proxy} %, cum. return)   " + "  ".join(f"h={h:>3d}" for h in hs))
        for q in list(FAN_QUANTILES) + ["mean", "hit"]:
            row = fan[fan["q"] == q]
            if not len(row):
                continue
            vals = dict(zip(row["h"], row["value"]))
            qlabel = f"p{int(round(q * 100)):>2d}" if isinstance(q, float) else q
            fmt = "{:7.1f}" if q == "hit" else "{:+7.1f}"
            lines.append(f"    {qlabel:<6}" + "  ".join(
                fmt.format(vals.get(h, float("nan"))) for h in hs))
    exc = st.get("exc", {})
    if exc.get("n"):
        lines.append(
            f"  excursion ({HOLD}d, n={exc['n']}): MAE med {exc['mae_med']:+.1f}  "
            f"q25 {exc['mae_q25']:+.1f}  p10 {exc['mae_p10']:+.1f}  worst {exc['mae_worst']:+.1f}"
            f"  trough~d{exc['trough_d']:.0f}   MFE med {exc['mfe_med']:+.1f}  "
            f"q75 {exc['mfe_q75']:+.1f}  peak~d{exc['peak_d']:.0f}   "
            f"give-back {exc['giveback_med']:+.1f}")
        lines.append(
            f"    dips <-3/-5/-8/-12%: {exc['dip3']:.0f}% {exc['dip5']:.0f}% "
            f"{exc['dip8']:.0f}% {exc['dip12']:.0f}%")
    exc2 = st.get("exc2", {})
    if exc2.get("n") and H != HOLD:
        lines.append(
            f"  excursion ({H}d, n={exc2['n']}): MAE med {exc2['mae_med']:+.1f}  "
            f"q25 {exc2['mae_q25']:+.1f}   MFE med {exc2['mfe_med']:+.1f}")
    bar = st.get("bar", {})
    if bar.get("n"):
        lines.append(
            f"  barriers (closes, {HOLD}d): touch -3/-5/-8/-12%: "
            f"{bar['touch_m3']:.0f}% {bar['touch_m5']:.0f}% {bar['touch_m8']:.0f}% "
            f"{bar['touch_m12']:.0f}%   touch +3/+5/+8/+12%: "
            f"{bar['touch_p3']:.0f}% {bar['touch_p5']:.0f}% {bar['touch_p8']:.0f}% "
            f"{bar['touch_p12']:.0f}%")
    brk = st.get("brk", {}).get("brackets", {})
    if brk:
        lines.append("  bracket P(target first)% [rows=stop, cols=target +3/+5/+8]:")
        for x in BRACKET_LEVELS:
            row = [brk.get((x, y), {}).get("p_target", float("nan")) for y in BRACKET_LEVELS]
            lines.append(f"    -{x}%: " + "  ".join(f"{v:5.0f}" for v in row))
    vol = st.get("vol", {})
    if np.isfinite(vol.get("rv21_med", np.nan)):
        lines.append(
            f"  vol: fwd rv{HOLD} med {vol['rv21_med']:.1f} p90 {vol['rv21_p90']:.1f}   "
            f"fwd/trail ratio med {vol['vratio_med']:.2f} (>1 in {vol['vratio_gt1']:.0f}%)   "
            f"VR{HOLD} {vol['vr21']:.2f}   |z|>1 {vol['z_gt1']:.0f}%  |z|>2 {vol['z_gt2']:.0f}%"
            "   (Gaussian: 32%/4.6%)")
    size = st.get("size", {})
    if size and np.isfinite(size.get("size_q25", np.nan)):
        lines.append(
            f"  sizing: per 1% NAV at q25 MAE {size['size_q25']:.0f}%, "
            f"at p10 MAE {size['size_p10']:.0f}%   RR {size['rr']:.2f}")
    ci = st.get("ci", {})
    parts = [f"{k} [{lo:+.1f},{hi:+.1f}]" for k, (lo, hi) in ci.items() if np.isfinite(lo)]
    if parts:
        lines.append("  epCI: " + "   ".join(parts))
    return "\n".join(lines)


def render_survival(surv):
    parts = "  ".join(
        f"d{h}: {v:.0f}%" if np.isfinite(v) else f"d{h}: --" for h, v in surv.items())
    return f"  survival: still in cell at   {parts}"


def render_transition(hit, valid, sub_all, h):
    v = valid.to_numpy(dtype=bool)
    if not v.any():
        return None
    share = float(np.mean(hit.to_numpy(dtype=bool)[v]) * 100)
    hit_full = hit.reindex(sub_all.index).fillna(False).to_numpy(dtype=bool)
    exc_hit = excursion_stats(sub_all[hit_full], h)
    exc_not = excursion_stats(sub_all[~hit_full], h)
    mq_hit = exc_hit.get("mae_q25", float("nan"))
    mq_not = exc_not.get("mae_q25", float("nan"))
    return (f"    -> HighCorr within {h}d: {share:.0f}%   "
           f"MAE q25 if so {mq_hit:+.1f} (n={exc_hit.get('n', 0)}) "
           f"vs {mq_not:+.1f} if not (n={exc_not.get('n', 0)})")


def render_entry(st, proxy):
    if not st.get("n_events"):
        return "  ENTRY: n=0 events"
    if not st.get("gate"):
        return f"  ENTRY: n={st['n_events']} events  -- (below {GATE_EVENTS}-event gate)"
    exc = st.get("exc", {})
    s = (f"  ENTRY: n={st['n_events']} events   {proxy} mean {st['mean_r']:+.2f}%  "
        f"hit {st['hit']:.0f}%")
    if exc.get("n"):
        s += f"   MAE q25 {exc['mae_q25']:+.1f}"
    bar = st.get("bar", {})
    if np.isfinite(bar.get("touch_m5", np.nan)):
        s += f"   touch -5 {bar['touch_m5']:.0f}%"
    ci = st.get("ci_mean_r")
    if ci and np.isfinite(ci[0]):
        s += f"   CI [{ci[0]:+.1f},{ci[1]:+.1f}]"
    return s


def render_hold(st):
    if not st.get("n_eps"):
        return "  EPISODE-HOLD (confirm 3): n=0 episodes"
    if not st.get("gate"):
        return (f"  EPISODE-HOLD (confirm 3): n={st['n_eps']} episodes  "
                f"-- (below {GATE_EPISODES}-episode gate)")
    s = (f"  EPISODE-HOLD (confirm 3): n={st['n_eps']} episodes   "
        f"dur med {st['dur_med']:.0f}d q75 {st['dur_q75']:.0f}d max {st['dur_max']:.0f}d   "
        f"terminal med {st['term_med']:+.1f} hit {st['term_hit']:.0f}%   "
        f"MAE q25 {st['mae_q25']:+.1f}")
    if np.isfinite(st.get("post_exit_med", float("nan"))):
        s += f"   post-exit r21 med {st['post_exit_med']:+.1f} hit {st['post_exit_hit']:.0f}%"
    ci = st.get("ci_term")
    if ci and np.isfinite(ci[0]):
        s += f"   CI [{ci[0]:+.1f},{ci[1]:+.1f}]"
    return s


def report_index(name, M, meta, args):
    proxy = meta["proxy"]
    proxy_close = meta["proxy_close"]
    P = forward_path_panel(proxy_close, horizon=H)
    masks = cell_masks(M)
    print(f"##### {name} PATH STUDY: {meta['note']}  "
          f"({M.index.min().date()} -> {M.index.max().date()}, {len(M)} days) #####")
    print(f"NOTE: barriers/brackets are close-only (a floor on true intraday touch "
          f"rates); print gate {GATE_DAYS}d AND {GATE_EPISODES} episodes (ENTRY: "
          f"{GATE_EVENTS} events).\n")
    seed = 200
    for label, mask in masks.items():
        seed += 1
        st = cell_stats(P, M["rv"], mask, family="ENV", seed=seed)
        lines = [render_cell(f"{name} {label}", st, proxy)]
        if st.get("gate"):
            if label in R.ZONES:
                lines.append(render_survival(survival(mask)))
            if label == "LowCorr" or label.startswith("LowCorrx"):
                hit, valid = transition_within(M, mask, h=HOLD, to="HighCorr")
                sub_all = P.reindex(mask.index[mask.to_numpy(dtype=bool)])
                tline = render_transition(hit, valid, sub_all, HOLD)
                if tline:
                    lines.append(tline)
        lines.append(render_entry(entry_stats(M, P, mask, seed=seed + 1000), proxy))
        lines.append(render_hold(hold_stats(P, mask, seed=seed + 2000)))
        print("\n".join(lines))
        print()
    return P


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="docs/index.html",
                    help="built dashboard HTML carrying the payload (default docs/index.html)")
    ap.add_argument("--indices", default="ndx,spx,iwm",
                    help="comma list of ndx/spx/iwm (default all three; spx and iwm "
                         "need network or a warm cache)")
    ap.add_argument("--basket-size", type=int, default=R.BASKET_N)
    ap.add_argument("--cache-dir", default=".ndx_dark_cache")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--horizon", type=int, default=H, help="secondary path horizon (default 63)")
    ap.add_argument("--no-ci", action="store_true", help="skip episode-cluster CIs (faster)")
    ap.add_argument("--csv", default=None, help="(wired in a later phase)")
    ap.add_argument("--risk-csv", default=None, help="(wired in a later phase)")
    args = ap.parse_args()

    P_html = R.load_payload(args.html)
    print(f"Payload generated: {P_html.get('generated')}")
    wanted = [w.strip().upper() for w in args.indices.split(",") if w.strip()]

    for name in ("NDX", "SPX", "IWM"):
        if name not in wanted:
            continue
        try:
            if name == "NDX":
                M, meta = R.build_ndx_frame(P_html)
            else:
                M, meta = R.build_basket_frame(P_html, name, args.basket_size,
                                               args.cache_dir, refresh=args.refresh)
        except RuntimeError as e:
            print(f"##### {name}: SKIPPED ({e}) #####\n", file=sys.stderr)
            continue
        report_index(name, M, meta, args)


if __name__ == "__main__":
    main()
