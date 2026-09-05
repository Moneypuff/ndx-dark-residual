#!/usr/bin/env python3
"""
Regime return-path metrics -- pure computation.
================================================

`index_comovement_study.py` answers "what is the mean/median 1-month forward
return in each of the 27 NDX/SPX/IWM DIX comovement regimes?" That says
nothing about the *shape* of the month: whether it is calm or violent,
whether it trends or chops, how deep the drawdown along the way runs, and how
wide the range of plausible outcomes is. This module computes that path
layer -- see `REGIME_PATHS_PLAN.md` for the full spec.

No I/O, no network: every function here takes pandas/numpy objects already in
memory and returns pandas/numpy objects. `regime_paths_study.py` (the CLI)
and `build_comovement.py` (the page builder) are the only callers that touch
files or the network.

Reused, not reimplemented, from elsewhere in the repo:
  * `index_comovement_study.build_aligned` -- the per-day regime frame (DIX5
    deciles -> Low/Mid/High -> 3-letter code).
  * `index_comovement_study.block_boot_ci` -- moving-block bootstrap CI.
  * `gdx_chase_study.path_matrix` -- forward %-from-event-close path matrix,
    reused by `fan_quantiles`.
"""
import numpy as np
import pandas as pd

H = 21          # path horizon, sessions -- matches the existing 1-month forward return
TRAIL = 21      # trailing-vol window for rv_ratio
MIN_N = 10      # a cell with fewer scored paths than this reports n only, no numbers
FAN_Q = (10, 25, 50, 75, 90)

# Cell-classification thresholds (§2.3 of the plan). Deliberately coarse --
# the page always shows the underlying numbers alongside the label.
VOL_HOT = 1.15      # med(rv) >= VOL_HOT * base(rv)          -> "hot"
VOL_QUIET = 0.85    # med(rv) <= VOL_QUIET * base(rv)         -> "quiet"
ER_TREND = 1.20     # med(er) >= ER_TREND * base(er)          -> "trend"
ER_CHOP = 0.80      # med(er) <= ER_CHOP * base(er)           -> "chop"
DIR_MIN_PP = 1.0    # |med(r21)| below this (pp) -> "flat"


# ----------------------------------------------------------------------------
# Per-path metrics (§2.1)
# ----------------------------------------------------------------------------
def path_metrics(close, horizon=H, trail=TRAIL):
    """One row per date of `close` with the forward-path metrics for the
    `horizon` sessions starting at that date (close-to-close, close[0] is the
    observation day itself so `c_0 == 0`).

    `close` : pd.Series of adjusted closes, sorted, unique DatetimeIndex.

    Columns: r21, rv, rv_ratio, mae, mae_day, mfe, mfe_day, range, er, taw,
    xings, ac1, big_dn, big_up. NaN wherever the forward window (or, for
    rv_ratio, the trailing window) is incomplete. A date needs `horizon`
    sessions strictly *after* it to score; the last `horizon` dates in
    `close` are therefore always NaN here (same convention as
    `compute_forward_return` elsewhere in the repo).
    """
    idx = close.index
    p = close.to_numpy(dtype=float)
    n = len(p)
    cols = {k: np.full(n, np.nan) for k in
            ("r21", "rv", "rv_ratio", "mae", "mae_day", "mfe", "mfe_day",
             "range", "er", "taw", "xings", "ac1", "big_dn", "big_up")}

    # Trailing daily log returns (for rv_ratio's denominator), aligned so
    # trail_rv[i] uses the `trail` sessions ending at i (inclusive).
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.diff(np.log(p))  # dlog[i] = log(p[i+1]/p[i]), len n-1
    trail_rv = np.full(n, np.nan)
    for i in range(trail, n):
        seg = dlog[i - trail:i]  # the `trail` daily log-returns ending at i
        if np.all(np.isfinite(seg)):
            trail_rv[i] = np.std(seg, ddof=1) * np.sqrt(252) * 100.0

    for i in range(n - horizon):
        seg = p[i:i + horizon + 1]
        if not np.all(np.isfinite(seg)) or seg[0] == 0:
            continue
        c = seg / seg[0] - 1.0  # cumulative return path, c[0] == 0, in fraction
        r_daily = np.diff(np.log(seg))  # horizon daily log-returns

        cols["r21"][i] = c[-1] * 100.0
        rv = np.std(r_daily, ddof=1) * np.sqrt(252) * 100.0
        cols["rv"][i] = rv
        if np.isfinite(trail_rv[i]) and trail_rv[i] > 0:
            cols["rv_ratio"][i] = rv / trail_rv[i]

        mae_day = int(np.argmin(c))
        mfe_day = int(np.argmax(c))
        cols["mae"][i] = c[mae_day] * 100.0
        cols["mae_day"][i] = mae_day
        cols["mfe"][i] = c[mfe_day] * 100.0
        cols["mfe_day"][i] = mfe_day
        cols["range"][i] = cols["mfe"][i] - cols["mae"][i]

        daily_moves = np.abs(np.diff(c))  # horizon steps, in fraction
        total_move = daily_moves.sum()
        cols["er"][i] = abs(c[-1]) / total_move if total_move > 0 else np.nan

        cols["taw"][i] = np.mean(c[1:] > 0) * 100.0
        signs = np.sign(c[1:])
        nz = signs[signs != 0]
        cols["xings"][i] = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0

        if len(r_daily) > 2 and np.std(r_daily[:-1]) > 0 and np.std(r_daily[1:]) > 0:
            cols["ac1"][i] = float(np.corrcoef(r_daily[:-1], r_daily[1:])[0, 1])
        cols["big_dn"][i] = int(np.sum(np.diff(c) <= -0.01))
        cols["big_up"][i] = int(np.sum(np.diff(c) >= 0.01))

    return pd.DataFrame(cols, index=idx)


def fan_quantiles(close, dates, horizon=H, q=FAN_Q):
    """(len(q), horizon+1) array of cumulative-%-return quantiles across the
    forward paths starting on `dates`. Reuses `gdx_chase_study.path_matrix`
    (imported lazily to avoid a hard dependency on its Yahoo-adjacent module
    graph for callers that only need the pure metrics)."""
    from gdx_chase_study import path_matrix
    dates = [d for d in dates if d in close.index]
    if not dates:
        return np.full((len(q), horizon + 1), np.nan)
    P = path_matrix(close, dates, horizon)  # (horizon+1) x n, % from entry
    M = P.to_numpy()
    out = np.full((len(q), horizon + 1), np.nan)
    for r in range(horizon + 1):
        row = M[r]
        row = row[np.isfinite(row)]
        if len(row):
            out[:, r] = np.percentile(row, q)
    return out


# ----------------------------------------------------------------------------
# Episode / run-length accounting (§0a of the plan)
# ----------------------------------------------------------------------------
def run_lengths(code):
    """Contiguous-run (episode) table for a Series of regime codes indexed by
    date. Returns one row per episode: code, start, length (sessions)."""
    code = code.dropna()
    if code.empty:
        return pd.DataFrame(columns=["code", "start", "length"])
    vals = code.to_numpy()
    idx = code.index
    starts = [0] + [i for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
    starts.append(len(vals))
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        rows.append({"code": vals[a], "start": idx[a], "length": b - a})
    return pd.DataFrame(rows)


def run_length_at(code):
    """For each date, the number of consecutive sessions (including that
    date) its code has held so far -- i.e. how many sessions *into* the
    current episode that date sits. Used for the `--min-run` cut."""
    code = code.copy()
    vals = code.to_numpy()
    out = np.zeros(len(vals), dtype=int)
    run = 0
    prev = object()
    for i, v in enumerate(vals):
        run = run + 1 if v == prev else 1
        out[i] = run
        prev = v
    return pd.Series(out, index=code.index)


# ----------------------------------------------------------------------------
# Cell table (§2.4) and classification (§2.3)
# ----------------------------------------------------------------------------
_METRIC_COLS = ("r21", "rv", "rv_ratio", "mae", "mae_day", "mfe", "mfe_day",
                "range", "er", "taw", "xings", "ac1", "big_dn", "big_up")


def _cell_row(sub_metrics, sub_code, code_val, n_episodes, with_ci=False, seed=0):
    """One cell_table row from the metrics rows selected for one (regime,
    index, lens) combination."""
    from index_comovement_study import block_boot_ci

    out = {"regime": code_val, "n": int(len(sub_metrics)), "n_episodes": int(n_episodes)}
    for col in _METRIC_COLS:
        r = sub_metrics[col].dropna()
        if len(r) == 0:
            out[col + "_med"] = np.nan
            out[col + "_q25"] = np.nan
            out[col + "_q75"] = np.nan
            continue
        out[col + "_med"] = round(float(r.median()), 2)
        out[col + "_q25"] = round(float(r.quantile(0.25)), 2)
        out[col + "_q75"] = round(float(r.quantile(0.75)), 2)
    r21 = sub_metrics["r21"].dropna()
    if len(r21):
        out["r21_sd"] = round(float(r21.std(ddof=1)), 2) if len(r21) > 1 else np.nan
        out["r21_iqr"] = round(float(r21.quantile(0.75) - r21.quantile(0.25)), 2)
        out["r21_q10"] = round(float(r21.quantile(0.10)), 2)
        out["r21_q90"] = round(float(r21.quantile(0.90)), 2)
    else:
        out["r21_sd"] = out["r21_iqr"] = out["r21_q10"] = out["r21_q90"] = np.nan

    for col in ("rv", "er", "mae"):
        r = sub_metrics[col].dropna()
        if with_ci and len(r) >= 2 * 21:
            lo, hi = block_boot_ci(r.to_numpy(), seed=seed)
            out[f"{col}_ci_lo"] = round(lo, 2) if np.isfinite(lo) else np.nan
            out[f"{col}_ci_hi"] = round(hi, 2) if np.isfinite(hi) else np.nan
        else:
            out[f"{col}_ci_lo"] = np.nan
            out[f"{col}_ci_hi"] = np.nan
    return out


_BLANKABLE_SUFFIXES = ("_med", "_q25", "_q75", "_sd", "_iqr", "_q10", "_q90",
                      "_ci_lo", "_ci_hi")


def _blank_if_small(row, min_n=MIN_N):
    """Degrade-honestly rule (plan §1.5): below `min_n` scored paths, every
    computed statistic is blanked to NaN -- only regime/n/n_episodes/index
    survive. Callers render n plus a dash, never a number built on too few
    observations."""
    if row["n"] >= min_n:
        return row
    for k in list(row):
        if k.endswith(_BLANKABLE_SUFFIXES):
            row[k] = np.nan
    return row


def cell_table(A, metrics, index_name, min_run=1, with_ci=True, min_n=MIN_N):
    """Tidy per-regime table for one index.

    `A`       : index_comovement_study.build_aligned(...) output (must carry
                a `code` column -- NDX_z/SPX_z/IWM_z first letters).
    `metrics` : path_metrics(close_for_index_name) output, same date range.
    `index_name` : label only (carried into the 'index' column).
    `min_run` : keep only dates whose code has held for >= min_run
                consecutive sessions (the persistence cut, §2.2/§2.4.5).

    Returns one row per observed code, sorted by n descending, plus a first
    row labelled 'BASELINE' over every scored date (subject to the same
    min_run filter).
    """
    joined = A[["code"]].join(metrics, how="inner")
    if min_run > 1:
        rl = run_length_at(A["code"])
        joined = joined[rl.reindex(joined.index) >= min_run]

    ep = run_lengths(A["code"])
    ep_counts = ep["code"].value_counts()

    base_row = dict(_cell_row(joined, joined["code"], "BASELINE",
                              int(len(ep)), with_ci=with_ci, seed=0), index=index_name)
    rows = [_blank_if_small(base_row, min_n)]
    seed = 1
    for code_val, g in joined.groupby("code"):
        row = _cell_row(g, g["code"], code_val, int(ep_counts.get(code_val, 0)),
                        with_ci=with_ci, seed=seed)
        row["index"] = index_name
        rows.append(_blank_if_small(row, min_n))
        seed += 1
    out = pd.DataFrame(rows)
    is_base = out["regime"] == "BASELINE"
    out = pd.concat([out[is_base], out[~is_base].sort_values("n", ascending=False)],
                    ignore_index=True)
    return out


def vol_matched_baseline(A, metrics, code, base_metrics=None, terciles=None):
    """Vol-persistence control (§2.4.2): the forward `rv`/`er` a cell would
    show if it simply reproduced the trailing-vol tercile *mix* of the whole
    sample, rather than any DIX signal. Splits every scored day into trailing
    -vol terciles, then reweights the cell's own tercile composition against
    each tercile's baseline median.

    Returns {'rv': matched_rv_or_nan, 'er': matched_er_or_nan,
             'mix': {tercile_label: weight, ...}} or all-NaN if the cell or
    the trailing-vol column is unusable.
    """
    joined = A[["code"]].join(metrics, how="inner").dropna(subset=["rv_ratio"])
    # rv_ratio = fwd_rv / trail_rv, so trail_rv = fwd_rv / rv_ratio (avoids
    # recomputing trailing vol -- it's implicit in what path_metrics stored).
    with np.errstate(divide="ignore", invalid="ignore"):
        trail_rv = joined["rv"] / joined["rv_ratio"]
    joined = joined.assign(trail_rv=trail_rv).dropna(subset=["trail_rv"])
    if joined.empty:
        return {"rv": np.nan, "er": np.nan, "mix": {}}

    if terciles is None:
        terciles = joined["trail_rv"].quantile([1 / 3, 2 / 3]).to_numpy()
    lo_cut, hi_cut = terciles
    tercile = np.where(joined["trail_rv"] <= lo_cut, "low",
                       np.where(joined["trail_rv"] >= hi_cut, "high", "mid"))
    joined = joined.assign(tercile=tercile)

    base_med = (joined.groupby("tercile")[["rv", "er"]].median()
               if base_metrics is None else base_metrics)

    cell = joined[joined["code"] == code]
    if cell.empty:
        return {"rv": np.nan, "er": np.nan, "mix": {}}
    mix = cell["tercile"].value_counts(normalize=True).to_dict()

    out = {"mix": {k: round(float(v), 3) for k, v in mix.items()}}
    for col in ("rv", "er"):
        val = sum(w * base_med.loc[t, col] for t, w in mix.items() if t in base_med.index)
        out[col] = round(float(val), 2) if mix else np.nan
    return out


def classify(row, base):
    """The one-word (well, three-word) cell label of §2.3, from a cell_table
    row and the BASELINE row of the same table (both dict-like with the
    *_med keys)."""
    def get(d, k):
        v = d[k] if k in d else np.nan
        return v if pd.notna(v) else np.nan

    rv, base_rv = get(row, "rv_med"), get(base, "rv_med")
    er, base_er = get(row, "er_med"), get(base, "er_med")
    r21 = get(row, "r21_med")

    if pd.isna(rv) or pd.isna(base_rv) or base_rv == 0:
        vol_state = "n/a"
    elif rv >= VOL_HOT * base_rv:
        vol_state = "hot"
    elif rv <= VOL_QUIET * base_rv:
        vol_state = "quiet"
    else:
        vol_state = "normal"

    if pd.isna(er) or pd.isna(base_er) or base_er == 0:
        trend_state = "n/a"
    elif er >= ER_TREND * base_er:
        trend_state = "trend"
    elif er <= ER_CHOP * base_er:
        trend_state = "chop"
    else:
        trend_state = "mixed"

    if pd.isna(r21):
        direction = "n/a"
    elif r21 >= DIR_MIN_PP:
        direction = "up"
    elif r21 <= -DIR_MIN_PP:
        direction = "down"
    else:
        direction = "flat"

    return f"{vol_state} {trend_state} {direction}"


def entry_cell_table(A, metrics, index_name, codes, min_gap=21):
    """Same row shape as `cell_table`, scored only at the first day each
    regime forms (`index_comovement_study.entry_events`, `min_gap`-session
    cool-down). Unlike `cell_table`, never blanks small cells -- the entry
    lens is inherently low-n by construction (that's the point of it), and
    the caller decides which codes are worth showing."""
    from index_comovement_study import entry_events

    rows = []
    for code in codes:
        dates, _ = entry_events(A, code, min_gap=min_gap)
        dates = [d for d in dates if d in metrics.index]
        sub = metrics.loc[dates] if dates else metrics.iloc[0:0]
        row = _cell_row(sub, None, code, len(dates), with_ci=False)
        row["index"] = index_name
        rows.append(row)
    return pd.DataFrame(rows)
