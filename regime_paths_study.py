#!/usr/bin/env python3
"""
Regime return-path study: how the following month UNFOLDS, not just where it
ends.
=============================================================================

`index_comovement_study.py` scores each of the 27 NDX/SPX/IWM DIX
comovement regimes by the mean/median 1-month forward return. That answers
"how much" -- it says nothing about "how": whether the month is calm or
violent, whether it grinds in one direction or whipsaws, how deep the
drawdown along the way runs, and how wide the cone of plausible outcomes is.

This study adds that path layer, per regime and per index, using the metrics
in `regime_paths.py` (forward realized vol and vol-vs-trailing ratio, max
adverse/favourable excursion, Kaufman efficiency ratio, time above water,
zero crossings, terminal-return dispersion, fan-chart quantiles) under two
lenses -- every-day (matches the existing table, heavily overlapping,
descriptive of the environment) and entry-day (first day a regime forms,
21-session cool-down, closest to independent, the honest read on whether a
setup is tradeable) -- plus four honesty controls: block-bootstrap CIs, a
vol-persistence control (is "hot" just this cell's trailing-vol mix?),
expanding-window (no-look-ahead) deciles, and an out-of-sample split.

See `REGIME_PATHS_PLAN.md` for the full spec and `REGIME_PATHS_FINDINGS.md`
for the write-up this CLI's output feeds.

Data source
-----------
Same as `index_comovement_study.py` (DIX regimes, from the dashboard
payload) joined to QQQ/SPY/IWM adjusted closes via the repo's own Yahoo
loader (same call `build_comovement.py` makes) -- no live FINRA re-fetch.

Usage
-----
    python regime_paths_study.py --cache-dir .ndx_dark_cache
    python regime_paths_study.py --lens entry --min-run 5
    python regime_paths_study.py --csv regime_paths.csv --fan-csv regime_paths_fan.csv
"""
import argparse

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import regime_paths as R
from build_comovement import PROXY
from index_comovement_study import (IDX, EXP_MIN, OOS_SPLIT, load_payload,
                                    build_aligned, entry_events)

HEADLINE_CODES = ("LLH", "HML", "MML", "HHL", "LLL", "HHH")
MIN_RUN_DEFAULT = 3
KEY_COLS = ("r21", "rv", "rv_ratio", "mae", "mfe", "er", "taw", "ac1")


# ----------------------------------------------------------------------------
# Frame assembly
# ----------------------------------------------------------------------------
def load_prices(P, cache_dir):
    """{'NDX': close_series, 'SPX': ..., 'IWM': ...} of QQQ/SPY/IWM adjusted
    closes, over the DIX payload's date range (with a small buffer)."""
    all_dates = pd.to_datetime(
        P["rel"]["dates"] + P["spx"]["dates"] + P["iwm"]["dates"])
    lo = all_dates.min() - pd.Timedelta(days=10)
    hi = all_dates.max() + pd.Timedelta(days=2)
    panels = N.load_yahoo_panels(list(PROXY.values()), lo, hi,
                                 cache_dir=cache_dir or None, label="PATHS")
    adjclose = panels["adjclose"][list(PROXY.values())].dropna(how="all")
    return {k: adjclose[PROXY[k]].dropna() for k in IDX}


def build_metrics(prices, horizon=R.H, trail=R.TRAIL):
    return {k: R.path_metrics(s, horizon=horizon, trail=trail) for k, s in prices.items()}


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------
def episode_report(A):
    ep = R.run_lengths(A["code"])
    summ = (ep.groupby("code")["length"]
           .agg(episodes="count", days="sum", median_len="median", max_len="max")
           .sort_values("days", ascending=False))
    missing = R.unobserved_codes(A["code"])
    lines = ["=== SECTION 1: PERSISTENCE -- how long a regime code holds ===",
             f"  {len(A)} scored days, {len(ep)} contiguous episodes across "
             f"{ep['code'].nunique()} of 27 codes.",
             f"  Episode length: median {ep['length'].median():.0f}d, "
             f"75th pct {ep['length'].quantile(0.75):.0f}d, max {ep['length'].max():.0f}d.",
             "  A regime-conditioned statistic describes an ENVIRONMENT most of the",
             "  time, not a month spent continuously in that regime -- see the entry",
             "  lens (Section 4) for the closer-to-independent read.",
             f"  {len(missing)} of 27 codes NEVER observed in this sample: "
             + (", ".join(missing) if missing else "(none)")
             + " -- zero evidence, not just rare; see the note below the table.\n",
             summ.to_string()]
    return "\n".join(lines)


def baseline_report(cell_tables):
    lines = ["=== SECTION 2: BASELINE PATH PROFILE (every-day lens) ==="]
    for k in IDX:
        base = cell_tables[k].set_index("regime").loc["BASELINE"]
        lines.append(
            f"  {k:3s}  n={base['n']:5.0f}   r21 {base['r21_med']:+.2f}% "
            f"(IQR {base['r21_iqr']:.2f})   fwd rv {base['rv_med']:.1f}%   "
            f"rv/trail {base['rv_ratio_med']:.2f}x   max DD {base['mae_med']:+.2f}%   "
            f"max run-up {base['mfe_med']:+.2f}%   ER {base['er_med']:.2f}   "
            f"time-above-water {base['taw_med']:.0f}%")
    return "\n".join(lines)


def cell_report(cell_tables, A, metrics, label):
    lines = [f"=== SECTION 3: PER-REGIME PATH TABLE, every-day lens ({label}) ==="]
    for k in IDX:
        tbl = cell_tables[k].set_index("regime", drop=False)
        base = tbl.loc["BASELINE"]
        rows = []
        for _, row in tbl[tbl["regime"] != "BASELINE"].iterrows():
            lbl = R.classify(row, base)
            vm = R.vol_matched_baseline(A, metrics[k], row["regime"])
            ci = (f"[{row['rv_ci_lo']:+.1f},{row['rv_ci_hi']:+.1f}]"
                 if np.isfinite(row.get("rv_ci_lo", np.nan)) else "--")
            rows.append({
                "regime": row["regime"], "n": int(row["n"]), "ep": int(row["n_episodes"]),
                "r21": row["r21_med"], "rv": row["rv_med"], "rv_CI": ci,
                "rv_ratio": row["rv_ratio_med"], "mae": row["mae_med"],
                "mfe": row["mfe_med"], "er": row["er_med"], "taw": row["taw_med"],
                "r21_iqr": row["r21_iqr"], "vm_rv": vm.get("rv", np.nan),
                "vm_er": vm.get("er", np.nan), "label": lbl,
            })
        df = pd.DataFrame(rows).sort_values("n", ascending=False)
        lines.append(f"\n  --- {k} ---")
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", "{:+.2f}".format):
            lines.append(df.to_string(index=False))
    return "\n".join(lines)


def entry_report(A, metrics):
    codes = list(HEADLINE_CODES)
    for code in sorted(A["code"].unique()):
        if code not in codes:
            dates, _ = entry_events(A, code)
            if len(dates) >= R.MIN_N:
                codes.append(code)

    lines = ["=== SECTION 4: ENTRY LENS (first day a regime forms, "
             "21-session cool-down) ===",
             f"  Headline six ({', '.join(HEADLINE_CODES)}) always shown; "
             f"other codes shown only with >= {R.MIN_N} entries."]
    for k in IDX:
        et = R.entry_cell_table(A, metrics[k], k, codes).sort_values(
            "n", ascending=False)
        lines.append(f"\n  --- {k} ---")
        show = et[["regime", "n", "n_episodes", "r21_med", "rv_med", "er_med",
                  "mae_med", "mfe_med"]].rename(columns={"n_episodes": "n_ep"})
        with pd.option_context("display.width", 160, "display.float_format",
                               "{:+.2f}".format):
            lines.append(show.to_string(index=False))
    return "\n".join(lines)


def min_run_report(A, metrics, full_tables, min_run):
    lines = [f"=== SECTION 5: PERSISTENCE CUT (--min-run {min_run}: only days "
             f"whose code has already held {min_run}+ sessions) ===",
             "  Cells whose label CHANGES vs the every-day table (Section 3); "
             "unchanged cells are omitted."]
    any_change = False
    for k in IDX:
        cut = R.cell_table(A, metrics[k], k, min_run=min_run, with_ci=False)
        cut_by = cut.set_index("regime")
        full_by = full_tables[k].set_index("regime")
        base_cut = cut_by.loc["BASELINE"]
        base_full = full_by.loc["BASELINE"]
        rows = []
        for code in cut_by.index:
            if code == "BASELINE" or code not in full_by.index:
                continue
            lbl_cut = R.classify(cut_by.loc[code], base_cut)
            lbl_full = R.classify(full_by.loc[code], base_full)
            if lbl_cut != lbl_full:
                rows.append({"regime": code, "n_full": int(full_by.loc[code, "n"]),
                            "n_cut": int(cut_by.loc[code, "n"]),
                            "label_everyday": lbl_full, f"label_min{min_run}": lbl_cut})
        if rows:
            any_change = True
            lines.append(f"\n  --- {k} ---")
            lines.append(pd.DataFrame(rows).to_string(index=False))
    if not any_change:
        lines.append("\n  No cell's label changed under the persistence cut.")
    return "\n".join(lines)


def expanding_report_full(A_exp, metrics, full_tables):
    lines = ["=== SECTION 6: EXPANDING (NO-LOOK-AHEAD) DECILES ===",
             f"  {len(A_exp)} scored days (min {EXP_MIN} obs before a day is ranked). "
             "Cells whose label CHANGES vs the full-sample-decile table (Section 3); "
             "unchanged cells are omitted."]
    any_change = False
    for k in IDX:
        exp_tbl = R.cell_table(A_exp, metrics[k], k, with_ci=False)
        exp_by = exp_tbl.set_index("regime")
        full_by = full_tables[k].set_index("regime")
        if "BASELINE" not in exp_by.index:
            continue
        base_exp = exp_by.loc["BASELINE"]
        base_full = full_by.loc["BASELINE"]
        rows = []
        for code in exp_by.index:
            if code == "BASELINE" or code not in full_by.index:
                continue
            lbl_exp = R.classify(exp_by.loc[code], base_exp)
            lbl_full = R.classify(full_by.loc[code], base_full)
            if lbl_exp != lbl_full:
                rows.append({"regime": code, "n_full": int(full_by.loc[code, "n"]),
                            "n_exp": int(exp_by.loc[code, "n"]),
                            "label_full": lbl_full, "label_expanding": lbl_exp})
        if rows:
            any_change = True
            lines.append(f"\n  --- {k} ---")
            lines.append(pd.DataFrame(rows).to_string(index=False))
    if not any_change:
        lines.append("\n  No cell's label changed under expanding-window cutoffs.")
    return "\n".join(lines)


def oos_report(P, prices, full_tables, split=OOS_SPLIT):
    A_full = build_aligned(P, basis="full")
    tr = A_full[A_full.index < split]
    te = A_full[A_full.index >= split].copy()
    lines = [f"=== SECTION 7: OUT-OF-SAMPLE (cutoffs fit < {split}, scored >= {split}) ==="]
    if len(tr) < 300 or len(te) < 100:
        return "\n".join(lines + ["  insufficient data for the split"])

    for i in IDX:
        lo_c, hi_c = tr[i + "_dix5"].quantile([0.30, 0.70])
        te[i + "_z_oos"] = np.where(te[i + "_dix5"] <= lo_c, "L",
                                    np.where(te[i + "_dix5"] >= hi_c, "H", "M"))
    te["code"] = te["NDX_z_oos"] + te["SPX_z_oos"] + te["IWM_z_oos"]

    any_change = False
    for k in IDX:
        m = R.path_metrics(prices[k], horizon=R.H, trail=R.TRAIL)
        oos_tbl = R.cell_table(te[["code"]], m, k, with_ci=False)
        oos_by = oos_tbl.set_index("regime")
        full_by = full_tables[k].set_index("regime")
        if "BASELINE" not in oos_by.index:
            continue
        base_oos = oos_by.loc["BASELINE"]
        base_full = full_by.loc["BASELINE"]
        rows = []
        for code in oos_by.index:
            if code == "BASELINE" or code not in full_by.index or oos_by.loc[code, "n"] < 21:
                continue
            lbl_oos = R.classify(oos_by.loc[code], base_oos)
            lbl_full = R.classify(full_by.loc[code], base_full)
            rows.append({"regime": code, "n_oos": int(oos_by.loc[code, "n"]),
                        "label_full": lbl_full, "label_oos": lbl_oos,
                        "changed": lbl_oos != lbl_full})
            any_change = any_change or lbl_oos != lbl_full
        if rows:
            lines.append(f"\n  --- {k} (only codes with >= 21 OOS days) ---")
            lines.append(pd.DataFrame(rows).to_string(index=False))
    if not any_change:
        lines.append("\n  (no rows met the >= 21 OOS-day floor, or none changed label)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="docs/index.html",
                    help="built dashboard HTML carrying the DIX payload (default docs/index.html)")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR,
                    help="Yahoo price cache directory")
    ap.add_argument("--basis", choices=["full", "expanding"], default="full",
                    help="decile basis for the main tables (default: full, matching the page)")
    ap.add_argument("--lens", choices=["everyday", "entry", "both"], default="both",
                    help="which lens's main section(s) to print (default: both)")
    ap.add_argument("--min-run", type=int, default=MIN_RUN_DEFAULT,
                    help=f"persistence cut for Section 5 (default {MIN_RUN_DEFAULT})")
    ap.add_argument("--csv", default=None, help="write the tidy every-day cell table here")
    ap.add_argument("--fan-csv", default=None, help="write fan-quantile rows here")
    ap.add_argument("--skip-robustness", action="store_true",
                    help="skip Sections 5-7 (persistence/expanding/OOS -- slower)")
    args = ap.parse_args()

    P = load_payload(args.html)
    A = build_aligned(P, basis=args.basis)
    prices = load_prices(P, args.cache_dir)
    metrics = build_metrics(prices)

    print(f"Payload generated: {P.get('generated')}")
    print(f"Scored days: {len(A)}  [{A.index.min().date()} -> {A.index.max().date()}]  "
         f"basis={args.basis}\n")

    print(episode_report(A))
    print()

    full_tables = {k: R.cell_table(A, metrics[k], k, with_ci=True) for k in IDX}
    print(baseline_report(full_tables))
    print()

    if args.lens in ("everyday", "both"):
        print(cell_report(full_tables, A, metrics, args.basis))
        print()
    if args.lens in ("entry", "both"):
        print(entry_report(A, metrics))
        print()

    if not args.skip_robustness:
        print(min_run_report(A, metrics, full_tables, args.min_run))
        print()
        A_exp = build_aligned(P, basis="expanding")
        if len(A_exp) >= 300:
            print(expanding_report_full(A_exp, metrics, full_tables))
        else:
            print(f"=== SECTION 6: EXPANDING (NO-LOOK-AHEAD) DECILES ===\n"
                 f"  insufficient scored days ({len(A_exp)}) for a meaningful comparison")
        print()
        print(oos_report(P, prices, full_tables))
        print()

    if args.csv:
        rows = []
        for k in IDX:
            df = full_tables[k].copy()
            df["lens"] = "everyday"
            rows.append(df)
            et = R.entry_cell_table(A, metrics[k], k,
                                    sorted(set(HEADLINE_CODES) | set(A["code"].unique())))
            et["lens"] = "entry"
            rows.append(et)
        pd.concat(rows, ignore_index=True).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")

    if args.fan_csv:
        rows = []
        for k in IDX:
            codes = ["BASELINE"] + sorted(A["code"].unique())
            for code in codes:
                dates = (A.index if code == "BASELINE" else A.index[A["code"] == code])
                fan = R.fan_quantiles(prices[k], list(dates), horizon=R.H)
                for qi, q in enumerate(R.FAN_Q):
                    for day, val in enumerate(fan[qi]):
                        rows.append({"index": k, "regime": code, "q": q, "day": day,
                                    "value": round(float(val), 3) if np.isfinite(val) else None})
        pd.DataFrame(rows).to_csv(args.fan_csv, index=False)
        print(f"wrote {args.fan_csv}")


if __name__ == "__main__":
    main()
