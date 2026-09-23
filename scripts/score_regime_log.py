#!/usr/bin/env python3
"""
Append tonight's regime state to the scoring log and resolve outcomes.
======================================================================

Runs inside .github/workflows/regime_score.yml against a checkout of the
`regime-scoring-data` branch:

  1. reads the published docs/regime_state.json (fetched by the workflow),
  2. appends one row per NDX as-of date to state.csv (idempotent: a re-run
     for the same date REPLACES the row, so a catch-up rebuild wins),
  3. recomputes outcomes.csv from scratch: every state row with a row 21
     log entries later gets its realized 21-row forward returns from the
     log's own recorded closes. Fully deterministic from state.csv, so the
     branch history stays append-only + one regenerated file.

Log rows are one per successful nightly build; a missed night stretches
that row's horizon by a day. That drift is accepted and documented -- the
log measures the rules as they would actually have been consumed.

Rule flags are three-valued. True/False are genuine readings; an EMPTY
cell (None) means the rule was not evaluable that night (an index was
missing from the build). `split_by_flag` keeps such rows out of BOTH the
on and the off group, so a degraded night can never masquerade as a
"rule off" observation.

The tilt screen (rule A) is recorded as names only; its per-name outcomes
are resolved in batch by `intra_index_regime_study.py --score-log` against
the then-current payload, not here.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

STATE_COLS = ["date", "generated", "rules_hash",
              "ndx_zone", "spx_zone", "iwm_zone", "ndx_dz_l1",
              "ndx_zone_ewm", "spx_zone_ewm", "iwm_zone_ewm",
              "ruleA_active", "ruleB_active", "ruleC_active", "ruleD_active",
              "qqq_close", "spy_close", "iwm_close", "screen"]
FLAG_COLS = ["ruleA_active", "ruleB_active", "ruleC_active", "ruleD_active"]
RULE_KEYS = {"ruleA_active": "ndx_tilt_screen_v1",
             "ruleB_active": "ndx_dixlow_caution_v1",
             "ruleC_active": "all_dispersed_derisk_v1",
             "ruleD_active": "all_dispersed_derisk_v2"}
HORIZON = 21


def _flag(v):
    """Three-valued rule reading: None stays None (not evaluable), anything
    else collapses to a bool."""
    return None if v is None else bool(v)


def _parse_flag(v):
    """CSV cell -> True / False / None ('true'/'false' in any case; an empty
    or NaN cell is a not-evaluable night)."""
    t = str(v).strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    return None


def state_row(state):
    """Flatten one regime_state.json payload into a state.csv row dict."""
    idx = state.get("indices", {})
    ndx, spx, iwm = idx.get("NDX") or {}, idx.get("SPX") or {}, idx.get("IWM") or {}
    rules = state.get("rules", {})
    row = {
        "date": ndx.get("asof"),
        "generated": state.get("generated"),
        "rules_hash": state.get("rules_hash"),
        "ndx_zone": ndx.get("zone"),
        "spx_zone": spx.get("zone"),
        "iwm_zone": iwm.get("zone"),
        "ndx_dz_l1": ndx.get("dz_roll_l1"),
        "ndx_zone_ewm": ndx.get("zone_ewm"),
        "spx_zone_ewm": spx.get("zone_ewm"),
        "iwm_zone_ewm": iwm.get("zone_ewm"),
        "qqq_close": ndx.get("proxy_close"),
        "spy_close": spx.get("proxy_close"),
        "iwm_close": iwm.get("proxy_close"),
        "screen": ";".join(rules.get("ndx_tilt_screen_v1", {}).get("names") or []),
    }
    for col, key in RULE_KEYS.items():
        row[col] = _flag(rules.get(key, {}).get("active"))
    return row


def read_state_log(path):
    """state.csv -> frame with STATE_COLS (missing columns added empty, so
    a log written by an older schema still loads) and three-valued flags."""
    df = pd.read_csv(path, dtype=str)
    for c in STATE_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[STATE_COLS].astype(object)
    for c in FLAG_COLS:
        df[c] = [_parse_flag(v) for v in df[c]]
    return df


def append_state(df, row):
    """Insert/replace `row` by its date, keep the log date-sorted."""
    if row["date"] is None:
        raise SystemExit("state json carries no NDX as-of date")
    df = df[df["date"] != row["date"]]
    new = pd.DataFrame([row]).astype(object)
    df = new if df.empty else pd.concat([df.astype(object), new], ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)[STATE_COLS]


def resolve_outcomes(df, horizon=HORIZON):
    """Realized forward returns over the next `horizon` LOG ROWS, from the
    log's own closes. Returns an outcomes frame (one row per resolvable
    state row); the rule flags are carried through three-valued."""
    out = []
    closes = {k: pd.to_numeric(df[k], errors="coerce")
              for k in ("qqq_close", "spy_close", "iwm_close")}
    for i in range(len(df) - horizon):
        row = {"date": df["date"].iloc[i]}
        for c in FLAG_COLS:
            row[c] = _flag(df[c].iloc[i]) if c in df.columns else None
        row["resolved_on"] = df["date"].iloc[i + horizon]
        for k, col in (("fwd21_qqq", "qqq_close"), ("fwd21_spy", "spy_close"),
                       ("fwd21_iwm", "iwm_close")):
            a, b = closes[col].iloc[i], closes[col].iloc[i + horizon]
            row[k] = round((b / a - 1.0) * 100.0, 3) if pd.notna(a) and pd.notna(b) and a else None
        out.append(row)
    return pd.DataFrame(out, columns=["date", *FLAG_COLS, "resolved_on",
                                      "fwd21_qqq", "fwd21_spy", "fwd21_iwm"])


def split_by_flag(outcomes, flag):
    """(on, off) outcome rows for one rule flag. A not-evaluable row (None)
    belongs to neither group."""
    vals = outcomes[flag].map(_flag) if len(outcomes) else outcomes[flag]
    on = outcomes[[v is True for v in vals]]
    off = outcomes[[v is False for v in vals]]
    return on, off


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-json", required=True,
                    help="the fetched published docs/regime_state.json")
    ap.add_argument("--dir", required=True,
                    help="checkout of the regime-scoring-data branch")
    args = ap.parse_args()

    state = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    state_path = d / "state.csv"
    df = (read_state_log(state_path) if state_path.exists()
          else pd.DataFrame(columns=STATE_COLS))
    df = append_state(df, state_row(state))
    df.to_csv(state_path, index=False)

    outcomes = resolve_outcomes(df)
    (d / "outcomes.csv").write_text(outcomes.to_csv(index=False), encoding="utf-8")
    print(f"state rows: {len(df)}   resolved outcomes: {len(outcomes)}")
    if len(outcomes) >= 42:
        for flag in ("ruleB_active", "ruleC_active", "ruleD_active"):
            on, off = split_by_flag(outcomes, flag)
            on = on["fwd21_qqq"].dropna().astype(float)
            off = off["fwd21_qqq"].dropna().astype(float)
            if len(on) >= 21 and len(off) >= 21:
                print(f"{flag} ({RULE_KEYS[flag]}): QQQ fwd21 on {on.mean():+.2f}% "
                      f"(n={len(on)}) vs off {off.mean():+.2f}% (n={len(off)})")


if __name__ == "__main__":
    main()
