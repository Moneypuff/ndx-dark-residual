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

The tilt screen (rule A) is recorded as names only; its per-name outcomes
are resolved in batch by `intra_index_regime_study.py --score-log` against
the then-current payload, not here.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

STATE_COLS = ["date", "generated", "rules_hash", "ndx_zone", "spx_zone",
              "iwm_zone", "ndx_dz_l1", "ruleA_active", "ruleB_active",
              "ruleC_active", "qqq_close", "spy_close", "iwm_close", "screen"]
HORIZON = 21


def state_row(state):
    """Flatten one regime_state.json payload into a state.csv row dict."""
    idx = state.get("indices", {})
    ndx, spx, iwm = idx.get("NDX") or {}, idx.get("SPX") or {}, idx.get("IWM") or {}
    rules = state.get("rules", {})
    return {
        "date": ndx.get("asof"),
        "generated": state.get("generated"),
        "rules_hash": state.get("rules_hash"),
        "ndx_zone": ndx.get("zone"),
        "spx_zone": spx.get("zone"),
        "iwm_zone": iwm.get("zone"),
        "ndx_dz_l1": ndx.get("dz_roll_l1"),
        "ruleA_active": bool(rules.get("ndx_tilt_screen_v1", {}).get("active")),
        "ruleB_active": bool(rules.get("ndx_dixlow_caution_v1", {}).get("active")),
        "ruleC_active": bool(rules.get("all_dispersed_derisk_v1", {}).get("active")),
        "qqq_close": ndx.get("proxy_close"),
        "spy_close": spx.get("proxy_close"),
        "iwm_close": iwm.get("proxy_close"),
        "screen": ";".join(rules.get("ndx_tilt_screen_v1", {}).get("names") or []),
    }


def append_state(df, row):
    """Insert/replace `row` by its date, keep the log date-sorted."""
    if row["date"] is None:
        raise SystemExit("state json carries no NDX as-of date")
    df = df[df["date"] != row["date"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)[STATE_COLS]


def resolve_outcomes(df, horizon=HORIZON):
    """Realized forward returns over the next `horizon` LOG ROWS, from the
    log's own closes. Returns an outcomes frame (one row per resolvable
    state row)."""
    out = []
    closes = {k: pd.to_numeric(df[k], errors="coerce")
              for k in ("qqq_close", "spy_close", "iwm_close")}
    for i in range(len(df) - horizon):
        row = {"date": df["date"].iloc[i],
               "ruleA_active": df["ruleA_active"].iloc[i],
               "ruleB_active": df["ruleB_active"].iloc[i],
               "ruleC_active": df["ruleC_active"].iloc[i],
               "resolved_on": df["date"].iloc[i + horizon]}
        for k, col in (("fwd21_qqq", "qqq_close"), ("fwd21_spy", "spy_close"),
                       ("fwd21_iwm", "iwm_close")):
            a, b = closes[col].iloc[i], closes[col].iloc[i + horizon]
            row[k] = round((b / a - 1.0) * 100.0, 3) if pd.notna(a) and pd.notna(b) and a else None
        out.append(row)
    return pd.DataFrame(out)


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
    df = (pd.read_csv(state_path, dtype=str)
          if state_path.exists() else pd.DataFrame(columns=STATE_COLS))
    for c in ("ruleA_active", "ruleB_active", "ruleC_active"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).lower() == "true")
    df = append_state(df, state_row(state))
    df.to_csv(state_path, index=False)

    outcomes = resolve_outcomes(df)
    (d / "outcomes.csv").write_text(outcomes.to_csv(index=False), encoding="utf-8")
    print(f"state rows: {len(df)}   resolved outcomes: {len(outcomes)}")
    if len(outcomes) >= 42:
        for flag in ("ruleB_active", "ruleC_active"):
            on = outcomes[outcomes[flag].astype(bool)]["fwd21_qqq"].dropna().astype(float)
            off = outcomes[~outcomes[flag].astype(bool)]["fwd21_qqq"].dropna().astype(float)
            if len(on) >= 21 and len(off) >= 21:
                print(f"{flag}: QQQ fwd21 on {on.mean():+.2f}% (n={len(on)}) "
                      f"vs off {off.mean():+.2f}% (n={len(off)})")


if __name__ == "__main__":
    main()
