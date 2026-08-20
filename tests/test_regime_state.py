"""Tests for the nightly regime-state build (``build_regime_state``) and the
forward-scoring log (``scripts/score_regime_log``).

The state page and the log are the study's pre-registration machinery, so
the pieces pinned here are the ones a silent bug would corrupt forever:
the current-zone age, the frozen-rule evaluation (incl. the not-evaluable
degradation when an index is missing), the tilt avoid-screen extraction
from a payload-shaped dict, the append-by-date idempotency of the state
log, and the 21-row outcome resolution from the log's own closes.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_regime_state as B

_spec = importlib.util.spec_from_file_location(
    "score_regime_log",
    Path(__file__).resolve().parent.parent / "scripts" / "score_regime_log.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def _bdays(n, start="2024-01-02"):
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# zone_age / eval_rules
# ---------------------------------------------------------------------------
def test_zone_age_counts_current_run_only():
    z = pd.Series(["Mid", "Low", "Low", "High", "Low", "Low", "Low"],
                  index=_bdays(7))
    assert B.zone_age(z) == 3
    assert B.zone_age(z.iloc[:0]) == 0


def test_eval_rules_activation_matrix():
    ndx_low = {"zone": "LowCorr", "dz_roll_l1": "DIXLow"}
    ndx_low_mid = {"zone": "LowCorr", "dz_roll_l1": "DIXMid"}
    low = {"zone": "LowCorr"}
    mid = {"zone": "MidCorr"}
    r = B.eval_rules({"NDX": ndx_low, "SPX": low, "IWM": low}, ["AAA"])
    assert r["ndx_tilt_screen_v1"]["active"] and r["ndx_tilt_screen_v1"]["names"] == ["AAA"]
    assert r["ndx_dixlow_caution_v1"]["active"]
    assert r["all_dispersed_derisk_v1"]["active"]
    r2 = B.eval_rules({"NDX": ndx_low_mid, "SPX": mid, "IWM": low}, ["AAA"])
    assert not r2["ndx_dixlow_caution_v1"]["active"]
    assert not r2["all_dispersed_derisk_v1"]["active"]
    # missing index -> the 3-index rule reports not-evaluable, screen still works
    r3 = B.eval_rules({"NDX": ndx_low}, [])
    assert not r3["all_dispersed_derisk_v1"]["evaluable"]
    # NDX not dispersed -> the screen emits no names even if some were passed
    r4 = B.eval_rules({"NDX": {"zone": "MidCorr", "dz_roll_l1": "DIXLow"},
                       "SPX": low, "IWM": low}, ["AAA"])
    assert not r4["ndx_tilt_screen_v1"]["active"]
    assert r4["ndx_tilt_screen_v1"]["names"] == []
    assert not r4["ndx_dixlow_caution_v1"]["active"]


# ---------------------------------------------------------------------------
# ndx_tilt_screen from a payload-shaped dict
# ---------------------------------------------------------------------------
def test_ndx_tilt_screen_picks_recently_hot_names():
    n, n_names = 90, 60
    idx = _bdays(n)
    names = [f"N{i:02d}" for i in range(n_names)]
    d = {t: [0.4] * n for t in names}
    for t in names[:12]:                       # these run hot in the last week
        d[t] = [0.4] * (n - 5) + [0.7] * 5
    P = {"bench": "QQQ",
         "rel": {"dates": [x.strftime("%Y-%m-%d") for x in idx],
                 "d": {**d, "QQQ": [0.4] * n}}}
    out = B.ndx_tilt_screen(P, idx[-1].strftime("%Y-%m-%d"))
    assert set(out) == set(names[:12])
    # before enough history has accrued the screen abstains
    assert B.ndx_tilt_screen(P, idx[10].strftime("%Y-%m-%d")) == []


# ---------------------------------------------------------------------------
# scoring log: append idempotency + outcome resolution
# ---------------------------------------------------------------------------
def _state_json(date, qqq=100.0, active=False):
    return {"generated": f"{date} 03:00 UTC", "rules_hash": "abc",
            "indices": {"NDX": {"asof": date, "zone": "LowCorr",
                                "dz_roll_l1": "DIXLow", "proxy_close": qqq},
                        "SPX": {"asof": date, "zone": "MidCorr",
                                "proxy_close": 2 * qqq},
                        "IWM": {"asof": date, "zone": "MidCorr",
                                "proxy_close": 3 * qqq}},
            "rules": {"ndx_tilt_screen_v1": {"active": active, "names": ["AAA"]},
                      "ndx_dixlow_caution_v1": {"active": active},
                      "all_dispersed_derisk_v1": {"active": False}}}


def test_append_state_replaces_same_date_and_sorts():
    df = pd.DataFrame(columns=score.STATE_COLS)
    df = score.append_state(df, score.state_row(_state_json("2026-01-05", 100.0)))
    df = score.append_state(df, score.state_row(_state_json("2026-01-02", 99.0)))
    assert list(df["date"]) == ["2026-01-02", "2026-01-05"]
    # re-log the same date with a newer close -> replaced, not duplicated
    df = score.append_state(df, score.state_row(_state_json("2026-01-05", 101.0)))
    assert len(df) == 2
    assert float(df[df["date"] == "2026-01-05"]["qqq_close"].iloc[0]) == 101.0


def test_resolve_outcomes_forward_math():
    dates = [d.strftime("%Y-%m-%d") for d in _bdays(30)]
    df = pd.DataFrame(columns=score.STATE_COLS)
    for i, d in enumerate(dates):
        df = score.append_state(df, score.state_row(
            _state_json(d, qqq=100.0 * (1.01 ** i), active=(i == 0))))
    out = score.resolve_outcomes(df, horizon=21)
    assert len(out) == 30 - 21
    # 21 rows of +1%/row compounds to ~+23.24%
    assert out["fwd21_qqq"].iloc[0] == pytest.approx((1.01 ** 21 - 1) * 100, abs=0.01)
    assert bool(out["ruleB_active"].iloc[0]) is True
    assert out["resolved_on"].iloc[0] == dates[21]


def test_resolve_outcomes_handles_missing_closes():
    dates = [d.strftime("%Y-%m-%d") for d in _bdays(25)]
    df = pd.DataFrame(columns=score.STATE_COLS)
    for i, d in enumerate(dates):
        sj = _state_json(d, qqq=100.0)
        if i == 21:
            sj["indices"]["NDX"]["proxy_close"] = None
        df = score.append_state(df, score.state_row(sj))
    out = score.resolve_outcomes(df, horizon=21)
    assert out["fwd21_qqq"].iloc[0] is None or pd.isna(out["fwd21_qqq"].iloc[0])
    assert out["fwd21_spy"].iloc[0] == pytest.approx(0.0)
