"""Tests for the earnings-event construction in ``earnings_dpi_study`` --
the study's core methodology.

``build_events`` turns (earnings dates + timing) plus price/DPI panels into one
row per event: it picks the timing-aware base close T, measures the
pre-earnings DPI run-in window, and the post-earnings horizon returns. Bugs
here are silent methodology errors, not crashes, so the tests drive it with
deterministic synthetic panels (adjclose = 100 + t, dpi = t/100) where every
expected number is computable, and pin down the timing rules, the DPI-window
alignment and sufficiency guard, the within-name percentile ranks, and the
skip/empty edges. ``_pos_at_or_after`` and the diagnostic
``anchor_to_reaction`` are covered directly.
"""
import numpy as np
import pandas as pd
import pytest

import earnings_dpi_study as E


IDX = pd.bdate_range("2022-01-03", periods=60)


def _panels():
    adj = pd.DataFrame({"AAA": [100.0 + t for t in range(60)]}, index=IDX)
    dpi = pd.DataFrame({"AAA": [t / 100.0 for t in range(60)]}, index=IDX)
    return {"adjclose": adj, "dpi": dpi}


def _earn(ticker, report_date, timing="amc"):
    return pd.DataFrame({"ticker": [ticker], "report_date": [report_date], "timing": [timing]})


# ===========================================================================
# _pos_at_or_after
# ===========================================================================
def test_pos_at_or_after_exact_trading_day():
    assert E._pos_at_or_after(IDX, IDX[20]) == 20


def test_pos_at_or_after_rolls_forward_from_weekend():
    # 2022-01-29 is a Saturday; the next trading day is IDX[20] (Mon 2022-01-31).
    assert IDX[20] == pd.Timestamp("2022-01-31")
    assert E._pos_at_or_after(IDX, pd.Timestamp("2022-01-29")) == 20


def test_pos_at_or_after_past_end_is_none():
    assert E._pos_at_or_after(IDX, pd.Timestamp("2030-01-01")) is None


# ===========================================================================
# build_events -- timing-aware base close T
# ===========================================================================
def test_build_events_amc_exact_uses_announce_day_as_T():
    p = _panels()
    adj = p["adjclose"]["AAA"]
    ev = E.build_events(_earn("AAA", IDX[20], "amc"), p, dpi_windows=(5, 10))
    row = ev.iloc[0]
    # amc on a real trading day -> T = A = IDX[20].
    assert row["base_T"] == IDX[20].date().isoformat()
    assert row["next_day_ret"] == pytest.approx(adj.iloc[21] / adj.iloc[20] - 1)
    assert row["w1_ret"] == pytest.approx(adj.iloc[25] / adj.iloc[20] - 1)
    assert row["w2_ret"] == pytest.approx(adj.iloc[30] / adj.iloc[20] - 1)
    assert row["m1_ret"] == pytest.approx(adj.iloc[41] / adj.iloc[20] - 1)


@pytest.mark.parametrize("timing", ["bmo", "intraday"])
def test_build_events_bmo_and_intraday_use_prior_close(timing):
    p = _panels()
    adj = p["adjclose"]["AAA"]
    ev = E.build_events(_earn("AAA", IDX[20], timing), p)
    row = ev.iloc[0]
    # Reaction is the (A-1)->A move, so T = IDX[19].
    assert row["base_T"] == IDX[19].date().isoformat()
    assert row["next_day_ret"] == pytest.approx(adj.iloc[20] / adj.iloc[19] - 1)


def test_build_events_amc_on_nontrading_date_falls_back_to_prior_close():
    # amc but the report date is a weekend, so it does not exactly match the
    # announce session -> T = A - 1 (not the announce day).
    p = _panels()
    ev = E.build_events(_earn("AAA", pd.Timestamp("2022-01-29"), "amc"), p)
    assert ev.iloc[0]["base_T"] == IDX[19].date().isoformat()


# ===========================================================================
# build_events -- pre-earnings DPI window
# ===========================================================================
def test_build_events_dpi_window_is_w_sessions_ending_before_announce():
    p = _panels()
    ev = E.build_events(_earn("AAA", IDX[20], "amc"), p, dpi_windows=(5, 10))
    row = ev.iloc[0]
    # Window is [fp-w, fp): dpi5 over positions 15..19, dpi10 over 10..19.
    assert row["dpi5"] == pytest.approx(np.mean([t / 100.0 for t in range(15, 20)]))
    assert row["dpi10"] == pytest.approx(np.mean([t / 100.0 for t in range(10, 20)]))


def test_build_events_dpi_window_stays_before_announce_regardless_of_timing():
    # The DPI cut-off is always the day before A, so bmo (T=A-1) uses the same
    # run-in window as amc.
    p = _panels()
    amc = E.build_events(_earn("AAA", IDX[20], "amc"), p, dpi_windows=(5,)).iloc[0]
    bmo = E.build_events(_earn("AAA", IDX[20], "bmo"), p, dpi_windows=(5,)).iloc[0]
    assert amc["dpi5"] == pytest.approx(bmo["dpi5"])


def test_build_events_dpi_insufficient_data_is_nan_but_row_kept():
    p = _panels()
    p["dpi"] = p["dpi"].copy()
    p["dpi"].iloc[10:20] = np.nan       # blank out the whole run-in window
    ev = E.build_events(_earn("AAA", IDX[20], "amc"), p, dpi_windows=(5, 10))
    row = ev.iloc[0]
    assert np.isnan(row["dpi5"]) and np.isnan(row["dpi10"])
    assert row["has_data"] == 1          # price-based returns still present


def test_build_events_within_name_dpi_percentile_ranks():
    p = _panels()
    earn = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA"],
                         "report_date": [IDX[15], IDX[30], IDX[45]],
                         "timing": ["amc"] * 3})
    ev = E.build_events(earn, p, dpi_windows=(5, 10))
    # Monotonically increasing dpi (ramp) -> ranks 1/3, 2/3, 1.
    assert ev["dpi5_pct"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


# ===========================================================================
# build_events -- skips and empty result (regression for the empty-frame guard)
# ===========================================================================
def test_build_events_unknown_ticker_returns_empty_frame():
    # Regression: zero valid events must return an empty DataFrame, not raise
    # KeyError from grouping a frame with no 'ticker' column.
    ev = E.build_events(_earn("ZZZ", IDX[20], "amc"), _panels())
    assert ev.empty


def test_build_events_report_before_history_is_skipped():
    ev = E.build_events(_earn("AAA", IDX[0], "amc"), _panels())
    assert ev.empty


def test_build_events_horizon_running_off_the_end_is_nan():
    p = _panels()
    ev = E.build_events(_earn("AAA", IDX[58], "amc"), p)
    row = ev.iloc[0]
    assert np.isnan(row["m1_ret"])       # T+21 runs past the panel
    assert np.isfinite(row["next_day_ret"])
    assert row["has_data"] == 1          # keyed off next-day availability


# ===========================================================================
# anchor_to_reaction  (diagnostic anchoring)
# ===========================================================================
def _ret_with_spike(pos, mag, base=0.001):
    r = pd.Series(base, index=IDX)
    r.iloc[pos] = mag
    return r


def test_anchor_snaps_to_dominant_reaction():
    # A single large move at slot 22 dominates -> T = reaction - 1 = 21.
    r = _ret_with_spike(22, 0.08)
    assert E.anchor_to_reaction(r, IDX, 20, sigma=0.01) == (21, True)


def test_anchor_falls_back_when_no_move_stands_out():
    r = pd.Series(0.001, index=IDX)
    assert E.anchor_to_reaction(r, IDX, 20, sigma=0.01) == (20, False)


def test_anchor_falls_back_without_dominance_margin():
    # Two comparable moves -> neither is 1.3x the other, so no anchoring.
    r = pd.Series(0.001, index=IDX)
    r.iloc[22] = 0.08
    r.iloc[23] = 0.075
    assert E.anchor_to_reaction(r, IDX, 20, sigma=0.01) == (20, False)


def test_anchor_falls_back_below_absolute_floor():
    # Dominant but tiny (< min_abs=0.03) move must not anchor.
    r = _ret_with_spike(22, 0.02)
    assert E.anchor_to_reaction(r, IDX, 20, sigma=0.001) == (20, False)


def test_anchor_returns_fallback_when_window_collapses():
    # approx_pos past the end collapses the search window (hi <= lo).
    r = _ret_with_spike(22, 0.08)
    assert E.anchor_to_reaction(r, IDX, 200, sigma=0.01) == (200, False)
