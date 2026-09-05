"""Tests for the pure pieces of ``regime_path_study`` (Phase 1: the path
engine, ENV-family cell stats, fan/excursion/barrier/bracket blocks).

All synthetic; no payload or network required, matching the style of
``tests/test_intra_index_regime.py``.
"""
import numpy as np
import pandas as pd
import pytest

import regime_path_study as S


def _bdays(n, start="2022-01-03"):
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# forward_path_panel
# ---------------------------------------------------------------------------
def test_forward_path_panel_values_and_tail_nan():
    idx = _bdays(10)
    close = pd.Series([100, 102, 101, 105, 110, 108, 107, 112, 115, 120.0], index=idx)
    P = S.forward_path_panel(close, horizon=5)
    assert P.loc[idx[0], 0] == pytest.approx(0.0)
    assert P.loc[idx[0], 1] == pytest.approx((102 / 100 - 1) * 100)
    assert P.loc[idx[0], 5] == pytest.approx((108 / 100 - 1) * 100)
    # anchor near the end: the horizon runs past the data -> NaN
    assert np.isnan(P.loc[idx[8], 5])
    assert P.loc[idx[8], 1] == pytest.approx((120 / 115 - 1) * 100)


def test_forward_path_panel_column_zero_is_zero():
    idx = _bdays(6)
    close = pd.Series([50.0, 51, 49, 52, 53, 54], index=idx)
    P = S.forward_path_panel(close, horizon=3)
    assert (P[0] == 0.0).all()


# ---------------------------------------------------------------------------
# daily_rets_from_path
# ---------------------------------------------------------------------------
def test_daily_rets_from_path_matches_direct_pct_change():
    idx = _bdays(8)
    close = pd.Series([100, 103, 99, 101, 104, 108, 107, 110.0], index=idx)
    P = S.forward_path_panel(close, horizon=4)
    sub = P.loc[[idx[0]]]
    rets = S.daily_rets_from_path(sub, 4)
    expected = (close.pct_change() * 100).iloc[1:5].to_numpy()
    np.testing.assert_allclose(rets[0], expected)


def test_daily_rets_from_path_none_when_columns_missing():
    sub = pd.DataFrame({0: [0.0], 1: [1.0]})
    assert S.daily_rets_from_path(sub, 4) is None


# ---------------------------------------------------------------------------
# excursion_stats
# ---------------------------------------------------------------------------
def test_excursion_stats_trough_and_peak_day():
    row = {0: 0.0, 1: -2.0, 2: -5.0, 3: -8.0, 4: -4.0, 5: 6.0, 6: 3.0, 7: 1.0,
          8: 0.5, 9: 1.5, 10: 2.0}
    sub = pd.DataFrame([row])
    st = S.excursion_stats(sub, 10)
    assert st["n"] == 1
    assert st["mae_med"] == pytest.approx(-8.0)
    assert st["trough_d"] == pytest.approx(3.0)
    assert st["mfe_med"] == pytest.approx(6.0)
    assert st["peak_d"] == pytest.approx(5.0)
    assert st["giveback_med"] == pytest.approx(6.0 - 2.0)
    assert st["dip5"] == pytest.approx(100.0)
    assert st["dip12"] == pytest.approx(0.0)


def test_excursion_stats_requires_completeness_to_h():
    row_ok = {c: float(c) for c in range(6)}
    row_incomplete = {c: (float(c) if c < 4 else np.nan) for c in range(6)}
    sub = pd.DataFrame([row_ok, row_incomplete])
    st = S.excursion_stats(sub, 5)
    assert st["n"] == 1


def test_excursion_stats_empty_when_no_columns():
    sub = pd.DataFrame({0: [0.0], 1: [1.0]})
    assert S.excursion_stats(sub, 5) == {"n": 0}


# ---------------------------------------------------------------------------
# barrier_touch
# ---------------------------------------------------------------------------
def test_barrier_touch_known_case():
    base = {c: 0.0 for c in range(6)}
    row0 = dict(base); row0[2] = -6.0; row0[4] = 9.0
    row1 = dict(base); row1[3] = -2.0
    sub = pd.DataFrame([row0, row1])
    bar = S.barrier_touch(sub, 5, levels=(5, 8))
    assert bar["n"] == 2
    assert bar["touch_m5"] == pytest.approx(50.0)
    assert bar["touch_p8"] == pytest.approx(50.0)
    assert bar["touch_m8"] == pytest.approx(0.0)


def test_barrier_touch_excludes_incomplete_rows():
    base = {c: 0.0 for c in range(6)}
    row_ok = dict(base); row_ok[2] = -6.0
    row_bad = dict(base); row_bad[5] = np.nan
    sub = pd.DataFrame([row_ok, row_bad])
    bar = S.barrier_touch(sub, 5, levels=(5,))
    assert bar["n"] == 1
    assert bar["touch_m5"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# bracket_outcomes
# ---------------------------------------------------------------------------
def test_bracket_outcomes_first_passage_and_neither():
    r0 = {0: 0.0, 1: 1.0, 2: 5.5, 3: 3.0, 4: -6.0}     # target (+5) hits day2, before stop day4
    r1 = {0: 0.0, 1: -6.0, 2: -6.0, 3: -6.0, 4: -6.0}  # stop (-5) hits day1, target never
    r2 = {0: 0.0, 1: 1.0, 2: 2.0, 3: 1.0, 4: 0.5}      # neither hit; terminal +0.5
    sub = pd.DataFrame([r0, r1, r2])
    out = S.bracket_outcomes(sub, 4, stops=(5,), targets=(5,))
    b = out["brackets"][(5, 5)]
    assert out["n"] == 3
    assert b["p_target"] == pytest.approx(100 / 3)
    assert b["p_stop"] == pytest.approx(100 / 3)
    assert b["p_neither"] == pytest.approx(100 / 3)
    assert b["e_ret"] == pytest.approx((5.0 + (-5.0) + 0.5) / 3)


def test_bracket_outcomes_empty_without_columns():
    sub = pd.DataFrame({0: [0.0]})
    out = S.bracket_outcomes(sub, 4)
    assert out["n"] == 0 and out["brackets"] == {}


# ---------------------------------------------------------------------------
# fan_quantiles
# ---------------------------------------------------------------------------
def test_fan_quantiles_median_mean_and_outlier():
    idx = _bdays(6)
    vals = [1.0, 2.0, 3.0, 100.0, 4.0, 5.0]
    sub = pd.DataFrame({0: [0.0] * 6, 5: vals}, index=idx)
    fan = S.fan_quantiles(sub, checkpoints=(5,), qs=(0.5,))
    row = fan[(fan["h"] == 5) & (fan["q"] == 0.5)]
    assert row["n"].iloc[0] == 6
    assert row["value"].iloc[0] == pytest.approx(float(np.median(vals)))
    mean_row = fan[(fan["h"] == 5) & (fan["q"] == "mean")]
    assert mean_row["value"].iloc[0] == pytest.approx(float(np.mean(vals)))
    hit_row = fan[(fan["h"] == 5) & (fan["q"] == "hit")]
    assert hit_row["value"].iloc[0] == pytest.approx(100.0)  # all 6 values > 0


def test_fan_quantiles_skips_missing_checkpoint_column():
    sub = pd.DataFrame({0: [0.0, 0.0]})
    fan = S.fan_quantiles(sub, checkpoints=(63,), qs=(0.5,))
    assert fan.empty


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------
def test_sizing_matches_playbook_convention():
    size_q25, size_p10, rr = S.sizing(mae_q25=-6.1, mae_p10=-9.9, med_term=5.5)
    assert size_q25 == pytest.approx(100.0 / 6.1)
    assert size_p10 == pytest.approx(100.0 / 9.9)
    assert rr == pytest.approx(5.5 / 6.1)


def test_sizing_nan_on_zero_mae():
    size_q25, size_p10, rr = S.sizing(mae_q25=0.0, mae_p10=-4.0, med_term=1.0)
    assert np.isnan(size_q25)
    assert np.isnan(rr)
    assert np.isfinite(size_p10)


# ---------------------------------------------------------------------------
# cell_masks
# ---------------------------------------------------------------------------
def test_cell_masks_marginal_and_grid_labels():
    idx = _bdays(10)
    M = pd.DataFrame({
        "cz_roll": ["LowCorr"] * 5 + ["HighCorr"] * 5,
        "dz_roll_l1": ["DIXLow"] * 3 + ["DIXHigh"] * 7,
    }, index=idx)
    masks = S.cell_masks(M)
    assert masks["LowCorr"].sum() == 5
    assert masks["HighCorr"].sum() == 5
    assert masks["LowCorrxDIXLow(l1)"].sum() == 3
    assert masks["HighCorrxDIXHigh(l1)"].sum() == 5
    assert "vz_roll" not in "".join(masks.keys())  # no vol-parallel cells without the column


def test_cell_masks_includes_vol_parallel_when_present():
    idx = _bdays(4)
    M = pd.DataFrame({
        "cz_roll": ["LowCorr"] * 4,
        "dz_roll_l1": ["DIXLow"] * 4,
        "vz_roll": ["VolLow"] * 4,
    }, index=idx)
    masks = S.cell_masks(M)
    assert "VolLowxDIXLow(l1)" in masks


# ---------------------------------------------------------------------------
# cell_stats (integration of the blocks above under gating)
# ---------------------------------------------------------------------------
def test_cell_stats_gate_and_basic_blocks():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(20.0, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):     # 6 episodes x 10 days = 60 days, 6 episodes
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=7)
    assert st["n_days"] == 60
    assert st["n_eps"] == 6
    assert st["gate"] is True
    assert st["exc"]["n"] > 0
    assert np.isfinite(st["exc"]["mae_med"])
    assert np.isfinite(st["vol"]["vr21"])
    assert np.isfinite(st["vol"]["rv21_med"])
    assert np.isfinite(st["size"]["size_q25"])
    lo, hi = st["ci"]["mean_r"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    lo2, hi2 = st["ci"]["mae_q25"]
    assert np.isfinite(lo2) and np.isfinite(hi2)


def test_cell_stats_below_gate_reports_counts_only():
    n = 200
    idx = _bdays(n)
    close = pd.Series(100.0, index=idx)
    rv = pd.Series(15.0, index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[10:30] = True   # 20 days, 1 episode -- below both gate thresholds
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask)
    assert st["n_days"] == 20
    assert st["n_eps"] == 1
    assert st["gate"] is False
    assert st["ci"] == {}


def test_cell_stats_empty_mask():
    n = 50
    idx = _bdays(n)
    close = pd.Series(100.0, index=idx)
    rv = pd.Series(15.0, index=idx)
    mask = pd.Series(False, index=idx)
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask)
    assert st["n_days"] == 0
    assert st["n_eps"] == 0
    assert st["gate"] is False


# ---------------------------------------------------------------------------
# render_cell (gate suppression)
# ---------------------------------------------------------------------------
def test_render_cell_suppresses_detail_below_gate():
    st = {"n_days": 20, "n_eps": 1, "gate": False}
    out = S.render_cell("NDX LowCorr", st, "QQQ")
    assert "below gate" in out
    assert "excursion" not in out


# ---------------------------------------------------------------------------
# episode_hold_paths / hold_stats (Phase 2)
# ---------------------------------------------------------------------------
def test_episode_hold_paths_confirm_exit_and_post_exit():
    n = 60
    idx = _bdays(n)
    close = pd.Series(100.0 + np.arange(n), index=idx)   # strictly increasing, easy to hand-check
    mask = pd.Series(False, index=idx)
    mask.iloc[5:10] = True   # episode: positions 5..9 (5 days)
    P = S.forward_path_panel(close, horizon=S.H)
    holds = S.episode_hold_paths(P, mask, confirm=3, cap=S.H)
    assert len(holds) == 1
    row = holds.iloc[0]
    assert row["start"] == idx[7]      # 3rd consecutive True day (positions 5, 6, 7)
    assert row["exit"] == idx[10]      # first failing close after the run
    assert row["duration"] == 3
    expected_term = float((close.iloc[10] / close.iloc[7] - 1) * 100)
    assert row["terminal"] == pytest.approx(expected_term)
    expected_mae = float((close.iloc[8] / close.iloc[7] - 1) * 100)  # smallest of the 3 forward gains
    assert row["mae"] == pytest.approx(expected_mae)
    assert row["trough_d"] == 1
    assert row["peak_d"] == 3
    expected_post = float((close.iloc[10 + S.HOLD] / close.iloc[10] - 1) * 100)
    assert row["post_exit_r21"] == pytest.approx(expected_post)


def test_episode_hold_paths_skips_short_and_open_episodes():
    n = 30
    idx = _bdays(n)
    close = pd.Series(100.0 + np.arange(n), index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[3:5] = True    # only 2 days -- never confirms at confirm=3
    mask.iloc[20:30] = True  # runs through the end of the sample -- open episode
    P = S.forward_path_panel(close, horizon=S.H)
    holds = S.episode_hold_paths(P, mask, confirm=3, cap=S.H)
    assert holds.empty


def test_hold_stats_gate_and_ci():
    n = 500
    idx = _bdays(n)
    close = pd.Series(100.0 + np.arange(n) * 0.3, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (5, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 8] = True   # 6 episodes of 8 days -> 6 EPISODE-HOLD rows
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.hold_stats(P, mask, seed=3)
    assert st["n_eps"] == 6
    assert st["gate"] is True
    assert np.isfinite(st["term_med"])
    lo, hi = st["ci_term"]
    assert np.isfinite(lo) and np.isfinite(hi)


def test_hold_stats_below_gate():
    n = 100
    idx = _bdays(n)
    close = pd.Series(100.0 + np.arange(n), index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[5:13] = True   # 1 episode only
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.hold_stats(P, mask)
    assert st["n_eps"] == 1
    assert st["gate"] is False
    assert "ci_term" not in st


# ---------------------------------------------------------------------------
# entry_stats (Phase 2)
# ---------------------------------------------------------------------------
def test_entry_stats_gate_and_fields():
    n = 500
    idx = _bdays(n)
    rng = np.random.default_rng(9)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    M = pd.DataFrame(index=idx)
    mask = pd.Series(False, index=idx)
    starts = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340]
    for s in starts:
        mask.iloc[s:s + 5] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.entry_stats(M, P, mask, seed=1)
    assert st["n_events"] == len(starts)
    assert st["gate"] is True
    assert np.isfinite(st["mean_r"])
    assert np.isfinite(st["hit"])
    lo, hi = st["ci_mean_r"]
    assert np.isfinite(lo) and np.isfinite(hi)


def test_entry_stats_below_gate():
    n = 200
    idx = _bdays(n)
    close = pd.Series(100.0, index=idx)
    M = pd.DataFrame(index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[10:15] = True   # only 1 entry
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.entry_stats(M, P, mask)
    assert st["n_events"] == 1
    assert st["gate"] is False
    assert "ci_mean_r" not in st


# ---------------------------------------------------------------------------
# survival / transition_within (Phase 2)
# ---------------------------------------------------------------------------
def test_survival_known_share():
    idx = _bdays(20)
    mask = pd.Series(False, index=idx)
    mask.iloc[0:10] = True
    surv = S.survival(mask, checkpoints=(5,))
    # anchors 0..9; future = anchor+5; anchors 0..4 -> future 5..9 (all True);
    # anchors 5..9 -> future 10..14 (all False) -> 5 of 10 anchors survive
    assert surv[5] == pytest.approx(50.0)


def test_survival_nan_when_no_valid_future():
    idx = _bdays(10)
    mask = pd.Series(False, index=idx)
    mask.iloc[8:10] = True
    surv = S.survival(mask, checkpoints=(21,))
    assert np.isnan(surv[21])


def test_transition_within_hits_and_valid():
    idx = _bdays(30)
    M = pd.DataFrame({"cz_roll": ["LowCorr"] * 30}, index=idx)
    M.loc[idx[15], "cz_roll"] = "HighCorr"
    mask = pd.Series(False, index=idx)
    mask.iloc[10:14] = True
    hit, valid = S.transition_within(M, mask, h=5, to="HighCorr")
    assert bool(hit.loc[idx[10]]) is True    # window 11..15 includes position 15
    assert bool(hit.loc[idx[13]]) is True    # window 14..18 includes position 15
    assert bool(valid.loc[idx[10]]) is True


def test_transition_within_invalid_near_end():
    idx = _bdays(10)
    M = pd.DataFrame({"cz_roll": ["LowCorr"] * 10}, index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[8] = True   # near the end -- t+h runs past the frame
    hit, valid = S.transition_within(M, mask, h=5, to="HighCorr")
    assert bool(valid.loc[idx[8]]) is False


# ---------------------------------------------------------------------------
# render_entry / render_hold gate suppression
# ---------------------------------------------------------------------------
def test_render_entry_and_hold_suppress_below_gate():
    assert "below" in S.render_entry({"n_events": 3, "gate": False}, "QQQ")
    assert "below" in S.render_hold({"n_eps": 2, "gate": False})


def test_render_cell_prints_blocks_above_gate():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(11)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(18.0, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=5)
    out = S.render_cell("NDX LowCorr", st, "QQQ")
    assert "excursion" in out
    assert "vol:" in out
    assert "sizing:" in out
    assert "epCI" in out
