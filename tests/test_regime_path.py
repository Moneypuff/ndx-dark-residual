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


# ---------------------------------------------------------------------------
# daily_microstructure (Phase 3, block E)
# ---------------------------------------------------------------------------
def test_daily_microstructure_basic_stats():
    idx = _bdays(10)
    proxy = pd.Series([100.0, 102, 101, 105, 103, 106, 104, 108, 107, 110], index=idx)
    mask = pd.Series(True, index=idx)
    micro = S.daily_microstructure(proxy, mask)
    r = (proxy.pct_change() * 100).dropna().to_numpy()
    assert micro["n"] == len(r)
    assert micro["worst"] == pytest.approx(float(r.min()))
    assert micro["best"] == pytest.approx(float(r.max()))
    assert micro["up_frac"] == pytest.approx(float(np.mean(r > 0) * 100))


def test_daily_microstructure_below_min_n():
    idx = _bdays(4)
    proxy = pd.Series([100.0, 101, 99, 102], index=idx)
    mask = pd.Series(True, index=idx)
    micro = S.daily_microstructure(proxy, mask)
    assert micro["n"] < 5
    assert "ann_vol" not in micro


def test_daily_microstructure_positive_autocorrelation():
    # alternating +1%/-1% BLOCKS of 10 days each: mostly same-sign neighbors
    idx = _bdays(50)
    trend = np.concatenate([np.full(10, 1.0), np.full(10, -1.0)] * 3)[:49] / 100.0
    proxy = pd.Series(100.0 * np.cumprod(np.concatenate([[1.0], 1 + trend])), index=idx)
    mask = pd.Series(True, index=idx)
    micro = S.daily_microstructure(proxy, mask)
    assert micro["autocorr1"] > 0.3


# ---------------------------------------------------------------------------
# vol_parallel_comparison (Phase 3)
# ---------------------------------------------------------------------------
def test_vol_parallel_comparison_empty_without_vz_roll():
    idx = _bdays(10)
    M = pd.DataFrame({"cz_roll": ["LowCorr"] * 10, "dz_roll_l1": ["DIXLow"] * 10}, index=idx)
    P = pd.DataFrame({0: [0.0] * 10})
    assert S.vol_parallel_comparison(P, M) == []


def test_vol_parallel_comparison_renders_when_vz_roll_present():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(1)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(15.0, index=idx)
    cz = np.where(np.arange(n) % 20 < 10, "LowCorr", "MidCorr")
    vz = np.where(np.arange(n) % 15 < 7, "VolLow", "VolMid")
    dz = np.where(np.arange(n) % 8 < 4, "DIXLow", "DIXMid")
    M = pd.DataFrame({"cz_roll": cz, "dz_roll_l1": dz, "vz_roll": vz, "rv": rv}, index=idx)
    P = S.forward_path_panel(close, horizon=S.H)
    lines = S.vol_parallel_comparison(P, M, seed=1)
    assert any("DIX(l1)=DIXLow" in l for l in lines)


# ---------------------------------------------------------------------------
# primaries_report (P1, P3) -- decision logic on synthetic cell_stats dicts
# ---------------------------------------------------------------------------
def _fake_frame_for_primaries(n=600):
    """Minimal M frame (cz_roll, r1m, mae21) spanning several years, for
    primaries_report's era_split/year_mix side-lines -- the decision logic
    itself is exercised via the hand-built all_stats dicts."""
    idx = _bdays(n, start="2019-01-02")
    rng = np.random.default_rng(42)
    cz = np.random.default_rng(1).choice(["LowCorr", "MidCorr", "HighCorr"], size=n)
    return pd.DataFrame({"cz_roll": cz, "r1m": rng.normal(1.0, 3.0, n),
                         "mae21": -np.abs(rng.normal(4.0, 2.0, n))}, index=idx)


def test_primaries_report_p1_supported():
    M = _fake_frame_for_primaries()
    low = {"gate": True, "vol": {"z_gt2": 10.0}, "ci": {"z_gt2": (6.0, 14.0)}}
    high = {"gate": True, "vol": {"z_gt2": 3.0}, "ci": {}}
    out = S.primaries_report("NDX", M, {"LowCorr": low, "HighCorr": high})
    assert "-> SUPPORTED" in out


def test_primaries_report_p1_not_supported_when_ci_touches_benchmark():
    M = _fake_frame_for_primaries()
    low = {"gate": True, "vol": {"z_gt2": 5.0}, "ci": {"z_gt2": (3.0, 7.0)}}  # CI spans 4.6
    high = {"gate": True, "vol": {"z_gt2": 3.0}, "ci": {}}
    out = S.primaries_report("NDX", M, {"LowCorr": low, "HighCorr": high})
    assert "-> NOT SUPPORTED" in out


def test_primaries_report_p1_missing_when_below_gate():
    M = _fake_frame_for_primaries()
    out = S.primaries_report("NDX", M, {"LowCorr": {"gate": False}, "HighCorr": {"gate": True}})
    assert "P1 (jump risk): --" in out


def test_primaries_report_p3_both_legs_supported():
    M = _fake_frame_for_primaries()
    low = {"gate": True, "exc": {"n": 100, "mae_q25": -4.0}}
    mid = {"gate": True, "exc": {"n": 100, "mae_q25": -5.0}}
    high = {"gate": True, "exc": {"n": 100, "mae_q25": -9.0},
           "vol": {"vratio_med": 0.8}, "ci": {"vratio": (0.6, 0.9)}}
    out = S.primaries_report("NDX", M, {"LowCorr": low, "MidCorr": mid, "HighCorr": high})
    assert "SUPPORTED (both legs)" in out


def test_primaries_report_p3_mae_leg_only():
    M = _fake_frame_for_primaries()
    low = {"gate": True, "exc": {"n": 100, "mae_q25": -4.0}}
    mid = {"gate": True, "exc": {"n": 100, "mae_q25": -5.0}}
    high = {"gate": True, "exc": {"n": 100, "mae_q25": -9.0},
           "vol": {"vratio_med": 1.1}, "ci": {"vratio": (0.9, 1.3)}}
    out = S.primaries_report("NDX", M, {"LowCorr": low, "MidCorr": mid, "HighCorr": high})
    assert "MAE leg only" in out


def test_primaries_report_p3_not_supported_when_highcorr_not_worst():
    M = _fake_frame_for_primaries()
    low = {"gate": True, "exc": {"n": 100, "mae_q25": -9.0}}
    mid = {"gate": True, "exc": {"n": 100, "mae_q25": -5.0}}
    high = {"gate": True, "exc": {"n": 100, "mae_q25": -4.0},
           "vol": {"vratio_med": 0.8}, "ci": {"vratio": (0.6, 0.9)}}
    out = S.primaries_report("NDX", M, {"LowCorr": low, "MidCorr": mid, "HighCorr": high})
    assert "NOT SUPPORTED" in out


# ---------------------------------------------------------------------------
# Block H helpers: era_split, per_year_median, loyo_range_stat
# ---------------------------------------------------------------------------
def test_era_split_splits_on_oos_cutoff():
    idx = _bdays(20, start="2023-12-15")   # straddles the 2024-01-01 cutoff
    M = pd.DataFrame({"r1m": np.arange(20, dtype=float), "mae21": -np.arange(20, dtype=float) - 1},
                     index=idx)
    mask = pd.Series(True, index=idx)
    pre, post = S.era_split(M, mask, split="2024-01-01")
    expect_pre = int((idx < pd.Timestamp("2024-01-01")).sum())
    assert pre["n"] == expect_pre
    assert post["n"] == 20 - expect_pre
    assert 0 < expect_pre < 20   # sanity: the window really does straddle the cutoff


def test_per_year_median_gates_thin_years():
    idx = _bdays(5, start="2023-01-02").append(_bdays(15, start="2024-01-02"))
    vals = list(range(20))
    M = pd.DataFrame({"col": vals}, index=idx)
    mask = pd.Series(True, index=idx)
    out = S.per_year_median(M, mask, "col", min_days=10)
    assert "23:." in out    # only 5 obs in 2023 -- below min_days
    assert "24:" in out and "." not in out.split("24:")[1][:1]


def test_loyo_range_stat_q25():
    idx = _bdays(750, start="2020-01-02")   # spans 2020, 2021, 2022
    years = idx.year.to_numpy()
    vals = np.where(years == 2020, -10.0, np.where(years == 2021, -2.0, -5.0))
    M = pd.DataFrame({"col": vals.astype(float)}, index=idx)
    mask = pd.Series(True, index=idx)
    lo, hi = S.loyo_range_stat(M, mask, "col", lambda x: float(np.percentile(x, 25)))
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


# ---------------------------------------------------------------------------
# cluster_boot_ci_diff / mae_ids_for_diff / p2_report
# ---------------------------------------------------------------------------
def test_cluster_boot_ci_diff_detects_separation():
    rng = np.random.default_rng(5)
    idsA = np.repeat(np.arange(10), 5)
    idsB = np.repeat(np.arange(10), 5)
    a = rng.normal(-8.0, 0.5, 50)   # clearly more negative
    b = rng.normal(-2.0, 0.5, 50)
    q25 = lambda x: float(np.percentile(x, 25))
    lo, hi = S.cluster_boot_ci_diff(a, idsA, b, idsB, stat=q25, seed=1)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi < 0    # a's q25 is reliably below b's


def test_cluster_boot_ci_diff_gates_on_episode_count():
    idsA = np.array([0, 0, 1, 1])   # only 2 episodes -- below GATE_EPISODES
    idsB = np.repeat(np.arange(10), 5)
    lo, hi = S.cluster_boot_ci_diff(np.zeros(4), idsA, np.zeros(50), idsB)
    assert np.isnan(lo) and np.isnan(hi)


def test_mae_ids_for_diff_matches_excursion_stats():
    n = 200
    idx = _bdays(n)
    rng = np.random.default_rng(2)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[20:60] = True
    P = S.forward_path_panel(close, horizon=S.H)
    mins, ids = S.mae_ids_for_diff(P, mask)
    sub = P.reindex(mask.index[mask.to_numpy(dtype=bool)])
    exc = S.excursion_stats(sub, S.HOLD)
    assert len(mins) == exc["n"]
    np.testing.assert_allclose(np.sort(mins), np.sort(exc["_mins"]))


def test_p2_report_below_gate_when_legs_thin():
    n = 100
    idx = _bdays(n)
    close = pd.Series(100.0, index=idx)
    M = pd.DataFrame({"cz_roll": ["LowCorr"] * n, "dz_roll_l1": ["DIXLow"] * n}, index=idx)
    P = S.forward_path_panel(close, horizon=S.H)
    out = S.p2_report(M, P)
    assert "below gate" in out


# ---------------------------------------------------------------------------
# report_index integration smoke test
# ---------------------------------------------------------------------------
def test_report_index_smoke_includes_primaries_and_microstructure(capsys):
    import intra_index_regime_study as R

    rng = np.random.default_rng(0)
    n = 1000
    idx = _bdays(n, start="2020-01-02")
    names = [f"N{i:02d}" for i in range(30)]
    px = pd.DataFrame({t: 100.0 * np.cumprod(1 + rng.normal(0, 0.015, n)) for t in names},
                      index=idx)
    proxy = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n)), index=idx)
    dix = pd.Series(rng.normal(0.42, 0.05, n), index=idx)
    r1m = (proxy.shift(-21) / proxy - 1) * 100
    M = R.assemble_frame(px, proxy, dix, r1m)
    meta = {"proxy": "QQQ", "proxy_close": proxy, "note": "smoke", "dropped": {}}
    S.report_index("SYN", M, meta, type("A", (), {})())
    out = capsys.readouterr().out
    assert "PRIMARY HYPOTHESES" in out
    assert "microstructure" in out


# ---------------------------------------------------------------------------
# load_cboe_vol (Phase 4, implied-vol leg)
# ---------------------------------------------------------------------------
def test_load_cboe_vol_unknown_index_returns_empty():
    assert S.load_cboe_vol("XYZ").empty


def test_load_cboe_vol_skips_cleanly_when_fetch_fails(monkeypatch):
    import build_gex_dispersion as G
    monkeypatch.setattr(G, "fetch_text_cached", lambda *a, **k: None)
    out = S.load_cboe_vol("NDX", cache_dir=None)
    assert out.empty


def test_load_cboe_vol_parses_fetched_csv(monkeypatch):
    import build_gex_dispersion as G
    csv_text = "DATE,OPEN,HIGH,LOW,CLOSE\n01/03/2022,18.0,19.0,17.5,18.5\n01/04/2022,17.0,18.0,16.5,17.2\n"
    monkeypatch.setattr(G, "fetch_text_cached", lambda *a, **k: csv_text)
    out = S.load_cboe_vol("NDX", cache_dir=None)
    assert len(out) == 2
    assert out.iloc[0] == pytest.approx(18.5)


# ---------------------------------------------------------------------------
# cell_stats implied-vs-realized leg (Phase 4)
# ---------------------------------------------------------------------------
def test_cell_stats_implied_leg_populates_ivrp_and_ci():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(4)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(20.0, index=idx)
    implied = pd.Series(22.0, index=idx)   # implied consistently a few points above realized
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=2, implied=implied)
    assert np.isfinite(st["vol"]["ivrp_med"])
    assert st["vol"]["ivrp_med"] > 0     # implied ran above realized by construction
    lo, hi = st["ci"]["ivrp"]
    assert np.isfinite(lo) and np.isfinite(hi)


def test_cell_stats_without_implied_leaves_ivrp_nan():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(4)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(20.0, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=2)
    assert np.isnan(st["vol"]["ivrp_med"])
    assert "ivrp" not in st["ci"]


def test_render_cell_shows_implied_line_when_present():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(6)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(20.0, index=idx)
    implied = pd.Series(23.0, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=2, implied=implied)
    out = S.render_cell("NDX LowCorr", st, "QQQ")
    assert "implied-realized" in out


# ---------------------------------------------------------------------------
# n_dispersed / cross_index_masks / cross_index_section (Phase 5)
# ---------------------------------------------------------------------------
def _cz_frame(idx, pattern):
    return pd.DataFrame({"cz_roll": pattern}, index=idx)


def test_n_dispersed_counts_and_common_dates():
    idx = _bdays(10)
    ndx = _cz_frame(idx, ["LowCorr"] * 6 + ["HighCorr"] * 4)
    spx = _cz_frame(idx, ["LowCorr"] * 4 + ["MidCorr"] * 6)
    iwm = _cz_frame(idx, ["LowCorr"] * 8 + ["HighCorr"] * 2)
    nlow, cz = S.n_dispersed({"NDX": ndx, "SPX": spx, "IWM": iwm})
    assert len(cz) == 10
    assert nlow.iloc[0] == 3    # all three LowCorr on day 0
    # day 9: ndx=HighCorr, spx=MidCorr, iwm=HighCorr -> none LowCorr
    assert nlow.iloc[9] == 0


def test_n_dispersed_drops_na_rows():
    idx = _bdays(5)
    ndx = _cz_frame(idx, ["LowCorr", "NA", "HighCorr", "LowCorr", "LowCorr"])
    spx = _cz_frame(idx, ["LowCorr"] * 5)
    nlow, cz = S.n_dispersed({"NDX": ndx, "SPX": spx})
    assert len(cz) == 4    # the NA row is dropped


def test_cross_index_masks_labels_and_counts():
    idx = _bdays(10)
    M = pd.DataFrame(index=idx)
    nlow = pd.Series([3, 3, 2, 1, 0, 0, 1, 2, 3, 3], index=idx)
    masks = S.cross_index_masks(M, nlow, n_indices=3)
    assert set(masks) == {"0of3", "1of3", "2of3", "3of3"}
    assert masks["3of3"].sum() == 4
    assert masks["0of3"].sum() == 2


def test_cross_index_section_smoke():
    n = 300
    idx = _bdays(n)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    M = pd.DataFrame({"rv": np.full(n, 18.0)}, index=idx)
    nlow = pd.Series(rng.integers(0, 4, n), index=idx)
    P = S.forward_path_panel(close, horizon=S.H)
    out = S.cross_index_section("NDX", M, P, nlow, 3, "QQQ")
    assert "BY N-OF-3 INDICES DISPERSED" in out


# ---------------------------------------------------------------------------
# rule rows (Phase 5)
# ---------------------------------------------------------------------------
def test_rule_row_ndx_dixlow_caution_smoke():
    n = 500
    idx = _bdays(n)
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    cz = np.where(np.arange(n) % 10 < 6, "LowCorr", "MidCorr")
    dz = np.where(np.arange(n) % 6 < 2, "DIXLow", np.where(np.arange(n) % 6 < 4,
                                                           "DIXMid", "DIXHigh"))
    M = pd.DataFrame({"cz_roll": cz, "dz_roll_l1": dz, "rv": np.full(n, 18.0)}, index=idx)
    P = S.forward_path_panel(close, horizon=S.H)
    active_mask = (M["cz_roll"] == "LowCorr") & (M["dz_roll_l1"] == "DIXLow")
    active_st = S.cell_stats(P, M["rv"], active_mask, seed=1)
    out = S.rule_row_ndx_dixlow_caution(M, P, active_st)
    assert "ndx_dixlow_caution_v1" in out
    assert "LowCorr-not-active" in out


def test_rule_row_all_dispersed_smoke():
    n = 300
    idx = _bdays(n)
    rng = np.random.default_rng(9)
    close_a = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    close_b = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    Ma = pd.DataFrame({"cz_roll": ["LowCorr"] * n, "rv": np.full(n, 18.0)}, index=idx)
    Mb = pd.DataFrame({"cz_roll": np.where(np.arange(n) % 3 == 0, "LowCorr", "MidCorr"),
                       "rv": np.full(n, 15.0)}, index=idx)
    frames = {"NDX": Ma, "SPX": Mb}
    paths = {"NDX": S.forward_path_panel(close_a, horizon=S.H),
            "SPX": S.forward_path_panel(close_b, horizon=S.H)}
    metas = {"NDX": {"proxy": "QQQ"}, "SPX": {"proxy": "SPY"}}
    nlow, _ = S.n_dispersed(frames)
    out = S.rule_row_all_dispersed(frames, paths, metas, nlow, 2)
    assert "all_dispersed_derisk_v1" in out
    assert "NDX active" in out and "SPX active" in out


# ---------------------------------------------------------------------------
# CSV row builders (Phase 5)
# ---------------------------------------------------------------------------
def test_parse_cell_label_variants():
    assert S._parse_cell_label("LowCorrxDIXLow(l1)") == ("LowCorr", "DIXLow")
    assert S._parse_cell_label("HighCorr") == ("HighCorr", "")
    assert S._parse_cell_label("VolLow") == ("VolLow", "")
    assert S._parse_cell_label("2of3") == ("", "")


def test_cell_to_long_rows_empty_when_ungated():
    assert S.cell_to_long_rows("NDX", "ENV", "LowCorr", {"gate": False}) == []


def test_cell_to_long_rows_shape_when_gated():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(18.0, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=1)
    rows = S.cell_to_long_rows("NDX", "ENV", "LowCorrxDIXLow(l1)", st)
    assert len(rows) > 0
    assert all(r["corr_regime"] == "LowCorr" and r["dix_regime"] == "DIXLow" for r in rows)
    assert all(r["gate"] for r in rows)
    assert any("p50" in r for r in rows)


def test_cell_to_risk_row_ungated_has_no_numeric_fields():
    row = S.cell_to_risk_row("NDX", "ENV", "LowCorr", {"gate": False, "n_days": 10, "n_eps": 1})
    assert row["gate"] is False
    assert "mae_q25" not in row


def test_cell_to_risk_row_gated_has_expected_columns():
    n = 400
    idx = _bdays(n)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    rv = pd.Series(18.0, index=idx)
    mask = pd.Series(False, index=idx)
    for s in (10, 60, 120, 180, 240, 300):
        mask.iloc[s:s + 10] = True
    P = S.forward_path_panel(close, horizon=S.H)
    st = S.cell_stats(P, rv, mask, seed=1)
    row = S.cell_to_risk_row("NDX", "ENV", "LowCorr", st)
    assert row["mae_q25"] == pytest.approx(st["exc"]["mae_q25"])
    assert "ci_mae_q25_lo" in row and "ci_mae_q25_hi" in row


# ---------------------------------------------------------------------------
# full end-to-end main()-style integration (3 synthetic indices, CSVs)
# ---------------------------------------------------------------------------
def test_cross_index_and_csv_integration(tmp_path, capsys):
    import intra_index_regime_study as R

    rng = np.random.default_rng(0)
    n = 900
    frames, metas, paths, all_long, all_risk = {}, {}, {}, [], []
    for name, proxy_sym in (("NDX", "QQQ"), ("SPX", "SPY"), ("IWM", "IWM")):
        idx = _bdays(n, start="2020-01-02")
        names = [f"{name}{i:02d}" for i in range(35)]   # >= MIN_NAMES so avg_corr is defined
        px = pd.DataFrame({t: 100.0 * np.cumprod(1 + rng.normal(0, 0.015, n)) for t in names},
                          index=idx)
        proxy = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n)), index=idx)
        dix = pd.Series(rng.normal(0.42, 0.05, n), index=idx)
        r1m = (proxy.shift(-21) / proxy - 1) * 100
        M = R.assemble_frame(px, proxy, dix, r1m)
        meta = {"proxy": proxy_sym, "proxy_close": proxy, "note": "smoke", "dropped": {}}
        P, all_stats = S.report_index(name, M, meta, type("A", (), {})())
        frames[name], metas[name], paths[name] = M, meta, P
        for label, st in all_stats.items():
            all_long.extend(S.cell_to_long_rows(name, "ENV", label, st))
            all_risk.append(S.cell_to_risk_row(name, "ENV", label, st))

    capsys.readouterr()   # discard the per-index report output
    nlow, _ = S.n_dispersed(frames)
    for name, M in frames.items():
        print(S.cross_index_section(name, M, paths[name], nlow, 3, metas[name]["proxy"]))
    print(S.rule_row_all_dispersed(frames, paths, metas, nlow, 3))
    out = capsys.readouterr().out
    assert "BY N-OF-3 INDICES DISPERSED" in out
    assert "all_dispersed_derisk_v1" in out

    long_csv, risk_csv = tmp_path / "paths.csv", tmp_path / "risk.csv"
    pd.DataFrame(all_long).to_csv(long_csv, index=False)
    pd.DataFrame(all_risk).to_csv(risk_csv, index=False)
    long_df = pd.read_csv(long_csv)
    risk_df = pd.read_csv(risk_csv)
    assert {"index", "family", "cell", "h", "gate"}.issubset(long_df.columns)
    assert {"index", "cell", "mae_q25", "gate"}.issubset(risk_df.columns)
    assert (risk_df["gate"] == True).any()   # noqa: E712 -- at least one gated row present
