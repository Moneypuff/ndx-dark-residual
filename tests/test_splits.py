"""Tests for the stock-split detection and adjustment logic in
``ndx_dark_residual``.

FINRA off-exchange volume is reported on an as-traded (unadjusted) basis, so a
stock split makes the per-name dark ratio jump discontinuously. The pipeline
detects those jumps heuristically from the FINRA-total / consolidated-volume
ratio (``detect_split_factors``) and rescales the pre-split rows back onto the
split-adjusted basis (``_cumulative_split_factor`` -> ``apply_split_factors``
-> ``split_adjust_finra_panel``) so the ratio is continuous across the split.

There is no oracle for this heuristic, so the tests drive it with synthetic
ratio series that have a known split (or no split) and assert the documented
behaviour: which factors are detected and snapped, where the effective date
lands, which ambiguous steps are rejected, and that the rescaling makes the
ratio continuous.
"""
import numpy as np
import pandas as pd
import pytest

import ndx_dark_residual as N


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bdays(n, start="2022-01-03"):
    return pd.bdate_range(start, periods=n)


def make_ratio(n=120, split_at=60, factor=3.0, base=0.4, noise=0.0, seed=0):
    """Synthetic FINRA/volume ratio: flat at ``base``, stepping up by ``factor``
    at position ``split_at`` (an as-traded volume series jumps up on a split)."""
    rng = np.random.default_rng(seed)
    idx = _bdays(n)
    vals = np.full(n, float(base))
    vals[split_at:] *= factor
    if noise:
        vals = vals * np.exp(rng.normal(0.0, noise, n))
    return pd.Series(vals, index=idx)


# ---------------------------------------------------------------------------
# detect_split_factors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [3.0, 4.0, 5.0, 10.0])
def test_detects_clean_split_factor_and_date(factor):
    r = make_ratio(n=120, split_at=60, factor=factor)
    out = N.detect_split_factors(r)
    assert list(out.values()) == [factor]
    # Clean synthetic step -> effective date localizes exactly on the true split.
    assert list(out.keys()) == [r.index[60]]


def test_two_to_one_below_threshold_not_flagged():
    # SPLIT_JUMP_THR = log(2.2); a clean 2:1 step (log 2) sits just under it.
    # The detector is documented to catch >=3:1 robustly, so a pure 2:1 with no
    # noise is intentionally not flagged.
    out = N.detect_split_factors(make_ratio(factor=2.0))
    assert out == {}


def test_noisy_step_snaps_to_nearest_allowed_ratio():
    # A ~3.9x step under noise should snap to the allowed ratio 4, not 3.9.
    out = N.detect_split_factors(make_ratio(factor=3.9, noise=0.05, seed=1))
    assert list(out.values()) == [4]


def test_snapped_factor_is_from_allowed_set():
    out = N.detect_split_factors(make_ratio(factor=6.0))
    assert set(out.values()) <= set(N.SPLIT_ALLOWED)


def test_no_split_on_gradual_drift():
    # A slow monotone ramp is a regime shift, not a split: the windowed median
    # jump never exceeds the threshold, so nothing is flagged.
    idx = _bdays(120)
    ramp = pd.Series(np.linspace(0.3, 3.0, 120), index=idx)
    assert N.detect_split_factors(ramp) == {}


def test_no_split_on_flat_noisy_series():
    assert N.detect_split_factors(make_ratio(factor=1.0, noise=0.06, seed=2)) == {}


def test_too_short_series_returns_empty():
    # Needs at least 3 * window (default window=12 -> 36) points.
    assert N.detect_split_factors(make_ratio(n=30, factor=3.0)) == {}


def test_extreme_step_far_from_any_ratio_is_rejected():
    # A 25x step exceeds the jump threshold but its nearest allowed ratio (20)
    # is >22% away, so the snap guard rejects it rather than mislabel it.
    assert N.detect_split_factors(make_ratio(factor=25.0)) == {}


def test_all_nan_series_returns_empty():
    idx = _bdays(120)
    assert N.detect_split_factors(pd.Series(np.nan, index=idx)) == {}


# ---------------------------------------------------------------------------
# _cumulative_split_factor
# ---------------------------------------------------------------------------
def test_cumulative_factor_single_split():
    idx = _bdays(6)
    fac = N._cumulative_split_factor(idx, {idx[3]: 3.0})
    # Strictly-before rows scale by 3; the split day and after stay at 1.
    assert list(fac) == [3.0, 3.0, 3.0, 1.0, 1.0, 1.0]


def test_cumulative_factor_compounds_multiple_splits():
    idx = _bdays(6)
    fac = N._cumulative_split_factor(idx, {idx[2]: 3.0, idx[4]: 2.0})
    # Before both -> 3*2=6; between the two -> 2; on/after the last -> 1.
    assert list(fac) == [6.0, 6.0, 2.0, 2.0, 1.0, 1.0]


def test_cumulative_factor_empty_is_all_ones():
    idx = _bdays(5)
    assert list(N._cumulative_split_factor(idx, {})) == [1.0] * 5


# ---------------------------------------------------------------------------
# apply_split_factors
# ---------------------------------------------------------------------------
def test_apply_rescales_presplit_rows_to_continuity():
    idx = _bdays(6)
    # As-traded volume that halves-then-thirds visually: 100 pre, 300 post a 3:1.
    panel = pd.DataFrame({"AAA": [100., 100, 100, 300, 300, 300],
                          "BBB": [10.] * 6}, index=idx)
    adj = N.apply_split_factors(panel, {"AAA": {idx[3]: 3.0}})
    # Pre-split 100*3 == post-split 300 -> a single continuous level.
    assert list(adj["AAA"]) == [300.] * 6
    # Untouched symbol is unchanged.
    assert list(adj["BBB"]) == [10.] * 6


def test_apply_empty_panel_returns_empty():
    idx = _bdays(6)
    out = N.apply_split_factors(pd.DataFrame(), {"AAA": {idx[3]: 3.0}})
    assert out.empty


def test_apply_no_factors_is_identity():
    idx = _bdays(6)
    panel = pd.DataFrame({"AAA": [100., 100, 100, 300, 300, 300]}, index=idx)
    out = N.apply_split_factors(panel, {})
    pd.testing.assert_frame_equal(out, panel)


def test_apply_ignores_symbol_absent_from_panel():
    idx = _bdays(6)
    panel = pd.DataFrame({"AAA": [100., 100, 100, 300, 300, 300]}, index=idx)
    out = N.apply_split_factors(panel, {"ZZZ": {idx[3]: 3.0}})
    assert list(out["AAA"]) == [100., 100, 100, 300, 300, 300]


def test_apply_does_not_mutate_input():
    idx = _bdays(6)
    panel = pd.DataFrame({"AAA": [100., 100, 100, 300, 300, 300]}, index=idx)
    before = panel["AAA"].copy()
    N.apply_split_factors(panel, {"AAA": {idx[3]: 3.0}})
    pd.testing.assert_series_equal(panel["AAA"], before)


# ---------------------------------------------------------------------------
# panel_split_factors
# ---------------------------------------------------------------------------
def _panel_pair(n=120, split_at=60, factor=3.0):
    idx = _bdays(n)
    vol = pd.Series(1e6, index=idx)
    spl = pd.Series(0.4e6, index=idx)
    spl.iloc[split_at:] *= factor
    flat = pd.Series(0.4e6, index=idx)
    finra = pd.DataFrame({"SPL": spl, "FLAT": flat})
    volume = pd.DataFrame({"SPL": vol, "FLAT": vol})
    return finra, volume, idx


def test_panel_detects_only_split_symbols():
    finra, volume, idx = _panel_pair(factor=3.0)
    facs = N.panel_split_factors(finra, volume)
    assert set(facs) == {"SPL"}
    assert list(facs["SPL"].values()) == [3]
    assert list(facs["SPL"].keys()) == [idx[60]]


def test_panel_skips_symbol_missing_from_volume():
    finra, volume, _ = _panel_pair(factor=3.0)
    # Drop the split symbol from the volume panel -> it cannot be evaluated.
    facs = N.panel_split_factors(finra, volume.drop(columns=["SPL"]))
    assert "SPL" not in facs


# ---------------------------------------------------------------------------
# split_adjust_finra_panel (end-to-end)
# ---------------------------------------------------------------------------
def test_split_adjust_makes_ratio_continuous():
    finra, volume, _ = _panel_pair(factor=3.0)
    adj = N.split_adjust_finra_panel(finra, volume)
    ratio = adj["SPL"] / volume["SPL"]
    # Pre- and post-split dark ratios coincide after adjustment.
    assert ratio.iloc[30] == pytest.approx(ratio.iloc[90])
    # The flat symbol is passed through untouched.
    pd.testing.assert_series_equal(adj["FLAT"], finra["FLAT"])


def test_split_adjust_empty_panel_returns_empty():
    assert N.split_adjust_finra_panel(pd.DataFrame(), pd.DataFrame()).empty
