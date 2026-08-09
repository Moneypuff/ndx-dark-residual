"""Tests for the pure pieces of ``build_regime_log``.

The regime log's value rests on the break detector firing on the right day,
in the right direction, with the right context tag -- and on the forward
outcome column being strictly look-ahead-only relative to the signal date.
Each of those behaviors is pinned here on hand-built series.
"""
import numpy as np
import pandas as pd
import pytest

import build_regime_log as R


def _px(vals, start="2020-01-02"):
    return pd.Series(list(vals), index=pd.bdate_range(start, periods=len(vals)),
                     dtype="float64")


def _theme_break_prices(n=420, peak=340):
    """Pair where a runs a strong theme vs b (~+20pp/quarter, like semis 2026),
    then rolls over at `peak` fast enough to break but slow enough that the
    theme is still 'on' when the z-cross fires -- the realistic profile."""
    rng = np.random.default_rng(1)
    drift = np.where(np.arange(n) < peak, 0.003, -0.003)
    a = 100 * np.cumprod(1 + drift + rng.normal(0, 0.004, n))
    b = 100 * np.cumprod(1 + rng.normal(0, 0.004, n))
    return _px(a), _px(b)


# ---------------------------------------------------------------------------
# spread_frames
# ---------------------------------------------------------------------------
def test_spread_frames_values_and_forward_alignment():
    a = _px(100 * 1.01 ** np.arange(300))     # +1%/day
    b = _px(np.full(300, 100.0))              # flat
    s21, s63, z, fwd = R.spread_frames(a, b)
    assert s21.iloc[25] == pytest.approx((1.01 ** 21 - 1) * 100, rel=1e-9)
    assert s63.iloc[70] == pytest.approx((1.01 ** 63 - 1) * 100, rel=1e-9)
    # fwd at t covers exactly the NEXT 63 sessions
    t = 100
    assert fwd.iloc[t] == pytest.approx((a.iloc[t + 63] / a.iloc[t] - 1) * 100, rel=1e-9)
    assert fwd.iloc[-63:].isna().all()


def test_spread_frames_aligns_mismatched_indexes():
    a = _px(np.full(100, 100.0))
    b = _px(np.full(90, 100.0))               # shorter
    s21, s63, z, fwd = R.spread_frames(a, b)
    assert len(s21) == 90


# ---------------------------------------------------------------------------
# detect_breaks
# ---------------------------------------------------------------------------
def test_detect_breaks_fires_after_reversal_in_theme_direction():
    a, b = _theme_break_prices()
    s21, s63, z, _ = R.spread_frames(a, b)
    sigs = R.detect_breaks(s63, z)
    pos = [d for d, sgn in sigs if sgn > 0]
    assert pos, "the broken uptrend theme must produce a +1 signal"
    peak_date = a.index[340]
    # fires after the peak, within ~2 months of it
    assert all(d > peak_date for d in pos)
    assert min(pos) < peak_date + pd.Timedelta(days=60)


def test_detect_breaks_respects_cooldown():
    a, b = _theme_break_prices()
    _, s63, z, _ = R.spread_frames(a, b)
    sigs = [d for d, sgn in R.detect_breaks(s63, z) if sgn > 0]
    locs = [s63.index.get_loc(d) for d in sigs]
    assert all(j - i >= R.COOLDOWN for i, j in zip(locs, locs[1:]))


def test_detect_breaks_inverse_direction():
    a, b = _theme_break_prices()
    # swap: now the theme is b-under-a breaking means sign -1 on (b, a)
    _, s63, z, _ = R.spread_frames(b, a)
    sigs = R.detect_breaks(s63, z)
    assert any(sgn < 0 for _, sgn in sigs)
    assert not any(sgn > 0 for _, sgn in sigs)


def test_detect_breaks_quiet_pair_is_silent():
    rng = np.random.default_rng(3)
    a = _px(100 * np.cumprod(1 + rng.normal(0, 0.002, 400)))
    b = _px(100 * np.cumprod(1 + rng.normal(0, 0.002, 400)))
    _, s63, z, _ = R.spread_frames(a, b)
    # with no sustained >5pp theme, signals should be rare-to-none
    assert len(R.detect_breaks(s63, z)) <= 1


# ---------------------------------------------------------------------------
# shocks, gamma release, buckets
# ---------------------------------------------------------------------------
def test_shock_dates_flags_spike_only():
    idx = pd.bdate_range("2020-01-02", periods=400)
    cor = pd.Series(10.0, index=idx)
    cor.iloc[350:360] = 30.0                   # a violent correlation spike
    shocks = R.shock_dates(cor)
    assert shocks and all(idx[349] <= s <= idx[365] for s in shocks)


def test_near_shock_window():
    d = pd.Timestamp("2024-08-15")
    assert R.near_shock(d, [pd.Timestamp("2024-08-02")])          # before, within 34d
    assert R.near_shock(d, [pd.Timestamp("2024-08-20")])          # after, within 8d
    assert not R.near_shock(d, [pd.Timestamp("2024-06-01")])
    assert not R.near_shock(d, [pd.Timestamp("2024-09-15")])


def test_gamma_release_flag_lookback():
    idx = pd.bdate_range("2021-01-04", periods=80)
    g = pd.Series(80.0, index=idx)
    g.iloc[40:] = 20.0                          # 60-pt collapse at position 40
    f = R.gamma_release_flag(g)
    assert bool(f.iloc[41])
    # release stays flagged for GREL_LOOKBACK sessions after the drop leaves the diff window
    assert bool(f.iloc[40 + 21 + R.GREL_LOOKBACK - 2])
    assert not bool(f.iloc[30])


@pytest.mark.parametrize("shock,grel,expected", [
    (True, True, "fade"),                       # shock dominates
    (True, False, "fade"),
    (False, True, "rotation"),
    (False, False, "derisk"),
])
def test_bucket_for(shock, grel, expected):
    assert R.bucket_for(shock, grel) == expected


# ---------------------------------------------------------------------------
# leaderboard state + posture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r63,z21,expected", [
    (8.0, 0.5, "HOT"), (8.0, -1.4, "HOT, breaking"),
    (-8.0, -0.5, "UNFAVORED"), (-8.0, 1.4, "UNFAVORED, turning"),
    (2.0, 3.0, "NEUTRAL"),
])
def test_lb_state(r63, z21, expected):
    assert R.lb_state(r63, z21) == expected


def test_posture_priority_order():
    assert R.posture(True, True, True)[0] == "shock"
    assert R.posture(False, True, True)[0] == "open"
    assert R.posture(False, False, True)[0] == "settling"
    assert R.posture(False, False, False)[0] == "stable"


# ---------------------------------------------------------------------------
# conviction: value area, scoring, events
# ---------------------------------------------------------------------------
def test_value_area_concentrates_where_volume_is():
    idx = pd.bdate_range("2020-01-02", periods=300)
    # price wanders 100->120, but ALL the volume trades in the 100-105 leg
    c = pd.Series(np.linspace(100, 120, 300), index=idx)
    v = pd.Series(np.where(c < 105, 5e6, 1e4), index=idx)
    lo, hi = R.value_area_edges(c, v, window=252)
    assert lo.iloc[:251].isna().all() and np.isfinite(lo.iloc[251])
    assert hi.iloc[-1] < 112     # value area hugs the high-volume 100-105 zone
    assert lo.iloc[-1] >= 99


def test_conviction_scores_confirmed_breakout():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2019-01-02", periods=298)
    # flat coil, then a FRESH hard upside break (8 sessions old) on 2.5x volume --
    # the level/band components carry 10- and 5-session recency windows, so the
    # break must still be recent on the last bar
    px = np.concatenate([100 + rng.normal(0, 0.4, 290), np.linspace(101, 112, 8)])
    c = pd.Series(px, index=idx)
    v = pd.Series(np.concatenate([np.full(290, 1e6), np.full(8, 2.5e6)]), index=idx)
    conv = R.conviction_frame(c, v)
    last = conv.iloc[-1]
    assert last["dir"] == 1
    assert last["score"] >= 3            # MA side, level break, band thrust, volume
    assert bool(last["level"]) and bool(last["vol"])


def test_conviction_quiet_tape_scores_low():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2019-01-02", periods=320)
    c = pd.Series(100 + rng.normal(0, 0.3, 320).cumsum() * 0.05, index=idx)
    v = pd.Series(1e6, index=idx)
    conv = R.conviction_frame(c, v)
    assert conv["score"].iloc[-1] <= 2   # no level break, no thrust, no volume


def test_conviction_events_direction_sign_and_cooldown():
    idx = pd.bdate_range("2019-01-02", periods=400)
    c = pd.Series(np.concatenate([np.full(300, 100.0), np.linspace(100, 130, 100)]), index=idx)
    conv = pd.DataFrame({"dir": 0, "score": 0, "mas": False, "level": False,
                         "band": False, "vol": False}, index=idx)
    conv.loc[idx[310], ["dir", "score"]] = [1, 3]
    conv.loc[idx[315], ["dir", "score"]] = [1, 4]   # inside cooldown -> skipped
    conv.loc[idx[350], ["dir", "score"]] = [1, 3]
    ev = R.conviction_events(c, conv, horizon=21)
    assert list(ev["date"]) == [idx[310], idx[350]]
    assert (ev["fwd"] > 0).all()          # rising tape, up direction -> positive signed fwd
