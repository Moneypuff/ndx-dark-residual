"""Tests for the pure pieces of ``build_vol_surface``.

Pins three mechanics: the moneyness-grid resample with no extrapolation
(a hole is honest missing data, not a flat-carried edge), the
nearest-first expiry cap that keeps ``symbol_surface`` output bounded,
and the payload's day window / symbol inclusion rules. All on hand-built
snapshot frames -- no network, no real chains.
"""
import json

import numpy as np
import pandas as pd
import pytest

import build_vol_surface as V


def _rows(date, symbol, expiry, spot, quotes):
    """quotes: list of (right, strike, iv, oi)."""
    return [{"date": date, "symbol": symbol, "expiry": expiry, "right": r,
             "strike": k, "iv": iv, "oi": oi, "volume": 0, "bid": 1.0,
             "ask": 1.2, "last": 1.1, "spot": spot}
            for r, k, iv, oi in quotes]


FLAT_SMILE = [("P", 80.0, 0.30, 1000), ("P", 90.0, 0.30, 1000),
              ("C", 100.0, 0.30, 1000), ("C", 105.0, 0.30, 1000),
              ("C", 110.0, 0.30, 1000)]


# ---------------------------------------------------------------------------
# surface_grid
# ---------------------------------------------------------------------------
def test_surface_grid_flat_smile():
    day = pd.DataFrame(_rows("2026-08-10", "GDX", "2027-01-15", 100.0, FLAT_SMILE))
    grid = V.surface_grid(day, 100.0)
    assert grid is not None
    inside = (V.M_GRID >= 0.80) & (V.M_GRID <= 1.10)
    assert np.allclose(grid[inside], 0.30, atol=1e-9)
    at_060 = np.isclose(V.M_GRID, 0.60)
    at_140 = np.isclose(V.M_GRID, 1.40)
    assert np.all(np.isnan(grid[at_060]))
    assert np.all(np.isnan(grid[at_140]))


def test_surface_grid_thin_returns_none():
    thin = pd.DataFrame(_rows("2026-08-10", "GDX", "2027-01-15", 100.0, [
        ("P", 90.0, 0.30, 1000), ("C", 100.0, 0.30, 1000),
        ("C", 110.0, 0.30, 1000)]))
    assert V.surface_grid(thin, 100.0) is None

    one_sided = pd.DataFrame(_rows("2026-08-10", "GDX", "2027-01-15", 100.0, [
        ("C", 101.0, 0.30, 1000), ("C", 103.0, 0.30, 1000),
        ("C", 105.0, 0.30, 1000), ("C", 107.0, 0.30, 1000),
        ("C", 110.0, 0.30, 1000)]))
    assert V.surface_grid(one_sided, 100.0) is None


# ---------------------------------------------------------------------------
# symbol_surface
# ---------------------------------------------------------------------------
def _many_expiries(n, symbol="XYZ", spot=100.0, start="2026-09-01", step_days=30):
    rows = []
    for i in range(n):
        expiry = str((pd.Timestamp(start) + pd.Timedelta(days=step_days * i)).date())
        rows += _rows("2026-08-10", symbol, expiry, spot, FLAT_SMILE)
    return pd.DataFrame(rows)


def test_symbol_surface_orders_and_caps_expiries():
    day = _many_expiries(16)
    surf = V.symbol_surface(day, "XYZ", "2026-08-10")
    assert surf is not None
    assert len(surf["expiries"]) == V.MAX_EXPIRIES
    assert len(surf["atm"]) == V.MAX_EXPIRIES
    assert len(surf["iv"]) == V.MAX_EXPIRIES
    dtes = surf["dtes"]
    assert len(dtes) == V.MAX_EXPIRIES
    assert all(a < b for a, b in zip(dtes, dtes[1:]))


def test_symbol_surface_skips_dead_expiry():
    good = _rows("2026-08-10", "XYZ", "2026-09-18", 100.0, FLAT_SMILE)
    thin = _rows("2026-08-10", "XYZ", "2026-10-16", 100.0, [
        ("P", 90.0, 0.30, 1000), ("C", 100.0, 0.30, 1000),
        ("C", 110.0, 0.30, 1000)])
    day = pd.DataFrame(good + thin)
    surf = V.symbol_surface(day, "XYZ", "2026-08-10")
    assert surf is not None
    assert surf["expiries"] == ["2026-09-18"]


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------
def test_build_payload_shape_and_day_cap():
    dates = [str((pd.Timestamp("2026-07-01") + pd.Timedelta(days=i)).date())
             for i in range(12)]
    rows = []
    for d in dates:
        rows += _rows(d, "AAA", "2027-06-18", 100.0, FLAT_SMILE)
    df = pd.DataFrame(rows)

    payload = V.build_payload(df, days=10)
    assert payload["dates"] == dates[-10:]
    assert payload["dates"] == sorted(payload["dates"])
    assert "AAA" in payload["symbols"]

    dumped = json.dumps(payload)
    assert "NaN" not in dumped


def test_build_payload_excludes_surfaceless_symbol():
    d = "2026-08-10"
    good = _rows(d, "AAA", "2027-06-18", 100.0, FLAT_SMILE)
    thin = _rows(d, "BBB", "2027-06-18", 100.0, [
        ("P", 90.0, 0.30, 1000), ("C", 100.0, 0.30, 1000),
        ("C", 110.0, 0.30, 1000)])
    df = pd.DataFrame(good + thin)

    payload = V.build_payload(df, days=10)
    assert "AAA" in payload["symbols"]
    assert "BBB" not in payload["symbols"]
    assert "BBB" not in payload["surfaces"]


# ---------------------------------------------------------------------------
# model predictions: hybrid_weight / predict_row_strike / model_grids
# ---------------------------------------------------------------------------
def test_hybrid_weight_shape():
    atm_iv, dte = 0.30, 365
    sigma = atm_iv * (dte / 365.25) ** 0.5
    cases = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.5), (1.5, 1.0), (3.0, 1.0)]
    for d, expected in cases:
        m = 1.0 + d * sigma
        assert V.hybrid_weight(m, atm_iv, dte) == pytest.approx(expected, abs=1e-6)

    ds = np.linspace(0, 3, 50)
    ws = [V.hybrid_weight(1.0 + d * sigma, atm_iv, dte) for d in ds]
    assert all(a <= b + 1e-12 for a, b in zip(ws, ws[1:]))


def test_predict_row_strike_shift():
    prev_row = [round(0.30 - 0.20 * (m - 1.0), 6) for m in V.M_GRID]
    pred = V.predict_row_strike(prev_row, 1.10)

    i_atm = int(np.argmin(np.abs(V.M_GRID - 1.0)))
    target = float(V.M_GRID[i_atm]) * 1.10
    assert pred[i_atm] == pytest.approx(0.30 - 0.20 * (target - 1.0), abs=1e-6)

    i_far_wing = len(V.M_GRID) - 1   # m=1.40 -> target=1.54, past the 1.40 block edge
    assert pred[i_far_wing] is None


def test_predict_row_strike_flat_ratio1():
    n = len(V.M_GRID)
    prev_row = [None] * n
    for i in range(5, 28):
        prev_row[i] = 0.30
    pred = V.predict_row_strike(prev_row, 1.0)
    for i in range(5, 28):
        assert pred[i] == pytest.approx(0.30, abs=1e-9)
    assert pred[4] is None
    assert pred[28] is None


def test_model_grids_delta_is_prev_row():
    n = len(V.M_GRID)
    prev_surf = {"spot": 100.0, "expiries": ["2026-09-18"], "dtes": [36],
                 "atm": [0.30], "iv": [[0.30] * n]}
    cur_surf = {"spot": 101.0, "expiries": ["2026-09-18", "2026-12-18"],
                "dtes": [30, 120], "atm": [0.31, None],
                "iv": [[0.31] * n, [None] * n]}

    models = V.model_grids(prev_surf, cur_surf)
    assert models is not None
    assert models["delta"][0] == prev_surf["iv"][0]
    assert all(v is None for v in models["strike"][1])
    assert all(v is None for v in models["delta"][1])
    assert all(v is None for v in models["hybrid"][1])


def test_model_grids_hybrid_interpolates_between():
    m_grid = V.M_GRID
    skew = [round(0.30 - 0.20 * (m - 1.0), 6) for m in m_grid]
    prev_surf = {"spot": 100.0, "expiries": ["E"], "dtes": [180],
                 "atm": [0.20], "iv": [skew]}
    cur_surf = {"spot": 105.0, "expiries": ["E"], "dtes": [170],
                "atm": [0.21], "iv": [[None] * len(m_grid)]}

    models = V.model_grids(prev_surf, cur_surf)
    strike_row, delta_row, hybrid_row = (models["strike"][0], models["delta"][0],
                                          models["hybrid"][0])

    i_atm = int(np.argmin(np.abs(m_grid - 1.0)))
    assert hybrid_row[i_atm] == pytest.approx(delta_row[i_atm], abs=1e-9)

    i_wing = int(np.argmin(np.abs(m_grid - 1.30)))
    assert strike_row[i_wing] is not None
    assert hybrid_row[i_wing] == pytest.approx(strike_row[i_wing], abs=1e-9)
    assert abs(strike_row[i_wing] - delta_row[i_wing]) > 1e-6

    i_mid = int(np.argmin(np.abs(m_grid - 1.15)))
    lo, hi = sorted([delta_row[i_mid], strike_row[i_mid]])
    assert lo < hybrid_row[i_mid] < hi


def test_build_payload_models_presence_and_hygiene():
    dates = ["2026-08-10", "2026-08-11", "2026-08-12"]
    rows = []
    for d in dates:
        rows += _rows(d, "AAA", "2027-06-18", 100.0, FLAT_SMILE)
    df = pd.DataFrame(rows)

    payload = V.build_payload(df, days=10)
    d0, d1, d2 = dates
    assert "models" not in payload["surfaces"]["AAA"][d0]
    assert "models" in payload["surfaces"]["AAA"][d1]
    assert "models" in payload["surfaces"]["AAA"][d2]

    dumped = json.dumps(payload)
    assert "NaN" not in dumped


def test_sticky_strike_world_zero_residual():
    """IV a fixed function of ABSOLUTE STRIKE both days -- the sticky-
    strike model should reproduce day 2's observed grid almost exactly
    (residual bounded only by 4dp rounding), while the sticky-delta
    model, which ignores the strike shift, misses materially."""
    strike_iv = lambda k: 0.30 - 0.001 * (k - 100)

    def day(date, spot):
        quotes = []
        for k in np.arange(60, 141, 2.0):
            right = "P" if k < spot else "C"
            quotes.append((right, float(k), round(float(strike_iv(k)), 6), 1000))
        return _rows(date, "STRK", "2026-12-18", spot, quotes)

    df = pd.DataFrame(day("2026-08-10", 100.0) + day("2026-08-11", 105.0))
    payload = V.build_payload(df, days=10)
    surf2 = payload["surfaces"]["STRK"]["2026-08-11"]
    assert "models" in surf2

    obs2 = surf2["iv"][0]
    strike_pred = surf2["models"]["strike"][0]
    delta_pred = surf2["models"]["delta"][0]

    strike_resid = [o - p for o, p in zip(obs2, strike_pred) if o is not None and p is not None]
    delta_resid = [o - p for o, p in zip(obs2, delta_pred) if o is not None and p is not None]
    assert strike_resid and delta_resid
    assert max(abs(r) for r in strike_resid) < 1e-3
    assert max(abs(r) for r in delta_resid) >= 0.001


def test_sticky_delta_world_zero_residual():
    """Mirror of the sticky-strike world: IV a fixed function of
    MONEYNESS both days -- the sticky-delta model should reproduce day
    2 almost exactly, while sticky-strike (which shifts by the spot
    ratio) misses materially."""
    delta_iv = lambda m: 0.30 - 0.20 * (m - 1.0)

    def day(date, spot):
        quotes = []
        for m in np.arange(0.60, 1.401, 0.02):
            k = m * spot
            right = "P" if k < spot else "C"
            quotes.append((right, round(float(k), 2), round(float(delta_iv(m)), 6), 1000))
        return _rows(date, "DELT", "2026-12-18", spot, quotes)

    df = pd.DataFrame(day("2026-08-10", 100.0) + day("2026-08-11", 105.0))
    payload = V.build_payload(df, days=10)
    surf2 = payload["surfaces"]["DELT"]["2026-08-11"]
    assert "models" in surf2

    obs2 = surf2["iv"][0]
    strike_pred = surf2["models"]["strike"][0]
    delta_pred = surf2["models"]["delta"][0]

    delta_resid = [o - p for o, p in zip(obs2, delta_pred) if o is not None and p is not None]
    strike_resid = [o - p for o, p in zip(obs2, strike_pred) if o is not None and p is not None]
    assert delta_resid and strike_resid
    assert max(abs(r) for r in delta_resid) < 1e-3
    assert max(abs(r) for r in strike_resid) >= 0.001
