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
