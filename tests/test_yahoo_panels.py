"""Tests for ``load_yahoo_panels``' cache-preservation and backfill behavior.

The Yahoo price panel is a single pickle shared across every builder (the
dashboard, comovement, gex_dispersion, regime_log, earnings). A caller that
passes ``refresh=True`` previously skipped loading that shared cache before
overwriting it, so any symbol NOT requested by that particular call was
silently dropped from the file -- a --refresh dispatch of one builder could
gut every other builder's history. Separately, the incremental fetch path
only ever re-fetched a stale symbol forward from its last cached date, so a
symbol truncated by that clobber (or any other narrower-window write) could
never recover on its own. Both are pinned here with a fake ``fetch_yahoo_one``
and a tmp_path cache -- no network.
"""
from pathlib import Path

import pandas as pd
import pytest

import ndx_dark_residual as N


def _mkdf(dates, value=100.0):
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"close": value, "adjclose": value, "volume": 1000}, index=idx)


def _seed_cache(cache_dir, panels_by_symbol, synced=None, nodata=None):
    """Write a yahoo_prices.pkl whose wide frames hold exactly the given
    per-symbol date ranges, mirroring what load_yahoo_panels itself writes."""
    fields = ("close", "adjclose", "volume")
    out = {}
    for f in fields:
        out[f] = pd.DataFrame({s: df[f] for s, df in panels_by_symbol.items()})
    payload = {**out, "_synced": synced or "", "_nodata": sorted(nodata or [])}
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, cache_dir / N.YAHOO_CACHE)


def _fake_fetch(calls, series_by_symbol):
    """A fetch_yahoo_one stand-in that records (sym, start) and returns a daily
    series for `sym` covering [start, end] from a pre-baked source series."""
    def _fetch(sym, start, end, session=None, **kw):
        calls.append((sym, pd.Timestamp(start)))
        src = series_by_symbol.get(sym)
        if src is None:
            return pd.DataFrame(columns=["close", "adjclose", "volume"])
        win = src[(src.index >= pd.Timestamp(start)) & (src.index <= pd.Timestamp(end))]
        return win.copy()
    return _fetch


# ---------------------------------------------------------------------------
# no-clobber: a --refresh call for symbol A must not drop symbol B from the
# shared pickle it doesn't even request.
# ---------------------------------------------------------------------------
def test_refresh_preserves_untouched_symbols(tmp_path, monkeypatch):
    other_dates = pd.bdate_range("2004-01-02", "2004-06-30")
    _seed_cache(tmp_path, {"UNTOUCHED": _mkdf(other_dates)})

    fresh_dates = pd.bdate_range("2020-01-02", "2020-01-10")
    fresh_src = {"REFRESHED": _mkdf(fresh_dates, value=200.0)}
    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, fresh_src))

    out = N.load_yahoo_panels(["REFRESHED"], "2020-01-02", "2020-01-10",
                              cache_dir=tmp_path, refresh=True, label="T")
    assert "REFRESHED" in out["close"].columns
    assert out["close"]["REFRESHED"].notna().sum() == len(fresh_dates)

    # The rewritten pickle must still carry the untouched symbol's full history.
    reread = pd.read_pickle(tmp_path / N.YAHOO_CACHE)
    assert "UNTOUCHED" in reread["close"].columns
    assert reread["close"]["UNTOUCHED"].dropna().index.min() == other_dates.min()
    assert reread["close"]["UNTOUCHED"].dropna().index.max() == other_dates.max()


def test_refresh_forces_full_refetch_even_if_cache_looks_current(tmp_path, monkeypatch):
    # A symbol already cached through the target session must still be re-fetched in
    # full under refresh=True, not skipped because it "looks current".
    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    _seed_cache(tmp_path, {"SPY": _mkdf(dates)})

    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, {"SPY": _mkdf(dates, value=5.0)}))

    N.load_yahoo_panels(["SPY"], "2024-01-02", "2024-01-31",
                        cache_dir=tmp_path, refresh=True, label="T")
    assert calls, "refresh=True must fetch even an already-current symbol"
    assert calls[0][0] == "SPY"
    assert calls[0][1] == pd.Timestamp("2024-01-02")   # fetched from the full window start


# ---------------------------------------------------------------------------
# backfill: a symbol truncated well past `start` must be re-fetched in full,
# not merely forward from its last cached date.
# ---------------------------------------------------------------------------
def test_truncated_symbol_is_backfilled_not_just_extended(tmp_path, monkeypatch):
    # Cached history starts in 2018 even though the caller wants back to 2004 --
    # simulating exactly the mega-cap truncation the clobber bug produced.
    truncated_dates = pd.bdate_range("2018-01-02", "2018-12-31")
    _seed_cache(tmp_path, {"AAPL": _mkdf(truncated_dates)})

    full_dates = pd.bdate_range("2004-01-02", "2019-01-31")
    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one",
                        _fake_fetch(calls, {"AAPL": _mkdf(full_dates, value=7.0)}))

    out = N.load_yahoo_panels(["AAPL"], "2004-01-02", "2019-01-31",
                              cache_dir=tmp_path, refresh=False, label="T")

    assert calls, "a truncated symbol must trigger a fetch, not be treated as current"
    assert calls[0] == ("AAPL", pd.Timestamp("2004-01-02")), (
        "must backfill from the requested start, not from the symbol's last cached date")
    assert out["close"]["AAPL"].dropna().index.min() == pd.Timestamp("2004-01-02")


def test_mildly_stale_symbol_still_only_extends_forward(tmp_path, monkeypatch):
    # Regression guard: a symbol that is merely a few days behind (the ordinary
    # incremental case) must still be fetched only forward, not backfilled --
    # backfilling everything on every run would defeat the incremental cache entirely.
    dates = pd.bdate_range("2020-01-02", "2020-06-01")
    _seed_cache(tmp_path, {"MSFT": _mkdf(dates)})

    calls = []
    new_dates = pd.bdate_range(dates.max(), "2020-06-10")
    monkeypatch.setattr(N, "fetch_yahoo_one",
                        _fake_fetch(calls, {"MSFT": _mkdf(new_dates, value=9.0)}))

    N.load_yahoo_panels(["MSFT"], "2020-01-02", "2020-06-10",
                        cache_dir=tmp_path, refresh=False, label="T")

    assert calls == [("MSFT", dates.max())], (
        "a symbol within the backfill tolerance of `start` must only extend forward")


def test_symbol_within_tolerance_of_start_is_not_backfilled(tmp_path, monkeypatch):
    # A symbol whose cached start is within YAHOO_BACKFILL_TOL_DAYS of the requested
    # start is NOT truncated in the pathological sense -- must not force a full re-pull.
    tol = N.YAHOO_BACKFILL_TOL_DAYS
    cached_start = pd.Timestamp("2020-01-02") + pd.Timedelta(days=tol - 10)
    dates = pd.bdate_range(cached_start, cached_start + pd.Timedelta(days=200))
    _seed_cache(tmp_path, {"XOM": _mkdf(dates)})

    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one",
                        _fake_fetch(calls, {"XOM": _mkdf(pd.bdate_range(dates.max(),
                                                                        dates.max() + pd.Timedelta(days=5)))}))

    N.load_yahoo_panels(["XOM"], "2020-01-02", str((dates.max() + pd.Timedelta(days=5)).date()),
                        cache_dir=tmp_path, refresh=False, label="T")

    assert calls and calls[0][1] == dates.max(), (
        "within-tolerance staleness must extend forward, not trigger a full backfill")
