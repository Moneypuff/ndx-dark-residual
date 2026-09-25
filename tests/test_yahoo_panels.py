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

Also pinned: the re-basing guard. Yahoo re-bases a symbol's whole history on every
split (close/volume) and dividend (adjclose), so an incremental fetch that only
returns recent bars would leave older cached bars on a stale basis (a 2:1 split
would read as a -50% day). Incremental fetches re-request a few cached bars and
re-fetch the symbol in full when they no longer match; split events ride along
and are persisted so FINRA's as-traded volumes can be put on Yahoo's basis.
"""
from pathlib import Path

import pandas as pd
import pytest

import ndx_dark_residual as N


def _mkdf(dates, value=100.0):
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"close": value, "adjclose": value, "volume": 1000}, index=idx)


def _seed_cache(cache_dir, panels_by_symbol, synced=None, nodata=None, splits_known=None,
                splits=None):
    """Write a yahoo_prices.pkl whose wide frames hold exactly the given
    per-symbol date ranges, mirroring what load_yahoo_panels itself writes.
    `splits_known=None` writes a legacy cache (no recorded split history)."""
    fields = ("close", "adjclose", "volume")
    out = {}
    for f in fields:
        out[f] = pd.DataFrame({s: df[f] for s, df in panels_by_symbol.items()})
    payload = {**out, "_synced": synced or "", "_nodata": sorted(nodata or [])}
    if splits_known is not None:
        payload["_splits_known"] = sorted(splits_known)
        payload["_splits"] = splits or {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, cache_dir / N.YAHOO_CACHE)


def _fake_fetch(calls, series_by_symbol, splits_by_symbol=None):
    """A fetch_yahoo_one stand-in that records (sym, start) and returns a daily
    series for `sym` covering [start, end] from a pre-baked source series, with
    that window's split events attached the way the real fetch attaches them."""
    def _fetch(sym, start, end, session=None, **kw):
        calls.append((sym, pd.Timestamp(start)))
        src = series_by_symbol.get(sym)
        if src is None:
            return pd.DataFrame(columns=["close", "adjclose", "volume"])
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        win = src[(src.index >= lo) & (src.index <= hi)].copy()
        win.attrs["splits"] = {d: f for d, f in (splits_by_symbol or {}).get(sym, {}).items()
                               if lo <= d <= hi}
        return win
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
    _seed_cache(tmp_path, {"MSFT": _mkdf(dates)}, splits_known=["MSFT"])

    calls = []
    src = _mkdf(pd.bdate_range("2020-01-02", "2020-06-10"))     # same basis as the cache
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, {"MSFT": src}))

    N.load_yahoo_panels(["MSFT"], "2020-01-02", "2020-06-10",
                        cache_dir=tmp_path, refresh=False, label="T")

    # Forward from a short overlap of already-cached bars (the re-basing check), not
    # from `start`; the overlap matches, so there is no second (full) fetch.
    assert calls == [("MSFT", dates[-1 - N.YAHOO_OVERLAP_BARS])], (
        "a symbol within the backfill tolerance of `start` must only extend forward")


def test_symbol_within_tolerance_of_start_is_not_backfilled(tmp_path, monkeypatch):
    # A symbol whose cached start is within YAHOO_BACKFILL_TOL_DAYS of the requested
    # start is NOT truncated in the pathological sense -- must not force a full re-pull.
    tol = N.YAHOO_BACKFILL_TOL_DAYS
    cached_start = pd.Timestamp("2020-01-02") + pd.Timedelta(days=tol - 10)
    dates = pd.bdate_range(cached_start, cached_start + pd.Timedelta(days=200))
    _seed_cache(tmp_path, {"XOM": _mkdf(dates)}, splits_known=["XOM"])

    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one",
                        _fake_fetch(calls, {"XOM": _mkdf(pd.bdate_range(dates.min(),
                                                                        dates.max() + pd.Timedelta(days=5)))}))

    N.load_yahoo_panels(["XOM"], "2020-01-02", str((dates.max() + pd.Timedelta(days=5)).date()),
                        cache_dir=tmp_path, refresh=False, label="T")

    assert calls and calls[0][1] == dates[-1 - N.YAHOO_OVERLAP_BARS], (
        "within-tolerance staleness must extend forward, not trigger a full backfill")


# ---------------------------------------------------------------------------
# re-basing guard: a split / dividend after the symbol was cached re-bases its
# whole Yahoo history; the stale cached bars must be replaced, not merged.
# ---------------------------------------------------------------------------
def _rebased_source(dates, factor, from_date):
    """Yahoo's view AFTER a `factor`:1 split effective `from_date`: every bar, old or new,
    is on the post-split basis (price / factor)."""
    df = _mkdf(dates, value=100.0 / factor)
    return df


def test_split_after_caching_triggers_full_refetch_and_replaces_column(tmp_path, monkeypatch):
    cached = pd.bdate_range("2023-01-02", "2023-06-30")
    _seed_cache(tmp_path, {"MNST": _mkdf(cached, value=100.0)}, splits_known=["MNST"])
    split_day = pd.Timestamp("2023-07-05")
    src = _rebased_source(pd.bdate_range("2023-01-02", "2023-07-14"), 2.0, split_day)

    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one",
                        _fake_fetch(calls, {"MNST": src}, {"MNST": {split_day: 2.0}}))
    out = N.load_yahoo_panels(["MNST"], "2023-01-02", "2023-07-14",
                              cache_dir=tmp_path, refresh=False, label="T")

    # 1st call: incremental with overlap; 2nd: full re-fetch from the earliest cached bar.
    assert [c[0] for c in calls] == ["MNST", "MNST"]
    assert calls[0][1] == cached[-1 - N.YAHOO_OVERLAP_BARS]
    assert calls[1][1] == cached.min()
    close = out["close"]["MNST"].dropna()
    assert close.nunique() == 1 and close.iloc[0] == pytest.approx(50.0), (
        "the stale pre-split bars must be replaced -- no fake -50% step")
    assert out["splits"]["MNST"] == {split_day: 2.0}
    reread = pd.read_pickle(tmp_path / N.YAHOO_CACHE)
    assert reread["close"]["MNST"].dropna().eq(50.0).all()
    assert reread["_splits"]["MNST"] == {split_day: 2.0}
    assert "MNST" in reread["_splits_known"]


def test_dividend_rebase_of_adjclose_triggers_full_refetch(tmp_path, monkeypatch):
    cached = pd.bdate_range("2023-01-02", "2023-03-31")
    _seed_cache(tmp_path, {"KO": _mkdf(cached, value=60.0)}, splits_known=["KO"])
    src = _mkdf(pd.bdate_range("2023-01-02", "2023-04-10"), value=60.0)
    src["adjclose"] = 60.0 * 0.992          # an ex-dividend re-adjusted every earlier adjclose
    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, {"KO": src}))
    out = N.load_yahoo_panels(["KO"], "2023-01-02", "2023-04-10",
                              cache_dir=tmp_path, refresh=False, label="T")
    assert len(calls) == 2 and calls[1][1] == cached.min()
    assert out["adjclose"]["KO"].dropna().eq(60.0 * 0.992).all()


def test_float_noise_in_overlap_is_not_a_rebase(tmp_path, monkeypatch):
    # Yahoo's adjclose wobbles ~1e-7 between identical requests -- must not trigger re-fetches.
    cached = pd.bdate_range("2023-01-02", "2023-03-31")
    _seed_cache(tmp_path, {"PEP": _mkdf(cached, value=170.0)}, splits_known=["PEP"])
    src = _mkdf(pd.bdate_range("2023-01-02", "2023-04-10"), value=170.0)
    src["adjclose"] = 170.0 * (1 + 2e-7)
    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, {"PEP": src}))
    N.load_yahoo_panels(["PEP"], "2023-01-02", "2023-04-10",
                        cache_dir=tmp_path, refresh=False, label="T")
    assert len(calls) == 1, "sub-tolerance noise must stay an ordinary incremental fetch"


def test_unknown_split_event_in_increment_triggers_full_refetch(tmp_path, monkeypatch):
    # Belt and braces: a split event the cache never recorded forces a rebuild even if the
    # overlapping prices happen to match.
    cached = pd.bdate_range("2023-01-02", "2023-03-31")
    _seed_cache(tmp_path, {"APH": _mkdf(cached)}, splits_known=["APH"])
    src = _mkdf(pd.bdate_range("2023-01-02", "2023-04-10"))
    ev = pd.Timestamp("2023-04-05")
    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, {"APH": src}, {"APH": {ev: 2.0}}))
    out = N.load_yahoo_panels(["APH"], "2023-01-02", "2023-04-10",
                              cache_dir=tmp_path, refresh=False, label="T")
    assert len(calls) == 2
    assert out["splits"]["APH"] == {ev: 2.0}


def test_failed_rebuild_keeps_consistent_cache_instead_of_mixing_bases(tmp_path, monkeypatch):
    cached = pd.bdate_range("2023-01-02", "2023-06-30")
    _seed_cache(tmp_path, {"CRWD": _mkdf(cached, value=100.0)}, splits_known=["CRWD"])
    src = _mkdf(pd.bdate_range("2023-01-02", "2023-07-14"), value=25.0)   # re-based 4:1
    calls = []
    real = _fake_fetch(calls, {"CRWD": src})

    def flaky(sym, start, end, session=None, **kw):
        if calls:                                # the full re-fetch fails (throttled)
            calls.append((sym, pd.Timestamp(start)))
            return pd.DataFrame(columns=["close", "adjclose", "volume"])
        return real(sym, start, end, session=session)
    monkeypatch.setattr(N, "fetch_yahoo_one", flaky)
    out = N.load_yahoo_panels(["CRWD"], "2023-01-02", "2023-07-14",
                              cache_dir=tmp_path, refresh=False, label="T")
    close = out["close"]["CRWD"].dropna()
    assert close.eq(100.0).all(), "no post-split bars may be spliced onto the stale basis"
    assert close.index.max() == cached.max()


def test_legacy_cache_without_split_history_is_rebuilt_once(tmp_path, monkeypatch):
    # A cache written before split events were tracked gets one full re-fetch -- from the
    # EARLIEST cached bar -- which records the symbol's splits and marks it known.
    cached = pd.bdate_range("2019-01-02", "2023-03-31")
    _seed_cache(tmp_path, {"NVDA": _mkdf(cached)})                         # legacy: no _splits_known
    ev = pd.Timestamp("2021-07-20")
    src = _mkdf(pd.bdate_range("2019-01-02", "2023-04-10"))
    calls = []
    monkeypatch.setattr(N, "fetch_yahoo_one", _fake_fetch(calls, {"NVDA": src}, {"NVDA": {ev: 4.0}}))
    out = N.load_yahoo_panels(["NVDA"], "2020-01-02", "2023-04-10",
                              cache_dir=tmp_path, refresh=False, label="T")
    assert calls == [("NVDA", cached.min())], "rebuild must reach back to the earliest cached bar"
    assert out["splits"] == {"NVDA": {ev: 4.0}}
    reread = pd.read_pickle(tmp_path / N.YAHOO_CACHE)
    assert "NVDA" in reread["_splits_known"]
    assert reread["close"]["NVDA"].dropna().index.min() == cached.min(), (
        "the deeper history another caller cached must survive the rebuild")

    # ...and only once: the next run is an ordinary incremental fetch.
    calls.clear()
    N.load_yahoo_panels(["NVDA"], "2020-01-02", "2023-04-14",
                        cache_dir=tmp_path, refresh=False, label="T")
    assert len(calls) == 1 and calls[0][1] > cached.min()


def test_parse_yahoo_splits_ratios_and_malformed_rows():
    res = {"events": {"splits": {
        "a": {"date": 1718026200, "numerator": 10.0, "denominator": 1.0},     # NVDA 10:1
        "b": {"date": 1690000000, "numerator": 1, "denominator": 10},        # 1:10 reverse
        "c": {"date": 1600000000, "numerator": 2},                           # malformed
        "d": {"date": 1500000000, "numerator": 1, "denominator": 1},         # no-op ratio
    }}}
    out = N._parse_yahoo_splits(res)
    assert out[pd.Timestamp("2024-06-10")] == 10.0
    assert out[pd.to_datetime(1690000000, unit="s").normalize()] == pytest.approx(0.1)
    assert len(out) == 2
    assert N._parse_yahoo_splits({}) == {} and N._parse_yahoo_splits({"events": None}) == {}
