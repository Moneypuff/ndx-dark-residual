"""Dollar-DIX price basis: FINRA volume is as-traded, so the price it is multiplied by must be the
as-traded (raw) close, not Yahoo's split-adjusted close. Pins:
  * _raw_close_from_splits undoes forward and reverse splits (and ignores malformed events),
  * fetch_yahoo_one derives `rawclose` from the chart response's split events,
  * load_yahoo_panels re-fetches, once and in full, any cached symbol that predates `rawclose`,
  * compute_dollar_dix is invariant to a split when fed raw prices (and is not when fed adjusted ones).
No network.
"""
import pandas as pd
import pytest

import ndx_dark_residual as N


def _ts(d):
    return int(pd.Timestamp(d).timestamp())


def test_forward_split_multiplies_pre_split_bars():
    idx = pd.to_datetime(["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11"])
    close = pd.Series([120.0, 120.9, 121.8, 120.9], index=idx)              # split-adjusted
    raw = N._raw_close_from_splits(close, {"x": {"date": _ts("2024-06-10"), "numerator": 10, "denominator": 1}})
    assert raw.loc["2024-06-07"] == pytest.approx(1209.0)
    assert raw.loc["2024-06-10"] == pytest.approx(121.8)                    # ex-date and after unchanged


def test_reverse_split_divides_pre_split_bars():
    idx = pd.to_datetime(["2024-09-06", "2024-09-09", "2024-09-10"])
    close = pd.Series([27.3, 26.7, 26.0], index=idx)
    raw = N._raw_close_from_splits(close, {"x": {"date": _ts("2024-09-10"), "numerator": 1, "denominator": 10}})
    assert raw.loc["2024-09-06"] == pytest.approx(2.73)
    assert raw.loc["2024-09-10"] == pytest.approx(26.0)


def test_multiple_splits_compound_and_bad_events_are_ignored():
    idx = pd.to_datetime(["2020-01-02", "2021-01-04", "2022-01-03"])
    close = pd.Series([10.0, 10.0, 10.0], index=idx)
    ev = {"a": {"date": _ts("2021-01-04"), "numerator": 2, "denominator": 1},
          "b": {"date": _ts("2022-01-03"), "numerator": 3, "denominator": 1},
          "bad1": {"numerator": 5, "denominator": 1},                        # no date
          "bad2": {"date": _ts("2021-06-01"), "numerator": 0, "denominator": 1},
          "bad3": None, "bad4": "2:1"}                                       # non-mapping entries
    raw = N._raw_close_from_splits(close, ev)
    assert list(raw) == pytest.approx([60.0, 30.0, 10.0])
    assert N._raw_close_from_splits(close, None).equals(close.astype("float64"))


class _Resp:
    status_code = 200
    def __init__(self, j): self._j = j
    def json(self): return self._j


class _Session:
    def __init__(self, j): self.j, self.urls = j, []
    def get(self, url, **kw):
        self.urls.append(url); return _Resp(self.j)


def test_fetch_yahoo_one_derives_rawclose_from_split_events():
    days = pd.to_datetime(["2022-07-20", "2022-07-21", "2022-07-22"])
    j = {"chart": {"error": None, "result": [{
        "timestamp": [_ts(d) + 14 * 3600 for d in days],
        "indicators": {"quote": [{"close": [38.0, 38.37, 35.78], "volume": [1, 2, 3]}],
                       "adjclose": [{"adjclose": [38.0, 38.37, 35.78]}]},
        "events": {"splits": {"s": {"date": _ts("2022-07-22") + 14 * 3600 - 14 * 3600, "numerator": 4, "denominator": 1}}}}]}}
    sess = _Session(j)
    df = N.fetch_yahoo_one("GME", pd.Timestamp("2022-07-20"), pd.Timestamp("2022-07-22"), session=sess)
    assert "events=split" in sess.urls[0]
    assert df.loc["2022-07-21", "rawclose"] == pytest.approx(153.48)
    assert df.loc["2022-07-22", "rawclose"] == pytest.approx(35.78)
    assert df.loc["2022-07-21", "close"] == pytest.approx(38.37)          # adjusted series untouched


def test_cache_without_rawclose_triggers_one_full_refetch(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", "2024-03-01")
    old = pd.DataFrame({"close": 100.0, "adjclose": 100.0, "volume": 1000}, index=dates)   # pre-rawclose cache
    pd.to_pickle({"close": pd.DataFrame({"AAA": old.close}), "adjclose": pd.DataFrame({"AAA": old.adjclose}),
                  "volume": pd.DataFrame({"AAA": old.volume}), "_synced": "", "_nodata": []},
                 tmp_path / N.YAHOO_CACHE)
    calls = []
    def fake(sym, start, end, session=None, **kw):
        calls.append((sym, pd.Timestamp(start)))
        w = old[(old.index >= pd.Timestamp(start)) & (old.index <= pd.Timestamp(end))].copy()
        w["rawclose"] = w.close
        return w
    monkeypatch.setattr(N, "fetch_yahoo_one", fake)
    out = N.load_yahoo_panels(["AAA"], "2024-01-02", "2024-03-01", workers=1, cache_dir=tmp_path)
    assert calls == [("AAA", pd.Timestamp("2024-01-02"))]                    # full window, not forward-only
    assert out["rawclose"]["AAA"].notna().all()
    calls.clear()
    N.load_yahoo_panels(["AAA"], "2024-01-02", "2024-03-01", workers=1, cache_dir=tmp_path)
    assert calls == []                                                       # migrated: now current


def test_dollar_dix_is_split_invariant_only_with_raw_prices():
    days = pd.bdate_range("2024-01-02", periods=40)
    split_day = days[20]
    # two names, identical dark flow in dollars; BBB does a 1-for-10 reverse split on split_day
    short = pd.DataFrame({"AAA": 500.0, "BBB": 5000.0}, index=days)
    total = pd.DataFrame({"AAA": 1000.0, "BBB": 10000.0}, index=days)
    short.loc[days >= split_day, "BBB"] = 500.0; total.loc[days >= split_day, "BBB"] = 1000.0
    short["AAA"] = 400.0                                                     # AAA DPI 0.4, BBB DPI 0.5
    raw = pd.DataFrame({"AAA": 10.0, "BBB": 1.0}, index=days); raw.loc[days >= split_day, "BBB"] = 10.0
    adj = raw.copy(); adj["BBB"] = 10.0                                      # split-adjusted: flat 10
    dix_raw = N.compute_dollar_dix(short, total, raw, min_names=2)
    dix_adj = N.compute_dollar_dix(short, total, adj, min_names=2)
    assert dix_raw.dropna().round(10).nunique() == 1                         # 0.45 throughout
    assert dix_raw.dropna().iloc[0] == pytest.approx(0.45)
    assert dix_adj.dropna().round(10).nunique() > 1                          # pre-split BBB over-weighted 10x
