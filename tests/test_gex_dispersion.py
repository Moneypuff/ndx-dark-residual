"""Tests for the pure pieces of ``build_gex_dispersion``.

The barometer rests on three hand-rolled computations -- the Cboe-style
basket-identity realized correlation, the dispersion analog, and the
no-look-ahead percentile/forward-outcome plumbing -- plus two CSV parsers for
external documents whose shapes we don't control. A bug in any of them would
silently corrupt the published page rather than crash, so each is pinned here
against either a direct from-definition recomputation or a hand-built
document.
"""
import json

import numpy as np
import pandas as pd
import pytest

import build_gex_dispersion as B


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------
SQUEEZE_BODY = (
    "date,price,dix,gex\n"
    "2020-01-02,3257.85,0.412,5.2e9\n"
    "2020-01-03,3234.85,0.401,4.9e9\n"
    "2020-01-06,3246.28,0.399,4.7e9\n"
)


def test_parse_squeeze_happy_path():
    df = B.parse_squeeze_csv(SQUEEZE_BODY)
    assert list(df.columns) == ["price", "dix", "gex"]
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2020-01-02")
    assert df["gex"].iloc[0] == pytest.approx(5.2e9)


def test_parse_squeeze_rejects_html_and_wrong_columns():
    assert B.parse_squeeze_csv("<html><body>gate</body></html>").empty
    assert B.parse_squeeze_csv("date,foo\n2020-01-02,1\n").empty
    assert B.parse_squeeze_csv("").empty


GEXPLUS_BODY = (
    "DATE,SPX,CHG(%),GEX,VEX,GEX+,GIV,GIV(MAD%),NPD,VGR,VIX,VIX(MAD%),CR(x),"
    "SU,MID,MO,DIX,RISC,VIBE,OPEN,HIGH,LOW,CLOSE\n"
    "2004-01-02,1108.48,1.24,64521,72414,136935,16.07,0.81,-2.44,-2.22,18.22,"
    "0.92,0.84,1103.95,1110.74,1117.54,,,,1111.92,1118.85,1105.08,1108.48\n"
    "2004-01-05,1122.22,0.13,82869,73166,156035,15.56,0.78,-8.82,-1.91,17.49,"
    "0.88,0.84,1117.95,1124.35,1130.76,,,,1108.48,1122.22,1108.48,1122.22\n"
)


def test_parse_gexplus_takes_gexplus_column_in_public_dollar_units():
    df = B.parse_gexplus_csv(GEXPLUS_BODY)
    assert list(df.columns) == ["price", "dix", "gex", "high", "low"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2004-01-02")
    assert df["price"].iloc[0] == pytest.approx(1108.48)     # CLOSE, not SPX/OPEN
    assert df["gex"].iloc[0] == pytest.approx(136935 * B.GEXPLUS_DOLLARS)
    assert df["dix"].isna().all()                            # empty pre-2011 DIX cells
    # SPX intraday high/low ride along for the range stats (row 1: 1118.85/1105.08)
    assert df["high"].iloc[0] == pytest.approx(1118.85)
    assert df["low"].iloc[0] == pytest.approx(1105.08)


def test_parse_gexplus_without_ohlc_columns_leaves_high_low_nan():
    body = ("DATE,GEX+,CLOSE\n"
            "2020-01-02,5000,3257.85\n"
            "2020-01-03,4800,3234.85\n")
    df = B.parse_gexplus_csv(body)
    assert list(df.columns) == ["price", "dix", "gex", "high", "low"]
    assert df["high"].isna().all() and df["low"].isna().all()


def test_parse_gexplus_rejects_html_and_missing_columns():
    assert B.parse_gexplus_csv("<html>bad key</html>").empty
    assert B.parse_gexplus_csv("DATE,GEX,CLOSE\n2020-01-02,1,2\n").empty  # no GEX+
    assert B.parse_gexplus_csv("").empty


COR1M_BODY = (
    "DATE,OPEN,HIGH,LOW,CLOSE\n"
    "01/03/2006,23.50,24.00,23.10,23.80\n"
    "01/04/2006,24.33,24.40,24.00,24.10\n"
)
DSPX_BODY = (
    "DATE,DSPX\n"
    "06/19/2014,17.81\n"
    "06/20/2014,17.37\n"
)


def test_parse_cboe_ohlc_takes_close():
    s = B.parse_cboe_csv(COR1M_BODY)
    assert len(s) == 2
    assert s.index[0] == pd.Timestamp("2006-01-03")
    assert s.iloc[0] == pytest.approx(23.80)   # CLOSE, not OPEN


def test_parse_cboe_single_value_column():
    s = B.parse_cboe_csv(DSPX_BODY)
    assert len(s) == 2
    assert s.iloc[1] == pytest.approx(17.37)


def test_parse_cboe_garbage_returns_empty():
    assert B.parse_cboe_csv("<html>nope</html>").empty
    assert B.parse_cboe_csv("A,B\n1,2\n").empty          # no DATE column
    assert B.parse_cboe_csv("").empty


def test_parse_cboe_skips_malformed_rows():
    s = B.parse_cboe_csv("DATE,DSPX\n06/19/2014,17.81\nnot-a-date,9.9\n06/20/2014,\n")
    assert len(s) == 1


# ---------------------------------------------------------------------------
# parse_yahoo_ohlc -- the ^GSPC high/low fallback for the intraday range
# ---------------------------------------------------------------------------
def _yahoo_body(highs, lows, closes, ts=(1072972800, 1073232000)):
    return json.dumps({"chart": {"result": [{
        "timestamp": list(ts),
        "indicators": {"quote": [{"high": highs, "low": lows, "close": closes}]},
    }], "error": None}})


def test_parse_yahoo_ohlc_extracts_high_low_close():
    df = B.parse_yahoo_ohlc(_yahoo_body([1118.85, 1122.22],
                                        [1105.08, 1108.48],
                                        [1108.48, 1122.22]))
    assert list(df.columns) == ["high", "low", "close"]
    assert len(df) == 2
    assert df["high"].iloc[0] == pytest.approx(1118.85)
    assert df["low"].iloc[1] == pytest.approx(1108.48)


def test_parse_yahoo_ohlc_drops_all_nan_rows_and_handles_errors():
    df = B.parse_yahoo_ohlc(_yahoo_body([1118.85, None], [1105.08, None],
                                        [1108.48, None]))
    assert len(df) == 1                                    # the all-None day dropped
    assert B.parse_yahoo_ohlc("<html>gate</html>").empty
    assert B.parse_yahoo_ohlc(json.dumps({"chart": {"result": None,
                                                     "error": "x"}})).empty
    assert B.parse_yahoo_ohlc("").empty


# ---------------------------------------------------------------------------
# top_weights
# ---------------------------------------------------------------------------
def test_top_weights_orders_normalizes_and_reports_coverage():
    w, cov = B.top_weights(["AAA", "BBB", "CCC", "DDD"],
                           {"AAA": 5.0, "BBB": 7.0, "CCC": 2.0, "DDD": 1.0}, n=2)
    assert list(w.index) == ["BBB", "AAA"]
    assert w.sum() == pytest.approx(1.0)
    assert w["BBB"] == pytest.approx(7 / 12)
    assert cov == pytest.approx(12.0)


def test_top_weights_skips_missing_and_handles_empty():
    w, cov = B.top_weights(["AAA", "BBB"], {"AAA": 3.0}, n=5)
    assert list(w.index) == ["AAA"]
    w, cov = B.top_weights(["AAA"], {}, n=5)
    assert w.empty and cov == 0.0


# ---------------------------------------------------------------------------
# realized_cor_disp -- pinned against the from-definition covariance identity
# ---------------------------------------------------------------------------
def _panel(n_days=80, n_names=4, seed=3, rho=0.5):
    """Correlated daily return panel: common factor + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n_days)
    idio = rng.normal(0, 0.01, (n_days, n_names))
    lam = np.sqrt(rho / (1 - rho))
    X = lam * common[:, None] + idio
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(X, index=dates, columns=[f"S{i}" for i in range(n_names)])


def test_realized_cor_matches_direct_covariance_identity():
    R = _panel()
    w = pd.Series(0.25, index=R.columns)
    out = B.realized_cor_disp(R, w, window=21, min_names=3)
    p = 40                                   # any position with a full window
    win = R.iloc[p - 20:p + 1]
    C = np.cov(win.to_numpy().T, ddof=1)
    wv = np.full(4, 0.25)
    port_var = wv @ C @ wv
    sig = np.sqrt(np.diag(C))
    num = port_var - np.sum(wv ** 2 * sig ** 2)
    den = np.sum(wv * sig) ** 2 - np.sum(wv ** 2 * sig ** 2)
    assert out["cor"].iloc[p] == pytest.approx(100 * num / den, rel=1e-9)


def test_realized_disp_matches_direct_covariance_identity():
    R = _panel(seed=11)
    w = pd.Series(0.25, index=R.columns)
    out = B.realized_cor_disp(R, w, window=21, min_names=3)
    p = 55
    win = R.iloc[p - 20:p + 1]
    C = np.cov(win.to_numpy().T, ddof=1)
    wv = np.full(4, 0.25)
    expected = 100 * np.sqrt(252 * (np.sum(wv * np.diag(C)) - wv @ C @ wv))
    assert out["disp"].iloc[p] == pytest.approx(expected, rel=1e-9)


def test_realized_cor_recovers_factor_correlation_roughly():
    # long sample, strong common factor -> the estimate should sit near truth
    R = _panel(n_days=600, n_names=8, seed=7, rho=0.6)
    w = pd.Series(1 / 8, index=R.columns)
    out = B.realized_cor_disp(R, w, window=21, min_names=6)
    med = out["cor"].dropna().median()
    assert 40 < med < 80


def test_realized_cor_undefined_before_window_and_masked_below_min_names():
    R = _panel()
    R.iloc[:, 2:] = np.nan                    # only 2 of 4 names have data
    w = pd.Series(0.25, index=R.columns)
    out = B.realized_cor_disp(R, w, window=21, min_names=3)
    assert out["cor"].isna().all()            # 2 < min_names everywhere
    out2 = B.realized_cor_disp(_panel(), w, window=21, min_names=3)
    assert out2["cor"].iloc[:20].isna().all()  # vol window not yet full
    assert out2["cor"].iloc[20:].notna().all()


def test_realized_cor_ignores_weight_names_missing_from_panel():
    R = _panel()
    w = pd.Series(0.2, index=list(R.columns) + ["GHOST"])
    out = B.realized_cor_disp(R, w, window=21, min_names=3)
    assert out["cor"].iloc[30:].notna().all()


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------
def test_rolling_pct_no_lookahead_and_rank_semantics():
    s = pd.Series(np.arange(30, dtype=float))
    p = B.rolling_pct(s, window=10, min_periods=10)
    assert p.iloc[:9].isna().all()
    # strictly increasing input: every defined day is the max of its window
    assert (p.dropna() == 95.0).all()          # (9 below + 0.5*1 equal)/10 = 95%


def test_rolling_pct_mid_rank_on_ties():
    s = pd.Series([1.0, 1.0, 1.0, 1.0])
    p = B.rolling_pct(s, window=4, min_periods=4)
    assert p.iloc[-1] == pytest.approx(50.0)


def test_full_pct_bounds():
    s = pd.Series([3.0, 1.0, 2.0])
    p = B.full_pct(s)
    assert p.idxmax() == 0 and p.idxmin() == 1
    assert p.max() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# forward outcomes -- strictly after t
# ---------------------------------------------------------------------------
def test_forward_return_alignment():
    px = pd.Series([100.0, 101, 102, 103, 104, 105])
    r = B.forward_return(px, horizon=2)
    assert r.iloc[0] == pytest.approx(2.0)     # 100 -> 102
    assert r.iloc[-2:].isna().all()


def test_forward_vol_uses_only_future_returns():
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 50)))
    fv = B.forward_vol(px, horizon=5)
    t = 10
    fut = px.pct_change().iloc[t + 1:t + 6]    # sessions t+1 .. t+5
    assert fv.iloc[t] == pytest.approx(fut.std() * np.sqrt(252) * 100, rel=1e-9)
    assert fv.iloc[-5:].isna().all()
    # changing the past must not change the forward vol at t
    px2 = px.copy(); px2.iloc[:t - 3] *= 1.5
    assert B.forward_vol(px2, horizon=5).iloc[t] == pytest.approx(fv.iloc[t], rel=1e-9)


# ---------------------------------------------------------------------------
# quadrants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("g,c,expected", [
    (80, 80, "HH"), (80, 20, "HL"), (20, 80, "LH"), (20, 20, "LL"),
    (50, 50, "HH"),                    # boundary counts as High
])
def test_quad_code(g, c, expected):
    assert B.quad_code(g, c) == expected


def test_quad_code_undefined_dials():
    assert B.quad_code(np.nan, 50) == ""
    assert B.quad_code(50, np.nan) == ""


def test_quad_stats_aggregates_and_sorts_by_frequency():
    df = pd.DataFrame({
        "quad": ["HL", "HL", "HL", "LH", "LH", ""],
        "r21":  [1.0, 2.0, -1.0, -4.0, 2.0, 99.0],
        "fvol": [10.0, 12.0, 11.0, 30.0, 34.0, 99.0],
        "tvol": [9.0, 12.0, 13.0, 20.0, 24.0, 0.0],
    })
    rows = B.quad_stats(df)
    assert [r[0] for r in rows] == ["HL", "LH"]      # "" excluded, sorted by n
    hl = rows[0]
    assert hl[2] == 3
    assert hl[3] == pytest.approx(0.67, abs=0.01)    # mean of 1,2,-1
    assert hl[5] == 67                               # hit: 2 of 3 positive
    assert hl[6] == pytest.approx(11.0)
    assert hl[7] == pytest.approx(-0.3, abs=0.05)    # mean fvol - tvol
    assert rows[1][1] == B.QUAD_LABEL["LH"]


# ---------------------------------------------------------------------------
# quartile_bin + cell_stats -- the 4x4 intraday-range matrix
# ---------------------------------------------------------------------------
def test_quartile_bin_fixed_cuts_and_preserves_nan():
    b = B.quartile_bin(pd.Series([0.0, 24.9, 25.0, 50.0, 74.9, 75.0, 100.0, np.nan]))
    assert list(b.iloc[:7]) == [0, 0, 1, 2, 2, 3, 3]     # 25/50/75 cuts, top clipped
    assert pd.isna(b.iloc[-1])


def test_hl_stats_mean_mad_median_p90():
    st = B._hl_stats(pd.Series([1.0, 3.0]))
    assert st["n"] == 2
    assert st["mean"] == pytest.approx(2.0)
    assert st["mad"] == pytest.approx(1.0)               # mean(|1-2|,|3-2|)
    assert st["med"] == pytest.approx(2.0)
    assert st["p90"] == pytest.approx(2.8)               # linear-interp 90th pct
    empty = B._hl_stats(pd.Series([], dtype="float64"))
    assert empty == {"n": 0, "mean": None, "mad": None, "med": None, "p90": None}


def test_cell_stats_bins_and_aggregates_full_16_grid():
    # gexp/corp chosen to land in known quartiles; hlpct hand-set per cell
    df = pd.DataFrame({
        "gexp":  [10.0, 10.0, 90.0, 90.0, 90.0],   # gq: 0,0,3,3,3
        "corp":  [10.0, 10.0, 10.0, 10.0, 90.0],   # cq: 0,0,0,0,3
        "hlpct": [1.0,  3.0,  2.0,  4.0,  10.0],
    })
    df["gq"] = B.quartile_bin(df["gexp"])
    df["cq"] = B.quartile_bin(df["corp"])
    grid = B.cell_stats(df)
    assert len(grid) == 16                              # every cell present
    by = {(c["gq"], c["cq"]): c for c in grid}
    assert by[(0, 0)]["n"] == 2 and by[(0, 0)]["mean"] == pytest.approx(2.0)
    assert by[(0, 0)]["mad"] == pytest.approx(1.0)
    assert by[(3, 0)]["n"] == 2 and by[(3, 0)]["mean"] == pytest.approx(3.0)
    assert by[(3, 3)]["n"] == 1 and by[(3, 3)]["mean"] == pytest.approx(10.0)
    assert by[(1, 1)]["n"] == 0 and by[(1, 1)]["mean"] is None    # empty cell


def test_cell_stats_ignores_rows_missing_bin_or_range():
    df = pd.DataFrame({
        "gq":    [0.0, np.nan, 0.0, 0.0],
        "cq":    [0.0, 0.0,    np.nan, 0.0],
        "hlpct": [5.0, 5.0,    5.0, np.nan],
    })
    grid = B.cell_stats(df)
    by = {(c["gq"], c["cq"]): c for c in grid}
    assert by[(0, 0)]["n"] == 1                          # only the fully-defined row


def test_cell_stats_reads_named_column():
    # the same range values under an arbitrary column name (e.g. next-day hl1)
    df = pd.DataFrame({
        "gq":  [0.0, 0.0, 3.0],
        "cq":  [0.0, 0.0, 3.0],
        "hl1": [1.0, 3.0, 9.0],
    })
    by = {(c["gq"], c["cq"]): c for c in B.cell_stats(df, "hl1")}
    assert by[(0, 0)]["n"] == 2 and by[(0, 0)]["mean"] == pytest.approx(2.0)
    assert by[(3, 3)]["n"] == 1 and by[(3, 3)]["mean"] == pytest.approx(9.0)


def test_next_day_range_is_the_following_sessions_range():
    # the build's tradeable basis: session t's cell scored by session t+1's range.
    # hl1 = hl0.shift(-1) puts tomorrow's travel on today's row; last row -> NaN.
    hl0 = pd.Series([1.0, 2.0, 3.0, 4.0])
    hl1 = hl0.shift(-1)
    assert list(hl1.iloc[:3]) == [2.0, 3.0, 4.0]
    assert pd.isna(hl1.iloc[-1])


# ---------------------------------------------------------------------------
# series_corr
# ---------------------------------------------------------------------------
def test_series_corr_levels_and_changes():
    idx = pd.bdate_range("2023-01-02", periods=200)
    rng = np.random.default_rng(5)
    a = pd.Series(np.cumsum(rng.normal(0, 1, 200)), index=idx)
    b = a + rng.normal(0, 0.1, 200)
    r, n = B.series_corr(a, b)
    assert n == 200 and r > 0.99
    r2, _ = B.series_corr(a, b, diff=21)
    assert r2 > 0.9


def test_series_corr_insufficient_overlap():
    idx = pd.bdate_range("2023-01-02", periods=10)
    a = pd.Series(range(10), index=idx, dtype=float)
    r, n = B.series_corr(a, a)
    assert r is None and n == 10
