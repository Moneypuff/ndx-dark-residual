"""Point-in-time DIX history splice: the committed data/pit_dix_history.csv (2009+) is used for
dates up to its last row and the live build only after it; the comovement builders prefer the
long NDX series (payload key `ndxi`) over the live replica when it is present. No network."""
import pandas as pd
import pytest

import ndx_dark_residual as N


def test_committed_history_file_loads_and_covers_2009_onward():
    h = N.load_pit_dix_history()
    assert h is not None, "data/pit_dix_history.csv should be committed"
    assert {"ndx", "spx", "iwm"} <= set(h.columns)
    assert h.index.min() <= pd.Timestamp("2009-12-31")
    assert h.index.is_monotonic_increasing and not h.index.duplicated().any()
    v = h[["ndx", "spx", "iwm"]].stack()
    assert v.between(0.2, 0.8).all()                      # DIX is a short share of dark volume


def test_missing_history_file_means_live_only(tmp_path):
    assert N.load_pit_dix_history(tmp_path / "nope.csv") is None
    live = pd.Series([0.4, 0.5], index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
    assert N.splice_dix_history(live, None).equals(live)


def test_splice_uses_history_then_live_after_its_last_row():
    hist = pd.Series([0.40, 0.41, 0.42], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    live = pd.Series([0.99, 0.99, 0.50, 0.51],
                     index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]))
    out = N.splice_dix_history(live, hist, "T")
    assert list(out.index) == list(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]))
    assert out.loc["2020-01-06"] == pytest.approx(0.42)   # history wins on overlap
    assert out.loc["2020-01-07"] == pytest.approx(0.50)   # live only after the history's last row
    assert N.splice_dix_history(pd.Series(dtype=float), hist).equals(hist)


def test_splice_with_long_overlap_prints_diagnostic(capsys):
    # >60 overlapping days exercises the overlap diagnostic (a DataFrame attribute clash once crashed it)
    idx = pd.bdate_range("2020-01-01", periods=200)
    hist = pd.Series(0.45, index=idx[:150]) + pd.Series(range(150), index=idx[:150]) * 1e-4
    live = pd.Series(0.46, index=idx) + pd.Series(range(200), index=idx) * 1e-4
    out = N.splice_dix_history(live, hist, "TEST")
    assert "TEST DIX: PIT history" in capsys.readouterr().err
    assert len(out) == 200 and out.iloc[149] == pytest.approx(hist.iloc[149])


def test_comovement_builders_prefer_long_ndx_series():
    import build_comovement as BC
    d = ["2010-01-04", "2010-01-05"]
    P = {"ndxi": {"dates": d, "dix": [0.41, 0.42], "r21": [1.0, 2.0]},
         "rel": {"dates": ["2020-01-02"], "ndx_dix": [0.9], "r21": {"QQQ": [9.0]}}, "bench": "QQQ",
         "spx": {"dates": d, "dix": [0.4, 0.4], "r21": [0.1, 0.2]},
         "iwm": {"dates": d, "d": [0.5, 0.5], "r21": [0.3, 0.4]}}
    dix, ret = BC.dix_and_returns(P)
    assert dix["NDX"].index.min() == pd.Timestamp("2010-01-04") and ret["NDX"].iloc[1] == 2.0
    del P["ndxi"]
    dix, _ = BC.dix_and_returns(P)
    assert dix["NDX"].iloc[0] == pytest.approx(0.9)       # falls back to the live replica
