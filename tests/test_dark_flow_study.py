"""Offline smoke/regression tests for ``dark_flow_study``.

The study is orchestration over ``dark_flow`` and the repo's data layer; these tests
drive every report section with small synthetic panels shaped like
``build_universe_panels`` output (no network), so a refactor that breaks the
pipeline -- a renamed panel key, a misaligned control, a section that silently
drops a signal -- fails here rather than halfway through a 10-minute live run.
"""
import numpy as np
import pandas as pd
import pytest

import dark_flow_study as S


def _synthetic_panels(n_names=140, n_days=460, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-08-01", periods=n_days)
    cols = [f"N{i:03d}" for i in range(n_names)]
    ret = rng.normal(0, 0.015, (n_days, n_names))
    adj = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    vol = pd.DataFrame(rng.lognormal(14, 0.4, (n_days, n_names)), index=idx, columns=cols)
    share = np.clip(rng.normal(0.45, 0.05, (n_days, n_names)), 0.1, 0.9)
    total = vol * share
    dpi = np.clip(0.45 + rng.normal(0, 0.08, (n_days, n_names)), 0.05, 0.95)
    short = total * dpi
    P = {"short": short, "total": total, "short_raw": short, "total_raw": total,
         "close": adj, "adjclose": adj, "volume": vol, "splits": {}}
    return P, cols, idx


def test_dix_validation_scores_each_gauge_against_published_dix():
    idx = pd.bdate_range("2024-01-01", periods=6)
    sqz = pd.DataFrame({"dix": [0.40, 0.42, 0.41, 0.43, 0.44, 0.42]}, index=idx)
    fixed = sqz["dix"] + 0.001
    naive = sqz["dix"] * 0.5 + 0.18            # right direction, wrong weights
    out = S.dix_validation(fixed, naive, sqz)
    (mae_old, dc_old, n_old), (mae_new, dc_new, n_new) = out.values()
    assert mae_new == pytest.approx(0.001)
    assert dc_new == pytest.approx(1.0)
    assert mae_old == pytest.approx((sqz["dix"] - naive).abs().mean())
    assert dc_old == pytest.approx(1.0)           # a scaled copy still moves in lockstep
    assert n_old == n_new == 6


def test_study_sections_run_offline_on_synthetic_panels():
    P, names, idx = _synthetic_panels()
    sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, vols = S.build_signals(P, names)
    assert set(sig) == {k for k, _, _ in S.SIGNALS}
    assert set(ctl) == set(S.CONTROLS)
    dates = idx[idx >= idx[260]][::5]
    lines = []
    idx_res = S.section_index(lines, dix, adj[names[0]], "TEST", None)
    assert set(idx_res["signal"]) == {"level", "detrended"}
    dec, prof = S.section_structure(lines, dpi1, ret_1d)
    assert dec["share_idio"] > 0.5                     # iid synthetic DPI is all idiosyncratic
    assert set(prof) == {-5, -2, -1, 0, 1, 2, 5}
    res = S.section_signals(lines, sig, ctl, adj, dates, min_names=50)
    assert list(res["signal"]) == [k for k, _, _ in S.SIGNALS]
    assert res[["h5_uni_coef", "h21_ctrl_coef", "q5q1_h5_mean"]].notna().all().all()
    # pure-noise synthetic data: no signal should look significant
    assert (res["h21_ctrl_t"].abs() < 4).all()
    S.section_robust(lines, sig, ctl, vols, adj, dates, min_names=50)
    V = S.section_vol(lines, sig, ctl, adj, dates, min_names=50)
    assert set(V["horizon"]) == set(S.HORIZONS)
    text = "\n".join(lines)
    for hdr in ("1. INDEX LEVEL", "2. STRUCTURE", "3. SIGNALS", "3b. ROBUSTNESS", "4. VOLATILITY"):
        assert hdr in text
    assert "window 20 sessions" in text and "tails of dpi_z" in text


def test_section_ats_runs_on_synthetic_otc_weeks():
    P, names, idx = _synthetic_panels(n_names=120, n_days=460, seed=1)
    _, ctl, *_ = S.build_signals(P, names)
    rng = np.random.default_rng(2)
    weeks = pd.date_range(idx[0], idx[-40], freq="W-MON")
    rows = []
    for w in weeks:
        pub = w + pd.Timedelta(days=21)
        for s in names:
            rows.append((s, "ATS_W_SMBL", w, float(rng.lognormal(12, 0.3)), pub))
            rows.append((s, "OTC_W_SMBL", w, float(rng.lognormal(12.5, 0.3)), pub))
    A = pd.DataFrame(rows, columns=["sym", "type", "week", "shares", "published"])
    lines = []
    out = S.section_ats(lines, A, P, names, ctl, min_names=50)
    assert set(out["signal"]) == {"ats_share_z", "ats_of_volume_z", "ats_vol_surprise"}
    assert out["coef"].notna().any()
    assert "5. ATS SPLIT" in lines[0]


def test_fetch_otc_weekly_reuses_fresh_cache_with_numeric_dtypes(tmp_path, monkeypatch):
    # Every requested symbol is cached and fresh -> no API call, and the frame handed to
    # section_ats keeps numeric shares (concat with an empty fetch once upcast them to object).
    today = pd.Timestamp.today().normalize()
    cached = pd.DataFrame({"sym": ["AAA", "AAA"], "type": ["ATS_W_SMBL", "OTC_W_SMBL"],
                           "week": [today - pd.Timedelta(days=21)] * 2, "shares": [1.5e6, 4.0e6],
                           "published": [today - pd.Timedelta(days=2)] * 2})
    cached.to_csv(tmp_path / S.OTC_CACHE, index=False)

    def no_network(*a, **k):
        raise AssertionError("fresh cache must not hit the API")
    monkeypatch.setattr(S.N, "make_session", lambda *a, **k: type("X", (), {"post": no_network})())
    out = S.fetch_otc_weekly(["AAA"], cache_dir=tmp_path)
    assert len(out) == 2
    assert pd.api.types.is_float_dtype(out["shares"])
    assert pd.api.types.is_datetime64_any_dtype(out["week"])


def test_by_name_section_runs_offline_and_reports_against_chance():
    P, names, idx = _synthetic_panels(n_names=80, n_days=1300, seed=3)
    sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, vols = S.build_signals(P, names)
    dates = idx[idx >= idx[260]][::5]
    lines = []
    S.section_by_name(lines, sig, ctl, adj, dates, names, dpi1, ret_1d, vols, min_names=40,
                      k_placebo=3)
    text = "\n".join(lines)
    assert text.startswith("6. BY NAME")
    for key in ("d_5d h5", "dpi_z h21", "persistence:", "beats", "interaction t",
                "structurally dark tercile"):
        assert key in text
    assert "NDX-100" not in text          # synthetic names are not NDX members


def test_market_relative_fwd_is_demeaned_over_names_with_a_signal():
    idx = pd.bdate_range("2024-01-01", periods=8)
    adj = pd.DataFrame({"A": np.exp(np.arange(8) * 0.01), "B": np.ones(8), "C": np.exp(np.arange(8) * 0.05)},
                       index=idx)
    sig = pd.DataFrame({"A": 1.0, "B": 1.0, "C": np.nan}, index=idx)     # C has no signal
    y = S.market_relative_fwd(adj, sig, 2, idx[:3])
    assert y["C"].isna().all()
    assert (y[["A", "B"]].sum(axis=1).abs() < 1e-9).all()
    assert y["A"].iloc[0] == pytest.approx(1.0)                        # (+2% - 0%) / 2, in pp


def test_by_name_section_declines_short_samples():
    P, names, idx = _synthetic_panels(n_names=40, n_days=460, seed=4)
    sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, vols = S.build_signals(P, names)
    lines = []
    S.section_by_name(lines, sig, ctl, adj, idx[260:][::5], names, dpi1, ret_1d, vols, min_names=20)
    assert "too few for per-name tests" in lines[1]


def _ss(rows):
    cell = lambda v: f'<Cell><Data ss:Type="String">{v}</Data></Cell>'  # noqa: E731
    body = "".join("<Row>" + "".join(cell(v) for v in r) + "</Row>\n" for r in rows)
    return ('<?xml version="1.0"?>\n<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"\n'
            '          xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n'
            ' <Worksheet ss:Name="Holdings"><Table>\n' + body + ' </Table></Worksheet>\n</Workbook>')


def test_sectors_from_spreadsheetml_equity_rows_and_bare_ampersands():
    doc = _ss([["Fund Holdings as of", "Sep 24, 2026"],
               ["Ticker", "Name", "Sector", "Asset Class"],
               ["T", "AT&T INC", "Communication", "Equity"],          # bare '&' must not break parsing
               ["XOM", "EXXON MOBIL CORP", "Energy", "Equity"],
               ["USD", "USD CASH", "Cash and/or Derivatives", "Cash"],
               ["BRK.B", "BERKSHIRE HATHAWAY INC CLASS B", "Financials", "Equity"]])
    assert S.sectors_from_spreadsheetml(doc) == {"T": "Communication", "XOM": "Energy",
                                                  "BRK.B": "Financials"}
    assert S.sectors_from_spreadsheetml("<html>consent gate</html>") == {}
    assert S.sectors_from_spreadsheetml(_ss([["Ticker", "Name"], ["AAPL", "APPLE"]])) == {}


def test_universe_sectors_ndx_uses_static_map_with_gics_names():
    out = S.universe_sectors("ndx")                                  # no network for ndx
    assert out["NVDA"] == "Information Technology"
    assert out["COST"] == "Consumer Staples"
    assert out["AMGN"] == "Health Care"


def test_by_name_section_reports_sectors_when_mapped():
    P, names, idx = _synthetic_panels(n_names=60, n_days=1300, seed=5)
    sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, vols = S.build_signals(P, names)
    dates = idx[idx >= idx[260]][::5]
    smap = {t: ("Alpha" if i < 30 else "Beta") for i, t in enumerate(names)}
    lines = []
    S.section_by_name(lines, sig, ctl, adj, dates, names, dpi1, ret_1d, vols, min_names=30,
                      k_placebo=2, sector_map=smap)
    sector_lines = [ln for ln in lines if "by sector" in ln]
    assert len(sector_lines) == 2 and "Alpha (30)" in sector_lines[0] and "Beta (30)" in sector_lines[0]


def test_section_index_explains_with_implied_vol():
    P, names, idx = _synthetic_panels(n_names=60, n_days=1300, seed=6)
    sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, vols = S.build_signals(P, names)
    rng = np.random.default_rng(1)
    iv = pd.Series(18 + np.cumsum(rng.normal(0, 0.3, len(idx))), index=idx).clip(lower=9)
    lines = []
    S.section_index(lines, dix, adj[names[0]], "TEST", None, iv=iv, ivname="VIX")
    text = "\n".join(lines)
    assert "Why it looks predictive" in text and "+ VIX" in text
    assert "VIX high:" in text and "lead-lag" in text
