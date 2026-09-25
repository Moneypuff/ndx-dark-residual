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
