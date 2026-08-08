"""Tests for the GEX-regime study: trailing-percentile bucketing (Q1 must be
the negative regime when negatives are present), vol adjustment arithmetic, and
proof the heatmap / IC-by-regime machinery recovers a planted 'DIX only works
in the low-GEX regime' interaction.
"""
import numpy as np
import pandas as pd
import pytest

import spx_gex_regime_study as G


def test_to_band_cuts():
    assert G.to_band([0, 19.9, 20, 39, 55, 79.9, 80, 100]).tolist() == \
        [1, 1, 2, 2, 3, 4, 5, 5]


def test_trailing_pct_q1_is_negative_regime():
    # GEX oscillates around 0 with occasional deep negatives: the lowest
    # trailing-percentile quintile must be dominated by the negative days.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    gex = pd.Series(rng.normal(2.0, 2.5, 1500), index=idx)
    pct = G.trailing_pct(gex)
    q = pd.Series(G.to_band(pct), index=idx).where(pct.notna())
    neg_share_q1 = (gex[q == 1] < 0).mean()
    neg_share_q5 = (gex[q == 5] < 0).mean()
    assert neg_share_q1 > 0.5
    assert neg_share_q5 < 0.05
    # today's value is not compared against itself: a fresh all-time max ranks 100
    s = pd.Series(np.arange(300, dtype=float), index=pd.bdate_range("2015-01-01", periods=300))
    p = G.trailing_pct(s, win=252, min_obs=100)
    assert p.iloc[-1] == pytest.approx(100.0)


def test_vol_adjustment_arithmetic():
    # constant daily vol sigma -> zrv = fwd_frac / (sigma*sqrt(h))
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(1)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=idx)
    df = pd.DataFrame({"price": price, "dix": rng.uniform(40, 46, 300),
                       "gex": rng.normal(2, 1, 300)}, index=idx)
    d = G.build_frame(df)
    h = G.HORIZONS[0]
    i = 100
    fwd_frac = price.iloc[i + h] / price.iloc[i] - 1.0
    assert d[f"fwd{h}"].iloc[i] == pytest.approx(fwd_frac * 100, rel=1e-9)
    assert d[f"zrv{h}"].iloc[i] == pytest.approx(
        fwd_frac / (d["rv21"].iloc[i] * np.sqrt(h)), rel=1e-9)


def _planted(interaction, seed=0, n=2500):
    """Frame with gexq/dband/pd126 and an outcome where %D predicts returns
    ONLY inside GEX quintile 1 (strength = `interaction`)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2013-01-01", periods=n)
    gexq = rng.integers(1, 6, n).astype(float)
    pd126 = rng.uniform(0, 100, n)
    dband = G.to_band(pd126)
    z = rng.normal(0, 1, n) + np.where(gexq == 1, interaction * (pd126 - 50) / 50.0, 0.0)
    return pd.DataFrame({"gexq": gexq, "dband": dband, "pd126": pd126,
                         "zrv21": z, "fwd21": z}, index=idx)


def test_ic_by_regime_recovers_planted_interaction():
    d = _planted(interaction=1.0, seed=2)
    rows = {r["q"]: r for r in G.ic_by_regime(d, 21)}
    assert rows[1]["ic"] > 0.3
    assert rows[1]["ci"][0] > 0                     # Q1 IC clearly positive
    for q in (2, 3, 4, 5):
        # other regimes carry no planted signal; finite-sample noise ICs can
        # graze zero, so bound the magnitude instead of demanding a straddle
        assert abs(rows[q]["ic"]) < 0.15
        assert rows[q]["ic"] < rows[1]["ic"] - 0.3


def test_heatmap_shows_gradient_only_in_q1():
    d = _planted(interaction=1.0, seed=3, n=6000)
    gm, gn = G.heatmap(d, 21)
    assert gm[0, 4] - gm[0, 0] > 0.8                # Q1: strong D1->D5 gradient
    assert abs(gm[2, 4] - gm[2, 0]) < 0.4           # Q3: flat
    assert gn.sum() == len(d)


def test_ic_by_regime_flat_on_null():
    d = _planted(interaction=0.0, seed=4)
    rows = {r["q"]: r for r in G.ic_by_regime(d, 21)}
    lo, hi = rows[1]["ci"]
    assert lo < 0 < hi
