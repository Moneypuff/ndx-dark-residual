"""Tests for the hand-rolled statistics in ``earnings_dpi_study``.

These functions (``_gammaln``, ``_betainc``/``_betacf``, ``_t_sf``,
``_pearson``, ``_spearman``, ``_welch``) reimplement pieces of scipy from
scratch so the production pipeline can stay scipy-free. Everything downstream
-- the correlations, terciles and p-values reported in the study and the HTML
report -- rests on them, and a bug here would silently corrupt results rather
than crash.

Strategy: validate each function against scipy as a trusted oracle (scipy is a
test-only dependency, see requirements-dev.txt) across a range of inputs, and
separately pin down the documented edge-case branches (too-few samples, perfect
correlation, zero variance, degenerate degrees of freedom).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import earnings_dpi_study as E


# ---------------------------------------------------------------------------
# _gammaln -- log-gamma (Lanczos approximation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.5, 1.0, 1.5, 2.0, 3.7, 5.0, 10.0, 50.0, 123.4])
def test_gammaln_matches_scipy(x):
    assert E._gammaln(x) == pytest.approx(special.gammaln(x), rel=1e-9, abs=1e-9)


def test_gammaln_known_values():
    # gamma(n) = (n-1)!  ->  gammaln is the log of that
    assert E._gammaln(1.0) == pytest.approx(0.0, abs=1e-10)   # 0! = 1
    assert E._gammaln(2.0) == pytest.approx(0.0, abs=1e-10)   # 1! = 1
    assert E._gammaln(5.0) == pytest.approx(np.log(24.0), abs=1e-9)  # 4! = 24


# ---------------------------------------------------------------------------
# _betainc -- regularized incomplete beta I_x(a, b)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b,x", [
    (2.0, 3.0, 0.4),
    (0.5, 0.5, 0.3),
    (5.0, 2.0, 0.9),
    (1.0, 1.0, 0.25),   # I_x(1,1) == x
    (10.0, 20.0, 0.35),
    (2.5, 7.5, 0.1),    # x below the continued-fraction switch point
    (2.5, 7.5, 0.8),    # x above the switch point (exercises the 1 - ... branch)
])
def test_betainc_matches_scipy(a, b, x):
    assert E._betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), rel=1e-8, abs=1e-10)


def test_betainc_boundaries():
    # Documented clamps: I_x is 0 at/below 0 and 1 at/above 1.
    assert E._betainc(2.0, 3.0, 0.0) == 0.0
    assert E._betainc(2.0, 3.0, -0.5) == 0.0
    assert E._betainc(2.0, 3.0, 1.0) == 1.0
    assert E._betainc(2.0, 3.0, 1.5) == 1.0


def test_betainc_identity_uniform():
    # I_x(1, 1) is the identity on [0, 1].
    for x in (0.1, 0.5, 0.9):
        assert E._betainc(1.0, 1.0, x) == pytest.approx(x, abs=1e-10)


# ---------------------------------------------------------------------------
# _t_sf -- Student-t survival function P(T > t)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t,df", [
    (0.0, 10),
    (0.5, 5),
    (2.0, 10),
    (3.3, 30),
    (1.0, 1),
    (5.0, 100),
    (0.75, 2.5),   # non-integer df (Welch produces these)
])
def test_t_sf_matches_scipy(t, df):
    assert E._t_sf(t, df) == pytest.approx(stats.t.sf(t, df), rel=1e-7, abs=1e-10)


def test_t_sf_at_zero_is_half():
    # The t-distribution is symmetric about 0, so P(T > 0) == 0.5 for any df.
    for df in (1, 5, 30):
        assert E._t_sf(0.0, df) == pytest.approx(0.5, abs=1e-9)


def test_t_sf_nonpositive_df_is_nan():
    # Documented guard: df <= 0 has no valid distribution.
    assert np.isnan(E._t_sf(2.0, 0))
    assert np.isnan(E._t_sf(2.0, -3))


# ---------------------------------------------------------------------------
# _pearson -- Pearson r with two-sided p via the t-approximation
# ---------------------------------------------------------------------------
def test_pearson_matches_scipy():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=60))
    y = pd.Series(0.4 * x + rng.normal(size=60))
    r, p, n = E._pearson(x, y)
    ref = stats.pearsonr(x, y)
    assert n == 60
    assert r == pytest.approx(ref.statistic, rel=1e-9)
    assert p == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-12)


def test_pearson_masks_nans_and_reports_pair_count():
    # Only rows finite in BOTH series should count; n reflects the overlap.
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0])
    y = pd.Series([2.0, np.nan, 6.0, 8.0, 10.0, 12.0, 14.0])
    r, p, n = E._pearson(x, y)
    mask = x.notna() & y.notna()
    assert n == int(mask.sum()) == 5
    ref = stats.pearsonr(x[mask], y[mask])
    assert r == pytest.approx(ref.statistic, rel=1e-9)


def test_pearson_too_few_samples_returns_nan():
    # Documented guard: fewer than 5 valid pairs -> (nan, nan, count).
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    r, p, n = E._pearson(x, y)
    assert np.isnan(r) and np.isnan(p)
    assert n == 4


def test_pearson_perfect_positive_correlation():
    # |r| >= 1 takes the p == 0 fast path (avoids divide-by-zero in the t-stat).
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 3.0 * x + 1.0
    r, p, n = E._pearson(x, y)
    assert r == pytest.approx(1.0, abs=1e-12)
    assert p == 0.0
    assert n == 6


def test_pearson_perfect_negative_correlation():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = -2.0 * x + 5.0
    r, p, n = E._pearson(x, y)
    assert r == pytest.approx(-1.0, abs=1e-12)
    assert p == 0.0


# ---------------------------------------------------------------------------
# _spearman -- rank correlation (delegates to _pearson on the ranks)
# ---------------------------------------------------------------------------
def test_spearman_matches_scipy():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=50))
    y = pd.Series(x ** 3 + 0.1 * rng.normal(size=50))  # monotone-ish, non-linear
    r, p, n = E._spearman(x, y)
    ref = stats.spearmanr(x, y)
    assert r == pytest.approx(ref.statistic, rel=1e-9)
    assert p == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-12)


def test_spearman_monotonic_is_unit():
    # A strictly increasing relationship has rank correlation exactly 1.
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = pd.Series([10.0, 11.0, 13.0, 20.0, 21.0, 30.0, 100.0])
    r, p, n = E._spearman(x, y)
    assert r == pytest.approx(1.0, abs=1e-12)
    assert p == 0.0


def test_spearman_too_few_samples_returns_nan():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([4.0, 3.0, 2.0, 1.0])
    r, p, n = E._spearman(x, y)
    assert np.isnan(r) and np.isnan(p)
    assert n == 4


# ---------------------------------------------------------------------------
# _welch -- Welch's unequal-variance two-sample t-test
# ---------------------------------------------------------------------------
def test_welch_matches_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.5, 2.0, size=30)
    diff, t, p, na, nb = E._welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert na == 40 and nb == 30
    assert diff == pytest.approx(a.mean() - b.mean(), rel=1e-12)
    assert t == pytest.approx(ref.statistic, rel=1e-9)
    assert p == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-12)


def test_welch_sign_of_difference():
    # diff is mean(a) - mean(b); a clearly larger mean -> positive diff and t.
    a = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 10.2])
    b = np.array([1.0, 2.0, 0.0, 1.5, 0.5, 1.2])
    diff, t, p, na, nb = E._welch(a, b)
    assert diff > 0
    assert t > 0
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert t == pytest.approx(ref.statistic, rel=1e-9)


def test_welch_drops_nonfinite_before_counting():
    # NaN/inf are filtered first, so na/nb reflect only the finite entries and
    # the result equals scipy run on the cleaned arrays.
    a = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
    b = np.array([2.0, 3.0, 4.0, -np.inf, 5.0])
    diff, t, p, na, nb = E._welch(a, b)
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    assert na == len(a_clean) == 4
    assert nb == len(b_clean) == 4
    ref = stats.ttest_ind(a_clean, b_clean, equal_var=False)
    assert t == pytest.approx(ref.statistic, rel=1e-9)
    assert p == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("na,nb", [(2, 10), (10, 2), (2, 2)])
def test_welch_too_few_samples_returns_nan(na, nb):
    # Documented guard: need at least 3 per side.
    a = np.arange(na, dtype=float)
    b = np.arange(nb, dtype=float) + 0.5
    diff, t, p, ra, rb = E._welch(a, b)
    assert np.isnan(diff) and np.isnan(t) and np.isnan(p)
    assert ra == na and rb == nb


def test_welch_zero_standard_error_returns_mean_diff_but_nan_t():
    # Both samples constant -> se == 0; documented behaviour is to still return
    # the mean difference but leave t and p as nan (no divide-by-zero).
    a = np.array([5.0, 5.0, 5.0, 5.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    diff, t, p, na, nb = E._welch(a, b)
    assert diff == pytest.approx(3.0)
    assert np.isnan(t) and np.isnan(p)
    assert na == 4 and nb == 4
