#!/usr/bin/env python3
"""
Relative-darkness toolkit
=========================

Name-level dark-flow measures built to separate the pieces the D-vs-DIX residual
(`compute_residuals` in ndx_dark_residual.py) mixes together, plus the statistics
to test them honestly. The findings they produced are in DARK_FLOW_FINDINGS.md;
`dark_flow_study.py` runs the full study.

Why not the residual
--------------------
Per-name DPI is ~13% structural level (names differ persistently), ~7% common
(market-wide, what the DIX measures) and ~80% idiosyncratic day-to-day movement.
The common and idiosyncratic parts relate to PRICE with opposite signs: the
market-wide part rises on down days (dip-buying absorbed off-exchange), while a
name's idiosyncratic part rises WITH its own up-moves -- an echo of buy pressure
the price already shows. Residualizing D against the DIX keeps the echo, throws
away the contrarian part, and does it with a rolling beta against a benchmark that
is mega-cap dominated and contains the name itself. The measures below replace it:

Measures (all causal: a value at t uses only data through session t)
----------------------------------------------------------------------
vw_dpi              k-session VOLUME-weighted DPI = sum(short)/sum(total). The repo's
                    D is an equal-weight mean of daily ratios, so a thin day counts as
                    much as a heavy one.
abnormal_dpi        vw_dpi minus the name's OWN volume-weighted DPI over the `base`
                    sessions before the signal window, and that deviation in units of
                    its own trailing sigma ("relative darkness vs its own norm").
                    Cross-sectional use then needs no market benchmark: ranking names
                    per date removes the common component exactly, with no beta to
                    estimate.
dark_share /        off-exchange share of consolidated volume (FINRA total on Yahoo's
abnormal_dark_share split-adjusted basis / Yahoo volume) -- literal "darkness" -- and
                    its log deviation from the name's own baseline.
dark_imbalance      abnormal dark BUYING in average-daily-volume units: the excess
                    short (= market-maker-facing buy) volume over what the name's
                    normal DPI implies, scaled by its normal volume. Weighs the DPI
                    deviation by how much volume actually went dark.
price_echo_split    per-date cross-sectional regression of a signal on the name's own
                    same-day and k-session returns -> (echo, hidden): the part of the
                    dark buying the price already reflects vs the part it does not --
                    the quantity an "informed dark accumulation" story is about.

Evaluation
----------
forward_log_return  FINRA posts day t's file after that close, so the earliest
                    tradable entry is the close of t+1 (lag=1).
xs_rank             per-date cross-sectional ranks centred on 0, in [-0.5, 0.5]; a
                    regression slope on it reads as the return spread from the
                    lowest- to the highest-ranked name.
fama_macbeth        per-date OLS slopes -> time-series mean with Newey-West t.
quantile_spread     top-minus-bottom quantile mean of an outcome per date.
decompose_dpi       structural / common / idiosyncratic variance shares and the
                    persistence of the idiosyncratic part.
echo_profile        mean per-date rank correlation of a signal with returns at
                    leads/lags.
index_predictability  time-series tests of an index DIX on its index's forward
                    returns: level and detrended, with realized-vol and past-return
                    controls.
index_mechanism     why an index gauge looks predictive: its slope with and without
                    implied vol (VIX/VXN), a vol-regime x gauge table of forward
                    returns, and its lead-lag profile against index returns.

Per-name ("does DPI work better on certain names?")
---------------------------------------------------
per_name_slopes     each name's own time-series slope of forward return on its signal
                    (per 1 SD) with a Newey-West t.
placebo_slope_t     the same t-stats with each name's signal circularly shifted in time
                    -- the null distribution for "how many names look like DPI works
                    here by chance". Overlapping 1-month returns put ~8% of names past
                    |t| = 2 even under this null.
selection_pnl       the honest test of name picking: select names on an estimation
                    window, trade each in its own estimated direction in a later window.
shift_names         independent circular shift per name (placebo helper for the above).
"""
import numpy as np
import pandas as pd

from index_comovement_study import ols_nw


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------
def _paired(short, total):
    """Mask both panels to the cells where both are present (FINRA reports them together;
    this keeps a stray one-sided NaN from unbalancing a rolling sum)."""
    m = short.notna() & total.notna()
    return short.where(m), total.where(m)


def vw_dpi(short, total, window=5, min_periods=None):
    """`window`-session volume-weighted DPI: sum(short) / sum(total), clipped 0..1. NaN when
    fewer than `min_periods` sessions report (default: all but one) or the total is 0."""
    sh, tv = _paired(short, total)
    mp = max(1, window - 1) if min_periods is None else min_periods
    s = sh.rolling(window, min_periods=mp).sum()
    t = tv.rolling(window, min_periods=mp).sum()
    return (s / t.replace(0, np.nan)).clip(lower=0, upper=1)


def _baseline_ratio(num, den, window, base, min_base):
    """sum(num) / sum(den) over the `base` sessions ending just before the current
    `window`-session signal window (so the baseline never overlaps the signal)."""
    n = num.shift(window).rolling(base, min_periods=min_base).sum()
    d = den.shift(window).rolling(base, min_periods=min_base).sum()
    return n / d.replace(0, np.nan)


def abnormal_dpi(short, total, window=5, base=126, sd_window=252, min_base=None):
    """(abn, z) -- the name's `window`-session volume-weighted DPI minus its own
    volume-weighted DPI over the prior `base` sessions, and that deviation divided by its
    own trailing standard deviation (through the PREVIOUS session, so z_t is knowable at t).
    """
    sh, tv = _paired(short, total)
    mb = base // 2 if min_base is None else min_base
    abn = vw_dpi(sh, tv, window) - _baseline_ratio(sh, tv, window, base, mb)
    sd = abn.rolling(sd_window, min_periods=max(2, sd_window // 2)).std().shift(1)
    return abn, abn / sd.replace(0, np.nan)


def dark_share(total, volume, window=5):
    """Off-exchange share of consolidated volume over `window` sessions: sum(FINRA total) /
    sum(consolidated volume). Both must be on the same split basis -- build_universe_panels'
    'total' (Yahoo-adjusted) with Yahoo's 'volume'."""
    v = volume.where(volume > 0)
    tv, vv = _paired(total, v)
    mp = max(1, window - 1)
    return tv.rolling(window, min_periods=mp).sum() / \
        vv.rolling(window, min_periods=mp).sum().replace(0, np.nan)


def abnormal_dark_share(total, volume, window=5, base=126, min_base=None):
    """log(current `window`-session dark share / the name's dark share over the prior `base`
    sessions): > 0 means more of its trading went off-exchange than usual."""
    tv, vv = _paired(total, volume.where(volume > 0))
    mb = base // 2 if min_base is None else min_base
    ratio = dark_share(tv, vv, window) / _baseline_ratio(tv, vv, window, base, mb)
    return np.log(ratio.where(ratio > 0))


def dark_imbalance(short, total, volume, window=5, base=126, min_base=None):
    """Abnormal dark buying in average-daily-volume units:

        (sum_k short - baseline_dpi * sum_k total) / (k * ADV)

    where baseline_dpi and ADV (consolidated) come from the `base` sessions before the
    window. All three volumes must be on one split basis (build_universe_panels' 'short' /
    'total' with Yahoo 'volume')."""
    sh, tv = _paired(short, total)
    mb = base // 2 if min_base is None else min_base
    mp = max(1, window - 1)
    base_dpi = _baseline_ratio(sh, tv, window, base, mb)
    adv = volume.where(volume > 0).shift(window).rolling(base, min_periods=mb).mean()
    excess = sh.rolling(window, min_periods=mp).sum() - base_dpi * tv.rolling(window, min_periods=mp).sum()
    return excess / (window * adv)


def xs_rank(df):
    """Per-date cross-sectional percentile ranks centred on zero (range ~[-0.5, 0.5]);
    NaN cells stay NaN."""
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1).replace(0, np.nan)
    return r.sub(0.5 * (1 + 1 / n), axis=0).where(df.notna())


def xs_residualize(signal, regressors, min_names=30):
    """Per-date cross-sectional OLS residual of `signal` on [1, regressors...]: the part of
    each name's value that day not explained by the regressors. Dates with fewer than
    `min_names` complete names are NaN."""
    regs = [r.reindex(index=signal.index, columns=signal.columns) for r in regressors]
    S = signal.to_numpy(dtype=float)
    Rs = [r.to_numpy(dtype=float) for r in regs]
    res = np.full(S.shape, np.nan)
    for i in range(len(signal.index)):
        y = S[i]
        X = np.column_stack([np.ones_like(y)] + [R[i] for R in Rs])
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < max(min_names, X.shape[1] + 2):
            continue
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        res[i, m] = y[m] - X[m] @ b
    return pd.DataFrame(res, index=signal.index, columns=signal.columns)


def price_echo_split(signal, ret_1d, ret_k, min_names=30):
    """(echo, hidden) parts of a dark-flow signal. Each date, the signal's cross-sectional
    rank is regressed on the names' own same-day and k-session return ranks; `hidden` is the
    residual (dark buying the price has not reflected), `echo` the fitted part."""
    S = xs_rank(signal)
    hidden = xs_residualize(S, [xs_rank(ret_1d), xs_rank(ret_k)], min_names=min_names)
    return S - hidden, hidden


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def forward_log_return(adjclose, horizon, lag=1):
    """Forward log return (%) from the close `lag` sessions after t to `horizon` sessions
    after that. lag=1: FINRA publishes day t's file after the close, so t+1 is the first
    close a signal from t can trade."""
    la = np.log(adjclose.where(adjclose > 0))
    return (la.shift(-(lag + horizon)) - la.shift(-lag)) * 100.0


def newey_west_t(x, lags):
    """(mean, Newey-West t, n) of a series (NaN-dropped)."""
    v = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(v)
    if n < max(8, lags + 3):
        return np.nan, np.nan, n
    beta, se, t, p, r2 = ols_nw(v, np.ones((n, 1)), lags=lags)
    return float(beta[0]), float(t[0]), n


def fama_macbeth(y, X, dates, min_names=30, nw_lags=0):
    """Per-date cross-sectional OLS of `y` on [1, X...] (X: {name: panel}) over `dates`.

    The outcome is demeaned per date (the intercept absorbs the market), so slopes are
    pure cross-sectional return spreads. Returns (summary DataFrame with columns coef / t /
    n_dates per regressor, per-date slope DataFrame)."""
    names = list(X)
    Ys = y.reindex(index=dates)
    Xs = {k: v.reindex(index=dates, columns=y.columns) for k, v in X.items()}
    rows, used = [], []
    for d in dates:
        yy = Ys.loc[d].to_numpy(dtype=float)
        cols = [Xs[k].loc[d].to_numpy(dtype=float) for k in names]
        A = np.column_stack([np.ones_like(yy)] + cols)
        m = np.isfinite(yy) & np.isfinite(A).all(axis=1)
        if m.sum() < max(min_names, A.shape[1] + 2):
            continue
        b, *_ = np.linalg.lstsq(A[m], yy[m] - yy[m].mean(), rcond=None)
        rows.append(b[1:])
        used.append(d)
    slopes = pd.DataFrame(rows, index=pd.DatetimeIndex(used), columns=names)
    out = {}
    for k in names:
        mu, t, n = newey_west_t(slopes[k], nw_lags) if len(slopes) else (np.nan, np.nan, 0)
        out[k] = {"coef": mu, "t": t, "n_dates": n}
    return pd.DataFrame(out).T, slopes


def quantile_spread(signal, y, dates, q=5, min_names=30):
    """Series over `dates` of mean(y | top signal quantile) - mean(y | bottom quantile),
    equal-weighted; dates with fewer than `min_names` paired names are skipped."""
    out = {}
    S, Y = signal.reindex(index=dates), y.reindex(index=dates, columns=signal.columns)
    for d in dates:
        s, v = S.loc[d], Y.loc[d]
        m = s.notna() & v.notna()
        if m.sum() < max(min_names, 2 * q):
            continue
        b = pd.qcut(s[m].rank(method="first"), q, labels=False)
        out[d] = float(v[m][b == q - 1].mean() - v[m][b == 0].mean())
    return pd.Series(out, dtype=float)


def _pooled_acf(e, k):
    a, b = e, e.shift(k)
    m = a.notna() & b.notna()
    a, b = a.where(m), b.where(m)
    a, b = a - a.stack().mean(), b - b.stack().mean()
    den = np.sqrt((a ** 2).sum().sum() * (b ** 2).sum().sum())
    return float((a * b).sum().sum() / den) if den > 0 else np.nan


def decompose_dpi(dpi, lags=(1, 2, 5, 10, 20, 60)):
    """Two-way variance decomposition of a (dates x names) DPI panel.

    dpi_it = mu_i (structural name level) + delta_t (common, per-date cross-sectional mean
    of the demeaned panel) + e_it (idiosyncratic). Returns shares of total variance, the
    pooled autocorrelations of e, and an AR(1)+white-noise fit to them
    (acf(k) = s * rho**k -> persistent share s, rho, half-life in sessions)."""
    x = dpi.astype(float)
    total = float(x.stack().var())
    mu = x.mean()
    dm = x - mu
    delta = dm.mean(axis=1)
    e = dm.sub(delta, axis=0)
    acf = {k: _pooled_acf(e, k) for k in lags}
    out = {"n_obs": int(x.notna().sum().sum()), "n_names": int(x.shape[1]),
           "share_structural": float(mu.var() / total),
           "share_common": float(delta.var() / total),
           "share_idio": float(e.stack().var() / total),
           "sd_structural": float(mu.std()), "sd_common": float(delta.std()),
           "sd_idio": float(e.stack().std()), "acf": acf}
    a1, a2 = acf.get(1), acf.get(2)
    if a1 and a2 and a1 > 0 and a2 > 0 and a2 < a1:
        rho = a2 / a1
        out.update({"ar_rho": rho, "ar_persistent_share": a1 * a1 / a2,
                    "ar_half_life": float(np.log(0.5) / np.log(rho))})
    return out


def echo_profile(signal, returns, lags=(-5, -2, -1, 0, 1, 2, 5), min_names=30):
    """Mean per-date cross-sectional Spearman correlation between signal_t and
    returns_{t+k}; k < 0 looks at past returns, k = 0 same day, k > 0 future."""
    R = returns.rank(axis=1)
    S = signal.rank(axis=1)
    out = {}
    for k in lags:
        Rk = R.shift(-k)
        cs = []
        for d in S.index:
            a, b = S.loc[d], Rk.loc[d]
            m = a.notna() & b.notna()
            if m.sum() >= min_names:
                cs.append(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])
        out[k] = float(np.mean(cs)) if cs else np.nan
    return out


def index_predictability(dix, price, horizons=(5, 21, 63), smooth=5, trend=252):
    """Time-series tests of an index dark gauge on the index's own forward log returns (%).

    For each horizon h: slope and Newey-West(h) t of the forward return on the z-scored
    `smooth`-session DIX mean -- alone and with the index's 21-session realized vol and
    past 21-session return as controls -- and the same for the DETRENDED gauge (minus its
    trailing `trend`-session mean; the DIX drifts up over the years, which a raw-level test
    can mistake for predictive power). Returns a tidy DataFrame."""
    df = pd.DataFrame({"dix": dix, "px": price}).dropna()
    lp = np.log(df["px"])
    g = df["dix"].rolling(smooth, min_periods=max(1, smooth - 2)).mean()
    sig = {"level": g, "detrended": g - g.rolling(trend, min_periods=trend // 2).mean()}
    rv = lp.diff().rolling(21, min_periods=15).std() * np.sqrt(252) * 100
    past = (lp - lp.shift(21)) * 100
    rows = []
    for h in horizons:
        fwd = (lp.shift(-h) - lp) * 100
        for name, s in sig.items():
            for ctl in (False, True):
                parts = {"s": s} if not ctl else {"s": s, "rv21": rv, "past21": past}
                D = pd.DataFrame({**parts, "y": fwd}).dropna()
                if len(D) < 3 * h + 10:
                    continue
                Z = (D.drop(columns="y") - D.drop(columns="y").mean()) / D.drop(columns="y").std()
                X = np.column_stack([np.ones(len(D))] + [Z[c].to_numpy() for c in Z])
                beta, se, t, p, r2 = ols_nw(D["y"].to_numpy(), X, lags=h)
                row = {"horizon": h, "signal": name, "controls": ctl, "coef_pp_per_sd": beta[1],
                       "t": t[1], "n": len(D), "start": D.index[0].date(), "end": D.index[-1].date()}
                if ctl:
                    row.update({"t_rv21": t[2], "t_past21": t[3]})
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Per-name heterogeneity
# --------------------------------------------------------------------------
def nw_slope(x, y, lags):
    """Time-series OLS slope of y on x -- x standardized to 1 SD, so the slope is the outcome's
    move per 1 SD of the signal -- with a Newey-West (Bartlett) t. Pairs with a NaN on either
    side are dropped. Returns (slope, t, n); (nan, nan, n) when n < 30 or x is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 30 or x.std() == 0:
        return np.nan, np.nan, n
    x = (x - x.mean()) / x.std()
    yd = y - y.mean()
    sxx = x @ x
    b = (x @ yd) / sxx
    g = x * (yd - b * x)
    s = g @ g
    for lag in range(1, lags + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * (g[lag:] @ g[:-lag])
    se = np.sqrt(s) / sxx
    return float(b), (float(b / se) if se > 0 else np.nan), n


def per_name_slopes(X, Y, lags, min_obs=150):
    """DataFrame [b, t, n] indexed by name: each column of X (signal) against the same column
    of Y (outcome), both dates x names on the same (e.g. weekly) dates. Names with fewer than
    `min_obs` paired observations are left out."""
    rows = {}
    for c in X.columns:
        if c not in Y.columns:
            continue
        b, t, n = nw_slope(X[c].to_numpy(float), Y[c].to_numpy(float), lags)
        if n >= min_obs and np.isfinite(t):
            rows[c] = (b, t, n)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["b", "t", "n"])


def shift_names(X, rng, min_shift=26):
    """Copy of X with each column circularly shifted by its own random offset of at least
    `min_shift` rows from the true alignment (keeps each series' autocorrelation, breaks its
    timing relation to anything else)."""
    a = X.to_numpy(float)
    n = a.shape[0]
    if n <= 2 * min_shift:
        raise ValueError("series too short for the requested minimum shift")
    sh = rng.integers(min_shift, n - min_shift, size=a.shape[1])
    return pd.DataFrame(np.column_stack([np.roll(a[:, j], sh[j]) for j in range(a.shape[1])]),
                        index=X.index, columns=X.columns)


def placebo_slope_t(X, Y, lags, k=40, min_obs=150, min_shift=26, seed=0):
    """Null distribution of per-name t-stats: `k` independent circular shifts of every name's
    signal. Compare the real per_name_slopes t's against it -- tail counts under this null are
    what 'DPI works on this name' looks like by chance."""
    rng = np.random.default_rng(seed)
    ts = [per_name_slopes(shift_names(X, rng, min_shift), Y, lags, min_obs)["t"].to_numpy()
          for _ in range(k)]
    return np.concatenate(ts) if ts else np.array([])


def selection_pnl(X, Y, est_mask, test_mask, lags, sel_t=1.5, min_obs=80, demean=False,
                  orient="own"):
    """Out-of-sample test of "trade DPI only on the names where it worked".

    Names whose estimation-window slope has |t| > `sel_t` are traded in the test window: each
    position is the name's signal standardized with ESTIMATION-window mean/SD (no look-ahead),
    times the sign of its own estimated slope (`orient="own"`) or of the average slope across
    all names (`orient="pooled"` -- separates name-specific direction from a common tilt).
    `demean=True` removes each name's test-window mean position (timing only; attribution, not
    tradeable). Returns (per-date P&L = mean over selected names of position x outcome, number
    of names selected)."""
    est = per_name_slopes(X[est_mask], Y[est_mask], lags, min_obs=min_obs)
    sel = est[est["t"].abs() > sel_t]
    if sel.empty:
        return pd.Series(dtype=float), 0
    xe = X[est_mask][sel.index]
    z = (X[test_mask][sel.index] - xe.mean()) / xe.std()
    if demean:
        z = z - z.mean()
    sign = np.sign(sel["b"]) if orient == "own" else np.sign(est["b"].mean())
    return (z * sign * Y[test_mask][sel.index]).mean(axis=1).dropna(), len(sel)


# --------------------------------------------------------------------------
# Index level: what the gauge tracks, and what absorbs its predictive power
# --------------------------------------------------------------------------
def index_mechanism(dix, price, iv, horizons=(21, 63), smooth=5, trend=252, lags=(-3, -2, -1, 0, 1, 2, 3)):
    """Why an index dark gauge looks predictive, in three views.

    `coef`: slope (pp of forward log return per 1 SD, Newey-West t with lags = horizon) of the
    `smooth`-session gauge alone, with implied vol `iv` (e.g. VIX) as a control, of `iv` alone,
    and of the detrended gauge (minus its trailing `trend`-session mean) with and without `iv`.
    `table`: mean 1-month (21-session) forward return by terciles of each series' trailing
    1-year percentile (rows: `iv`, columns: gauge; no look-ahead), with counts in `counts`.
    `leadlag`: correlation of the detrended daily gauge at t with the index return at t+k.
    `loo`: the gauge-alone 1-month slope re-estimated leaving out each calendar year -- which
    episodes the raw relation leans on. `corr_iv` / `corr_iv_detrended`: how closely the
    (detrended) gauge tracks implied vol.
    """
    df = pd.DataFrame({"dix": dix, "px": price, "iv": iv}).dropna()
    lp = np.log(df["px"])
    r = lp.diff() * 100
    g = df["dix"].rolling(smooth, min_periods=max(1, smooth - 2)).mean()
    det = g - g.rolling(trend, min_periods=trend // 2).mean()
    specs = {"gauge alone": ["g"], "gauge + implied vol": ["g", "iv"], "implied vol alone": ["iv"],
             "detrended gauge": ["det"], "detrended gauge + implied vol": ["det", "iv"]}
    base = pd.DataFrame({"g": g, "det": det, "iv": df["iv"]})
    rows = []
    for h in horizons:
        y = (lp.shift(-h) - lp) * 100
        for name, xs in specs.items():
            D = pd.concat([base[xs], y.rename("y")], axis=1).dropna()
            if len(D) < 3 * h + 10:
                continue
            Z = (D[xs] - D[xs].mean()) / D[xs].std()
            beta, se, t, p, r2 = ols_nw(D["y"].to_numpy(), np.column_stack([np.ones(len(D))] + [Z[c] for c in xs]),
                                        lags=h)
            rows.append({"horizon": h, "spec": name, "coef": beta[1], "t": t[1], "n": len(D)})

    def tpct(s):
        return s.rolling(trend, min_periods=trend // 2).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
    B = pd.DataFrame({"gp": tpct(g), "vp": tpct(df["iv"]), "f": (lp.shift(-21) - lp) * 100}).dropna()
    labels = ["low", "mid", "high"]
    B["gauge"] = pd.cut(B["gp"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=labels)
    B["implied vol"] = pd.cut(B["vp"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=labels)
    table = B.pivot_table(index="implied vol", columns="gauge", values="f", aggfunc="mean", observed=True)
    counts = B.pivot_table(index="implied vol", columns="gauge", values="f", aggfunc="count", observed=True)
    x = df["dix"] - df["dix"].rolling(trend, min_periods=trend // 2).mean()
    leadlag = {k: float(x.corr(r.shift(-k))) for k in lags}
    # which years carry the raw 1-month slope: re-estimate leaving each calendar year out
    D = pd.concat([g.rename("g"), ((lp.shift(-21) - lp) * 100).rename("y")], axis=1).dropna()
    loo = {}
    for yr in sorted(set(D.index.year)):
        d = D[D.index.year != yr]
        z = (d["g"] - d["g"].mean()) / d["g"].std()
        beta, se, t, p, r2 = ols_nw(d["y"].to_numpy(), np.column_stack([np.ones(len(d)), z]), lags=21)
        loo[yr] = (float(beta[1]), float(t[1]))
    return {"coef": pd.DataFrame(rows), "table": table, "counts": counts, "leadlag": leadlag,
            "loo": pd.DataFrame.from_dict(loo, orient="index", columns=["coef", "t"]),
            "corr_iv": float(g.corr(df["iv"])), "corr_iv_detrended": float(det.corr(df["iv"]))}
