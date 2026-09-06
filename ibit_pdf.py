#!/usr/bin/env python3
"""Risk-neutral probability density for the IBIT option chain.

Extracts the risk-neutral (Breeden-Litzenberger) probability density of the
iShares Bitcoin Trust (IBIT) share price at each monthly expiration from the
October 2026 monthly through the January 2027 monthly.

Method
------
Breeden & Litzenberger (1978): the risk-neutral density of the underlying at
expiry T is the (undiscounted) second derivative of the call price with
respect to strike,

    q(K) = e^{rT} d^2C/dK^2 .

Differentiating raw, noisy quotes directly amplifies the noise, so instead we

  1. estimate the implied forward F and discount factor D = e^{-rT} for each
     expiry from put-call parity  (C - P = D (F - K)),
  2. take the out-of-the-money wing of each side (puts below F, calls above F),
     invert each mid price to a Black-76 implied volatility,
  3. fit a smooth volatility smile across strike and extrapolate it flat into
     the wings,
  4. re-price a dense grid of calls off the fitted smile and take a clean
     second derivative,
  5. clip tiny negatives and renormalise so the density integrates to one.

Data source
-----------
By default the chain is pulled live from Yahoo Finance (public delayed quotes).
Alternatively point --csv at a snapshot in the repo's ``optsnap`` schema
(columns: date,symbol,expiry,right,strike,iv,oi,volume,bid,ask,last,spot) and
the same computation runs fully offline.

Outputs (written to --outdir, default ./ibit_pdf_out):
  * ibit_pdf_<expiry>.csv   -- price grid and density for each expiry
  * ibit_pdf_summary.csv    -- forward, discount, moments and key probabilities
  * ibit_pdf.png            -- density and CDF plots (unless --no-plot)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

TICKER = "IBIT"

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# The valuation date used for time-to-expiry. Overridable with --asof; defaults
# to today so the script stays correct as it is re-run over time.
DEFAULT_ASOF = dt.date.today()

# Monthly-expiry window requested: October monthly through January monthly.
# Monthly options expire on the third Friday of the month.
DEFAULT_MONTHS = [(2026, 10), (2026, 11), (2026, 12), (2027, 1)]


# --------------------------------------------------------------------------- #
# Black-76 pricing / implied vol on the forward
# --------------------------------------------------------------------------- #
def _black76(F, K, T, sigma, D, call: bool):
    """Black-76 price times the discount factor D. Works on scalars or arrays."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    intrinsic = np.maximum(F - K, 0.0) if call else np.maximum(K - F, 0.0)
    safe = (sigma > 0) & (T > 0)
    vsqrt = np.where(safe, sigma * np.sqrt(T), 1.0)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vsqrt
    d2 = d1 - vsqrt
    if call:
        priced = F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        priced = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    out = D * np.where(safe, priced, intrinsic)
    return out if out.ndim else float(out)


def _implied_vol(price: float, F: float, K: float, T: float, D: float, call: bool) -> float:
    """Invert Black-76 for the implied vol; NaN if the price is not arbitrage-sound."""
    if price <= 0 or T <= 0:
        return np.nan
    intrinsic = D * (max(F - K, 0.0) if call else max(K - F, 0.0))
    upper = D * (F if call else K)  # price of the extreme (sigma -> inf) option
    if price <= intrinsic + 1e-9 or price >= upper - 1e-12:
        return np.nan
    try:
        return brentq(
            lambda s: _black76(F, K, T, s, D, call) - price,
            1e-4, 8.0, maxiter=200, xtol=1e-8,
        )
    except (ValueError, RuntimeError):
        return np.nan


# --------------------------------------------------------------------------- #
# Chain container + forward/discount estimation
# --------------------------------------------------------------------------- #
# One option quote. two_sided marks a genuine live bid/ask market (as opposed
# to a mid synthesised from a possibly stale last trade); the smile fit trusts
# only two-sided quotes with a sane relative spread.
Quote = namedtuple("Quote", "mid bid ask oi two_sided")


@dataclass
class ExpiryChain:
    expiry: dt.date
    asof: dt.date
    spot: float
    # per-strike quotes, keyed by strike
    calls: dict = field(default_factory=dict)  # strike -> Quote
    puts: dict = field(default_factory=dict)

    @property
    def T(self) -> float:
        return max((self.expiry - self.asof).days, 0) / 365.0


def _quote(bid, ask, last, oi) -> Quote:
    """Build a Quote from raw fields; mid of a two-sided market, else last trade."""
    oi = float(oi or 0)
    b = float(bid) if bid is not None else 0.0
    a = float(ask) if ask is not None else 0.0
    if b > 0 and a > 0 and a >= b:
        return Quote(0.5 * (b + a), b, a, oi, True)
    if last is not None and float(last) > 0:
        return Quote(float(last), b, a, oi, False)
    return Quote(np.nan, b, a, oi, False)


def _usable(q: Quote, max_rel_spread: float) -> bool:
    """A quote fit for the smile: live two-sided market, positive, sane spread."""
    if q is None or not q.two_sided or np.isnan(q.mid) or q.mid <= 0:
        return False
    return (q.ask - q.bid) <= max_rel_spread * q.mid


def estimate_forward_discount(chain: ExpiryChain, r_guess: float = 0.045):
    """Fit  C - P = D (F - K)  by OLS over strikes quoted on both sides.

    Returns (F, D). Falls back to spot / exp(-r_guess T) when parity is too thin.
    """
    ks, diffs, weights = [], [], []
    for k in sorted(set(chain.calls) & set(chain.puts)):
        c, p = chain.calls[k], chain.puts[k]
        if not (c.two_sided and p.two_sided) or np.isnan(c.mid) or np.isnan(p.mid):
            continue
        w = np.sqrt(c.oi + p.oi + 1.0)
        # near-ATM strikes carry the cleanest parity signal
        w *= np.exp(-((k / chain.spot - 1.0) ** 2) / (2 * 0.15 ** 2))
        ks.append(k); diffs.append(c.mid - p.mid); weights.append(w)

    T = chain.T
    if len(ks) >= 3:
        ks = np.array(ks); diffs = np.array(diffs); weights = np.array(weights)
        # weighted linear fit  diffs = a + b*K  ->  b = -D, a = D*F
        W = np.sqrt(weights)
        A = np.vstack([np.ones_like(ks), ks]).T * W[:, None]
        coef, *_ = np.linalg.lstsq(A, diffs * W, rcond=None)
        a, b = coef
        D = -b
        if 0.5 < D <= 1.0 + 1e-6 and a > 0:
            D = min(D, 1.0)
            return a / D, D
    # fallback
    D = float(np.exp(-r_guess * T))
    return chain.spot / D, D


# --------------------------------------------------------------------------- #
# Breeden-Litzenberger density from a fitted smile
# --------------------------------------------------------------------------- #
@dataclass
class DensityResult:
    expiry: dt.date
    T: float
    spot: float
    forward: float
    discount: float
    grid: np.ndarray          # strike / terminal-price grid
    density: np.ndarray       # risk-neutral pdf on the grid
    cdf: np.ndarray
    smile_k: np.ndarray       # strikes used for the smile
    smile_iv: np.ndarray      # implied vols used for the smile
    n_quotes: int


def compute_density(chain: ExpiryChain, grid_points: int = 801,
                    smooth: float | None = None,
                    max_rel_spread: float = 0.5) -> DensityResult:
    F, D = estimate_forward_discount(chain)
    T = chain.T

    # Out-of-the-money wing: puts below the forward, calls above it. Only
    # genuine two-sided quotes with a sane spread feed the smile.
    pts = []  # (strike, iv, open interest)
    for k, p in chain.puts.items():
        if k > F or not _usable(p, max_rel_spread):
            continue
        iv = _implied_vol(p.mid, F, k, T, D, call=False)
        if not np.isnan(iv):
            pts.append((k, iv, p.oi))
    for k, c in chain.calls.items():
        if k < F or not _usable(c, max_rel_spread):
            continue
        iv = _implied_vol(c.mid, F, k, T, D, call=True)
        if not np.isnan(iv):
            pts.append((k, iv, c.oi))

    if len(pts) < 5:
        raise ValueError(f"too few usable quotes for {chain.expiry} ({len(pts)})")

    pts.sort()
    ks = np.array([p[0] for p in pts])
    ivs = np.array([p[1] for p in pts])
    ois = np.array([p[2] for p in pts], dtype=float)

    # collapse duplicate strikes (e.g. a call and put both quoted at the forward)
    uniq_k, inv = np.unique(ks, return_inverse=True)
    uniq_iv = np.array([ivs[inv == i].mean() for i in range(len(uniq_k))])
    uniq_oi = np.array([ois[inv == i].sum() for i in range(len(uniq_k))])

    # Trim IV outliers: points more than 4 robust deviations from a local
    # median smile are stale quotes that would kink the density.
    if len(uniq_k) >= 9:
        from scipy.ndimage import median_filter
        med = median_filter(uniq_iv, size=5, mode="nearest")
        resid = uniq_iv - med
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        keep = np.abs(resid) < 4 * 1.4826 * mad
        if keep.sum() >= 5:
            uniq_k, uniq_iv, uniq_oi = uniq_k[keep], uniq_iv[keep], uniq_oi[keep]

    # Smooth the smile in strike space. Weights (mean-normalised so the
    # smoothing budget keeps its scale) lean on liquid strikes; `resid_vol` is
    # the vol-point noise the fit is allowed to absorb, so the spline follows
    # the smile's curvature without tracing quote noise into the density.
    w = np.sqrt(uniq_oi + 1.0)
    w = w / w.mean()
    resid_vol = smooth if smooth is not None else 0.01
    s = len(uniq_k) * resid_vol ** 2
    spline = UnivariateSpline(uniq_k, uniq_iv, w=w, k=3, s=s)
    dspline = spline.derivative()

    def iv_at(k):
        # C1-continuous extrapolation: continue the smile linearly off each
        # boundary using the spline's own slope there, so IV and IV' stay
        # continuous and the reconstructed density has no kink at the join.
        k = np.asarray(k, dtype=float)
        lo, hi = uniq_k[0], uniq_k[-1]
        out = spline(np.clip(k, lo, hi))
        left = k < lo
        right = k > hi
        out = np.where(left, spline(lo) + dspline(lo) * (k - lo), out)
        out = np.where(right, spline(hi) + dspline(hi) * (k - hi), out)
        return np.clip(out, 1e-3, 8.0)

    # Dense grid. Extend past the quoted strikes by a volatility-scaled amount
    # (not a fixed fraction) so both tails have room to decay without inventing
    # a fat wing beyond where the smile is anchored.
    atm_iv = float(iv_at(F))
    pad = max(4.0 * F * atm_iv * np.sqrt(T), 0.10 * F)
    lo = max(1e-6, min(uniq_k[0], F - pad))
    hi = max(uniq_k[-1], F + pad)
    grid = np.linspace(lo, hi, grid_points)
    calls = _black76(F, grid, T, iv_at(grid), D, call=True)

    # q(K) = e^{rT} d^2C/dK^2 = (1/D) C''(K), central differences.
    dK = grid[1] - grid[0]
    d2 = np.gradient(np.gradient(calls, dK), dK)
    density = d2 / D
    # Light Gaussian pass removes the finite-difference kinks left where the
    # smile joins its flat wings; too small to distort the body's shape.
    from scipy.ndimage import gaussian_filter1d
    density = gaussian_filter1d(density, sigma=max(grid_points / 250.0, 1.0))
    density = np.clip(density, 0.0, None)

    area = _trapz(density, grid)
    if area > 0:
        density = density / area
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * dK)])

    return DensityResult(
        expiry=chain.expiry, T=T, spot=chain.spot, forward=F, discount=D,
        grid=grid, density=density, cdf=cdf,
        smile_k=uniq_k, smile_iv=uniq_iv, n_quotes=len(pts),
    )


def summarize(res: DensityResult) -> dict:
    g, d = res.grid, res.density
    mean = _trapz(g * d, g)
    var = _trapz((g - mean) ** 2 * d, g)
    std = np.sqrt(max(var, 0.0))

    def quantile(q):
        return float(np.interp(q, res.cdf, g))

    def prob_above(level):
        return float(1.0 - np.interp(level, g, res.cdf))

    return {
        "expiry": res.expiry.isoformat(),
        "days": int(round(res.T * 365)),
        "spot": round(res.spot, 4),
        "forward": round(res.forward, 4),
        "discount": round(res.discount, 6),
        "mean": round(mean, 4),
        # a clean risk-neutral density integrates to the forward; this is the
        # key quality check on the tails
        "mean/fwd-1%": round(100 * (mean / res.forward - 1), 2),
        "std": round(std, 4),
        "std_pct": round(100 * std / mean, 2) if mean else np.nan,
        "p05": round(quantile(0.05), 2),
        "p25": round(quantile(0.25), 2),
        "median": round(quantile(0.50), 2),
        "p75": round(quantile(0.75), 2),
        "p95": round(quantile(0.95), 2),
        "P(>spot)": round(prob_above(res.spot), 4),
        "P(>1.1*spot)": round(prob_above(1.1 * res.spot), 4),
        "P(<0.9*spot)": round(float(np.interp(0.9 * res.spot, g, res.cdf)), 4),
        "n_quotes": res.n_quotes,
    }


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    # first Friday
    d += dt.timedelta(days=(4 - d.weekday()) % 7)
    return d + dt.timedelta(days=14)


def _ca_bundle() -> str | None:
    for p in (os.environ.get("REQUESTS_CA_BUNDLE"),
              os.environ.get("SSL_CERT_FILE"),
              "/root/.ccr/ca-bundle.crt"):
        if p and os.path.exists(p):
            return p
    return None


def fetch_yahoo(expiries: list[dt.date], asof: dt.date) -> list[ExpiryChain]:
    """Pull the IBIT chain from Yahoo Finance for the given expiries."""
    import requests

    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
    sess = requests.Session()
    sess.headers.update({"User-Agent": ua})
    verify = _ca_bundle() or True

    # Prime cookies + crumb (Yahoo rejects the options endpoint without them).
    sess.get("https://fc.yahoo.com", verify=verify, timeout=30)
    crumb = sess.get("https://query2.finance.yahoo.com/v1/test/getcrumb",
                     verify=verify, timeout=30).text.strip()

    want = {e: int(dt.datetime(e.year, e.month, e.day,
                               tzinfo=dt.timezone.utc).timestamp()) for e in expiries}
    chains = []
    for exp, ts in want.items():
        url = (f"https://query2.finance.yahoo.com/v7/finance/options/{TICKER}"
               f"?date={ts}&crumb={crumb}")
        r = sess.get(url, verify=verify, timeout=30)
        r.raise_for_status()
        result = r.json()["optionChain"]["result"][0]
        spot = float(result["quote"]["regularMarketPrice"])
        opt = result["options"][0]
        chain = ExpiryChain(expiry=exp, asof=asof, spot=spot)
        for row in opt.get("calls", []):
            chain.calls[float(row["strike"])] = _quote(
                row.get("bid"), row.get("ask"), row.get("lastPrice"), row.get("openInterest"))
        for row in opt.get("puts", []):
            chain.puts[float(row["strike"])] = _quote(
                row.get("bid"), row.get("ask"), row.get("lastPrice"), row.get("openInterest"))
        chains.append(chain)
    return chains


def load_csv(path: str, expiries: list[dt.date], asof: dt.date) -> list[ExpiryChain]:
    """Load the chain from a snapshot in the optsnap schema."""
    opener = pd.read_csv
    df = opener(path)
    df = df[df["symbol"].str.upper() == TICKER].copy()
    if df.empty:
        raise ValueError(f"{path} has no {TICKER} rows")
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    want = set(expiries)
    chains = []
    for exp in expiries:
        sub = df[df["expiry"] == exp]
        if sub.empty:
            continue
        spot = float(sub["spot"].iloc[0])
        chain = ExpiryChain(expiry=exp, asof=asof, spot=spot)
        for _, row in sub.iterrows():
            k = float(row["strike"])
            q = _quote(row.get("bid"), row.get("ask"), row.get("last"), row.get("oi"))
            if str(row["right"]).upper().startswith("C"):
                chain.calls[k] = q
            else:
                chain.puts[k] = q
        chains.append(chain)
    if not chains:
        raise ValueError(f"none of the requested expiries {sorted(want)} found in {path}")
    return chains


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot(results: list[DensityResult], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, res in enumerate(results):
        color = cmap(i / max(len(results) - 1, 1))
        label = f"{res.expiry:%b %d %Y}  ({int(round(res.T*365))}d)"
        ax1.plot(res.grid, res.density, color=color, label=label, lw=1.8)
        ax2.plot(res.grid, res.cdf, color=color, label=label, lw=1.8)
    spot = results[0].spot
    for ax in (ax1, ax2):
        ax.axvline(spot, color="0.4", ls="--", lw=1, label=f"spot {spot:.2f}")
        ax.set_xlabel(f"{TICKER} price at expiry ($)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax1.set_title(f"{TICKER} risk-neutral density (Breeden-Litzenberger)")
    ax1.set_ylabel("probability density")
    ax2.set_title(f"{TICKER} risk-neutral CDF")
    ax2.set_ylabel("cumulative probability")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="read the chain from an optsnap-schema CSV instead of Yahoo")
    p.add_argument("--asof", type=lambda s: dt.date.fromisoformat(s),
                   default=DEFAULT_ASOF, help="valuation date (YYYY-MM-DD); default today")
    p.add_argument("--outdir", default="ibit_pdf_out", help="output directory")
    p.add_argument("--grid-points", type=int, default=801)
    p.add_argument("--smooth", type=float, default=None,
                   help="smile smoothing: vol-points of noise the fit may absorb "
                        "(default 0.01 = 1 vol pt; larger = smoother)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    expiries = [third_friday(y, m) for (y, m) in DEFAULT_MONTHS]

    print(f"IBIT risk-neutral density  |  asof {args.asof}")
    print("monthly expiries:", ", ".join(e.isoformat() for e in expiries))
    print("source:", args.csv if args.csv else "Yahoo Finance (live)")
    print()

    if args.csv:
        chains = load_csv(args.csv, expiries, args.asof)
    else:
        chains = fetch_yahoo(expiries, args.asof)

    os.makedirs(args.outdir, exist_ok=True)
    results, rows = [], []
    for chain in chains:
        try:
            res = compute_density(chain, grid_points=args.grid_points, smooth=args.smooth)
        except ValueError as e:
            print(f"  skip {chain.expiry}: {e}", file=sys.stderr)
            continue
        results.append(res)
        rows.append(summarize(res))
        out = os.path.join(args.outdir, f"ibit_pdf_{res.expiry.isoformat()}.csv")
        pd.DataFrame({"price": res.grid, "density": res.density, "cdf": res.cdf}).to_csv(
            out, index=False)

    if not results:
        print("no densities computed", file=sys.stderr)
        return 1

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.outdir, "ibit_pdf_summary.csv")
    summary.to_csv(summary_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    print(f"\nper-expiry grids + summary written to {args.outdir}/")

    if not args.no_plot:
        png = os.path.join(args.outdir, "ibit_pdf.png")
        plot(results, png)
        print(f"plot written to {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
