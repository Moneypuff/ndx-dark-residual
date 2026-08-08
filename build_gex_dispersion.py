#!/usr/bin/env python3
"""
Build the GEX x dispersion barometer (docs/gex_dispersion.html).

Two dials, one tape:

  * GAMMA -- SqueezeMetrics' public daily SPX gamma exposure (GEX, $ per 1%
    move; the free ancestor of the gated GEX+ series), ranked against its own
    trailing year so the dollar drift of a growing options market washes out.
  * CORRELATION -- how much S&P 500 stocks move together. We compute our OWN
    1-month (21-session) realized correlation with Cboe's implied-correlation
    weighting -- the top-50 IVV constituents by index weight -- via the basket
    identity

        rho = (sigma_P^2 - sum(w_i^2 sigma_i^2))
              / ((sum(w_i sigma_i))^2 - sum(w_i^2 sigma_i^2))

    and a realized dispersion analog  sqrt(sum(w_i sigma_i^2) - sigma_P^2),
    then lay both against Cboe's published gauges (COR1M implied correlation,
    DSPX implied dispersion) as an external cross-check.

Crossing the two dials at their midpoints gives four regimes ("quadrants"),
each scored by the SPX return AND realized volatility over the following 21
sessions -- the 1-month horizon where gamma positioning has its bite.

Data sources (all free, all fetched with a day-stamped cache):
  * https://squeezemetrics.com/monitor/static/DIX.csv      (SPX close, DIX, GEX; 2011->)
  * https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv  (2006->)
  * https://cdn.cboe.com/api/global/us_indices/daily_prices/DSPX_History.csv   (2014->)
  * iShares IVV holdings (constituents + weights) and Yahoo daily adjusted
    closes via the repo's own loaders in ndx_dark_residual.py.

The HTML shell lives in gex_dispersion_template.html with three placeholders
(/*__SER__*/, /*__QUAD__*/ and /*__META__*/). Mirrors build_comovement.py.

    python build_gex_dispersion.py --docs-out docs/gex_dispersion.html --cache-dir .ndx_dark_cache
"""
import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N

SQUEEZE_CSV_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
CBOE_HISTORY_TMPL = ("https://cdn.cboe.com/api/global/us_indices/daily_prices/"
                     "{sym}_History.csv")

TOP_N = 50            # Cboe's implied-correlation indices use the top 50 SPX names
WINDOW = 21           # 1 month of sessions -- the vol window AND the outcome horizon
GEX_PCT_WIN = 252     # trailing-year window for the GEX percentile (kills dollar drift)
GEX_PCT_MIN = 126     # half a year of history before the percentile is defined
MIN_NAMES = 35        # a day needs >= this many of the top-50 with a full vol window
ANN = np.sqrt(252.0)

QUAD_LABEL = {
    "HL": "Pinned & dispersed",
    "HH": "Pinned but synchronized",
    "LH": "Unclamped & synchronized",
    "LL": "Unclamped & dispersed",
}


# ----------------------------------------------------------------------------
# Fetch layer: day-stamped text cache with stale fallback
# ----------------------------------------------------------------------------
def fetch_text_cached(url, cache_path, refresh=False, label="doc", session=None,
                      retries=3, pause=1.0):
    """Body of `url` as text, cached at `cache_path` with a same-day stamp.
    A fresh cache short-circuits the fetch; a failed fetch falls back to any
    stale cache rather than killing the build. Returns None only when there is
    neither a live body nor a cached one."""
    import time as _time
    cache = Path(cache_path) if cache_path else None
    stamp = cache.with_suffix(cache.suffix + ".stamp") if cache else None
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    if cache and not refresh and cache.exists() and stamp is not None and stamp.exists():
        if stamp.read_text().strip() == today:
            print(f"{label}: reusing today's cache", file=sys.stderr)
            return cache.read_text(encoding="utf-8")
    if N.requests is None:
        return cache.read_text(encoding="utf-8") if cache and cache.exists() else None
    get = (session or N.requests).get
    last = None
    for a in range(retries):
        try:
            r = get(url, timeout=60, headers={"User-Agent": N._YF_UA})
            if r.status_code == 200 and r.text and not r.text.lstrip().lower().startswith("<html"):
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(r.text, encoding="utf-8")
                    stamp.write_text(today)
                return r.text
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        _time.sleep(pause * (a + 1))
    if cache and cache.exists():
        print(f"  ! {label}: fetch failed ({last}); using stale cache", file=sys.stderr)
        return cache.read_text(encoding="utf-8")
    print(f"  ! {label}: fetch failed ({last}); no cache to fall back on", file=sys.stderr)
    return None


# ----------------------------------------------------------------------------
# Parsers (pure -- unit tested)
# ----------------------------------------------------------------------------
def parse_squeeze_csv(text):
    """SqueezeMetrics DIX.csv -> DataFrame indexed by date with columns
    price (SPX close), dix, gex ($). Empty frame if the body is unusable."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    need = {"date", "price", "dix", "gex"}
    if not need.issubset({c.strip().lower() for c in df.columns}):
        return pd.DataFrame()
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    for c in ("price", "dix", "gex"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["price", "dix", "gex"]].dropna(how="all")


def parse_cboe_csv(text, value_col=None):
    """A Cboe *_History.csv -> daily Series. The files come in two shapes:
    OHLC (DATE,OPEN,HIGH,LOW,CLOSE -- e.g. COR1M) and single-value
    (DATE,DSPX). `value_col` picks a column by name; default is CLOSE when
    present, else the last column. Empty Series if unusable."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:  # noqa: BLE001
        return pd.Series(dtype="float64")
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" not in df.columns or len(df.columns) < 2:
        return pd.Series(dtype="float64")
    col = (value_col.upper() if value_col else
           ("CLOSE" if "CLOSE" in df.columns else df.columns[-1]))
    if col not in df.columns:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx)
    return s[s.index.notna()].dropna().sort_index()


# ----------------------------------------------------------------------------
# Computation (pure -- unit tested)
# ----------------------------------------------------------------------------
def top_weights(tickers, weight_map, n=TOP_N):
    """Normalized weight Series over the `n` heaviest names that actually have
    a weight, plus the share of the index those names covered pre-normalization."""
    w = pd.Series({t: weight_map[t] for t in tickers if weight_map.get(t)},
                  dtype="float64").sort_values(ascending=False).head(n)
    if w.empty:
        return w, 0.0
    coverage = float(w.sum())
    return w / w.sum(), coverage


def realized_cor_disp(returns, w, window=WINDOW, min_names=MIN_NAMES):
    """Realized implied-correlation analog and dispersion from a wide daily
    return panel (`returns`: dates x names, decimal) and index weights `w`.

    Per day, weights are renormalized over the names whose trailing `window`
    vol exists (IPOs enter the basket as their history accrues). Returns a
    DataFrame with columns:
        cor   -- basket-identity average pairwise correlation, in % (0-100 scale)
        disp  -- sqrt(sum(w sigma_i^2) - sigma_P^2), annualized vol points
        n     -- names contributing that day (NaN rows where n < min_names)
    """
    cols = [c for c in w.index if c in returns.columns]
    R = returns[cols]
    wv = w[cols]

    vol = R.rolling(window, min_periods=window).std() * ANN
    have = vol.notna()
    W = have.mul(wv, axis=1)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0)

    # basket return, weighted over the names with data each day (so its own
    # trailing vol exists as soon as the single-name vols do), then its vol
    Wr = R.notna().mul(wv, axis=1)
    Wr = Wr.div(Wr.sum(axis=1).replace(0, np.nan), axis=0)
    rp = (R.fillna(0.0) * Wr).sum(axis=1, min_count=1)
    sp = rp.rolling(window, min_periods=window).std() * ANN

    v = vol.fillna(0.0)
    s1 = (W * v).sum(axis=1)              # sum(w sigma)
    s2 = ((W ** 2) * (v ** 2)).sum(axis=1)  # sum(w^2 sigma^2)
    wv2 = (W * (v ** 2)).sum(axis=1)      # sum(w sigma^2)

    den = s1 ** 2 - s2
    cor = 100.0 * (sp ** 2 - s2) / den.where(den > 0)
    disp = 100.0 * np.sqrt((wv2 - sp ** 2).clip(lower=0.0))

    out = pd.DataFrame({"cor": cor.clip(0.0, 100.0), "disp": disp,
                        "n": have.sum(axis=1)})
    out.loc[out["n"] < min_names, ["cor", "disp"]] = np.nan
    return out


def rolling_pct(s, window=GEX_PCT_WIN, min_periods=GEX_PCT_MIN):
    """Percentile (0-100, mid-rank) of each value within its own trailing
    `window` observations -- no look-ahead, and level drift washes out."""
    def _pct(a):
        x = a[-1]
        if not np.isfinite(x):
            return np.nan
        h = a[np.isfinite(a)]
        return 100.0 * ((h < x).mean() + 0.5 * (h == x).mean())
    return s.rolling(window, min_periods=min_periods).apply(_pct, raw=True)


def full_pct(s):
    """Full-sample percentile (0-100, average rank on ties)."""
    return s.rank(pct=True) * 100.0


def forward_return(price, horizon=WINDOW):
    """% return over the NEXT `horizon` sessions (no look-ahead: t's value is
    realized strictly after t)."""
    return (price.shift(-horizon) / price - 1.0) * 100.0


def forward_vol(price, horizon=WINDOW):
    """Annualized realized vol (in vol points) of the NEXT `horizon` daily
    returns -- sessions t+1 .. t+horizon."""
    r = price.pct_change()
    return (r.rolling(horizon, min_periods=horizon).std() * ANN * 100.0).shift(-horizon)


def quad_code(gex_pct, cor_pct, mid=50.0):
    """'H'/'L' for gamma then correlation, e.g. 'HL' = long-gamma, low-corr.
    Empty string where either dial is undefined."""
    if not (np.isfinite(gex_pct) and np.isfinite(cor_pct)):
        return ""
    return ("H" if gex_pct >= mid else "L") + ("H" if cor_pct >= mid else "L")


def quad_stats(df):
    """Per-quadrant scoreboard rows [code, label, days, mean r21, median r21,
    hit%, mean fwd vol, mean (fwd vol - trailing vol)], most frequent first."""
    rows = []
    for code, g in df[df["quad"] != ""].groupby("quad"):
        r = g["r21"].dropna()
        fv = g["fvol"].dropna()
        dv = (g["fvol"] - g["tvol"]).dropna()
        rows.append([
            code, QUAD_LABEL.get(code, code), int(len(g)),
            round(float(r.mean()), 2) if len(r) else None,
            round(float(r.median()), 2) if len(r) else None,
            round(float((r > 0).mean() * 100)) if len(r) else None,
            round(float(fv.mean()), 1) if len(fv) else None,
            round(float(dv.mean()), 1) if len(dv) else None,
        ])
    rows.sort(key=lambda x: -x[2])
    return rows


def series_corr(a, b, diff=0):
    """(pearson r, overlap n) of two Series aligned on their common dates;
    with diff>0, of their `diff`-day changes (comovement, not shared level)."""
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if diff:
        d = d.diff(diff).dropna()
    if len(d) < 30:
        return None, int(len(d))
    return round(float(d["a"].corr(d["b"])), 3), int(len(d))


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
def rnd(s, nd=2):
    return [None if pd.isna(x) else round(float(x), nd) for x in s.values]


def build_payloads(F, cor1m, dspx, meta_extra):
    """SER / QUAD / META payload dicts from the aligned daily frame `F`
    (index: dates; columns gex, gexp, cor, corp, disp, spx, r21, fvol, tvol,
    quad) plus the full Cboe series for the overlay panes."""
    ser = {
        "dates": [d.strftime("%Y-%m-%d") for d in F.index],
        "gex": rnd(F["gex"] / 1e9, 3),          # $bn
        "gexp": rnd(F["gexp"], 1),
        "cor": rnd(F["cor"], 2),
        "corp": rnd(F["corp"], 1),
        "cor1m": rnd(cor1m.reindex(F.index), 2),
        "disp": rnd(F["disp"], 2),
        "dspx": rnd(dspx.reindex(F.index), 2),
        "spx": rnd(F["spx"], 2),
        "r21": rnd(F["r21"], 2),
        "quad": list(F["quad"]),
    }
    quad = quad_stats(F)

    r = F["r21"].dropna()
    fv = F["fvol"].dropna()
    last = F.index[-1]
    cur = F.iloc[-1]
    c_lvl, c_n = series_corr(F["cor"], cor1m)
    c_chg, _ = series_corr(F["cor"], cor1m, diff=WINDOW)
    d_lvl, d_n = series_corr(F["disp"], dspx)
    d_chg, _ = series_corr(F["disp"], dspx, diff=WINDOW)

    meta = {
        "range": f"{F.index[0].strftime('%Y-%m-%d')} → {last.strftime('%Y-%m-%d')}",
        "n_days": int(len(F)),
        "n_indep": int(round(len(F) / WINDOW)),
        "built": "built " + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "baseline": {
            "mean": round(float(r.mean()), 2), "med": round(float(r.median()), 2),
            "hit": round(float((r > 0).mean() * 100)),
            "fvol": round(float(fv.mean()), 1),
        },
        "today": {
            "date": last.strftime("%Y-%m-%d"),
            "gex": round(float(cur["gex"]) / 1e9, 2),
            "gexp": round(float(cur["gexp"]), 1) if pd.notna(cur["gexp"]) else None,
            "cor": round(float(cur["cor"]), 1) if pd.notna(cur["cor"]) else None,
            "corp": round(float(cur["corp"]), 1) if pd.notna(cur["corp"]) else None,
            "cor1m": (round(float(cor1m.loc[:last].iloc[-1]), 2) if len(cor1m.loc[:last]) else None),
            "dspx": (round(float(dspx.loc[:last].iloc[-1]), 2) if len(dspx.loc[:last]) else None),
            "disp": round(float(cur["disp"]), 1) if pd.notna(cur["disp"]) else None,
            "quad": cur["quad"], "quadLabel": QUAD_LABEL.get(cur["quad"], "—"),
        },
        "cmp": {"cor_r": c_lvl, "cor_dr": c_chg, "cor_n": c_n,
                "dsp_r": d_lvl, "dsp_dr": d_chg, "dsp_n": d_n},
    }
    meta.update(meta_extra)

    # findings figures, all derived from the table so the prose never drifts
    by = {q[0]: q for q in quad}

    def f(code, i):
        return by[code][i] if code in by and by[code][i] is not None else None

    pct = lambda v: None if v is None else f"{'+' if v >= 0 else ''}{v:.2f}%"
    meta["find"] = {
        "dangerRet": pct(f("LH", 3)), "dangerVol": f("LH", 6), "dangerHit": f("LH", 5),
        "calmRet": pct(f("HL", 3)), "calmVol": f("HL", 6), "calmHit": f("HL", 5),
        "volSpread": (round(by["LH"][6] - by["HL"][6], 1)
                      if f("LH", 6) is not None and f("HL", 6) is not None else None),
        "corR": c_lvl, "corDR": c_chg, "dspR": d_lvl,
    }
    return ser, quad, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="gex_dispersion_template.html")
    ap.add_argument("--docs-out", default="docs/gex_dispersion.html")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--top", type=int, default=TOP_N,
                    help=f"basket size for the realized gauges (default {TOP_N}, "
                         "matching Cboe's implied-correlation universe)")
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download of the GEX / Cboe / holdings docs")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = (lambda name: cache_dir / name if cache_dir else None)
    session = N.make_session(4) if N.requests else None

    # ---- GEX (+ SPX close) --------------------------------------------------
    body = fetch_text_cached(SQUEEZE_CSV_URL, cpath("gexdisp_squeeze.csv"),
                             refresh=args.refresh, label="SqueezeMetrics GEX",
                             session=session)
    G = parse_squeeze_csv(body or "")
    if G.empty:
        raise SystemExit("No usable GEX series (SqueezeMetrics fetch failed and no cache).")
    print(f"GEX: {len(G)} days [{G.index.min().date()} -> {G.index.max().date()}]",
          file=sys.stderr)

    # ---- Cboe reference gauges ---------------------------------------------
    cor1m = parse_cboe_csv(fetch_text_cached(
        CBOE_HISTORY_TMPL.format(sym="COR1M"), cpath("gexdisp_cor1m.csv"),
        refresh=args.refresh, label="Cboe COR1M", session=session) or "")
    dspx = parse_cboe_csv(fetch_text_cached(
        CBOE_HISTORY_TMPL.format(sym="DSPX"), cpath("gexdisp_dspx.csv"),
        refresh=args.refresh, label="Cboe DSPX", session=session) or "")
    print(f"COR1M: {len(cor1m)} days   DSPX: {len(dspx)} days", file=sys.stderr)

    # ---- top-50 basket: holdings + weights (with stale-cache fallback) ------
    wcache = cpath("gexdisp_weights.json")
    tickers, weights = N.fetch_ishares_holdings(N.IVV_PORTFOLIO_ID,
                                                label="IVV S&P 500",
                                                session=session,
                                                return_weights=True)
    if tickers and weights:
        if wcache:
            wcache.write_text(json.dumps({"tickers": tickers, "weights": weights}))
    elif wcache and wcache.exists():
        print("  ! IVV holdings fetch failed; using cached holdings", file=sys.stderr)
        d = json.loads(wcache.read_text())
        tickers, weights = d["tickers"], d["weights"]
    if not tickers or not weights:
        raise SystemExit("No IVV holdings/weights available (fetch failed, no cache).")
    w, coverage = top_weights(tickers, weights, n=args.top)
    print(f"Basket: top {len(w)} IVV names cover {coverage:.1f}% of the index",
          file=sys.stderr)

    # ---- constituent prices -> realized correlation & dispersion ------------
    start = (G.index.min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().normalize()
    panels = N.load_yahoo_panels(list(w.index), start, end,
                                 cache_dir=args.cache_dir or None,
                                 refresh=args.refresh, label="GEXDISP")
    adj = panels["adjclose"].dropna(how="all")
    returns = adj.pct_change(fill_method=None)
    RC = realized_cor_disp(returns, w, window=WINDOW, min_names=MIN_NAMES)
    print(f"Realized gauges: {RC['cor'].notna().sum()} defined days", file=sys.stderr)

    # ---- aligned frame, dials, outcomes -------------------------------------
    F = G.join(RC[["cor", "disp"]], how="left").rename(columns={"price": "spx"})
    F["gexp"] = rolling_pct(F["gex"])
    F["corp"] = full_pct(F["cor"])
    # outcomes computed on the full series BEFORE trimming, so late rows still
    # see their (already realized) futures where possible
    F["r21"] = forward_return(F["spx"])
    F["fvol"] = forward_vol(F["spx"])
    F["tvol"] = (F["spx"].pct_change().rolling(WINDOW, min_periods=WINDOW).std()
                 * ANN * 100.0)
    F = F[F["gexp"].notna() & F["cor"].notna()].copy()
    if F.empty:
        raise SystemExit("No days where both dials are defined -- nothing to build.")
    F["quad"] = [quad_code(g, c) for g, c in zip(F["gexp"], F["corp"])]

    ser, quad, meta = build_payloads(
        F, cor1m, dspx,
        {"basket": f"top {len(w)} IVV names · {coverage:.0f}% of index weight",
         "top_n": int(len(w))})

    body = (Path(args.template).read_text(encoding="utf-8")
            .replace("/*__SER__*/", json.dumps(ser, separators=(",", ":")))
            .replace("/*__QUAD__*/", json.dumps(quad, separators=(",", ":")))
            .replace("/*__META__*/", json.dumps(meta, separators=(",", ":"))))
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>GEX × Dispersion Barometer</title></head>'
           '<body>\n' + body + '\n</body></html>\n')
    out = Path(args.docs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc)//1024} KB, {len(F)} days, "
          f"{len(quad)} quadrants, today: {meta['today']['quad'] or '—'})")


if __name__ == "__main__":
    main()
