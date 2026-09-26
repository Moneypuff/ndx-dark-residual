#!/usr/bin/env python3
"""
Relative-darkness study
=======================

Does name-level dark flow carry an edge once it is measured properly -- and why does
the index-level DIX look like it works while the single-name residual does not?
Reproduces DARK_FLOW_FINDINGS.md end to end from FINRA + Yahoo (no key).

Sections
--------
1. Index level   the universe's reconstructed dollar-DIX (and, when reachable,
                 SqueezeMetrics' published DIX, 2011->) vs the index's forward returns:
                 raw level and detrended, alone and with realized-vol / past-return
                 controls, Newey-West; then why it looks predictive -- the same slope
                 with implied vol (VIX/VXN) as a control, a vol-regime x gauge table of
                 forward returns, and the gauge's lead-lag profile against the index.
2. Structure     variance decomposition of per-name DPI (structural / common /
                 idiosyncratic) and the "price echo": how abnormal DPI co-moves with the
                 name's own returns, vs how the common component co-moves with the market.
3. Signals       the current methodology (raw 1-day DPI, 5-day D, the grid's rolling-OLS
                 residual vs the DIX, the L/S tab's DPI-minus-expanding-mean) against the
                 relative-darkness measures in dark_flow.py, in weekly Fama-MacBeth
                 cross-sections of forward returns (entry at the close AFTER the FINRA file
                 posts), alone and with reversal / momentum / size / volatility / volume
                 controls, full sample and both halves, plus a quintile long-short.
   Robustness    other signal windows (1/10/20 sessions), tail-only events (|z| > 2),
                 and the conditional "dark accumulation into weakness" double sort.
4. Volatility    the same signals against FUTURE realized volatility, controlling for
                 past volatility, volume and absolute returns.
6. By name       (--by-name) does DPI work better on certain names? Each name's own
                 time-series slope vs a circular-shift placebo, persistence across halves,
                 out-of-sample name selection (with its own placebo), characteristic
                 interactions (size, off-exchange share, structural DPI level, volatility,
                 DPI persistence, price echo), and an NDX-100 name table.
5. ATS (--ats)   FINRA's weekly OTC-transparency totals split off-exchange volume into
                 ATS dark pools (institutional) vs wholesaler internalization (retail):
                 the most literal "relative darkness". Abnormal ATS share / volume vs
                 forward returns, entered after FINRA's publication date (~3 weeks after
                 the week starts).

Usage
-----
    python dark_flow_study.py --cache-dir ~/.ndx_dark_cache \\
        --summary-out dark_flow_summary.txt --csv-out dark_flow_signals.csv
    python dark_flow_study.py --universe ndx        # NDX-100 only (thin: ~100 names)
    python dark_flow_study.py --universe russell    # IWM holdings, liquid names only
    python dark_flow_study.py --no-sqz              # skip the SqueezeMetrics DIX download
    python dark_flow_study.py --ats                 # add the ATS-vs-internalizer section
    python dark_flow_study.py --by-name             # add the per-name heterogeneity section

Universes use CURRENT index membership (survivorship: names that left the index are
missing), which flatters average returns but matters far less for the cross-sectional
signal spreads tested here.
"""
import argparse
import io
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import dark_flow as F
import ndx_dark_residual as N

SQZ_DIX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
FINRA_OTC_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
OTC_CACHE = "finra_otc_weekly.csv"
OTC_START = "2021-06-01"      # the public API currently serves symbol totals from Dec 2021 on
WINDOW, BASE = 5, 126          # signal window / baseline window (sessions)
HORIZONS = (5, 21)             # forward-return horizons (sessions), entered at t+1
MIN_DOLLAR_ADV = 5e6           # russell screen: trailing-60d $ADV floor ...
MIN_PRICE = 3.0                # ... and price floor
SIGNALS = [
    # key, label, family
    ("dpi_1d", "raw 1-day DPI (dashboard scatter / decile x-axis)", "current"),
    ("d_5d", "D = 5-day mean of daily DPI (the repo's D)", "current"),
    ("resid_reg", "grid residual: rolling OLS of D on the dollar-DIX", "current"),
    ("rel_expanding", "L/S tab 'name-specific': DPI minus expanding mean", "current"),
    ("dpi_level", "structural level: trailing 126-session mean DPI", "new"),
    ("dpi_z", "abnormal 5d volume-weighted DPI, own-sigma units", "new"),
    ("dark_share_abn", "abnormal off-exchange share of volume (log)", "new"),
    ("dark_imbalance", "abnormal dark buying in ADV units", "new"),
    ("dpi_hidden", "dpi_z minus its price echo (dark buying price hasn't shown)", "new"),
]
CONTROLS = ("ret_5d", "ret_21d", "mom_12_1", "rv_21d", "log_dollar_adv", "vol_surprise")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def universe_symbols(universe):
    """(constituent symbols, index proxy ETF whose returns the index-level test uses)."""
    if universe == "ndx":
        return [t for t in N.NDX100], "QQQ"
    if universe == "russell":
        return N.fetch_ishares_holdings(N.IWM_PORTFOLIO_ID, label="IWM Russell 2000"), "IWM"
    sp = N.fetch_ishares_holdings(N.IVV_PORTFOLIO_ID, label="IVV S&P 500")
    return list(dict.fromkeys(sp + N.NDX100)), "SPY"


def load_sqz_dix(timeout=60):
    """SqueezeMetrics' public DIX.csv (SPX price + DIX, 2011->) or None if unreachable."""
    try:
        r = N.make_session(1).get(SQZ_DIX_URL, timeout=timeout, headers={"User-Agent": N._YF_UA})
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["date"]).set_index("date").sort_index()
        return df if {"price", "dix"} <= set(df.columns) else None
    except Exception:  # noqa: BLE001
        return None


_SS_NS = "{urn:schemas-microsoft-com:office:spreadsheet}"
# the dashboard's static NDX labels -> iShares' GICS sector names
_NDX_SECTOR_ALIASES = {"Technology": "Information Technology", "Cons. Discretionary": "Consumer Discretionary",
                       "Cons. Staples": "Consumer Staples", "Comm. Services": "Communication"}


def sectors_from_spreadsheetml(text):
    """{ticker: sector} from an iShares SpreadsheetML holdings document's 'Sector' column
    (equity rows only); {} when the body isn't parseable or has no such column."""
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", text or "")
    try:
        root = ET.fromstring(text)
    except Exception:  # noqa: BLE001
        return {}

    def cells(row):
        out, col = {}, 0
        for c in row.findall(_SS_NS + "Cell"):
            i = c.get(_SS_NS + "Index")
            col = int(i) if i else col + 1
            d = c.find(_SS_NS + "Data")
            out[col] = d.text if d is not None else None
        return out
    rows = [cells(r) for r in root.iter(_SS_NS + "Row")]
    hi = next((i for i, r in enumerate(rows) if "Ticker" in r.values()), None)
    if hi is None:
        return {}
    hdr = rows[hi]
    tcol = next(k for k, v in hdr.items() if v == "Ticker")
    scol = next((k for k, v in hdr.items() if v == "Sector"), None)
    acol = next((k for k, v in hdr.items() if v == "Asset Class"), None)
    if scol is None:
        return {}
    out = {}
    for r in rows[hi + 1:]:
        t = (r.get(tcol) or "").strip().upper()
        if not N._TICKER_RE.match(t) or (acol and (r.get(acol) or "").strip().lower() != "equity"):
            continue
        if (r.get(scol) or "").strip():
            out[t] = r[scol].strip()
    return out


def universe_sectors(universe):
    """{ticker: GICS sector} for the universe: iShares holdings' Sector column, with the
    dashboard's static NDX map filling any NDX-only names. {} if the holdings can't be read."""
    out = {}
    pid = {"spx_ndx": N.IVV_PORTFOLIO_ID, "russell": N.IWM_PORTFOLIO_ID}.get(universe)
    if pid:
        try:
            r = N.make_session(1).get(N.ISHARES_DOC_TMPL.format(pid=pid), timeout=60,
                                      headers={"User-Agent": N._YF_UA})
            if r.status_code == 200:
                out = sectors_from_spreadsheetml(r.text)
        except Exception:  # noqa: BLE001
            out = {}
    if universe in ("spx_ndx", "ndx"):
        for t, sec in N.TICKER_SECTOR.items():
            out.setdefault(t, _NDX_SECTOR_ALIASES.get(sec, sec))
    return out


def build_signals(P, names, window=WINDOW, base=BASE):
    """All signal and control panels (dates x names) from build_universe_panels output."""
    sh, tv = P["short"][names], P["total"][names]
    ok = P["total_raw"][names] >= 1000                      # ignore near-empty prints
    sh, tv = sh.where(ok), tv.where(ok)
    vol = P["volume"][names].where(P["volume"][names] > 0)
    # Yahoo occasionally serves non-positive adjusted closes on old bars of names with large
    # special dividends; treat them as missing rather than letting log() produce garbage.
    close = P["close"][names].where(P["close"][names] > 0)
    adj = P["adjclose"][names].where(P["adjclose"][names] > 0)
    lret = np.log(adj).diff()

    dpi1 = (sh / tv.replace(0, np.nan)).clip(0, 1)
    d5 = dpi1.rolling(5, min_periods=1).mean()
    dix = N.compute_dollar_dix(sh, tv, close)                   # the universe's own dollar-DIX
    reg = N.compute_residuals(d5, "DIX", window=126, min_periods=40,
                              bench_series=dix.rolling(5, min_periods=1).mean())["reg"]
    _, z = F.abnormal_dpi(sh, tv, window=window, base=base)
    ret_1d = lret * 100
    ret_k = (np.log(adj) - np.log(adj).shift(window)) * 100
    _, hidden = F.price_echo_split(z, ret_1d, ret_k)
    adv_dollar = (vol * close).rolling(60, min_periods=30).mean()
    sig = {
        "dpi_1d": dpi1,
        "d_5d": d5,
        "resid_reg": reg,
        "rel_expanding": dpi1 - dpi1.expanding(min_periods=20).mean(),
        "dpi_level": dpi1.rolling(base, min_periods=base // 2).mean(),
        "dpi_z": z,
        "dark_share_abn": F.abnormal_dark_share(tv, vol, window=window, base=base),
        "dark_imbalance": F.dark_imbalance(sh, tv, vol, window=window, base=base),
        "dpi_hidden": hidden,
    }
    ctl = {
        "ret_5d": ret_k,
        "ret_21d": (np.log(adj) - np.log(adj).shift(21)) * 100,
        "mom_12_1": (np.log(adj.shift(21)) - np.log(adj.shift(252))) * 100,
        "rv_21d": lret.rolling(21, min_periods=15).std() * np.sqrt(252) * 100,
        "log_dollar_adv": np.log(adv_dollar.where(adv_dollar > 0)),
        "vol_surprise": np.log(vol.rolling(window, min_periods=window - 1).sum()
                               / (window * vol.shift(window).rolling(base, min_periods=base // 2).mean())),
    }
    return sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, (sh, tv, vol)


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------
def fmt(c, t):
    return "      --      " if not np.isfinite(c) else f"{c:+.3f} ({t:+.1f})"


IMPLIED_VOL = {"spx_ndx": "^VIX", "ndx": "^VXN", "russell": "^VIX"}   # Yahoo has no RVX


def load_implied_vol(symbol, start, cache_dir=None):
    """Daily close of an implied-vol index (VIX / VXN) from Yahoo, or None if unavailable."""
    try:
        px = N.load_yahoo_panels([symbol], start, pd.Timestamp.today().normalize(), workers=1,
                                 cache_dir=cache_dir, label="implied vol")["close"]
        s = px[symbol.upper()].dropna() if symbol.upper() in px else None
        return s if s is not None and len(s) > 500 else None
    except Exception:  # noqa: BLE001
        return None


def mechanism_lines(lines, label, gauge, price, iv, ivname):
    """Why the gauge looks predictive: its slope with/without implied vol, the vol x gauge
    return table, and its lead-lag profile against the index."""
    M = F.index_mechanism(gauge, price, iv)
    lines.append(f"   Why it looks predictive -- {label} (implied vol = {ivname}; corr of the 5d gauge with "
                 f"{ivname} {M['corr_iv']:+.2f}, detrended {M['corr_iv_detrended']:+.2f}):")
    for h, g in M["coef"].groupby("horizon"):
        cells = {r.spec: fmt(r.coef, r.t).strip() for r in g.itertuples()}
        lines.append(f"     h{h}: gauge alone {cells.get('gauge alone')} | + {ivname} {cells.get('gauge + implied vol')} | "
                     f"{ivname} alone {cells.get('implied vol alone')} | detrended {cells.get('detrended gauge')} | "
                     f"detrended + {ivname} {cells.get('detrended gauge + implied vol')}")
    T, C = M["table"], M["counts"]
    lines.append(f"     1-month forward return (%) by trailing-1y terciles, rows {ivname} low/mid/high, cols gauge low/mid/high:")
    for ix in T.index:
        lines.append(f"       {ivname} {ix:4s}: " + "  ".join(f"{T.loc[ix, c]:+.2f} (n={int(C.loc[ix, c])})" for c in T.columns))
    lines.append("     lead-lag corr(detrended gauge_t, index return_t+k): "
                 + "  ".join(f"k={k:+d} {v:+.3f}" for k, v in M["leadlag"].items()))
    full = M["coef"][(M["coef"]["horizon"] == 21) & (M["coef"]["spec"] == "gauge alone")]
    if len(full) and len(M["loo"]):
        worst = M["loo"].sort_values("coef").head(2)
        lines.append(f"     leave-one-year-out, gauge alone h21 (full {fmt(full['coef'].iloc[0], full['t'].iloc[0]).strip()}): "
                     + "; ".join(f"without {yr} {fmt(r.coef, r.t).strip()}" for yr, r in worst.iterrows())
                     + f"; median over years {M['loo']['coef'].median():+.3f}")
    return M


def dix_validation(fixed, naive, sqz):
    """How closely a reconstructed S&P-universe DIX tracks SqueezeMetrics' published DIX, for
    the split-fixed gauge vs the old pairing of as-traded FINRA shares with split-adjusted
    prices. Returns {label: (level MAE, corr of daily changes, n)}."""
    out = {}
    for label, g in (("old: as-traded shares x split-adjusted price", naive),
                     ("fixed: shares on Yahoo's split basis", fixed)):
        j = pd.DataFrame({"g": g, "s": sqz["dix"]}).dropna()
        out[label] = (float((j["g"] - j["s"]).abs().mean()), float(j["g"].diff().corr(j["s"].diff())), len(j))
    return out


def section_index(lines, dix, proxy_px, proxy, sqz, naive_dix=None, iv=None, ivname="VIX", vix=None):
    lines.append("1. INDEX LEVEL -- does the dark gauge predict the index? (pp of forward return "
                 "per 1 SD of the 5d gauge; Newey-West t, lags = horizon)")
    if sqz is not None and naive_dix is not None:
        lines.append("   Split-basis check vs SqueezeMetrics' published DIX (same S&P-style universe):")
        for label, (mae, dc, n) in dix_validation(dix, naive_dix, sqz).items():
            lines.append(f"     {label:46s} level MAE {mae:.4f}   corr of daily changes {dc:.3f}   (n={n})")
    feeds = [(f"reconstructed dollar-DIX vs {proxy}", dix, proxy_px)]
    if sqz is not None:
        feeds += [("SqueezeMetrics DIX vs SPX, full history", sqz["dix"], sqz["price"]),
                  ("SqueezeMetrics DIX vs SPX, FINRA window (2018-08->)",
                   sqz["dix"].loc["2018-08-01":], sqz["price"].loc["2018-08-01":])]
    rows = []
    for label, g, px in feeds:
        R = F.index_predictability(g, px)
        if R.empty:
            continue
        lines.append(f"   {label}  [{R['start'].min()} -> {R['end'].max()}]")
        lines.append(f"     {'h':>3}  {'level':>15} {'level+ctrl':>15} {'detrended':>15} {'detr+ctrl':>15}   rv21 t (ctrl)")
        for h, g2 in R.groupby("horizon"):
            cell = {(r.signal, r.controls): fmt(r.coef_pp_per_sd, r.t) for r in g2.itertuples()}
            rv = g2[(g2.signal == "level") & g2.controls]["t_rv21"]
            lines.append(f"     {h:>3}  {cell.get(('level', False), '--'):>15} {cell.get(('level', True), '--'):>15} "
                         f"{cell.get(('detrended', False), '--'):>15} {cell.get(('detrended', True), '--'):>15}"
                         f"   {float(rv.iloc[0]) if len(rv) else np.nan:+.1f}")
        R.insert(0, "feed", label)
        rows.append(R)
    if iv is not None and proxy_px is not None:
        mechanism_lines(lines, f"reconstructed dollar-DIX vs {proxy}", dix, proxy_px, iv, ivname)
    if vix is not None and sqz is not None:
        mechanism_lines(lines, "SqueezeMetrics DIX vs SPX, full history", sqz["dix"], sqz["price"], vix, "VIX")
    lines.append("")
    return pd.concat(rows) if rows else pd.DataFrame()


def section_structure(lines, dpi1, ret_1d):
    lines.append("2. STRUCTURE -- what per-name DPI is made of")
    dec = F.decompose_dpi(dpi1.loc["2018-08-01":])
    lines.append(f"   {dec['n_obs']:,} name-days, {dec['n_names']} names. Variance shares: structural "
                 f"level {dec['share_structural']:.1%} | common (market) {dec['share_common']:.1%} | "
                 f"idiosyncratic {dec['share_idio']:.1%}")
    lines.append(f"   SDs: structural {dec['sd_structural']:.3f}, common {dec['sd_common']:.4f}, idio "
                 f"{dec['sd_idio']:.4f}. Idio ACF " + ", ".join(f"lag{k} {v:+.2f}" for k, v in dec["acf"].items()))
    if "ar_rho" in dec:
        lines.append(f"   AR(1)+noise fit: {dec['ar_persistent_share']:.0%} of idio variance persistent, "
                     f"rho {dec['ar_rho']:.2f} (half-life {dec['ar_half_life']:.1f} sessions); the rest "
                     f"is day-to-day flow-composition noise")
    abn = dpi1 - dpi1.rolling(60, min_periods=20).mean()
    prof = F.echo_profile(abn.loc["2018-08-01":], ret_1d.loc["2018-08-01":])
    lines.append("   Price echo -- xs rank corr of abnormal DPI_t with the name's own return at t+k:  "
                 + "  ".join(f"k={k:+d}: {v:+.3f}" for k, v in prof.items()))
    common = abn.mean(axis=1)
    mkt = ret_1d.mean(axis=1)
    lines.append("   Common component vs equal-weight market return, same day: "
                 f"{common.corr(mkt):+.3f}  (the index-level relation has the OPPOSITE sign)")
    lines.append("")
    return dec, prof


def section_signals(lines, sig, ctl, adj, dates, min_names):
    lines.append("3. SIGNALS -- weekly Fama-MacBeth, forward log return entered at the close of t+1.")
    lines.append("   Cell = return spread (pp) from the lowest- to the highest-ranked name, Newey-West t;")
    lines.append("   '+ctrl' adds 5d/21d reversal, 12-1 momentum, 21d vol, $ADV and volume surprise.")
    Y = {h: F.forward_log_return(adj, h, lag=1) for h in HORIZONS}
    C = {k: F.xs_rank(v) for k, v in ctl.items()}
    mid = dates[len(dates) // 2]
    halves = (("1st half", dates[dates < mid]), ("2nd half", dates[dates >= mid]))
    hdr = "   " + f"{'signal':15s}" + "".join(f"{f'h{h} alone':>16}{f'h{h} +ctrl':>16}" for h in HORIZONS) \
        + "".join(f"{f'h21 +ctrl {lab}':>22}" for lab, _ in halves) + f"{'Q5-Q1 h5':>16}"
    lines.append(hdr)
    rows = []
    for key, label, fam in SIGNALS:
        S = F.xs_rank(sig[key])
        cells, rec = [], {"signal": key, "family": fam, "label": label}
        for h in HORIZONS:
            lags = max(1, h // 5)
            for ctrl in (False, True):
                X = {"s": S, **(C if ctrl else {})}
                res, _ = F.fama_macbeth(Y[h], X, dates, min_names=min_names, nw_lags=lags)
                c, t = res.loc["s", "coef"], res.loc["s", "t"]
                cells.append(fmt(c, t))
                rec[f"h{h}_{'ctrl' if ctrl else 'uni'}_coef"], rec[f"h{h}_{'ctrl' if ctrl else 'uni'}_t"] = c, t
        for lab, dd in halves:
            res, _ = F.fama_macbeth(Y[21], {"s": S, **C}, dd, min_names=min_names, nw_lags=4)
            c, t = res.loc["s", "coef"], res.loc["s", "t"]
            cells.append(fmt(c, t))
            rec[f"h21_ctrl_{lab.replace(' ', '')}_coef"], rec[f"h21_ctrl_{lab.replace(' ', '')}_t"] = c, t
        qs = F.quantile_spread(sig[key], Y[5], dates, min_names=min_names)
        qm, qt, _ = F.newey_west_t(qs, 1)
        rec["q5q1_h5_mean"], rec["q5q1_h5_t"] = qm, qt
        lines.append("   " + f"{key:15s}" + "".join(f"{c:>16}" for c in cells[:4])
                     + "".join(f"{c:>22}" for c in cells[4:]) + f"{fmt(qm, qt):>16}")
        rows.append(rec)
    lines.append(f"   ({len(dates)} weekly dates {dates.min().date()} -> {dates.max().date()}; halves split at "
                 f"{mid.date()})")
    for key, label, fam in SIGNALS:
        lines.append(f"     {key:15s} [{fam}] {label}")
    lines.append("")
    return pd.DataFrame(rows)


def section_robust(lines, sig, ctl, vols, adj, dates, min_names, mask=None):
    lines.append("3b. ROBUSTNESS -- giving the flow signals every chance")
    sh, tv, vol = vols
    msk = (lambda df: df.where(mask)) if mask is not None else (lambda df: df)  # noqa: E731
    Y = {h: F.forward_log_return(adj, h, lag=1) for h in HORIZONS}
    C = {k: F.xs_rank(v) for k, v in ctl.items()}
    for k in (1, 10, 20):
        cells = []
        for key, S in (("dpi_z", F.abnormal_dpi(sh, tv, window=k)[1]),
                       ("dark_imbalance", F.dark_imbalance(sh, tv, vol, window=k))):
            for h in HORIZONS:
                res, _ = F.fama_macbeth(Y[h], {"s": F.xs_rank(msk(S)), **C}, dates,
                                        min_names=min_names, nw_lags=max(1, h // 5))
                cells.append(f"{key} h{h} {fmt(res.loc['s', 'coef'], res.loc['s', 't'])}")
        lines.append(f"   window {k:>2} sessions, +ctrl: " + " | ".join(cells))
    z = sig["dpi_z"]

    def rel(h):
        # market-relative = demeaned over the cross-section actually being tested (names with a
        # signal that date), not the whole panel -- otherwise screened-out names shift every bin
        yy = Y[h].where(z.notna())
        return yy.sub(yy.mean(axis=1), axis=0).reindex(dates)
    for h in HORIZONS:
        y = rel(h)
        zz = z.reindex(dates)
        cells = []
        for lo, hi, lab in ((-np.inf, -2, "z<-2"), (-2, -1, "-2..-1"), (-1, 1, "-1..1"),
                            (1, 2, "1..2"), (2, np.inf, "z>2")):
            per = y.where((zz > lo) & (zz <= hi)).mean(axis=1)
            mu, t, _ = F.newey_west_t(per, max(1, h // 5))
            cells.append(f"{lab} {fmt(mu, t)}")
        lines.append(f"   tails of dpi_z, h{h} market-relative return: " + " | ".join(cells))
    rk = ctl["ret_5d"].reindex(dates).rank(axis=1, pct=True)
    zr = z.reindex(dates).rank(axis=1, pct=True)
    terc = ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0 + 1e-9))
    mid = dates[len(dates) // 2]
    for h in HORIZONS:
        y = rel(h)
        cells = []
        for (a, b), lab in zip(terc, ("losers", "middle", "winners")):
            inr = (rk > a) & (rk <= b)
            hi = y.where(inr & (zr > terc[2][0])).mean(axis=1)
            lo = y.where(inr & (zr <= terc[0][1])).mean(axis=1)
            spread = (hi - lo).dropna()
            mu, t, _ = F.newey_west_t(spread, max(1, h // 5))
            cell = f"{lab} {fmt(mu, t)}"
            if h == max(HORIZONS):
                halves = [F.newey_west_t(part, max(1, h // 5)) for part in
                          (spread[spread.index < mid], spread[spread.index >= mid])]
                cell += f" [halves {fmt(*halves[0][:2]).strip()} / {fmt(*halves[1][:2]).strip()}]"
            cells.append(cell)
        lines.append(f"   high-minus-low dpi_z within past-5d-return terciles, h{h}: " + " | ".join(cells))
    lines.append("")


def section_vol(lines, sig, base_ctl, adj, dates, min_names):
    lines.append("4. VOLATILITY -- signals vs FUTURE log realized vol (sessions t+2..t+1+h), controls: "
                 "5d & 21d log RV, volume surprise, $ADV, |1d| and |5d| return, 5d return")
    lret = np.log(adj.where(adj > 0)).diff()
    lrv = lambda s: np.log((s * np.sqrt(252)).where(s > 0))  # noqa: E731
    ctl = {"rv5": F.xs_rank(lrv(lret.rolling(5, min_periods=4).std())),
           "rv21": F.xs_rank(lrv(lret.rolling(21, min_periods=15).std())),
           "absr": F.xs_rank(lret.abs()), "absk": F.xs_rank((np.log(adj) - np.log(adj).shift(5)).abs()),
           "rk": F.xs_rank(np.log(adj) - np.log(adj).shift(5)),
           "vsurp": F.xs_rank(base_ctl["vol_surprise"]), "adv": F.xs_rank(base_ctl["log_dollar_adv"])}
    rows = []
    for h in HORIZONS:
        y = lrv(lret.rolling(h, min_periods=max(3, h - 2)).std().shift(-(1 + h)))
        for key in ("d_5d", "dpi_z", "dark_share_abn", "dark_imbalance", "resid_reg"):
            res, _ = F.fama_macbeth(y, {"s": F.xs_rank(sig[key]), **ctl}, dates,
                                    min_names=min_names, nw_lags=max(1, h // 5))
            rows.append({"signal": key, "horizon": h, "coef": res.loc["s", "coef"], "t": res.loc["s", "t"]})
    V = pd.DataFrame(rows)
    for key, g in V.groupby("signal", sort=False):
        lines.append(f"   {key:15s} " + "  ".join(f"h{r.horizon}: {fmt(r.coef, r.t)}" for r in g.itertuples())
                     + "   (log-RV spread, lowest -> highest ranked)")
    lines.append("")
    return V


# ----------------------------------------------------------------------------
# 6. By name -- does DPI work better on certain names?
# ----------------------------------------------------------------------------
def market_relative_fwd(adj, sig, h, dates):
    """Forward log return (entered at t+1) minus that date's mean over the names that have a
    signal -- the per-name outcome, free of the market's move."""
    y = F.forward_log_return(adj, h, lag=1).where(sig.notna())
    return y.sub(y.mean(axis=1), axis=0).reindex(dates)


def section_by_name(lines, sig, ctl, adj, dates, names, dpi1, ret_1d, vols, min_names,
                    k_placebo=100, seed=0, sector_map=None):
    lines.append("6. BY NAME -- does DPI work better on certain names? Each name's own weekly time-series "
                 "slope of market-relative forward return on its signal (pp per 1 SD, Newey-West t)")
    if len(dates) < 200:          # per-name slopes need ~3 years; halves ~80 weeks each
        lines.append(f"   (only {len(dates)} weekly dates -- too few for per-name tests)\n")
        return
    mid = dates[len(dates) // 2]
    H1, H2 = np.asarray(dates < mid), np.asarray(dates >= mid)
    rng = np.random.default_rng(seed)
    tables = {}
    for skey in ("d_5d", "dpi_z"):
        X = sig[skey].reindex(dates)
        for h in HORIZONS:
            lags = max(1, h // 5)
            Y = market_relative_fwd(adj, sig[skey], h, dates)
            full = F.per_name_slopes(X, Y, lags)
            null = F.placebo_slope_t(X, Y, lags, k=40, seed=seed)
            t = full["t"].to_numpy()
            h1 = F.per_name_slopes(X[H1], Y[H1], lags, min_obs=80)
            h2 = F.per_name_slopes(X[H2], Y[H2], lags, min_obs=80)
            both = h1.join(h2, lsuffix="_1", rsuffix="_2", how="inner")
            rho = both["b_1"].corr(both["b_2"], method="spearman")
            top = both[both["t_1"].rank(pct=True) > 0.8]
            lines.append(f"   {skey} h{h}: {len(full)} names | SD of t: real {t.std():.2f} vs placebo {null.std():.2f} | "
                         f"|t|>2 real {np.mean(np.abs(t) > 2):.1%} vs placebo {np.mean(np.abs(null) > 2):.1%} | "
                         f"|t|>3 {int((np.abs(t) > 3).sum())} names vs {np.mean(np.abs(null) > 3) * len(t):.0f} by chance")
            lines.append(f"      persistence: Spearman(slope 1st half, slope 2nd half) {rho:+.3f} (n={len(both)}); "
                         f"top-quintile-by-1st-half-t names: t {top['t_1'].mean():+.2f} -> {top['t_2'].mean():+.2f}")
            cells = []
            for dm in (False, True):
                pnl, nsel = F.selection_pnl(X, Y, H1, H2, lags, demean=dm)
                m, tt, _ = F.newey_west_t(pnl, lags)
                nt = [F.newey_west_t(F.selection_pnl(F.shift_names(X, rng), Y, H1, H2, lags, demean=dm)[0],
                                     lags)[1] for _ in range(k_placebo)]
                beat = float(np.nanmean(np.array(nt) < tt)) if np.isfinite(tt) else np.nan
                cells.append(f"{'timing-only' if dm else 'as traded'} {fmt(m, tt).strip()} beats {beat:.0%} of placebo")
            pooled, _ = F.selection_pnl(X, Y, H1, H2, lags, orient="pooled")
            pm, pt, _ = F.newey_west_t(pooled, lags)
            lines.append(f"      trade 1st-half |t|>1.5 names ({nsel}) in their own direction in the 2nd half: "
                         + "; ".join(cells) + f"; same names, one common direction {fmt(pm, pt).strip()}")
            tables[(skey, h)] = (full, both, null)
    # characteristic interactions: signal x characteristic in the full cross-section
    sh, tv, vol = vols
    abn = dpi1 - dpi1.rolling(60, min_periods=20).mean()
    chars = {"size ($ADV)": ctl["log_dollar_adv"], "off-exchange share": F.dark_share(tv, vol, window=126),
             "structural DPI level": sig["dpi_level"], "volatility": ctl["rv_21d"],
             "DPI persistence": abn.rolling(252, min_periods=126).corr(abn.shift(1)),
             "price echo": abn.rolling(252, min_periods=126).corr(ret_1d)}
    C = {k: F.xs_rank(v) for k, v in ctl.items()}
    lines.append("   characteristic x signal interaction t (full cross-section, +ctrl):")
    for skey in ("d_5d", "dpi_z"):
        for h in HORIZONS:
            Yf = F.forward_log_return(adj, h, lag=1)
            S_ = F.xs_rank(sig[skey])
            cells = []
            for cname, ch in chars.items():
                Cr = F.xs_rank(ch.where(sig[skey].notna()))
                res, _ = F.fama_macbeth(Yf, {"s": S_, "c": Cr, "sxc": S_ * Cr, **C}, dates,
                                        min_names=min_names, nw_lags=max(1, h // 5))
                cells.append(f"{cname} {res.loc['sxc', 't']:+.1f}")
            lines.append(f"      {skey} h{h}: " + " | ".join(cells))
    lvl = sig["dpi_level"].where(sig["dpi_z"].notna()).rank(axis=1, pct=True)
    for h in HORIZONS:
        Yf = F.forward_log_return(adj, h, lag=1)
        cells = []
        for lab, (lo, hi) in (("structurally lit tercile", (0, 1 / 3)), ("structurally dark tercile", (2 / 3, 1.01))):
            m = (lvl > lo) & (lvl <= hi)
            Xs = {"s": F.xs_rank(sig["dpi_z"].where(m)), **{k: F.xs_rank(ctl[k].where(m)) for k in ctl}}
            parts = []
            for dd in (dates, dates[dates < mid], dates[dates >= mid]):
                res, _ = F.fama_macbeth(Yf, Xs, dd, min_names=max(20, min_names // 3), nw_lags=max(1, h // 5))
                parts.append(fmt(res.loc["s", "coef"], res.loc["s", "t"]).strip())
            cells.append(f"{lab}: {parts[0]} [halves {parts[1]} / {parts[2]}]")
        lines.append(f"   dpi_z h{h} within " + " | ".join(cells))
    groups = {}
    for t in names:
        if (sector_map or {}).get(t):
            groups.setdefault(sector_map[t], []).append(t)
    groups = {k: v for k, v in sorted(groups.items()) if len(v) >= 20}
    if groups:
        h = max(HORIZONS)
        Yf = F.forward_log_return(adj, h, lag=1)
        for skey in ("d_5d", "dpi_z"):
            cells, n2 = [], 0
            for secname, cols in groups.items():
                m = pd.DataFrame(False, index=sig[skey].index, columns=sig[skey].columns)
                m[[c for c in cols if c in m.columns]] = True
                Xs = {"s": F.xs_rank(sig[skey].where(m)), **{k: F.xs_rank(ctl[k].where(m)) for k in ctl}}
                res, _ = F.fama_macbeth(Yf, Xs, dates, min_names=12, nw_lags=max(1, h // 5))
                c, t = res.loc["s", "coef"], res.loc["s", "t"]
                n2 += int(abs(t) >= 2) if np.isfinite(t) else 0
                cells.append(f"{secname} ({len(cols)}) {fmt(c, t).strip()}")
            lines.append(f"   {skey} h{h} by sector (+ctrl), {n2} of {len(groups)} at |t|>=2: " + " | ".join(cells))
    ndx = [t for t in N.NDX100 if t in names]
    if len(ndx) >= 50:
        full, both, null = tables[("dpi_z", max(HORIZONS))]
        d = full.join(both[["t_1", "t_2"]], how="left").reindex([t for t in ndx if t in full.index])
        p_up, p_dn = np.mean(null > 1), np.mean(null < -1)
        agree = d.dropna(subset=["t_1", "t_2"])
        same = int(((agree["t_1"] > 1) & (agree["t_2"] > 1)).sum() + ((agree["t_1"] < -1) & (agree["t_2"] < -1)).sum())
        d = d.sort_values("t")
        pick = pd.concat([d.head(4), d.tail(4)])
        lines.append(f"   NDX-100, dpi_z h{max(HORIZONS)}: {len(d)} names, |t|>2: {int((d['t'].abs() > 2).sum())} "
                     f"(chance {np.mean(np.abs(null) > 2) * len(d):.1f}); |t|>1 the same way in both halves: {same} "
                     f"(chance {len(agree) * (p_up ** 2 + p_dn ** 2):.1f})")
        tf = lambda v: f"{v:+.1f}" if np.isfinite(v) else "--"  # noqa: E731
        lines.append("      strongest either way (t full / 1st half / 2nd half): " + ", ".join(
            f"{i} {tf(r.t)}/{tf(r.t_1)}/{tf(r.t_2)}" for i, r in pick.iterrows()))
    lines.append("")


# ----------------------------------------------------------------------------
# 5. ATS vs internalizer split (FINRA OTC transparency, weekly)
# ----------------------------------------------------------------------------
def fetch_otc_weekly(symbols, cache_dir=None, workers=4, retries=5):
    """Weekly symbol totals from FINRA's public OTC-transparency API: 'ATS_W_SMBL' (volume on
    alternative trading systems -- the dark pools) and 'OTC_W_SMBL' (non-ATS OTC volume --
    mostly wholesaler internalization of retail orders). Long frame [sym, type, week,
    shares, published]. Cached to `OTC_CACHE`; only symbols missing from it are fetched."""
    import concurrent.futures
    import time
    cols = ["sym", "type", "week", "shares", "published"]
    path = Path(cache_dir) / OTC_CACHE if cache_dir else None
    have = pd.read_csv(path, parse_dates=["week", "published"]) if path and path.exists() \
        else pd.DataFrame(columns=cols)
    stale = have["published"].max() < pd.Timestamp.today() - pd.Timedelta(days=10) if len(have) else True
    todo = [s for s in symbols if stale or s not in set(have["sym"])]
    session = N.make_session(workers)

    def one(job):
        sym, code = job
        body = {"limit": 1000, "compareFilters": [
            {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": sym},
            {"compareType": "EQUAL", "fieldName": "summaryTypeCode", "fieldValue": code}],
            "dateRangeFilters": [{"fieldName": "weekStartDate", "startDate": OTC_START,
                                  "endDate": str(pd.Timestamp.today().date())}]}
        for k in range(retries):
            try:
                r = session.post(FINRA_OTC_URL, json=body, timeout=60,
                                 headers={"Accept": "application/json"})
                if r.status_code == 204:
                    return []
                if r.status_code == 200:
                    return [(sym, code, x["weekStartDate"], x["totalWeeklyShareQuantity"],
                             x.get("initialPublishedDate")) for x in r.json()]
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.5 * (k + 1))
        return []
    rows = []
    if todo:
        print(f"FINRA OTC transparency: fetching {len(todo)} symbol(s)...", file=sys.stderr)
        jobs = [(s, c) for s in todo for c in ("ATS_W_SMBL", "OTC_W_SMBL")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(one, jobs):
                rows += r
    new = pd.DataFrame(rows, columns=cols)
    keep = have[~have["sym"].isin(todo)]
    parts = [f for f in (keep, new) if len(f)]
    # (concat with an EMPTY object-dtype frame would silently upcast `shares` to object)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce")
    out["week"] = pd.to_datetime(out["week"])
    out["published"] = pd.to_datetime(out["published"])
    if path is not None and len(new):
        out.to_csv(path, index=False)
    return out


def section_ats(lines, A, P, names, ctl, min_names):
    lines.append("5. ATS SPLIT -- abnormal dark-POOL activity (FINRA weekly OTC transparency), entered at "
                 "the first close after FINRA publishes the week; weekly Fama-MacBeth, spread pp (NW t)")
    if A is None or A.empty:
        lines.append("   (no OTC-transparency data)\n")
        return None
    ats = A[A["type"] == "ATS_W_SMBL"].pivot_table(index="week", columns="sym", values="shares", aggfunc="sum")
    otc = A[A["type"] == "OTC_W_SMBL"].pivot_table(index="week", columns="sym", values="shares", aggfunc="sum")
    cols = [c for c in names if c in ats.columns and c in otc.columns]
    ats, otc = ats[cols], otc[cols]
    published = A.groupby("week")["published"].max()
    vol = P["volume"][cols]
    wvol = vol.resample("W-MON", label="left", closed="left").sum().reindex(ats.index)

    def abn(df, w=26):
        b = df.shift(1).rolling(w, min_periods=w // 2)
        return (df - b.mean()) / b.std()
    share = ats / (ats + otc)
    sigs = {"ats_share_z": abn(share),
            "ats_of_volume_z": abn(ats / wvol.where(wvol > 0)),
            "ats_vol_surprise": np.log(ats / ats.shift(1).rolling(26, min_periods=13).mean())}
    lines.append(f"   median ATS share of off-exchange volume {float(share.median().median()):.2f} "
                 f"(mega-caps ~0.1-0.2: retail internalization dominates), of all volume "
                 f"{float((ats / wvol).median().median()):.2f}; {len(cols)} names, weeks "
                 f"{ats.index.min().date()} -> {ats.index.max().date()}")
    tidx = P["adjclose"].index
    entry = {}
    for wk, pub in published.dropna().items():
        pos = tidx.searchsorted(pub, side="right")          # first session after publication
        if pos < len(tidx):
            entry[wk] = tidx[pos]
    ent = pd.Series(entry)
    ent = ent[~ent.duplicated(keep="last")]
    Y = {h: F.forward_log_return(P["adjclose"][cols], h, lag=0) for h in HORIZONS}
    C = {k: F.xs_rank(v[cols]).shift(1) for k, v in ctl.items()}     # known at entry - 1
    rows = []
    for key, S in sigs.items():
        Se = S.reindex(ent.index)
        Se.index = pd.DatetimeIndex(ent.values)
        dates = Se.index
        cells = []
        for h in HORIZONS:
            for ctrl in (False, True):
                X = {"s": F.xs_rank(Se), **(C if ctrl else {})}
                res, _ = F.fama_macbeth(Y[h], X, dates, min_names=min_names, nw_lags=max(1, h // 5))
                cells.append(fmt(res.loc["s", "coef"], res.loc["s", "t"]))
                rows.append({"signal": key, "horizon": h, "controls": ctrl,
                             "coef": res.loc["s", "coef"], "t": res.loc["s", "t"]})
        mid = dates[len(dates) // 2]
        halves = []
        for dd in (dates[dates < mid], dates[dates >= mid]):
            res, _ = F.fama_macbeth(Y[21], {"s": F.xs_rank(Se), **C}, dd, min_names=min_names, nw_lags=4)
            halves.append(fmt(res.loc["s", "coef"], res.loc["s", "t"]))
        lines.append(f"   {key:17s} h5 {cells[0]} +ctrl {cells[1]} | h21 {cells[2]} +ctrl {cells[3]} "
                     f"| h21 +ctrl halves {halves[0]} / {halves[1]} (split {mid.date()})")
    lines.append("")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=("spx_ndx", "ndx", "russell"), default="spx_ndx")
    ap.add_argument("--start", default="2018-06-01", help="price history start (FINRA begins 2018-08-01)")
    ap.add_argument("--test-start", default="2019-03-01",
                    help="first signal date (leaves the 126-session baseline room to fill)")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-sqz", action="store_true", help="skip SqueezeMetrics' published DIX")
    ap.add_argument("--ats", action="store_true",
                    help="add the FINRA OTC-transparency ATS-vs-internalizer section (~1,000 API calls "
                         "on first run, cached after)")
    ap.add_argument("--by-name", action="store_true",
                    help="add the per-name heterogeneity section (placebo loops: a few minutes)")
    ap.add_argument("--placebo-k", type=int, default=100,
                    help="placebo draws for the --by-name name-selection test")
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--csv-out", default="", help="per-signal results table")
    args = ap.parse_args()

    syms, proxy = universe_symbols(args.universe)
    if not syms:
        sys.exit("could not load universe holdings")
    end = pd.Timestamp.today().normalize()
    P = N.build_universe_panels(list(dict.fromkeys([proxy] + syms)), pd.Timestamp(args.start), end,
                                workers=args.workers, cache_dir=args.cache_dir or None,
                                ns=f"flowstudy_{args.universe}", refresh=args.refresh, label=args.universe,
                                heal_frac=1.0 if args.universe == "russell" else 0.5)
    idx = P["short"].index
    for k in ("close", "adjclose", "volume"):
        P[k] = P[k].reindex(idx)
    names = [s for s in syms if s in P["short"].columns and s != proxy]
    sig, ctl, dix, dpi1, ret_1d, adj, adv_dollar, close, vols = build_signals(P, names)
    min_names = 30 if args.universe == "ndx" else 100
    liq = None
    if args.universe == "russell":        # untradeable microcaps out of the cross-section
        liq = (adv_dollar >= MIN_DOLLAR_ADV) & (close >= MIN_PRICE)
        sig = {k: v.where(liq) for k, v in sig.items()}
        ctl = {k: v.where(liq) for k, v in ctl.items()}
    dates = idx[idx >= pd.Timestamp(args.test_start)][::5]

    lines = [f"RELATIVE-DARKNESS STUDY -- universe {args.universe} ({len(names)} names), "
             f"FINRA {idx.min().date()} -> {idx.max().date()}", ""]
    sqz = None if args.no_sqz else load_sqz_dix()
    naive = None
    if args.universe == "spx_ndx":        # the pre-fix pairing, for the split-basis validation
        ok = P["total_raw"][names] >= 1000
        naive = N.compute_dollar_dix(P["short_raw"][names].where(ok), P["total_raw"][names].where(ok),
                                     close)
    ivsym = IMPLIED_VOL[args.universe]
    iv = load_implied_vol(ivsym, "2010-06-01", cache_dir=args.cache_dir or None)
    vix = iv if ivsym == "^VIX" else (load_implied_vol("^VIX", "2010-06-01", cache_dir=args.cache_dir or None)
                                     if sqz is not None else None)
    idx_res = section_index(lines, dix, P["adjclose"][proxy] if proxy in P["adjclose"] else None, proxy,
                            sqz, naive_dix=naive, iv=iv, ivname=ivsym.strip("^"), vix=vix)
    section_structure(lines, dpi1, ret_1d)
    sig_res = section_signals(lines, sig, ctl, adj, dates, min_names)
    section_robust(lines, sig, ctl, vols, adj, dates, min_names, mask=liq)
    section_vol(lines, sig, ctl, adj, dates, min_names)
    if args.ats:
        A = fetch_otc_weekly(names, cache_dir=args.cache_dir or None)
        section_ats(lines, A, P, names, ctl, min_names)
    if args.by_name:
        section_by_name(lines, sig, ctl, adj, dates, names, dpi1, ret_1d, vols, min_names,
                        k_placebo=args.placebo_k, sector_map=universe_sectors(args.universe))
    text = "\n".join(lines)
    print(text)
    if args.summary_out:
        Path(args.summary_out).write_text(text + "\n", encoding="utf-8")
    if args.csv_out:
        sig_res.to_csv(args.csv_out, index=False)
        if not idx_res.empty:
            idx_res.to_csv(args.csv_out.replace(".csv", "_index.csv"), index=False)


if __name__ == "__main__":
    main()
