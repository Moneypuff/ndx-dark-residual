#!/usr/bin/env python3
"""
Build the regime log (docs/regime_log.html) -- the operational companion to
the GEX x dispersion barometer.

Turns the barometer's playbook (GEX_DISPERSION_GUIDE.md) into a page you
check each morning:

  * CONTEXT -- today's quadrant, the GEX+ percentile path (is a rotation
    window open?), the dispersion premium (DSPX vs our realized), and the
    correlation-shock state; distilled into one posture line.
  * SIGNAL LOG -- dated break signals across the backtested pair universe,
    each tagged with its context bucket (fade / rotation / de-risk) and, once
    63 sessions have passed, the realized old-theme outcome -- the page keeps
    score on itself.
  * LEADERBOARD -- every sector/theme ETF vs SPY: 63d/21d relative strength,
    break z, trend state. A screen, not a signal (sector-vs-index spreads are
    tamer than pair spreads; the thresholds run hot here, and the page says so).

The bucket statistics shown against each tag are the backtest's own numbers
(12 pairs x both directions, 2005-2026, 150 signals) and are deliberately
hardcoded with their sample sizes: the page must not overclaim as data
accrues nightly.

Reads the gamma series through the same cached fetch layer as the barometer
(GEXPLUS_KEY env -> GEX+ feed, else the public classic CSV) and reads the
quadrant/dispersion series straight out of the built barometer page, so build
docs/gex_dispersion.html first (the workflow does).

    python build_regime_log.py --docs-out docs/regime_log.html --cache-dir .ndx_dark_cache
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ndx_dark_residual as N
import build_gex_dispersion as B

# ---- detector parameters (the backtest's stable-plateau values) ------------
ON_CUT = 5.0          # |63d spread| above this = theme "on" (pp)
Z_CUT = 1.5           # 21d spread z crossing against the theme = break
GREL_DROP = 40        # GEX+ 1y-pct fall over 21 sessions = gamma release
GREL_LOOKBACK = 10    # release counts if it happened within this many sessions
SHOCK_Z = 2.0         # COR1M 21d-change z at/above this = correlation shock
SHOCK_BEFORE_D = 34   # calendar days before a break a shock still taints it
SHOCK_AFTER_D = 8     # ... and after
COOLDOWN = 63         # sessions between counted signals per pair-direction
FWD = 63              # outcome horizon (sessions)
LOG_MONTHS = 24       # how far back the signal log reaches

# The backtest's bucket statistics -- shown verbatim next to every tag.
BUCKETS = {
    "fade":     {"label": "Fade",        "n": 28, "mean": 3.7,  "neg": 43,
                 "text": "breaks near a correlation shock historically reversed"},
    "rotation": {"label": "Rotation",    "n": 36, "mean": -1.6, "neg": 64,
                 "text": "gamma-released breaks were the genuine leadership changes"},
    "derisk":   {"label": "De-risk",     "n": 86, "mean": 0.6,  "neg": 51,
                 "text": "edge gone, direction a coin flip"},
}

PAIRS = [
    ("SMH", "IGV", "Semis vs Software"),
    ("SOXX", "IGV", "Semis (SOXX) vs Software"),
    ("QQQ", "IWM", "NDX-100 vs Small caps"),
    ("VUG", "VTV", "Growth vs Value"),
    ("XLE", "XLK", "Energy vs Tech"),
    ("XLF", "XLU", "Financials vs Utilities"),
    ("XLY", "XLP", "Discretionary vs Staples"),
    ("XBI", "XLV", "Biotech vs Health care"),
    ("EEM", "SPY", "Emerging mkts vs S&P"),
    ("KRE", "XLF", "Regional banks vs Financials"),
    ("GDX", "GLD", "Gold miners vs Gold"),
    ("SMH", "QQQ", "Semis vs NDX-100"),
]
SECTORS = {"XLK": "Tech", "XLF": "Financials", "XLV": "Health care",
           "XLY": "Discretionary", "XLP": "Staples", "XLE": "Energy",
           "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
           "XLRE": "Real estate", "XLC": "Comm services"}
THEMES = {"SMH": "Semis", "IGV": "Software", "XBI": "Biotech",
          "KRE": "Regional banks", "GDX": "Gold miners", "IWM": "Small caps",
          "VTV": "Value", "VUG": "Growth", "EEM": "Emerging mkts",
          "QQQ": "NDX-100"}


# ----------------------------------------------------------------------------
# Pure computation (unit tested)
# ----------------------------------------------------------------------------
def spread_frames(pa, pb):
    """(s21, s63, z, fwd) for the a-over-b spread of two aligned price Series:
    21d/63d return spreads in pp, the 21d spread's z vs its trailing year, and
    the (look-ahead) forward-63-session spread used only for scoring the past."""
    ix = pa.dropna().index.intersection(pb.dropna().index)
    a, b = pa[ix], pb[ix]
    s21 = (a / a.shift(21) - b / b.shift(21)) * 100
    s63 = (a / a.shift(63) - b / b.shift(63)) * 100
    z = (s21 - s21.rolling(252).mean()) / s21.rolling(252).std()
    fwd = (a.shift(-FWD) / a - b.shift(-FWD) / b) * 100
    return s21, s63, z, fwd


def detect_breaks(s63, z, on_cut=ON_CUT, z_cut=Z_CUT, cooldown=COOLDOWN):
    """Break signals in BOTH directions: list of (date, sign) where sign=+1
    means the a-over-b theme broke (was on, z crossed below -z_cut) and
    sign=-1 means the b-over-a theme broke. Cooldown per direction."""
    out = []
    for sgn in (1, -1):
        on = (s63 * sgn) > on_cut
        zz = z * sgn
        trig = on.shift(1).fillna(False) & (zz < -z_cut) & (zz.shift(1) >= -z_cut)
        last = None
        for d in s63.index[trig]:
            if last is not None and (s63.index.get_loc(d) - s63.index.get_loc(last)) < cooldown:
                continue
            out.append((d, sgn))
            last = d
    return sorted(out)


def shock_dates(cor1m, z_cut=SHOCK_Z):
    """Days the COR1M 21d change ran z >= z_cut vs its trailing year."""
    chg = cor1m.diff(21)
    z = (chg - chg.rolling(252).mean()) / chg.rolling(252).std()
    return list(z.index[z >= z_cut])


def near_shock(d, shocks, before_d=SHOCK_BEFORE_D, after_d=SHOCK_AFTER_D):
    lo, hi = d - pd.Timedelta(days=before_d), d + pd.Timedelta(days=after_d)
    return any(lo <= s <= hi for s in shocks)


def gamma_release_flag(gexp, drop=GREL_DROP, lookback=GREL_LOOKBACK):
    """Boolean Series: a >=`drop`-point fall of the gamma percentile over 21
    sessions occurred within the last `lookback` sessions."""
    rel = (gexp.diff(21) <= -drop)
    return rel.rolling(lookback, min_periods=1).max().astype(bool)


def bucket_for(is_shock, is_grel):
    return "fade" if is_shock else ("rotation" if is_grel else "derisk")


def value_area_edges(close, volume, window=252, pct=0.70, nbins=40):
    """Rolling volume-profile value area: for each day, bin the trailing
    `window` closes weighted by volume and keep the tightest set of bins
    holding `pct` of it. Returns (va_lo, va_hi) Series. Close-anchored (daily
    closes, not intraday ranges) -- coarse but honest at daily granularity."""
    c = close.to_numpy(dtype=float)
    v = np.nan_to_num(volume.reindex(close.index).to_numpy(dtype=float))
    lo_s = np.full(len(c), np.nan)
    hi_s = np.full(len(c), np.nan)
    for i in range(window - 1, len(c)):
        cs, vs = c[i - window + 1:i + 1], v[i - window + 1:i + 1]
        m = np.isfinite(cs)
        cs, vs = cs[m], vs[m]
        if len(cs) < window // 2 or vs.sum() <= 0:
            continue
        lo, hi = cs.min(), cs.max()
        if hi <= lo:
            continue
        idx = np.clip(((cs - lo) / (hi - lo) * nbins).astype(int), 0, nbins - 1)
        prof = np.bincount(idx, weights=vs, minlength=nbins)
        order = np.argsort(prof)[::-1]
        k = int(np.searchsorted(prof[order].cumsum(), pct * prof.sum())) + 1
        sel = order[:k]
        lo_s[i] = lo + sel.min() * (hi - lo) / nbins
        hi_s[i] = lo + (sel.max() + 1) * (hi - lo) / nbins
    return (pd.Series(lo_s, index=close.index), pd.Series(hi_s, index=close.index))


def conviction_frame(close, volume, va_window=252):
    """Per-day directional conviction score, 0-4 -- one point per confirmed
    component, all judged in the direction of the trailing 21-session move:

      mas    -- close on the move's side of >=2 of the 21/50/200d SMAs
      level  -- major level broken within 10 sessions: 200d SMA cross, or an
                exit through the trailing-year volume-profile value area edge
      band   -- a close beyond the 21d +/-2-sigma Bollinger band within 5 sessions
      vol    -- 5-day mean volume >= 1.25x the 63-day mean

    Returns a DataFrame with dir (+1/-1/0), score, and the four booleans."""
    c = close.dropna()
    v = volume.reindex(c.index)
    s21, s50, s200 = (c.rolling(w).mean() for w in (21, 50, 200))
    sd21 = c.rolling(21).std()
    up_band, dn_band = s21 + 2 * sd21, s21 - 2 * sd21
    dirn = np.sign(c - c.shift(21)).fillna(0)

    side = sum(((c > s) & (dirn > 0)) | ((c < s) & (dirn < 0))
               for s in (s21, s50, s200))
    mas = (side >= 2) & (dirn != 0)

    cross_up_200 = (c > s200) & (c.shift(1) <= s200.shift(1))
    cross_dn_200 = (c < s200) & (c.shift(1) >= s200.shift(1))
    va_lo, va_hi = value_area_edges(c, v, window=va_window)
    exit_up = (c > va_hi) & (c.shift(1) <= va_hi.shift(1))
    exit_dn = (c < va_lo) & (c.shift(1) >= va_lo.shift(1))
    brk_up = (cross_up_200 | exit_up).rolling(10, min_periods=1).max().astype(bool)
    brk_dn = (cross_dn_200 | exit_dn).rolling(10, min_periods=1).max().astype(bool)
    level = (brk_up & (dirn > 0)) | (brk_dn & (dirn < 0))

    thrust_up = (c > up_band).rolling(5, min_periods=1).max().astype(bool)
    thrust_dn = (c < dn_band).rolling(5, min_periods=1).max().astype(bool)
    band = (thrust_up & (dirn > 0)) | (thrust_dn & (dirn < 0))

    vol_ok = v.rolling(5).mean() >= 1.25 * v.rolling(63).mean()

    F = pd.DataFrame({"dir": dirn, "mas": mas, "level": level,
                      "band": band, "vol": vol_ok.fillna(False)})
    F["score"] = F[["mas", "level", "band", "vol"]].sum(axis=1).astype(int)
    return F


def conviction_events(close, conv, min_score=3, cooldown=21, horizon=63):
    """First-day high-conviction events (score >= min_score, `cooldown`
    sessions apart per direction) with the direction-signed forward return
    over `horizon` sessions -- 'did the break follow through?'."""
    c = close.dropna()
    out = []
    last = {1: None, -1: None}
    hot = (conv["score"] >= min_score) & (conv["dir"] != 0)
    for d in conv.index[hot]:
        sgn = int(conv.loc[d, "dir"])
        i = c.index.get_loc(d)
        if last[sgn] is not None and i - last[sgn] < cooldown:
            continue
        last[sgn] = i
        fwd = (float((c.iloc[i + horizon] / c.iloc[i] - 1) * 100 * sgn)
               if i + horizon < len(c) else np.nan)
        out.append({"date": d, "dir": sgn, "score": int(conv.loc[d, "score"]),
                    "fwd": fwd})
    return pd.DataFrame(out)


def lb_state(r63, z21, on_cut=ON_CUT):
    if r63 > on_cut:
        return "HOT" if z21 > -1.0 else "HOT, breaking"
    if r63 < -on_cut:
        return "UNFAVORED" if z21 < 1.0 else "UNFAVORED, turning"
    return "NEUTRAL"


def posture(shock_recent, grel_now, rotation_recent):
    if shock_recent:
        return ("shock", "Systemic wobble in progress — fade theme breaks; "
                "historically they snapped back.")
    if grel_now:
        return ("open", "Rotation window OPEN — gamma released; tighten stops on "
                "crowded themes. Breaks fired now carry the strongest historical lean.")
    if rotation_recent:
        return ("settling", "Rotation recently completed — new leadership settling. "
                "Fresh breaks without the gamma tag get de-risk weight, not conviction.")
    return ("stable", "Stable regime — let the quadrant set the risk budget.")


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
def load_barometer(path):
    """(SER, META) payloads out of the built barometer page."""
    html = Path(path).read_text(encoding="utf-8")
    ser = re.search(r"const SER = (\{.*?\});\n", html)
    meta = re.search(r"const META = (\{.*?\});\n", html)
    if not ser or not meta:
        raise SystemExit(f"No barometer payload in {path} -- build docs/gex_dispersion.html first.")
    return json.loads(ser.group(1)), json.loads(meta.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--barometer", default="docs/gex_dispersion.html",
                    help="built barometer page carrying the dials payload")
    ap.add_argument("--template", default="regime_log_template.html")
    ap.add_argument("--docs-out", default="docs/regime_log.html")
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download of the gamma/Cboe docs AND the Yahoo "
                         "price panel")
    ap.add_argument("--refresh-text", action="store_true",
                    help="force re-download of just the gamma/Cboe text feeds "
                         "(bypassing the post-close cache-freshness check), without "
                         "forcing the Yahoo price panel -- see build_gex_dispersion.py")
    args = ap.parse_args()
    text_refresh = args.refresh or args.refresh_text

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = (lambda name: cache_dir / name if cache_dir else None)
    session = N.make_session(4) if N.requests else None

    # ---- dials: gamma percentile + COR1M shocks + barometer payload ---------
    key = os.environ.get("GEXPLUS_KEY", "").strip()
    G = pd.DataFrame()
    if key:
        G = B.parse_gexplus_csv(B.fetch_text_cached(
            B.GEXPLUS_URL_TMPL.format(key=key), cpath("gexdisp_gexplus.csv"),
            refresh=text_refresh, label="SqueezeMetrics GEX+", session=session) or "")
    if G.empty:
        G = B.parse_squeeze_csv(B.fetch_text_cached(
            B.SQUEEZE_CSV_URL, cpath("gexdisp_squeeze.csv"),
            refresh=text_refresh, label="SqueezeMetrics GEX", session=session) or "")
    if G.empty:
        raise SystemExit("No gamma series available.")
    gexp = B.rolling_pct(G["gex"])
    grel = gamma_release_flag(gexp)

    cor1m = B.parse_cboe_csv(B.fetch_text_cached(
        B.CBOE_HISTORY_TMPL.format(sym="COR1M"), cpath("gexdisp_cor1m.csv"),
        refresh=text_refresh, label="Cboe COR1M", session=session) or "")
    shocks = shock_dates(cor1m)

    SERB, METB = load_barometer(args.barometer)

    # ---- prices -------------------------------------------------------------
    syms = sorted({s for a, b, _ in PAIRS for s in (a, b)}
                  | set(SECTORS) | set(THEMES) | {"SPY", "GLD"})
    panels = N.load_yahoo_panels(syms, "2004-01-01", pd.Timestamp.today().normalize(),
                                 cache_dir=args.cache_dir or None,
                                 refresh=args.refresh, label="REGLOG")
    adj = panels["adjclose"].dropna(how="all")
    volp = panels["volume"]

    # ---- signal log ---------------------------------------------------------
    log_start = pd.Timestamp.today() - pd.DateOffset(months=LOG_MONTHS)
    sig_rows, rotation_recent = [], False
    for a, b, label in PAIRS:
        if a not in adj or b not in adj:
            continue
        s21, s63, z, fwd = spread_frames(adj[a], adj[b])
        g = grel.reindex(s63.index).fillna(False)
        for d, sgn in detect_breaks(s63, z):
            if d < log_start:
                continue
            is_shock = near_shock(d, shocks)
            bucket = bucket_for(is_shock, bool(g.loc[d]))
            f = fwd.loc[d] * sgn
            broke = f"{a} > {b}" if sgn > 0 else f"{b} > {a}"
            if bucket == "rotation" and (pd.Timestamp.today() - d).days <= 63:
                rotation_recent = True
            sig_rows.append({
                "date": d.strftime("%Y-%m-%d"), "pair": label, "broke": broke,
                "z": round(float(z.loc[d] * sgn), 2), "bucket": bucket,
                "fwd63": None if pd.isna(f) else round(float(f), 1),
            })
    sig_rows.sort(key=lambda r: r["date"], reverse=True)

    # ---- leaderboard + conviction ------------------------------------------
    spy = adj["SPY"]
    lb_rows, all_ev = [], []
    for t, name in {**SECTORS, **THEMES}.items():
        if t not in adj:
            continue
        s21, s63, z, _ = spread_frames(adj[t], spy)
        if not len(s63.dropna()):
            continue
        r63, r21, zz = float(s63.iloc[-1]), float(s21.iloc[-1]), float(z.iloc[-1])
        recent = ""
        for d, sgn in reversed(detect_breaks(s63, z)):
            if (pd.Timestamp.today() - d).days <= 63:
                recent = ("broke" if sgn > 0 else "turned") + " " + d.strftime("%m-%d")
            break
        # conviction: absolute price action of the ETF itself, plus its own
        # historical follow-through in the current direction
        c = adj[t].dropna()
        conv = conviction_frame(c, volp[t])
        ev = conviction_events(c, conv)
        ev["etf"] = t
        all_ev.append(ev)
        cur = conv.iloc[-1]
        score, cdir = int(cur["score"]), int(cur["dir"])
        comps = "·".join(lbl for k, lbl in (("mas", "MA"), ("level", "LVL"),
                                            ("band", "BB"), ("vol", "VOL")) if cur[k])
        ft = None
        if cdir:
            evd = ev[(ev["dir"] == cdir)].dropna(subset=["fwd"])
            if len(evd) >= 10:
                ft = [round(float(evd["fwd"].median()), 1),
                      int(round(100 * float((evd["fwd"] > 0).mean()))), int(len(evd))]
        lb_rows.append([name, t, round(r63, 1), round(r21, 1), round(zz, 2),
                        lb_state(r63, zz), recent, score, cdir, comps, ft])
    lb_rows.sort(key=lambda r: -r[2])
    AE = pd.concat(all_ev, ignore_index=True).dropna(subset=["fwd"]) if all_ev else pd.DataFrame()

    def _pool(d):
        s = AE[AE["dir"] == d]
        return ([round(float(s["fwd"].median()), 1),
                 int(round(100 * float((s["fwd"] > 0).mean()))), int(len(s))]
                if len(s) else None)

    # ---- context / posture --------------------------------------------------
    last_shock = max([s for s in shocks], default=None)
    shock_recent = bool(last_shock and (pd.Timestamp.today() - last_shock).days <= 15)
    grel_now = bool(grel.iloc[-1])
    pk, ptext = posture(shock_recent, grel_now, rotation_recent)

    dspx = [v for v in SERB["dspx"] if v is not None]
    dspx_chg21 = (round(dspx[-1] - dspx[-22], 1) if len(dspx) > 22 else None)
    t = METB["today"]
    ctx = {
        "date": t["date"],
        "quad": t["quad"], "quadLabel": t["quadLabel"],
        "gexLabel": METB.get("gexLabel", "GEX"),
        "gexp": round(float(gexp.iloc[-1]), 1),
        "gexp_chg21": round(float(gexp.iloc[-1] - gexp.iloc[-22]), 1) if len(gexp) > 22 else None,
        "grel": grel_now,
        "disp": t["disp"], "dspx": t["dspx"], "dspx_chg21": dspx_chg21,
        "cor1m": t["cor1m"],
        "lastShock": last_shock.strftime("%Y-%m-%d") if last_shock else None,
        "shockRecent": shock_recent,
        "posture": pk, "postureText": ptext,
        "built": "built " + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "logMonths": LOG_MONTHS,
        "conv": {"up": _pool(1), "down": _pool(-1)},
    }

    body = (Path(args.template).read_text(encoding="utf-8")
            .replace("/*__CTX__*/", json.dumps(ctx, separators=(",", ":")))
            .replace("/*__SIG__*/", json.dumps(sig_rows, separators=(",", ":")))
            .replace("/*__LB__*/", json.dumps(lb_rows, separators=(",", ":")))
            .replace("/*__BK__*/", json.dumps(BUCKETS, separators=(",", ":"))))
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>Regime Log</title></head>'
           '<body>\n' + body + '\n</body></html>\n')
    out = Path(args.docs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc)//1024} KB, {len(sig_rows)} signals in log, "
          f"{len(lb_rows)} leaderboard rows, posture: {pk})")


if __name__ == "__main__":
    main()
