# Dark-residual methodology review — "relative darkness", measured properly

**The question.** Study after study found no name-level edge in the dark residual,
even though the DIX seems to work at the index level. Is the residual neutralizing a
real signal, and would a better way to quantify the *relative darkness* of a name's
flow recover it?

**The answer, in order of importance:**

1. **There is less to inherit than it looked — and part of the index-level edge was a
   data bug.** The dollar-DIX multiplied Yahoo's *split-adjusted* prices by FINRA's
   *as-traded* share counts, so every name was under-weighted by its split factor
   before each split (AMZN got 1.4% of the 2019 NDX gauge instead of 14.3%). Fixed and
   validated against SqueezeMetrics' published DIX. On the corrected gauges, the repo's
   one "tradeable-looking" index result (large-cap DIX firm while small-cap DIX is Low
   → strong NDX months, 86–100% hit) falls to baseline. On SqueezeMetrics' own 15-year
   DIX, the level's 1-month edge is t = 1.8 raw and ≈ 0 once detrended or once realized
   volatility is controlled for: the DIX mostly works as a stress gauge.
2. **The residual isolates the wrong component.** A name's DPI is 13% structural level,
   7% market-wide, 81% idiosyncratic — and the two moving parts relate to price with
   **opposite signs**. The market-wide part rises on down days (−0.25 with the market's
   same-day return: dip-buying absorbed off-exchange). The idiosyncratic part rises
   *with the name's own* up-moves (+0.11 same day, +0.12 with the previous day) — an echo
   of buy pressure already in the price, which then slightly reverses. Subtracting the
   DIX throws away the contrarian part and keeps the echo.
3. **Measured every sensible way, relative darkness has no directional edge in large
   caps.** Nine constructions (the four the dashboard uses and five new ones), 518
   S&P 500 ∪ NDX-100 names, 381 weekly cross-sections, entry after FINRA publishes,
   reversal / momentum / size / volatility / volume controls, both halves, windows of
   1–20 sessions, tail-only events and the conditional "accumulation into weakness"
   sort: no flow measure reaches t = 2 in the bullish direction, and the few |t| ≈ 2
   cells are *negative* (next-day price-pressure reversal). The NDX-100 alone is flat,
   and so is the literal darkness measure — the *share* of off-exchange volume executed
   in ATS dark pools.
4. **What survives is marginal and mostly regime-dependent:** in *small caps*,
   abnormal dark buying leaned bullish in 2019–22 (t ≈ 2, gone since 2023), and "dark
   accumulation into weakness" held its sign in both halves (+0.36pp/month, t = 2.3);
   a bullish ATS-volume surprise appears only in 2024–26, and a *bearish*
   structural-darkness effect only in 2023–26. All sit within what several hundred
   correlated tests produce by chance — a watch list, not a strategy.
5. **It does not work better on particular large-cap or NDX names.** Per-name results
   match a placebo almost exactly — about one name in twelve looks significant at one
   month from noise alone — and 2019–22's "working" names stop working in 2023–26.
   Small caps are the exception: names picked on 2019–22 and traded in their own
   direction beat every placebo run in 2023–26 (t = 3.0), and abnormal dark buying
   leans bullish in *structurally dark* names in both universes (§5f).
6. **Dark flow does carry volatility information** — abnormal off-exchange share
   predicts lower future realized vol (t ≈ −8) — but the effect is ~4% of vol,
   too small to trade on its own.

Reproduce (FINRA + Yahoo, no key; ~5 min per universe with a warm cache):
```
python dark_flow_study.py --ats --by-name --summary-out dark_flow_summary.txt --csv-out dark_flow_signals.csv
python dark_flow_study.py --universe russell --no-sqz --by-name   # IWM holdings, liquid names
python dark_flow_study.py --universe ndx --no-sqz         # NDX-100 only
```
`dark_flow.py` holds the measures and statistics (tested in `tests/test_dark_flow.py`);
`dark_flow_study.py` is the study. Numbers below are from the 2026-09-25 runs, whose
full output is committed as `dark_flow_summary.txt` (all three universes) and
`dark_flow_signals.csv` (the S&P ∪ NDX signal table).

---

## 1. Two data bugs, now fixed

### 1a. The dollar-DIX weighted pre-split days by a fraction of their dollars

Yahoo's `close` and `volume` are split-adjusted across the whole history — Yahoo
re-bases them on every split. FINRA reports the shares that actually printed that day.
`compute_dollar_dix(short, total, close)` paired the two directly (the code comment
said it paired a *raw* close with as-traded volume; Yahoo's close is not raw), so each
pre-split day's dollar volume came out divided by the split factor:

| 2019 share of NDX dark dollars | repo (mis-weighted) | true |
|---|---:|---:|
| AMZN (20:1 in 2022) | 1.4% | **14.3%** |
| AAPL (4:1 in 2020) | 5.8% | **11.5%** |
| TSLA (5:1 in 2020, 3:1 in 2022) | 0.8% | **5.7%** |
| NVDA (4:1 in 2021, 10:1 in 2024) | 0.2% | **4.7%** |
| GOOGL (20:1 in 2022) | 0.3% | **2.6%** |
| MSFT (no split) | 9.0% | 4.5% |

87 of the 518 S&P/NDX stocks split during the FINRA window (Aug 2018 →). The old
volume-ratio split *detector* (used only for the aggregate dark ratio) flagged 50 —
46 real, plus four false positives (SCHW in March 2023 was the SVB-crisis volume
shift, not a split; also IR, DOC, NWS). It missed 41, 39 of them because its
threshold skips every split of about 2:1 or less by design, and it read TSLA's 5:1 as
6:1.

**Fix:** the price request now carries `events=split` (same call, no extra cost), the
Yahoo cache stores each symbol's split history, and `build_universe_panels` returns
FINRA volumes on Yahoo's split-adjusted share basis (`finra_to_price_basis`; as-traded
originals kept as `short_raw` / `total_raw`). DPI is a ratio and is unchanged.
**Validation** against SqueezeMetrics' published S&P DIX over the same universe:

| reconstructed DIX (518 S&P ∪ NDX names) vs SqueezeMetrics | level MAE | corr of daily changes |
|---|---:|---:|
| old pairing (as-traded shares × split-adjusted price) | 0.0063 | 0.945 |
| **fixed (shares on Yahoo's split basis)** | **0.0038** | **0.975** |

On the 502 S&P 500 members alone (SqueezeMetrics' own universe) the same fix takes
MAE 0.0062 → 0.0033 and change-correlation 0.947 → 0.978, and over Aug 2022 – Jul
2024 — when AMZN, GOOGL, NVDA and AVGO split — level correlation 0.964 → **0.996**
(MAE 0.0048 → 0.0018).

The error averaged 0.45 SD of the NDX gauge before mid-2022 and reached a full SD on
5% of days (SPX: 0.31 SD, p95 0.66 SD). It flows into the NDX/SPX/IWM/sector gauges,
the top-contributor shares and the grid residual's benchmark.

**It changes the documented index-level findings.** Re-running the comovement study's
own `build_aligned` / `entry_report` / `two_factor_report` on both versions of the three
gauges (2020-01 → 2026-09; the mis-weighted version reproduces
`INDEX_COMOVEMENT_FINDINGS.md` — e.g. 7 entries, NDX +6.80%, 100% hit):

| comovement result | mis-weighted gauges | **corrected gauges** |
|---|---|---|
| entries, N=High S=Mid I=Low | 7 — NDX +6.80%, hit 100% | 10 — **NDX +0.80%, hit 60%** |
| entries, N=Mid S=Mid I=Low | 21 — NDX +4.98%, hit 90% | 16 — **NDX +1.95%** (baseline +1.76%) |
| entries, N=Low S=Low I=High (requested) | 13 — NDX +0.84% | 9 — NDX +2.00% |
| two-factor SPREAD → NDX 1m return | +0.63 (t = 1.76) | **−0.12 (t = −0.23)** |
| two-factor LEVEL → NDX 1m return | −0.03 (t = −0.06) | +0.48 (t = 0.77) |

The "large-cap DIX firm while small-cap DIX is Low" family — the one the comovement
doc called tradeable-looking — does not survive correct weighting.

### 1b. The Yahoo price cache kept pre-split history on a stale basis

An incremental fetch only returned bars from the last cached date forward and
`combine_first`-ed them onto the cache — but a split re-bases the *whole* history, so
older cached bars stayed on the pre-split basis. Since the CI cache began (July 2026)
that hit **MNST** (2:1, Aug 11 — an NDX name) and **APH** (2:1, Sep 3): each shows a
fake −50% day inside every forward return that spans it. Dividends did the same to
`adjclose` every quarter, by the dividend yield.

**Fix:** incremental fetches re-request `YAHOO_OVERLAP_BARS` (5) cached sessions; if
they no longer match within `YAHOO_REBASE_TOL` (1e-4 — Yahoo's adjclose wobbles ~2e-7
between identical requests), or a split event appears that the cache didn't know, the
symbol is re-fetched in full and its column *replaced*, reaching back to the earliest
bar any caller cached. A cache written before split events were recorded gets one full
re-fetch per symbol (the first nightly run after this change does that automatically).

---

## 2. Is there an index-level edge to inherit?

Time-series regressions of the index's forward log return on the 5-session gauge
(pp per 1 SD, Newey-West t with lags = horizon). "Detrended" subtracts the trailing
252-session mean (the DIX drifts up over the years — 0.38 in 2011, 0.47 now — which a
raw-level test can mistake for predictive power). "+ctrl" adds 21-session realized vol
and the past 21-session return.

| gauge | h | level | level +ctrl | detrended | detrended +ctrl | realized vol t (+ctrl) |
|---|---:|---:|---:|---:|---:|---:|
| SqueezeMetrics DIX → SPX, 2011–2026 | 21 | +0.42 (1.8) | +0.23 (0.9) | +0.17 (0.8) | −0.12 (−0.6) | 2.3 |
| | 63 | +0.65 (1.2) | +0.08 (0.1) | −0.37 (−0.6) | **−1.12 (−2.1)** | 3.8 |
| SqueezeMetrics DIX → SPX, 2018-08 → | 21 | +0.43 (1.0) | +0.35 (0.8) | −0.02 (−0.1) | −0.34 (−1.1) | 3.8 |
| reconstructed S&P DIX (fixed) → SPY | 21 | +0.33 (0.9) | +0.22 (0.6) | −0.16 (−0.5) | −0.53 (−1.7) | 3.8 |
| | 63 | +0.19 (0.2) | −0.26 (−0.3) | −1.39 (−1.5) | **−2.36 (−3.0)** | 4.4 |

Over 15 years the raw level earns +0.4pp/month per SD at t = 1.8. Nothing survives
detrending or a volatility control, and the only significant coefficients are
*negative* (3-month, detrended, vol-controlled). The DIX rises in selloffs (corr with
realized vol +0.28, 2011–2026), and in a mostly-bull sample buying stress pays; realized
vol captures that better than the DIX does. That is not an effect a name-level residual
could be expected to carry.

---

## 3. Anatomy of name-level DPI — why the residual can't inherit it

518 S&P/NDX names, 1,004,585 name-days, daily DPI = FINRA short / total off-exchange:

| component | share of variance | SD |
|---|---:|---:|
| structural name level (persistent differences between names) | 13.4% | 0.053 |
| common, market-wide (what the DIX measures) | 6.7% | 0.037 |
| idiosyncratic day-to-day | **80.6%** | 0.129 |

The idiosyncratic part is mostly *flow-composition noise*, not sampling noise: AAPL,
with 10–30M off-exchange shares a day, still swings its DPI ±7 points day to day
(0.63 on Sep 9 2026, 0.49 on Sep 11, 0.40 on Sep 15, 0.59 on Sep 17) — binomial
sampling noise at those volumes would be ~0.0002 — and the idiosyncratic SD barely falls from the
smallest to the largest dark-volume decile (0.137 → 0.111). Its informative piece is
short-lived (AR(1) fit: 66% persistent, ρ = 0.81, half-life **3.3 sessions**) on top
of a slow multi-month drift (lag-60 autocorrelation 0.12).

The two moving parts point in **opposite directions** relative to price:

| rank correlation with return at t+k | k=−5 | −2 | −1 | **0** | +1 | +2 | +5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| a name's abnormal DPI vs *its own* return | +0.02 | +0.06 | +0.12 | **+0.11** | −0.02 | −0.00 | +0.00 |

vs. the common component against the equal-weight market's same-day return: **−0.25**
(−0.48 in the Russell 2000). Market-wide dark buying is contrarian — it absorbs
selloffs. Name-specific dark buying is an echo of the name's own buy pressure, already
in the price, then partly reversed the next day.

What that does to each part of the current method:

1. **`D − DIX` and the rolling-OLS residual** subtract exactly the contrarian common
   part and keep the echo.
2. **The benchmark** is dollar-weighted, so it is a handful of mega-caps (the top 10
   are 58% of NDX dark dollars over the last year) that includes the name itself
   (NVDA's residual is taken against a gauge that is 11% NVDA) — and until this fix
   it was mis-weighted by splits (§1a).
3. **The rolling β** is estimated on 5-day-smoothed series over 126 sessions, and the
   residual is mean-zero over its own window by construction.
4. **D is an equal-weight mean of daily ratios**, so a thin day counts as much as a
   heavy one.
5. **The dashboard's name-level tests don't use the residual at all.** The scatter,
   decile bars, cross-sectional L/S and D-streak tabs use the raw 1-day DPI (or DPI
   minus its expanding mean), with forward returns measured from the close of the
   signal day — before FINRA has published that day's file.

---

## 4. The tweak: relative darkness (`dark_flow.py`)

| measure | definition | what it isolates |
|---|---|---|
| `vw_dpi` | Σshort / Σtotal over k sessions | DPI with heavy days weighted as heavy |
| `abnormal_dpi` → `dpi_z` | vw_dpi − the name's own vw DPI over the prior 126 sessions (baseline never overlaps the window), ÷ its own trailing σ (through t−1) | darkness relative to the name's own norm, in its own units; no look-ahead |
| `abnormal_dark_share` | log(k-session off-exchange share of consolidated volume ÷ own 126-session share) | literal darkness: how much more of the trading went off-exchange than usual |
| `dark_imbalance` | (Σshort − baseline DPI × Σtotal) ÷ (k × ADV) | abnormal dark *buying* in ADV units — the DPI deviation weighted by how much volume actually went dark |
| `price_echo_split` → `dpi_hidden` | per-date cross-sectional residual of the signal on the name's own 1-day and k-day returns | dark buying the price has **not** reflected — the quantity an informed-accumulation story is actually about |

Cross-sectional use ranks each measure per date instead of regressing on the DIX:
ranking removes the common component exactly, with no β to estimate and no
mega-cap benchmark. Everything is evaluated with a one-session implementation lag.

---

## 5. Does relative darkness predict returns?

Weekly Fama-MacBeth cross-sections; forward log return from the close of t+1. Cell =
return spread (pp) from the lowest- to the highest-ranked name, Newey-West t in
parentheses. "+ctrl" adds 5- and 21-session reversal, 12-1 momentum, 21-session vol,
$ADV and volume surprise. Halves split at 2022-12-06.

### 5a. S&P 500 ∪ NDX-100 — 518 names, 381 weeks (2019-03 → 2026-09)

| signal | h5 | h5 +ctrl | h21 | h21 +ctrl | h21 +ctrl 2019–22 | h21 +ctrl 2023–26 | Q5−Q1 h5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| *current:* raw 1-day DPI | −0.11 (−2.2) | −0.14 (−3.1) | −0.27 (−2.0) | −0.37 (−2.9) | −0.13 (−0.8) | −0.59 (−3.2) | −0.08 (−1.7) |
| *current:* D (5-day mean) | −0.08 (−1.5) | −0.11 (−2.3) | −0.33 (−2.0) | −0.41 (−2.5) | −0.08 (−0.3) | −0.70 (−3.5) | −0.07 (−1.6) |
| *current:* **grid residual (OLS vs DIX)** | −0.01 (−0.2) | −0.02 (−0.4) | −0.02 (−0.2) | +0.05 (+0.4) | +0.19 (+1.2) | −0.08 (−0.6) | −0.03 (−0.8) |
| *current:* DPI − expanding mean | −0.10 (−2.2) | −0.12 (−2.8) | −0.26 (−2.3) | −0.27 (−2.5) | −0.07 (−0.5) | −0.44 (−2.8) | −0.09 (−2.2) |
| *new:* structural level (126-session mean) | −0.10 (−1.7) | −0.13 (−2.4) | −0.35 (−1.8) | −0.55 (−2.7) | −0.22 (−0.7) | −0.84 (−3.4) | −0.09 (−1.8) |
| *new:* **`dpi_z`** | −0.00 (−0.1) | −0.04 (−0.9) | +0.03 (+0.2) | −0.01 (−0.1) | +0.15 (+0.9) | −0.16 (−1.1) | −0.01 (−0.2) |
| *new:* abnormal dark share | −0.02 (−0.5) | −0.08 (−1.9) | +0.06 (+0.4) | −0.14 (−1.2) | −0.33 (−2.0) | +0.03 (+0.2) | −0.01 (−0.3) |
| *new:* dark imbalance (ADV units) | −0.02 (−0.4) | −0.04 (−0.9) | −0.00 (−0.0) | −0.00 (−0.0) | +0.19 (+1.0) | −0.18 (−1.3) | −0.01 (−0.4) |
| *new:* **`dpi_hidden`** (echo removed) | −0.04 (−0.7) | −0.04 (−1.0) | −0.02 (−0.1) | −0.02 (−0.1) | +0.14 (+0.8) | −0.16 (−1.1) | −0.03 (−0.7) |

- The grid residual is flat everywhere, and so is every *flow* measure — including
  the echo-removed one. Standard errors are ~0.04pp/week and ~0.1–0.15pp/month, so a
  top-vs-bottom spread of ~0.1pp/week or ~0.3pp/month (≈4–5%/yr gross) would have
  registered at t ≈ 2.
- The signals that *do* register — raw DPI, D, DPI minus its expanding mean, the
  structural level — all load on the **structural level**, and the sign is
  **negative**: persistently dark names underperformed, entirely in 2023–26
  (−0.84pp/month, t = −3.4) and not in 2019–22. The cross-sectional L/S tab's
  "name-specific" signal (DPI minus its expanding mean) is one of these: a
  structural-level bet, not a flow bet.

### 5b. Robustness — every chance for the flow signals

Same universe and controls (the robustness block of the study output):

| check | 1 week | 1 month |
|---|---|---|
| `dpi_z` built on 1 / 10 / 20-session windows | −0.08 (−1.8) / −0.06 (−1.5) / −0.04 (−0.9) | −0.07 (−0.7) / −0.06 (−0.5) / −0.11 (−0.8) |
| `dark_imbalance` on 1 / 10 / 20-session windows | −0.09 (−2.1) / −0.04 (−1.0) / −0.03 (−0.7) | −0.09 (−0.9) / −0.05 (−0.4) / −0.12 (−0.9) |
| tails: market-relative return when `dpi_z` < −2 / > +2 | +0.04 (0.4) / +0.00 (0.0) | −0.11 (−0.6) / −0.13 (−0.8) |
| "accumulation into weakness": high − low `dpi_z` among past-week losers | −0.04 (−0.7) | −0.07 (−0.5); halves −0.01 / −0.12 |

Nothing. The only |t| ≈ 2 cell is the *one-session* window at one week, and it is
negative — a day of heavy dark buying is followed by slight underperformance, the
price-pressure reversal the echo profile predicts (§3). Extreme flow is no more
informative than ordinary flow, and the general version of the earnings study's
"accumulation into a down-gap" does not hold in large caps.

### 5c. NDX-100 alone (~100 names)

Nothing at 1 month with controls (every |t| ≤ 1.2, the grid residual +0.28, t = 1.0).
The one large half-sample cell — raw DPI in 2019–22, +1.13pp (t = 3.3) — **flips** to
−0.59 in 2023–26: a style exposure (the retail-heavy, high-DPI names ran in 2020–21),
not a stable edge. With ~100 names a decile holds ~10 stocks; the NDX alone cannot
power a cross-sectional test, which is one reason earlier studies came back empty.

### 5d. Russell 2000 — liquid current members (~950 names/week; $ADV ≥ $5M, price ≥ $3)

The part of the market where retail flow dominates and published order-flow effects
are strongest. Structure differs in the expected direction: the market-wide component
is even more contrarian (−0.48 with the market's same-day return), the price echo is
much weaker (+0.04 same day), and 88% of DPI variance is idiosyncratic.

| signal | h21 | h21 +ctrl | h21 +ctrl 2019–22 | h21 +ctrl 2023–26 |
|---|---:|---:|---:|---:|
| raw 1-day DPI | +0.11 (0.6) | +0.06 (0.4) | +0.19 (0.7) | −0.05 (−0.2) |
| grid residual (OLS vs DIX) | +0.16 (1.1) | +0.21 (1.4) | +0.37 (1.9) | +0.06 (0.3) |
| structural level | +0.01 (0.1) | −0.18 (−0.7) | −0.30 (−0.7) | −0.08 (−0.3) |
| **`dpi_z`** | **+0.42 (2.3)** | +0.26 (1.6) | **+0.45 (2.1)** | +0.09 (0.4) |
| dark imbalance | +0.26 (1.6) | +0.25 (1.5) | **+0.49 (2.3)** | +0.03 (0.1) |
| **`dpi_hidden`** | **+0.37 (2.1)** | +0.25 (1.6) | **+0.45 (2.1)** | +0.08 (0.3) |

| robustness (1 month, +ctrl) | spread (t) |
|---|---|
| `dark_imbalance`, 1-session window | +0.26 (2.1) |
| tails: `dpi_z` > +2 | +0.37 (1.7) |
| **"accumulation into weakness": high − low `dpi_z` among past-week losers** | **+0.36 (2.3)** — halves +0.49 (2.0) / +0.23 (1.2) |

This is the only place the flow measures lean the way the dark-accumulation story
predicts — **bullish** — and the relative-darkness constructions do lean harder than
the raw ratio or the residual. But the effect is small (≈ +0.3–0.4pp/month
top-to-bottom), does not clear t = 2 with controls, and lives in 2019–22 (the retail
boom) with little left since 2023. The one cell whose sign holds in both halves is
the conditional one: small caps that fell over the past week *while* taking abnormal
dark buying out-returned those that fell without it by +0.36pp over the next month —
the small-cap echo of the earnings study's gap-down result. It is on the watch list
(§7), not in a strategy.

### 5e. ATS dark pools vs internalizers (FINRA OTC transparency, weekly, Dec 2021 →)

FINRA's weekly OTC-transparency totals split off-exchange volume into **ATS** (the
dark pools — mostly institutional) and **non-ATS OTC** (mostly wholesaler
internalization of retail orders). That is the most literal "relative darkness"
available: the median S&P/NDX name runs 55% of its off-exchange volume through ATSs,
mega-caps only 10–20% (retail internalization dominates them). Entered at the first
close after FINRA publishes the week (~3 weeks after it starts):

| signal | h5 +ctrl | h21 +ctrl | h21 +ctrl, 2022 – May 2024 | May 2024 – Aug 2026 |
|---|---:|---:|---:|---:|
| abnormal ATS share of off-exchange volume | +0.03 (0.7) | +0.02 (0.1) | +0.17 (1.1) | −0.13 (−0.7) |
| abnormal ATS share of *all* volume | +0.01 (0.2) | +0.09 (0.6) | +0.05 (0.3) | +0.14 (0.6) |
| ATS volume surprise (vs own 26-week mean) | +0.17 (2.5) | +0.36 (1.9) | +0.07 (0.3) | **+0.63 (2.4)** |

Shifting flow into or out of dark pools carries nothing. Unusually *heavy* dark-pool
volume has leaned bullish, but only in the last two years — before May 2024 it was
flat — so it too is a watch-list item rather than a finding.

### 5f. Does DPI work better on certain names?

Two ways to ask it: pick **names** (each name's own history), or pick
**characteristics** (what kind of name). `python dark_flow_study.py --by-name` runs
both.

**Picking names.** Each name gets its own weekly time-series slope of
market-relative forward return on its own signal, with a Newey-West t — the question
the dashboard's per-name decile bars answer by eye. The baseline is a placebo: every
name's signal circularly shifted in time 40 times (its autocorrelation kept, its
alignment with returns destroyed). Then the practical test: select the names with
|t| > 1.5 in 2019–22 and trade each *in its own direction* in 2023–26, scored against
100 placebo runs of the same procedure.

| universe, signal, horizon | SD of per-name t, real / placebo | names past \|t\| = 2, real / placebo | rank-corr of slopes, 2019–22 vs 2023–26 | trade 2019–22's names in their own direction in 2023–26 |
|---|---:|---:|---:|---|
| S&P ∪ NDX, D, 1 month | 1.15 / 1.14 | 8.7% / 8.1% | +0.06 | +0.25 (t 1.5) — the same names in *one common* direction do +0.19: it's the bearish level effect, not name picking |
| S&P ∪ NDX, `dpi_z`, 1 week | 0.99 / 1.01 | 4.5% / 4.9% | −0.11 | −0.00 (t −0.0), beats 47% of placebo runs |
| S&P ∪ NDX, `dpi_z`, 1 month | 1.15 / 1.15 | 8.0% / 8.2% | +0.05 | +0.10 (t 1.2), beats 77% |
| Russell, D, 1 month | 1.23 / 1.18 | 10.5% / 8.7% | +0.07 | +0.19 (t 1.5); timing only +0.24 (t 2.8), beats 99% |
| Russell, `dpi_z`, 1 week | 1.06 / 1.00 | 5.5% / 4.6% | +0.04 | +0.06 (t 1.4), beats 87% |
| **Russell, `dpi_z`, 1 month** | 1.19 / 1.13 | 9.6% / 7.4% | +0.02 | **+0.29 (t 3.0), beats all 100 placebo runs**; timing only t 3.3; one common direction −0.04 |

- **Large caps: no.** How many names "work" matches the placebo almost exactly, and
  the names that worked in 2019–22 were back to zero in 2023–26 (top quintile's
  average t: +1.6 → +0.2). Mind the base rate: with overlapping one-month returns,
  about **one name in twelve clears |t| = 2 under the placebo** — ~40 S&P names that
  look like DPI works on them from noise alone.
- **NDX-100:** 6 names clear |t| = 2 on abnormal DPI at one month (chance: 7.6), and 8
  keep |t| > 1 the same way in both halves (chance: 6.3). The most consistent-looking
  — XEL, INTU and BKR negative; ROP and FAST positive; NFLX, +4.2 in 2019–22 and −0.2
  since — are the names chance would hand you.
- **Small caps: yes, modestly.** Per-name dispersion runs a little above chance, and
  picking the names where abnormal DPI predicted returns in 2019–22 — long or short,
  each in its own direction — earned +0.29pp per 1-SD position per month in 2023–26
  (t = 3.0), better than every placebo run. It is name-specific (one common direction
  earns −0.04) and not a static tilt (timing only, t = 3.3). But slope ranks barely
  correlate across halves (+0.02), so it lives in a minority of names, and it is a
  few percent a year gross on small caps whose positions turn over weekly.

**Picking characteristics.** Signal × characteristic interaction t-stats in the full
cross-section (with the usual controls; each characteristic measured on trailing
data), one week / one month:

| interaction t | size | off-exchange share | **structural DPI level** | volatility | DPI persistence | price echo |
|---|---:|---:|---:|---:|---:|---:|
| S&P ∪ NDX, `dpi_z` | −0.2 / −0.3 | +1.1 / +0.7 | **+2.2 / +2.3** | +0.7 / +1.5 | +1.1 / +0.3 | −2.0 / −1.2 |
| S&P ∪ NDX, D | −1.8 / −1.3 | +0.6 / +0.4 | **+1.9 / +1.4** | −0.4 / −0.2 | +1.6 / +1.0 | −1.2 / −0.3 |
| Russell, `dpi_z` | +2.6 / +1.0 | +0.9 / +2.0 | **+1.7 / +2.3** | +1.6 / +2.5 | +1.0 / −0.3 | −0.2 / −0.9 |
| Russell, D | +1.1 / +0.7 | −0.1 / +0.5 | **+2.1 / +2.0** | −0.0 / −0.3 | +0.4 / −0.5 | −0.4 / −0.3 |

In large caps, 3 of 24 interactions reach |t| ≈ 2 (two of them the structural-level
one below); in small caps, 6 of 24 do, against ~1 expected at 5%. The one
characteristic that lines up in *every* cell of both universes
is **structural darkness** — how dark the name normally is (its trailing 126-session
DPI level). Abnormal dark buying leans bullish in structurally dark names and
neutral-to-bearish in structurally lit ones:

| `dpi_z`, 1 month, +ctrl | structurally lit tercile | structurally dark tercile | dark tercile, 2019–22 / 2023–26 |
|---|---:|---:|---:|
| S&P ∪ NDX | −0.10 (−0.6) | +0.27 (+1.6) | +0.50 (+1.7) / +0.07 (+0.4) |
| Russell | −0.13 (−0.6) | **+0.65 (+2.9)** | +0.98 (+2.9) / +0.35 (+1.2) |

**Sectors add nothing.** At one month, abnormal DPI reaches |t| = 2 in 0 of 11 S&P
sectors and 1 of 11 Russell sectors (small-cap utilities: 30 names, −0.81, t = −2.4)
— what chance gives. The level signal is negative in 9 of 11 large-cap sectors
(strongest in Staples and Financials, t ≈ −2.5): the broad bearish level effect of
§5a, not a sector story.

**Bottom line.** DPI does not work better on particular large-cap or NDX names. The
dashboard's per-name decile bars will always show some names with striking patterns,
because about one in twelve clears |t| = 2 at a one-month horizon by chance, and
those names don't keep working. In small caps there is a real but modest name-level
component. Its best describable form is structural darkness: abnormal dark buying in
names that are normally dark. Like everything else here, it was strongest in 2019–22.

---

## 6. What dark flow *does* carry: volatility

Same cross-sections, outcome = log realized vol over the next 5 / 21 sessions,
controlling for 5- and 21-session realized vol, volume surprise, $ADV, absolute 1- and
5-session returns and the 5-session return (log-vol spread, lowest → highest ranked):

| signal | S&P ∪ NDX h5 | S&P ∪ NDX h21 | Russell h21 |
|---|---:|---:|---:|
| structural D (5-day) | +0.022 (4.5) | +0.022 (4.3) | −0.031 (−6.3) |
| `dpi_z` | −0.013 (−2.9) | −0.019 (−4.4) | −0.017 (−4.5) |
| **abnormal dark share** | **−0.037 (−7.8)** | **−0.038 (−7.7)** | **−0.035 (−7.9)** |
| dark imbalance | −0.016 (−3.5) | −0.018 (−4.4) | −0.014 (−3.8) |
| grid residual | −0.013 (−2.8) | −0.015 (−3.7) | −0.012 (−3.3) |

When more of a name's trading moves off-exchange than usual, the next weeks are
calmer (information-driven trading goes to lit venues for immediacy; quiet periods
are retail- and internalization-heavy). It is highly significant because the cross-
section is large, but the magnitude is ~4% of volatility (e.g. 30% vs 28.9%) —
inside the typical implied-vol bid/ask. Useful as one input to a vol forecast; not a
standalone trade.

---

## 7. Recommendations

1. **Retire the name-level dark residual as a directional signal.** Keep the grid as
   a descriptive monitor if it is useful, but plot `dpi_z` (own-σ abnormal,
   volume-weighted DPI) rather than `D − DIX`, and say what the echo is. Don't read a
   single name's decile bars as evidence: at one month, ~1 name in 12 looks like DPI
   works on it by chance (§5f).
2. **Re-run the index studies on corrected gauges.** The next nightly build uses the
   fixed dollar-DIX; `INDEX_COMOVEMENT_FINDINGS.md`'s tables were computed on
   mis-weighted gauges and its surviving family does not survive correction (§1a).
3. **If dark flow stays in the toolkit, use it for risk, not direction**: as a
   volatility input next to the vol tracker, or as conditioning inside event studies
   (the earnings gap-down drift result is the only conditional survivor, and still
   needs an out-of-sample test).
4. **Pre-registered watch list** — each needs t > 2 with the *same* construction in
   Oct 2026 → Dec 2027 before it is taken seriously: (a) small-cap "accumulation into
   weakness" (high − low `dpi_z` among past-week Russell losers, bullish, both halves),
   (b) Russell abnormal flow (`dpi_z`, bullish, 2019–22 only), (c) ATS volume surprise
   (bullish, 2024–26 only), (d) structural darkness (bearish, 2023–26 only), (e)
   small-cap name selection — Russell names picked on their own `dpi_z` history and
   traded in their own direction, rolled forward yearly — and (f) `dpi_z` within the
   structurally dark tercile.
5. **Better data, not better transforms, is what would move this.** Daily FINRA short
   ratios mix market-maker inventory mechanics with customer flow and cannot separate
   informed from uninformed orders. Trade-level retail identification (sub-penny price
   improvement in TAQ) or venue-level ATS prints are where published order-flow
   effects live.

## Caveats

- **Survivorship:** universes are *current* members; names that left are missing. This
  flatters average returns more than the cross-sectional spreads tested here.
- **One regime of history:** FINRA's consolidated files start 2018-08; the public ATS
  split Dec 2021. A 2019–26 sample is mostly a bull market with two sharp drawdowns.
- **Multiple testing:** several hundred (correlated) signal × horizon × control ×
  period cells were run across the three universes; at 5%, a couple of dozen |t| > 2
  cells are expected by chance. The |t| ≈ 2–2.5 cells in §5 sit inside that envelope,
  which is why only effects that hold across both halves get attention — and the one
  that does (§5d, small-cap accumulation into weakness) is still marginal.
- FINRA's daily short-sale files cover regular-hours trades reported to the TRFs/ADF —
  ~75–90% of the weekly OTC-transparency volume for the same names.
