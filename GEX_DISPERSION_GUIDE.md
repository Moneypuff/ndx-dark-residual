# How to Use the GEX × Dispersion Barometer

A field guide to the two dials, the four quadrants, and the regime-break
detector — what each reading means, what to do with it, and exactly how much
to trust it. All statistics below come from the GEX+ era build
(5,560 sessions, Jul 2004 → Aug 2026) and the 150-signal pair backtest run
against it; rerun the commands in §7 to refresh them.

---

## 1. The two dials, in one breath

* **Gamma (GEX+, trailing-year percentile).** How hard dealer hedging leans
  *against* the tape. High percentile = pinned: rallies and selloffs both get
  faded, realized vol gets suppressed. Low percentile = unclamped: hedging
  flows amplify moves instead of damping them. Use the *percentile*, never the
  dollar level — the dollar series drifts upward with the options market.
* **Correlation (realized 1-month, top-50 S&P names).** Whether the index is
  one trade or five hundred. Low correlation = dispersed, stock-picking pays,
  index vol stays under single-stock vol. High = macro has the wheel.
  Cboe's COR1M and DSPX are on the chart as the implied cross-check; ours are
  realized, so ours lag and theirs lead.

**The single most important calibration:** these dials forecast *volatility*,
not *direction*. Across four quadrants, mean 1-month SPX forward returns span
less than 0.7pp — but forward realized vol spans **11.6 to 21.8 points**.
Size positions off the quadrant; don't pick direction off it.

## 2. The quadrant playbook

| Quadrant | Days | Fwd 1-mo SPX | Hit | Fwd vol | How to treat it |
|---|---|---|---|---|---|
| **Pinned & dispersed** (G+ C−) | 1,961 | +0.51% | 64% | **11.6** | The grind. Carry, pair trades, and stock-picking work; index options are sellable; moves mean-revert intraday. Complacency risk builds the longer it runs. |
| **Pinned but synchronized** (G+ C+) | 1,126 | +1.13% | 70% | 14.0 | Macro-driven but damped. Trends persist gently; the best hit-rate bucket. |
| **Unclamped & synchronized** (G− C+) | 1,660 | +1.17% | 67% | **21.8** | The storm quadrant. Widest outcomes in both directions — returns average fine *because* rebounds live here, but paths are violent. Cut gross, buy convexity, don't sell vol. |
| **Unclamped & dispersed** (G− C−) | 813 | +0.54% | 63% | 14.9 | The transition state. Leadership rotations happen here (see §4). Vol tends to *expand* out of it (+3.3 pts vs trailing). |

Baseline for comparison: +0.84% mean, 66% hit, 15.6 fwd vol.

Read the current quadrant off the hero card of `docs/gex_dispersion.html`;
the map (section 01) shows where today sits inside the continuum — a
51st-percentile day is not a 99th-percentile day, and the table above is
deliberately blunt about that.

### 2a. Intraday range by cell — the 4×4 (page section 05)

The four-quadrant table smooshes a barely-pinned day into the same bucket as a
deeply-pinned one. Section 05 of the page splits **each dial into quartiles**
(fixed 25/50/75 cuts on the same percentiles the quadrants use) for a **4×4
grid**, and scores each of the 16 cells by SPX's **daily high-low range** —
`100 × (High − Low) / prior close`. This answers the sizing question the forward
tables don't: *while I sit in this regime, how far does SPX actually travel in a
day?*

Three things to keep straight:

* **Two bases, and the default is the tradeable one.** The regime is fixed at
  the *close* (both dials are computed from the day's data), so you can't trade
  the same session. The matrix defaults to **Next day** — session *t*'s cell
  scored by session *t+1*'s high-low range, the swing you can actually act on
  given today's reading — with a toggle to **Same day**, the contemporaneous
  range (the environment you're sitting in). They come out **nearly identical**
  because regimes are persistent: the grind corner is 0.65% same-day / 0.69%
  next-day, the storm corner 2.22% / 2.18% — the extremes just ease a hair one
  day out (mean reversion), and the structure is unchanged. The map is not an
  artifact of same-day conditioning.
* **Each cell reports mean · median · MAD · p90 · n.** **MAD** is the mean
  absolute deviation *about the mean* — `mean(|x − mean(x)|)` — i.e. how much the
  daily range itself wobbles day to day inside the cell, a spread-of-the-spread.
  **p90** is the hot session (the 90th-percentile range). As everywhere in this
  study, trust the **median and n** over the mean in thin corner cells — the
  high-gamma/high-corr and low-gamma/low-corr diagonals are the sparse ones.
* **Colour = distance from the all-day baseline.** Cooler cells run tighter than
  a typical day, hotter cells wider. The grind corner (pinned & dispersed) sits
  cool; the storm corner (unclamped & synchronized) runs hottest — the same
  vol story the forward table tells, now in plain intraday points.

Range source: SPX high/low come from the GEX+ yacht-club feed (which ships OHLC)
and fall back to Yahoo `^GSPC` for the classic-GEX path, which carries close
only.

## 3. Reading the dials day to day (the 2-minute routine)

1. **Quadrant + how deep into it** (hero card, then the map). Near a boundary?
   Expect flicker; the bands in section 03 show how choppy the recent regime
   has been.
2. **GEX+ percentile path, not level.** A collapse of 40+ points inside a
   month is the "rotation window open" tell — leadership changes become
   *possible*. It is not by itself bearish.
3. **DSPX vs our realized dispersion.** Implied leads, realized lags:
   * DSPX rising while realized is flat → street paying up for dispersion
     (single-stock event risk ahead: earnings, product cycles).
   * DSPX falling hard (21d change z ≤ −1.5) while realized stays high → the
     dispersion trade is being *taken down*. Crowded relative-value themes are
     being unwound — check your pair exposures.
4. **COR1M.** Quiet (single digits/low teens) = whatever is happening is
   rotation, not crisis. A 21d-change z ≥ +2 spike = systemic unwind in
   progress; switch to the §5 shock rules.

## 4. Detecting a theme break (the pair detector)

For any long/short theme (semis−software, growth−value, QQQ−IWM…):

* **Theme "on":** |63-day return spread| > 5pp.
* **Break signal:** the 21-day return spread's z-score (vs its trailing year)
  crosses ±1.5 *against* the theme.
* **Gamma-release tag:** GEX+ percentile fell ≥ 40 points over the prior 21
  sessions, within 10 sessions of the break.
* **Shock tag:** a COR1M spike (21d-change z ≥ +2) within roughly a month
  before to a week after the break.

The historical sequence of a *real* rotation, worth pattern-matching:
**gamma releases → theme's ratio tops → 21d spread flips (often on the exact
DSPX peak) → DSPX premium unwinds while realized dispersion stays high.**
That is July 2024 and July 2026, beat for beat.

## 5. What to do with a break signal — the three buckets

Backtest: 12 pairs × both directions, 2005–2026, 150 signals. Old-theme
forward 63-session spread (positive = theme resumed):

| Context at the break | n | Fwd 63d | % neg | Action |
|---|---|---|---|---|
| **Near a correlation shock** | 28 | **+3.7pp** | 43% | **Fade the break.** Shocks knock every spread down together; it says nothing about the theme. The strongest, sturdiest rule in the study (Mar 2026 "breaks" reversed +43 to +61pp). |
| **No shock, no gamma release** | 86 | +0.6pp | 51% | **De-risk, don't flip.** The theme's edge is gone (vs +0.4 baseline it's noise); direction is a coin flip. |
| **No shock + gamma release** | 36 | **−1.6pp** (med −2.2) | 64% | **Treat the theme as over.** This bucket *is* the famous-rotation list: Jan 2017 semis, Jul 2024, DeepSeek Jan 2025. |

**Confidence calibration — read this twice.** The shock-fade rule is the
strongest stat in the table. The gamma-release lean is real-looking but
*statistically soft*: ~2pp/quarter of edge at placebo p ≈ 0.10–0.15. It is a
tilt on exits, not a standalone short signal. What upgrades any individual
signal materially: **multiple themes breaking simultaneously** with gamma
released (Jul 2024 and Jul 2026 both printed triple breaks within two weeks).
One pair breaking = interesting; three = the tape is rotating.

## 6. What this framework cannot do

* **No direction calls on the index.** 0.7pp of return spread across
  quadrants is noise; 10 points of vol spread is signal. Use it for sizing,
  vol posture, and theme management.
* **The gamma-release bucket is n=36, p≈0.1.** Robust across a parameter
  plateau (z 1.5–1.75, drop 30–40) but degrades outside it. Never bet the
  book on it alone.
* **Realized gauges use today's top-50 basket across all history** —
  survivorship and weight drift grow with lookback. Cboe's series are the
  point-in-time clean versions; when ours and theirs disagree on *level*,
  ignore it (implied ≠ realized by construction); when they disagree on
  *direction of change*, trust theirs.
* **In-sample cuts.** The correlation dial's High/Low midpoint is full-sample;
  quadrant stats are descriptive history, not out-of-sample forecasts.
* **Overlapping windows.** 5,560 sessions ≈ 265 independent months. Medians
  and hit-rates over means, always.
* **GEX+ is SqueezeMetrics' model.** We rank it; we don't re-derive it. Treat
  the percentile as ours and the dollar level as theirs.

## 7. Running it

```bash
# the page (docs/gex_dispersion.html); GEXPLUS_KEY env enables the GEX+ feed,
# without it the build falls back to the public classic-GEX series
GEXPLUS_KEY=... python build_gex_dispersion.py --docs-out docs/gex_dispersion.html \
    --cache-dir .ndx_dark_cache
```

```bash
# the playbook page (docs/regime_log.html): posture, dated break signals with
# bucket tags and realized outcomes, sector leaderboard. Needs the barometer
# page built first.
GEXPLUS_KEY=... python build_regime_log.py --docs-out docs/regime_log.html \
    --cache-dir .ndx_dark_cache
```

The nightly Pages workflow rebuilds both after each close (set the
`GEXPLUS_KEY` Actions secret). The regime log shows every signal with the
bucket tags and the n's from §5 verbatim — the framework's honesty about its
own sample sizes is the feature — and scores each past signal once its
forward 63 sessions have elapsed.

The leaderboard also carries a **conviction column** (0–4, one point per
confirmed component: MA side of the 21/50/200d stack; a 200d cross or
trailing-year volume-profile value-area exit within 10 sessions; a close
beyond the 21d ±2σ Bollinger band within 5; 5d volume ≥ 1.25× 63d) next to
each ETF's **own historical follow-through** for breaks in that direction.
Two hard-won calibrations: conviction is *asymmetric* — high-conviction
up-breaks followed through (+3.0% median next quarter, 67%, n≈1,600) while
down-breaks historically *faded* (the ETF rallied ~5% on average after them;
capitulation reads contrarian) — and follow-through is a property of the
ETF, not the setup: SMH chases well (+9.8% med, 74%), GDX punishes chasers
(−0.5%, 48%) even though GDX's *turn signals* are 5-for-5. High conviction
tells you the break is real; the follow-through column tells you whether
that ETF rewards buying it late.

## 8. Worked example — semis vs software, 2026

* **May 29:** Pinned & dispersed; GEX+ p86; SMH/IGV theme +20pp/quarter.
* **Jun 18–22:** GEX+ p86 → p7 in three weeks (rotation window opens);
  quadrant slips to Unclamped & dispersed; ratio peaks Jun 22. *Action per
  §3: theme is now break-able; tighten stops on the crowded expression.*
* **Jul 14:** 21d spread flips negative — on the exact day DSPX peaks (47.4).
  No COR1M shock (last one Mar 6). Gamma-release tag on. *Bucket: rotation.
  Action per §5: exit, lean against re-entry.*
* **Jul 1–15:** QQQ−IWM and VUG−VTV also break, gamma-flagged — the triple-
  break upgrade.
* **Aug 3–7:** DSPX unwinds to 36.7 (−2.5σ 21d change) with realized
  dispersion still 47; COR1M never left single digits; gamma re-pins (p70).
  *Read: a completed leadership rotation inside an intact dispersion regime —
  not a systemic unwind. New leadership settling.*

Spread at writing: −13 to −16pp. Per §5, the old theme's edge after such
signals was zero-to-negative historically; the dip is not the old regime
on sale until the bucket context changes (a fresh COR1M shock would
reclassify everything).
