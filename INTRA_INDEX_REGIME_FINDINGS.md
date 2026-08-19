# Intra-index comovement regimes × DIX → 1-month forward returns

The cross-index comovement study asked how the three DIX gauges line up
*across* indices. This study divides the data into regimes *inside* the
NDX-100: at times all ~100 stocks rally together and correlation is high; in
lower-correlation periods only a certain group rallies while the rest sells
off. It then asks how the DIX reads inside each regime — at the index level
and per name. **30 Aug 2018 – 18 Aug 2026, 2,001 sessions, 102 grid names.**

Reproduce:
```
python intra_index_regime_study.py --csv intra_index_regimes.csv
```
Data is read from the payload embedded in `docs/index.html` (per-name
split-adjusted closes, raw dark ratios, adjusted 1-month forward returns —
both payload encodings supported). The optional external cross-check reads
`docs/gex_dispersion.html`.

## Definitions

- **AVG_CORR** = equal-weight average pairwise correlation of daily returns
  across all names with a full trailing 21-session window (min 30 names).
  The NDX-internal realized analog of Cboe's implied-correlation gauges.
- **DISP21** = 21-session mean of the daily cross-sectional std of returns.
- **BREADTH** = fraction of names with a positive trailing 21-session return.
- **Comovement regime** = Low/Mid/High on a 30/40/30 split of AVG_CORR, on
  two bases: **full-sample** cutoffs (mild look-ahead) and **expanding**
  cutoffs (each day vs its own past only, min 250 obs — live-knowable).
- **DIX5** = 5-day MA of the payload's NDX dollar-DIX, zoned the same way.
- **Tilt** (per name) = 5d MA of the raw daily dark ratio minus the name's
  own expanding mean (min 60 obs) — the dashboard's "name-specific vs own
  average" signal. **Tilt spread** = mean 1-month forward return of the
  top-20%-tilt names minus the bottom-20%, each day.
- **Forward return** = QQQ 21 sessions ahead (payload `r21`, %); per-name
  outcomes use the per-name adjusted `r21`. No look-ahead in any outcome.
- **CI** = 95% moving-block bootstrap (21-day blocks); suppressed under 42
  scored days.
- Packed closes are split-adjusted but not dividend-adjusted; every name's
  compounded 21d close return is validated against its adjusted `r21`
  (0.98 gate; nothing dropped in the current build — worst name 0.995,
  dividends only).

## The regimes themselves (full-sample basis)

| Regime | n | avg corr | dispersion | breadth | QQQ vol | QQQ 1m fwd | 95% CI | hit% |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| LowCorr  | 601 | 0.13 | 2.31% | 0.62 | 14.6 | **+0.33%** | [−0.9, +1.5] | 57% |
| MidCorr  | 799 | 0.27 | 1.95% | 0.60 | 17.9 | +2.08% | [+0.8, +3.3] | 70% |
| HighCorr | 601 | 0.48 | 2.14% | 0.48 | 31.8 | +2.47% | [+0.4, +4.5] | 66% |

High-comovement tapes are panic tapes (vol 32, breadth 0.48 — everything
falling together) and precede the *best* months — mean reversion. The
low-correlation, dispersed tape precedes the *weakest* months. Under
expanding cutoffs the gap narrows (LowCorr +1.36%, HighCorr +2.29%) — part
of the regime-alone effect is full-sample relabeling.

Longest episodes land where they should: **HighCorr** = the 2022 bear
(Feb–Jul 2022, Aug 2022–Feb 2023), the COVID crash (Feb–May 2020), late
2018; **LowCorr** = Jul–Nov 2025, Apr–Aug 2026 (the current episode),
Dec 2025–Mar 2026, May–Jul 2024, Aug–Sep 2021. **As of 18 Aug 2026 the
index sits at AVG_CORR 0.07 — the 2.5th percentile of the whole history —
with cross-sectional dispersion at 3.6%/day**: the most dispersed tape in
the sample.

## Headline: the DIX gradient lives only in the low-correlation regime

QQQ 1-month forward by comovement regime × DIX regime, **expanding
(no-look-ahead) basis**:

| | DIX Low | DIX Mid | DIX High |
|---|---:|---:|---:|
| **LowCorr**  | −0.11% (n=120, hit 46%) | +0.87% (n=386) | **+2.76% (n=249, hit 82%, CI [+1.3, +4.2])** |
| **MidCorr**  | +3.18% (n=61) | +3.20% (n=197) | +1.51% (n=388) |
| **HighCorr** | +6.22% (n=30) | +2.50% (n=101) | +1.65% (n=220) |

- **In the dispersed tape, DIX discriminates hard**: a ~2.9pp Low→High
  gradient with the only tight, positive CI in the row. The same pattern
  holds on the full-sample basis (−0.71% → +1.45%).
- **In mid/high-correlation tapes the gradient is flat or inverted** — in
  panic regimes low DIX (capitulation in dark flow) precedes the biggest
  bounces (+6.22%, but n=30 is anecdote-sized). Mean reversion dominates
  and the DIX level adds nothing.
- The interaction regression (`r1m ~ zDIX + zCORR + zDIX·zCORR`, NW-21)
  agrees in sign but is not significant at 5%: interaction −0.63
  (t=−1.43), −0.66 (t=−1.56) with a realized-vol control. AVG_CORR and
  index vol are 0.77 correlated; with the vol control neither carries the
  level effect alone. Treat the 3×3 as descriptive of when DIX has worked,
  not a proven coefficient.

## Which group rallies? Per-name dark flow in the dispersed tape

Daily Q5-minus-Q1 tilt spread of 1-month forward returns, by regime
(full-sample basis):

| Regime | n | spread | 95% CI | Q5 (high tilt) | Q1 (low tilt) |
|---|---:|---:|---|---:|---:|
| LowCorr  | 552 | **−1.59pp** | **[−2.6, −0.6]** | +0.64% | +2.23% |
| MidCorr  | 797 | +0.02pp | [−0.6, +0.6] | +2.15% | +2.14% |
| HighCorr | 593 | −0.07pp | [−0.8, +0.6] | +3.44% | +3.51% |

**In low-correlation regimes, per-name dark flow does identify the groups —
with a negative sign.** Names whose dark ratio runs above their own norm lag
by ~1.6pp/month (the only CI excluding zero); the group that rallies is the
one *without* an elevated dark-flow tilt. In mid/high-correlation tapes
everything comoves and the cross-sectional signal carries nothing. The
effect is nonlinear (concentrated in the low-corr tercile): the continuous
slope vs zCORR is only +0.37pp/z (t=+1.22).

This matches the dashboard's long-standing per-name finding (elevated
name-specific dark ratio skews forward returns negative) and sharpens it:
**that signal is a low-correlation-regime phenomenon.**

## Tape taxonomy — "all rally" vs "only a certain group rallies"

Trailing 21d QQQ sign × breadth tercile:

| Tape | n | avg corr | QQQ 1m fwd | 95% CI |
|---|---:|---:|---:|---|
| broad rally | 667 | 0.28 | +1.18% | [−0.2, +2.5] |
| mid rally | 561 | 0.21 | +1.44% | [+0.5, +2.4] |
| narrow rally | 74 | 0.25 | +2.48% | [+1.3, +3.7] |
| selective selloff | 2 | 0.39 | — | — |
| mid selloff | 96 | 0.28 | −0.14% | [−1.9, +1.8] |
| broad selloff | 601 | 0.39 | +2.68% | [+1.0, +4.5] |

- **Selective selloffs barely exist** (2 days in eight years): when the
  index falls, breadth collapses and correlation spikes with it. The
  asymmetry the question described is real — selectivity is a *rally-side*
  phenomenon.
- Narrow rallies (index up, breadth in the bottom tercile) have *not* been
  fragile: +2.48%, 72% hit — leadership persisting, not topping.
- DIX splits the rally tapes: broad rally → DIXLow +0.31% / DIXHigh +2.47%;
  mid rally → −0.27% / +2.77%. In broad selloffs the *low*-DIX washouts
  bounce hardest (+3.40%).

## Regime-entry event study (expanding basis, 21-session cool-down)

| Entry condition | entries | QQQ 1m | hit% |
|---|---:|---:|---:|
| enter LowCorr & DIXHigh | 21 | **+3.17%** (med +3.96) | 81% |
| enter LowCorr | 18 | +2.92% | 83% |
| enter LowCorr & DIXLow | 13 | +1.41% (med +0.99) | 58% |
| enter HighCorr | 17 | +0.91% | 76% |
| enter HighCorr & DIXLow | 6 | −0.76% | — |

Unlike the cross-index LLH divergence (which failed entry-day scoring), the
low-corr + high-DIX setup survives it: the edge is present at formation, not
only deep inside episodes.

## Out-of-sample split (fit < 2024, evaluate 2024+; 659 test days)

- Test baseline: QQQ +1.96% (hit 69%).
- **The tilt spread holds OOS**: −1.94pp in LowCorr test days (vs −0.3pp /
  −7.9pp on thin Mid/High samples).
- **The index-level DIX gradient does not stay monotone OOS**: LowCorr row
  DIXLow +1.80% (n=57) / DIXMid +0.21% (n=305) / DIXHigh +1.79% (n=105) —
  the High leg still beats Mid, but the Low leg stopped being bad in
  2024+.
- The train-fitted interaction model transfers modestly: OOS corr(pred,
  realized) = +0.18; days it liked beat the test average by +1.4pp. (The
  cross-index two-factor model managed 0.00 on the same kind of test.)

## External gauge cross-check

AVG_CORR vs the GEX/dispersion barometer: corr(level) = **+0.95** against
the realized top-50 SPX correlation and **+0.73** against Cboe COR1M
implied correlation (+0.90 / +0.34 on 21d changes). The NDX-internal gauge
is measuring the same comovement the market-wide gauges see, computed from
nothing but the payload.

## What the pattern says

1. **Comovement is a real conditioning variable for the DIX.** The
   unconditional "level carries nothing" result from the cross-index study
   decomposes: DIX level carries a ~3pp/month monotone gradient in
   low-correlation tapes and nothing (or the opposite) in high-correlation
   tapes, where mean reversion dominates.
2. **In dispersed tapes, per-name dark flow tells you which group** — avoid
   the names with elevated dark-flow tilt (−1.6pp/month, CI excludes zero,
   holds OOS). This is the study's most robust cell.
3. **Selectivity is rally-side only**: selloffs are broad by construction
   (correlation spikes), so "only a certain group rallies" regimes are
   low-corr *up* tapes — and narrow rallies have continued, not died.
4. **The regime now** (Aug 2026): record-low pairwise correlation, high
   dispersion, DIX mid — the exact regime where the per-name tilt spread
   has been most informative and the index-level DIX zone is the thing to
   watch.
5. The interaction coefficient is only marginal (t ≈ −1.5) and the
   index-level gradient softened OOS — treat the 3×3 as a conditioning map,
   not a standalone trigger.

## Caveats

- **Survivorship.** The panel is the *current* NDX grid (102 names incl.
  recent IPOs with short history); early-sample gauges are computed over
  the subset that existed. Comovement gauges are fairly insensitive to
  membership, but per-name results describe today's constituents.
- **Overlapping windows.** 2,001 sessions ≈ ~95 independent months; block
  bootstrap and the entry study are the honest lenses.
- **Corr ↔ vol.** AVG_CORR and QQQ realized vol are 0.77 correlated; with a
  vol control neither is separately significant. "Low-corr regime" and
  "quiet tape" largely overlap in this sample.
- **Equal-weight gauges.** AVG_CORR/BREADTH weight AAPL and the 100th name
  equally; a cap-weighted variant would track the index's own variance mix.
- Tape/breadth cutoffs are full-sample (descriptive); the tape table is not
  a live signal as printed.
- NDX only: the payload packs full-history per-name closes only for the
  NDX grid. Extending to SPX/IWM needs a constituent price fetch (see
  `build_gex_dispersion.py`'s top-50 basket for the pattern).
- The 3×3 table (with CIs) is written to `intra_index_regimes.csv`.
