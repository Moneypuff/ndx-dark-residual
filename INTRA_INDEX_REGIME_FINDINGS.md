# Intra-index comovement regimes × DIX → 1-month forward returns

The cross-index comovement study asked how the three DIX gauges line up
*across* indices. This study divides the data into regimes *inside* each
index — NDX-100, S&P 500, Russell 2000/IWM: at times all constituents rally
together and correlation is high; in lower-correlation periods only a
certain group rallies while the rest sells off. It then asks how each
index's DIX reads inside each regime — at the index level and per name.
**NDX: 30 Aug 2018 – 18 Aug 2026, 2,001 sessions, 102 grid names. SPX/IWM:
6 Jan 2020 →, 1,664 sessions, top-99 iShares baskets.**

Reproduce:
```
python intra_index_regime_study.py --csv intra_index_regimes.csv   # all three
python intra_index_regime_study.py --indices ndx                   # offline
```
NDX is read entirely from the payload embedded in `docs/index.html`
(per-name split-adjusted closes, raw dark ratios, adjusted 1-month forward
returns — both payload encodings supported). SPX/IWM comovement gauges come
from fetched constituent baskets (top-100 IVV / IWM holdings by weight,
Yahoo adjusted closes via the shared incremental cache; holdings cached to
JSON), with DIX and outcomes from the payload's own `spx` / `iwm` series.
The optional external cross-check reads `docs/gex_dispersion.html`.

## Definitions

- **AVG_CORR** = equal-weight average pairwise correlation of daily returns
  across all names with a full trailing 21-session window (min 30 names).
  The index-internal realized analog of Cboe's implied-correlation gauges.
- **DISP21** = 21-session mean of the daily cross-sectional std of returns.
- **BREADTH** = fraction of names with a positive trailing 21-session return.
- **Comovement regime** = Low/Mid/High on a 30/40/30 split of AVG_CORR, on
  two bases: **full-sample** cutoffs (mild look-ahead) and **expanding**
  cutoffs (each day vs its own past only, min 250 obs — live-knowable).
- **DIX5** = 5-day MA of each index's dollar-DIX, zoned the same way.
- **Tilt** (per name) = 5d MA of the raw daily dark ratio minus the name's
  own expanding mean (min 60 obs) — the dashboard's "name-specific vs own
  average" signal. **Tilt spread** = mean 1-month forward return of the
  top-20%-tilt names minus the bottom-20%, each day. Daily panels exist
  only for NDX; the SPX variant runs at the payload's weekly `spx_rel`
  cadence on the raw single-day print; IWM has no per-name panel.
- **Forward return** = each index's own proxy (QQQ/SPY/IWM) 21 sessions
  ahead (payload `r21`, %). No look-ahead in any outcome.
- **CI** = 95% moving-block bootstrap (21-day blocks); suppressed under 42
  scored days.
- NDX packed closes are split-adjusted but not dividend-adjusted; every
  name's compounded 21d close return is validated against its adjusted
  `r21` (0.98 gate; nothing dropped in the current build — worst name
  0.995, dividends only). SPX/IWM baskets use Yahoo adjusted closes.

## The regimes themselves (full-sample basis)

| Index | Regime | n | avg corr | dispersion | breadth | vol | own 1m fwd | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NDX | LowCorr  | 601 | 0.13 | 2.31% | 0.62 | 14.6 | **+0.33%** | [−0.9, +1.5] |
| NDX | MidCorr  | 799 | 0.27 | 1.95% | 0.60 | 17.9 | +2.08% | [+0.8, +3.3] |
| NDX | HighCorr | 601 | 0.48 | 2.14% | 0.48 | 31.8 | +2.47% | [+0.4, +4.5] |
| SPX | LowCorr  | 499 | 0.12 | 1.97% | 0.63 | 10.4 | **+0.32%** | [−0.5, +1.1] |
| SPX | MidCorr  | 666 | 0.25 | 1.65% | 0.60 | 13.4 | +1.47% | [+0.3, +2.6] |
| SPX | HighCorr | 499 | 0.47 | 1.85% | 0.50 | 26.7 | +2.14% | [+0.2, +4.0] |
| IWM | LowCorr  | 499 | 0.16 | 3.95% | 0.59 | 16.1 | **−0.24%** | [−2.0, +1.4] |
| IWM | MidCorr  | 666 | 0.26 | 3.13% | 0.58 | 20.8 | +0.81% | [−0.9, +2.4] |
| IWM | HighCorr | 499 | 0.39 | 3.71% | 0.46 | 32.7 | **+2.84%** | [+1.0, +4.8] |

The shape is universal: high-comovement tapes are panic tapes (vol up,
breadth down, everything falling together) and precede the *best* months —
mean reversion — while the low-correlation, dispersed tape precedes the
weakest. The monotone regime effect is strongest for IWM (−0.24% → +2.84%,
zCORR NW-t +4.4 before the vol control) and smallest after expanding
cutoffs (part of the effect is full-sample relabeling). Small-cap
dispersion runs structurally ~2× large-cap (3–4%/day vs ~2%) on a narrower
correlation range.

Longest episodes land where they should for all three: **HighCorr** = the
2022 bear, the COVID crash, (NDX) late 2018; **LowCorr** = Jul–Nov 2025,
Dec 2025–Mar 2026, and Apr–Aug 2026 — the current episode. **As of 18 Aug
2026 all three indices sit in LowCorr simultaneously** (NDX AVG_CORR 0.07 —
the 2.5th percentile of its whole history — with dispersion at 3.6%/day).

## Headline: where the DIX gradient lives — and where it doesn't

Own-proxy 1-month forward by comovement regime × DIX regime, **expanding
(no-look-ahead) basis**:

| LowCorr regime only | DIX Low | DIX Mid | DIX High |
|---|---:|---:|---:|
| **NDX** | −0.11% (n=120, hit 46%) | +0.87% (n=386) | **+2.76% (n=249, hit 82%, CI [+1.3, +4.2])** |
| **SPX** | +0.82% (n=93) | +0.87% (n=424) | +1.09% (n=224) |
| **IWM** | +0.29% (n=133) | −0.31% (n=288) | **−1.29% (n=138, hit 42%)** |

- **NDX**: in the dispersed tape, DIX discriminates hard — a ~2.9pp
  Low→High gradient with the only tight, positive CI in the row (same
  pattern on the full-sample basis, −0.71% → +1.45%). In mid/high-corr
  regimes the gradient is flat or inverted; in panic regimes low DIX
  (capitulation) precedes the biggest bounces (+6.22%, n=30 —
  anecdote-sized).
- **SPX does not inherit the NDX result**: the gradient is ~flat
  (+0.8→+1.1) on both bases. Whatever information the dark-flow level
  carries in dispersed tapes, it is an NDX-100 phenomenon, not a
  large-cap-index generic.
- **IWM inverts it**: high IWM DIX in a dispersed small-cap tape has been
  *bad* (−1.29%, hit 42%; entry-day version: 14 entries, −1.01%, hit 43%).
  This extends the cross-index study's finding #4 — "a high IWM DIX alone
  is a poor omen for small caps" — and locates it: the damage is done in
  the low-correlation regime.
- Interaction regressions (`r1m ~ zDIX + zCORR + zDIX·zCORR`, NW-21) agree
  in sign everywhere (interaction −0.4 to −0.7) but are not significant at
  5% for any index, and for SPX/IWM the zCORR level effect migrates to the
  realized-vol control (zRV t=+2.0/+2.4; corr↔vol 0.8). Treat the 3×3s as
  conditioning maps of when DIX has worked, not proven coefficients.

## Which group rallies? Per-name dark flow in the dispersed tape

Daily Q5-minus-Q1 tilt spread of 1-month forward returns, by regime
(full-sample basis):

| | Regime | n | spread | 95% CI | Q5 (high tilt) | Q1 (low tilt) |
|---|---|---:|---:|---|---:|---:|
| NDX (daily) | LowCorr  | 552 | **−1.59pp** | **[−2.6, −0.6]** | +0.64% | +2.23% |
| NDX (daily) | MidCorr  | 797 | +0.02pp | [−0.6, +0.6] | +2.15% | +2.14% |
| NDX (daily) | HighCorr | 593 | −0.07pp | [−0.8, +0.6] | +3.44% | +3.51% |
| SPX (weekly) | LowCorr  | 93 | −0.07pp | [−0.4, +0.2] | +0.44% | +0.52% |
| SPX (weekly) | MidCorr  | 129 | −0.11pp | [−0.4, +0.2] | +1.63% | +1.75% |
| SPX (weekly) | HighCorr | 95 | −0.69pp | [−1.2, −0.1] | +2.60% | +3.29% |

**In NDX low-correlation regimes, per-name dark flow does identify the
groups — with a negative sign.** Names whose dark ratio runs above their
own norm lag by ~1.6pp/month (the only CI excluding zero; −1.9pp on 2024+
test days); the group that rallies is the one *without* an elevated
dark-flow tilt. In mid/high-correlation tapes everything comoves and the
cross-section carries nothing. This matches the dashboard's long-standing
per-name finding (elevated name-specific dark ratio skews forward returns
negative) and sharpens it: **that signal is a low-correlation-regime
phenomenon.**

The SPX check runs at a weekly cadence on the noisier single-day print, so
it is a weaker instrument: it shows the same overall negative constant
(−0.28pp, t=−2.3) but its (small) regime concentration sits in HighCorr
rather than LowCorr. Read it as "the negative per-name signal exists in the
S&P too", not as a regime contradiction with equal evidence.

## Tape taxonomy — "all rally" vs "only a certain group rallies"

Trailing 21d index return sign × breadth tercile (NDX shown; SPX/IWM in
the script output are shaped the same):

| Tape (NDX) | n | avg corr | QQQ 1m fwd | 95% CI |
|---|---:|---:|---:|---|
| broad rally | 667 | 0.28 | +1.18% | [−0.2, +2.5] |
| mid rally | 561 | 0.21 | +1.44% | [+0.5, +2.4] |
| narrow rally | 74 | 0.25 | +2.48% | [+1.3, +3.7] |
| selective selloff | 2 | 0.39 | — | — |
| mid selloff | 96 | 0.28 | −0.14% | [−1.9, +1.8] |
| broad selloff | 601 | 0.39 | +2.68% | [+1.0, +4.5] |

- **Selective selloffs barely exist anywhere** (NDX 2 days, SPX 0, IWM 4
  in the shared window): when an index falls, breadth collapses and
  correlation spikes with it. Selectivity is a *rally-side* phenomenon.
- Narrow rallies have *not* been fragile in any index (NDX +2.48%, SPX
  +1.43%, IWM +2.34% — leadership persisting, not topping).
- DIX splits the rally tapes in NDX (broad rally: DIXLow +0.31% / DIXHigh
  +2.47%) and more weakly in SPX (+0.49% / +1.79%); in broad selloffs the
  *low*-DIX washouts bounce hardest in every index (NDX +3.40%, SPX
  +2.55%, IWM +3.22%).

## Regime-entry event study (expanding basis, 21-session cool-down)

| Entry condition | NDX | SPX | IWM |
|---|---:|---:|---:|
| enter LowCorr | +2.92% (18, hit 83%) | +1.71% (22, 73%) | +1.18% (27, 59%) |
| enter LowCorr & DIXHigh | **+3.17% (21, 81%)** | +0.84% (21, 67%) | **−1.01% (14, 43%)** |
| enter LowCorr & DIXLow | +1.41% (13, 58%) | +2.36% (13, 92%) | −0.47% (18, 44%) |
| enter HighCorr | +0.91% (17, 76%) | +0.73% (10, 60%) | +1.27% (16, 50%) |

The entry view repeats the per-index story: the low-corr + high-DIX setup
survives entry-day scoring only for NDX; for IWM the same setup is the
*worst* row. (Each cell is a couple dozen events — direction, not
precision.)

## Out-of-sample split (fit < 2024, evaluate 2024+; ~660 test days each)

- Test baselines: QQQ +1.96%, SPY +1.69%, IWM +1.64%.
- **NDX tilt spread holds OOS**: −1.94pp on LowCorr test days.
- **The NDX index-level DIX gradient does not stay monotone OOS** (LowCorr:
  DIXLow +1.80/DIXMid +0.21/DIXHigh +1.79 — High still beats Mid, but the
  Low leg stopped being bad). IWM's inversion partially reverses OOS on a
  thin DIXHigh leg (+5.07%, n=19) — small-cap cells get very sparse after
  2024 under train cutoffs.
- Train-fitted interaction models transfer modestly everywhere: OOS
  corr(pred, realized) = +0.18 (NDX), +0.39 (SPX), +0.28 (IWM) — but the
  SPX/IWM numbers leaning on sparse high-corr test cells; the cross-index
  two-factor model managed 0.00 on the same kind of test.

## Cross-index comovement agreement

Common window 6 Jan 2020 – 18 Aug 2026 (1,663 days), full-sample basis:

- corr(AVG_CORR): NDX↔SPX **+0.96** (same regime 84% of days), NDX↔IWM
  +0.81 (61%), SPX↔IWM +0.83 (62%). All three agree 54% of days — the
  large-cap gauges are nearly one gauge; small caps genuinely decouple.
- Forward 1m by number of indices in LowCorr:

| dispersed | n | NDX | SPX | IWM |
|---|---:|---:|---:|---:|
| 0 of 3 | 963 | +2.37% | +1.82% | +1.71% |
| 1 of 3 | 191 | +2.31% | +1.32% | +0.05% |
| 2 of 3 | 202 | +1.78% | +1.27% | +0.86% |
| 3 of 3 | 307 | **−0.70%** | **−0.24%** | +0.05% |

**Everything-dispersed is the weakest environment on record for all three
indices** — and it is the current one (all three in LowCorr since late
Apr 2026).

External cross-check: NDX AVG_CORR vs the GEX/dispersion barometer —
corr(level) **+0.95** against the realized top-50 SPX correlation and
**+0.73** against Cboe COR1M (+0.90 / +0.34 on 21d changes).

## What the pattern says

1. **Comovement is a real conditioning variable for the DIX — but the
   conditioning differs by index.** The unconditional "level carries
   nothing" result decomposes three ways: in low-correlation tapes the DIX
   level carries a ~3pp/month monotone gradient for NDX, roughly nothing
   for SPX, and the *opposite* sign for IWM. A single pooled rule would
   have averaged these away.
2. **In dispersed tapes, per-name dark flow tells you which group** (NDX):
   avoid the names with elevated dark-flow tilt (−1.6pp/month, CI excludes
   zero, holds OOS). This remains the study's most robust cell; the weekly
   SPX check sees the same negative constant.
3. **Selectivity is rally-side only, in every index**: selloffs are broad
   by construction (correlation spikes), so "only a certain group rallies"
   regimes are low-corr *up* tapes — and narrow rallies have continued,
   not died.
4. **The regime now** (Aug 2026): all three indices dispersed at once —
   historically the weakest forward environment (NDX −0.70%/month) — with
   the NDX at record-low correlation. Watch each index's own DIX zone
   through its own map: high NDX DIX would be constructive, high IWM DIX
   would not.
5. Interaction coefficients are only marginal (|t| ≈ 1–1.6), the corr
   level effect is entangled with vol for SPX/IWM, and several OOS legs
   are thin — treat the 3×3s as conditioning maps, not standalone
   triggers.

## Caveats

- **Survivorship.** All universes are *current* membership (NDX grid
  names; top-100 present-day IVV/IWM holdings — for IWM that is also a
  winners-by-weight tilt). Comovement gauges are fairly insensitive to
  membership; per-name results describe today's constituents.
- **Basket coverage.** The SPX basket covers 75% of index weight; the IWM
  basket only 22.6% (99 of ~2,000 names in the current build) — a
  behavioral proxy for the small-cap tape, not a replication of the index.
- **Overlapping windows.** ~2,000/1,660 sessions ≈ 95/79 independent
  months; block bootstrap and the entry studies are the honest lenses.
- **Corr ↔ vol.** AVG_CORR and each proxy's realized vol are ~0.8
  correlated; with a vol control neither is separately significant.
  "Low-corr regime" and "quiet tape" largely overlap in this sample.
- **Equal-weight gauges.** AVG_CORR/BREADTH weight the largest and
  smallest basket names equally.
- Tape/breadth cutoffs are full-sample (descriptive); the tape tables are
  not live signals as printed.
- The SPX tilt check runs on the payload's weekly-sampled `spx_rel` block
  (raw single-day print, ~330 rows) — a much weaker instrument than the
  NDX daily panel; IWM has no per-name dark-flow panel at all.
- The per-index 3×3 tables (with CIs) are written to
  `intra_index_regimes.csv`.
