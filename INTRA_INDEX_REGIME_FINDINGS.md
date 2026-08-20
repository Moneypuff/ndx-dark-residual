# Intra-index comovement regimes × DIX → 1-month forward returns

Divide each index's history into comovement regimes — at times all
constituents rally together and correlation is high; in dispersed periods
only a certain group rallies — and study how each index's DIX reads inside
each regime, at the index level and per name. **NDX: 30 Aug 2018 – 18 Aug
2026, 2,001 sessions, 102 grid names (point-in-time panel: 180 of 182
ever-members). SPX/IWM: 6 Jan 2020 →, ~1,660 sessions, top-99 iShares
baskets.**

This is the post-review revision: the study was refereed for era
confounds, episode-level uncertainty, survivorship, timing and mechanism,
and every number below is produced under the corrected standards. Earlier
versions of this document quoted larger effects; where a headline shrank,
that is the correction working, not a data change.

Reproduce:
```
python intra_index_regime_study.py --transitions --csv intra_index_regimes.csv
python intra_index_regime_study.py --indices ndx --point-in-time   # + PIT tilt panel
```

## Inference standards (what every number below has survived)

- **Rolling-basis zones**: Low/Mid/High are 30/40/30 splits against a
  trailing 504-session window — live-knowable *and* drift-proof. (Expanding
  percentiles on the drifting dollar-DIX had turned zones into calendar
  labels: the old headline cell compared mostly-2020-23 DIXHigh days
  against mostly-2025-26 DIXLow days. Expanding and full-sample tables
  remain in the script output as comparisons.)
- **Episode-cluster CIs** (`epCI`): regimes arrive in multi-week episodes;
  CIs resample whole episodes, not 21-day blocks. Day counts are printed
  with episode counts everywhere.
- **Era-adjusted means** (`ex`): each cell also reported net of its own
  calendar-year baseline (diagnostic — the demeaning uses full-year info).
- **Print gate**: no mean is quoted below 42 scored days AND 5 episodes
  (10 events for entry studies).
- **Timing**: FINRA publishes day t's file after the close; headline cells
  are re-run with the DIX signal lagged one session.
- **Two PRIMARY cells** (pre-specified): the NDX LowCorr×DIXHigh forward
  return, and the NDX LowCorr tilt spread. Everything else is descriptive;
  with ~150 printed cells, isolated "significant" cells are expected by
  chance.

## What the gauge measures (mechanism, stated honestly)

FINRA's daily file records the short-marked share of **all off-exchange
volume** — today mostly wholesaler internalization of retail flow, and
flow rather than positioning. "High DIX = dark accumulation" is one
interpretation (SqueezeMetrics'); a high per-name ratio can equally read
as crowded retail intensity. This study treats the ratios as signals to be
conditioned and tested. The evidence below (index-level positive, per-name
negative, per-name effect concentrated in high off-exchange-share names)
sits more comfortably with the crowding reading at the name level.

## The regimes themselves (rolling basis)

| Index | Regime | n days/eps | breadth | vol | own 1m fwd | ex (era-adj) | epCI |
|---|---|---:|---:|---:|---:|---:|---|
| NDX | LowCorr | 720/30 | 0.63 | 14.9 | +1.46% | **−0.91** | [+0.1, +2.5] |
| NDX | HighCorr | 400/24 | 0.44 | 34.7 | +2.70% | **+2.22** | [+0.2, +6.0] |
| SPX | LowCorr | 675/39 | 0.63 | 10.7 | +0.82% | −0.88 | [−0.1, +1.5] |
| SPX | HighCorr | 310/15 | 0.46 | 24.2 | +2.01% | +1.84 | [−0.2, +4.4] |
| IWM | LowCorr | 529/45 | 0.58 | 16.8 | −0.65% | −1.84 | [−2.4, +1.3] |
| IWM | HighCorr | 310/27 | 0.44 | 28.5 | +2.15% | +1.77 | [+0.4, +4.5] |

The ordering — dispersed tapes precede below-era months, panic tapes
above-era months — survives era adjustment in all three indices and
strengthens excluding COVID/2022. It is entangled with volatility
(corr(zCORR, zRV) ≈ 0.8); the script's vol-parallel panel shows a vol-only
version reproduces part but not all of it (the NDX DIX gradient inside
low-VOL terciles is +1.0→+2.5 vs +(−0.2)→+2.9 inside low-CORR terciles,
and in the nested regression neither zRV nor zCORR is separately
significant).

## PRIMARY 1 — the NDX DIX gradient in dispersed tapes (corrected)

QQQ 1-month forward inside the NDX LowCorr regime, rolling basis:

| | DIX Low | DIX Mid | DIX High |
|---|---:|---:|---:|
| raw mean | −0.23% (n=269/35ep) | +2.16% | **+2.90% (n=184/31ep, hit 82%)** |
| era-adjusted | **−2.35** | −0.19 | **+0.18** |
| epCI | [−1.4, +1.2] | [+1.1, +3.2] | [+0.9, +4.8] |
| lag-1 signal | −0.34% | +2.19% | +2.97% |

What survived and what didn't:

- **The gradient survived** (~2.5–3.2pp Low→High, monotone, on 26–35
  episodes per leg with balanced year mixes, intact at lag-1 timing;
  leave-one-year-out range of the High cell [+1.85, +3.69]).
- **The level claim did not**: the High cell's era-adjusted mean is
  +0.18pp — the celebrated +2.9%/82% is mostly the unconditional strength
  of the years the cell lives in. The informative cell is actually the
  **DIX-Low leg: −2.35pp below its own era** — in dispersed tapes, *low*
  dark-flow share has been the warning, more than high share being a
  buy signal.
- **Risk of holding the High cell**: fwd p10/p90 [−3.0, +7.8]; max adverse
  excursion inside the 21-session hold: median −1.6%, worst −13.6%.
- **Not NDX-specific.** Crossed inputs on the common 2020+ window: NDX
  gauge+DIX→SPY reproduces ~59% of the gradient (+0.02→+1.94) and SPX
  gauge+DIX→QQQ carries +0.08→+2.03, while SPX gauge+DIX→SPY is flat
  (+0.50→+1.09). The contrast the earlier draft called "an NDX-100
  phenomenon" decomposes into: **the outcome leg (QQQ responds harder
  than SPY) plus a smaller DIX-gauge leg** — both large-cap dark-flow
  gauges carry it. (The SPX-ex-NDX DIX leg is packed by the pipeline as
  `spx.dix_ex` and will populate on the next nightly build.)
- **OOS (2024+) is mixed**: under live rolling zones the DIXHigh test cell
  has only 37 days (gated); under train-frozen cutoffs High beats Mid
  (+1.79 vs +0.21) but Low isn't bad (+1.80, n=57/11ep). The train-fitted
  interaction transfers weakly (OOS corr +0.18, NW t=+1.62).

Entry-day version (rolling basis) — and one more casualty: at *formation*
the DIX split does not differentiate. Enter LowCorr & DIXHigh: 17 entries,
+2.63%, hit 82% (lag-1: 16 entries, +2.77%); enter LowCorr & DIXLow: 20
entries, +2.57%, hit 80%; enter LowCorr alone: 21 entries, +3.05%. The
old expanding-basis entry gradient was another zone-drift artifact. The
gradient is an *environment* description (it accrues inside episodes,
via the DIX-Low leg), not an entry trigger.

## PRIMARY 2 — the per-name tilt spread (corrected: point-in-time panel)

Q5-minus-Q1 one-month spread on the name-specific tilt (5d-MA raw dark
ratio minus own expanding mean), NDX LowCorr days:

| Panel / variant | spread | epCI | pre-2024 / 2024+ |
|---|---:|---|---|
| current members (survivorship-censored) | −1.07pp | [−2.3, −0.0] | — |
| **point-in-time (180 ever-members)** | **−0.60pp** | [−1.3, +0.1] | −0.37 / −0.80 |
| PIT, momentum-neutral | −0.64pp | [−1.4, +0.1] | −0.46 / −0.81 |
| PIT, beta-neutral | −0.57pp | [−1.2, +0.1] | −0.28 / −0.83 |
| PIT, momentum+beta-neutral (primary) | −0.65pp | [−1.3, +0.1] | −0.32 / −0.94 |
| PIT, sector-neutral | −0.48pp | [−1.0, +0.0] | −0.28 / −0.66 |
| PIT, high off-exch-share half | −0.93pp | [−1.9, +0.1] | — |
| PIT, low off-exch-share half | −0.42pp | [−1.0, +0.3] | — |

Regrading of what was previously called "the study's most robust cell":

- **Survivorship was nearly half the magnitude** (−1.07 → −0.60 once
  departed members are included with membership-masked histories).
- **The sign is consistent** (negative in 5 of 7 years, LOYO range
  [−1.57, −0.71] on the current-members panel) and is **not** a momentum,
  beta or sector bet — neutralizations barely move it. But the
  episode-cluster CI now touches zero and the effect is 2× stronger
  post-2024, so "regime-specific alpha" and "post-2023 era effect" remain
  unidentified.
- **It reads as a retail-crowding signal**: twice the size among names
  with a high off-exchange share of volume (−0.93 vs −0.42).
- The buckets are quasi-static (Q5 lived in CTSH/KHC/NXPI/GILD/CHTR; Q1 in
  AMZN/GOOGL/GOOG/NFLX/MU; 21d rank autocorrelation +0.29) — closer to one
  persistent tilt than 720 independent bets, which is exactly what the
  episode CI prices in.
- Practical form: an **avoid-screen** (don't hold high-tilt names in
  dispersed tapes), not a long/short spread — the Q5 leg still went *up*
  +1.4%/month; there is no decline to short, only a lag to avoid, and a
  40-name daily-rebalanced spread would spend most of 0.6pp on costs.

## SPX and IWM inside their own dispersed regimes

- **SPX: flat.** LowCorr row +0.50 → +1.03 → +1.09 (era-adjusted all
  negative: −1.24 → −0.55); flagship cell ex −0.55, epCI [−0.5, +2.5]. The
  weekly `spx_rel` tilt check is ~0 in LowCorr (−0.08, [−0.3, +0.2]) — a
  weak instrument (weekly raw prints), reported for completeness, no
  longer cited as corroboration of anything.
- **IWM: the "inversion" is demoted to suggestive.** LowCorr×DIXHigh is
  −0.91% (ex −2.29, hit 43%) but epCI [−2.9, +1.0] spans zero, the whole
  LowCorr row is negative regardless of DIX (−0.20/−0.86/−0.91 — the
  regime, not the DIX, carries IWM's bad news), and the cell partially
  reverses on a thin 2024+ leg. The doubly-reconstructed IWM instrument
  (current-holdings DIX, 22.6%-coverage winners basket) caps how much this
  row can ever say.

## Do dispersed regimes announce their own end? (transitions, corrected)

Episode-clustered hazard models on mature (age ≥ 21d) LowCorr days, every
spec controlling for distance-to-boundary (the rolling percentile of the
gauge — a driftless random walk "predicts" its own exits through proximity
alone, and the test suite pins that this control kills such a signal):

- **The gauge's own 5-day slope survives for the large caps**: +0.15/sd
  (t=+4.0, 13 episodes) NDX, +0.16/sd (t=+2.7, 12 eps) SPX — correlation
  turning up inside a mature dispersed tape raises the exit hazard beyond
  boundary mechanics. For IWM it flips (−0.17/sd) with the boundary term
  dominant (+0.35/sd, t=+5.5) — small-cap exits are mostly proximity.
- **Deteriorating breadth helps at the margin** (d21_breadth −0.14/sd,
  t=−2.4 NDX; −0.11, t=−1.8 SPX).
- **The "rising DIX precedes the exit" precursor died** under honest
  construction (d21_dix t = −0.6 NDX / +1.0 SPX / +0.2 IWM).
- **"Last one dispersed → exits 100%" died**: pooled at the episode level
  it is 42/54 (Wilson [0.65, 0.87]) vs 82/114 (Wilson [0.63, 0.79]) for
  *all* LowCorr episode starts — indistinguishable.
- **The durable null stands**: entries into HighCorr are jumps, not creeps
  (~35% of the whole 40-day gauge rise lands in the final 5 sessions;
  12–22 entries per index). Correlation spikes are not forecastable from
  the comovement gauges, and Cboe COR1M gives no usable early warning
  either.
- Episode-level counts are small everywhere (9–13 mature episodes per
  index): treat even the surviving slopes as one-cycle evidence.

## Cross-index agreement (rolling basis, 1,414 common days)

NDX↔SPX gauges are nearly one gauge (corr +0.96, same regime 83% of days);
IWM decouples (+0.81/+0.83, ~61%). Forward 1m by number of indices in
LowCorr:

| dispersed | n days/eps | NDX (ex) | SPX (ex) | IWM (ex) |
|---|---:|---:|---:|---:|
| 0 of 3 | 572/35 | +2.15 (+1.66) | +1.99 (+1.31) | +1.90 (+1.51) |
| 3 of 3 | 387/38 | +0.12 (−2.11), epCI [−1.4,+2.0] | +0.17 (−1.55), [−0.9,+1.4] | −0.31 (−1.65), [−2.1,+1.5] |

The earlier draft's "weakest environment on record" is **not** a
statistical claim — every 3-of-3 epCI includes zero. What holds is the
consistent within-year drag (era-adjusted −1.6 to −2.1pp across all three
indices, sign right in 6 of 6 years for NDX) and the ordering vs 0-of-3.
All three indices have been jointly dispersed since late April 2026;
current-regime reads carry an as-of date and are computed only on complete
sessions.

## Tape structure (kept for its structural facts only)

Selective selloffs barely exist (NDX 2 days, SPX 0, IWM 4): when an index
falls, breadth collapses and correlation spikes — selectivity is a
rally-side phenomenon (partly by construction: equal-weight breadth vs a
cap-weighted index leg). Narrow rallies have not been fragile in any index.
The tape×DIX sub-splits duplicate the 3×3s and are no longer quoted.

## What the pattern says (post-review grading)

1. **The regime axis is real but era-entangled**: dispersed tapes run
   ~1–2pp/month below their own year's baseline, panic tapes ~2pp above,
   in all three indices — with vol as an inseparable co-driver at this
   sample size.
2. **Dark flow inside dispersed tapes**: the surviving index-level fact is
   the *gradient* — mostly the DIX-Low leg sitting ~2.4pp below era — and
   it belongs to both large-cap gauges, expressed strongest in QQQ. The
   +2.9%/82% cell as previously quoted was era-inflated.
3. **The per-name tilt is a modest, style-clean, retail-flavored drag**
   (−0.6pp PIT, CI touching zero): defensible as an avoid-screen in
   dispersed tapes, no longer as "the most robust cell".
4. **Regime endings**: watch the gauge's own short-term slope and breadth
   deterioration; ignore DIX changes and "last one dispersed" as timing
   signals; nothing forecasts the correlation spike itself.
5. Everything rests on ~10–40 episodes per claim from one macro cycle.
   The pre-registered forward-scoring log (see `frozen_rules.json` once
   Phase 10 of the improvement plan lands) is the only thing that will
   settle points 2–4.

## Caveats

- One macro cycle (2018–2026); 9–59 episodes behind any cell; era-adjusted
  columns use within-year future information (diagnostic only).
- Corr ↔ vol ≈ 0.8: "dispersed" and "quiet" largely overlap in-sample.
- IWM instruments are doubly reconstructed (current-holdings DIX,
  22.6%-coverage winners basket); SPX basket covers 75% of index weight.
- The PIT panel drops 2 of 182 ever-members (HONA, SPCX — no history) and
  keeps both Alphabet share classes (mild Q1 double-count); membership
  windows from `data/ndx_membership.csv` (public reconstitution history,
  hand-reviewed; regenerate with `fetch_ndx_membership.py`).
- NDX packed closes are split-adjusted, dividend-unadjusted (validated
  ≥0.995 vs adjusted r21); SPX/IWM baskets use Yahoo adjusted closes.
- Equal-weight gauges vs cap-weighted outcomes; the tape taxonomy's
  selloff asymmetry is partly definitional.
- Per-index 3×3 tables (all bases, with both CI families, year mixes and
  gates) are in `intra_index_regimes.csv`; the full report adds the
  expanding/full-sample comparisons, vol parallels, lag-1 grids and
  transition models.
