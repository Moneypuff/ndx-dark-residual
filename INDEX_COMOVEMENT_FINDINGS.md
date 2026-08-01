# Cross-index DIX comovement → 1-month forward returns

When the three index dark-flow gauges — **NDX-100 DIX**, **S&P 500 DIX** and
**Russell 2000 / IWM DIX** — line up vs. pull apart, what do the indices do over
the following month? Study of the **5-day moving average** of each dollar-DIX
against each index's own 1-month (21 trading day) forward return, **6 Jan 2020 –
31 Jul 2026, 1,651 common sessions**.

Reproduce:
```
python index_comovement_study.py --csv index_comovement_regimes.csv   # text/CSV study
python build_comovement.py --docs-out docs/comovement.html            # interactive tab
```
Data is read from the payload embedded in `docs/index.html` (no live FINRA
re-fetch; both the plain and compressed payload encodings are supported); pass
`--html`/`--payload` to point at another build. The interactive version is
published as the **Comovement** tab in the dashboard (`docs/comovement.html`),
rebuilt nightly by the refresh workflow.

## Definitions

- **DIX** = dollar-weighted `Σ($ short volume) / Σ($ off-exchange volume)` across
  an index's constituents each day (SqueezeMetrics' construction;
  `compute_dollar_dix` in `ndx_dark_residual.py`).
- **DIX5** = 5-day moving average of DIX (min 3 obs).
- **Decile → regime**, per index: **Low** = deciles 1–3, **Mid** = 4–7,
  **High** = 8–10. Reported on two bases: **full-sample** deciles (the original
  construction; mild look-ahead) and **expanding-window** deciles (each day
  ranked only against its own past, min 250 obs — what a live trader could have
  known).
- **Forward return** = each index's own price proxy (QQQ / SPY / IWM), 21
  sessions ahead, in percent. No look-ahead in the outcome.
- **CI** = 95% moving-block bootstrap (21-day blocks) on the regime's mean,
  respecting the overlap autocorrelation; suppressed for regimes with < 42
  scored days, where the block resample degenerates.

## Baseline (all 1,651 common days)

| Index | mean 1m | 95% CI | median | hit% |
|---|---:|---|---:|---:|
| NDX (QQQ) | +1.76% | [+0.6, +2.9] | +2.35% | 65% |
| SPX (SPY) | +1.33% | [+0.4, +2.2] | +1.94% | 69% |
| IWM       | +1.11% | [−0.2, +2.3] | +1.42% | 59% |

Under expanding (no-look-ahead) deciles the scored sample is 1,402 days and the
baseline is a touch lower (NDX +1.44%, SPX +1.24%, IWM +0.79%).

## The requested divergence — SPX & NDX DIX **Low**, IWM DIX **High**

**Full-sample deciles: 94 days (83 with a full forward window). All three
indices beat their baseline** — but the block-bootstrap CIs are wide, and two
sharper tests below take most of the shine off.

| Index | mean 1m | 95% CI | median | hit% |
|---|---:|---|---:|---:|
| NDX (QQQ) | +2.73% | [+0.2, +5.0] | +3.82% | 71% |
| SPX (SPY) | +2.12% | [+0.3, +3.8] | +2.69% | 72% |
| IWM       | +2.49% | [−1.2, +6.6] | +2.67% | 64% |

Two honesty checks:

1. **Expanding cutoffs collapse the sample.** Ranked only against its own past,
   this regime appears on just **17 live days (6 scored)** — the 94-day count is
   mostly an artifact of full-sample deciles re-labeling history. (Those few
   live occurrences were strongly positive, but n = 6 is anecdote, not
   evidence.)
2. **It fails the entry test.** Scoring only the **first day** the regime forms
   (12 entries, 21-session cool-down): NDX **+0.33%**, SPX **+0.44%**, IWM
   **−0.94%** (42% hit) — *below* baseline. The attractive every-day average
   was earned deep inside long episodes, not when the setup appears — i.e. it
   describes environments, not a trade trigger.

## Divergence vs. level: the two-factor regression

Replacing the 27 bins with two continuous factors —
`LEVEL` = mean DIX5 z-score of the three gauges, and
`SPREAD` = (z_NDX + z_SPX)/2 − z_IWM (large-cap-minus-small-cap dark flow) —
and regressing each index's 1-month forward return on both (Newey-West, 21
lags):

| Index | LEVEL β | (t) | SPREAD β | (t) | R² |
|---|---:|---:|---:|---:|---:|
| NDX | −0.02 | −0.03 | **+0.66** | +1.76 (p=0.08) | 1.5% |
| SPX | −0.19 | −0.41 | +0.32 | +1.12 | 0.5% |
| IWM | +0.03 | +0.05 | −0.07 | −0.16 | 0.0% |

This directly confirms half of the original conclusion and disciplines the
other half: **level carries nothing** (β ≈ 0 everywhere), and the
**large-vs-small divergence tilts only the large-cap indices** — about +0.7pp
of NDX forward return per 1z of spread, marginally significant — while IWM
itself is *not* reliably helped or hurt by the spread once you leave the bins.

## Regime-entry event study (first day a regime forms, 21-session cool-down)

| Regime | entries | NDX | SPX | IWM |
|---|---:|---:|---:|---:|
| N=High, S=Mid, I=Low | 7 | **+6.80%** (hit 100%) | +4.93% (hit 100%) | +4.03% |
| N=Mid, S=Mid, I=Low | 21 | **+4.62%** (hit 86%) | +3.54% (hit 81%) | +2.87% |
| all High | 20 | +2.66% | +1.78% | +2.05% |
| N=High, S=High, I=Low | 14 | +1.49% | +1.27% | +2.11% |
| all Low | 15 | +1.70% | +0.42% | −1.20% |
| N=Low, S=Low, I=High (requested) | 12 | +0.33% | +0.44% | −0.94% |

**The one family that survives event-style counting is "large-cap DIX firm
while small-cap DIX is Low"** — 28 entries across its two variants, NDX
+4.6–6.8% with 86–100% hit rates. The requested LLH divergence does not.

## Sector dark-flow gauge (defensive minus cyclical)

Defensive (XLP/XLU/XLV) minus cyclical (XLI/XLF/XLE/XLB) sector-DIX z-spread,
from the dashboard's reconstructed sector series (1,653 days):

| Outcome | slope per 1z (NW t) | defensive-tilt vs cyclical-tilt tercile |
|---|---:|---:|
| NDX | −0.42 (t=−0.46) | +2.00% vs +1.93% |
| SPX | −0.04 (t=−0.07) | +1.66% vs +1.41% |
| IWM | +0.10 (t=+0.09) | +1.69% vs +0.83% |

**A null result**: where dark flow rotates between defensives and cyclicals
carries no measurable information about the next month at the index level.
Reported so it doesn't have to be re-discovered.

## Out-of-sample split (fit < 2024, evaluate 2024+)

- Test-window baseline (647 days): NDX +1.98%, SPX +1.67%, IWM +1.62%.
- The requested divergence under train-fitted cutoffs fired on only **4 test
  days** (all strongly positive — too few to score).
- The two-factor model fitted pre-2024 **does not generalize**: OOS
  correlation of predicted vs realized returns is +0.00 (NDX), −0.13 (SPX),
  −0.14 (IWM). Days the model liked beat the test average by +0.38pp for NDX
  and *lagged* it for SPX/IWM.

## What the pattern says (revised)

1. **Level carries nothing** — confirmed by bins, factors, and entries alike.
2. **The tradeable-looking signal is one-sided:** large-cap DIX firm + small-cap
   DIX Low → strong NDX/SPX months, and it survives entry-day scoring (28
   entries, 86–100% NDX hit). It remains a small-n, one-cycle observation.
3. **The requested LLH divergence describes an environment, not a trigger** —
   its every-day average is positive, but at formation it is baseline-or-worse,
   and under real-time cutoffs it has barely ever existed.
4. **A high IWM DIX alone remains a poor omen for small caps** (N=Mid,S=Mid,
   I=High: IWM −0.27%, hit 46%), and the worst cell (N=Mid,S=Low,I=Low: IWM
   −6.81%, CI [−11.8, −1.7]) is one of the few that stays significant under
   block bootstrap.
5. **Nothing here generalizes cleanly out of sample** — treat every cell,
   including the good ones, as descriptive of 2020–2023 until 2024+ accumulates
   more independent months.

## Caveats

- **Overlapping windows.** 1,651 sessions ≈ only ~79 independent months; the
  block-bootstrap CIs and the entry study are the honest lenses. Regimes of
  30–90 days represent a handful of distinct episodes.
- **Sample window.** Starts Jan 2020; COVID crash-recovery and the 2022
  drawdown sit inside it.
- **IWM DIX is reconstructed** from iShares' Russell 2000 holdings and spans a
  wider range than the NDX/SPX gauges; deciles are internally consistent but
  raw levels aren't comparable.
- The full 27-regime table (with CIs) is written to
  `index_comovement_regimes.csv`.
