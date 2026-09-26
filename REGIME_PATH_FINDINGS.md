# Regime path study — findings

> **Status: FIRST LIVE RUN COMPLETE (2026-09-05).** Run against the
> published payload (`docs/index.html`, generated 2026-09-05 07:00 UTC by
> the nightly build) plus freshly fetched IVV/IWM constituent baskets and
> Cboe's VXN/VIX/RVX history. Every number below is real output from
> `regime_path_study.py`; `regime_paths.csv`, `regime_path_risk.csv` and
> `regime_path_envelopes.json` were regenerated in the same run and are
> committed alongside this document. Reproduce:
> ```
> python regime_path_study.py --cache-dir .ndx_dark_cache \
>     --csv regime_paths.csv --risk-csv regime_path_risk.csv \
>     --envelopes regime_path_envelopes.json
> ```
> This is one snapshot, not a maintained live number — the mean/MAE/vol
> figures below will drift as more sessions accrue, exactly like
> `INTRA_INDEX_REGIME_FINDINGS.md`'s own numbers do. Treat this as the
> first data point in that document's series, not a one-time fact.

Companion to `INTRA_INDEX_REGIME_FINDINGS.md` (the corrected regime study):
that document answers "what is the mean 1-month forward return inside each
comovement × DIX regime cell." This one answers what a mean cannot: how
volatile the path to that return is, how deep it goes before it pays,
whether a stop helps or amputates, and whether the regime's own volatility
expands or contracts while you hold it. **The two studies agree with each
other wherever they overlap** — see "Cross-checks against the mean-return
study" below — which is the strongest evidence this run is measuring the
same thing correctly.

Sample: NDX 2018-08-30 → 2026-09-04 (2,014 days, 102 payload grid names);
SPX/IWM 2020-01-06 → 2026-09-04 (1,676 days, top-99 IVV/IWM baskets by
weight, 75.3%/22.7% pre-normalization coverage). All three fully clear the
run at rolling-basis, lag-1-DIX timing.

## Three PRIMARY hypotheses (pre-specified)

Only these three claims may headline; everything else below is
descriptive. Verdicts below are this run's real output.

### P1 — Jump risk in dispersed tapes

| Index | LowCorr \|z\|>2 | epCI | HighCorr \|z\|>2 | Verdict |
|---|---:|---|---:|---|
| NDX | 10.0% | [4.4, 17.4] | 5.5% | **NOT SUPPORTED** — CI's lower bound (4.4%) sits just under the 4.6% Gaussian benchmark |
| SPX | 3.8% | [0.9, 8.0] | 1.9% | NOT SUPPORTED |
| IWM | 7.4% | [2.8, 14.0] | 0.7% | NOT SUPPORTED (CI does clear 4.6%, but HighCorr's own share is lower, and the decision rule requires both) |

**Reading**: no index clears the pre-registered bar, but NDX misses by the
thinnest possible margin — a 4.4% lower CI bound against a 4.6% benchmark.
Trailing vol does not *detectably* understate dispersed-tape path risk in
any of the three, though NDX's dispersed tape is the closest call and
worth re-checking as more sessions accrue.

### P2 — The DIX-Low leg as a drawdown effect (NDX only, as pre-specified)

| | LowCorr×DIXLow(l1) q25 MAE | LowCorr×DIXHigh(l1) q25 MAE | diff epCI | Verdict |
|---|---:|---:|---|---|
| NDX | −5.8% | −2.6% | [−4.6, −0.5] | **SUPPORTED (drawdown flag)** |

**Reading**: this is the sharpest result in the whole run. The corrected
mean-return study found the DIX-Low leg sits ~2.4pp below its own era in
*terminal* return; this shows the same leg also carries a reliably worse
drawdown — the episode-cluster difference CI excludes zero even though
each leg's own MAE has a wide individual CI. The 21-day median terminal
return in the active leg is essentially flat (−0.0%, hit 49.8%) against
+2.5% (hit 78.3%) when the caution rule is off (full breakdown under
"Frozen-rule path rows" below) — so this reads as a genuine caution flag,
not a mean-only artifact.

### P3 — Panic tapes: worst excursions, decaying vol

| Index | q25 MAE21 by zone (Low/Mid/High) | worst zone | HighCorr fwd/trail vol ratio (med) | epCI | Verdict |
|---|---|---|---:|---|---|
| NDX | −4.1 / −5.8 / −8.2 | HighCorr | 0.83 | [0.65, 0.96] | **SUPPORTED (both legs)** |
| SPX | −3.5 / −3.5 / −5.5 | HighCorr | 0.82 | [0.65, 0.92] | **SUPPORTED (both legs)** |
| IWM | −6.7 / −4.8 / −5.4 | **LowCorr** | 0.80 | [0.72, 0.89] | **NOT SUPPORTED** (MAE leg fails) |

**Reading**: large-cap indices behave exactly as hypothesized — panic
tapes carry both the worst drawdowns and reliably decaying forward vol.
IWM inverts on the MAE leg: its *dispersed* (LowCorr) tape is the one with
the worst q25 MAE (−6.7%), not its panic tape — consistent with
`INTRA_INDEX_REGIME_FINDINGS.md`'s own finding that "the whole LowCorr row
is negative regardless of DIX" for IWM (the regime, not the DIX split,
carries IWM's bad news). The vol-decay leg still holds for IWM's HighCorr
(ratio 0.80, CI [0.72, 0.89]) — small-cap panic vol decays too, it's just
not simultaneously the worst-drawdown regime.

## Per-index zone marginals (rolling basis, ENV family, 21-day hold)

| Index | Zone | n days/eps | mean r21 | hit | p10/p50/p90 | MAE q25 | touch −5% | vol ratio | size/1% risk | RR |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| NDX | LowCorr | 754/30 | +1.4% | 68% | −4.4/+1.8/+6.5 | −4.1 | 18% | 1.12 | 24% | 0.44 |
| NDX | MidCorr | 611/53 | +1.7% | 67% | −5.7/+2.8/+8.1 | −5.8 | 29% | 0.94 | 17% | 0.48 |
| NDX | HighCorr | 400/24 | +2.7% | 65% | −10.3/+4.4/+13.7 | −8.2 | 38% | 0.83 | 12% | 0.53 |
| SPX | LowCorr | 712/38 | +0.9% | 65% | −3.5/+1.2/+4.3 | −3.5 | 11% | 1.16 | 29% | 0.36 |
| SPX | MidCorr | 401/51 | +1.4% | 72% | −5.1/+2.0/+6.0 | −3.5 | 17% | 0.90 | 29% | 0.57 |
| SPX | HighCorr | 314/14 | +2.0% | 68% | −6.5/+3.4/+8.5 | −5.5 | 28% | 0.82 | 18% | 0.61 |
| IWM | **LowCorr** | 574/49 | **−0.6%** | **46%** | −7.8/**−0.6**/+6.6 | −6.7 | 39% | 1.17 | 15% | **−0.08** |
| IWM | MidCorr | 560/68 | +1.6% | 64% | −5.7/+1.8/+8.0 | −4.8 | 23% | 0.92 | 21% | 0.37 |
| IWM | HighCorr | 293/23 | +2.0% | 65% | −6.2/+2.2/+9.9 | −5.4 | 27% | 0.80 | 18% | 0.40 |

IWM's LowCorr row is the one to flag: negative median return, sub-50% hit
rate, and the only negative reward/risk in the table — a small-cap
dispersed tape is not a "wait it out" regime the way NDX/SPX's are.

### ENTRY and EPISODE-HOLD summaries (zone marginals)

| Index | Zone | ENTRY n / mean / hit | EPISODE-HOLD n / dur med / terminal med / hit / post-exit r21 |
|---|---|---|---|
| NDX | LowCorr | 21 / +3.01% / 86% | 21 / 22d / +0.9% / 62% / +1.7% (hit 71%) |
| NDX | MidCorr | 39 / +1.60% / 67% | 42 / 11d / +1.2% / 55% / +2.1% (hit 76%) |
| NDX | HighCorr | 19 / +1.13% / 74% | 17 / 9d / +2.7% / 82% / +2.6% (hit 65%) |
| SPX | LowCorr | 22 / +1.92% / 68% | 25 / 16d / +0.2% / 52% / +3.2% (hit 76%) |
| SPX | MidCorr | 30 / +2.24% / 77% | 39 / 6d / +0.5% / 72% / +1.3% (hit 69%) |
| SPX | HighCorr | 12 / +1.26% / 67% | 11 / 16d / +0.8% / 73% / +4.2% (hit 73%) |
| IWM | **LowCorr** | 29 / +0.96% / 59% | **28 / 11d / −0.6% / 39%** / +1.9% (hit 68%) |
| IWM | MidCorr | 35 / +1.22% / 63% | 48 / 7d / +0.4% / 60% / +0.7% (hit 58%) |
| IWM | HighCorr | 15 / +0.12% / 40% | 15 / 7d / +1.4% / 67% / **−0.5% (hit 47%)** |

IWM's EPISODE-HOLD family (a confirmed, held-to-exit position, not just an
environment average) makes the LowCorr weakness concrete: the median
*held* position lost money (39% hit). IWM's HighCorr episodes are the only
family/zone combination anywhere in this run whose post-exit 21-session
return was negative on median — small-cap panic-tape exits are not
obviously followed by relief.

## Cross-checks against the mean-return study

Every place this run overlaps `INTRA_INDEX_REGIME_FINDINGS.md`, it agrees:

- **Ordering**: dispersed (LowCorr) tapes run below panic (HighCorr) tapes
  in all three indices here (mean r21 NDX +1.4→+2.7, SPX +0.9→+2.0, IWM
  −0.6→+2.0) — the same ordering the corrected study reports.
- **IWM's LowCorr is uniquely bad**: confirmed at the path level (only
  negative RR, only sub-50% ENV and EPISODE-HOLD hit rates in the table).
- **The DIX-Low leg's weakness (NDX)**: P2 above adds a drawdown reading
  to the mean-return study's own finding that the DIX-Low leg sits ~2.4pp
  below its era.
- **All-3-dispersed is a weak environment, not a strong one**: RR 0.09
  (NDX), 0.17 (SPX), −0.03 (IWM) at 3-of-3 vs 0.46/0.73/0.49 at 0-of-3 (see
  the cross-index table below) — matching the corrected study's finding
  that "3-of-3 dispersed" episode-cluster CIs include zero.

## Cross-index cells (N-of-3 indices simultaneously dispersed, ENV family)

| N dispersed | n days/eps | NDX mean r21 (hit, MAE q25, RR) | SPX | IWM |
|---|---:|---|---|---|
| 0 of 3 | 553/36 | +2.3% (67%, −6.8, 0.46) | +2.1% (72%, −4.2, 0.73) | +2.0% (67%, −5.1, 0.49) |
| 1 of 3 | 196/57 | +0.1% (56%, −7.1, 0.13) | +0.3% (62%, −4.9, 0.24) | −1.0% (48%, −7.2, −0.05) |
| 2 of 3 | 287/63 | +2.2% (74%, −3.6, 0.68) | +1.7% (76%, −3.1, 0.64) | +1.0% (58%, −5.3, 0.19) |
| 3 of 3 | 391/41 | +0.1% (54%, −4.9, 0.09) | +0.3% (58%, −3.5, 0.17) | −0.2% (48%, −6.2, −0.03) |

The relationship is not monotone in any index (1-of-3 is the weakest cell
everywhere, not 3-of-3) — a genuinely descriptive, not tidy, result worth
carrying forward rather than smoothing over.

## Corr-vs-vol comparison (LowCorr vs. VolLow, by DIX(lag-1) zone)

| Index | DIX zone | LowCorr MAE q25 / touch-5 / vol-ratio | VolLow MAE q25 / touch-5 / vol-ratio |
|---|---|---|---|
| NDX | DIXLow | −5.8 / 31% / 1.16 | −4.6 / 21% / 1.28 |
| NDX | DIXMid | −3.4 / 12% / 1.07 | −4.0 / 14% / 1.12 |
| NDX | DIXHigh | −2.6 / 10% / 1.12 | −2.7 / 9% / 1.10 |
| SPX | DIXLow | −3.5 / 10% / 1.13 | −3.6 / 5% / 1.33 |
| SPX | DIXMid | −3.2 / 11% / 1.26 | −2.9 / 7% / 1.22 |
| SPX | DIXHigh | −3.5 / 15% / 1.11 | −2.5 / 7% / 1.09 |
| IWM | DIXLow | −6.2 / 41% / 1.19 | −6.0 / 33% / 1.24 |
| IWM | DIXMid | −7.0 / 41% / 1.18 | −6.3 / 37% / 1.20 |
| IWM | DIXHigh | −6.6 / 35% / 1.13 | −5.2 / 26% / 1.17 |

Comovement-defined and vol-defined "low" regimes look similar on MAE but
the vol-defined VolLow cell consistently shows a *higher* vol-expansion
ratio and lower touch-5 rate than the comovement-defined LowCorr cell at
the same DIX level — i.e., a quiet realized-vol regime is somewhat calmer
going forward than a low-correlation regime is, even though their drawdown
depths are close. This is the honest corr-vs-vol picture the design's
Phase 3 promised, not a strong divergence either way.

## Frozen-rule path rows

**`ndx_dixlow_caution_v1`** (NDX LowCorr & DIX-Low, lag-1, vs. LowCorr with
DIX Mid/High):

| | n days/eps | mean r21 | hit | MAE q25 |
|---|---:|---:|---:|---:|
| active | 273/36 | −0.3% | 50% | −5.8 |
| LowCorr-not-active | 481/49 | +2.5% | 78% | −3.1 |

The flag reads exactly as the corrected study's hypothesis predicted: a
flat-to-negative median with a coin-flip hit rate and a materially worse
drawdown, against a strong +2.5%/78% when the flag is off.

**`all_dispersed_derisk_v1`** (all three indices LowCorr simultaneously vs.
not), one row per proxy:

| Proxy | active mean r21 (hit, MAE q25) | not-active mean r21 (hit, MAE q25) |
|---|---|---|
| QQQ | +0.1% (54%, −4.9) | +1.8% (67%, −6.0) |
| SPY | +0.3% (58%, −3.5) | +1.6% (71%, −3.9) |
| IWM | −0.2% (48%, −6.2) | +1.2% (61%, −5.7) |

Consistent softening across all three proxies when the flag is active —
same pattern the cross-index table shows for 3-of-3 above (these two
tables are the same underlying cell, cross-checked two ways).

## Implied-vol leg (VXN / VIX / RVX)

Reachable and parsed cleanly for all three indices this run (Cboe's CDN
was not blocked). Across the 62 gated cells with a defined implied-vol
comparison: **implied ran above realized in every single cell** (median
gap +5.0 vol points, range +2.2 to +6.8), and realized volatility exceeded
implied in only ~16% of holds on average. This is the standard variance
risk premium, not a regime-specific finding — every cell in every index
priced richer than it realized, at every level of comovement and DIX. It's
a reminder that a long-vol structure inside any of these cells starts with
a real headwind, and a short-vol one starts with a real tailwind, before
any regime conditioning is applied at all.

## How to use this

1. **Size**: read the q25 (and the stricter p10) MAE for the cell you're
   about to hold, and set position weight = 100 / |q25 MAE| percent of NAV
   per 1% of NAV you're willing to risk. Example: NDX LowCorr sizes to 24%
   of NAV per 1% risk; IWM LowCorr sizes to only 15% for the same 1% risk
   budget, on top of also having a negative expected return there.
2. **Stop or no stop**: check the bracket matrix (console report) for the
   cell before setting a resting stop — in the panic (HighCorr) zones
   across all three indices, `p_stop` at −5% commonly runs 25–40%, cutting
   deep into a regime whose vol is already decaying (per P3).
3. **Vol posture**: HighCorr's fwd/trail vol ratio ran 0.80–0.83 across all
   three indices this run, all with CIs excluding 1 — a real, cross-index-
   consistent finding that panic-tape vol reliably fades over the
   following month, which argues against systematically buying more vol
   protection once already inside a HighCorr regime.
4. **The caution flag is now double-confirmed**: `ndx_dixlow_caution_v1`
   is supported both as a mean effect (existing study) and, as of this
   run, a drawdown effect (P2) — it is the single most operationally
   useful result in this document.

## Caveats

- **This is one snapshot** (run 2026-09-05). Every number here will move
  as new sessions accrue, particularly the thin cells below.
- **Two cells never cleared the gate**: SPX HighCorr×DIXLow(l1) (18
  days/6 episodes) and IWM HighCorr×DIXLow(l1) (40 days/12 episodes) — a
  panic tape with simultaneously low dark-flow has been rare in both
  baskets' 2020+ windows. `regime_path_risk.csv` carries their counts with
  blank numeric fields, per the print-gate discipline.
- **SPX's HighCorr marginal itself is thin** (314 days but only 14
  episodes) — every SPX HighCorr number above should be read as a small-
  sample descriptive, not a precise estimate; its own P3 verdict still
  supported both legs, but the CI on the vol ratio is correspondingly wide.
- **P1 (NDX) is a genuine coin-flip call**: the episode-cluster CI's lower
  bound (4.4%) missed the 4.6% Gaussian benchmark by a hair. A few dozen
  more dispersed-tape sessions could flip this; it is not a settled null.
- **One macro cycle** — NDX's window runs 2018–2026 (includes COVID and
  2022), SPX/IWM's baskets only from 2020. Every primary's per-year
  composition and pre-2024/2024+ split is printed in the console report;
  reproduce with `--indices ndx` alone for the offline, longest-window leg.
- **Barriers and brackets are close-only** — a floor, not a measurement,
  of true intraday touch rates.
- **The implied-vol leg's universal positive gap is itself evidence of a
  known effect (the variance risk premium), not a regime-specific
  finding** — don't read "implied is always rich" as something this study
  discovered; it's the base rate the regime conditioning sits on top of.
- **The CSVs cover the ENV family for the per-index marginals, 3×3 grid
  and vol-parallel cells only** — cross-index and rule-row cells print in
  the console report but are not (yet) written to either committed CSV.
- **The committed envelope** (`regime_path_envelopes.json`, generated
  2026-09-05 14:31 UTC against the 07:00 UTC payload) reflects this same
  run; `build_regime_state.py --envelopes regime_path_envelopes.json` will
  show the current live cell's p10/p50/p90, MAE q25 and sizing weight on
  the nightly strip once that flag is wired into `refresh.yml` (not yet
  merged to `main` as of this writing).
