# Regime path study — findings

> **Status: CODE COMPLETE, FIRST LIVE RUN PENDING.** `regime_path_study.py`
> is built and tested (see `tests/test_regime_path.py`, synthetic panels
> only) against the design in `REGIME_PATH_STUDY_PLAN.md`. This document
> is the findings-doc scaffold Phase 5 of that plan calls for: every
> section, table shape and decision rule below is real and wired to real
> code; the numbers are not, because this sandboxed development session
> has neither the built dashboard payload (`docs/index.html`, produced by
> the nightly Pages workflow, gitignored) nor network access for the
> SPX/IWM iShares baskets or the Cboe implied-vol CDN. **Do not read any
> number in this document as a result** — every table below is a
> placeholder (`—`) until the first real run, which happens the moment
> someone with the built payload (or CI) runs the reproduce commands.
> When that run lands, replace this banner and fill in the tables; do not
> add new sections — the shape below is the one to populate.

Companion to `INTRA_INDEX_REGIME_FINDINGS.md` (the corrected regime study,
implemented and quoting real numbers): that document answers "what is the
mean 1-month forward return inside each comovement × DIX regime cell."
This one answers what a mean cannot: how volatile the path to that return
is, how deep it goes before it pays, whether a stop helps or amputates,
and whether the regime's own volatility expands or contracts while you
hold it.

Reproduce (once a built payload exists):
```
python regime_path_study.py                                   # all three indices, console report
python regime_path_study.py --indices ndx                     # offline (NDX only, no network)
python regime_path_study.py --csv regime_paths.csv --risk-csv regime_path_risk.csv
python regime_path_study.py --no-implied                      # skip the Cboe VXN/VIX/RVX leg
```
Data source: identical to `intra_index_regime_study.py` — the built
dashboard payload for NDX (fully offline), a fetched iShares basket for
SPX/IWM (network or a warm `.ndx_dark_cache`), and (optionally) Cboe's
published VXN/VIX/RVX history for the implied-vol leg (skips cleanly
offline).

## What this study adds over the mean

- **Fan quantiles** (p5…p95) of the cumulative return at nine checkpoints
  (day 1 through day 63) — the chartable object a mean can't give you.
- **Excursions**: median/q25/p10/worst max adverse excursion and its
  timing, max favorable excursion, give-back, and the share of paths
  dipping past −3/−5/−8/−12% — the numbers `ETF_PATH_PLAYBOOK.md` sizes
  positions to, now computed for every regime cell instead of two
  ETF-study analogs.
- **Barriers and brackets**: touch probabilities and a stop×target
  first-passage matrix, so a reader can see whether a resting stop helps
  or amputates in a given regime (close-only — a floor on true intraday
  touch rates).
- **Volatility**: forward realized vol vs. the cell's own trailing
  realized vol (expansion ratio), a variance ratio (trending vs.
  mean-reverting), vol-scaled tail shares against the Gaussian benchmarks,
  day-level microstructure (skew, excess kurtosis, autocorrelation), and
  — when reachable — the implied-minus-realized gap against Cboe's
  VXN/VIX/RVX.
- **Regime survival**: how often the cell is still active at d5/10/21/
  42/63, and for LowCorr cells specifically, how often a HighCorr jump
  lands inside the hold and what that costs the MAE distribution — the
  jump-risk price the corrected regime study's transition section left
  unpriced.
- **Three anchor families** per cell: ENV (every day the cell holds — an
  environment description), ENTRY (first day of formation, 21-session
  cool-down — directly comparable to the regime study's own entry
  section), and EPISODE-HOLD (a confirmed, variable-duration position:
  enter on the 3rd consecutive session, exit on the first failing close,
  capped at 63 sessions, with the 21-session post-exit leg that prices
  what the regime's ending pays or costs).
- **Sizing**: the position weight that loses 1% of NAV at the q25/p10 max
  adverse excursion, in the same units `ETF_PATH_PLAYBOOK.md` uses, so a
  regime cell and a chase-signal ETF sit on the same size scale.

Every conditional number is gated (42 anchors AND 5 episodes for ENV; 10
events for ENTRY; 5 episodes for EPISODE-HOLD) and, where gated, carries an
episode-cluster bootstrap CI (resampling whole contiguous episodes, the
same estimator the corrected regime study uses). Cells below the gate
print counts only — never a quoted statistic.

## Three PRIMARY hypotheses (pre-specified)

These are the only claims this document's headline may rest on; everything
else below is descriptive. Each carries a decision rule fixed before any
run, per `REGIME_PATH_STUDY_PLAN.md`.

### P1 — Jump risk in dispersed tapes

NDX LowCorr's share of 21-session outcomes beyond 2σ of trailing vol
(`|z|>2`) must exceed both the Gaussian 4.6% benchmark (episode-cluster CI
excluding it) **and** the same share inside HighCorr, or the claim is
written down as not supported.

| | LowCorr \|z\|>2 | epCI | HighCorr \|z\|>2 | Verdict |
|---|---:|---|---:|---|
| NDX | — | [—, —] | — | *(pending)* |

### P2 — The DIX-Low leg as a drawdown effect

Within NDX LowCorr, the DIX-Low leg's (lag-1) q25 max adverse excursion
must be reliably worse than the DIX-High leg's — an episode-cluster
**difference** CI (both legs' episodes resampled independently) excluding
zero — or the finding is "mean effect only, do not resize on it," matching
the corrected regime study's own finding for the terminal return.

| | LowCorr×DIXLow(l1) q25 MAE | LowCorr×DIXHigh(l1) q25 MAE | diff epCI | Verdict |
|---|---:|---:|---|---|
| NDX | — | — | [—, —] | *(pending)* |

### P3 — Panic tapes: worst excursions, decaying vol

NDX HighCorr must carry both the worst q25 MAE21 of the three comovement
zones **and** a median forward/trailing vol ratio below 1 with its
episode-cluster CI excluding 1, or the verdict demotes to "MAE leg only."

| Zone | q25 MAE21 | HighCorr fwd/trail vol ratio (med) | epCI | Verdict |
|---|---:|---:|---|---|
| LowCorr | — | — | — | — |
| MidCorr | — | — | — | — |
| HighCorr | — | [—, —] | — | *(pending)* |

## Per-index marginals (rolling basis, ENV family)

For each of NDX/SPX/IWM and each comovement-zone marginal (LowCorr,
MidCorr, HighCorr): fan quantiles at nine checkpoints, 21-day excursion
(MAE/MFE, trough/peak day, give-back, dip shares), close-only barrier
touch probabilities, the stop×target bracket matrix, the volatility block
(forward vs. trailing vol, variance ratio, vol-scaled tails, day-level
microstructure, implied-vol gap when reachable), the sizing translation,
survival at d5–d63, and — for LowCorr — the HighCorr-transition MAE split.
ENTRY and EPISODE-HOLD family summaries follow each marginal.

*(Table shape: see `regime_path_study.py:render_cell` / `render_entry` /
`render_hold` for the exact printed layout; `regime_path_risk.csv` carries
the same numbers in one row per cell once a run exists.)*

## The 3×3 risk table (comovement zone × DIX zone, lag-1)

Per index, per cell: n days/episodes, gate, q25/p10/worst MAE21, MFE,
touch −5%/−8%, vol-ratio median, sizing weight. *(Pending — see
`regime_path_risk.csv` after a run; the console report's per-cell blocks
carry the same numbers with fan quantiles and CIs alongside.)*

## Corr-vs-vol comparison (LowCorr vs. VolLow, by DIX zone)

For each DIX(lag-1) zone: MAE q25, the −5% touch probability and the
vol-expansion-ratio median inside the comovement-defined LowCorr cell
versus the realized-vol-defined VolLow cell — the corr-vs-vol honesty
owed since realized correlation and realized vol run ~0.8 correlated
in-sample (the corrected regime study's own vol-parallel finding).
*(Pending.)*

## Cross-index cells (N-of-3 dispersed)

Per index proxy (QQQ/SPY/IWM), by how many of the three indices sit in
LowCorr simultaneously (0 of 3 through 3 of 3, common dates): the same
ENV-family blocks as the marginals. *(Pending.)*

## Frozen-rule path rows

The path-study versions of the two evaluable rules in `frozen_rules.json`:

- **`ndx_dixlow_caution_v1`**: active (NDX LowCorr & DIX-Low, lag-1) vs.
  LowCorr-but-not-active (DIX Mid/High). *(Pending.)*
- **`all_dispersed_derisk_v1`**: active (all three indices LowCorr
  simultaneously) vs. not, one row per proxy. *(Pending.)*

(`ndx_tilt_screen_v1` names per-name tickers rather than a regime cell and
has no path-study analog here; it stays scored by
`intra_index_regime_study.py --score-log`.)

## Implied-vol leg (VXN / VIX / RVX)

Median implied-minus-realized (vol points) and the share of holds where
realized exceeded implied, per cell, when Cboe's CDN is reachable
(`load_cboe_vol`; skips cleanly and says so otherwise — this environment
had no network access, so every leg below reads "skipped"). *(Pending.)*

## How to use this (once populated)

1. **Size**: read the q25 (and the stricter p10) MAE for the cell you're
   about to hold, and set position weight = 100 / |q25 MAE| percent of
   NAV per 1% of NAV you're willing to risk — `ETF_PATH_PLAYBOOK.md`'s own
   convention, so a regime cell and a chase-signal ETF compare directly.
2. **Stop or no stop**: read the bracket matrix for the cell. A stop that
   sits at a level with a high `p_stop` relative to `p_target` is cutting
   into the cell's own favorable-excursion tail, not just limiting loss —
   check `e_ret` at a few stop/target combinations before choosing one.
3. **Vol posture**: read the vol-expansion-ratio median and the
   |z|>1/|z|>2 tail shares. A ratio reliably above 1 with heavy tails says
   size down and expect volatility to expand further; a ratio below 1
   says the regime's realized vol is more likely to fade than persist
   through the hold (as P3 tests for HighCorr specifically).

## Caveats

- **One macro cycle** (the payload's own window; 2018–2026 for NDX, per
  `INTRA_INDEX_REGIME_FINDINGS.md`). Every per-year line, pre-2024/2024+
  split and leave-one-year-out range printed alongside the three primaries
  exists precisely because a handful of years can move any single cell's
  q25 a lot.
- **Episode counts are thin in parts of the HighCorr row.** The corrected
  regime study counts 14–29 HighCorr episodes per index depending on the
  DIX cross; several HighCorr×DIX cells will print `-- (below gate)`
  rather than a number, and that is the honest result, not a bug to chase.
- **DIX-zone episodes are short** (the regime study counts under-five-day
  average episode length in several LowCorr×DIX cells): the EPISODE-HOLD
  family's 3-day confirmation exists for exactly this reason, and its
  counts will be visibly thinner than ENV's for the 3×3 grid.
- **Barriers and brackets are close-only** — a floor, not a measurement,
  of true intraday touch rates; every barrier/bracket table says so in
  its own line.
- **The variance ratio uses a linear (summed daily-return) proxy** for the
  compounded h-session return, standard at daily-return scale but an
  approximation nonetheless (see `variance_ratio`'s docstring in
  `regime_path_study.py`).
- **The CSVs (`regime_paths.csv`, `regime_path_risk.csv`) cover the ENV
  family for the per-index marginals, 3×3 grid and vol-parallel cells
  only** — the cross-index and rule rows print in the console report but
  are not (yet) written to either committed CSV; a future pass can extend
  the writer if the committed surface needs them.
- **This document itself is the biggest caveat**: it is a scaffold, not a
  study. Treat every `—` as exactly that until a run replaces it.
