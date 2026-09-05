# Regime return paths — how the month unfolds, not just where it ends

`INDEX_COMOVEMENT_FINDINGS.md` scores each of the 27 NDX/SPX/IWM DIX comovement
regimes by the mean/median 1-month forward return. A mean says nothing about
the **road there**: whether the month is calm or violent, whether it trends in
one direction or chops back and forth, how deep the drawdown along the way
runs, and how wide the range of plausible outcomes is. This adds that path
layer — forward realized volatility, a trend/chop measure, drawdown and
run-up, and the full quantile fan — under the same 27-regime framework, with
the persistence, vol-clustering, and out-of-sample checks the headline numbers
need before they're trusted.

Data: **1,673 common sessions, 6 Jan 2020 → 4 Sep 2026**, same DIX5-decile
regimes as the comovement study (Low = deciles 1–3, Mid = 4–7, High = 8–10,
per index, over the full sample). Own-price (QQQ/SPY/IWM) forward paths, 21
trading sessions, no look-ahead in the outcome.

Reproduce:
```
python regime_paths_study.py --cache-dir .ndx_dark_cache
python regime_paths_study.py --lens entry --min-run 5
python regime_paths_study.py --csv regime_paths.csv --fan-csv regime_paths_fan.csv
```
Interactive version: the **"How the month unfolds"** section of the
Comovement tab (`docs/comovement.html`), which carries fan charts, the full
metrics strip, and a lens toggle for every regime, driven by
`regime_paths.py`.

## 1. Read the episode table before anything else

A 3-letter regime code changes constantly. Across the sample:

| | |
|---|--:|
| contiguous episodes | 549 |
| median episode length | **2 sessions** |
| 75th-percentile length | 4 sessions |
| longest episode | 31 sessions (all-Low) |

So "the path after regime X" almost always describes an **environment**, one
or two days into a run that will likely flip tomorrow — not a month spent
continuously inside that regime. Every path statistic below is scored under
two lenses for that reason:

- **Every-day** (matches the existing return table): every session with that
  code and a full 21-session forward window. Descriptive of the environment,
  heavily overlapping.
- **Entry-day** (`index_comovement_study.entry_events`, 21-session cool-down):
  only the first day a regime forms. Closest to independent, small n — the
  honest read on whether a *setup* looks different from the *environment*.

**6 of the 27 possible codes never occur at all** (`regime_paths.unobserved_codes`,
reported by `regime_paths_study.py`'s Section 1) — zero evidence, not merely
rare: **LHL, LHM, LHH, HLL, HLM, HLH**. Read the pattern: these are *exactly*
the six codes where NDX and SPX sit at *opposite* DIX5 deciles (one Low, the
other High) — for all three IWM values. **NDX and SPX dark flow never
diverge to opposite extremes on the same day, anywhere in 2020–2026.** Every
regime in §3–§6 below therefore has at least Mid-level agreement between the
two large-cap gauges; the comovement study's own large-cap correlation is the
likely reason, and it means a live reading of "NDX Low, SPX High" (or the
reverse) has no historical precedent to size against at all.

## 2. Baseline path profile (the yardstick)

| Index | fwd vol | vol/trailing | max DD | max run-up | efficiency | time above water | r21 IQR |
|---|--:|--:|--:|--:|--:|--:|--:|
| NDX (QQQ) | 19.4% | 1.00× | −2.3% | +4.3% | 0.21 | 76% | 7.7% |
| SPX (SPY) | 13.9% | 0.99× | −1.8% | +3.2% | 0.22 | 76% | 5.3% |
| IWM       | 21.3% | 1.00× | −2.9% | +3.8% | 0.19 | 62% | 8.4% |

Efficiency ratio (ER) is Kaufman's |net move| ÷ Σ|daily moves|: 1.0 is a
straight line, → 0 is pure chop. A baseline ER of ~0.20–0.22 says the average
month is already fairly noisy — most of the day-to-day motion cancels out.
IWM's much lower time-above-water (62% vs 76%) is the small-cap tax showing up
in the path, not just the mean.

## 3. The path map — every-day lens

**All 21 regimes observed in the sample** (NDX shown, largest first; SPX/IWM
in `regime_paths.csv` and the page — no cell is left out of either). `ER` is
the efficiency ratio (→1 trend, →0 chop); `vol CI` is the 95% block-bootstrap
CI on mean forward vol, suppressed below 42 scored days. `vm rv` is the
vol-persistence control (§4) — skip to that section before trusting
"hot"/"quiet" at face value. The 6 codes with **no row at all** (never
observed — see §1) are LHL, LHM, LHH, HLL, HLM, HLH.

| Regime | n | r21 | fwd vol | vol CI | ER | max DD | time above water | label |
|---|--:|--:|--:|---|--:|--:|--:|---|
| MMM (all Mid) | 209 | +2.4% | 19.2% | [17.7, 25.1] | 0.19 | −2.3% | 71% | normal mixed up |
| HHM | 179 | +3.1% | 17.3% | [16.6, 22.9] | 0.27 | −1.8% | 76% | normal trend up |
| LLL | 174 | +2.0% | 18.9% | [16.8, 28.9] | 0.19 | −3.2% | 71% | normal mixed up |
| HHH | 163 | +3.4% | 19.4% | [17.9, 26.1] | 0.24 | −1.8% | 81% | normal mixed up |
| **MML** | 133 | +2.5% | 17.6% | [16.1, 19.2] | **0.15** | −1.5% | 81% | **normal chop up** |
| MMH | 119 | +0.6% | 21.6% | [17.7, 30.4] | 0.18 | −3.2% | 69% | normal mixed flat |
| LLM | 104 | +0.5% | 23.8% | [20.4, 33.3] | 0.16 | −3.0% | 62% | hot chop flat |
| **LLH** (requested) | 96 | +3.6% | **24.0%** | [19.8, 35.9] | 0.23 | −2.1% | 74% | **hot mixed up** |
| HHL | 80 | +4.1% | 16.8% | [16.1, 25.4] | 0.22 | −1.6% | 86% | normal mixed up |
| LMM | 69 | −1.1% | 20.1% | [19.6, 25.4] | 0.17 | −3.6% | 48% | normal mixed down |
| MLL | 48 | +0.3% | 22.2% | [17.4, 44.2] | 0.20 | −2.6% | 74% | normal mixed flat |
| MLM | 46 | +2.6% | 21.2% | [17.3, 34.3] | 0.21 | −1.8% | 81% | normal mixed up |
| **MHM** | 43 | +3.8% | **15.7%** | [13.8, 28.5] | **0.30** | −1.6% | 86% | **quiet trend up** |
| MLH | 34 | +1.1% | 21.2% | — | 0.24 | −2.8% | 64% | normal mixed up |
| LMH | 33 | +3.5% | 23.7% | — | 0.16 | −5.8% | 33% | hot chop up |
| **HML** | 30 | +6.7% | 22.3% | — | **0.28** | −0.2% | 95% | **hot trend up** |
| **HMH** | 30 | −2.8% | **26.3%** | — | 0.23 | **−5.8%** | 43% | **hot mixed down** |
| LML | 26 | −0.3% | 16.8% | — | 0.09 | −2.8% | 71% | normal chop flat |
| MHH | 26 | +2.0% | 17.5% | — | 0.27 | −2.4% | 76% | normal trend up |
| HMM | 20 | +1.8% | 17.2% | — | 0.18 | −2.1% | 79% | normal mixed up |
| MHL | 11 | +1.8% | 12.9% | — | 0.11 | −0.9% | 90% | quiet chop up |

Four cells worth naming (NDX letters are read N/S/I — e.g. LLH = NDX Low, SPX
Low, IWM High):

1. **MML (NDX Mid, SPX Mid, IWM Low) has an attractive mean and the choppiest
   path in the table.** ER 0.15 is the lowest of any cell with a meaningful
   sample — well under the baseline's already-noisy 0.21 — despite 81% of
   days finishing above water. The +2.5% mean is earned by grinding, not
   trending: expect whipsaw, not a clean ride.
2. **LLH — the originally-requested divergence — earns its return through a
   violent month, not a calm one.** Forward vol (24.0%) is the second-hottest
   cell in the table, a full 24% above baseline. This directly confirms the
   read implied in the feasibility pass and sharpens
   `INDEX_COMOVEMENT_FINDINGS.md`'s finding: the LLH mean is real, but a
   trader sizing into it should expect NDX-vol-on-steroids, not a quiet
   drift higher.
3. **MHM (NDX Mid, SPX High, IWM Mid) is the standout calm-trend cell.**
   Lowest forward vol (15.7%, 19% below baseline), highest efficiency ratio
   (0.30 — the most trend-like path observed), shallow drawdown (−1.6%), and
   86% time above water. Good mean, low vol, low chop, shallow risk — the one
   cell that looks attractive on every path axis at once.
4. **HML and HMH are the table's two vol extremes, in opposite directions.**
   HML (NDX High, SPX Mid, IWM Low — the single best NDX-mean regime in the
   comovement study) is *hot and trending*: high vol, but ER 0.28 says that
   vol is mostly directional, and the median max drawdown is a shallow
   −0.2%. HMH (NDX High, SPX Mid, IWM High — the worst NDX-mean regime) is
   *hot and directionless*: the hottest forward vol of any decent-n cell
   (26.3%) paired with the deepest median drawdown (−5.8%) and only 43% of
   days above water. Same "large-cap DIX High, SPX Mid" setup, opposite IWM
   leg, opposite path character entirely.

## 4. Vol-persistence control: does "hot"/"quiet" survive?

Realized vol is sticky, and DIX regimes aren't independent of the vol
environment a cell happens to occur in. The control: split every scored day
into trailing-vol terciles, then ask what a cell's forward vol *would* be if
it just reproduced its own trailing-vol tercile mix (`vm rv` below) — the
gap between that and the cell's actual forward vol is what the DIX pattern
itself adds.

| Regime | index | raw fwd vol | vol-matched baseline | survives? |
|---|---|--:|--:|---|
| MHM | NDX | 15.7% | 20.6% | **yes, strongly** — 5pp quieter than even its own matched baseline |
| MHM | SPX | 11.8% | 15.5% | **yes, strongly** |
| HHL | NDX | 16.8% | 19.5% | **yes** — quieter than matched baseline too |
| LLH | NDX | 24.0% | 22.6% | **partially** — still hot, but most of the raw gap vs. the 19.4% overall baseline is really "LLH occurs in already-volatile stretches" |
| HML | NDX | 22.3% | 23.1% | **no** — at or slightly *below* its matched baseline; the raw "hot" tag is a vol-clustering artifact, not a HML-specific effect |
| HMH | NDX | 26.3% | 24.4% | **yes** — ~2pp hotter than matched baseline |
| HMH | IWM | 24.9% | 22.4% | **yes** — consistent with NDX |

**MHM's quiet-trend character is the most robust finding in this study** — it
holds on both large-cap indices and survives the vol-matching control by a
wide margin, meaning the DIX pattern itself (not just a calm macro backdrop)
is associated with a dampened, trending month. HML's "hot" tag, by contrast,
mostly evaporates under the control — its return is real (confirmed
separately by the comovement study's entry-day count of 28 across HML's two
variants), but its elevated raw volatility is largely just the vol regime it
tends to occur in.

## 5. Entry-day lens: does the character hold at formation?

For the headline regimes, comparing every-day medians to the entry-day
(first-day-of-regime) medians:

| Regime | index | entries | entry rv (med) | entry ER | every-day rv (med) | every-day ER |
|---|---|--:|--:|--:|--:|--:|
| LLH | NDX | 13 | 23.1% | 0.27 | 24.0% | 0.23 |
| LLH | IWM | 13 | 24.2% | 0.29 | 21.9% | 0.21 |
| HML | NDX | 7 | 17.0% | 0.34 | 22.3% | 0.28 |
| MML | NDX | 21 | 18.0% | 0.23 | 17.6% | 0.15 |

LLH is **already hot on the day it forms** — the every-day vol reading isn't
an artifact of averaging over long episodes; a fresh occurrence looks about
as volatile as the steady-state. MML, by contrast, looks *less* choppy at
formation (ER 0.23) than in its every-day average (ER 0.15) — the
whipsaw character builds as the episode runs, not on day one. HML's n=7 is
too thin to read as more than a hint (lower vol, even higher ER at entry) —
noted, not relied on.

## 6. What survives persistence, expanding cutoffs, and out-of-sample

Three robustness cuts, each reporting only the cells whose **label changes**
(full detail in the CLI output / `regime_paths.csv`):

- **`--min-run 3`** (only sessions where a regime has already held 3+ days):
  roughly a third of cells with enough surviving n change label. **NDX MMM
  goes flatter the longer it persists** (normal mixed up → normal mixed
  flat — the most common regime has momentum on day one that fades; SPX and
  IWM's MMM label doesn't move). **LLH gets *hotter* as it persists on SPX**
  (normal mixed up → hot mixed up) — the violence isn't front-loaded, it
  builds. **HHL calms with persistence on both NDX and SPX** (normal → quiet
  mixed up), the mirror image; IWM's HHL label is unchanged.
- **Expanding-window (no-look-ahead) deciles**: relabels roughly half the
  higher-n cells. NDX HHH picks up a "hot" tag it didn't carry under
  full-sample deciles (normal mixed up → hot mixed up); NDX MML swaps its
  "chop" tag for a "quiet" one (normal chop up → quiet mixed up); NDX MHM
  *loses* its "quiet" tag (quiet trend up → normal trend up). Sample sizes
  shift too (HHM 179→246 days) because live-knowable cutoffs redraw the
  Low/Mid/High boundaries as the sample accrues. Treat every hot/quiet/
  trend/chop call in §3 as **descriptive of hindsight cutoffs**, not
  something a live trader could have read off identically in real time.
- **Out-of-sample (cutoffs fit pre-2024, scored 2024+)**: of the 8–9 cells
  with ≥ 21 OOS days per index, **roughly half changed label** — e.g. NDX
  MHM loses its "quiet" tag OOS (normal vol, not quiet), SPX MML loses
  "quiet" too, and IWM MMH flips from "mixed down" to "quiet chop up"
  entirely. This is a genuinely rough OOS record: path *character*, like the
  comovement study's own returns, should be read as descriptive of
  2020–2023 more than as a forward-looking rule.

## 7. What it says

1. **Mean return and path character are different axes, and they don't move
   together.** MML has a good mean and the choppiest path in the sample;
   LLH has a good mean and the most violent path outside HMH. Sizing or
   holding-period decisions built on the mean alone will be wrong-footed by
   the actual month.
2. **MHM (NDX Mid / SPX High / IWM Mid) is the one cell that is good on every
   axis** — above-baseline return, the lowest forward vol, the highest
   trend-efficiency, a shallow drawdown — and it is also the one hot/quiet
   finding that survives the vol-persistence control on both large-cap
   indices. It is the strongest single read in this study.
3. **HML and HMH are the same "large-cap DIX High, SPX Mid" setup with
   opposite path character depending on the IWM leg** — trending-hot when
   IWM is Low, directionless-hot with the deepest drawdown in the table when
   IWM is High. The comovement study already knew these were the best/worst
   NDX-mean regimes; the path layer shows *why* the good one is investable
   and the bad one is a landmine (deep, non-trending drawdown, not a
   controlled pullback).
4. **The requested LLH divergence is a hot regime, not a calm one, and it is
   already hot on the day it forms** — not an artifact of long episode
   averaging. A position sized for LLH's attractive mean should be sized for
   NDX-vol running well above baseline, not for a quiet drift higher.
5. **None of this generalizes cleanly out of sample.** Roughly half the
   labeled cells flip character 2024+, mirroring the comovement study's own
   OOS caveat. Read §3–§5 as a map of what 2020–2023 looked like, and §6 as
   the honest account of how much of that map survives contact with new
   data.

## Caveats

- **Overlapping windows.** Every-day-lens statistics overlap heavily (~1,673
  sessions ≈ 80 independent months); entry-day is the closer-to-independent
  lens but is thin for most cells (n as low as 7–15) — treat medians as
  sturdier than means, small cells as suggestive.
- **Path metrics are noisy per day.** A single day's forward vol or
  efficiency ratio is one realization of a 21-day path; only the cell-level
  medians here are meaningful.
- **In-sample deciles by default.** §3–§5 use full-sample DIX5 deciles (mild
  look-ahead), matching the existing comovement table; §6 reports what
  changes under expanding, no-look-ahead cutoffs.
- **Sample window and IWM DIX reconstruction** carry the same caveats as
  `INDEX_COMOVEMENT_FINDINGS.md` (COVID crash-recovery and the 2022 drawdown
  sit inside the sample; IWM's DIX deciles are internally consistent but not
  comparable in raw level to NDX/SPX).
- The full tidy table (every-day + entry lens, all metrics, both bases) is in
  `regime_paths.csv`; the fan-chart quantiles are in `regime_paths_fan.csv`.
