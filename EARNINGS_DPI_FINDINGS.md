# DPI and earnings: positioning into the report, and what happens to it after

Two questions on the same event set — the NDX-100 (93 entities — Alphabet's GOOG
folded into GOOGL), **2,880 quarterly earnings events** (2,657 with dark-pool
data), Aug 2018 – Jul 2026:

1. Does an elevated dark-pool indicator (DPI) heading **into** a report line up
   with how the stock performs **after** it?
2. What happens to **DPI itself** in the week, fortnight and month after the
   report?

Reproduce:
```
python fetch_earnings_edgar.py --out earnings_dates_edgar.csv        # SEC EDGAR dates
python earnings_dpi_study.py --earnings earnings_dates_edgar.csv     # -> earnings_dpi_events.csv
```
Visual report: `earnings_dpi_report.html` (`docs/earnings.html` for Pages).

## Definitions

- **DPI** = per-name short ÷ total off-exchange volume (FINRA), 0–1 — the same
  per-name construction used across `ndx_dark_residual.py`.
- **DPI5 / DPI10** = mean daily DPI over the 5 / 10 trading days ending the
  session **before** the report — strictly pre-announcement (no look-ahead).
- **T** = last clean pre-news close (timing-aware: report day for after-hours
  reporters, prior session for before-open reporters).
- **Forward returns** = adjClose(T+h)/adjClose(T) − 1 at h = 1 (next-day
  reaction), 5 (1-week), 10 (2-week) and 21 (1-month) trading sessions.
  Split-adjusted.
- **Within-name percentile**: each event's DPI ranked against that same name's
  own history, so "high DPI" means high for that stock.
- **Post-earnings DPI** = mean DPI over T+1…T+5 / T+10 / T+21, and the **change**
  vs. that event's own DPI10, in percentage points of the ratio. Session T is
  excluded from both sides (for an AMC reporter it is the report day itself —
  part pre-news tape, part after-hours reaction).

---

# Part 1 — DPI into the report vs. the return after it

**A weak positive tilt that only shows up at the one-month horizon.** Direction
is consistent with the conventional dark-pool reading (high short-volume share =
market-makers shorting to fill buy orders = accumulation = bullish), but the
magnitude is small and the shorter horizons are indistinguishable from zero.

DPI10 vs. forward return, and the high-minus-low within-name tercile spread:

| Horizon | DPI10 Pearson r | p | High−low tercile spread | spread p |
|---|---:|---:|---:|---:|
| Next-day (T+1) | +0.012 | 0.54 | −0.35 pp | 0.32 |
| 1-week (T+5)   | +0.018 | 0.35 | −0.65 pp | 0.16 |
| 2-week (T+10)  | +0.042 | 0.031 | +0.11 pp | 0.84 |
| 1-month (T+21) | **+0.059** | **0.003** | +0.81 pp | 0.24 |

Only the pooled 1-month correlation clears significance; the tercile buckets do
not on their own. At one month the high-DPI tercile averaged **+2.41%** (55% up)
vs the low-DPI tercile's **+1.59%** (53% up). DPI5 is a touch weaker still
(1-month r=+0.049).

## Realized volatility

Annualized realized volatility over each post-earnings window,
RV_h = sqrt(252/h · Σ_{i=1..h} r_i²) with r_i the daily log return on session
T+i (columns `next_day_rvol` … `m1_rvol`):

| Window | mean RV | median | DPI10 corr | high-DPI vs low-DPI |
|---|---:|---:|---:|---:|
| Next-day (T+1) | 85.0% | 63.6% | +0.054 | 89.5% vs 82.5% |
| 1-week (T+5)   | 53.1% | 43.9% | +0.066 | 54.5% vs 53.4% |
| 2-week (T+10)  | 45.1% | 37.9% | +0.063 | 45.9% vs 45.9% |
| 1-month (T+21) | 39.7% | 34.1% | +0.061 | 40.1% vs 41.0% |

RV is highest the day after the report (the earnings gap, annualized) and decays
as the spike averages into calmer sessions. DPI barely moves it, so what
directional tilt exists above is **drift, not a volatility effect**.

## Robustness

- **Timing:** 1-month positive for both after-hours (r=+0.080) and before-open
  (r=+0.026) reporters. Next-day ~0 for both.
- **Cohort:** 1-month positive for both mega-caps (r=+0.058) and the other 73
  names (r=+0.056). Next-day is −0.01 for mega-caps, +0.01 for the rest.
- **By year:** positive in all nine years, ranging +0.02 to +0.09 — no single
  year carries it, and none reverses it.
- **Per name:** 47 of 90 names show a positive DPI10→1-month correlation — a
  bare majority.

---

# Part 2 — how DPI itself changes after the report

## Headline

**Dark-pool positioning jumps hard on the reaction day, is still slightly
elevated across week one, and is back to its run-in level by week two.** Over
the full month the net change is not distinguishable from zero.

| Window | mean DPI | change vs run-in DPI10 | median | % of events up | t | p |
|---|---:|---:|---:|---:|---:|---:|
| Reaction day (T+1) | 0.4972 | **+3.94 pp** | +4.07 pp | 62% | +20.6 | <0.001 |
| 1 week (T+1…T+5)   | 0.4679 | **+1.00 pp** | +0.99 pp | 51% | +6.2 | <0.001 |
| 2 weeks (T+1…T+10) | 0.4626 | +0.50 pp | +0.51 pp | 49% | +3.2 | 0.001 |
| 1 month (T+1…T+21) | 0.4600 | +0.26 pp | +0.41 pp | 48% | +1.7 | 0.089 |

(Pre-earnings DPI10 averages 0.4581; changes are in percentage points of the
0–1 ratio.)

Cut into **non-overlapping** slices, the lift is entirely a week-one effect —
and really a T+1 effect:

| Slice | change vs run-in | p |
|---|---:|---:|
| Week 1 (T+1…T+5)     | +1.00 pp | <0.001 |
| Week 2 (T+6…T+10)    | +0.00 pp | 0.99 |
| Weeks 3–4 (T+11…T+21)| +0.04 pp | 0.83 |

## What the change is made of

**1. Mean reversion — real, but a third the size the naive sort suggests.**
Because the change is `post − DPI10`, DPI10's own measurement noise enters with a
minus sign, so correlating the change against DPI10 is mechanically negative.
Ranking events instead on an **earlier, non-overlapping window** (sessions
−60…−11, column `dpi_prior`) removes that artifact:

| Horizon | corr vs DPI10 (biased) | corr vs prior window (clean) | high-tercile Δ | low-tercile Δ | high−low | p |
|---|---:|---:|---:|---:|---:|---:|
| 1 week   | −0.388 | −0.087 | −0.14 pp | +2.07 pp | −2.21 pp | <0.001 |
| 2 weeks  | −0.443 | −0.106 | −0.81 pp | +1.72 pp | −2.53 pp | <0.001 |
| 1 month  | −0.509 | −0.125 | −1.28 pp | +1.75 pp | −3.03 pp | <0.001 |

Names that go into a print with dark-pool positioning already high give some of
it back over the following month; names that go in low drift up. Positioning is
still persistent overall — corr(pre, post) ≈ +0.63 at every horizon — it just
pulls toward the name's own normal level.

**2. DPI follows the tape.** The reaction-day spike is *larger* after bad news
(+4.84 pp vs +3.14 pp), but from T+2 on the two paths separate the other way: DPI
stays ~+0.5 to +1.3 pp elevated after an up reaction and sits ~0.5–0.9 pp *below*
run-in after a down one.

| Horizon | after an up reaction | after a down reaction | difference | p |
|---|---:|---:|---:|---:|
| 1 week  | +1.48 pp | +0.48 pp | +1.00 pp | 0.002 |
| 2 weeks | +1.14 pp | −0.20 pp | +1.34 pp | <0.001 |
| 1 month | +0.87 pp | −0.41 pp | +1.28 pp | <0.001 |

## Does the change predict anything? No.

A change measured over T+1…T+5 is only observable at T+5, so it is tested
against the return **from** that point — never a window it overlaps. Against the
same forward windows, the pre-earnings *level* keeps working and the change adds
nothing:

| Signal | Forward window | r (change) | p | r (pre-earnings DPI10) | p |
|---|---|---:|---:|---:|---:|
| Δ DPI over 1 week  | T+5 → T+21  | −0.022 | 0.26 | **+0.070** | <0.001 |
| Δ DPI over 2 weeks | T+10 → T+21 | −0.033 | 0.09 | **+0.044** | 0.025 |
| Δ DPI over 1 month | T+21 → T+42 | +0.008 | 0.67 | −0.014 | 0.49 |

Sorting events into terciles by the size of their week-one DPI change gives a
−0.16 pp spread in the T+5→T+21 return (p=0.71). A double sort confirms it: mean
T+5→T+21 return by run-in tercile (rows) × week-one-change tercile (columns) —

| run-in \ ΔDPI | falling | flat | rising |
|---|---:|---:|---:|
| low  | −0.29% | +0.29% | +0.23% |
| mid  | +0.99% | +1.64% | +1.47% |
| high | +1.68% | +1.48% | +2.08% |

— the gradient runs down the rows (the pre-earnings level), not across the
columns.

## How to read Part 2

- Dark-pool activity **reacts** to earnings: a large, reliable one-day jump in
  short-volume share, decaying within about a week.
- The direction of that reaction **tracks the print** — DPI holds up after good
  news and fades after bad — which makes it a coincident indicator, not a
  leading one.
- Anything that looks like "DPI is building after the report" is mostly
  **mean reversion** toward the name's ordinary level, and roughly a third of
  the apparent reversion is a measurement artifact.
- For forecasting, **where DPI was going in is what matters**; what it did
  afterwards is noise.

## Caveats

- Small effect sizes throughout; single index; a mostly-bull 2018–2026 sample;
  DPI is a noisy daily series.
- The month-long DPI averages overlap the price windows they are compared
  against; the forward-return tests above avoid that by construction, the
  descriptive tables do not (they are not meant as signals).
- **93 entities from 100 index members**: 6 foreign filers (ASML, ARM, PDD,
  CCEP, FER, NBIS/TRI) file 6-K rather than 8-K and are omitted; SPCX/HONA had no
  matching filings; and Alphabet's two share classes are merged (see below).
- **Dual-class merge:** GOOG and GOOGL are the same company (same CIK → identical
  report dates), so keeping both double-counts Alphabet. They are folded into one
  entity: DPI is re-derived **volume-weighted** from the summed off-exchange
  short/total across both classes — not an average of the two ratios, which would
  over-weight the thinner, higher-DPI Class C (GOOG ≈ 40% of combined off-exchange
  volume, and its DPI runs ~0.42 vs GOOGL's ~0.37) — and GOOGL's prices are used
  for returns. Pass `--no-merge-classes` to keep them separate.
- Earnings 8-Ks were isolated by matching Item 2.02 filings to each 10-Q/10-K,
  which drops non-earnings 2.02s (Tesla delivery numbers, monthly sales,
  guidance pre-announcements). AMC/BMO is inferred from the 8-K acceptance
  timestamp converted to US/Eastern.

## Note on an earlier preliminary cut

A first pass on **only 20 mega-caps over 2023–2026** with hand-typed report dates
suggested high DPI preceded a *weaker* next-day move (r ≈ −0.16). That did **not**
survive scaling up: with authoritative EDGAR dates, AMC/BMO timing, the full
93-name universe and 8 years of history, the mega-cap next-day effect is −0.01
(i.e. noise), and what is left is the modest *positive* 1-month relationship in
Part 1. The earlier result was small-sample / single-window fragility — a useful
reminder to treat any single narrow cut with suspicion. The 20-name curated input
is retained as `earnings_dates.csv`.

An earlier version of this document also quoted a stronger Part 1 (1-month
r=+0.098, tercile spread +2.83 pp). Those numbers predate the fix in #23 for a
DPI/price index misalignment that read the wrong pre-earnings window for every
event; the tables above are from the corrected data, and the effect is weaker
than the pre-fix figures suggested.
