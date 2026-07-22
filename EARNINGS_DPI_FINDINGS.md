# DPI into earnings vs. post-earnings performance

Does an elevated dark-pool indicator (DPI) heading **into** an earnings report
line up with how a stock performs **after** the report? Study of the NDX-100
(93 entities — Alphabet's GOOG folded into GOOGL), **2,663 quarterly earnings
events**, Aug 2018 – Jul 2026.

Reproduce:
```
python fetch_earnings_edgar.py --out earnings_dates_edgar.csv        # SEC EDGAR dates
python earnings_dpi_study.py --earnings earnings_dates_edgar.csv     # -> earnings_dpi_events.csv
```
Visual report: `earnings_dpi_report.html`.

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

## Headline result

**Higher DPI into a report lines up with better performance after it — modestly
next-day, and more clearly over the following month.** Direction is consistent
with the conventional dark-pool reading (high short-volume share = market-makers
shorting to fill buy orders = accumulation = bullish).

**The edge compounds with the holding period.** DPI10 vs. forward return, and
the high-minus-low within-name tercile spread, at four horizons:

| Horizon | DPI10 Pearson r | p | High−low tercile spread | spread p |
|---|---:|---:|---:|---:|
| Next-day (T+1) | +0.049 | 0.012 | +0.78 pp | 0.026 |
| 1-week (T+5)   | +0.071 | <0.001 | +1.08 pp | 0.017 |
| 2-week (T+10)  | +0.068 | <0.001 | +1.31 pp | 0.013 |
| 1-month (T+21) | **+0.098** | **<0.001** | **+2.83 pp** | **<0.001** |

The gap between high- and low-DPI events widens the longer you hold, and its
significance tightens with it. At one month the high-DPI tercile averaged
**+3.96%** (61% up) vs the low-DPI tercile's **+1.13%** (50% up). DPI5 tells the
same story a touch weaker (next-day r=+0.038, 1-month r=+0.063).

Next-day return by within-name DPI10 quintile (1 = lowest → 5 = highest) is
roughly flat then up; the 1-month gradient is monotonically increasing.

## Realized volatility

Annualized realized volatility over each post-earnings window,
RV_h = sqrt(252/h · Σ_{i=1..h} r_i²) with r_i the daily log return on session
T+i (columns `next_day_rvol` … `m1_rvol`):

| Window | mean RV | median | DPI10 corr | high-DPI vs low-DPI |
|---|---:|---:|---:|---:|
| Next-day (T+1) | 84.5% | 62.8% | +0.049 | 83.9% vs 83.5% |
| 1-week (T+5)   | 53.7% | 44.0% | +0.079 | 54.0% vs 52.9% |
| 2-week (T+10)  | 45.6% | 38.2% | +0.087 | 45.6% vs 45.0% |
| 1-month (T+21) | 40.3% | 34.4% | +0.081 | 39.7% vs 40.4% |

RV is highest the day after the report (the earnings gap, annualized) and decays
as the spike averages into calmer sessions. DPI barely moves it — high- and
low-DPI events realize almost identical volatility — so the directional DPI edge
above is **drift, not a volatility effect**.

## Robustness

- **Timing:** 1-month effect positive for both after-hours (r ≈ +0.11) and
  before-open (r ≈ +0.08) reporters. Next-day: +0.05 (AMC) / +0.06 (BMO).
- **Cohort:** 1-month positive for both mega-caps (r ≈ +0.11) and the other 74
  names (r ≈ +0.09). Next-day is ~flat for mega-caps (+0.00) and positive for
  the rest (+0.06).
- **By year:** absent in 2018–2019, builds from 2020 on; same sign in nearly
  every year thereafter. Stronger in 2022+ (r ≈ +0.07 next-day, +0.11 1-month).
- **Per name:** 61 of 90 names show a positive DPI10→1-month correlation.

## How to read it

- The edge is **modest**: r ≈ 0.10 at a month explains ~1% of the variance in
  post-earnings monthly returns — a real aggregate tilt, not a single-name trade.
- It is **broad and reasonably stable** across timing, cohort, and sub-period,
  which is what separates it from noise.

## Caveats

- Small effect sizes; single index; a mostly-bull 2018–2026 sample; DPI is a
  noisy daily series.
- **93 entities from 100 index members**: 6 foreign filers (ASML, ARM, PDD,
  CCEP, FER, NBIS/TRI) file 6-K rather than 8-K and are omitted; SPCX/HONA had no
  matching filings; and Alphabet's two share classes are merged (see below).
- **Dual-class merge:** GOOG and GOOGL are the same company (same CIK → identical
  report dates), so keeping both double-counts Alphabet. They are folded into one
  entity: DPI is re-derived **volume-weighted** from the summed off-exchange
  short/total across both classes — not an average of the two ratios, which would
  over-weight the thinner, higher-DPI Class C (GOOG ≈ 40% of combined off-exchange
  volume, and its DPI runs ~0.42 vs GOOGL's ~0.37) — and GOOGL's prices are used
  for returns. Merged GOOGL corr(DPI10, 1-month) = +0.32, between the classes'
  separate +0.14 / +0.41. Removing the double-count leaves the pooled headline
  essentially unchanged (DPI10 1-month r held at +0.098). Pass
  `--no-merge-classes` to keep them separate.
- Earnings 8-Ks were isolated by matching Item 2.02 filings to each 10-Q/10-K,
  which drops non-earnings 2.02s (Tesla delivery numbers, monthly sales,
  guidance pre-announcements). AMC/BMO is inferred from the 8-K acceptance
  timestamp converted to US/Eastern.

## Note on an earlier preliminary cut

A first pass on **only 20 mega-caps over 2023–2026** with hand-typed report dates
suggested high DPI preceded a *weaker* next-day move (r ≈ −0.16). That did **not**
survive scaling up: with authoritative EDGAR dates, AMC/BMO timing, the full
94-name universe and 8 years of history, the mega-cap next-day effect is −0.05
(p ≈ 0.4, i.e. noise), and the durable signal is the *positive* 1-month
relationship above. The earlier result was small-sample / single-window
fragility — a useful reminder to treat any single narrow cut with suspicion.
The 20-name curated input is retained as `earnings_dates.csv`.
