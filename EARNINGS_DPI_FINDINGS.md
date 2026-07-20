# DPI into earnings vs. post-earnings performance

Does an elevated dark-pool indicator (DPI) heading **into** an earnings report
line up with how a stock performs **after** the report? Study of the NDX-100
(94 names with 8-K earnings filings), **2,694 quarterly earnings events**,
Aug 2018 – Jul 2026.

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
- **Next-day** = adjClose(T+1)/adjClose(T) − 1 (the earnings reaction).
- **1-month** = adjClose(T+21 sessions)/adjClose(T) − 1. Split-adjusted.
- **Within-name percentile**: each event's DPI ranked against that same name's
  own history, so "high DPI" means high for that stock.

## Headline result

**Higher DPI into a report lines up with better performance after it — modestly
next-day, and more clearly over the following month.** Direction is consistent
with the conventional dark-pool reading (high short-volume share = market-makers
shorting to fill buy orders = accumulation = bullish).

| Signal | Horizon | Pearson r | p | Spearman r | within-name r |
|---|---|---:|---:|---:|---:|
| DPI5  | next-day | +0.039 | 0.042 | +0.036 | +0.046 |
| DPI10 | next-day | +0.050 | 0.010 | +0.045 | +0.050 |
| DPI5  | 1-month | +0.064 | 0.001 | +0.084 | +0.051 |
| DPI10 | 1-month | **+0.099** | **<0.001** | +0.115 | +0.093 |

**Tercile buckets** (by within-name DPI10 percentile):

| Bucket | Next-day mean | % up | 1-month mean | % up |
|---|---:|---:|---:|---:|
| Low DPI  (n=859) | +0.46% | 51% | +1.10% | 50% |
| Mid DPI  (n=882) | +0.05% | — | +1.82% | — |
| High DPI (n=953) | **+1.27%** | 56% | **+3.95%** | 61% |

High−Low **1-month** spread = **+2.85 pp** (Welch t = +4.10, **p < 0.001**).
High−Low next-day spread = +0.81 pp (t = +2.36, p = 0.018).

Next-day return by within-name DPI10 quintile (1 = lowest → 5 = highest) is
roughly flat then up; the 1-month gradient is monotonically increasing.

## Robustness

- **Timing:** 1-month effect positive for both after-hours (r ≈ +0.11) and
  before-open (r ≈ +0.08) reporters. Next-day: +0.05 (AMC) / +0.06 (BMO).
- **Cohort:** 1-month positive for both mega-caps (r ≈ +0.11) and the other 74
  names (r ≈ +0.09). Next-day is ~flat for mega-caps (+0.00) and positive for
  the rest (+0.06).
- **By year:** absent in 2018–2019, builds from 2020 on; same sign in nearly
  every year thereafter. Stronger in 2022+ (r ≈ +0.07 next-day, +0.11 1-month).
- **Per name:** 62 of 91 names show a positive DPI10→1-month correlation.

## How to read it

- The edge is **modest**: r ≈ 0.10 at a month explains ~1% of the variance in
  post-earnings monthly returns — a real aggregate tilt, not a single-name trade.
- It is **broad and reasonably stable** across timing, cohort, and sub-period,
  which is what separates it from noise.

## Caveats

- Small effect sizes; single index; a mostly-bull 2018–2026 sample; DPI is a
  noisy daily series.
- **94 of 100 names**: 6 foreign filers (ASML, ARM, PDD, CCEP, FER, NBIS/TRI)
  file 6-K rather than 8-K and are omitted; SPCX/HONA had no matching filings.
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
