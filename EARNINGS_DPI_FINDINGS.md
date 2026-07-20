# DPI into earnings vs. post-earnings performance

Does an elevated dark-pool indicator (DPI) heading **into** an earnings report line
up with how a stock performs **after** the report? Study of 20 NDX mega-caps,
**202 after-hours earnings events**, Jul 2023 – Jan 2026.

Reproduce: `python earnings_dpi_study.py` (writes `earnings_dpi_events.csv`).
Visual report: `earnings_dpi_report.html`.

## Definitions

- **DPI** = per-name short ÷ total off-exchange volume (FINRA), 0–1 — the same
  per-name construction used across `ndx_dark_residual.py`.
- **DPI5 / DPI10** = mean daily DPI over the 5 / 10 trading days ending the day
  **before** the report date (T−1). These names report after the close, so the
  cut-off is strictly pre-announcement — no look-ahead.
- **Next-day** = adjClose(T+1) / adjClose(T) − 1, where T = report day (the
  earnings reaction gap).
- **1-month** = adjClose(T+21 sessions) / adjClose(T) − 1. Split-adjusted.
- **Within-name percentile**: each event's DPI ranked against that *same name's*
  history, so "high DPI" means high for that stock, not just a high-DPI name.

## Headline result

**Yes — modestly, and only in the immediate reaction.** Higher DPI going in is
associated with a *weaker* next-day move; the effect is gone by a month.

| Signal | Horizon | Pearson r | p | Spearman r | within-name r |
|---|---|---:|---:|---:|---:|
| DPI5  | next-day | −0.151 | 0.032 | −0.149 | −0.152 |
| DPI10 | next-day | **−0.159** | **0.023** | −0.162 | −0.165 |
| DPI5  | 1-month | −0.005 | 0.94 | −0.037 | −0.032 |
| DPI10 | 1-month | −0.014 | 0.84 | −0.049 | −0.036 |

**Tercile buckets** (by within-name DPI10 percentile):

| Bucket | Next-day mean | % up | 1-month mean |
|---|---:|---:|---:|
| Low DPI  (n=60) | **+0.67%** | 53% | +3.68% |
| Mid DPI  (n=62) | −0.13% | — | +0.85% |
| High DPI (n=80) | **−2.69%** | 34% | +0.99% |

High−Low next-day spread = **−3.36 pp** (Welch t = −2.41, **p = 0.018**).
High−Low 1-month spread = −2.69 pp (t = −1.03, p = 0.31, **not significant**).

Next-day return by within-name DPI10 quintile (1 = lowest → 5 = highest):
`+0.2% / +1.9% / −1.4% / −3.0% / −2.4%` — a roughly monotonic fade for the top
half. The 1-month quintile pattern is flat/noisy.

## How to read it

- The edge is a **next-day phenomenon**. That it evaporates by a month is
  consistent with elevated pre-earnings DPI marking **crowded positioning that
  unwinds on the news** ("sell the news") rather than a durable directional call.
- It is a **pooled tendency, not a law**: 11/20 names show the negative tilt, and
  it is carried by high-beta / semiconductor names (MU −0.82, QCOM −0.60,
  PANW −0.59, AVGO −0.57, TSLA −0.47) while a few reverse (TXN +0.52, ADBE +0.33).
  Per-name samples are ~10 events — texture, not precision.
- **Base rate:** only 42% of these events were up the next day at all (median
  −1.43%); this mega-cap cohort "sold the news" often over the sample window.

## Caveats

- Modest effect size; single 2½-year bull-market regime; DPI is a noisy daily
  series; 20 mega-caps ≠ the broad market.
- Earnings dates are **hand-curated** — no bulk earnings feed was reachable from
  the build environment (Yahoo's earnings module needs a crumb from a blocked
  host; SEC/Nasdaq/aggregators are blocked by egress policy). Dates were validated
  against the price tape: known reactions (META +20.3%, NVDA −8.5%, AVGO +24.4%,
  TSLA +12.1%, GOOGL −7.3%) reproduce to a tenth of a percent. See
  `earnings_dates.csv` to edit/extend the universe.
- An optional `--anchor` mode snaps each event to its dominant nearby price move;
  it is **off by default** because it biases the sample toward large (and, via
  volatility clustering, disproportionately negative) moves. Results above use the
  curated dates as-is.
