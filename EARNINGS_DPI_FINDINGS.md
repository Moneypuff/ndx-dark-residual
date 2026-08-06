# DPI into earnings vs. post-earnings performance

Does an elevated dark-pool indicator (DPI) heading **into** an earnings report
line up with how a stock performs **after** the report? Study of the NDX-100
(**100 entities** — Alphabet's GOOG folded into GOOGL, and the foreign 6-K
filers now included), **2,985 quarterly earnings events**, Aug 2018 – Jul 2026.

Reproduce:
```
python fetch_earnings_edgar.py --out earnings_dates_edgar.csv        # SEC EDGAR dates (8-K + 6-K)
python earnings_dpi_study.py --earnings earnings_dates_edgar.csv \
    --summary-out earnings_dpi_summary.txt                           # -> earnings_dpi_events.csv
```
Visual report: `earnings_dpi_report.html` (docs/earnings.html). The refresh
workflow commits `earnings_dpi_summary.txt` on every run — the numbers below
are from the 2026-08-01 run.

## Definitions

- **DPI** = per-name short ÷ total off-exchange volume (FINRA), 0–1.
- **DPI5 / DPI10** = mean daily DPI over the 5 / 10 trading days ending the
  session **before** the report — strictly pre-announcement.
- **T** = last clean pre-news close (timing-aware: report day for after-hours
  reporters, prior session for before-open reporters).
- **Forward returns** = adjClose(T+h)/adjClose(T) − 1 at h = 1, 5, 10, 21
  sessions, split-adjusted — and, the headline outcome, their **market-excess
  twins** (`*_xret` = the name's return minus **QQQ's** over the identical
  window), since raw returns in a mostly-bull sample partly just ride beta.
- **Within-name percentile**: each event's DPI ranked against that name's own
  history — in two flavours: full-history (mild look-ahead) and **expanding**
  (ranked only against the name's prior events, min 8 — knowable on the day).
- **Season-cluster bootstrap**: earnings cluster in reporting weeks, so the
  ~2,985 events are cross-sectionally correlated. Headline CIs/p-values
  resample **calendar quarters** (~33 seasons) with replacement.

## Headline result (revised)

**The simple "high DPI into earnings → better month" reading does not survive
honest measurement.** Raw and pooled, DPI10 vs the 1-month return is still
mildly positive (r = +0.051, p = 0.008) — but that association is carried by
market beta and season effects, not by the names themselves:

| Test, 1-month horizon | value | inference |
|---|---:|---|
| DPI10 vs raw return, pooled | r = +0.051 | p = 0.008 (overstated: events not independent) |
| DPI10 vs **QQQ-excess** return | r = +0.011 | cluster p = 0.55 — nothing |
| High−low DPI tercile, raw | +0.49 pp | t = 0.71, p = 0.48 |
| High−low tercile, **excess** | −0.67 pp | cluster CI [−1.63, +0.29], p = 0.18 |
| High−low tercile, excess, **expanding (tradable) ranks** | **−1.08 pp** | cluster p = 0.03 |
| High−low tercile, excess, 1-week | **−0.92 pp** | cluster p = 0.007 |

Net of the market, high within-name DPI into a report has been associated with
*slightly worse*, not better, relative performance — and the one cut that is
statistically solid (1-week, −0.92pp, p=0.007) points the wrong way for the
bullish reading. The momentum double-sort agrees: within every pre-earnings
momentum tercile, the high-DPI minus low-DPI excess spread is ~0 or negative
(−1.9pp among stocks that ran +14.5% into the print, p = 0.16).

## The one cut that does work: dark accumulation into a down-gap

Splitting the **post-reaction drift** (T+1 → T+21, i.e. after the earnings gap
is known) by the direction of the reaction:

| Reaction | n | drift, high-DPI | drift, low-DPI | spread | corr(DPI10, drift) |
|---|---:|---:|---:|---:|---|
| **Gap down < −2%** | 879 | **+2.36%** | +0.19% | **+2.17 pp (p = 0.015)** | r = +0.092 (p = 0.006) |
| Flat ±2% | 854 | +1.53% | +1.35% | +0.18 pp (p = 0.83) | r = +0.024 (ns) |
| Gap up > +2% | 980 | +2.06% | +1.73% | +0.33 pp (p = 0.77) | r = +0.047 (ns) |

This is exactly where an "informed dark accumulation" story should show up and
the only place it does: **names that were being accumulated in dark pools into
the report and then gapped down recover over the following month; names that
gap down without that dark bid don't** (+0.19%). It is conditional (you only
know the gap after it happens), it is one cut among several, and it has not
been through an out-of-sample split — but it is the strongest surviving result
in the study.

## Realized volatility

High pre-report DPI is a mild **volatility** tell for the reaction itself:
corr(DPI10, next-day RV) = +0.061 (p = 0.001), high-DPI events realize 86.5%
annualized next-day vol vs 78.3% for low-DPI. The gap fades by the 2-week
window — DPI says a bigger reaction is coming, not which direction.

## Signal decay across data vintages

Worth recording: an earlier vintage of this study (2,663 events) reported
r(DPI10, 1-month) = +0.098 and a +2.83pp raw tercile spread. Re-running the
*identical* raw statistics on last night's pre-upgrade data already gave
+0.061 / +0.84pp, and today's build (with the foreign filers added) gives
+0.051 / +0.49pp. The raw effect has been shrinking as the sample grows — the
usual signature of a period-specific artifact rather than a stable edge, and
consistent with the excess-return analysis above attributing most of it to
beta in a rising market.

## Universe completeness

- The 6 foreign private issuers previously dropped (no 8-K/2.02 trail) are now
  recovered via a scored **6-K heuristic** — candidate 6-Ks inside each
  quarter's expected reporting window (5–80 days after quarter end), scored by
  filing-index contents (EX-99 exhibit, earnings-keyword document names) —
  contributing 81 events across ASML, ARM, PDD, CCEP, FER, NBIS and TRI
  (`source=6k` in `earnings_dates_edgar.csv`). Heuristic by construction:
  dates are authoritative EDGAR acceptance timestamps, but a mis-picked 6-K
  would inject a non-earnings event. Mostly before-open reporters (53 bmo /
  18 intraday / 10 amc).
- Alphabet's dual classes remain merged (volume-weighted DPI, GOOGL returns);
  pass `--no-merge-classes` to keep them separate.

## How to read it

- The interesting object is no longer a pooled "buy high-DPI reporters" tilt —
  that was mostly beta. It is the **conditional** result: dark accumulation
  into a report that then gaps down has marked names that recover.
- All inference here should use the season-clustered numbers; the pooled
  p-values overstate precision by roughly an order of magnitude (2,985 events
  ≈ 33 independent seasons).

## Caveats

- Single index, mostly-bull 2018–2026 sample, DPI is a noisy daily series.
- The gap-down result is one conditional cut (n = 879, p = 0.015 clustered by
  season for the tercile spread) and has not been validated out of sample.
- The 6-K date heuristic is best-effort (see above); excluding `source=6k`
  rows reproduces the previous 93-entity universe.
- Earnings 8-Ks are isolated by matching Item 2.02 filings to each 10-Q/10-K;
  AMC/BMO timing is inferred from acceptance timestamps converted to
  US/Eastern.
