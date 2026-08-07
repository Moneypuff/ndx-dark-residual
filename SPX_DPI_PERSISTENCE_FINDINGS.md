# DPI persistence on a downtrend: a high-decile DPI *streak*, not a single print

Pushback tested (per request): the single-stock tests used a single-day DPI
reading and found no edge — but a stock on a **high-decile DPI streak** may do
better over the 1–2 month range. So on a downtrend, require the **10-day average
DPI to sit in the higher deciles** (persistent dark accumulation), not one print.

**Short answer: the pushback is substantially right — persistence surfaces a real,
beta-adjusted ~+0.7%/month alpha the single print missed, and it reproduces out of
sample at the 1-month horizon. Two caveats: the 2–3 month version is an in-sample
artifact that fails OOS, and it's a long-only (not a clean market-neutral) edge.**

Full S&P 500, 2019–2026 (848,632 name-days; 40% of days are in a 3-month
downtrend, 24% have the 10-day-avg DPI in the top quintile). Signals are
self-relative (ranked within each name's own trailing year). Outcomes: forward
21/42/63-day return as RAW, excess-vs-SPY, and **CAPM alpha** (`fwd − β·SPY_fwd`,
β on trailing 252 daily returns). Reproduce:
```
python spx_dpi_persistence_study.py --start 2019-01-01 --out spx_dpi_persistence.csv
```

## 1. A streak-length gradient — longer streak, more alpha (1 month)

Forward CAPM alpha by DPI-streak bucket, among downtrend names (a streak = run of
consecutive sessions with the 5-day-MA D in the top quintile):

| streak (days) | 21d alpha [95% CI] |
|---|---|
| 0 | +0.29% [−0.15, +0.74] |
| 1–4 | +0.34% [−0.10, +0.77] |
| 5–9 | +0.47% [−0.02, +0.96] |
| **10+** | **+0.71% [+0.14, +1.29]** |

Monotone in streak length, and the **10+ streak clears zero** — the first
beta-adjusted, dose-responsive single-stock signal in the whole investigation.
The single-print tests missed this because they never conditioned on *duration*.
The streak-specific piece is ~+0.42 pp on top of the no-streak downtrend bounce
(+0.29%).

## 2. The out-of-sample split — 1 month holds, 2–3 months don't

The 10+ streak alpha (long-only), evaluated in-sample (2019–22) vs OOS (2023–26):

| horizon | full [95% CI] | IS 2019–22 | OOS 2023–26 |
|---:|---|---|---|
| **21d** | **+0.71% [+0.14, +1.29]** | **+0.71%** [−0.16, +1.56] | **+0.71%** [−0.06, +1.52] |
| 42d | +0.60% [−0.22, +1.49] | +1.30% [+0.21, +2.49] | −0.05% [−1.29, +1.22] |
| 63d | +0.71% [−0.34, +1.85] | +1.75% [+0.51, +3.16] | −0.29% [−1.78, +1.38] |

**The 1-month effect is the real one.** Its point estimate is essentially
identical across full / in-sample / out-of-sample (**+0.710% / +0.714% / +0.707%**,
hit rate 57 / 60 / 54%) — the effect *size* reproduces almost exactly out of
sample, which is far stronger evidence than a cell that merely stays positive.

**The 2–3 month versions do not survive.** They were strong in-sample (42d +1.30%,
63d +1.75%, both CI-positive) but collapse to ~0/negative out of sample (42d
−0.05%, 63d −0.29%) — textbook in-sample overfitting. So of the "1–2 month range",
only the **1-month** horizon holds up; the 2-month does not.

## 3. What does NOT confirm it

- **The market-neutral long-short is null.** Darkest-minus-least-dark by the
  10-day-avg percentile, on CAPM alpha, is +0.07% [−0.31, +0.44] full (IS −0.12%,
  OOS +0.26%). So it is **not** "more DPI percentile → more alpha" (a smooth
  cross-sectional factor); it is a **threshold/persistence regime** — sustained
  10+ days in the top quintile is a distinct state, which is exactly the streak
  intuition, not a rankable continuum.
- **Per-name conditional is only a whisper.** For each name, [signal-high &
  downtrend] minus its own downtrend baseline: the 10-day-avg beats the single
  print (53.3% vs 52.5% of names positive on raw 21d) — consistent direction —
  but washes out on alpha (51.3% positive, sign-p=0.56). The gradient/streak
  result is where the signal lives, not the per-name average.
- **Each OOS half's CI grazes zero** (21d IS lower bound −0.16, OOS −0.06). The
  full sample is significant and the halves are near-identical, but neither half
  is independently significant at 95%.

## Verdict

**Going about it with a single print was the wrong way — persistence matters.**
A 10+ session high-decile DPI streak in a downtrend earns ~+0.7% CAPM alpha over
the next month, monotone in streak length, and the effect size reproduces out of
sample. This is the one single-stock dark-flow result in the whole thread worth
taking seriously.

But keep it honest: it is a **1-month, long-only, threshold** effect. The 2–3
month extension is an in-sample artifact; the market-neutral long-short is null;
and each OOS half only grazes significance. Before trading it, it needs (a)
adjustment for the short-term-reversal factor (downtrend + accumulation overlaps
reversal — CAPM removes market beta, not that), (b) transaction-cost and
capacity checks (it is long-only turnover in mid/large caps), and (c) a true
forward out-of-sample as more data accrues. Treat it as a live, promising
hypothesis — the best lead here — not a finished signal.

## Caveats

- **Long-only alpha can carry residual factor tilt** beyond CAPM beta (notably
  short-term reversal, since the setup is "bounce off a downtrend"). The streak-0
  downtrend bucket already earns +0.29% (the generic bounce); the +0.42 pp
  marginal from the streak is the part attributable to persistent dark flow, and
  that is what needs factor-adjusting to confirm.
- **Survivorship** — current S&P 500 members; the CAPM-alpha and per-name-baseline
  framings remove the level/tilt bias.
- **Overlap** — 21-day windows; daily-basket block-bootstrap CIs and the IS/OOS
  stability are the honest lenses, not any single pooled mean.
- The single-print vs 10-day-avg and full/IS/OOS numbers are in
  `spx_dpi_persistence.csv`; event mechanics (run-length, gradient, per-name) are
  unit-tested in `tests/test_dpi_persistence.py`.
