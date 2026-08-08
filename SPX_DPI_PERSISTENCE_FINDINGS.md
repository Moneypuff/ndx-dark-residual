# DPI persistence on a downtrend: a high-decile DPI *streak*, not a single print

Pushback tested (per request): the single-stock tests used a single-day DPI
reading and found no edge — but a stock on a **high-decile DPI streak** may do
better over the 1–2 month range. So on a downtrend, require the **10-day average
DPI to sit in the higher deciles** (persistent dark accumulation), not one print.

**Short answer (updated after the reversal control): the pushback was a real
methodological improvement — persistence surfaced the most stable single-stock
number in the whole thread (a beta-adjusted ~+0.7%/month long-only alpha at 1
month that reproduces out of sample). But it does NOT survive a short-term-reversal
+ benchmark control: that +0.7% is almost entirely the generic tendency of
downtrend names to bounce, and the streak's OWN marginal is ~+0.15pp and not
significant. See "Short-term-reversal control" below — it is the decisive test.**

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

## 4. Short-term-reversal control — the decisive test

**Short-term reversal (STR)** is the 1-month anomaly where recent *losers* bounce
and recent *winners* give back (liquidity provision / overreaction correction).
Our setup buys stocks in a *downtrend* — recent losers by construction — so the
+0.7% could be STR dressed up as dark flow. CAPM alpha removes market beta but
*not* reversal. Two controls, among downtrend names:

**Confound check (a surprise that helps the mechanism):** the 10+ streak names had
a prior-21d return of **+0.91%** vs **−2.52%** for the no-streak downtrend names —
i.e. the streak names are the ones *stabilizing* inside a longer downtrend, **not**
deeper losers. So classic STR would favor the *no-streak* names; it is not
*selecting* the streak.

**Fama-MacBeth (daily cross-sectional `alpha ~ const + rback + streak10`, averaged):**
the streak-dummy coefficient is the alpha it adds *beyond* recent return and
*relative to other downtrend names*:

| horizon | raw 10+ long-only | reversal-adjusted streak coefficient |
|---:|---|---|
| **21d** | +0.71% [+0.14, +1.29] | **+0.15% [−0.35, +0.66]** · IS −0.16% · OOS +0.46% |
| 42d | +0.60% | −0.11% [−0.81, +0.62] |
| 63d | +0.71% | +0.11% [−0.79, +1.13] |

Double-sort (10+ streak − no-streak alpha within recent-return terciles): **+0.18 /
−0.02 / +0.45 pp** — averages ~+0.2pp, inconsistent across buckets.

**The +0.71% deflates.** It decomposes as **[common downtrend-name bounce ~+0.55%]
+ [streak-specific marginal ~+0.15%]**. Almost all of it is the generic bounce of
beaten-down names (which hits streak and no-streak alike); the streak's own
contribution, once recent return is controlled and it is benchmarked against other
downtrend names, is **~+0.15pp and not significant** in full, IS, or OOS.

## Verdict (revised)

**The single print was indeed the wrong lens — persistence is the right idea, and
it produced the best-looking single-stock number in the thread** (+0.71% CAPM
alpha, 1-month, monotone in streak length, stable OOS). That much stands.

**But it does not survive the reversal + benchmark control.** Stripped of the
common downtrend-bounce and the recent-return characteristic, the streak's own
alpha is ~+0.15pp and indistinguishable from zero (full/IS/OOS); the 2–3 month
versions were in-sample artifacts; and the market-neutral long-short is null. So
the honest conclusion is: **beaten-down names bounce, and a high-decile DPI streak
does not reliably add to that bounce.** Notably it is *not* classic reversal
*selecting* the names (they are stabilizers, not deep losers) — but the dark-flow
streak is not a confirmed independent edge on this data. Consistent with the
thread's throughline: dark flow's tradeable signal is systematic/index-level, not
single-stock selection.

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
