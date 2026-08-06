# SPX DIX: absolute-decile vs trailing-1yr breakout — which predicts better?

**Question.** The reconstructed SPX dollar-DIX level is non-stationary — its "bullish
floor" drifted up from ~0.42 (2020–21) to ~0.47–0.49 (2022–23), settling ~0.45–0.47
since. A day at 0.45 was a *high* reading in 2020 and an *ordinary* one in 2024. So does
the classic **absolute-level decile** signal (the dashboard's "forward return by DIX
decile" bars) actually predict SPX forward returns, or does a signal measured **relative
to the trailing 1-year average** — a breakout above / breakdown below its own recent
range — carry more predictive power?

**Data.** Reconstructed SPX dollar-DIX (Σ$-short ÷ Σ$-off-exchange over IVV constituents,
5-day MA) + SPY 1/2/3-month forward returns, pulled from the built dashboard payload
(`docs/index.html`, generated 2026-08-04). Daily, 2020-01-02 → 2026-08-04 (n≈1,655).
Reproduce with `python3 spx_dix_signal_study.py`.

Returns are overlapping, so all significance uses Newey–West HAC t-stats (lags = horizon)
and moving-block bootstrap CIs — the repo's existing machinery.

---

## Verdict

**The trailing-1yr breakout/breakdown signal has more — and the only statistically
meaningful — predictive power. The absolute-level decile is essentially non-predictive,
exactly because the level is non-stationary. But two big caveats: the edge is (a) weak and
only clear at the 3-month horizon, and (b) *mean-reverting*, not the bullish-breakout
direction folklore assumes.**

Your non-stationarity instinct is correct and is the whole story: a fixed absolute-decile
cut mostly labels *which regime a day is in*, not a tradable signal.

---

## 1. Absolute level carries almost no edge

| SPX signal | 1mo IC | 2mo IC | 3mo IC (NW t) |
|---|---|---|---|
| **abs** (raw DIX level) | +0.055 | +0.058 | −0.018 (t −0.4) |
| **dev** (DIX − trailing-1yr avg) | −0.009 | −0.092 | **−0.201 (t −2.4)** ✱ |
| **z** (standardized breakout) | −0.011 | −0.085 | −0.189 (t −2.2) ✱ |

IC = Spearman rank correlation. ✱ = p < 0.05. The absolute level never reaches |t| > 0.6 at
any horizon; the trailing-relative breakout is significant at 3 months.

**Decile spread (mean 3-month fwd return, top decile − bottom):**

| method | 1mo | 2mo | 3mo |
|---|---|---|---|
| abs decile — *look-ahead* (dashboard bars) | +0.4 | +0.3 | −1.9 |
| abs decile — expanding, real-time | +1.1 | −1.3 | −4.3 |
| **breakout-dev decile — real-time** | −0.9 | −2.5 | **−6.6** |

The picture is cleanest in the **shape** of the 3-month decile curves:

- **Absolute deciles → flat/noise.** `[7.1, 2.9, 3.4, 4.0, 3.9, 3.9, 3.7, 4.9, 4.0, 5.1]` pp.
  No monotonic structure; the top decile (7.1) is just the low-DIX 2020 days.
- **Breakout deciles → a clean monotone decline.** `[6.8, 5.8, 4.7, 4.8, 4.1, 4.3, 4.3, 3.4, 2.9, 0.2]` pp.
  Below-average DIX (breakdown, D1) precedes the best returns; the biggest breakout (D10)
  precedes the worst. A ~6.6pp gradient, in order.

## 2. Breakout vs breakdown states (DIX above / below its trailing 1yr avg)

| horizon | breakout (>1yr avg) | breakdown (<1yr avg) | spread | NW t |
|---|---|---|---|---|
| 1mo | +1.24pp (hit 68%) | +1.57pp (hit 70%) | −0.34 | −0.5 |
| 2mo | +2.28pp (hit 70%) | +3.26pp (hit 80%) | −0.98 | −0.8 |
| 3mo | +3.02pp (hit 70%) | +5.25pp (hit 84%) | −2.23 | −1.3 |

Breakdowns *beat* breakouts at every horizon — the opposite of "high DIX = bullish."
(High hit-rates everywhere are just the market's upward drift, not signal.)

## 3. Robustness — it's weak and regime-dependent

Across indices at 3 months, the breakout deviation beats the absolute level (larger |t|) in
**5 of 6** index/subsample cells (SPX, NDX robustly; IWM only post-2021). The mean-reversion
sign (dev IC < 0) is consistent across SPX/NDX/IWM in the 2022+ subsample.

**But nothing is sign-stable year to year.** 3-month IC of the breakout signal by year:
`2020 −0.06, 2021 −0.44, 2022 −0.44, 2023 −0.27, 2024 +0.05, 2025 +0.43, 2026 −0.11`.
The edge is a pooled/full-sample average; the *direction itself flips* in 2024–25. Treat
this as "the absolute level is unusable and the trailing-relative version is at best a weak,
regime-dependent mean-reversion tilt at ~3 months" — not a standalone trading rule.

---

## Practical takeaways

1. **Don't read the absolute DIX level as a fixed signal.** Its ~0.44–0.45 floor is a moving
   target; a raw-level decile mostly encodes the year. This is why the dashboard's
   **"trailing deciles (no look-ahead)"** toggle is the honest default — it normalizes to the
   trailing window, exactly the fix this study validates.
2. **If you use DIX directionally, normalize to its trailing 1-year range** (percentile, or
   deviation from the 252-day mean). That is where what little edge exists lives.
3. **Mind the sign and the horizon.** On this reconstructed dollar-DIX the usable relationship
   is mean-reverting and only shows up at ~3 months — a DIX *breakdown* below its year has
   led higher forward returns, not a breakout above it.

## Caveats

- This is the repo's **reconstructed** SPX dollar-DIX, not the SqueezeMetrics DIX product;
  sign and strength may differ from the commercial series.
- Sample is 2020–2026 (the payload's plot window) — it excludes the 2019 period you
  referenced. A live rebuild with `--plot-start 2019-01-01` would extend it (heavier: refetches
  ~500 constituents).
- Overlapping returns and a 6.5-year window mean modest statistical power; the honest reading
  is "the absolute level is non-predictive; the trailing-relative version is weakly, and only
  sometimes, informative."
