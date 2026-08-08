# Falling-knife filter: do LOW-DPI downtrend stocks keep selling off?

Reframe (per request): downtrend stocks bounce on average, but catching a falling
knife is dangerous. Instead of asking whether HIGH DPI *wins* (null), ask whether
**LOW 10-day-average DPI** in a downtrend flags the knives that *keep dropping* —
worse hit rate, fatter left tail. If DPI could **screen those out**, the loss
reduction would be alpha even with an unchanged mean.

**Short answer: no. Low-DPI downtrend stocks bounce just as hard, hit positive
just as often, blow up (>−10% / >−20%) just as often, and have the same left tail
as high-DPI ones. DPI does not separate falling knives from bouncers, so using it
as a downside filter buys no loss reduction.** If anything the reversal-controlled
big-loss probability is *slightly lower* for low-DPI names — the opposite of the
falling-knife fear.

Full S&P 500, 2019–2026; among names in a 3-month downtrend (40% of the panel),
split by the self-relative percentile of the 10-day-average DPI: **LOW** = bottom
quintile, **HIGH** = top quintile, **KEEP** = everything except LOW (the filtered
book). Forward 21/42d, RAW and CAPM alpha; daily-cross-section block-bootstrap CIs.
Reproduce:
```
python spx_falling_knife_study.py --start 2019-01-01 --out spx_falling_knife.csv
```

## The loss/bounce profile is flat across DPI (RAW forward return)

**21-day:**

| group | n | mean | hit% | P(<−10%) | P(<−20%) | 5th-pct | worst-decile |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LOW** | 74,726 | +1.64% | 57.1% | 9.1% | 2.0% | −13.0 | −15.6 |
| MID | 185,505 | +1.51% | 56.6% | 9.8% | 2.3% | −13.6 | −16.2 |
| **HIGH** | 82,028 | +1.72% | 56.9% | 9.2% | 2.1% | −13.4 | −16.1 |
| ALL | 342,259 | +1.58% | 56.8% | 9.5% | — | −13.4 | −16.1 |

**42-day:** LOW +3.29% / hit 59.7% / P(<−10%) 13.6% / worst-decile −19.6 vs HIGH
+3.28% / 59.6% / 13.9% / −21.0 — again indistinguishable (LOW's tail is if
anything *shallower*).

Every LOW-vs-HIGH gap is inside the noise:

| difference (21d, RAW) | mean | hit% | P(<−10%) |
|---|---:|---:|---:|
| HIGH − LOW | +0.05 [−0.30, +0.41] | −0.32 [−1.67, +1.13] | +0.14 [−0.76, +1.19] |
| KEEP − ALL (the filter) | −0.01 [−0.07, +0.04] | −0.11 [−0.35, +0.14] | +0.09 [−0.07, +0.25] |

**Filtering out the low-DPI names does not lift the hit rate or cut the big-loss
rate** — the KEEP-minus-ALL differences are ~0, and P(<−10%) even ticks slightly
*up* when you drop LOW (i.e. LOW isn't even the worst bucket). Same story on CAPM
alpha and at 42 days.

## Reversal control — LOW *is* the recent loser, but that doesn't mean it keeps falling

LOW-DPI names had a prior-21d return of **−4.44%** vs **+0.19%** for HIGH — so LOW
is indeed the still-being-sold group. But that recent weakness does **not** carry
into worse forward outcomes: a Fama-MacBeth linear-probability model of a big-loss
dummy (fwd < −10%) on `[recent-return, LOW-dummy]` gives a LOW coefficient of
**−0.006 [−0.01, +0.00]** at 21d (−0.010 [−0.02, −0.00] at 42d) — essentially
zero, and *negative* if anything. Controlling for how much it recently fell, being
low-DPI adds **no** extra falling-knife probability.

The OOS split says the same: KEEP−ALL hit-rate improvement is ~0 in both halves,
and the HIGH−LOW big-loss gap flips sign (IS +1.08, OOS −0.79) — noise.

## The path test — "kept selling off" measured on the path, not the endpoint

A fair objection to everything above: those are **endpoint** statistics. A stock
that plunges −18% mid-window and claws back to −2% by day 21 looks harmless in
every endpoint table — yet it is exactly the knife that stops a real position
out. So the decisive metric is **forward max adverse excursion (MAE)**: the
*minimum* of the price path over the next h sessions (today excluded).

**21-day path profile:**

| group | mean MAE | P(path < −10%) | P(path < −15%) | hidden knives* | MAE 5th-pct |
|---|---:|---:|---:|---:|---:|
| **LOW** | −5.22 [−6.13, −4.55] | 16.3% | 7.0% | 1.4% | −16.9 |
| MID | −5.43 | 17.2% | 7.6% | 1.6% | −17.6 |
| **HIGH** | −5.39 [−6.44, −4.65] | 16.4% | 7.4% | 1.5% | −17.6 |

*\*dipped >10% intraperiod but ended positive — the knives the endpoint tables
cannot see.*

Differences: HIGH−LOW mean MAE **−0.19 [−0.43, +0.03]** (HIGH's paths are
marginally *worse*, grazing zero), P(path<−10%) +0.10 [−1.20, +1.25]. The
KEEP−ALL filter shifts mean MAE by **−0.05pp** — statistically resolvable,
economically nothing, and in the *wrong* direction. The 42-day window tells the
same story (LOW −7.62 vs HIGH −7.87 mean MAE; P(path<−10%) 28.2% vs 28.0%).

**The path test confirms the endpoint result rather than overturning it.**
Low-DPI downtrend names do not sell off harder *along the way* either — their
intraperiod drawdowns, deep-dip probabilities, and hidden-knife rates are
indistinguishable from (if anything a touch milder than) high-DPI names. The
absolute danger is worth staring at — **~17% of downtrend names dip more than
10% at some point in the next month, ~29% within two months** — but DPI does not
tell you which.

## What this means

The danger is real in absolute terms — **~9% of downtrend names drop another >10%
over the next month (14% at two months), with a 5th-percentile around −13% to
−17%.** But **DPI gives you no help distinguishing which ones.** The downtrend
bounce is remarkably uniform across dark-flow: knives and bouncers look identical
on DPI, whether you read a single print, the 10-day average, or a streak.

Combined with the prior studies, this closes both sides of the single-stock
question: dark flow does not help pick the **winners** (selection alpha ≈ 0 after
beta/reversal control) *and* does not help avoid the **losers** (no falling-knife
discrimination). The tradeable dark-flow signal remains **systematic / index-level**
(option B, on the dashboard's index tabs), not a single-stock screen.

## Caveats

- ~~"Keep selling off" is proxied by the endpoint distribution~~ — resolved: the
  path-based MAE analysis above measures intraperiod drawdowns directly (daily
  closes; intraday lows would be deeper still, but symmetrically across groups).
- **Downtrend = 3-month OLS slope < 0**; a different dip definition could shift
  composition, but the earlier 1–6-month downtrend sweep was also flat.
- **Survivorship** — current S&P 500 members; the CAPM-alpha and within-downtrend
  framings remove the level/tilt bias, and a delisting would show up as a big loss
  (so if anything survivorship *understates* falling-knife risk equally across DPI).
- Machinery unit-tested against a planted fat left tail (`tests/test_falling_knife.py`).
