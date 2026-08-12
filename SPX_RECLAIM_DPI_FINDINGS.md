# "GOOGL-like reclaim": MA reclaim + persistent high DPI as a bottoming signal

Motivating observation: a name rallies earlier in the year, pulls back into a
*mild* downtrend, its **DPI rises sharply and stays in the upper deciles for a
few weeks**, then it **reclaims its moving average** and rallies again. Does that
specific bottoming/reclaim pattern generalize across the S&P 500 — and, crucially,
does the persistent-high-DPI condition **add anything** to a plain MA reclaim?

**Short answer: the reclaim catches bounces, but DPI adds ~nothing.** A moving-
average reclaim inside an uptrend does produce positive *raw* forward returns
(it's a momentum/bounce entry) — but that is market beta: its **excess vs SPY is
~0**, and layering on the "persistent high DPI" condition does **not** improve
the mean: the DPI marginal is **negative at every horizon**, with fewer than half
of names positive. Reframing the question from "bigger winner" to **"does the
bounce stick"** (the conviction angle — see that section) is kinder to the idea
but still thin: the only thing that clears its CI is a **~2 pp higher chance the
name is still above its MA one month out**; give-back depth and blow-up rate are
unchanged, and the edge fades by 2–3 months. The GOOGL case was a good anecdote,
not a repeatable cross-sectional edge.

Reproduce:
```
python spx_reclaim_dpi_study.py --start 2019-01-01 --out spx_reclaim_dpi.csv
python spx_reclaim_dpi_study.py --start 2019-01-01 --reclaim-ma 20   # robustness
```

## Entry rule (real-time, no look-ahead)

A **reclaim event** on day t requires all of:
1. **Reclaim** — close crosses back above its 50-day MA today.
2. **Uptrend** — close > its 200-day MA ("rallied earlier"; a pullback within an
   uptrend, not a broken name).
3. **Real dip** — it had been below the 50-day MA for ≥ half of the prior 10
   sessions (a genuine multi-week pullback, not one-day chop).

Events are split by the dark-flow condition:
4. **DPI-hi** — the name's self-relative DPI percentile (within its own trailing
   year) sat in the top quintile on ≥ half of the last 15 sessions ("DPI rose and
   stayed in the upper deciles for a few weeks").

- **R_dpi** = reclaims with persistent high DPI (the requested setup)
- **R_nodpi** = reclaims without it (plain reclaim — the control that isolates
  what the DPI condition is worth)

Events de-overlapped with a 21-session per-name cooldown; scored on 21/42/63-day
forward returns, raw / excess-vs-SPY / cross-sectionally demeaned; block-bootstrap
CIs, a per-name sign test, and a regime split. Universe: 502 S&P 500 names,
2019–2026 (`spx_xs_dip_dix_study.load_universe`).

## Result (primary, 50-day reclaim)

De-overlapped events: **R_dpi = 2,211 · R_nodpi = 7,956**.

| horizon | group | raw | raw hit | excess-SPY | 95% CI | names +% |
|---|---|---:|---:|---:|---|---:|
| 21d | R_dpi | +1.72% | 58% | **+0.02%** | [−0.55, +0.60] | 48.5% |
| 21d | R_nodpi | +1.59% | 58% | +0.23% | [−0.08, +0.56] | 51.4% |
| 42d | R_dpi | +2.91% | 60% | +0.22% | [−0.47, +0.93] | 47.9% |
| 42d | R_nodpi | +2.64% | 59% | +0.31% | [−0.16, +0.73] | 48.0% |
| 63d | R_dpi | +4.27% | 61% | +0.19% | [−0.67, +1.05] | 45.7% |
| 63d | R_nodpi | +4.15% | 60% | +0.38% | [−0.17, +0.94] | 44.4% |

**DPI marginal (R_dpi − R_nodpi), excess-vs-SPY: −0.21 / −0.09 / −0.19 pp** at
21 / 42 / 63 days — negative throughout.

Reading it:

- **The reclaim is a real bounce entry — in raw terms.** +1.7% to +4.3% forward,
  58–61% hit. But those raw numbers are mostly beta: a stock reclaiming its 50-day
  in an uptrend tends to keep rising *with the market*.
- **On a market-neutral basis it's a wash.** Excess-vs-SPY is +0.0 to +0.4% for
  both groups with CIs through zero; cross-sectionally demeaned is ~0 or slightly
  negative. The reclaim doesn't select *which* stock beats the market.
- **The DPI confirmation does not help.** Persistent high DPI *subtracts* a
  little at every horizon, and the DPI branch has **< 50% of names positive**
  (48.5 / 47.9 / 45.7%) — at 63d the sign test is p≈0.06 in the *wrong*
  direction. So it isn't just "no edge"; adding the dark-flow filter mildly
  worsens a plain reclaim.
- **No regime rescues it** (R_dpi 21d excess): pre-2021 +0.41% [−0.53, +1.24],
  2021+ −0.06% [−0.67, +0.58].

## Does the bounce *stick*? (the conviction reframe)

The test above asks whether high DPI picks the *bigger winner* (mean return) —
and it doesn't. A different question: conditional on the bounce already being
underway (the reclaim fired), does persistent high DPI make it **hold** rather
than fail — fewer whipsaws back below the MA, shallower give-back, fewer
blow-ups? Path metrics over the next N sessions, R_dpi vs R_nodpi (by-name
cluster-bootstrap 95% CI on the difference):

| metric (forward 1mo) | R_dpi | R_nodpi | diff [95% CI] |
|---|---:|---:|---|
| whipsaw back below MA % | 67.6 | 69.2 | −1.5 [−3.7, +0.6] |
| still above MA at N % | 68.1 | 65.8 | **+2.2 [+0.2, +4.2]** |
| blow-up (<−10%) % | 14.9 | 14.8 | +0.2 [−1.4, +1.7] |
| mean give-back % | −4.8 | −4.8 | −0.1 [−0.3, +0.2] |

**Weak, inconsistent support — not the clean "high DPI makes bounces stick"
signal you'd hope for.** The honest read across 1/2/3-month windows:

- **The one thing that clears its CI:** at **1 month**, high-DPI reclaims are
  **~2.2 pp more likely to still be above the reclaimed MA** (68.1% vs 65.8%,
  CI [+0.2, +4.2]). A small short-horizon "hold" edge. It fades by 2–3 months
  (+1.9 pp then ~0).
- **Whipsaw rate** is directionally lower for R_dpi at every window (−1.2 to
  −1.5 pp) but never significant (all CIs include zero), and the per-name paired
  test *contradicts* it at 3 months (only 35.8% of names stickier, sign-p≈0 —
  i.e. most names' high-DPI reclaims whipsaw *more* over a quarter).
- **Give-back depth and blow-up rate: no difference** — high DPI does *not*
  reduce the size of the pullback or the >10% failure rate (blow-ups are if
  anything a touch higher).
- **Per-name give-back** leans faintly toward high-DPI being shallower
  (~54% of names, sign-p≈0.06–0.08) — a whisper, not a result.

So reframing from "bigger winner" to "holds better" does move the needle off the
clearly-negative mean-return marginal to **roughly flat with a faint, short-lived
hold edge** (~2 pp higher 1-month above-MA rate). That is the most that survives
— it is not enough to build a conviction filter on, and it does not persist past
a month or reduce the real failure mode (deep give-backs / blow-ups).

## Why the anecdote didn't generalize

The GOOGL episode is a textbook example of the pattern — and examples like it
exist. But a single vivid case is exactly what a systematic test is for. Across
2,211 comparable setups the persistent-high-DPI condition carried no forward
information beyond the price action itself. This is the same mechanism the
downtrend and dip studies found: **per-name DPI is dominated by wholesaler /
market-maker internalization of retail flow, not informed accumulation**, so it
does not sharpen a single-stock timing rule. Dark-flow's edge lives at the
**index** level (systematic dark buying → market direction), where option B is
already surfaced on the dashboard — not as a single-stock bottoming filter.

The honest, usable takeaway: **the MA-reclaim-in-uptrend is a fine momentum/bounce
entry on its own** (positive raw returns, ~60% hit), but treat it as beta capture,
and don't expect the DPI overlay to add stock-selection alpha.

## Robustness & caveats

- **20-day reclaim** (`spx_reclaim_dpi_ma20.csv`) — reported for the "reclaimed
  *moving averages*" (plural) framing — tells the same story more sharply: the
  DPI marginal is **−0.33 / −0.54 / −0.60 pp** at 21 / 42 / 63d and R_dpi's own
  excess-vs-SPY turns negative (~−0.3%), with only ~43–44% of names positive. On
  a faster MA the DPI overlay hurts a plain reclaim more, not less.
- **Overlap & clustering.** Reclaims fire market-wide after market dips, so the
  pooled CIs and the per-name sign test are the honest lenses; the R_dpi − R_nodpi
  *difference* is the cleanest read since market-timing bias hits both branches
  equally.
- **Survivorship** — today's S&P 500 members only (names that left the index are
  absent). The cross-sectional-demean / excess-vs-SPY outcomes remove the level
  bias; a bias would have to correlate with the DPI condition to distort the
  marginal, and none is apparent.
- **Parameter choices** were pre-specified (50/200-day MAs, top-quintile DPI on
  ≥ half of 15 sessions, 10-day prior-dip, 21-day cooldown) to avoid fitting the
  rule to one name; the flatness across horizons, both MAs, per-name, and regimes
  argues against a strong effect hiding in a nearby parameterization.
- **Event mechanics are unit-tested** (`tests/test_reclaim_dpi.py`): a synthetic
  dip-then-reclaim fires exactly one event, DPI persistence routes it correctly,
  a below-200-day downtrend fires none, and the cooldown de-overlaps repeats.
