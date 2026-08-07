# Cross-sectional "buy the dip on high DIX" over the S&P 500 — a null

Does the index-level option-B signal (dark flow rising into a price dip → above
-average forward returns) **generalize to single stocks**? I.e. across the whole
S&P 500, is *buying a stock's dip when its own dark-flow is high* a cross-sectional
edge — **no matter the stock**?

**Short answer: no.** On 499 names × 1,769 days (**850,689 name-day observations,
Jun 2019 → Jul 2026**), high per-name dark flow does **not** improve dip-buying.
If anything the *darkest* dips slightly **underperform** the least-dark dips, and
fewer than half of individual names show any positive edge. The index-level
divergence signal does not survive the move to single-stock stock-selection.

Reproduce:
```
python spx_xs_dip_dix_study.py --start 2019-01-01 --out spx_xs_dip_dix.csv
```

## Definitions (all real-time, no look-ahead)

- **Per-name D** = FINRA `ShortVolume / TotalVolume` (consolidated off-exchange),
  5-day MA — the single-stock analog of DIX (`build_universe_panels` in
  `ndx_dark_residual.py`, ~500 IVV constituents).
- **"High DIX" for a stock**, two lenses: **self-relative** = D's percentile
  within the name's own trailing 252 sessions (primary; "unusually dark vs its
  own norm"); **cross-sectional** = D ranked across all names that day ("the
  darkest names today", cross-check). High/low = top/bottom quintile.
- **Dip** = a recent pullback. Two definitions, both tested: the simple
  **trailing 21-session return < 0** (the 2×2/decile tables just below), and — the
  stronger one — a **downtrend of duration W** (OLS log-price slope over the
  trailing W sessions < 0), with **W swept 1 → 6 months** (see the follow-up
  section). The conclusion is the same under both.
- **Outcome** = 21-session forward return, **cross-sectionally demeaned** (minus
  the universe's daily mean) so it measures *stock selection*, not market beta;
  a market-excess-vs-SPY twin is also reported and agrees.
- **Inference** = a **daily portfolio** (each day form the equal-weight basket
  the rule holds → one return per day → moving-block-bootstrap 95% CI), because
  the 850k pooled obs are massively overlapping and cross-correlated. Plus a 2×2
  interaction, decile-within-dips, a per-name breakdown, and a regime split.

## The result — flat everywhere (primary, self-relative signal)

**2×2** (cross-sectionally demeaned forward return, pp):

| cell | n | mean | hit |
|---|---:|---:|---:|
| high-D & dip | 78,243 | **−0.00** | 47% |
| low-D & dip | 85,228 | **+0.19** | 49% |
| high-D & no-dip | 125,666 | −0.05 | 48% |
| low-D & no-dip | 83,229 | +0.10 | 48% |

Interaction `(highD−lowD | dip) − (highD−lowD | no-dip)` = **−0.03 pp** — zero.
Note the wrong-way tilt: among dips, **low-D did marginally better than high-D**.

**Decile of D within dipped names** — no monotonicity; long-short (darkest −
least-dark dips) = **−0.26 pp**.

**Daily portfolio** (1,767 days):

| leg | mean/day | hit | 95% CI |
|---|---:|---:|---|
| long-short (darkest − least-dark dips) | **−0.19%** | 46% | [−0.49, +0.13] |
| long-only (darkest dips vs universe) | −0.01% | 47% | [−0.29, +0.27] |

**"No matter the stock"** — per-name edge (high-D dips vs each name's other
dips): **47.1% of 497 names positive** (sign-test p=0.19), median edge −0.12 pp.
Fewer than half of stocks benefit; the median name is slightly negative.

**Regime split** (long-short mean/day): pre-2021 −0.24% · 2021+ −0.18%. No era
rescues it.

**Robustness.** The cross-sectional-rank signal tells the same story (long-short
−0.16 pp, interaction +0.12 pp, per-name 44.9% positive). The market-excess
(vs-SPY) outcome agrees with the cross-sectional-demean outcome throughout
(`spx_xs_dip_dix.csv`): every long-short CI straddles zero.

## Follow-up — "dip" redefined as a downtrend, swept 1 → 6 months

A single-point trailing return is a weak proxy for a *dip*. Redefine it properly:
a name is **in a downtrend of duration W** when the OLS slope of its log price
over the trailing W sessions is negative (`trend_slope` in the study), and
**sweep W across 1, 2, 3, 4, 6 months**. Daily-portfolio long-short (darkest −
least-dark names *in that downtrend*), cross-sectionally demeaned outcome:

| downtrend | self-relative D L/S | 95% CI | interaction | names +% | x-sect D L/S | 95% CI |
|---|---:|---|---:|---:|---:|---|
| 1-month | −0.19% | [−0.50, +0.14] | −0.08 | 48% | −0.07% | [−0.44, +0.30] |
| 2-month | −0.06% | [−0.39, +0.28] | +0.08 | 49% | −0.03% | [−0.43, +0.37] |
| **3-month** | +0.02% | [−0.32, +0.39] | **+0.20** | 50% | +0.10% | [−0.30, +0.53] |
| 4-month | −0.09% | [−0.45, +0.28] | +0.04 | 48% | −0.05% | [−0.46, +0.36] |
| 6-month | −0.06% | [−0.43, +0.33] | +0.16 | 48% | +0.09% | [−0.35, +0.57] |

**The null is robust to the redefinition and to every duration.** Every
long-short 95% CI straddles zero, for both the self-relative and cross-sectional
signal; the point estimates are all within ±0.2%/month of zero; the per-name
fraction positive stays ~48–50% (a coin flip) at every W; and the regime split
is flat too (pre-2021 and 2021+ CIs both straddle zero for all durations,
`spx_xs_dip_dix.csv`). The long-only "darkest downtrenders vs universe" leg is
similarly indistinguishable from zero (+0.05% to +0.19%/mo, CIs include zero).

The one faint pattern worth naming honestly: the **interaction** (does high-vs-
low D pay more inside a downtrend than outside?) is mildly positive at the
**longer** durations (3-month +0.20 pp, 6-month +0.16 pp) and slightly negative
at 1-month — i.e. *if* dark flow ever carries dip-buying information, it is in
**longer, more established downtrends**, not short pullbacks. But it is a whisper
(the 3-month detail 2×2 still shows low-D marginally *ahead* of high-D, +0.27%
vs +0.23%, and the decile ladder within downtrends is non-monotone), nowhere
near significant, and does not turn into a tradeable long-short. It is a
direction to remember, not an edge to trade.

## Why the index signal worked but this doesn't

Option B fired at the **index** level because a high, rising DIX is a
**market-wide, systematic** read — broad dark accumulation is SqueezeMetrics'
risk-on tell, and "DIX up while SPX down" caught the whole market being bought
into weakness. That is a *market-timing* signal.

A single stock's high dark ratio is a different animal. Per-name off-exchange
short volume is dominated by **wholesaler/market-maker internalization of retail
flow and hedging** — mechanics of *how* the stock trades, not a forecast of
*that stock's* idiosyncratic future. Once you demean out the market (which is
exactly what a cross-sectional / stock-selection test must do), you remove the
systematic component that made option B work and are left with the idiosyncratic
part, which carries no dip-buying edge. **The signal is systematic, not
cross-sectional** — so it belongs on the index tabs (where it now lives), not as
a single-stock screen.

## Caveats

- **Survivorship.** The universe is *today's* ~500 IVV members, so names that
  left the index (often the losers) are absent — a mild upward bias on raw
  returns. The cross-sectional demean removes the level bias; only a bias
  *correlated with the dark-flow signal itself* would distort the null, and none
  is apparent. Still, treat the pre-2021 slice (2 years, fewer names warmed up)
  as thin.
- **Overlap.** 850k pooled obs ≈ far fewer independent ones; the daily-portfolio
  bootstrap CIs and the per-name sign test are the honest lenses, not the pooled
  cell means.
- **One definition of "dip."** Trailing-21d return < 0. A sharper dip (e.g. near
  a 3-month low, or an RSI trigger) or a different horizon could be probed, but
  the flatness across the 2×2, deciles, daily portfolio, per-name, and both
  signal lenses makes a hidden strong effect unlikely.
- **Not wired into the dashboard** — by design: there is no signal to surface.
  The machinery is unit-tested against a planted effect (`tests/test_xs_dip.py`)
  to confirm the null is real, not a bug.
