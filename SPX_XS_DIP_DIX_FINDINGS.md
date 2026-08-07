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
- **Dip** = trailing 21-session return < 0 (a one-month pullback). 43% of the
  panel qualifies.
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
