# GEX-regime quintiles × DIX, on vol-adjusted SPX forward returns

Request: instead of conditioning DIX on a *price* downtrend, condition on the
**GEX regime** — quintiles of GEX with the lowest = the negative-gamma regime,
highest = ultra-high positive gamma — and vol-adjust the forward returns (both
by trailing 21d realized vol and by VIX). 2012–2026, 3,641 sessions; GEX
bucketed by trailing-1-year percentile (no look-ahead, handles GEX's secular
growth); DIX read as %D(126) bands; outcomes in **sigma units**
(`fwd / (vol·√h)`). Reproduce:
```
python spx_gex_regime_study.py --out spx_gex_regime.csv
```

**Short answer: the premise is right — GEX quintile is a clean, monotone
downtrend/stress regime variable, and the low/negative-gamma regime pays a real
premium even per unit of volatility. But the hoped-for interaction runs
backwards: *within* the negative-gamma regime DIX adds nothing (IC ≈ 0, even
slightly negative) — in that regime everything bounces and dark flow doesn't
rank the days. What works is the stack, not the interaction: the GEX-Q1/Q2
regime supplies the fat expected bounce, and the %D(126)≥70 crossing supplies
the timing (n=73 entries: +2.52% fwd 21d, 75% hit, +0.46σ).**

## 1. Premise check — GEX quintile *is* the downtrend variable

| GEX Q | n | %GEX<0 | % in 3-mo downtrend | raw fwd 21d | zrv (σ) | zvix (σ) |
|---:|---:|---:|---:|---|---:|---:|
| **1 (neg regime)** | 689 | 41.4 | **46.7** | **+1.99 [+0.92,+3.13]** | **+0.391** | **+0.270** |
| 2 | 610 | 5.2 | 32.5 | +1.38 | +0.284 | +0.230 |
| 3 | 656 | 0.0 | 22.1 | +1.05 | +0.271 | +0.194 |
| 4 | 678 | 0.0 | 14.9 | +0.55 | +0.159 | +0.120 |
| **5 (ultra-high)** | 1008 | 0.0 | **8.0** | +0.67 [+0.14,+1.16] | +0.221 | +0.156 |

Downtrend share falls monotonically 47% → 8% across the quintiles — GEX regime
and price trend are strongly linked exactly as suspected, and the trailing
bucketing puts ~90% of all negative-GEX days in Q1 (GEX<0 is only 8.7% of all
days). The forward-return gradient is monotone Q1→Q4 in raw terms **and
survives both vol adjustments**: that matters, because low-gamma regimes are
high-vol regimes, and the raw +1.99% could have been "just more vol." Per unit
of risk, Q1 still pays ~1.8× Q5 (+0.39σ vs +0.22σ at 21d; +0.50σ vs +0.35σ at
42d; sign split: GEX<0 +0.42σ vs GEX≥0 +0.25σ). Caveat: adjacent-bucket CIs
overlap — the evidence is the monotone gradient across five buckets, two vol
scalings and two horizons, not any single pairwise test.

## 2. The heatmap — and why the interaction hypothesis fails

Mean sigma-unit fwd 21d, GEX quintile (rows) × %D(126) band (cols):

|  | D1 (dark-light) | D2 | D3 | D4 | D5 (dark-heavy) |
|---|---:|---:|---:|---:|---:|
| **Q1** | **+0.67** | +0.40 | +0.28 | +0.46 | +0.44 |
| Q2 | — | +0.19 | +0.26 | +0.38 | +0.28 |
| Q3 | −0.30 | +0.17 | +0.32 | +0.41 | +0.23 |
| Q4 | +0.27 | −0.10 | +0.22 | +0.44 | +0.03 |
| Q5 | +0.19 | +0.08 | +0.31 | +0.41 | −0.03 |

Read the rows: in Q1 there is **no left-to-right gradient** — even the
lowest-DIX cell bounces (+0.67σ, its best cell in fact, small n=32). The formal
version, Spearman IC of %D vs σ-adjusted forward return *within* each regime:

| GEX Q | IC 21d [95% CI] | IC 42d |
|---:|---|---:|
| 1 | **−0.05 [−0.21, +0.12]** | −0.11 |
| 2 | +0.02 | −0.01 |
| 3 | +0.07 | +0.15 |
| 4 | +0.12 [−0.06, +0.26] | +0.09 |
| 5 | +0.13 [−0.04, +0.29] | +0.17 |

**The hypothesis was that DIX would fit best in the low-GEX regime; the data
says the opposite.** In the negative-gamma regime the bounce is regime-wide —
vol is spiking, dealers are short gamma, mean-reversion kicks in regardless of
what dark pools did — and DIX cannot rank the days (if anything, faint
contrarian at 42d). What faint ordering DIX retains lives in the *calm*
mid/high-gamma regimes (Q3–Q5, IC +0.07..+0.17, none individually clearing
zero). So GEX-Q1 *replaces* rather than *sharpens* the DIX read: when gamma is
negative, the regime itself is the signal.

## 3. What does combine: regime + entry timing (stacking, not interaction)

The optimized entry (%D(126) crossing ≥70, hold 21) grouped by the GEX regime
on the entry day:

| regime at entry | n | raw fwd 21d | hit | zrv (σ) |
|---|---:|---|---:|---|
| **Q1–2 (low/neg)** | 73 | **+2.52% [+1.55, +3.53]** | **75%** | **+0.461 [+0.30, +0.62]** |
| Q4–5 (high) | 30 | +1.25% | 77% | +0.385 |

Most %D entries fire in low-gamma regimes anyway (73 vs 30 — dark buying into
weakness and low gamma co-occur), and those entries carry both the regime
premium and the timing edge. This also **refines the earlier "GEX>0 gate is
defensive" finding: the gate was cutting off the best raw days.** Per sigma the
entry works in both regimes; in raw terms the low-gamma entries pay twice as
much because they're taken where vol (and the regime premium) is.

## Verdict

- **Adopt the reframing**: GEX quintile (trailing-percentile, Q1 = negative
  regime) is a better-behaved conditioning variable than the price downtrend —
  forward-looking, monotone in both downtrend share and vol-adjusted forward
  return, and it cleanly isolates the negative-gamma bounce regime.
- **But drop the interaction hope**: DIX does not get sharper in the low-GEX
  regime — it goes mute there. The tradeable structure is a **stack**: GEX Q1/Q2
  tells you the tape pays a bounce premium (even per unit of risk); the %D(126)
  crossing tells you *when* within it. Don't spend the DIX read trying to rank
  days inside Q1.
- Dashboard-ready version if wanted: show the current GEX quintile as a regime
  chip next to the %D/divergence badges (Q1–2 = "bounce regime, signals carry
  extra weight in raw terms"), rather than gating signals off GEX sign as the
  earlier defensive framing suggested.

## Caveats

- ~2012–2026 spans one secular bull; the Q1 premium is partly "buying stress in
  a market that always recovered." The vol adjustment handles risk-per-day, not
  regime survivorship.
- Cell-level heatmap n's get small at the corners (D1/D5 in any quintile:
  30–120 days, overlapping windows); read rows and the IC summary, not corner
  cells.
- GEX and VIX and realized vol are all entangled (negative gamma ⇒ higher vol);
  the two vol scalings agree, which is the point of running both.
- Entry-by-regime buckets share the entry-day overlap discipline (first-day
  crossings only) but n=30 in the high bucket is thin.
- Machinery unit-tested: trailing-percentile bucketing (Q1 captures the
  negative regime; today excluded from its own rank), vol-adjustment
  arithmetic, planted DIX-only-in-Q1 interaction recovered by both the IC and
  the heatmap, flat on null (`tests/test_gex_regime.py`).
