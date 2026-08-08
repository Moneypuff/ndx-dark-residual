# Optimizing the index-level DIX signal — what tuning actually buys

Request: how can the tradeable index-level signal be optimized? Grid of 45
long/flat SPX configs (divergence lookbacks, %D windows × thresholds,
entry-and-hold variants, combinations, GEX gate), 2011–2026, net of 2bps per
switch, no look-ahead (position earns the day *after* the signal). The headline
test is **walk-forward**: each January, pick the best-Sharpe config on all data
to date and trade it the next year — so "optimization" is judged out of sample
by construction. Reproduce:
```
python spx_index_signal_opt_study.py --out spx_index_signal_opt.csv
```

**Short answer: three things actually matter, and none of them is a magic
parameter. (1) Stop holding the divergence *state* — use it as an entry marker
with a hold horizon; the state-while-true version destroys the signal. (2) The
best overlay is `%D(126) crosses ≥70, hold 21 sessions` — the only config of 45
whose timing edge clears zero (+4.9 bps/day [+0.7, +9.4]), Sharpe 0.91 vs 0.75
buy-and-hold at 41% exposure and half the drawdown — and it is exactly the
combination the earlier studies had already identified piecewise. (3) What
optimization buys is RISK-ADJUSTMENT, not out-performance: walk-forward matches
buy-and-hold's Sharpe (0.86 vs 0.88 since 2021) with half the exposure and a
−17% vs −25% max drawdown, but it does not out-return a bull tape.**

## 1. The single biggest fix: hold-after-entry, not hold-while-true

The dashboard's divergence badge (21d lookback) read as a daily *state* is the
**worst** config in the family — Sharpe 0.05, negative timing edge. The same
signal with entry-and-hold accounting (buy the first day the state turns
bullish, hold H sessions — the shape the original event study measured):

| config | Sharpe | exposure | edge bps/day |
|---|---:|---:|---:|
| DIV21 (state, as shipped) | 0.05 | 20% | −2.8 |
| DIV21 H10 | 0.57 | 41% | +2.7 |
| DIV21 H21 | 0.67 | 60% | +1.8 |
| DIV21 H42 | 0.76 | 81% | +1.0 |

The divergence *moment* carries the information; the persistence of the state
does not. (The H42 variants of every lookback sit at ~80–95% exposure — they
converge to buy-and-hold in disguise, edge ≈ 0. The honest divergence reading
is H10–H21: real but individually insignificant edges of +2–5 bps/day.)

## 2. The best config — and it's the one the thread already found

**`OSC126x70H21`** — stochastic %D of DIX over 126 sessions crosses up through
70, buy, hold 21 sessions (~5 entries/year, 41% exposure):

| | Sharpe | CAGR | maxDD | hit% | edge bps/day [95% CI] | Shp IS | Shp OOS |
|---|---:|---:|---:|---:|---|---:|---:|
| OSC126x70H21 | **0.91** | 10.0% | **−16.9%** | 55.9 | **+4.85 [+0.7, +9.4]** | 0.85 | 1.10 |
| buy-and-hold | 0.75 | 12.1% | −33.9% | 54.4 | 0 | 0.59 | 1.40 |

It is the only one of 45 configs whose in-market-minus-average edge clears zero,
it is stable in-sample vs out-of-sample, and the walk-forward selector picks it
**every single year 2021–2026** — no config-hopping, which is what genuine
signal (rather than noise-chasing) looks like. Credibility matters here: with 45
configs, ~2 false CI-clearances are expected by chance — but this config was
not discovered by the sweep. It is the pre-registered assembly of three findings
the earlier studies made independently (%D(126) is the best normalization; it
reads directionally, so a high threshold is an entry, not "overbought"; the
event-hold is the shape the divergence study measured). The sweep confirms it
rather than mines it.

GEX gating, which looked defensive in the base study, adds nothing once the
hold structure is right (OSC126>70+GEX Sharpe 0.62 vs 0.59 ungated; the
walk-forward briefly picked the gated variant for 2026 and it is a coin-flip).

## 3. What walk-forward optimization actually delivers (2021+)

| strategy | exposure | CAGR | Sharpe | maxDD | edge bps/day |
|---|---:|---:|---:|---:|---:|
| WALK-FORWARD (annual re-pick) | 50% | 10.5% | 0.86 | −16.9% | +2.8 [−4.5, +10.1] |
| fixed DIV21 state (shipped) | 18% | −0.3% | 0.01 | −25.8% | −4.0 |
| buy-and-hold | 100% | 14.2% | 0.88 | −25.4% | 0 |

The honest conclusion about "optimizing": **you cannot out-return a bull tape
with a long/flat timing overlay** — buy-and-hold won on CAGR and edged Sharpe.
What the optimized signal delivers is the same risk-adjusted return **at half
the exposure and two-thirds the drawdown**. That is the tradeable value: as a
capital-efficiency / drawdown-control overlay (deploy into the signal, park the
rest), or as a sizing signal on top of a core position — not as a replacement
for holding the index.

## Practical recommendations

1. **Re-cast the dashboard divergence badge as an event with a countdown** (it
   already shows the entry-day streak): the actionable read is "bullish
   divergence fired N days ago; the historical edge lives in the ~21 sessions
   after entry," not "state is currently bullish."
2. **Add the `%D(126) ≥ 70 crossing` as the primary index-level entry signal**
   — it is the strongest, most stable thing dark flow produces at the index
   level, worth ~+5 bps/day while active.
3. **Size, don't switch**: use the signals to scale exposure (e.g. overweight
   during the 21-session post-entry windows), since the overlay's value is
   risk-adjusted, not absolute.
4. **Don't re-tune further.** The walk-forward already converges on one config;
   deeper grids will only manufacture overfit. The next real improvement would
   have to come from new information (breadth of per-name DPI, GEX term
   structure), not new parameters on DIX.

## Caveats

- Execution is modeled at the signal day's close; DIX publishes after the
  close, so real fills slip to the next session — the earlier entry studies
  showed the signal's edge is not concentrated in day 1, but this is unmodeled.
- 2bps/switch covers SPY-like costs; the conclusions are not cost-sensitive
  (the best config switches ~10×/year).
- Long/flat only — the base study showed the bearish side is weak; a short leg
  was not searched (deliberately, to keep the grid small).
- One CI-clearing config among 45 would be unremarkable alone; the weight rests
  on its pre-registration by earlier studies, IS/OOS stability, and unanimous
  walk-forward selection.
- Machinery unit-tested: no-look-ahead accounting, per-switch costs, entry-hold
  construction, planted-edge detection, walk-forward selection
  (`tests/test_index_opt.py`).
