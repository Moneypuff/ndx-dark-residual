# SPX raw-DIX: decile (level) vs. trailing-1-year breakout/breakdown

Two questions on the **raw** S&P 500 Dark Index (SqueezeMetrics' published
`DIX.csv` — the actual series people quote when they say "DIX is at 45%", not
this repo's reconstructed dollar-DIX):

1. **Which framing predicts SPX forward returns better** — the DIX **decile**
   (level within its own history) or a trailing **1-year breakout/breakdown**
   (today's DIX vs. its own 252-session average)?
2. **The "45% floor" puzzle** — a 45% print was rare and bullish around 2019
   but looks like the *floor* in 2024–today. **Did something break?**

Reproduce:
```
python spx_dix_decile_vs_breakout_study.py                 # live fetch of DIX.csv
python spx_dix_decile_vs_breakout_study.py --csv DIX.csv    # offline snapshot
python spx_dix_decile_vs_breakout_study.py --out spx_dix_decile_vs_breakout.csv
```
Data: `https://squeezemetrics.com/monitor/static/DIX.csv` (`date, price, dix,
gex`), **2011-05-02 → present**. Forward returns use the file's own SPX `price`
column at 21 / 42 / 63 sessions (no look-ahead in the outcome). All signals are
constructed to be knowable in real time; the head-to-head runs on the common
sample where all four are defined (**2012-04-30 → present, ~3,566 sessions**).

Definitions (all real-time):
- **Expanding decile** — DIX ranked 1–10 within *all* of its own history to date
  (min 252 obs). The long-memory "level" signal.
- **Trailing-1y decile** — DIX ranked 1–10 within only the last 252 sessions.
  A fully de-trended "where does today sit within the past year".
- **Breakout z (1y)** — `(DIX − mean₂₅₂) / std₂₅₂`. Trailing-year sigmas above
  (breakout) / below (breakdown) the DIX's own 1-year average. This *is* the
  "vs a trailing 1-year average" signal.
- High DIX = the bullish reading (SqueezeMetrics' convention), so a **positive**
  signal→return relationship is the hypothesis throughout.

---

## Q2 — nothing broke in the data; the *level drifted*, and then the *signal faded*

**The raw DIX is strongly non-stationary.** The off-exchange share of volume
stepped up structurally, so the whole DIX distribution slid upward:

| era | median DIX | % of days ≥ 45% | where a fixed **45%** ranks that year |
|---|---:|---:|---|
| 2013–2019 | **~41–42%** | 0–14% | **86th–100th pct** — a genuine top-decile outlier |
| 2020–2021 | ~43% | 14–31% | ~69th–87th pct |
| 2022 | 47.1% | 75% | 25th pct |
| **2023–2025** | **~46–48%** | **76–96%** | **4th–24th pct — now the floor** |

Linear drift is **+0.42 pp/year** (fitted 40.4% in 2011 → 46.8% in 2026). Your
memory is exactly right: in 2019 a 45% DIX sat at the **88th percentile** of the
year (rare, top-decile → bullish); by 2023 it sat at the **3.6th percentile**
(basically the annual low). The step-change is **2021→2022** — median jumps
42.7% → 47.1% and "% of days ≥ 45%" jumps 14% → 75%.

So a **fixed 45% "bullish" rule broke** — but only because the *ruler* stayed
still while the *level* moved. That part is fixable: rank the DIX against its own
recent history (decile / percentile / z-score) instead of an absolute cutoff and
the drift washes out.

**But there is a second, deeper break that de-trending does *not* fix.** Split
the sample at the regime shift and score each real-time signal's Spearman IC
(1-month forward return):

| period | expanding-decile IC | trailing-1y-decile IC | breakout-z IC |
|---|---:|---:|---:|
| **pre-2021** (n=2,184) | **+0.196** | **+0.142** | **+0.137** |
| **2021+** (n=1,382) | **+0.022** | **−0.037** | **−0.038** |

The DIX→forward-return relationship — in *every* framing, de-trended or not —
went from clearly positive to roughly zero (slightly negative) in the current
regime. So "did something break?" has two honest answers:

1. **The scale drifted** (cosmetic, expected): more volume prints off-exchange,
   so 45% is no longer special. Use a relative framing.
2. **The information content decayed** (the real break): even a de-trended DIX
   has carried little next-month signal since ~2021. The most likely mechanism
   is the same one that lifted the level — the post-2020 explosion of 0DTE /
   index-option hedging and PFOF retail internalization means more of "short +
   off-exchange volume" is now mechanical dealer/retail flow and less of it is
   informed accumulation. The gauge measures a different mix than it did in
   2015.

---

## Q1 — the decile (level) beats the trailing-1-year breakout/breakdown

Head-to-head on the common real-time sample, **1-month (21-session)** forward
return:

| signal | Spearman IC | NW slope (pp/1sd) | t | top-band − bottom-band spread |
|---|---:|---:|---:|---:|
| **expanding decile (level)** | **+0.130** | **+0.38** | **+1.88** | **+1.09 pp** |
| trailing-1y decile | +0.072 | +0.18 | +1.04 | +0.52 pp |
| **breakout z (1y avg)** | +0.069 | +0.18 | +0.98 | +0.54 pp |
| breakout gap (1y avg) | +0.066 | +0.16 | +0.82 | +0.51 pp |

*(top/bottom band = deciles 9–10 vs 1–2 for decile signals, quintile 5 vs 1 for
the continuous ones; NW = Newey-West HAC, 21 lags.)*

The **level (decile) framing carries about twice the IC and twice the long-short
spread** of the trailing-1-year breakout/breakdown, and it is the only one whose
Newey-West t clears ~1.9. The pattern holds at 2-month and strengthens at
3-month, where the trailing-1y signals **flip to zero/negative** while the
expanding-level signal stays positive (see `spx_dix_decile_vs_breakout.csv`):

| horizon | expanding-decile IC | breakout-z IC |
|---:|---:|---:|
| 21d | +0.130 | +0.069 |
| 42d | +0.141 | +0.057 |
| 63d | +0.106 | **−0.007** |

**Does the breakout add anything on top of the level?** No. Joint regression
(both standardized, Newey-West):

```
r21 ~ LEVEL(exp decile) + BREAKOUT(1y z)
  LEVEL     +0.53 pp/1sd   t = +1.78
  BREAKOUT  −0.20 pp/1sd   t = −0.76      (n=3,566, R² = 0.9%)
```

The breakout term is small and **wrong-signed** once the level is in the model —
the trailing-1-year deviation contributes nothing the level had not already
said. Intuitively: what matters is *how dark the tape is relative to its recent
range* (a slow, persistent level), not *whether DIX just poked above its own
one-year average* (a fast change).

### Honest counting — entry events (de-overlapped)

Every-day averages over overlapping 21-day windows overstate precision, so score
only the **first day into each band**, 21-session cool-down:

| setup | high/up | low/down | spread |
|---|---|---|---:|
| **LEVEL** (into top-2 / bottom-2 expanding deciles) | n=81, **+2.00%**, hit 74% | n=50, +0.13%, hit 66% | **+1.86 pp** |
| **BREAKOUT** (into ±1σ of 1-yr average) | n=76, +1.34%, hit 70% | n=62, +0.58%, hit 71% | +0.76 pp |

Same verdict on an overlap-free basis: entering when DIX is **high within its own
history** paid ~+1.9 pp vs. low; a trailing-year **breakout** paid less than half
that, and its "breakdown" leg still had a 71% hit rate (a weak short signal).

### The subtlety that ties Q1 to Q2

The expanding decile's edge comes from the **slow, multi-year level**. Collapse
the window to one year — whether you call it a "trailing-1y decile" or a
"breakout z" — and the two become nearly identical (IC +0.072 vs +0.069) and
weak. And that extra juice in the long-window level is *exactly* what died after
2021. So:

- **Historically, level > breakout** — clearly, and out to 3 months.
- **Over a 1-year window the two methods are the same thing** (level-within-a-
  year), and both are weak.
- **In the current regime neither works** — the whole DIX→return edge has faded
  since the off-exchange step-up, which is the same event that broke the fixed
  45% rule you noticed.

---

## Follow-up — how fast should the breakout baseline be? (3-month vs 1-year)

The original breakout used a **1-year (252-session)** baseline. Would a faster
baseline "pick up signals faster"? Sweeping the baseline window (1-month forward
return, `spx_dix_decile_vs_breakout_window_sweep.csv`):

| baseline | Spearman IC | NW t | LS spread | pre-2021 IC | 2021+ IC |
|---|---:|---:|---:|---:|---:|
| 1-month | +0.021 | +0.69 | +0.27 | +0.048 | −0.022 |
| 2-month | +0.078 | +1.88 | +0.65 | +0.102 | +0.041 |
| **3-month** | **+0.096** | **+2.04** | +0.71 | +0.121 | +0.053 |
| **6-month** | **+0.124** | **+2.57** | +1.05 | +0.173 | +0.043 |
| 1-year (original) | +0.069 | +1.01 | +0.54 | +0.135 | −0.038 |

**There is an inverted-U in the baseline window, and the instinct is right: a
3-month baseline beats the 1-year one.** IC roughly doubles vs. the 1-month
baseline and rises from +0.069 → +0.096 vs. the 1-year, and the Newey-West t
crosses from insignificant (+1.01) to significant (+2.04). The peak is at
~**6 months** (IC +0.124, t +2.57) — which roughly matches the expanding-decile
*level* signal (+0.130). Read the ends of the curve as the two failure modes:

- **1-month is too fast** — it de-trends against last month only, so the
  "deviation" is mostly daily noise (IC +0.021, indistinguishable from zero).
- **1-year is too slow** — it de-trends away the very medium-term level that
  carries the signal.

So the predictive information in DIX lives at a **medium frequency (≈3–6 month
swings relative to a recent baseline)**. This holds and strengthens at the 2- and
3-month forward horizons (6-month baseline: r42 IC +0.151, r63 IC +0.118 — the
best cells in the whole study).

Two honest qualifiers:

1. **It still doesn't survive the regime.** Every window's **2021+ IC is ≤
   +0.05** (the 1-year even goes negative). The faster baseline recovers
   significance in the *full* sample — which is dominated by the strong pre-2021
   era — but does **not** resurrect a live edge in the current regime. The
   3-month is marginally the most alive recently (+0.053), which is something,
   but it is not a signal to bet on alone.
2. **"Rolling 3-month *average of the level*" is the wrong version of the idea.**
   Smoothing raw DIX with a 63-session MA *before* ranking it monotonically
   *lowers* IC (+0.130 raw → +0.040 with a 3-month MA), because smoothing lags
   and averages the signal away. "Faster" has to mean **a shorter breakout
   baseline**, not a smoothed level. (Contrast table in the script output.)

**Revised verdict on breakout:** with a **3–6 month** baseline the
breakout/breakdown is competitive with the level/decile framing — the earlier
"level clearly beats breakout" gap was an artifact of using an over-long 1-year
baseline. But the level (expanding decile) is still at least as good with less
tuning, adds the same medium-frequency information, and — like every framing
here — has faded since 2021.

## Follow-up — normalizing DIX into an oscillator (+ GEX gating + divergence)

Would turning DIX into a bounded **oscillator** generate a signal? The
breakout-z and trailing decile already *were* oscillators, so the real questions
are which normalization, and how to read it. Tested against 1-month forward
returns (`spx_dix_decile_vs_breakout_oscillator.csv`):

| oscillator | IC | NW t | pre-2021 | 2021+ | top-dec | bot-dec |
|---|---:|---:|---:|---:|---:|---:|
| **stochastic %D (126d)** | **+0.139** | +2.43 | +0.209 | +0.028 | +1.82 | +0.66 |
| percentile-rank (126d) | +0.121 | +2.69 | +0.171 | +0.039 | +1.83 | +0.53 |
| RSI(DIX) 14 *(momentum)* | +0.034 | +0.13 | +0.055 | +0.001 | +1.02 | +0.81 |
| MACD-hist(DIX) *(momentum)* | −0.017 | −0.34 | +0.005 | −0.051 | +1.26 | +1.54 |

Three conclusions:

1. **A range/rank-bounded oscillator is the best normalization found in this
   whole study.** A smoothed **stochastic %D on a 126-day window** posts IC
   +0.139 (pre-2021 +0.209) — edging the z-score, because min-max/rank bounding
   is robust to exactly the drift and fat tails behind the "45% floor".
2. **Read it directionally, not overbought/oversold.** Top decile (+1.82%) beats
   bottom (+0.66%) and the response is monotone — DIX extremes **persist**, they
   don't reverse. The textbook "fade the extreme" oscillator play is backwards
   here.
3. **Momentum oscillators on DIX are dead** — RSI ≈ 0, MACD-hist *negative*. The
   *rate of change* of DIX carries nothing; only the *level within its range*
   does. (Kept in the script so it isn't re-discovered.)

### GEX gating — the defensive lever that survives best post-2021

Conditioning the oscillator on the gamma regime (GEX is in the same file):

| regime | IC | pre-2021 | 2021+ |
|---|---:|---:|---:|
| GEX > 0 (long gamma, ~91% of days) | **+0.148** | +0.206 | **+0.056** |
| GEX ≤ 0 (short gamma) | +0.021 | +0.119 | −0.117 |

The gate is **defensive, not additive**: long-gamma is ~91% of days, so gating
barely moves the average return, but it removes the short-gamma regime where the
oscillator **inverts** (post-2021 GEX≤0 IC goes negative). GEX>0 gating roughly
*doubles* the (still-weak) post-2021 IC (+0.056 vs +0.028 ungated) and is the
least-dead signal anywhere after 2021.

### Option B — DIX-vs-price divergence (the one you flagged as useful)

`sign(21d DIX change)` vs `sign(21d price change)`:

| cell | n | fwd 1mo | hit | vs base (+1.07%) |
|---|---:|---:|---:|---:|
| **DIX up / price down** (bullish div) | 774 | **+1.53%** | 69% | +0.46 |
| both down (capitulation buying) | 476 | +1.99% | 71% | +0.91 |
| both up | 1,074 | +0.97% | 68% | −0.10 |
| DIX down / price up (bearish div) | 1,472 | +0.61% | 65% | −0.46 |

De-overlapped entry events on the bullish-divergence flag (84 entries, 21-day
cool-down): **+1.61%, 70% hit**. It is a genuinely distinct signal — dark buying
*into* price weakness — and reads intuitively. **This flag is now surfaced live
on the dashboard's NDX, SPX and IWM "DIX vs Return" tabs** as a dark-flow-
divergence badge (`compute_divergence_signal` in `ndx_dark_residual.py`; NDX uses
the reconstructed NDX-100 dollar-DIX vs QQQ, SPX/IWM use their reconstructed
dollar-DIX vs SPY/IWM). Note the dashboard computes it on each index's
*reconstructed* dollar-DIX rather than the raw published DIX validated here —
cousins, so treat the badge as directional colour, not the exact numbers above. Honesty check on the regime,
though: the bullish-divergence *edge over baseline* was **+0.72 pp pre-2021** but
**−0.07 pp in 2021+**, so like everything else here its edge has thinned in the
current regime even though its raw hit rate stays ~70%.

**Net of the oscillator work:** the best single construction is a **126-day
stochastic %D, read directionally, gated on GEX>0**, with **DIX-vs-price
divergence** as a second, independent flag. It repackages the level signal more
robustly and the GEX gate defends it best post-2021 — but none of it escapes the
regime fade, so size it as one input, not a standalone system.

## Bottom line

- **The 45% "floor" is drift, not a broken feed.** Off-exchange share rose ~0.42
  pp/yr; 45% went from the ~88th percentile of 2019 to the ~4th–24th percentile
  of 2023–25. Stop reading raw levels; rank the DIX against its own recent
  history.
- **The level (decile) wins against a 1-year breakout, but the breakout's
  weakness was mostly a baseline-length problem.** A trailing-*1-year*
  breakout/breakdown adds nothing on top of the level (joint-regression t =
  −0.76), but shortening the baseline to **3–6 months** roughly doubles its IC
  and makes it competitive with the level (see the follow-up sweep). Too fast
  (1-month) is noise; too slow (1-year) over-de-trends; DIX's information lives
  at a **medium 3–6 month frequency**. Smoothing the level itself is
  counterproductive.
- **But the honest headline is that both have stopped working since ~2021.**
  The DIX→next-month relationship, in every de-trended form, is ~0 in the new
  regime. Treat pre-2021 DIX edges as regime-specific until more post-shift
  months accumulate.

## Caveats

- **Overlapping windows.** ~3,566 daily obs ≈ ~170 independent months; the
  block-bootstrap CIs, the Newey-West t-stats, and the entry-event counts are
  the honest lenses, not the every-day means.
- **Raw vs. reconstructed DIX.** This study uses SqueezeMetrics' published
  index-level DIX (the "45%" series). The dashboard's SPX tab uses a
  reconstructed dollar-DIX from IVV constituents; the two are cousins but not
  identical, and their absolute levels are not directly comparable.
- **Regime split is a single event.** The pre-2021 vs. 2021+ contrast rests on
  one structural break; the "signal faded" conclusion will firm up (or not) as
  the new regime lengthens.
- The full head-to-head table across all three horizons is written to
  `spx_dix_decile_vs_breakout.csv`.
