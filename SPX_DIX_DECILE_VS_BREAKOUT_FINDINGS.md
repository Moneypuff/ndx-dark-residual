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

## Bottom line

- **The 45% "floor" is drift, not a broken feed.** Off-exchange share rose ~0.42
  pp/yr; 45% went from the ~88th percentile of 2019 to the ~4th–24th percentile
  of 2023–25. Stop reading raw levels; rank the DIX against its own recent
  history.
- **Between the two relative framings, the decile (level) wins** — ~2× the IC,
  ~2× the long-short spread, the only Newey-West t near significance, and it
  dominates out to 3 months. A trailing-1-year breakout/breakdown adds nothing
  on top of it (joint-regression t = −0.76).
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
