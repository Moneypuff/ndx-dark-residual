# Cross-index DIX comovement → 1-month forward returns

When the three index dark-flow gauges — **NDX-100 DIX**, **S&P 500 DIX** and
**Russell 2000 / IWM DIX** — line up vs. pull apart, what do the indices do over
the following month? Study of the **5-day moving average** of each dollar-DIX
against each index's own 1-month (21 trading day) forward return, **6 Jan 2020 –
28 Jul 2026, 1,648 common sessions**.

Reproduce:
```
python index_comovement_study.py --csv index_comovement_regimes.csv   # text/CSV study
python build_comovement.py --docs-out docs/comovement.html            # interactive tab
```
Data is read from the payload already embedded in `docs/index.html` (no live
FINRA re-fetch); pass `--html`/`--payload` to point at another build. The
interactive version is published as the **Comovement** tab in the dashboard
(`docs/comovement.html`), rebuilt nightly by the refresh workflow.

## Definitions

- **DIX** = dollar-weighted `Σ($ short volume) / Σ($ off-exchange volume)` across
  an index's constituents each day (SqueezeMetrics' construction;
  `compute_dollar_dix` in `ndx_dark_residual.py`). NDX and SPX sit ~0.33–0.59;
  IWM runs wider (~0.12–0.70). High DIX is the conventionally *bullish* reading
  (market-makers shorting to fill dark buy orders = accumulation).
- **DIX5** = 5-day moving average of DIX (min 3 obs) — the same 5d MA the
  dashboard uses as its residual benchmark.
- **Decile → regime**, per index, over the common sample: **Low** = deciles 1–3
  (bottom 30% of that index's own DIX5 history), **Mid** = 4–7, **High** = 8–10.
- **Forward return** = each index's own price proxy, 21 sessions ahead, in
  percent: **QQQ** for NDX, **SPY** for SPX, **IWM** for the Russell 2000
  (`compute_forward_return`). Scored on the date the regime is observed, so no
  look-ahead in the outcome.
- **hit%** = share of observations with a positive 1-month return.

## Baseline (all 1,648 common days)

| Index | mean 1m | median | hit% |
|---|---:|---:|---:|
| NDX (QQQ) | +1.77% | +2.36% | 65% |
| SPX (SPY) | +1.33% | +1.94% | 69% |
| IWM       | +1.12% | +1.43% | 60% |

## Headline: the requested divergence — SPX & NDX DIX **Low**, IWM DIX **High**

**91 days (82 with a full forward window). All three indices beat their baseline,
and it is the small cap that lifts the most relative to its own norm.**

| Index | mean 1m | median | hit% | vs. baseline mean |
|---|---:|---:|---:|---:|
| **NDX (QQQ)** | **+2.86%** | +4.00% | 72% | +1.09 pp |
| **SPX (SPY)** | **+2.15%** | +2.70% | 73% | +0.82 pp |
| **IWM**       | **+2.55%** | +2.69% | 65% | **+1.43 pp** |

Read plainly: when large-cap dark flow has cooled to its lower deciles while
small-cap dark flow is running hot, the next month has historically been
**broadly bullish for all three** — a "dash-for-trash / breadth-catch-up" setup —
with IWM roughly **doubling** its unconditional 1-month mean.

## The full 27-regime picture

`N`/`S`/`I` = NDX / SPX / IWM regime. Sorted by frequency; means are in %.
(Full table in `index_comovement_regimes.csv`.) The signal lives in the
**divergences** — pure comovement is comparatively muted.

**Comovement (all three aligned) is only mildly positive:**

| Regime | days | NDX | SPX | IWM |
|---|---:|---:|---:|---:|
| all **High** (N,S,I High) | 169 | +2.15% | +1.53% | +1.60% |
| all **Low**  (N,S,I Low)  | 170 | +1.71% | +1.85% | +1.73% |
| all **Mid**               | 213 | +2.32% | +2.00% | +2.01% |

Whether all three gauges are hot together or cold together, the following month
is a modest positive near baseline — the joint level alone barely discriminates.

**The strongest bullish divergences — large-cap DIX firm, small-cap DIX Low:**

| Regime | days | NDX | SPX | IWM | notable |
|---|---:|---:|---:|---:|---|
| N=High, S=Mid, I=**Low** | 32 | **+6.79%** | +4.53% | +5.04% | NDX hit 94%, SPX 88% |
| N=Mid, S=Mid, I=**Low**  | 137 | +3.32% | +2.58% | +1.76% | NDX hit 85%, SPX 82% |
| N=High, S=High, I=**Low**| 74 | +3.10% | +1.93% | +1.37% | NDX hit 84%, SPX 86% |
| N=Low, S=Mid, I=**High** | 40 | +3.59% | +3.15% | +4.31% | IWM hit 79% |

**The weakest — the mirror image, small-cap DIX High while large-cap sags:**

| Regime | days | NDX | SPX | IWM | notable |
|---|---:|---:|---:|---:|---|
| N=Mid, S=**Low**, I=Low | 47 | −0.91% | **−2.84%** | **−6.82%** | IWM hit **17%** |
| N=High, S=Mid, I=**High** | 30 | −2.16% | −1.32% | −1.75% | NDX hit 37% |
| N=Mid, S=**High**, I=High | 27 | +0.45% | −0.91% | **−3.49%** | IWM hit 30% |
| N=Mid, S=Mid, I=**High**  | 102 | +0.36% | +0.55% | −0.20% | IWM hit 46% |

## What the pattern says

1. **Divergence, not level, carries the signal.** All-hot and all-cold regimes
   both sit near baseline; the large positive and negative outcomes cluster in
   the mixed regimes.
2. **A *high* IWM DIX on its own is a poor omen for small caps** — a contrarian
   / mean-reverting tell. Every "IWM High while large-cap ≤ Mid" bucket has weak
   or negative IWM forward returns (−0.2% to −3.5%, hit ≤ 46%).
3. **…but the *combination* in the request flips it bullish.** Pairing High IWM
   DIX with **Low** SPX *and* NDX DIX (the requested regime) is the one
   high-IWM setup that has led higher across the board — the divergence, not the
   IWM level alone, is what matters.
4. **The cleanest large-cap tailwind is the opposite tilt:** large-cap DIX firm
   (Mid/High) while small-cap DIX is Low → the best NDX/SPX months in the sample
   (up to +6.8% NDX at a 94% hit rate), consistent with dark accumulation
   concentrating in mega-caps.

## Caveats

- **Overlapping windows.** 21-day forward returns overlap heavily, so
  observations are autocorrelated — ~1,648 sessions is only ~78 *independent*
  months. Regimes of 30–90 days represent only a handful of distinct episodes;
  treat **medians and hit-rates as sturdier than means**, and the small buckets
  as suggestive, not significant.
- **Sample window.** Starts Jan 2020, so the COVID crash-and-recovery and the
  2022 drawdown both sit inside it and inflate some means and hit-rates.
- **In-sample deciles.** Regime thresholds are the full-history deciles of DIX5,
  so live use would key off provisional (expanding-window) cutoffs.
- **IWM DIX is reconstructed** from iShares' Russell 2000 holdings and spans a
  wider range than the NDX/SPX gauges; its deciles are internally consistent but
  not directly comparable in raw level.
