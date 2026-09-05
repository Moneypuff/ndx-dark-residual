# Does DPI strength affect realized volatility?

**Question.** Does the strength of a name's dark ratio (DPI) predict its realized volatility?

**Method** (`dpi_volatility_study.py`, live payload, 2018-08-01 → 2026-09-01, 102 names).
DPI strength = the name's trailing (no-look-ahead) decile of its 5-day-MA dark ratio (1–10).
Target = forward realized vol `RV_h` = annualized std of daily log returns over `[t+1, t+h]`
(h = 10/21/42). Two confounds are removed: **vol persistence** (RV is highly autocorrelated) and
**market-vol regimes** (high DPI clusters when the whole market is more volatile). So the reported
metrics are market-relative `ln(RV/RV_QQQ)`, a within-name high-vs-low contrast, and — decisively —
a pooled OLS of `ln(fwd RV)` on the DPI decile **controlling for trailing RV (HAR) and market RV**.

## Answer: no usable effect

**1. Raw forward RV is flat across DPI deciles** (21d): ~35.3% at every decile, no gradient.
The absolute volatility of a name is unrelated to how strong its DPI is.

**2. Market-relative RV drifts slightly *down* with DPI — but that is the regime confound, not DPI.**

| DPI decile (21d) | raw RV % | ln(RV/QQQ) | concurrent QQQ RV % |
|---|---:|---:|---:|
| D1 | 35.3 | +0.499 | 20.3 |
| D5 | 35.2 | +0.493 | 20.5 |
| D10 | 35.6 | +0.463 | 21.5 |

`D10−D1 ln(RV/QQQ) = −0.036` (high-DPI names ≈ 4% *less* volatile relative to QQQ). But the name's
own RV barely moves (35.3 → 35.6); the ratio dips only because **the market itself is more volatile
in the high-DPI buckets** (QQQ RV 21.5% at D10 vs 20.3% at D1). High DPI coincides with modestly
higher-vol market days — it does not raise the stock's own vol.

**3. Within-name high(≥D8) vs low(≤D3) market-relative RV ≈ 0**: mean `−0.008` (0.99×), 41/99 names
positive, cross-name t = −0.9 (not significant). Inside a given name, high- and low-DPI days have
essentially the same forward vol.

**4. Incremental test (the decisive one).** With vol persistence and market vol controlled, the DPI
decile coefficient on `ln(fwd RV)` is:

| horizon | DPI-decile coef | per-decile effect | full-sample t | pre-2022 t | 2022+ t |
|---|---:|---:|---:|---:|---:|
| 10d | −0.0018 | ×0.998 | −4.0 | −3.8 | −1.8 |
| 21d | −0.0020 | ×0.998 | −4.8 | −4.8 | −1.5 |
| 42d | −0.0032 | ×0.997 | −8.1 | −8.3 | −2.1 |

The coefficient is "significant" only because n ≈ 174,000; the **effect size is trivial** — each
decile multiplies forward RV by ~0.997, so the entire D1→D10 span moves vol by ~2–3%. Forward vol is
explained by **trailing RV** (t = 54–65) and **market RV** (t = 24–31); DPI adds almost nothing. And
the small effect is a **pre-2022 artifact** — it fades to t ≈ −1.5…−2.1 after 2022 (same on a
continuous within-name DPI z-score: pre-2022 t −5.0, 2022+ t −1.2).

## Verdict

**DPI strength does not meaningfully or reliably affect realized volatility.** Absolute vol is flat
across DPI deciles; the faint market-relative dip is a market-regime coincidence, not a DPI effect;
within-name it is zero; and once vol persistence and market vol are controlled, DPI's incremental
contribution is economically trivial and not stable across regimes. Insofar as the sign means
anything, high DPI is associated with marginally *calmer* idiosyncratic vol — consistent with
orderly institutional absorption rather than pre-move informed accumulation that would spike vol.
For forecasting realized vol, use vol persistence and the market; DPI is not a vol signal.

## Reproduce

```
python dpi_volatility_study.py --payload docs/index.html   # writes data/dpi_volatility_by_decile.csv
```
`data/dpi_volatility_by_decile.csv` is a descriptive decile summary (raw fwd RV, market-relative RV,
concurrent market RV, n) of the run above.
