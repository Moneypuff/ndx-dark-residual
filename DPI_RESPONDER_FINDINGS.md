# DPI responder segregation — which names respond to a high dark ratio

**Question.** Bucket the NDX-100 names into those whose forward return responds *positively* to
having their DPI in the top deciles (> D7), those that don't respond, and those that respond
*negatively*.

**Method** (`dpi_responder_study.py`, run on the live dashboard payload, 2018-08-01 → 2026-09-01,
102 names with both a dark ratio and price). Response is measured **within each name** and **in
excess of QQQ**:

- `high` day = the name's trailing (no-look-ahead, 252-day) DPI decile ≥ 8, i.e. above D7;
- `x(h)` = the name's h-day forward return minus QQQ's over the same window;
- **`response = mean(x | high) − mean(x | all days)`** — how much better/worse the name does after
  a high-DPI reading than on an average day.

Within-name-and-excess-of-QQQ is deliberate: it isolates the DPI effect, cancels the name's own
drift (a name that merely trended up is not scored a "responder"), and cancels the market-timing of
*when* high-DPI clusters occur. Per-name significance is a moving-block bootstrap (block = h).
Buckets at |t| ≥ 1.5. Reference horizon 42 trading days (1/2/3-month all reported).

## The in-sample buckets exist…

| horizon | responders (t ≥ +1.5) | neutral | negatives (t ≤ −1.5) | cross-name mean response |
|--------:|----------------------:|--------:|---------------------:|-------------------------:|
| 21d | 8 | 74 | 17 | −0.02% |
| **42d** | **7** | **78** | **14** | **−0.18%** |
| 63d | 7 | 77 | 15 | −0.31% |

Positive bucket (42d): PLTR, PDD, ABNB, VRTX, TXN, ODFL, TTWO.
Negative bucket (42d, strongest last): SBUX, AMAT, FTNT, DDOG, ARM, ALNY, TRI, INTU, NXPI, PANW, …
Full per-name table: `data/dpi_responder_buckets.csv`.

## …but only the *negative* bucket is more than chance, and neither bucket is stable

Three honesty checks decide whether the segregation is real. It mostly is not.

**1. Chance / multiple testing.** With ~100 names, a t-test manufactures "responders" by luck.
Under the null, ≈2.3 names clear |t| ≥ 2 on each side.

| horizon | positive at |t|≥2 | negative at |t|≥2 | expected each side |
|--------:|------------------:|------------------:|-------------------:|
| 21d | 3 | 9 | ~2.3 |
| 42d | 1 | 10 | ~2.3 |
| 63d | 3 | 10 | ~2.3 |

- The **positive-responder bucket is indistinguishable from noise** (1–3 names vs ~2.3 expected).
  There is **no reliable set of names that respond positively to high DPI.**
- The **negative bucket is a real excess** (9–10 vs ~2.3; binomial p < 0.001): for a genuine subset,
  a high dark ratio weakly *precedes underperformance*. This agrees with the earnings-DPI study
  (high pre-report DPI → mildly negative excess) and the episode study (no positive post-episode
  edge; the dark ratio is contemporaneous, not a bullish forecast).

**2. Persistence (name-stability).** Split the sample in half, classify on the first half, measure
the second-half response of the same names:

| horizon | Spearman(H1 response, H2 response) |
|--------:|-----------------------------------:|
| 21d | +0.27 |
| 42d | +0.14 |
| 63d | −0.02 |

Weak at best and gone by 63d. The specific names in each bucket **do not carry over** — a
first-half "responder" is a coin-flip in the second half. You cannot build a fixed
"these-names-respond-to-DPI" watchlist and expect it to hold.

**3. Structure (ex-ante features).** Nothing knowable in advance separates responders:

| feature | Spearman with 42d response |
|---------|---------------------------:|
| index weight (size) | −0.10 |
| mean DPI level | +0.09 |
| DPI volatility | −0.04 |

By sector the spread is small and not robust (Technology is the most negative at −0.32 with only 2
of 42 names positive; Cons. Discretionary / Health Care mildly positive on tiny counts). No sector
gives a dependable responder set.

## Verdict

- **No positive-responder bucket.** The names that look bullish-on-high-DPI in-sample are chance;
  they neither exceed the multiple-testing baseline nor persist.
- **A real but unrostered negative tilt.** High DPI weakly precedes underperformance for more names
  than chance allows, but *which* names is unstable and unexplained by size, dark-share, DPI
  volatility, or sector. Treat it as a faint cross-sectional headwind, not a name-selection rule.
- **Practical use.** Do not segregate names into a durable DPI-responder list — it will not hold out
  of sample. The dark ratio remains a contemporaneous confirmation signal (strongest in the
  down-gap earnings cut), not a per-name forward alpha.

## Reproduce

```
python dpi_responder_study.py --payload docs/index.html   # writes data/dpi_responder_buckets.csv
```

`data/dpi_responder_buckets.csv` is a **descriptive snapshot** of the run above (per-name bucket,
42d/63d response, t, hit rate, high-day count); by finding (2) it is not a forward-looking
selection and should not be traded as one.
