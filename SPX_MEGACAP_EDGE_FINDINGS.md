# Mega-cap edge test: does dark-flow selection beat beta on the high-weight names?

Hypothesis (per request): the index-level DIX signal works, and the index is
dollar-weighted — so it *is* the mega-caps. If per-name dark flow carries anything
it should show up in the highest-weight names, not the long tail. Restrict the
S&P 500 to the top-N by IVV index weight and ask: can a high-DPI / dip setup
produce a small **edge over beta** there?

**Short answer: no confirmed edge over beta.** Buying high-DPI mega-caps beats SPY
and even carries real positive **CAPM alpha** — but that alpha is the **mega-cap
alpha of 2019–26** (it appears at *every* weight cutoff, in-sample and out), not
dark-flow selection. The moment you isolate selection — a beta-adjusted
darkest-minus-least-dark long-short — the alpha **straddles zero in the full
sample, in-sample, and out-of-sample**. The one cell that survived a first
excess-vs-SPY OOS split (top-50 in a downtrend) did so only because its long-only
leg inherited the mega-cap alpha; beta-adjusted and isolated to selection, nothing
survives. The point estimates are also **weakest at the very top (15–25 names)** —
the opposite of the hypothesis that the edge lives where the index weight is.

Reproduce:
```
python spx_megacap_edge_study.py --start 2019-01-01 --out spx_megacap_edge.csv
```
Universe: top-N of the IVV (S&P 500) holdings by index weight, 2019–2026 (199
names built). "High DIX" = the name's self-relative DPI percentile in its own
trailing-year top quintile. Daily equal-weight baskets, overlapping 21-day holds,
moving-block-bootstrap 95% CIs.

## The two honest, ~beta-neutral reads

- **LONG-SHORT** = darkest-quintile − least-dark-quintile mega-caps (fully
  market-neutral → any non-zero mean is pure dark-flow selection).
- **SELECTION** = high-DPI mega-caps minus the mega-cap basket's own daily mean
  (within-set demean → alpha over "just own the mega-caps", ~beta-neutral since
  all are high-cap).

**All days:**

| top-N | LONG-SHORT [95% CI] | SELECTION [95% CI] | high-DPI vs SPY | all-mega vs SPY | edge over beta |
|---:|---|---|---:|---:|---:|
| 15 | −0.70% [−1.94, +0.59] | −0.01% [−0.57, +0.61] | +1.60% | +1.61% | −0.01 |
| 25 | −0.06% [−1.19, +1.17] | −0.17% [−0.77, +0.42] | +1.09% | +1.28% | −0.19 |
| 50 | +0.13% [−0.60, +0.91] | +0.26% [−0.17, +0.71] | +1.27% | +1.01% | +0.26 |
| 100 | −0.02% [−0.54, +0.52] | +0.07% [−0.23, +0.37] | +0.78% | +0.72% | +0.07 |
| 200 | −0.29% [−0.65, +0.04] | −0.08% [−0.27, +0.10] | +0.50% | +0.57% | −0.08 |

**In a 3-month downtrend (the "buy the dip on high DIX" case):**

| top-N | LONG-SHORT [95% CI] | SELECTION [95% CI] | high-DPI vs SPY | all-mega vs SPY | edge over beta |
|---:|---|---|---:|---:|---:|
| 15 | −1.09% (n/a) | +0.33% [−0.53, +1.34] | +1.78% | +1.44% | +0.34 |
| 25 | −0.13% [−3.39, +3.88] | +0.21% [−0.77, +1.35] | +1.25% | +1.29% | −0.03 |
| **50** | **+0.73% [−0.64, +2.31]** | **+0.65% [−0.05, +1.46]** | +1.73% | +0.86% | **+0.87** |
| 100 | +0.39% [−0.38, +1.25] | +0.28% [−0.29, +0.86] | +1.01% | +0.67% | +0.34 |
| 200 | +0.03% [−0.43, +0.50] | +0.12% [−0.36, +0.59] | +0.70% | +0.60% | +0.10 |

## Reading it

1. **The "beats SPY" number is beta, not alpha.** The high-DPI mega-cap basket's
   excess vs SPY (+0.8% to +1.8%/mo) is matched almost one-for-one by the
   *all*-mega-cap basket's excess vs SPY — that is the mega-cap tilt of 2019–26,
   which any equal-weight big-cap basket earned. The dark-flow *selection* on top
   of it (`edge over beta`) is ±0.1–0.3 pp, indistinguishable from zero.

2. **Every beta-neutral CI straddles zero.** Neither the long-short nor the
   within-mega-cap selection mean is significant at any cutoff, on all days or in
   a downtrend.

3. **The one cell that flirts with significance is top-50 in a 3-month
   downtrend** — SELECTION +0.65%/mo with CI [−0.05, +1.46] (lower bound just
   grazes zero) and edge-over-beta +0.87 pp. It is *suggestive*, but: it does not
   appear at top-15/25 (the actual mega-caps), it fades by top-100/200, and it is
   one cell out of ~20 — multiple testing says treat it as noise until it
   reproduces out of sample.

4. **The mechanism is backwards.** The hypothesis was "strongest at the highest
   weights." Instead the very-top names (15–25) show *null-to-negative* selection,
   and whatever faint positive there is sits at 50–100. That is the signature of
   an **aggregate/systematic** signal (the index-level DIX reflects market-wide
   dark flow), not one where each mega-cap's own DPI forecasts its own return.
   Concentrating into the biggest names does not concentrate the edge — it
   removes it.

## Out-of-sample split (in-sample 2019–2022 vs OOS 2023–2026)

The lone suggestive cell — top-50 mega-caps, high-DPI selection, in a 3-month
downtrend — was found by scanning ~20 configurations, so the only honest test is
whether it reproduces out of sample. The configuration has no fitted parameters,
so I evaluate the *same* rule on 2019–2022 (in-sample) and 2023–2026 (OOS):

| top-N, in-3mo-downtrend | IS selection [95% CI] | OOS selection [95% CI] | IS L/S | OOS L/S |
|---:|---|---|---:|---:|
| 15 | +0.40% [−0.79, +1.67] | +0.21% [−1.22, +2.00] | −0.17% | −1.77% |
| 25 | +0.37% [−0.87, +1.80] | −0.11% [−1.58, +1.50] | −0.11% | −0.15% |
| **50** | **+0.45% [−0.22, +1.26]** | **+0.74% [−0.64, +2.58]** | +0.44% | +0.80% |
| 100 | +0.01% [−0.42, +0.47] | +0.55% [−0.47, +1.68] | −0.30% | +1.06% |
| 200 | +0.10% [−0.50, +0.69] | +0.15% [−0.63, +0.91] | −0.35% | +0.41% |

**Verdict: the top-50 cell survives the *weak* test and fails the *strong* one.**
Its sign holds in both halves — selection **+0.45% in-sample, +0.74% OOS** (and
long-short +0.44% → +0.80%), so it did not flip or vanish; if anything the OOS
point estimate is a touch larger. But **neither half's CI clears zero** — split
in two, each sub-sample (and especially the OOS downtrend days, a subset of 3.5
years) is too small to confirm, and the full-sample borderline result
(+0.65% [−0.05, +1.46]) came from pooling both halves. So it is a **consistent-
sign, statistically-unconfirmed** hypothesis, not a validated edge.

Two honesty notes: the *very top* names (15/25) do **not** reproduce — top-25 OOS
selection is −0.11% and its long-short −0.15%, so the "biggest names carry it"
thesis still fails out of sample; and top-100's downtrend selection is ~0 IS but
+0.55% OOS, i.e. *not* consistent, which is what a noise cell looks like. Only
the top-50 slice is positive in both halves.

**Practical read:** don't trade it. It is the single thread in this whole
investigation that hasn't died — worth re-checking as another year of data
accrues, and worth a proper per-name CAPM-alpha version rather than excess-vs-SPY
— but on the evidence it is a maybe, not an edge.

## CAPM-alpha confirmation (the decisive test)

Excess-vs-SPY is not a clean beta control (a high-beta name beats SPY in an up
market with no skill). So I re-scored everything as **CAPM alpha**: for each name,
estimate beta on the trailing 252 daily returns (real-time), then
`alpha = fwd_ret − beta·SPY_fwd_ret`. Two baskets:

- **ALPHA long-only** = the high-DPI basket's alpha (buy high-DPI mega-caps,
  beta-adjusted). *But this still contains whatever alpha any mega-cap basket had.*
- **ALPHA long-short** = darkest-quintile − least-dark-quintile alpha. This
  **cancels the common mega-cap alpha and isolates DPI selection** — it is the
  clean test.

Top-50, in a 3-month downtrend (the surviving cell):

| period | ALPHA long-only [95% CI] | ALPHA long-short [95% CI] |
|---|---|---|
| full | +1.65% [+0.74, +2.68] | +0.92% [−0.36, +2.34] |
| IS 2019–22 | +1.00% [+0.17, +2.00] | +0.66% [−0.90, +2.57] |
| OOS 2023–26 | +2.42% [+0.78, +4.54] | +0.90% [−0.90, +3.04] |

And the long-only alpha across **all** cutoffs (all-days) is positive and
CI-clearing in full/IS/OOS — even at **top-200** (+0.47% / +0.43% / +0.50%).

**This is the tell, and it kills the edge.** The high-DPI basket's long-only
alpha is real and robust — but it is the **mega-cap alpha of 2019–26** (big-cap
outperformance beyond beta), not dark flow: it shows up at *every* cutoff
including the entire top-200, so owning the names produces it, not selecting them
by DPI. The moment you go **long-short** — removing that common mega-cap alpha to
isolate what DPI actually picks — the alpha **straddles zero in the full sample,
in-sample, and out-of-sample** (top-50-downtrend: +0.92% [−0.36, +2.34] / +0.66%
[−0.90, +2.57] / +0.90% [−0.90, +3.04]), and across the other cutoffs the
long-short alpha is null with signs that flip between IS and OOS (e.g. top-15
all-days: IS +1.36%, OOS −1.57%).

So the surviving whisper was **long-only construction leaking the mega-cap alpha**,
not a DPI selection edge. Beta-adjusted and isolated to selection, it is **not
confirmed** — full, IS, and OOS all include zero.

## Bottom line

Restricting to the high-weight names does **not** turn the single-stock dark-flow
setup into a confirmed edge over beta. The CAPM-alpha test is decisive: high-DPI
mega-caps carry real positive alpha, but it is the **era's mega-cap alpha** (present
at every cutoff, IS and OOS), and the **DPI-selection component (long-short alpha)
is zero within CIs in every period**. The one cell that survived the excess-vs-SPY
OOS split did so only because its long-only leg inherited that mega-cap alpha;
beta-adjusted and isolated to selection, nothing remains. High-DPI mega-caps beat SPY, but that is the
well-known mega-cap tilt; the dark-flow-specific, beta-neutral component is
zero within CIs at every cutoff. This is consistent with the whole thread: dark
flow's tradeable edge is **systematic and index-level** (option B, already on the
dashboard's index tabs), not single-stock selection — not even among the names
that dominate the index. The lone top-50-in-a-downtrend cell is the only thread
worth an out-of-sample look, and it should be treated as a hypothesis, not a
result.

## Caveats

- **Survivorship** — today's IVV members and today's weights (a name's weight
  changed over 2019–26; we rank by the current weight). The beta-neutral metrics
  remove the level/tilt bias; a distortion would have to correlate weight-rank
  *changes* with the DPI signal, which is second-order.
- **Overlap** — 21-day forward windows; the block-bootstrap CIs and the
  cutoff/condition consistency (or lack of it) are the honest lenses, not any
  single point estimate.
- **Small baskets** — at top-15/25 a quintile is 3–5 names, so the long-short CIs
  are very wide (and undefined on downtrend days with too few names); the top-50+
  rows carry the weight of the read.
- **Machinery is unit-tested** against a planted selection edge
  (`tests/test_megacap_edge.py`) so the null is real, not a bug.
