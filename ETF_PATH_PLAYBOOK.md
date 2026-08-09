# The leaderboard universe after its own signals — path map and playbook

`GDX_CHASE_FINDINGS.md` mapped what happens after GDX's conviction events.
This study runs the identical map across **all 21 leaderboard ETFs** (the
regime log's SECTORS + THEMES universe) and turns the result into a
playbook: which ETFs reward chasing a break, which round-trip, how to size,
how long to hold, and what return / drawdown to expect while in the trade.
Same detectors as the page (`conviction_frame` score ≥ 3, `detect_breaks`
vs SPY), Yahoo adjusted closes 2004 → Aug 2026, own-price forward paths.

Reproduce:
```
python etf_path_study.py --cache-dir .ndx_dark_cache --csv-out etf_path_stats.csv
```
Numbers below are from the 2026-08-07 close. Event families as in the GDX
study: **up-breaks** (score ≥ 3, 21d move up — the "chase"), **down-breaks**
(same score, move down — capitulation), **turns** (ETF-vs-SPY 63d spread
≤ −5pp cracking upward through z +1.5).

## The three structural results

1. **Most of the universe rewards chasing; GDX is the outlier, not the
   rule.** 16 of 21 ETFs are CHASERs (median fwd 63d ≥ +2%, hit ≥ 55%);
   the pooled chaser median is +3.8% at 63 sessions (69% hit) and the
   median path is still rising at day 126 for essentially every ETF — the
   natural hold is six months, not one quarter. Only five are ROUND-TRIPs
   (XLE, XLP, XLRE, KRE, GDX), and only GDX has a negative chase median.
2. **Buying capitulation works on all 21.** Every down-break median fwd 63d
   is positive (+3.4% to +7.9%), the bounce starts immediately (median
   trough day 6–28, most within ~2 weeks), and the best names are the same
   growth complex that also chases well (QQQ +7.9%, 80% hit).
3. **The mean−median gap tells you which tail you own.** Chasers run
   *negative* skew (QQQ mean−median −1.2pp, XLI −2.0pp): steady grinders
   whose risk is an occasional large loss — sizing must respect the q25
   drawdown, not the median. Round-trippers run *positive* skew (GDX
   +3.2pp): flat medians with a right-tail lottery — entries must be priced
   (dip limits), and stops must not amputate the tail.

## Up-breaks — the chase table

Sorted by median fwd 63d. "Size" = position weight that loses 1% of NAV at
the q25 max adverse excursion (the sizing rule of this playbook); RR =
median fwd 63d ÷ |q25 MAE|.

| ETF | class | n | med 63d | hit | med 126d | mean−med | DD med | DD q25 | size/1% risk | RR |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| QQQ | CHASER | 91 | +5.5 | 76% | +8.7 | −1.2 | −2.6 | −6.1 | 16% | 0.89 |
| SMH | CHASER | 105 | +5.2 | 66% | +12.5 | +0.1 | −5.4 | −9.9 | 10% | 0.52 |
| XLC | CHASER | 34 | +4.7 | 71% | +12.6 | −1.7 | −3.9 | −6.4 | 16% | 0.73 |
| VUG | CHASER | 100 | +4.4 | 76% | +7.7 | −1.3 | −2.8 | −5.3 | 19% | 0.82 |
| EEM | CHASER | 85 | +4.2 | 71% | +5.5 | −1.1 | −4.1 | −8.9 | 11% | 0.47 |
| IGV | CHASER | 113 | +4.1 | 70% | +7.6 | −1.2 | −3.3 | −8.2 | 12% | 0.50 |
| XLI | CHASER | 96 | +4.0 | 69% | +7.7 | −2.0 | −3.0 | −5.7 | 18% | 0.71 |
| XLK | CHASER | 109 | +3.9 | 73% | +8.1 | −0.3 | −3.5 | −7.3 | 14% | 0.54 |
| IWM | CHASER | 91 | +3.7 | 65% | +4.3 | −1.9 | −4.1 | −8.0 | 13% | 0.46 |
| XLV | CHASER | 110 | +3.1 | 67% | +3.5 | −0.9 | −2.4 | −5.8 | 17% | 0.53 |
| XBI | CHASER | 95 | +2.7 | 61% | +7.2 | −0.2 | −6.1 | −10.4 | 10% | 0.26 |
| VTV | CHASER | 110 | +2.7 | 76% | +6.0 | −0.2 | −2.6 | −5.1 | 20% | 0.53 |
| XLU | CHASER | 105 | +2.6 | 65% | +6.2 | −1.0 | −3.3 | −6.7 | 15% | 0.39 |
| XLF | CHASER | 89 | +2.6 | 66% | +5.1 | −0.8 | −3.7 | −7.0 | 14% | 0.37 |
| XLB | CHASER | 99 | +2.5 | 60% | +5.0 | −0.8 | −4.1 | −7.6 | 13% | 0.33 |
| XLY | CHASER | 96 | +2.3 | 70% | +4.8 | −0.5 | −3.0 | −6.9 | 15% | 0.33 |
| XLE | ROUND-TRIP | 85 | +1.9 | 60% | +6.9 | +0.6 | −4.8 | −9.7 | 10% | 0.20 |
| XLP | ROUND-TRIP | 113 | +1.5 | 65% | +5.0 | −0.2 | −3.0 | −4.7 | 21% | 0.32 |
| XLRE | ROUND-TRIP | 50 | +1.1 | 56% | +2.8 | −0.3 | −3.5 | −8.2 | 12% | 0.13 |
| KRE | ROUND-TRIP | 85 | +0.4 | 51% | +4.3 | +0.3 | −6.7 | −11.0 | 9% | 0.04 |
| GDX | ROUND-TRIP | 86 | −0.5 | 48% | +5.6 | +3.2 | −10.9 | −16.1 | 6% | −0.03 |

The dip-entry alternative (limit −8% under the event close, exit 63
sessions after the fill) only makes sense where events actually dip: for
QQQ/VUG/XLV it fills < 13% of the time (the strong chasers rarely look
back — take the close). For the high-vol round-trippers it *is* the trade:
GDX fills 53% at a +3.9% median, XLE 26% at +6.6%, KRE 33% at +2.2%.

## Down-breaks — capitulation is a universal buy

Median fwd 63d after a high-conviction down-break, all positive:

| tier | ETFs (med 63d · hit) | expectations |
|---|---|---|
| Best | QQQ (+7.9 · 80%), IGV (+7.4 · 67%), XLE (+7.3 · 71%), VUG (+7.0 · 74%), SMH (+6.5 · 69%), XLK (+6.3 · 72%) | +2 to +3% already by day 21; trough day 6–17; DD q25 −8 to −14 → size 7–13% per 1% risk |
| Middle | XBI, VTV, XLV, XLF, XLRE, XLI, XLY, GDX, XLC (+4.5 to +5.9) | trough within ~3 weeks; XLV/XLP the calm versions (DD q25 −4 to −6 → size 17–23%) |
| Weakest | KRE, XLU, EEM (+3.4 to +3.5 · 60–70%) | still positive, thinner edge; EEM/KRE dead money first month (med 21d ≈ +0.4) |

The path shape is the same everywhere: the residual downside is front-loaded
(median trough inside the first three weeks, median MAE −2 to −8%), then a
grind up through day 126. Scale in over the first two weeks rather than all
at the close.

## Turn signals — themes turn, financials trap

Own-price median fwd 63d after a relative-strength upturn vs SPY (only
ETFs with n ≥ 8):

| ETF | n | med 63d | hit | DD q25 | read |
|---|--:|--:|--:|--:|---|
| SMH | 8 | +11.2 | 100% | −6.7 | the real 8-for-8 of the study |
| EEM | 10 | +6.9 | 80% | −4.8 | clean |
| XBI | 10 | +5.1 | 60% | −11.5 | works, rough ride |
| XLE | 9 | +5.0 | 78% | −8.1 | clean |
| GDX | 13 | +1.3 | 62% | −15.3 | works vs SPY (+ own right tail), violent path |
| KRE | 11 | −1.9 | 45% | −15.9 | trap |
| XLF | 9 | −7.8 | 44% | **−35.9** | trap — these "turns" were 2008/2011/2023 bank knives |

A deep-underperformance regime cracking upward is a buy in **themes with
mean-reverting fundamentals** (semis cycle, EM, energy, gold) and a value
trap in **credit-driven financials**, where the same signature marked the
early innings of balance-sheet crises. GDX's celebrated 5-for-5 is the
*relative* grade; SMH is the family's true star in absolute terms.

## The playbook

**1. Chasers (16 ETFs, led by QQQ/SMH/XLC/VUG/EEM/IGV).**
Buy the event close, full unit — waiting for an 8% dip forfeits most fills.
*Hold:* the median path rises into day 126; treat 6 months as the default,
one quarter as the minimum. *Expectations:* +2.5 to +5.5% median by day 63
(60–76% hit), +4 to +12.5% by day 126; en-route drawdown median −2.5 to
−6%, q25 −5 to −10%. *Sizing:* the negative skew means the q25 number is
the one to size to — 10–20% of NAV per 1% NAV risk (table above). No price
stops inside the quarter; the exit is the thesis (a fresh opposite-direction
signal or a pair-detector break), not a level.

**2. Round-trippers (GDX, KRE, XLE, XLRE; XLP is just low-amplitude).**
Never chase full-size. Half-unit at most at the close, rest of the order
6–8% below, working two months; hold 63 sessions *from the fill*.
*Expectations (on the filled dip):* GDX +3.9% median, XLE +6.6% from limit
levels; en-route q25 excursions −10 to −16% from the event close — size
6–10% per 1% risk. The positive skew is the reason to be there at all, so
no stops: the position must be small enough to sit through the median
−7 to −11% swing and still hold the right tail.

**3. Capitulation (all 21).**
Buy high-conviction down-breaks, scaling over days 0–10 (the median trough
lands there). *Hold:* one quarter minimum, six months where the tier-1
names are involved. *Expectations:* +3.4 to +7.9% median by day 63, hit
60–80%; residual drawdown median −2 to −8% (q25 −4 to −17 — XBI/SMH/GDX
are the violent ones). This is the highest-hit-rate family in the study and
the natural budget for adding risk during shocks — consistent with the pair
backtest's fade-the-shock rule (§5 of the guide).

**4. Turns.**
Trade them long in SMH/EEM/XLE/XBI/GDX with capitulation-style sizing
(q25 excursions −5 to −16%); hold two quarters (median paths peak near day
100–125). Never take the same signature in XLF/KRE — a financials
underperformance regime "cracking" has historically been the middle of the
crisis, not the end (q25 excursion −36%).

**Portfolio arithmetic.** With every position sized to 1% NAV at its q25
excursion, a book of 4–6 concurrent signals risks ~4–6% NAV in the bad
quantile with an expected quarter of roughly +0.3 to +0.9% NAV per position
(median × weight) — the edge compounds through hit rate and the six-month
tail, not through any single trade.

## Caveats

- Everything is long-only, mostly-bull-sample (2004–2026, one deep bear).
  The chaser class in particular inherits the market's upward drift;
  chase medians net of SPY would be thinner. The *relative* ranking across
  ETFs is the robust object, not the absolute levels.
- Events overlap and cluster in trends (2009, 2016, 2020, 2024–25);
  effective n is well below nominal. XLC (n=34) and XLRE (n=50) are short
  histories; turn-signal n's are 8–13 — direction of lean, not precision.
- Class thresholds (med63 ±2pp, hit 55%) are one in-sample cut; borderline
  names (XLE, XLP, XLY, XLB) can migrate classes as data accrues — rerun
  the script rather than trusting the table's edges.
- Limit fills assume a touch fills at the level (no gap-through modeling);
  fixed-horizon exits are for comparability, not optimized.
- Sizing translates historical q25 excursions into forward risk; regime
  shifts (a 2008) exceed them — the portfolio cap, not the per-name size,
  is the real backstop.
