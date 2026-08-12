# Does per-name DPI lead single-stock trend reversals?

Request: in the rotation tape (indices quiet, stocks moving hard against each
other), is there a DPI signal for *when* a stock's trend reverses? This study
flips the usual lens: find the **actual ex-post turning points** — 5,516 troughs
(local ±21-session low in a 3-month downtrend) and 7,314 peaks (mirror image) —
and stack the 10-day-average-DPI percentile (p10) in event time around them,
with cluster-bootstrap-by-name CIs. Then the tradeable version: does high DPI
raise P(trough within 21 sessions)? Full S&P 500, 2019–2026. Reproduce:
```
python spx_trend_reversal_study.py --start 2019-01-01 --out spx_trend_reversal.csv
```

**Short answer: DPI does not lead reversals — it lags them. Dark-pool buying
pressure *sags* into troughs (0.515 → 0.453 by τ=0, significantly below its
regime baseline from three weeks out) and only surges *after* the low is in
(0.567 at τ+21, significantly above baseline). At peaks it is the mirror image:
elevated into the top, falling after. And the tradeable version is mildly
backwards: a HIGH 10-day-avg DPI in a downtrend makes a near-term trough *less*
likely (−7pp raw; still −1.1pp [−2.1, −0.1] after the recent-return control,
reproducing OOS at −1.6pp). Per-name dark flow is a trend-confirming variable,
not a turn-timing one.**

## 1. The event-time picture

**p10 around 5,516 troughs** (baseline = mean p10 over all downtrend days, 0.511):

| τ | −42 | −31 | −21 | −10 | −5 | −1 | **0** | +5 | +10 | +21 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p10 | .515 | .516 | .503* | .489* | .477* | .461* | **.453*** | .468* | .528* | **.567*** |

(*) CI excludes baseline. The shape is unambiguous: dark buying *dries up* as
the sell-off climaxes into the low, bottoms exactly at the trough, and recovers
with price — overshooting the baseline two–four weeks *after* the turn. If you
wait for the DPI surge as your reversal tell, you are ~2–4 weeks late, which is
most of the bounce (t-trough ≈ day 10 of 21 from the path study).

**p10 around 7,314 peaks** (baseline 0.534): elevated the whole way up (.550–.561,
significant into the top), peaks *at* the peak, then falls below baseline after
(.496 at τ+10). Same message from the other side: DPI rides the trend it is in;
it does not flinch before the turn.

## 2. The tradeable version — mildly backwards

P(trough within next 21 sessions), among downtrend name-days (base rate 24.4%):

| group | P(trough ≤21d) |
|---|---:|
| LOW p10 (≤0.20) | **28.7%** |
| HIGH p10 (≥0.80) | **21.0%** |
| no streak | 25.6% |
| 10+ streak | 18.6% |

HIGH−LOW daily difference **−6.97pp [−8.52, −5.36]**; streak-vs-none −6.53pp.
Most of that is the recent-return characteristic (deeper recent losers are
mechanically closer to their local low, and LOW-DPI names are the deeper
losers), but the Fama-MacBeth control (`tw ~ rback + HIGH-dummy`) leaves
**−1.10pp [−2.10, −0.07]** full-sample, and it *strengthens* out of sample
(IS −0.66 n.s., OOS 2023+ **−1.58 [−2.96, −0.10]**) — the one split where the
user's "last ~2 years" tape lives, and the effect is most clearly negative
there. High dark flow in a downtrend means the trough is, if anything, *further
away*, not closer.

## 3. How this reconciles everything else in the thread

This result is the missing mechanism behind the earlier nulls — the pieces now
fit together:

- **The streak names were "stabilizers" (prior 21d +0.91%)**: their high DPI
  shows up *after* their local turn — exactly the post-trough DPI surge in the
  event curve. That's why they bounce a bit higher (the MFE result) yet carry
  no timing information: the turn already happened.
- **The falling-knife null**: LOW DPI troughs *sooner* (28.7%) yet earns the
  same forward return — because it keeps falling *into* that nearer trough. The
  earlier trough and the deeper path net out to the same endpoint.
- **The GOOGL-reclaim null**: "reclaim + high DPI" is precisely the post-turn
  confirmation state; by then the edge is spent.

So per-name DPI is best understood as a **coincident/lagging trend thermometer**
— it tells you what the stock has been doing, at dark-pool resolution, and its
surge is a *confirmation* that a bottom formed, not a forecast that one will.

## Verdict

The reversal-timing hope is refuted, cleanly and with the strongest design
available (conditioning on the actual turns, not on the signal). There is no
lens — level, average, streak, path, or event-time — in which per-name DPI
*precedes* single-stock trend reversals; its only consistent lead is the mildly
perverse one (high DPI → turn later). For the rotation game, the tradeable
information stays where the whole thread has located it: systematic (index
DIX/GEX regime + %D entries), with per-name DPI usable at most as a *lagging
confirmation* that a turn you already traded is holding.

## Caveats

- Turning points are ex-post by construction (a local extremum needs ±21 future
  sessions); that is deliberate — the event study asks whether the information
  exists at all, and the answer is no. The forward-probability test contains no
  look-ahead in its conditioning variables; edge-of-sample labels are NaN,
  never "no reversal".
- Trough/peak definitions (±21 local extremum, 3-month prior trend, 42-session
  de-overlap) are one reasonable choice; the event-curve shape (sag-into,
  surge-after) is strong enough that small definition changes won't flip its
  sign.
- Cluster bootstrap is by name; date clustering (many names troughing together
  in index selloffs) is partially absorbed by the daily-difference tests, which
  agree.
- Machinery unit-tested: V/W-shape turn detection with de-overlap, NaN edge
  handling of the forward label, planted DPI-ramp recovery in the event curve,
  planted probability-lift recovery and null flatness
  (`tests/test_trend_reversal.py`).
