# GDX after its own signals — the paths behind "punishes chasers"

The regime-log leaderboard carries two GDX statistics that look contradictory:
high-conviction breaks don't follow through (−0.5% median next quarter, 48%),
yet its relative-strength **turn signals are 5-for-5**. This study maps the
forward price paths behind both, then tests entry rules against each other.
All events come from the page's own detectors (`conviction_frame` /
`conviction_events` / `detect_breaks` in `build_regime_log.py`), Yahoo
adjusted closes 2004 → Aug 2026.

Reproduce:
```
python gdx_chase_study.py --cache-dir .ndx_dark_cache
```
Numbers below are from the 2026-08-07 close.

## The three event families

- **Up-breaks (n=86)** — conviction score ≥ 3 with the 21d move up: the
  "chase" entry. First-day events, 21-session cooldown per direction.
- **Down-breaks (n=73)** — the same score in a down move: capitulation.
- **Turn signals (n=13)** — `detect_breaks` on the GDX-vs-SPY 63d spread,
  sgn = −1: an entrenched underperformance regime (≤ −5pp/63d) cracking
  upward (21d spread z crossing +1.5). The last five (Oct 2018, Jan 2022,
  Oct 2022, Mar 2024, Jan 2025) are the 5-for-5; the 2008–2015 eight went
  2-for-8. Graded vs SPY; own-price paths shown below.

## The path map

Median % from event close (sessions after the event):

| family | d5 | d10 | d21 | d42 | d63 | d126 | max DD d1-63 (med) | trough | max fav (med) | peak |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| up-breaks | +0.0 | −0.4 | −0.4 | −0.9 | −0.5 | **+5.6** | **−10.9%** | ~d30 | +11.1% | ~d37 |
| down-breaks | +0.7 | +1.6 | **+2.6** | +2.5 | **+4.8** | +4.7 | −7.6% | ~d28 | +14.4% | ~d39 |
| turns | +0.2 | −0.1 | +1.0 | **+7.4** | +1.3 | +5.7 | −9.4% | ~d23 | +14.3% | ~d46 |

The chase's true shape: **the median up-break round-trips ±11% inside the
quarter and ends where it started, then resolves higher by six months.**
74% of chases see a >5% drawdown from the entry, 61% see >8%, median trough
near day 30. "Punishes chasers" is not a drawdown-to-zero story — it's a
you-bought-the-midpoint-of-a-round-trip story. Down-breaks are the mirror:
GDX is a mean-reverter around its own conviction events in both directions.

## What the −0.5% median is actually made of

| chase (day0 → d63) | n | median | hit | mean |
|---|--:|--:|--:|--:|
| all | 85 | −0.5% | 48% | +2.7% |
| **2006–2015 era** | 36 | **−2.4%** | **36%** | −1.9% |
| **2016– era** | 49 | **+2.6%** | **57%** | +6.1% |
| GLD above its 200d | 73 | −0.3% | 49% | +3.3% |
| GLD below its 200d | 12 | −2.3% | 42% | −0.4% |
| score 4/4 | 8 | +3.3% | 62% | +9.8% |
| already +15%/21d at event | 14 | +0.3% | 50% | +5.2% |

Two things worth reading twice. First, the headline stat is **an average of
two regimes**: chasing GDX in the 2006–2015 secular bear lost (36% hit);
chasing it in the 2016– bull worked (+2.6%, 57%) — same signal, opposite
sign, exactly like the turn-signal record (2-for-8 then 5-for-5). Second,
mean ≫ median everywhere: the chase's +2.7% mean against a −0.5% median
says the payoff is a **right tail** (2016, 2020, 2024–25 runs), not a
steady edge. A rule that always avoids the chase also always misses the tail.

## Entry rules, head to head

| rule | fills | median | hit |
|---|--:|--:|--:|
| chase the event close, exit d63 | 100% | −0.5% | 48% |
| limit −4% (working d1–42), exit 63 sessions after fill | 71% | −0.3% | 50% |
| limit −6%, exit fill+63 | 61% | +1.6% | 58% |
| **limit −8%, exit fill+63** | **53%** | **+3.9%** | **58%** |
| wait 21 sessions, buy the close, exit d63 | 100% | +0.7% | 54% |
| chase with a −12% stop | 100% | **−7.8%** | **38%** |

- **The dip entry is the edge.** A resting limit 8% under the event close
  fills roughly half the time (the median event dips ~11%), lands near the
  day-30 trough, and holding a quarter from the fill beats the chase by
  ~4.4pp median with a better hit rate. The cost is missing the other half —
  the strongest events never look back (that's the right tail).
- **Stops are the actual punishment.** A −12% stop converts the *typical*
  path's temporary drawdown into a realized loss and then misses the
  recovery: median −7.8%, 38% hit — far worse than doing nothing. GDX's
  post-event volatility must be handled by **position size, not stop
  distance**.
- Simply waiting 21 sessions (the flat part of the median path) flips the
  median positive with no fill risk.

## The strategy this implies

1. **Never chase full-size on event day.** Size a half (or third) position at
   most; the median path gives a better price 74% of the time.
2. **Rest the adds 6–8% below the event close, working ~2 months.** Filled
   → hold a quarter *from the fill*, not from the event.
3. **No price stops inside the first quarter.** The median winner is down
   ~10% at some point before working. Cap risk by sizing the whole position
   to survive a −16% excursion (the q25 drawdown).
4. **Let the regime set the aggression.** In a gold bull (GLD over its 200d,
   turn-signal era) the chase itself is +EV and the dip-buy is very +EV; in
   a bear both are −EV — the same rules would have lost 2006–2015. The
   regime call does the heavy lifting, exactly as the guide's era warning
   says.
5. **Treat turn signals and capitulation as the good entries.** Both buy
   weakness by construction (deep relative underperformance / a downside
   break) and both carry positive medians at d21–d63. The conviction
   up-break is a *confirmation* that arrives late; the money entry is the
   weakness that precedes or follows it.

At writing this is live: GDX printed a fresh 4/4-conviction up-break on
2026-08-07 at 89.89 after +18.6%/21d. History for this exact spot (score 4,
bull era, extended): favorable but tail-driven — and the playbook above says
half-size now, rest adds at ~83–84.5, no stop, full risk budget ~−16%.

## Caveats

- n=13 turn signals and n=8 score-4 events — direction-of-lean numbers, not
  precision. The 5-for-5 itself is what a 60%-accurate process produces ~8%
  of the time by luck.
- The era split is one in-sample cut, chosen with hindsight; "GLD vs 200d"
  is its knowable-on-the-day proxy and is weaker (n=12 below).
- Overlapping horizons: 86 events over 20 years cluster in trends
  (2016, 2020, 2024–25), so effective n is smaller than nominal.
- Adjusted closes only — no intraday fills; the limit-order results assume
  a touch of the level fills, and ignore gaps through it (conservative on
  entry price, optimistic on fill certainty).
- Exit rules are fixed-horizon by design (comparability), not optimized.
