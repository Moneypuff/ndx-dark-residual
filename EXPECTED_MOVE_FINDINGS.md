# Expected moves after the signals — pricing the playbook in options

`ETF_PATH_PLAYBOOK.md` says what to buy after each signal and how to size
it in stock. To decide whether **options** are the better expression you
need the signal-conditional *expected move* — how big |forward return| is
over the next 3–6 months given the event — against what the option market
charges for that window. This study computes both sides for the 21
leaderboard ETFs: the empirical conditional distributions (no model), and
a live implied overlay from option chains.

Reproduce:
```
python etf_expected_move_study.py --cache-dir .ndx_dark_cache
```
Historical numbers are from the 2026-08-07 close; the implied snapshot is
whatever the chains show on the day you run it (`--no-options` skips it;
the script degrades to historical-only when the endpoint is unreachable).

## Definitions

- **E[|r|]** — mean absolute forward return at the horizon (63 / 126
  sessions), i.e. the payoff of an ATM straddle held to expiry: the
  *realized expected move*.
- **Move ratio** — conditional E[|r|] ÷ unconditional E[|r|] over all
  overlapping same-length windows. Above 1: the signal predicts a bigger
  move than normal. Below 1: the signal only tilts the direction of a
  normal-or-smaller move.
- **Fair premia** — expiry-style payoff values under the conditional
  distribution, in % of spot: ATM call = E[max(r,0)], ATM put =
  E[max(−r,0)] (they sum to the straddle), +5/+10% OTM calls, and the
  short-put loss odds P(r < −5%), P(r < −10%).
- **Implied move** — ATM straddle mid ÷ spot at the expirations nearest
  ~91 and ~182 calendar days (0.8 · IV · √T when quotes are unusable). A
  frictionless market ATM call or put costs roughly half of it.

## Result 1 — signals move the *direction*, barely the *size*

| family | move ratio, 63d | move ratio, 126d | read |
|---|---|---|---|
| Up-breaks | **0.82–0.98** (all 21 ≤ 1) | 0.78–1.09 | a chase signal predicts a *calmer*-than-normal quarter with an upward tilt |
| Down-breaks | **1.09–1.42** (all 21 ≥ 1) | 0.96–1.41 | capitulation predicts a genuinely bigger move, tilted up |
| Turns | GDX 0.77, KRE 0.99, EEM 1.00 (63d) | GDX 0.98, EEM 1.40 | turns start quiet; the size shows up in the second quarter |

This is the core options fact: **after an up-break, long premium is
structurally the wrong vehicle** — you'd be buying a straddle whose
realized payoff runs below even the unconditional base, while the market
prices at-or-above it. The edge is drift, and drift is expressed with
intrinsic-heavy structures. After a down-break the realized move genuinely
widens — but entry IV after capitulation is usually elevated too, so even
there the long-premium case rests on the market *under*-reacting, which
the snapshot below can check case by case.

## Result 2 — the call/put fair-value asymmetry

Because up-break distributions are shifted right with thin left tails, the
conditional fair value splits very unevenly across the strike (63d, % of
spot; market would price each side ≈ implied straddle ÷ 2):

| ETF (up-break) | fair straddle | fair ATM call | fair ATM put | P(r<−10%) |
|---|--:|--:|--:|--:|
| QQQ | 7.3 | **5.7** | **1.5** | 3% |
| VTV | 4.9 | 3.7 | 1.2 | 4% |
| VUG | 6.1 | 4.6 | 1.5 | 6% |
| XLK | 6.9 | 5.2 | 1.6 | 5% |
| SMH | 10.2 | 7.7 | 2.5 | 9% |
| GDX | 13.2 | 8.0 | 5.2 | 22% |

For the strong chasers ~75–80% of the straddle's conditional value lives
in the call. A symmetric market premium therefore systematically
overprices the put side after these events — **short ATM/−5% puts and call
spreads are the natural expressions**, and outright straddles the worst.
GDX is again the outlier: its up-break put side keeps real value (22%
odds of a >10% drawdown at the horizon), which is the options version of
"don't chase GDX unhedged."

## Result 3 — live snapshot: implied vs conditional (run date Aug 2026)

Implied ~3M straddle vs conditional E[|r|] 63d after an up-break
(imp3M/up63): the market is **rich against the post-up-break world for
essentially the whole universe** — XLF 0.98 and XLY 1.03 are the only
fair ones; the median ratio is ~1.4; IGV 2.10, SMH 1.92, XLU/VUG 1.65,
XLK 1.59 lead the rich list. Even against the *bigger* down-break moves,
most implieds still clear conditional fair value. Full table prints from
the script; it is a snapshot, not a stable property — rerun before using.

## Worked example — the live GDX up-break (2026-08-07, spot 89.89)

| | ~3M | ~6M |
|---|--:|--:|
| Implied move (ATM straddle) | 19.0% | 22.9% |
| Conditional E\|r\| (up-break) | 13.2% | 21.5% |
| Conditional fair ATM call | 8.0% | **14.1%** |
| Conditional fair ATM put | 5.2% | 7.5% |
| Market ATM call ≈ implied ÷ 2 | ≈9.5% | ≈11.5% |

Three months of GDX optionality is ~44% overpriced versus the
post-up-break history — but six months is priced **at fair** on the
straddle while the conditional *call* is worth ~2.5pp more than the market
side (the drift and the right tail live in the second quarter, exactly the
path map's flat-quarter-then-resolve shape). The structure this implies:
**own the 6M call (or ATM/+10% call spread), finance it by selling the
rich 3M downside** — which is also the options translation of the stock
playbook (half-size now, add at −8%): a short 3M put struck ~8% below spot
*is* the resting limit order, collected at a premium instead of placed for
free. Sized cash-secured, its assignment is the entry you wanted anyway.

## Structure selection by family (the summary rule)

- **Chaser up-breaks** (QQQ, VUG, XLK, SMH, IGV…): stock or ATM/+5–10%
  call spreads; sell puts only where P(r<−10) is single-digit (QQQ 3%,
  VTV 4%). Never buy the straddle — conditional move ratio < 1.
- **Round-tripper up-breaks** (GDX, KRE, XLE): short near-dated puts at
  the dip-entry strike (−5 to −8%) as the paid limit order; own the
  longer-dated call side only where 6M implied ≤ conditional (GDX today).
- **Capitulation** (all 21): the one family whose realized move beats the
  unconditional base — long option structures are *fair* whenever
  post-crash IV hasn't already spiked above the conditional numbers
  (compare per name); otherwise short puts monetize both the drift and
  the elevated IV.
- **Turns**: quiet first quarter (ratio ≤ 1) then wide second quarter —
  favor 6M+ tenors; 3M structures die before the move arrives.

## Caveats

- Payoff-at-expiry arithmetic: no early exercise, path P&L, rolls, or
  transaction costs; short puts carry assignment risk exactly at the q25
  excursions in the playbook — size them as stock entries, not premium.
- Overlapping event windows inflate effective n; conditional
  distributions cluster by regime (the playbook's era caveat applies —
  bear-era conditionals would look different).
- Implied side is a Yahoo snapshot: mids can be stale/wide (the IV
  fallback is an approximation), one ATM strike per tenor, no smile — a
  real fill needs live markets.
- The ≈ implied ÷ 2 market-call approximation ignores carry and skew;
  it's a screening yardstick, not an execution price.
