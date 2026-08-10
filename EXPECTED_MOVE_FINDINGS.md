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

## Result 4 — skew: is the upside already positioned?

> **Errata (Aug 2026):** the wing IVs below were read from Yahoo's IV
> field, which is inverted off a spot-like forward — same-strike call/put
> IVs split by +3.5 vol pts at ~3M up to ~9 at LEAPs, inflating long-tenor
> call wings. The live pipeline (`trade_structures.forward_smile`) now
> re-inverts every IV against the chain's own implied forward (put-call
> parity regression), which removes the artifact; the vol-tracker page's
> 25Δ heatmap carries the corrected numbers. Direction of this table's
> conclusions survives; long-tenor magnitudes are overstated.

The same chains give the smile. Wings are measured at the ±1
**sigma-move** strikes (σ = ATM IV·√T, so "one expected move" is the same
yardstick for a 15-vol and a 47-vol ETF): `put_skew` / `call_skew` = wing
IV − ATM IV, and **rr = call wing − put wing** — equity smiles normally
run rr well below zero (downside protection over upside lottery).
Positioning reads: put/call open interest and the share of call OI struck
≥ +0.5σ. Snapshot (Aug 2026, ~3M tenor, liquid names):

| posture | names (rr, vol pts) | read |
|---|---|---|
| **Upside already bid** | **GDX +4.3** (put wing *below* ATM, 31% of call OI ≥ +0.5σ, 47% at 6M), EEM +1.7, XLU +0.6 | the market is paying up for calls — the crowd is positioned for the move |
| Normal put skew | QQQ −4.0, IWM −4.3, SMH −3.8, XBI −4.0, XLE −4.3, XLF −2.3 | no upside crowding; put wing carries the premium |
| Steep put skew | VUG −8.7, VTV −6.5, XLK −6.7, KRE −8.9 | downside fear still dominates the smile |

(Thin sector chains — XLB, XLC, XLY, XLRE, XLV — print wing IVs from
near-dead strikes in this snapshot; treat their rows as unreliable.)

Two uses. First, **rr is the "already positioned?" dial the playbook
needs**: a chaser up-break with *normal* put skew (SMH, QQQ, IWM now)
means the option market has not pre-paid the move — short puts collect a
genuinely rich wing and calls are clean. A positive rr (GDX now) means
speculators beat you to the upside: outright calls carry an extra skew
tax on top of the level effect. Second, skew flips the *structure*, not
the *direction*: an inverted call wing makes **call spreads** better (the
short +1σ leg is sold at a skew premium) exactly where outright calls got
worse — and it cheapens the put side, so the short-put financing leg
collects less than the symmetric ≈implied/2 yardstick suggests.

For the live GDX trade this refines the §Worked-example structure: the
6M ATM/+10% call *spread* is now clearly preferred over the outright 6M
call (buy ATM at fair-ish, sell the +1σ wing 2 vol points over ATM into
the speculative bid), and the short 3M put leg — struck at the dip-entry
level — is collecting a slightly *below*-symmetric premium (put wing −2
vols under ATM), which is the options market telling you the same thing
as the 8.67 put/call OI ratio: the downside side of GDX is where the
hedgers are, not the speculators.

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
  it's a screening yardstick, not an execution price (Result 4 is the
  skew correction, where the chain is liquid enough to measure it).
- Open interest is two-sided: a fat put OI can be hedgers, put *sellers*,
  or spreads — rr (what the market *pays*) is the cleaner positioning
  read; OI corroborates, it doesn't prove.
- Skew has no history here (chains are a daily snapshot, no archive), so
  "inverted vs normal" is judged cross-sectionally and against the usual
  equity-smile prior, not against each ETF's own past skew.
