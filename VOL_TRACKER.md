# Fixed-strike vol tracker — design and status

Tracks the big open-interest strikes day by day with their IVs, to read
**who is the aggressor** at each strike and to monitor option trades
entered after a regime-log signal fires. Companion to
`EXPECTED_MOVE_FINDINGS.md` (the static snapshot analysis this makes
dynamic).

## Why fixed strikes

As spot moves, ATM IV changes mechanically — the smile slides under the
spot. A **fixed strike's** IV change is the market genuinely re-pricing
that insurance. The clean per-contract object is the *local re-pricing*:

    dIV_local = dIV(fixed strike) − dIV(ATM, same expiry)

## The aggressor read (ΔOI × ΔIV)

For a big-OI strike, day over day:

| ΔOI | local ΔIV | read |
|---|---|---|
| rises | rises | buyer-initiated opening — paying up at the strike |
| rises | falls | **seller-initiated opening** — overwriters/note desks leaning on it |
| falls | falls | longs closing out |
| falls | rises | shorts covering — a squeeze at the strike |

Aggregated (ΔOI-weighted local ΔIV over the top-N OI strikes, per side)
this becomes a daily per-symbol **speculative-pressure index**. Two
mechanics are load-bearing: Yahoo's OI is **T+1** (lag ΔOI one day
against ΔIV before classifying), and contracts are keyed by
`(symbol, expiry, strike, right)` for their whole life — never chained
across expiries, so rolls can't fake a flow.

## Capture (phase 1 — LIVE)

- `snapshot_option_chains.py` + `data/optsnap_universe.csv` (edit the CSV
  to change coverage): the 22 leaderboard ETFs + SPY/KWEB/FXI and the
  single-name list (AAPL TSLA GOOGL NVDA AMZN MSFT META AMD MU CRWV IONQ
  OWL RKLB ONDS SPCX DRAM).
- Expiry policy: nearest 2 listings, the monthly (nearest third Friday)
  per month out to 9 months, and **every January expiration at any
  horizon** — the LEAPs where structured-note flow (autocallable
  barriers, buffered notes, note-hedge call stacks) concentrates. First
  capture confirmed the thesis: the largest Jan-2027+ OI lines are IGV
  puts at 68–78% moneyness, deep-ITM NVDA calls, META call stacks.
- Contract policy: alive contracts (bid or OI) within ±25% moneyness,
  widened to **±65% for January LEAPs** (note barriers sit 30–50% under
  spot), plus the top-20 OI strikes per side unconditionally.
- Storage: one `optsnap/YYYY-MM-DD.csv.gz` (~0.7 MB, ~34k rows) per
  trading day on the **`optsnap-data` branch**, appended by
  `.github/workflows/optsnap.yml` (6:30pm ET, DST-gated like
  refresh.yml). Main's history stays clean; the workflow bootstraps the
  branch on first run and pushes only there. Yahoo has no chain history —
  this capture is the only unrepeatable step in the repo.

## Derived analytics (phase 2 — CODE LIVE, outputs accrue with data)

`build_vol_tracker.py`: fixed-strike IV series per contract, per-expiry
ATM series, local re-pricing, the ΔOI×ΔIV classification, per-symbol
pressure indices, and the big-OI strike map (the standing structured-note
positions and their drift). Output degrades honestly with history: day 1
gives entry marks and the OI map, IV deltas start with snapshot 2, the
lagged aggressor read with snapshot 3, and the pressure index is worth
reading after ~5.

`build_vol_surface.py` renders the same capture as a rotatable 3D
implied-vol surface (`docs/vol_surface.html`, linked from this page's
strip): each captured expiry's live despiked smile resampled onto a
shared 60%–140% moneyness grid and stacked by time to expiry, with a
date scrubber over the last 10 snapshot days. No extrapolation — a grid
cell outside a smile's own strike range ships as a hole, not a
flat-carried edge.

## Risk reversals & skew (delta pillars)

The vol tracker page also charts per-symbol **delta-pillar skew history**:
IVs at the 10Δ put / 25Δ put / 50Δ (ATM) / 25Δ call / 10Δ call pillars,
interpolated to fixed **30d and 90d** constant maturities. Derived series:
RR25 = 25Δ call IV − 25Δ put IV, RR10 likewise, ATM = the 50Δ level (all
in vol points; negative RR = the put wing is bid). The skew card draws the
five pillars as a smile — today vs ~1 week and ~1 month ago.

Method is the same end to end regardless of data source: our own
Black-Scholes implied vol from the quoted mid → our own BS delta
(`trade_structures.bs_delta`) → linear interpolation **in delta space**
per OTM side (put side below spot, call side at/above, live + despiked) →
linear-in-IV interpolation across the two expiries bracketing the tenor.
A pillar whose target delta falls outside the side's observed span is
NaN — no extrapolation, gaps are honest.

Data channel: `build_rr_history.py` (LOCAL-ONLY, needs the ORATS duckdb)
computed the deep history once — 2007+ for the whole universe — and its
output lives as `optsnap/rr_history.csv.gz` on the **optsnap-data**
branch, fetched by the nightly build alongside the snapshots. The build
recomputes the same pillars from each live snapshot day and splices the
two series at the first snapshot date (the live pipeline wins). The page
payload keeps daily resolution for the last 2 years and W-FRI weekly
before that.

## Signal linkage (phase 3 — CODE LIVE)

Same script: for each live regime-log signal (≤63 sessions), the playbook
structure (ETF_PATH_PLAYBOOK class → `STRUCTURES`) is resolved to actual
listed contracts on the first snapshot at/after the event date — e.g. the
Aug 2026 GDX up-break resolves to long Jan-2027 ATM call / short +10%
wing / short Nov −8% put — and each leg is tracked daily: entry vs
current fixed-strike IV, OI drift, and (once flows exist) the aggressor
read at that strike. Entries that predate the first capture are marked
`entry proxied`. Rendering into the docs artifact next to the regime log
is deliberately deferred until a couple of weeks of data make the page
worth deploying (`--out-dir` already writes the CSVs the page will
consume).

## Structure suggester + trade log (LIVE, in the scanner)

`trade_structures.py` makes the findings docs' structure-selection table
executable. Per live signal, `suggest_structure()` maps (family, playbook
class, implied-vs-conditional richness, risk reversal) to legs specified
in **sigma-move units** (σ = ATM IV·√T), resolved to listed strikes at
suggestion time: chaser up-breaks get short put spreads (−0.5σ/−1.5σ,
~3M) when premium is rich and the put wing carries it, call spreads
(ATM/+1σ, ~6M) when fair; round-trippers get the dip-financed call
spread (long 6M ATM / short +1σ wing / short 3M −1σ put = the paid dip
limit); capitulation gets long calls only when implied < conditional,
else short put spreads; turns get long-tenor call spreads (ATM/+1.5σ)
and are refused outright in XLF/KRE.

The **trade log** is a deterministic replay, not mutable state:
`build_vol_tracker.py` re-derives entries (first snapshot at/after each
event, flagged when proxied) and exits (event + 63 sessions, the
playbook horizon) from snapshot history on every run, writing
`trade_log.csv` with entry/exit/current marks and P&L in % of spot.

**Marking** never trusts a closing quote: each leg is priced off the
**despiked live smile** (interpolated IV of neighboring live strikes →
Black-Scholes, r=0), and the quote only bounds the mark when the market
is tight (spread ≤ 25%) — wide or stale closing markets are exactly what
the surface mark exists to overrule. Marks are valuations for the log,
not executable prices; each leg's quoted spread is stored alongside.

## Calibration (phase 4 — after a quarter)

Does the pressure index lead follow-through? Score it against the
path-map outcomes (ETF_PATH_PLAYBOOK.md) with honest n, the way the
guide backtests everything else.

## Known limits

- Close-of-day marks only; ΔOI×ΔIV is an inference about aggression, not
  a trade tape. OI is T+1 and occasionally restated.
- Corporate actions re-key strikes (adjusted contracts appear as new
  series); phase 2 must detect strike-grid discontinuities and quarantine
  affected series.
- Wing IVs need the liveness filter (dead quotes carry garbage IVs).
- The endpoint is Yahoo's crumb-gated chain API — proven from the dev
  environment, to be validated from GitHub runners via workflow_dispatch;
  if runners are throttled, the fallback is running the same script on a
  schedule elsewhere and pushing to the same branch.
- History starts the day capture starts, with one exception:
  `backfill_optsnap_from_orats.py` can extend it backward from a local
  ORATS options-history duckdb when one is available (not part of the
  CI pipeline — a local, one-off tool). `build_rr_history.py` is the
  same idea for the delta-pillar RR/skew series (full 2007+ history,
  pushed as `optsnap/rr_history.csv.gz` on the data branch).
- The live snapshots cap strikes at ±25% moneyness, so on high-vol
  names the 10Δ (especially 90d) pillar can sit outside the captured
  smile — the RR10 series gaps honestly from the first live date on
  those names while the ORATS-derived history (full strike range)
  fills it before that.
- Every `iv` in a loaded snapshot is OUR OWN Black-Scholes (r=0) implied
  vol (`load_snapshots` → `recompute_iv` → `trade_structures.implied_vol`),
  inverted from the quoted price (mid bid/ask, or last trade when the
  market isn't two-sided) — never the vendor's own impliedVolatility/mid
  vol field. This is what keeps a Yahoo-captured day and an ORATS-backfilled
  day on one consistent methodology instead of carrying a seam wherever
  the sources meet. The big-OI strike map (`big_oi_map`) also drops any
  strike whose expiry has already passed real calendar time as of the
  build, not just the last snapshot date, so a stale build (a skipped
  nightly run, a weekend) can't show an already-expired contract's last
  OI as if it were still live.
