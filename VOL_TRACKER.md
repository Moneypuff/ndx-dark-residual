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

**IVs are re-derived, not trusted.** Yahoo's IV field is inverted off a
spot-like forward and splits same-strike call/put IVs by up to ~9 vol pts
at LEAP tenors. `forward_smile` extracts each expiry's implied forward
and discount from the chain itself (regressing tight C−P mids on strike:
slope = −D, zero-crossing = F — no rate/dividend assumptions), then
re-inverts every tight-quoted contract's mid via Black-76 against that
forward; quoteless-but-open contracts fall back to the feed IV, then the
despike. Same-strike call/put IVs agree again (residual split ~0.5 vol
pt), ATM is measured at the forward, and 25Δ wings use forward deltas.
The day-over-day fixed-strike panel keeps feed IVs (a stable bias cancels
in differences); every *level* comparison — rr, ATM, marks — uses the
forward-consistent smile.

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
- History starts the day capture starts. There is no backfill.
