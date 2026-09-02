# NDX historical membership (point-in-time)

Point-in-time Nasdaq-100 membership for the dark-flow study's window
(2018-08-01 → present). Built to replace the survivorship-biased practice of
applying **today's** constituent list backwards — the flaw that inflates every
"matched peer" baseline in the single-name event tabs (a survivor basket beats
QQQ by ~1pp/month over 2019–2026, which is not a real edge, it is hindsight
selection).

This is **Phase 1: the data artifact only.** Wiring it into the panels, the
cross-sectional baselines and the placebo resamples is a later phase (see
"Next phases" below). Nothing in the study reads this file yet.

## File

`data/ndx_historical_membership.csv` — one row per **membership stint**:

| column    | meaning |
|-----------|---------|
| `ticker`  | symbol as it traded **during that stint** (Yahoo convention: class shares use `-`) |
| `added`   | effective index-change date the stint began (ISO `YYYY-MM-DD`) |
| `removed` | effective date the stint ended; **empty = still a member** |

Membership test: a ticker is a member on date `d` iff `added <= d < removed`
(`added` inclusive, `removed` exclusive). One company can have several rows —
a genuine re-add, or an in-window ticker rename encoded as
`remove(old) + add(new)` on the rename date so every stint's symbol is valid
for its own window (e.g. `FISV → FI`, `WLTW → WTW`).

Current shape: **188 stints, 182 distinct tickers.**

## Provenance

Reconstructed from Wikipedia's Nasdaq-100 change log
(`Historical_components_of_the_Nasdaq-100`) by taking the current component
list and replaying the change log **backward** to 2018-08-01, opening and
closing stints as it goes. `fetch_ndx_membership.py` regenerates the file
deterministically (network fetch of the two Wikipedia pages; a re-run is
byte-identical). It is committed alongside the data so the artifact is
reproducible, not a black box.

Effective dates are index-change dates (the annual December reconstitution,
Nasdaq's special rebalances, and ad-hoc M&A / delisting replacements), not the
announcement dates.

## Verification (run against the saved file)

- **Structural:** 0 malformed rows, 0 `removed <= added`, 0 overlapping stints
  within a ticker.
- **Counts on checkpoint dates:** 101–103 securities on every June/annual
  checkpoint from 2018 to 2026 — within the [98, 104] sanity band. The count
  sits slightly above 100 because of the dual-class pair GOOG/GOOGL plus brief
  overlap windows around replacements.
- **Reconciles with our own data.** Every one of the 102 constituents in the
  live dashboard payload is covered by this file. The strongest independent
  check: **EA** is marked removed on **2026-08-04**, and EA's FINRA dark-ratio
  series in the payload stops on **exactly** that date. The removal date was
  not fitted to the data — it came from the change log and the data agreed.
- **More accurate than the repo's static list.** `NDX100_WEIGHTS` in
  `ndx_dark_residual.py` (a hand-maintained 2026-07-18 snapshot) still carries
  EA; this file correctly retires it. When Phase 2 wires membership in, this
  file is the authority for *when* a name was in the index; the static list
  remains only the source of approximate index *weights*.

## Known caveats (spot-check before relying on the edges)

- **Two sub-two-week stints** — `SOLS` (2025-10-30 → 2025-11-06) and `VSNT`
  (2026-01-05 → 2026-01-09). These look like change-log quirks (a same-week
  rename/correction recorded as add+remove). **Neither has a FINRA panel in our
  data**, so they do not affect the study either way; they are flagged here
  rather than silently deleted, since the source recorded them.
- Dates are calendar effective dates; if a database load needs trading-day
  alignment, snap `added`/`removed` to the next session on ingest.
- Wikipedia is the single source. It is good for the liquid Nasdaq-100 but is
  not an exchange-of-record; a licensed constituent history (e.g. from the
  index provider) would supersede this file if one becomes available.

## Next phases (not done here)

1. Have `build_universe_panels` restrict each date's cross-section to that
   date's members, so a name contributes to the DIX aggregate and the peer
   baseline only while it was actually in the index.
2. Rebuild the matched cross-sectional baseline and the circular-shift placebo
   off the point-in-time membership, then re-report the single-name event and
   episode edges. The survivor-biased peer base should collapse toward zero.
3. Backfill FINRA/price panels for names that **left** the index inside the
   window (currently absent), so exited members are not silently dropped.
