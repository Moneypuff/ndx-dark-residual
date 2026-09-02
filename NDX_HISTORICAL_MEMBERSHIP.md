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

## Phase 2 — wired into the matched baseline (done)

The membership file is now packed into the dashboard payload
(`rel.member = {ticker: [[i0,i1), …]}`, position ranges into `rel.dates`;
built by `load_ndx_membership_ranges` in `build_html`) and consumed by a
**"point-in-time universe"** toggle (default **ON**) on the *Episodes vs
streaks* and *D-streak events* tabs. When on, the matched cross-sectional
baseline counts a name as a peer only on days it was an actual index member
(gated at the window's start day); a name absent from the file is treated as
always a member, and a payload without `rel.member` disables the feature.
Single-name event/episode returns are unchanged — only the baseline and the
`edge = mean − base` move.

**How large the bias was (live payload, 2018-08 → 2026-09, excess of QQQ):**

| horizon | baseline, all names | baseline, point-in-time | shift |
|--------:|--------------------:|------------------------:|------:|
| 21d | +0.52 | +0.08 | −0.45 |
| 42d | +1.08 | +0.14 | −0.95 |
| 63d | +1.67 | +0.18 | −1.49 |

The all-names peer average beat QQQ by ~1.5pp/63d out of pure hindsight; the
point-in-time baseline sits at ~0, which is the efficient-market null. This was
almost entirely **look-ahead inclusion** — names that had price data but were
not yet in the index on the early dates.

**What it does to the headline edges** (exit-anchored, 63d, excess of QQQ):

| signal | edge, all-names base | edge, point-in-time base |
|--------|---------------------:|-------------------------:|
| 5-day-MA high-D | −1.48 | **−0.00** |
| raw 1-day high-D | −0.61 | **+0.92** |

The "post-episode short signal" was a **baseline artifact** — it vanishes under
an honest universe. The raw-signal high-D episode instead shows a small
*positive* post-exit excess that the biased baseline had masked (still needs
clustered inference before it is called real). The shipped JS reproduces these
numbers exactly (verified end-to-end in a browser against the live payload).

## Exited-name backfill — wired (populates on the next live refresh)

The **80 names that left** the index in-window (≈30% of all member-days: BIIB,
ANSS, ATVI, SPLK, FISV, …) are now fetched as **price-only** columns and merged
into the baseline:

- `ndx_exited_members()` derives them from this file (ever-a-member, not a
  current constituent); `fetch_exited_price_panels()` pulls their Yahoo close /
  adjusted close best-effort and merges them into the return panels *only* —
  the DIX aggregate, contributors and small-multiples grids keep reading the
  untouched live-constituent panels. On (default; `--no-exited-baseline` skips).
- These names carry **no dark series**, so their all-NaN `d` column drops out of
  `rel["d"]`: they enter the study **solely as baseline peers**, never as
  event/episode subjects, and are gated by their membership ranges like everyone
  else.
- **Safety:** `TICKER_VALID_FROM` masking is applied so a recycled symbol can't
  splice a predecessor, and any name whose Yahoo history doesn't overlap its own
  membership window is dropped rather than trusted. Delisted / renamed / acquired
  tickers that Yahoo can no longer resolve (e.g. ATVI, CELG, SGEN, XLNX, FB→META)
  simply return nothing and are skipped — **never fabricated**. The build logs
  the resolved-vs-dropped count.

Until the next refresh runs the fetch, the baseline still spans surviving names
only; and because Yahoo cannot resolve every exited ticker, coverage will remain
partial. Both directions of the residual gap only *lower* the baseline (exits
skew to laggards), so the corrected edges stay **conservative**.

### Durable equity-price store (so the fetch isn't re-run / re-rate-limited)

The exited-name prices (and, going forward, the whole Yahoo equity pull) are
persisted in a dedicated store, **kept separate from the ORATS options duckdb**:

- `equity_store.py` — a duckdb table `equity_eod(ticker, date, close, adj_close,
  volume)` (default `~/.ndx_dark_cache/equity_prices.duckdb`) with a committed
  columnar mirror **`data/equity_prices.parquet`** so it survives ephemeral
  sessions and imports into any database with one `read_parquet(...)`. A guard
  refuses to open anything named like `orats.duckdb`.
- `load_yahoo_panels` now reads through this store and writes fresh pulls back
  (both guarded — no duckdb ⇒ silent no-op), so once a symbol's history is on
  disk it is **never re-queried from Yahoo**, which is what triggered the 429s.
- `fetch_equity_prices.py` — resumable fetch (default target = the exited
  members). It skips symbols already stored, upserts incrementally, and on a
  hard 429 writes a resume-state file and exits 75 so it can be re-run
  (`--resume`) once the limit clears. A membership-window overlap check drops
  recycled/renamed tickers that return wrong-era data (e.g. a reused `FB`),
  never fabricating.

**Current state:** the fetch has run — **58 exited names stored** with real
history (delisted/renamed/recycled tickers such as ATVI, CELG, XLNX, SGEN, FB
correctly skipped). Measured effect of adding them to the point-in-time
baseline: **63d baseline +0.18 → −1.15** (the true equal-weight member basket
*underperformed* cap-weighted QQQ), so the honest edges move a further ~1.3pp in
the favourable direction. The dashboard picks this up on the next refresh, when
`fetch_exited_price_panels` reads the now-populated store instead of hitting the
Yahoo rate limit.

## Remaining

- **DIX aggregate & cross-sectional L/S ranking.** The reconstructed dollar-DIX
  and the *Cross-sectional L/S* decile ranking still use the full name set each
  day; restricting *those* to point-in-time members (a ranking change, not a
  baseline de-bias) is a separate follow-up.
- **Dark series for exited names.** Only price is backfilled, which is all the
  matched baseline needs. Adding their FINRA dark history (so they could also be
  event/episode subjects, not just peers) would require extending the recycled-
  ticker alias map and is out of scope here.
