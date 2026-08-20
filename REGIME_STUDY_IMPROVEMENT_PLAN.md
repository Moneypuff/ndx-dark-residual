# Intra-index regime study — improvement plan (Tiers 1–3)

> **Status: IMPLEMENTED** (all ten phases, Aug 2026), plus one follow-up
> beyond the original plan: Phase 3's point-in-time tilt panel was
> extended to SPX (`data/spx_membership.csv`, 667 stints from Wikipedia's
> S&P 500 change log; `--point-in-time ndx,spx`). Result: the per-name
> tilt effect does **not** generalize to the S&P 500 (clean null, CI
> centered on zero, ~50% hit rate) — sharpening the NDX finding to
> "concentrated in mega-cap tech/growth names," not a broad-market
> phenomenon. IWM's basket remains current-membership only (no free
> point-in-time source found for Russell 2000 reconstitution history;
> official data sits behind an FTSE Russell subscription). The corrected
> results live in `INTRA_INDEX_REGIME_FINDINGS.md`; the SPX-ex-NDX
> crossed leg populates on the first nightly build carrying
> `spx.dix_ex`, and the forward-scoring log accumulates on
> `regime-scoring-data` once `regime_score.yml` starts firing. This
> document is kept as the design record; its "Rules" and "Decision log"
> still govern follow-up work.

Execution plan for hardening `intra_index_regime_study.py` /
`INTRA_INDEX_REGIME_FINDINGS.md` against the four-referee critique
(econometrics, data/domain, practitioner, robustness). The critique's
verified defects, in one line each:

- **Era confound**: expanding percentiles on the non-stationary dollar-DIX
  turn DIX zones into calendar labels (headline cell ≈ "2023 vs 2026";
  excess over own-era mean ≈ +0.2pp; hit 82% → 57% within-year).
- **Episodes, not days**: regimes arrive in 3–24 contiguous episodes;
  21-day block CIs materially overstate precision (episode-cluster CI on
  "3-of-3 dispersed" includes zero).
- **Tilt "holds OOS" is partly circular**: 66% of LowCorr days are 2024+;
  the pre-2024 holdout CI includes zero; post-2024 the spread is negative
  in *every* regime, so "regime-specific" vs "era" is unidentified. The
  Q5/Q1 buckets are quasi-static (defensives vs growth leaders) and the
  panel is survivorship-censored (current members only).
- **NDX-vs-SPX contrast confounded**: different windows/warm-ups, two
  near-identical mega-cap gauges (corr +0.96), never tested crossed.
- **Mechanism overclaimed**: "dark accumulation" is not what FINRA
  off-exchange short volume measures; index-level (+) and per-name (−)
  signs are incoherent under it.
- **~200 untracked comparisons**, no print gates, several n=1..30 cells
  quoted; hazard "precursors" partly mechanical (distance-to-cutoff) and
  in-sample; day-t DIX is not tradeable at day-t close.
- **Nothing operational**: no rules, no risk columns, regime state not in
  the nightly build.

What the critique left standing (protect these while refactoring): the
regime ordering (survives within-year excess and ex-COVID/2022), the tilt
spread's *sign* (survives momentum- and beta-neutralization at −1.4 to
−1.7pp), corr-not-just-vol for the NDX gradient (concentrates in mid-vol),
the within-year 6-of-6 sign consistency of "3-of-3 dispersed", and the
"correlation spikes are jumps, not creeps" null.

---

## Rules (read first, apply throughout)

1. **Branch**: all work on `claude/dix-regime-comovement-scw5zm`. Never
   push to `main`; the scoring log (Phase 10) pushes only to its own
   orphan data branch, mirroring `optsnap.yml`.
2. **Dependencies**: numpy + pandas only. No scipy, no statsmodels — the
   repo implements its own stats (see `ols_nw`, `block_boot_ci`); new
   estimators (cluster bootstrap, cluster-robust OLS) follow that pattern.
3. **Offline NDX path stays intact**: `--indices ndx` must keep running
   against `docs/index.html` alone. Anything needing a fetch (membership
   panels, baskets, ATS data) degrades to a printed skip, never an error.
4. **Tests per phase**: every new pure function gets synthetic-panel tests
   in `tests/test_intra_index_regime.py` (style of the existing 30);
   `python -m pytest` green before each commit.
5. **Print gates are global**: once Phase 2 lands, no conditional mean or
   CI is printed anywhere (3×3s, tapes, entries, OOS, transitions) below
   **42 scored days AND 5 distinct episodes** (entries: 10 events). Cells
   below the gate print counts and `--`.
6. **Findings doc edits land with their phase**, not in a final pass —
   each phase's acceptance criteria include the doc reflecting its result.
7. Keep the existing CSV/report surface backward compatible where cheap
   (`intra_index_regimes.csv` gains columns; existing ones keep names).

---

# Tier 1 — fixes that can flip the headlines

## Phase 1 — Rolling-window zoning (kill the era confound)

**Why**: expanding percentiles on drifting series (dollar-DIX yearly mean
0.41→0.49→0.43) make zone membership a calendar label; this is the
critique's most consequential mechanical flaw.

**Changes** (`intra_index_regime_study.py`):
- New helper `rolling_pctile(s, window=504, min_obs=EXP_MIN)`: percentile
  (0..1, mid-rank on ties) of each value within its **trailing 504
  sessions** (~2 years), NaN until `min_obs`. Same loop structure as
  `expanding_pctile`, sliced window.
- `zones_30_40_30` gains `basis="rolling"` using it; module constant
  `ROLL_WIN = 504`.
- `assemble_frame` adds `cz_roll` / `dz_roll` columns alongside the
  existing bases.
- `report_index` promotes **rolling** to the headline basis: the per-index
  block prints rolling first, expanding second, full-sample last (Phase 9
  later moves full-sample to an appendix section of the doc).
- Entry study and OOS run on the rolling basis (OOS keeps train-fitted
  *cutoff levels* as a second, stricter variant — both printed).
- New diagnostic in `dix_by_corr_table`: per-cell **year composition** —
  a `years` column like `20:31% 23:28% 25:22%` (top-3 shares). This is the
  direct check that zones no longer sort by era.

**Tests**: rolling percentile has no look-ahead (append-invariance, as in
`test_zones_expanding_no_lookahead`); equals expanding when
`window >= len(series)`; a series with a level shift older than the window
re-normalizes (a value that is "High" vs 2020 history is "Mid" vs 2025
history — construct explicitly).

**Acceptance**: study runs all three indices; in the headline NDX LowCorr
row, no DIX zone's top year exceeds ~50% share (print it either way);
findings doc headline table replaced by the rolling-basis numbers with the
year-composition line quoted.

**Effort**: small. No new data.

## Phase 2 — Era- and episode-honest inference layer

**Why**: day-counts overstate information ~20×; block CIs ignore episode
clustering; ~200 cells with no multiplicity discipline; several quoted
cells are 1–4 episodes.

**Changes** (`intra_index_regime_study.py`):
- `episode_ids(zone_series)`: integer id per contiguous same-zone run
  (calendar-contiguity on the frame's own index; a gap in the frame does
  not split a run — match `episodes()` semantics).
- `cluster_boot_ci(r, ids, B=BOOT_B, seed)`: resample whole episodes with
  replacement (draw `n_episodes` episodes per replicate, concatenate,
  mean). Returns (nan, nan) below **5 episodes**. This becomes the CI
  printed for every conditional mean; the 21-day block CI stays in the CSV
  as `blk_ci_lo/hi` for continuity.
- `excess_vs_period(r1m, index_r1m_all)`: per-day forward return minus the
  same-calendar-year unconditional mean of that index's r1m. Printed as
  `excess` next to every conditional mean. **Label it "era-adjusted
  (diagnostic — uses full-year info)" in both report and doc**; it is a
  lens, not a signal.
- Per-year sign string for headline cells (`+ + - + 0`, years with <10
  days as `.`), and leave-one-year-out min/max mean for the two flagship
  cells (NDX LowCorr×DIXHigh; NDX LowCorr tilt spread).
- **Print gate** (Rule 5) implemented once in `fmt_stats` /
  `dix_by_corr_table` / `entry_report` / `oos_report`: below gate → counts
  + `--`. `n_episodes` column added everywhere `n_days` appears.
- Multiplicity note: the report header states how many conditional cells
  the run prints; the two flagship cells are marked `PRIMARY` and
  everything else `descriptive` (formal FDR is overkill for a text report;
  the gate + labeling is the enforceable part).

**Tests**: `episode_ids` on constructed zone runs; `cluster_boot_ci`
degenerate (<5 episodes → NaN) and coverage on synthetic clustered data
(episode-level effect variance → wider CI than `block_boot_ci`, assert
strictly wider on a constructed case); `excess_vs_period` arithmetic;
gate behavior (41 days or 4 episodes → `--`).

**Acceptance**: every printed mean carries (n_days, n_episodes, excess,
cluster CI); the "3-of-3 dispersed" table in `cross_index_report` gains
episode counts + cluster CIs + within-year excess (per the robustness
referee, expect the CI to include zero — the doc's claim is rewritten to
the within-year sign consistency, which is what actually held).

**Effort**: small-medium. No new data. Depends on Phase 1 (zones feed
episode ids).

## Phase 3 — Point-in-time tilt panel + factor-neutral spread

**Why**: the study's most-promoted result is estimated on a
winner-conditioned cross-section; its buckets are quasi-static style
baskets; its "OOS" is era-circular. The neutralized variants *survive* —
make them the primary number, on an honest panel.

**Changes**:
1. **Membership file** `data/ndx_membership.csv` (`ticker, added,
   removed`, ISO dates, `removed` empty = current). New script
   `fetch_ndx_membership.py` builds it from the public Nasdaq-100
   reconstitution history (Wikipedia's changes tables as primary source),
   normalized via `to_yahoo_symbol`, then **manually reviewed and
   committed** — the CSV is the source of truth; the fetcher is a
   regeneration aid, never run in CI. Cross-check row counts against the
   known ~5–10 changes/year; spot-check 2020/2023 reconstitutions against
   press releases before committing.
2. **Panels for ever-members**: new `--point-in-time` mode in the study
   builds the tilt panel via `ndx_dark_residual.build_universe_panels(
   ever_members, start, end, cache_dir=..., ns="pit")` — FINRA daily files
   are whole-market, so departed names re-filter the **same cached
   files** (no extra FINRA downloads; use a distinct cache namespace so
   the nightly cache is untouched). Prices: Yahoo still lists most index
   *deletions* (they keep trading); true delistings (acquisitions) that
   Yahoo lacks are logged and counted in the report (`dropped_pit` list).
   Expect ~130–150 tickers total.
3. **Membership masking**: tilt and per-name forward returns are NaN
   outside each name's `[added, removed)` window; the expanding tilt mean
   also only accrues inside it.
4. **Factor-neutralized spread**: extend `tilt_spread` with an optional
   `groups` argument (dict name→bucket). Neutralizations, each reported:
   momentum (halves of trailing 63d return, computed per day), beta
   (halves of 126d beta vs the proxy), sector (payload `sector_map`;
   pipeline TICKER_SECTOR for departed names, `NA` bucket allowed).
   Spread = equal-weight average of within-bucket Q5−Q1. Primary quoted
   number = **point-in-time, momentum+beta-neutralized, LowCorr row**.
5. **Concentration disclosure**: top-10 most-frequent Q5 and Q1 names with
   day counts, plus the 21d cross-sectional rank autocorrelation of tilt.
6. **Era honesty** (uses Phase 2): tilt table gains pre-2024 / post-2024
   split per regime, and the doc's "holds OOS" sentence is replaced by the
   two-sided statement the robustness referee wrote: *negative in LowCorr
   pre-2024 (CI including zero), strongly negative everywhere post-2024*.

**Tests**: membership masking (a name's tilt/returns NaN outside its
window on a synthetic panel); grouped `tilt_spread` (construct a case
where the raw spread is negative but vanishes within groups — assert the
neutralized number is the within-group one); rank-autocorrelation helper.

**Acceptance**: study prints raw vs point-in-time vs neutralized spreads
side by side with episode-cluster CIs; findings doc §"Which group rallies"
rewritten around the neutralized point-in-time number; survivorship caveat
converted from prose to measured delta (raw-current-members minus
point-in-time).

**Effort**: the largest single phase (new data source + fetch + review).
Network needed once to build panels; thereafter cached. Depends on Phase 2
for reporting.

---

# Tier 2 — identification

## Phase 4 — Common-window, crossed-input NDX/SPX test (+ SPX-ex-NDX DIX)

**Why**: "the gradient is an NDX phenomenon" currently rides on different
windows, warm-ups and two 0.96-correlated gauges.

**Changes**:
- Study section `crossed_report(frames)` run when both NDX and SPX frames
  exist: restrict both to their **common dates**, re-zone on the common
  window (rolling basis), then print the LowCorr DIX gradient for the four
  crosses — (NDX gauges, NDX DIX)→QQQ, →SPY; (SPX gauges, SPX DIX)→QQQ,
  →SPY. `assemble_frame` already takes `dix` and `r1m` independently, so
  this is frame plumbing, not new math. Interpretation table in the doc:
  which leg (gauge universe, DIX universe, outcome) moves the gradient.
- **SPX-ex-NDX DIX** (pipeline change, `ndx_dark_residual.py` main): where
  the S&P panels exist, also compute
  `compute_dollar_dix(SP["short"], SP["total"], SP["close"],
  exclude=set(NDX names)|{bench})` and pack it as `P["spx"]["dix_ex"]`
  (plus a `dix_ex_names` count). The study uses it as a third DIX input in
  the crossed section when present; prints a skip line on older payloads.
  This tests whether SPX's flat gradient is dilution by non-tech names.

**Tests**: crossed frame construction (synthetic two-index fixture:
identical gauges, different DIX — assert the crossed table attributes the
gradient to the DIX leg); pipeline: `compute_dollar_dix` exclusion already
covered — add a payload-shape test that `dix_ex` survives
`build_reconstructed_index_payload`-style packing (pure function level).

**Acceptance**: doc's headline #1 rewritten to whatever the crosses show —
pre-commit to the decision rule: *if NDX-DIX→SPY reproduces ≥half the
gradient and SPX-DIX→QQQ shows none, the claim becomes "the NDX dark-flow
gauge carries the signal"; if the common window flattens NDX to SPX
levels, the claim is retired to "one gauge, one window".*

**Effort**: small in the study; the pipeline half ships with the next
nightly build (payload regenerates automatically).

## Phase 5 — Vol-parallel panel + nested regression

**Why**: corr↔vol ≈ 0.8; the doc claims comovement conditioning without
showing the vol-only competitor.

**Changes** (study): standing section per index — (a) the 3×3 with
**realized-vol terciles** (rolling basis on `rv`) replacing AVG_CORR;
(b) nested NW table `r1m ~ zRV` vs `r1m ~ zRV + zCORR` vs full interaction
(report ΔR² and the added-term t); (c) double sort: DIX gradient within
vol-tercile × corr-tercile cells that pass the print gate (expect mostly
`--` at 3×3×3 — that's informative too).

**Tests**: none beyond a smoke assertion (section renders with gates) —
all math reuses tested pieces.

**Acceptance**: doc gains a "Comovement vs volatility" subsection stating
where corr adds to vol (per the robustness referee: the NDX gradient
concentrates in mid-vol; regime *means* stay vol-entangled) — with the
numbers from this run, not the critique's.

**Effort**: small. Depends on Phases 1–2.

## Phase 6 — Realistic timing (one-day DIX lag) + latest-day guard

**Why**: FINRA publishes day-t files after the close; day-t DIX zones are
not tradeable at day-t close. Latest-day gauge reads can sit on incomplete
basket data.

**Changes** (study):
- `assemble_frame(..., dix_lag=0)` parameter; a **paired run** per index
  prints the headline 3×3 and entry table at `dix_lag=1` next to lag-0,
  with a delta column. (Forward return stays the payload r1m at t — the
  lag makes the *signal* stale, which is the conservative direction; note
  this in the doc rather than re-deriving t+1-open returns the payload
  doesn't carry.)
- Latest-day completeness guard: the "latest" line and current-regime
  read use the last date where ≥95% of basket names have a close AND the
  DIX series has a print; later partial rows are excluded from gauges'
  "current" reporting (they remain in the frame).

**Tests**: lag plumbing (zone at t under lag-1 equals zone at t−1 under
lag-0); completeness guard on a synthetic panel with a ragged last row.

**Acceptance**: doc quotes the lag-1 numbers for both flagship cells; if
the NDX entry cell (+3.17%) degrades materially, the entry table's claim
is rewritten. All "current read" lines carry their as-of date from the
guard.

**Effort**: small.

## Phase 7 — Transition/hazard redo (episode-clustered, boundary-controlled)

**Why**: the precursor table's strongest feature is partly
distance-to-cutoff mechanics; cuts were in-sample; "100% last-one-
dispersed" is 3–5 episodes.

**Changes**: fold the scratchpad transition analysis into the study as an
optional `--transitions` section (one tool, not a second script):
- Label: `exit21` on mature (age ≥ 21d) LowCorr days, rolling basis.
- Estimator: **linear probability model with episode-clustered SEs** (new
  `ols_cluster(y, X, ids)` — OLS beta, sandwich with per-cluster score
  sums; numpy only). Regressors: the candidate features **plus the gap to
  the exit cutoff** (`cutoff − avg_corr`, from the rolling zone cutoffs)
  as a mandatory control, plus `tr21` when testing `d21_dix` (the
  robustness referee showed d21_dix works only in up-tapes — encode that
  interaction explicitly).
- Cut discipline: any tercile display fits cuts on pre-2024 only.
- Episode-level view: one row per episode (did it end within k of first
  reaching age 21; features at age 21) — the honest n is printed first.
- Keep, verbatim: the **jumps-not-creeps** result and the exit-to-Mid
  structure. Delete: every "100%" cell; replace with pooled
  3-index episode counts and a Clopper–Pearson interval.

**Tests**: `ols_cluster` recovers slope on synthetic clustered data and
its SE exceeds the naive OLS SE under injected cluster effects; boundary
control kills a constructed pure-random-walk "signal" (simulate a driftless
gauge, assert d5 slope loses significance once the gap is controlled).

**Acceptance**: a "Regime transitions" section lands in the findings doc
containing only gate-passing, boundary-controlled results; the chat-only
transition notes are superseded and deleted from the scratchpad workflow.

**Effort**: medium (one new estimator + section). Depends on Phases 1–2.

---

# Tier 3 — mechanism, presentation, operations

## Phase 8 — Mechanism language + retail-proxy test

**Why**: "dark accumulation" is unsupported; the index(+)/name(−) sign
split begs for the crowding/retail reading.

**Changes**:
- Doc-wide rewrite: "dark accumulation" → "off-exchange short-volume
  share"; one new paragraph stating the two-level sign structure and the
  candidate interpretations (SqueezeMetrics' passive-fill story vs retail
  internalization intensity), explicitly unresolved.
- Cheap test (data already in panels): per name, correlate tilt with the
  **off-exchange share of total volume** (offexch/total — retail-
  internalization proxy) and re-run the LowCorr spread **within halves of
  that share**. If the spread lives in the high-retail half, say so.
- Stretch (separate commit, may fail on network policy): weekly FINRA OTC
  Transparency ATS files to split ATS ("dark pool" proper) vs non-ATS
  (internalizer) share for a recent subsample; wire behind a flag with a
  graceful skip.

**Tests**: within-half spread reuses the Phase 3 grouped `tilt_spread`.

**Acceptance**: no instance of "accumulation" language survives in
findings/docstrings; the retail-half split is in the doc.

**Effort**: small (stretch item excluded).

## Phase 9 — Findings doc restructure + risk columns + deletion list

**Why**: presentation currently promotes selected extremes and reports
means without distributions; several cells should not be printable at all
(the Phase 2 gate removes most mechanically — this phase finishes the
editorial half).

**Changes**:
- Doc order per index: rolling-basis headline (with excess + episode
  counts) → OOS/lag-1 agreement flags *adjacent to each headline cell*
  (`OOS: agrees / mixed / reverses`) → expanding basis → full-sample
  demoted to a short appendix labeled *descriptive, look-ahead*.
- **Risk columns** for gate-passing cells: p10/p90 of forward return, and
  max adverse excursion within the 21-session window (min cumulative
  proxy return inside the hold, median and worst — computable from proxy
  closes already in the frames).
- Apply the deletion list: `+6.22% (n=30)`-class cells, all OOS cells
  under the gate, selective-selloff rows (kept only as the structural
  "barely exists" count), entry medians under 10 events, the SPX weekly
  tilt as "corroboration" (kept as an instrument-limitations note).
- "What the pattern says" rewritten to the post-critique grading: what
  survived (regime ordering; neutralized tilt sign; corr-beyond-vol for
  the gradient; jumps-not-creeps), what is era-entangled, what is retired.

**Tests**: none (doc + small report formatting); risk-column helpers get
unit tests (max-adverse-excursion on a constructed price path).

**Acceptance**: a reader encounters no number that fails the gate; both
flagship claims carry era-excess, episode count, cluster CI, lag-1 and
OOS flags within one table row.

**Effort**: small-medium. Depends on all earlier phases' numbers.

## Phase 10 — Pre-registered rules, forward-scoring log, dashboard surface

**Why**: six months of frozen-rule forward evidence outweighs every
in-sample cell; the regime state is the study's most useful daily output
and currently reaches no one.

**Changes**:
1. **`frozen_rules.json`** (committed, hash-stamped, freeze date = merge
   date). Exactly three rules, parameters fixed by the Tier 1/2 outputs
   (fill final cutbacks when freezing, structure now):
   - `ndx_tilt_screen`: when NDX rolling-basis comovement zone = Low,
     list the top-20%-tilt NDX names (point-in-time panel) as an
     avoid-screen; refresh weekly.
   - `ndx_dix_overweight`: NDX zone Low for ≥3 consecutive sessions AND
     rolling-basis DIX zone High on lag-1 data → 21-session QQQ
     overweight, no re-entry during hold.
   - `all_dispersed_derisk`: all three indices' zones Low → risk flag on
     until any exits.
2. **`build_regime_state.py`**: reads the payload + basket caches,
   computes per-index state (zone, basis values, episode age, N-of-3
   dispersed, DIX zone at lag 1, tilt screen list), emits
   `regime_state.json` + a compact HTML strip.
3. **Nightly integration** (`refresh.yml`): a step after the comovement
   build runs it; the strip is injected into `docs/regime_log.html`
   (extend `build_regime_log.py`'s existing strip mechanism) or shipped as
   `docs/regime_state.html` linked from there — decide in-PR by whichever
   template hook is smaller.
4. **Scoring log**: orphan branch `regime-scoring-data` (clone of the
   `optsnap.yml` pattern — `contents: write`, pushes only to that branch).
   Nightly append: one row per rule per day (state, triggered?, entry
   refs); rows whose 21-session window completed get their realized
   outcome filled. The study gains `--score-log <dir>` to summarize the
   accumulated live record.
5. Completeness guard from Phase 6 gates all of it (no state emitted off
   a partial session; the strip shows its as-of date).

**Tests**: rule-trigger logic on synthetic state histories (confirmation
window, cool-down, no re-entry during hold); scoring resolution (a
triggered row resolves exactly once, 21 sessions later, to the realized
proxy return); state JSON schema round-trip.

**Acceptance**: after one nightly cycle, the dashboard shows the per-index
regime line with an as-of date; the data branch holds dated rows; the
findings doc's "current read" section is replaced by a pointer to the live
strip. Rules are never edited after freeze — a changed rule is a new rule
id with its own start date.

**Effort**: medium. Independent of Tiers' analytics after Phase 1
(needs rolling zones + the lag guard); ship last so frozen parameters
reflect the corrected study.

---

## Sequencing and effort

| Order | Phase | Depends on | Size | Needs network |
|---|---|---|---|---|
| 1 | 1 Rolling zones | — | S | no |
| 2 | 2 Episode/era inference + gates | 1 | S-M | no |
| 3 | 5 Vol parallel | 1,2 | S | no |
| 4 | 6 Lag + latest-day guard | 1,2 | S | no |
| 5 | 4 Crossed inputs (+ pipeline dix_ex) | 1,2 | S (+nightly) | payload only |
| 6 | 3 Point-in-time tilt + neutralization | 2 | L | once (panels) |
| 7 | 7 Transition redo | 1,2 | M | no |
| 8 | 8 Mechanism + retail proxy | 3 | S | stretch only |
| 9 | 9 Doc restructure + risk columns | all above | S-M | no |
| 10 | 10 Frozen rules + nightly surface + log | 1,6 | M | nightly CI |

Phases 3–7 are mutually independent given 1–2 and can be reordered;
1→2 is strict; 9 consumes everything; 10 freezes last. Each phase is one
commit (or two where a pipeline change ships separately), tests green,
findings doc updated in the same commit.

## Decision log (already made — do not re-open)

- Rolling window = 504 sessions, min 250 obs; 30/40/30 split retained for
  cross-study comparability (cut-sensitivity 20/60/20 is a one-line
  robustness print in Phase 2, not a redesign).
- Cluster unit = contiguous same-zone episode; CI = percentile bootstrap
  over episodes; gates = 42 days AND 5 episodes (10 events for entries).
- Era-excess = same-calendar-year demeaning, labeled diagnostic.
- Membership source = committed CSV reviewed by hand; fetcher is offline
  tooling, never CI.
- Primary tilt number = point-in-time, momentum+beta-neutral, LowCorr row,
  episode-cluster CI.
- Lag convention = signal lagged one session against unchanged forward
  window (conservative), not synthetic t+1-open returns.
- Transition estimator = linear probability + episode-clustered sandwich
  SEs (no logit/IRLS — keeps the no-scipy rule and the coefficients
  readable as rate deltas).
- Scoring storage = orphan data branch (`optsnap-data` pattern), never
  commits to `main`.

## Open risks

- **Membership history accuracy**: Wikipedia change tables have known
  gaps pre-2015 — the study window needs 2018+, which is well covered;
  still, hand-verify the 2018–2020 rows before trusting the pre-2024
  holdout tilt numbers.
- **Delisted-name prices**: Yahoo lacks some acquired names; every drop is
  counted and reported. If drops exceed ~15% of departed names, add a
  secondary price source before quoting the point-in-time spread as
  primary.
- **FINRA cache pressure**: the point-in-time panel re-filters cached
  daily files under a new namespace — verify the Actions cache stays
  under budget before wiring anything into CI (the panel build is
  study-local, not nightly, so this only matters if Phase 10's tilt
  screen adopts it; if too heavy, the nightly screen falls back to
  current-member tilt with a label).
- **`dix_ex` payload growth**: one more float series (~1,660 values) —
  negligible, but confirm against the compressed-payload budget in
  `build_html`.
