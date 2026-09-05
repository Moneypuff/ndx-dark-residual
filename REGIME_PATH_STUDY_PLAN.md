# Regime path study — design (corr × DIX cells, cross-index states, rule states)

> **Status: DESIGN — not yet implemented.** Companion to
> `REGIME_STUDY_IMPROVEMENT_PLAN.md` (implemented, Aug 2026). That study
> answers "what is the mean 1-month return inside each regime combination".
> This one answers the question that a mean cannot: **how volatile is the
> path between entry and exit, how deep does it go before it pays, how often
> does a stop get hit before a target, and does the regime's own volatility
> expand or contract while you hold.** Every decision below is made; an
> implementer follows the phases in order, runs the tests after each, and
> keeps the guardrails in "Rules". Numbers in this document are layout
> placeholders (`—`), never results.

Reproduce (once implemented):
```
python regime_path_study.py                                  # all three indices
python regime_path_study.py --indices ndx                    # offline, payload only
python regime_path_study.py --csv regime_paths.csv --risk-csv regime_path_risk.csv \
    --envelopes regime_path_envelopes.json
```

---

## Why a path study, and why the mean is not enough

`intra_index_regime_study.py` reports, per index, a 3×3 grid of comovement
zone × DIX zone with the mean 21-session forward return, an episode-cluster
CI, and — for the two flagship cells only — one risk line (fwd p10/p90 and
the median/worst max adverse excursion). That is enough to rank regimes; it
is not enough to trade one:

- A +2.5% mean at 35 annualized vol (NDX HighCorr) and a +2.2% mean at 15
  vol (NDX LowCorr×DIXMid) are different products. The first spends part of
  most holds 8–15% under water; the second rarely does. The grid does not
  say which.
- The corrected findings say the informative leg is **LowCorr×DIXLow sitting
  ~2.4pp below its era**. Whether that is a *drawdown* effect (paths go down
  and stay down), a *chop* effect (paths oscillate and finish flat) or a
  *missing-upside* effect (paths simply fail to rally) decides whether the
  right response is to cut size, to sell premium, or to widen stops. A mean
  cannot separate the three; a path distribution can.
- The transition section established that exits from dispersed tapes into
  panic tapes are **jumps, not creeps** and are not forecastable from the
  gauges. The cost of that jump risk to someone holding through a LowCorr
  hold has never been priced: how often does a HighCorr transition land
  inside a 21-session hold, and what does the path look like when it does?
- Position sizing in this repo (`ETF_PATH_PLAYBOOK.md`) is anchored on the
  q25 max adverse excursion, not the mean. The regime cells have no such
  number outside two flagship lines.

So the deliverable is, for every regime combination the repo already
defines, the **distribution of the forward path**: fan quantiles by horizon,
adverse/favorable excursions and their timing, barrier-touch and
target-before-stop probabilities, forward realized vol against trailing and
implied vol, and the regime's own survival inside the hold — with the same
episode-honest inference and print gates the corrected study runs under.

---

## Rules (read first, apply throughout)

1. **Branch**: all work on `claude/regime-path-study-design-1xuaui` (or the
   implementation branch cut from it). Never push to `main`; nothing here
   writes to the scoring data branch except the optional Phase 6 scorer
   change, which follows the existing `regime_score.yml` pattern.
2. **Dependencies**: numpy + pandas only. No scipy, no statsmodels. New
   estimators follow the repo's hand-rolled pattern (`cluster_boot_ci`,
   `ols_cluster`). Tests may use scipy as an oracle (already in
   `requirements-dev.txt`).
3. **Offline NDX path stays intact**: `--indices ndx` must run against
   `docs/index.html` alone (QQQ closes are in the payload). SPX/IWM need
   the Yahoo cache; the implied-vol leg (Phase 4) needs the Cboe CDN. Every
   fetch degrades to a printed skip, never an error.
4. **Reuse, don't duplicate.** Frames, zones, episode ids, cluster CIs,
   entry events, print gates and the completeness guard come from
   `intra_index_regime_study`. Path-matrix helpers exist in
   `gdx_chase_study.py` but are per-event loops; this study builds one
   vectorized panel per index (Phase 1) and never re-implements zoning.
5. **Print gates are global** (the study's rule 5): no conditional
   statistic prints below **42 scored anchors AND 5 distinct episodes**
   (ENV family), **10 events** (ENTRY family), **5 episodes**
   (EPISODE-HOLD family). Below the gate a cell prints counts and `--`.
6. **Tradeable timing is the headline basis.** Anchors are the close of
   day t with the comovement zone known at t (`cz_roll`) and the DIX zone
   on the one-session-lagged signal (`dz_roll_l1`). The lag-0 DIX grid is
   not re-run here — the terminal-return comparison already exists in the
   study; paths are for trading.
7. **Pre-specified hypotheses**: exactly three PRIMARY path claims (below);
   everything else is descriptive and labeled so in the report header.
8. **Tests per phase** in `tests/test_regime_path.py`, synthetic panels
   only (style of `tests/test_intra_index_regime.py`); `python -m pytest`
   green before each commit. Findings-doc edits land with their phase.
9. **CSV/JSON outputs are committed** like `intra_index_regimes.csv`;
   they carry the payload's `generated` date and the as-of of the last
   complete session.

---

## What already exists (read before writing code)

| File | What this design takes from it |
|---|---|
| `intra_index_regime_study.py` | `load_payload`, `build_ndx_frame`, `build_basket_frame` (frames with `cz_roll`, `dz_roll_l1`, `vz_roll`, `rv`, `tr21`, `r1m`, `mae21`, `r1m_ex`); `run_ids` (episode ids), `cluster_boot_ci`, `entry_events`, `mask_stats`/`fmt_cell` (gate rendering), `year_mix`, `per_year_line`, `loyo_range`, gate constants, `IDX_PROXY`, `OOS_SPLIT`. |
| `gdx_chase_study.py`, `etf_path_study.py` | The excursion vocabulary (MAE/MFE, trough/peak day, dip5/dip8, mean−median skew) and the playbook's sizing rule (weight per 1% NAV at q25 MAE). Adopt the names; do not import the per-event loops. |
| `build_gex_dispersion.py` | `fetch_text_cached` + `parse_cboe_csv` + `CBOE_HISTORY_TMPL` — the Cboe CDN daily-history reader used for COR1M/DSPX; VIX, VXN and RVX ship as the same file shape (Phase 4). |
| `build_regime_state.py` / `frozen_rules.json` | The live state dict (`zone`, `dz_roll_l1`, rule flags). Phase 6 attaches the current cell's path envelope to the strip. |
| `scripts/score_regime_log.py` | The forward log records daily closes; Phase 6's optional extension resolves a realized min-path from them, which forward-tests the envelopes. |
| `tests/test_intra_index_regime.py` | Testing idiom: `_bdays`, hand-built frames, `pytest.approx`, no disk fixtures. |

One small change to the study itself (Phase 1): `build_ndx_frame` and
`build_basket_frame` add `meta["proxy_close"]` — the proxy's daily close
series on its own full calendar (QQQ packed closes from the payload; SPY/IWM
Yahoo adjclose). `assemble_frame` trims rows where the gauges are undefined,
so the path panel must be built on the proxy's own calendar and then
*reindexed* to the frame's anchor dates — never on the frame's index.

---

## Design overview: one engine, pre-specified cells

```
frame M (per index) ──┐
                      ├──> cell masks ──> anchor sets (3 families) ──┐
proxy close ──> forward_path_panel (dates × 0..63) ─────────────────┤
                fwd_realized_vol (dates × {21,63})                  ├──> per-cell stats ──> report / CSV / JSON
trailing rv_t, VXN_t ───────────────────────────────────────────────┘
```

### The path panel

`forward_path_panel(close, horizon=63)` → DataFrame indexed by the close's
own dates, columns `0..horizon`, value = `close[t+h] / close[t] − 1` in %,
NaN where the window runs past the data. Built with `shift(-h)` (vectorized,
~2,000 × 64 floats per index — trivial). `fwd_realized_vol(close, h)` →
annualized std of the daily returns `t+1..t+h`, NaN when incomplete. Both
are pure functions; every statistic below is a mask over their rows.

Horizon **H = 63** sessions (one quarter, the playbook's comparison
horizon). Checkpoints **(1, 2, 3, 5, 10, 15, 21, 42, 63)**. The
**21-session hold is the primary window** for every excursion, barrier,
vol and survival statistic (it is the rules' horizon and the study's
forward-return definition); 63 is reported alongside as the secondary
window. A path is scored at checkpoint h only when complete to h; the last
63 sessions of every sample are therefore unscored at h=63 and the last 21
at h=21 — counts are printed per horizon.

### Anchor families

| Family | Anchor | Hold | Independent unit / CI | Gate |
|---|---|---|---|---|
| **ENV** (environment) | every day t in the cell | fixed h | episode (`run_ids` of the mask), `cluster_boot_ci` | 42 anchors & 5 episodes |
| **ENTRY** (trigger) | first day the cell forms, 21-session cool-down (`entry_events`) | fixed h | event (≥21 apart; plain percentile bootstrap over events, labeled) | 10 events |
| **EPISODE-HOLD** (strategy) | close of the **3rd consecutive** session in the cell (confirmation; DIX zones flicker daily) | until the close of the first session the cell condition fails, capped at 63 | episode | 5 episodes |

ENV describes the environment (what the study's grid describes, now as a
distribution). ENTRY is comparable to the study's entry section. EPISODE-
HOLD is the only family that resembles a position: variable duration, exit
known at the close it happens, plus a **post-exit leg** (the 21 sessions
after the exit) so the reader sees what the regime's ending costs or pays.
Its rows report duration (median/q75/max), terminal return, MAE, MFE, hold
realized vol, and the post-exit 21-session return.

### The cells (masks), per index, rolling basis

| # | Cell family | Count | Printed |
|---|---|---:|---|
| 1 | comovement zone marginals: LowCorr / MidCorr / HighCorr | 3 | yes |
| 2 | comovement zone × DIX zone (lag-1) | 9 | yes |
| 3 | realized-vol zone (`vz_roll`) × DIX zone (lag-1) — the vol-parallel | 9 | CSV; one comparison block |
| 4 | cross-index: N-of-3 indices in LowCorr, N = 0..3, each of QQQ/SPY/IWM as outcome | 4×3 | yes (common dates) |
| 5 | rule states: `ndx_dixlow_caution_v1` active vs LowCorr-not-active; `all_dispersed_derisk_v1` active vs not | 4 | yes (named rows; subsets of 2 and 4) |

That is 12 printed cells × 3 families per index plus 12 cross-index rows
and 4 rule rows — with the gates doing the pruning. The report header
states the count of conditional cells printed, as the study does.

### PRIMARY hypotheses (pre-specified; the only three claims the findings doc may headline)

- **P1 — Jump risk in dispersed tapes.** In NDX **LowCorr** (marginal, ENV),
  the share of 21-session vol-scaled outcomes beyond 2σ of *trailing* vol
  (`|r21 / (rv_t·√(21/252))| > 2`) exceeds the Gaussian 4.6% **and** the same
  share in HighCorr. Decision rule: quote "trailing vol understates
  dispersed-tape path risk" only if the episode-cluster CI on the LowCorr
  share excludes 4.6%; otherwise write "no excess tail beyond trailing vol".
- **P2 — The DIX-Low leg is a drawdown effect, not only a mean effect.** NDX
  **LowCorr×DIXLow(l1)** has a worse q25 MAE21 than **LowCorr×DIXHigh(l1)**.
  Decision rule: the cluster-bootstrap CI on the difference of q25 MAEs
  excludes zero → the findings say "the caution flag is a drawdown flag";
  includes zero → "mean effect only; do not resize on it".
- **P3 — Panic tapes: worst excursions, decaying vol.** NDX **HighCorr**
  (marginal, ENV) has the worst q25 MAE21 of the three zones **and** a
  median forward/trailing vol ratio below 1 with a cluster CI excluding 1.
  Decision rule: both hold → "size HighCorr on MAE, expect vol to decay
  through the hold"; only the MAE part holds → "size on MAE; no vol claim".

All three are evaluated at lag-1 DIX timing, rolling basis, ENV family, on
the payload's full NDX window, with the 2024+ split printed beside them.

---

## Statistics catalogue (what "how volatile the paths can be" means here)

Every (cell, family) row carries the blocks below; primaries and marginals
print all of them, the 3×3 cells print A–D and G, the rest goes to CSV.

**A. Fan** — quantiles p5/p10/p25/p50/p75/p90/p95 of the cumulative return
at each checkpoint, plus mean and hit. This is the chartable object.

**B. Excursions (inside 21; secondary 63)** — MAE: median, q25, p10, worst,
median trough day; MFE: median, q75, p90, median peak day; **give-back** =
MFE − terminal (median): how much of the best point a fixed-horizon holder
returns; share of paths with MAE below −3/−5/−8/−12%.

**C. Barriers and brackets (close-to-close paths)** — P(touch −x within 21)
and P(touch +y within 21) for x, y ∈ {3, 5, 8, 12}; a **bracket matrix**
for stop x ∈ {3, 5, 8} × target y ∈ {3, 5, 8}: P(target first), P(stop
first), P(neither), and the bracket's expected return (+y / −x / terminal),
at 21 and 63 sessions. This is the block that decides whether a stop helps
or amputates in a given regime. Close-only paths understate intraday
touches — stated in every table title.

**D. Forward volatility** — fwd rv21 (annualized): median, p90; **vol
expansion ratio** fwd rv21 / trailing rv21 at t: median and share > 1;
**variance ratio** VR(21) = var(21-session return) / (21 · var(daily)) over
the cell's anchors (>1 trending, <1 mean-reverting inside the regime);
**vol-scaled terminal** z = r21 / (rv_t·√(21/252)): share |z| > 1 and > 2
(Gaussian benchmarks 31.7% / 4.6%); Phase 4 adds **implied**: VXN (NDX),
VIX (SPX), RVX (IWM) at t vs fwd rv21 — median implied − realized (the
regime's vol risk premium) and the share of holds where realized exceeded
implied.

**E. Daily microstructure (day level, not paths)** — annualized vol, skew,
excess kurtosis, lag-1 autocorrelation, worst and best day, share of
|r| > 2%, up-day fraction, all over the cell's days. Cheap and diagnostic:
it says whether a regime's path risk is gap-driven or grind-driven.

**F. Survival inside the hold** — P(cell still active at h) for h ∈ {5, 10,
21, 42, 63} from ENV anchors; for LowCorr cells, P(HighCorr reached within
21) and the MAE/terminal distribution **conditional on that transition vs
not** — the price of the jump risk the transition section left unpriced.

**G. Sizing translation (the playbook's rule)** — position weight that loses
1% NAV at the q25 MAE21 and at the p10 MAE21; reward/risk = median r21 /
|q25 MAE21|. Reported so the numbers land in the same units as
`ETF_PATH_PLAYBOOK.md`.

**H. Era and out-of-sample honesty** — per-year line of median r21 and
median MAE21 (`per_year_line` on both columns); pre-2024 / 2024+ split for
the three primaries; leave-one-year-out range of q25 MAE21 for the
primaries; year mix of every printed cell (`year_mix`).

Inference: `cluster_boot_ci` gains an optional `stat` callable (default
`np.mean`, backward compatible) so episode-resampled CIs exist for the
median MAE, the q25 MAE, a touch probability, the vol-ratio median and the
tail share. Primaries print those CIs; CSV carries them for every gated
cell. Differences between two cells (P2) resample both cells' episodes
jointly per replicate.

---

# Phases

## Phase 1 — Path engine + ENV family + fan / excursion / barrier blocks

**Changes**
- `intra_index_regime_study.py`: `meta["proxy_close"]` in both frame
  builders (Series on the proxy's own calendar); `cluster_boot_ci(...,
  stat=np.mean)`.
- New `regime_path_study.py`, pure functions under the usual banner:
  - `forward_path_panel(close, horizon=63)`, `fwd_realized_vol(close, h)`.
  - `excursion_stats(P, h)` → dict of block B for rows of a path panel.
  - `barrier_touch(P, h, levels)` → touch shares; `bracket_outcomes(P, h,
    stops, targets)` → first-passage matrix and expected bracket return
    (first day a close ≤ −x or ≥ +y; the earlier day wins; same-day
    impossible on closes).
  - `fan_quantiles(P, checkpoints, qs)` → long frame (h, q, value, n).
  - `cell_masks(M)` → ordered dict name → boolean Series for cell families
    1–3 (and 5's index-local rule row); cross-index masks come in Phase 5.
  - `cell_stats(P, rv, mask, ids, family="ENV")` → one flat dict (blocks
    A–D, G) plus counts and gate; `render_cell(...)` mirrors `fmt_cell`.
- CLI: `--html`, `--indices`, `--basket-size`, `--cache-dir`, `--refresh`,
  `--horizon` (default 63), `--no-ci`, `--csv`, `--risk-csv`. Report: per
  index, the marginals then the 3×3 (lag-1), ENV family only.

**Tests** — path panel on a constructed close series (known values, NaN
tail); excursion trough/peak days on a hand-built path; barrier first-
passage ordering (stop before target, target before stop, neither); bracket
expected return arithmetic; fan quantiles on a panel with a planted
outlier; gate behavior (41 anchors or 4 episodes → `--`); reindex safety (a
frame with an interior gap must not shift horizons — construct one).

**Acceptance** — `--indices ndx` runs offline; every printed cell shows
n anchors / n episodes / gate; the NDX LowCorr and HighCorr marginals print
blocks A–D and G. **Effort: medium.**

## Phase 2 — ENTRY and EPISODE-HOLD families, survival, post-exit leg

**Changes**
- `entry_anchors(mask)` wraps `entry_events` (unchanged semantics);
  `episode_hold_paths(close, mask, confirm=3, cap=63)` → per-episode rows
  (start, exit, duration, terminal, MAE, MFE, hold rv, post-exit r21).
- `survival(mask, ids, h)` → P(still in cell at h) from ENV anchors;
  `transition_within(M, mask, h, to="HighCorr")` → boolean per anchor, used
  to split the excursion stats (block F).
- Report gains the three-family layout per cell and a one-line survival
  strip per zone marginal.

**Tests** — episode-hold on a synthetic zone series (confirmation delays
the entry by two sessions; exit lands on the first failing close; cap
honored; a 2-day episode never enters); survival on constructed runs;
transition split reproduces hand-counted shares.

**Acceptance** — the report prints ENV / ENTRY / EPISODE-HOLD blocks for
each zone marginal and the 3×3; post-exit leg shown for EPISODE-HOLD.
**Effort: small-medium.**

## Phase 3 — Volatility layer and the corr-vs-vol comparison

**Changes** — vol-scaled terminals and tail shares, variance ratio, daily
microstructure (block E), vol expansion ratio distribution; the vol-
parallel cells (`vz_roll × dz_roll_l1`) computed into the CSV and one
printed comparison block per index: for each DIX zone, MAE q25 / touch −5%
/ vol ratio inside LowCorr vs inside VolLow (the same honesty the study's
vol-parallel section owes the reader). P1 and P3 become computable here.

**Tests** — variance ratio = 1 on iid synthetic returns (tolerance), > 1 on
an AR(1) with positive autocorrelation; vol-scaled tail share ≈ Gaussian
benchmark on Gaussian synthetic data; microstructure block on a series with
a planted −5% day.

**Acceptance** — primaries P1 and P3 print with cluster CIs and the
decision-rule verdict; the comparison block prints for every index.
**Effort: small.**

## Phase 4 — Implied-vol leg (VXN / VIX / RVX), graceful skip

**Changes** — `load_cboe_vol(index_key, cache_dir)` using
`build_gex_dispersion.fetch_text_cached` + `parse_cboe_csv` on
`CBOE_HISTORY_TMPL.format("VXN" | "VIX" | "RVX")`; align to anchors; per
cell: median implied − realized (annualized points), share of holds with
realized > implied, and the same split by 2024+. Any fetch/parse failure
prints `implied leg: skipped (...)` and the rest of the report is
unchanged. Flag `--no-implied` for offline runs.

**Tests** — parse of a hand-written Cboe-shaped CSV (OHLC and single-value
shapes) and alignment to a frame with a missing date; the skip path.

**Acceptance** — NDX block shows the VXN comparison when online; offline
run prints the skip line. **Effort: small.**

## Phase 5 — Cross-index and rule-state cells, era/OOS, findings doc, committed CSVs

**Changes**
- `n_dispersed(frames)` → Series on common dates (the study's
  `cross_index_report` logic, factored out); cross-index cells run the ENV
  and EPISODE-HOLD families for each outcome proxy.
- Rule rows: `ndx_dixlow_caution_v1` active vs LowCorr-not-active (NDX);
  `all_dispersed_derisk_v1` active vs not (all three proxies) — these are
  the path versions of the frozen rules and are labeled with the rule ids.
- Block H everywhere it applies; P2 computed (difference CI).
- `REGIME_PATH_FINDINGS.md` written in the corrected-study voice: the three
  primaries with verdicts first, then per-index marginals, the 3×3 risk
  tables, cross-index, rule rows, the corr-vs-vol block, the implied leg,
  caveats. A **"how to use this"** section translates the blocks into the
  three decisions the study is for: size (block G), stop-or-no-stop (block
  C's bracket matrix), and vol posture (block D).
- Commit `regime_paths.csv` (long: index, family, cell, h, n, mean, hit,
  p5…p95) and `regime_path_risk.csv` (one row per index × family × cell:
  blocks B–G plus CIs, year mix, gate).

**Tests** — `n_dispersed` on a three-frame synthetic; rule-row masks equal
the study's rule definitions (construct a frame where they differ from the
plain cell by the lag).

**Acceptance** — a reader meets no ungated number; each primary carries
its verdict, cluster CI, year mix, 2024+ split and LOYO range in one table
row; CSVs regenerate byte-identically from the same payload (seeded).
**Effort: medium.**

## Phase 6 — Operational surface: envelopes on the state strip, forward test in the log

**Changes**
- `--envelopes regime_path_envelopes.json`: per index and per (zone ×
  DIX-lag-1) cell, ENV family: p10/p50/p90 at h = 5, 10, 21; MAE q25;
  P(touch −5% in 21); vol expansion ratio median; weight per 1% NAV; n,
  episodes, gate, as-of, payload `generated`. **Committed**, regenerated
  on demand (not nightly — the estimates are in-sample and dated on
  purpose).
- `build_regime_state.py` reads the JSON when present and adds an
  "expected path envelope for the current cell" line per index with the
  estimate's as-of date; absent file → no line, no failure.
- Optional, same pattern as `regime_score.yml`: `scripts/score_regime_log.py`
  resolves `mae21` (min of the log's own recorded closes over the next 21
  rows) next to the realized return in `outcomes.csv`. Deterministic from
  `state.csv`, so the branch stays append-only + one regenerated file. This
  is the forward test of the envelopes: after ~6 months the study's
  `--score-log` can compare realized MAEs against the frozen q25 lines.

**Tests** — JSON schema round-trip; the state page renders with and without
the file; the scorer's `mae21` on a hand-built log (a −4% row inside the
window; the horizon-stretch on a missed night is documented, not tested
away).

**Acceptance** — `docs/regime_state.html` shows an envelope line with an
as-of date after the next nightly build; the scoring branch gains the
column without changing existing rows. **Effort: small-medium.**

---

## Report layout (one printed cell; placeholders, not results)

```
--- NDX  LowCorr x DIXLow (lag-1)   ENV: n=— anchors / — episodes   years 26:—% 24:—% 25:—%   gate ok
 fan (%)      h=1     3     5    10    15    21    42    63
   p10        —      —     —     —     —     —     —     —
   p50        —      —     —     —     —     —     —     —
   p90        —      —     —     —     —     —     —     —
   mean/hit   —/—%   ...
 excursion (21): MAE med — q25 — p10 — worst —  trough d—   MFE med — q75 —  peak d—   give-back —
   share MAE < -3/-5/-8/-12: —% —% —% —%
 barriers (21, closes): touch -3/-5/-8/-12: —% —% —% —%   touch +3/+5/+8/+12: —% —% —% —%
   bracket P(target first) stop\target  +3    +5    +8      E[bracket ret]
                             -3         —     —     —       —
                             -5         —     —     —       —
                             -8         —     —     —       —
 vol: fwd rv21 med — p90 —   fwd/trail ratio med — (>1 in —%)   VR21 —   |z|>1 —%  |z|>2 —%   VXN-realized —
 survival: in cell at d5/10/21/42/63: —% —% —% —% —%   -> HighCorr within 21: —%  (MAE q25 if so — vs —)
 sizing: per 1% NAV at q25 MAE — %, at p10 MAE — %   RR —
 epCI: mean r21 [—,—]  MAE q25 [—,—]  touch-5 [—,—]  vol ratio [—,—]
 ENTRY: n=— events   r21 mean — hit —%   MAE q25 —   touch -5 —%
 EPISODE-HOLD (confirm 3): n=— episodes   dur med — q75 — max —   terminal med — hit —%   MAE q25 —   post-exit r21 med —
```

Cells below a gate print the first line with counts and `-- (below gate)`.

## CSV / JSON schemas

- `regime_paths.csv` (long): `index, family, cell_type, corr_regime,
  dix_regime, h, n, n_eps, gate, mean, hit, p5, p10, p25, p50, p75, p90,
  p95`.
- `regime_path_risk.csv` (wide, one row per index × family × cell): counts
  and gate; `mae_med, mae_q25, mae_p10, mae_worst, trough_d, mfe_med,
  mfe_q75, peak_d, giveback_med, dip3, dip5, dip8, dip12, touch_m3 …
  touch_p12, br_{x}_{y}_target, br_{x}_{y}_stop, br_{x}_{y}_ret` (21 and
  63 suffixes), `rv21_med, rv21_p90, vratio_med, vratio_gt1, vr21, z_gt1,
  z_gt2, ivrp_med, iv_exceeded`, `surv_5 … surv_63, to_high_21,
  mae_q25_if_high, mae_q25_if_not`, `size_q25, size_p10, rr`, CI columns
  `ci_lo/hi` for mean r21, mae_q25, touch_m5, vratio_med; `years,
  per_year_r21, per_year_mae, pre24_*, post24_*` for primaries.
- `regime_path_envelopes.json`: `{schema: 1, generated, asof, gate_days,
  gate_episodes, indices: {NDX: {"LowCorr|DIXLow": {n, n_eps, gate, p10:
  {5,10,21}, p50: {...}, p90: {...}, mae_q25, touch_m5, vratio_med,
  size_q25}, ...}, ...}}`.

---

## Sequencing and effort

| Order | Phase | Depends on | Size | Needs network |
|---|---|---|---|---|
| 1 | Path engine + ENV + A–D, G | — | M | no (NDX) |
| 2 | ENTRY + EPISODE-HOLD + survival | 1 | S-M | no |
| 3 | Vol layer + corr-vs-vol block (P1, P3) | 1 | S | no |
| 4 | Implied leg | 1 | S | Cboe CDN, skippable |
| 5 | Cross-index + rules + era/OOS + findings + CSVs (P2) | 1–4 | M | Yahoo cache for SPX/IWM |
| 6 | Envelopes on the state strip + scorer column | 5 | S-M | nightly CI |

1 → 2/3/4 are independent given 1; 5 consumes all; 6 ships last so the
committed envelopes reflect the finished study. One commit per phase, tests
green, findings doc updated in the same commit from Phase 5 on.

## Decision log (made — do not re-open)

- **Cells** = the study's rolling-basis zones, DIX on the lag-1 signal, plus
  N-of-3 dispersed and the two evaluable frozen rules. No new regime
  definitions; a path study of a different partition is a different study.
- **Horizon** 63 with the 21-session hold primary; checkpoints (1, 2, 3, 5,
  10, 15, 21, 42, 63).
- **Three families**: ENV (every day), ENTRY (formation, 21 cool-down),
  EPISODE-HOLD (confirm 3, exit on first failing close, cap 63, post-exit
  21 leg). ENTRY keeps first-day semantics for comparability with the
  study's entry section; confirmation belongs only to the strategy family.
- **Paths are close-to-close** on the proxy (QQQ payload packed closes,
  split-adjusted, dividend-unadjusted — ~0.1% drift per quarter, accepted;
  SPY/IWM Yahoo adjclose). No intraday, no open-to-open synthetic returns.
- **Barrier grid** absolute {3, 5, 8, 12}%; bracket stops/targets {3, 5, 8}.
  Sigma-scaled risk appears only in the vol-scaled tail shares.
- **Vol reference** for scaling and the expansion ratio = the frame's own
  trailing 21-session realized vol (`rv`), not implied; implied is a
  separate, skippable leg.
- **Inference** = episode-cluster percentile bootstrap (`stat` argument);
  gates 42/5 (ENV), 10 (ENTRY), 5 (EPISODE-HOLD); pairwise differences
  resample both cells jointly.
- **Sizing units** = the playbook's (weight per 1% NAV at q25 MAE; p10 as
  the stricter line).
- **Envelopes are committed and dated**, not recomputed nightly; the
  forward log, not a re-estimate, is what updates the reader's belief.

## Open risks

- **Overlap.** ENV anchors inside one episode share most of their path;
  quantiles are fine as descriptions, but every CI is only as wide as the
  episode count allows (15–27 HighCorr episodes per index, 6–14 in the
  HighCorr×DIXLow cells → expect `--` in parts of the 3×3's HighCorr row).
  Say so rather than relaxing the gate.
- **Regime flicker.** DIX-zone episodes are short (59 episodes in 267
  NDX LowCorr×DIXMid days — under five sessions each on average);
  EPISODE-HOLD on 3×3 cells will be short and thin.
  The family's natural home is the zone marginals, N-of-3 and the rules;
  the 3×3 rows print whatever passes the gate.
- **Close-only barriers** understate touch frequencies; a −5% close-touch
  share is a floor on the intraday one. Every barrier table title says so.
- **One macro cycle.** Path shapes in 2020 and 2022 dominate the HighCorr
  rows; the per-year lines and the 2024+ split are mandatory beside every
  primary, and the envelopes carry the as-of date for that reason.
- **Payload tail.** The last 63 sessions are unscored at h=63; the state
  strip's envelope therefore describes history, never the current hold's
  realized path — that lives in the scoring log.
- **Cboe CDN policy.** VIX/VXN/RVX histories may be unreachable under the
  runner's network policy; the leg is designed to skip, and the findings
  doc must say "implied leg unavailable" rather than omit the section.
