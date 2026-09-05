# Regime return paths — implementation plan

How does the **next month unfold** after each of the 27 NDX/SPX/IWM DIX
comovement regimes on `docs/comovement.html`? The existing tab reports the
mean / median / hit-rate of the 21-session forward return per regime. A mean
says nothing about the *road* to it: whether the month is quiet or violent,
whether it trends in one direction or chops back and forth, how deep the
drawdown along the way is, and how wide the cone of outcomes is. This plan
adds that path layer — as a reproducible study (`regime_paths_study.py`), a
findings write-up (`REGIME_PATHS_FINDINGS.md`) and a new section on the
comovement page — without touching the existing regime table.

The plan is written to be executed as-is. Decisions are made below; do not
re-open them without a reason recorded in the findings doc.

---

## 0. What the data already says (feasibility check, live payload 2026-09-04)

Run against the `PX` payload of the published page (1,673 common sessions,
2020-01-06 → 2026-09-04, 21 of 27 regimes observed). Two facts shape
everything that follows.

**(a) Regimes are choppy labels.** Counting contiguous runs of the same
3-letter code:

| | |
|---|--:|
| episodes (contiguous runs) | 549 |
| median episode length | 2 sessions |
| 75th percentile | 4 sessions |
| longest run | 31 sessions (LLL) |

So "the path after regime X" cannot be read as "the path after a month spent
in X". Most days in a cell are a day or two into a run that will flip
tomorrow. The study therefore has to score paths under **two explicit lenses**
(every-day and entry-day, §2.2) and report the persistence question itself as a
result, not hide it.

**(b) The path metrics already separate cells the mean does not.** Median
forward 21-session realized vol (annualized, %) and Kaufman efficiency ratio
(ER = |net move| ÷ Σ|daily moves|, 1 = straight line, →0 = pure chop) for the
NDX proxy (QQQ), every-day lens, largest cells first:

| regime | n | mean r21 | fwd rv | max DD | ER | preliminary read |
|---|--:|--:|--:|--:|--:|---|
| baseline (all days) | 1,652 | +1.8 | 19.4 | −4.3 | 0.21 | |
| MMM | 209 | +2.2 | 19.3 | −4.8 | 0.19 | baseline-like |
| HHM | 179 | +2.4 | **17.3** | −3.4 | **0.27** | quieter, trendier grind up |
| LLL | 174 | +1.5 | 19.0 | −5.1 | 0.19 | baseline-like |
| HHH | 163 | +2.2 | 19.4 | −4.5 | 0.24 | |
| MML | 133 | +3.3 | 17.6 | −3.4 | **0.15** | good mean, but *choppy* |
| LLM | 104 | +0.6 | **24.0** | −4.9 | 0.16 | volatile and choppy |
| MMH | 98 | +0.3 | 21.6 | −5.6 | 0.18 | |
| LLH (requested) | 96 | +2.5 | **24.4** | −4.9 | 0.22 | good mean earned through a *violent* month |
| HHL | 80 | +3.2 | **16.8** | −3.7 | 0.22 | quiet grind up |
| MHM | 43 | +2.9 | **15.7** | −2.9 | **0.30** | quietest, most trending cell |
| HML | 30 | +6.8 | 22.3 | −4.2 | 0.28 | strong trend *with* vol |
| HMH | 30 | −2.2 | **26.3** | **−8.2** | 0.23 | the violent-down cell |

Same computation for SPY and IWM shows the same ordering (SPX baseline rv 14.0,
HMH 20.5; IWM baseline 21.3, HML/HMH 24.5–24.9). These are every-day,
overlapping, in-sample numbers — the study's job is to make them honest
(episode-aware CIs, entry lens, vol-persistence control, expanding cutoffs)
and to put them on the page.

---

## 1. Rules

1. **Branch**: `claude/regime-return-paths-analysis-60x2q0`, rebased on
   `main` (`807db3c`, PR #51). Never push elsewhere.
2. **No new dependencies.** numpy + pandas only; the page stays vanilla JS
   with zero external scripts, like every template in the repo.
3. **Reuse, don't duplicate.** Payload loading, `build_aligned`,
   `expanding_decile`, `entry_events`, `block_boot_ci` and `ols_nw` come from
   `index_comovement_study.py`; `path_matrix` comes from `gdx_chase_study.py`;
   prices come from `N.load_yahoo_panels` exactly as `build_comovement.py`
   calls it. Do not modify those functions; import them.
4. **Pure computation is unit tested and network-free.** All metric code
   lives in one module (`regime_paths.py`) with no I/O; tests use hand-built
   price series (`tests/test_stats.py` style). The study CLI and the page
   builder are thin wrappers.
5. **Degrade honestly.** A cell with fewer than `MIN_N` scored paths shows
   `n` and a dash, never a number; CIs are suppressed below 2 blocks (the
   existing `block_boot_ci` convention). The build step must never fail the
   nightly workflow because a cell is empty.
6. **Own price, own horizon.** Paths are the index's own proxy (QQQ / SPY /
   IWM adjusted close), 21 sessions, rebased to the close of the observation
   day. Same horizon as the existing table so the two layers agree.
7. Commit per phase; run `python -m pytest` before each commit.

---

## 2. Definitions (the spec)

### 2.1 Path and per-path metrics

For an observation day `t` and index `k`, let `p_0..p_21` be the 22 adjusted
closes from `t` through `t+21` and `r_i = ln(p_i / p_{i-1})` the 21 daily log
returns. Cumulative path `c_i = p_i / p_0 − 1` (%). Every metric below is one
number per (t, k):

| key | metric | formula | what it answers |
|---|---|---|---|
| `r21` | terminal return | `c_21` | (already on the page) |
| `rv` | forward realized vol | `std(r_1..r_21, ddof=1) · √252 · 100` | how volatile the month is |
| `rv_ratio` | vol expansion | `rv ÷ rv_trail`, where `rv_trail` is the same statistic over `r_{-20}..r_0` | does vol rise or fall from here — the part DIX might actually know |
| `mae` | max adverse excursion | `min_i c_i` (≤ 0) | worst drawdown from entry |
| `mae_day` | trough day | `argmin_i c_i` | is the pain early or late |
| `mfe` | max favourable excursion | `max_i c_i` (≥ 0) | best run-up |
| `mfe_day` | peak day | `argmax_i c_i` | |
| `range` | path range | `mfe − mae` | the cone's width along the way |
| `er` | efficiency ratio | `|c_21| ÷ Σ_i |c_i − c_{i−1}|` in price-% units | trend (→1) vs chop (→0) |
| `taw` | time above water | share of `i ∈ 1..21` with `c_i > 0` | did it spend the month up or down |
| `xings` | zero crossings | count of sign changes of `c_i` over `i ∈ 1..21` | direct chop count |
| `ac1` | lag-1 autocorrelation | `corr(r_1..r_20, r_2..r_21)` | momentum (+) vs mean-reversion (−) within the month |
| `big_dn`, `big_up` | ≥1% down / up days | counts | asymmetry of the shocks |

Distribution-level metrics per (regime, k) — computed across the paths, not
within one — answer "how wide is the cone":

| key | metric |
|---|---|
| `r21_sd`, `r21_iqr`, `r21_q10`, `r21_q90` | dispersion of terminal outcomes |
| `fan` | quantiles q10 / q25 / q50 / q75 / q90 of `c_i` at every `i = 0..21` (the fan chart) |
| `med_path_shape` | `argmax` of the q50 path (when the typical month tops) |

### 2.2 Lenses (which days count)

Both lenses use the same 27 codes built by `build_aligned` (5-day-MA DIX,
Low = deciles 1–3, Mid = 4–7, High = 8–10).

- **Every-day** (default, matches the existing table): every session whose
  code is X and which has a full 21-session forward window. Heavily
  overlapping; descriptive of the *environment*.
- **Entry** (`entry_events`, 21-session cool-down): the first day the code
  appears, one event per 21 sessions. Closest to independent; small n. This
  is the lens that says whether the setup is *tradeable*, and the one whose
  fan chart is the honest "what happens after I see this today".

Persistence is reported as a robustness cut, not a third lens: `--min-run K`
keeps only days whose code has held for ≥ K consecutive sessions (K = 3 is
the test value). Its purpose is to answer "does a regime that *sticks* look
different from one that flickers"; expect n to collapse for most cells.

### 2.3 Cell classification (the one-word answer)

Computed per (regime, k) from **medians vs the every-day baseline medians**
of the same index, using ratios so the three indices' different vol levels
don't matter:

```
vol_state   = "hot"   if med(rv) >= 1.15 * base(rv)
            = "quiet" if med(rv) <= 0.85 * base(rv)
            = "normal" otherwise
trend_state = "trend" if med(er) >= 1.20 * base(er)
            = "chop"  if med(er) <= 0.80 * base(er)
            = "mixed" otherwise
direction   = sign of med(r21) if |med(r21)| >= 1.0pp else "flat"
label       = f"{vol_state} {trend_state} {direction}"     # e.g. "hot chop flat", "quiet trend up"
```

Thresholds are constants at the top of `regime_paths.py` (`VOL_HOT`, `VOL_QUIET`,
`ER_TREND`, `ER_CHOP`, `DIR_MIN_PP`). They are deliberately coarse; the page
shows the label *and* the numbers, never the label alone.

### 2.4 Honesty controls

1. **Block-bootstrap CIs** (existing `block_boot_ci`, 21-day blocks) on the
   cell's *mean* `rv`, `er` and `mae`; suppressed below 42 scored days.
   `block_boot_ci` takes a 1-D array and returns a CI for its mean — call it
   once per metric; do not generalize it.
2. **Vol-persistence control.** Realized vol is sticky, and DIX regimes are
   not independent of the vol environment, so a cell with high forward `rv`
   may just be a cell that *occurs* in high-vol months. Two checks:
   `rv_ratio` (forward ÷ trailing, §2.1) and a **vol-matched baseline**: split
   all days into trailing-`rv` terciles; report each cell's forward `rv` and
   `er` against the baseline of *its own* trailing-vol tercile mix
   (weighted by the cell's tercile composition). A cell is only called
   "hot" or "quiet" in the findings if it clears the vol-matched baseline
   too.
3. **Expanding-window basis** (`--basis expanding`, existing
   `expanding_decile`, min 250 obs): rerun the every-day table on
   live-knowable cutoffs; report which cell labels survive.
4. **Out-of-sample split** at `OOS_SPLIT = 2024-01-01` (existing convention):
   cutoffs from pre-2024, paths scored 2024+; report cells with ≥ 21 OOS
   days only.
5. **Persistence cut** `--min-run 3` (§2.2).

---

## 3. Deliverables

| file | role |
|---|---|
| `regime_paths.py` | **new** — pure computation: `path_metrics`, `fan_quantiles`, `cell_table`, `classify`, `vol_matched_baseline`, `run_lengths`. No I/O, no network. |
| `regime_paths_study.py` | **new** — CLI: loads payload + Yahoo cache, prints the report, writes `regime_paths.csv` (tidy: one row per regime × index × lens) and `regime_paths_fan.csv` (fan quantiles). |
| `REGIME_PATHS_FINDINGS.md` | **new** — the write-up (§5 template). |
| `tests/test_regime_paths.py` | **new** — hand-built series tests (§4 Phase 1). |
| `build_comovement.py` | **edit** — compute the page payload (`PATHS`) from the same aligned frame + prices it already has; one new placeholder. |
| `comovement_template.html` | **edit** — new section "How the month unfolds" between the current 03 (price chart) and 04 (landscape); renumber 04–06 → 05–07. |
| `INDEX_COMOVEMENT_FINDINGS.md` | **edit** — one paragraph + link pointing to the path findings. |
| `.github/workflows/refresh.yml` | **no change** — `build_comovement.py` already runs nightly with the Yahoo cache; the new payload rides along. |

---

## 4. Phases

### Phase 1 — `regime_paths.py` + tests (pure, offline)

Functions (signatures are the contract):

```python
H = 21                      # path horizon, sessions
TRAIL = 21                  # trailing-vol window for rv_ratio
MIN_N = 10                  # below this a cell shows n and dashes
FAN_Q = (10, 25, 50, 75, 90)

def path_metrics(close: pd.Series, horizon=H, trail=TRAIL) -> pd.DataFrame:
    """One row per date of `close` with the §2.1 per-path metrics for the
    forward `horizon` sessions (NaN where the window is incomplete or the
    trailing window is missing). Vectorized where cheap; a plain loop over
    ~1,700 days is fine."""

def fan_quantiles(close: pd.Series, dates, horizon=H, q=FAN_Q) -> np.ndarray:
    """(len(q), horizon+1) array of cumulative-%-return quantiles across the
    paths starting on `dates` (reuses gdx_chase_study.path_matrix)."""

def cell_table(A: pd.DataFrame, metrics: dict[str, pd.DataFrame], lens="everyday",
               min_run=1) -> pd.DataFrame:
    """Tidy table: regime x index x lens -> n, n_episodes, medians and q25/q75
    of every §2.1 metric, r21 dispersion stats, block-bootstrap CIs on mean
    rv/er/mae (NaN when suppressed). `A` is index_comovement_study.build_aligned
    output; `metrics[k]` is path_metrics for index k."""

def classify(row, base) -> str:            # §2.3
def vol_matched_baseline(A, metrics, code, k) -> dict   # §2.4(2)
def run_lengths(code: pd.Series) -> pd.DataFrame        # episodes per regime (§0a table)
```

Tests (`tests/test_regime_paths.py`), each on a tiny hand-built close series:

- straight line up 1%/day → `er == 1.0`, `taw == 1.0`, `xings == 0`,
  `mae == 0`, `mfe_day == 21`.
- perfect ±1% zigzag → `er ≈ 0`, `taw ≈ 0.5`, `xings ≥ 19`, `ac1 < −0.9`.
- constant price → `rv == 0`, `rv_ratio` NaN (0/0 handled), no exception.
- known drawdown (up 5, down 10, up 12) → `mae`, `mae_day`, `mfe`, `mfe_day`
  match hand values.
- `rv_ratio`: doubled daily amplitude in the forward window → ratio ≈ 2.
- `fan_quantiles` on three parallel lines → q50 is the middle line, shape
  `(5, 22)`, day 0 all zeros.
- `run_lengths` on a hand-built code string → known episode counts.
- `cell_table` with n < MIN_N → NaN numbers, n still reported.
- `classify` boundary cases at exactly the thresholds.

### Phase 2 — `regime_paths_study.py` (the text/CSV study)

CLI mirrors `index_comovement_study.py`:

```
python regime_paths_study.py --html docs/index.html --cache-dir .ndx_dark_cache \
       [--basis full|expanding] [--lens everyday|entry|both] [--min-run K] \
       [--csv regime_paths.csv] [--fan-csv regime_paths_fan.csv]
```

Report sections, in order: (1) sample + episode-length table (§0a);
(2) baseline path profile per index; (3) the 27-cell table per index, sorted
by n, with medians of `r21 rv rv_ratio mae mfe er taw ac1`, `r21_iqr`, the
CI on `rv`, the vol-matched baseline `rv`/`er`, and the label; (4) the entry
lens for the six headline regimes (`LLH HML MML HHL LLL HHH` — the existing
`entry_report` list) plus any cell with ≥ 10 entries; (5) the `--min-run 3`
cut; (6) expanding basis; (7) OOS. Sections 5–7 print only the cells whose
label *changes* vs the full every-day table, plus n — the point is to see
what survives, not to reprint everything.

### Phase 3 — page section + payload

`build_comovement.py` gains `build_paths(A, adjclose)` → `PATHS`:

```
PATHS = {
  "h": 21, "q": [10,25,50,75,90],
  "base": { k: { "fan": [[...22 values]...5 rows], "m": {rv, er, mae, mfe, taw, r21_iqr, ...} } },
  "cell": { code: { "n": int, "ep": episodes, k: { "fan": ..., "m": {...}, "lbl": "hot chop flat",
                                                   "ci_rv": [lo,hi] | null } } },
  "entry": { code: { "n": int, k: { "fan": ..., "m": {...} } } }   # only cells with >= 10 entries
}
```

Injected via `/*__PATHS__*/`; the placeholder is added to the template and
`main()` replaces it alongside `DATA`, `PX`, `META`. Rounding: fan values to
2 dp, metrics to 2 dp — the payload stays a few tens of KB.

Template section "How the month unfolds" (new 04):

- Reuses the existing L/M/H segmented filter and the clickable heatmap rows
  as the regime selector (they already set a shared filter state; hook the
  new section to the same `apply()` path so all three sections move
  together). A regime is a full 3-letter code; with any "Any" the section
  falls back to the baseline and says so.
- Three small-multiple canvases (NDX / SPX / IWM): x = session 0..21, y =
  cumulative %; q10–q90 band (light), q25–q75 band (darker), q50 line
  (solid), baseline q50 (dotted) and baseline q25–q75 (hatched outline) for
  contrast. Zero line. Hover shows the five quantiles at that session.
- A metrics strip under each canvas: `n days · episodes`, `fwd vol` (with
  vs-baseline arrow and the CI when present), `vol vs trailing`, `max DD ·
  day`, `max run-up · day`, `efficiency`, `time above water`, `r21 IQR`, and
  the label chip. Numbers, then the word.
- Lens toggle: Every-day / Entry (Entry greyed with a tooltip when the cell
  has < 10 entries).
- A compact heatmap table (same `hm` styling as section 05) with one row per
  regime and, per index, `fwd vol`, `efficiency`, `max DD`, `r21 IQR` cells
  coloured on their own scales (vol: cool→warm; efficiency: chop→trend;
  DD: teal→rust; IQR: narrow→wide). Clicking a row selects it, like today.
- Prose: two sentences on what the fan shows and the persistence caveat
  (median episode length, pulled from the payload so it never drifts).

Keep the canvas drawing in the style of the existing `draw()` (device-pixel
scaling, theme tokens, `ptip` tooltip); copy its helpers, do not add a
library.

### Phase 4 — robustness runs + findings

Run the full CLI (all lenses, `--min-run 3`, `--basis expanding`, OOS) and
write `REGIME_PATHS_FINDINGS.md` using the §5 template. Add the cross-link
paragraph to `INDEX_COMOVEMENT_FINDINGS.md`. Regenerate
`docs/comovement.html` locally, open it, and check: filter → fan chart →
table stay in sync; "Any" fallback; a cell with n < MIN_N; light and dark
theme; mobile width (canvases stack).

### Phase 5 — PR

One PR against `main`, with the §0 feasibility table and the headline
findings in the body; screenshots of the new section for two contrasting
cells (e.g. MHM vs HMH).

---

## 5. Findings template (`REGIME_PATHS_FINDINGS.md`)

1. Question, data window, reproduce line.
2. The persistence fact (episode table) and what it means for reading any
   regime-conditioned statistic.
3. Baseline path profile per index (the yardstick).
4. **The path map**: per index, cells grouped by label — `quiet trend`,
   `hot chop`, `hot trend`, … — with n, `rv`, `rv_ratio`, `er`, `mae`, `r21_iqr`,
   and the CI. Call out (a) cells whose *mean* return is attractive but whose
   path is choppy or violent (MML, LLH are the candidates from §0), (b) cells
   that are quiet and trending (HHM, HHL, MHM candidates), (c) the violent
   cells (HMH, LLM candidates).
5. Vol-persistence control: which "hot"/"quiet" calls survive vs the
   vol-matched baseline and on `rv_ratio`.
6. Entry lens: for the six headline regimes, the fan at entry vs the
   every-day fan — does the shape change when you only count formation days?
7. Expanding cutoffs, `--min-run 3`, OOS: what survives.
8. What it says (3–5 numbered reads), then caveats (overlap, one cycle,
   in-sample deciles, IWM reconstruction — the existing list, plus: path
   metrics on 21 daily points are noisy per path; only cell-level medians
   are meaningful).

---

## 6. Decisions already made (do not re-open)

- **Horizon stays 21 sessions.** A 63-session path would be more
  informative for trend but would cut the sample by another third and break
  agreement with the existing table. Revisit only after this ships.
- **Efficiency ratio is the trend/chop primary**; `ac1`, `xings` and `taw` are
  supporting. ER is scale-free, bounded, and the same statistic
  `ETF_PATH_PLAYBOOK.md` readers already understand from `path_stats`.
- **Medians, not means, for every path metric** (skewed distributions); means
  only where a CI is attached.
- **The page's regime selector is shared**, not duplicated: one filter, three
  sections.
- **No new workflow step**: the payload is built inside `build_comovement.py`.
- **No signal/trade construction** in this iteration. The output is a map
  of environments; whether any cell is tradeable is the entry-lens question
  the comovement study already answers, and this study only adds the path
  shape to that answer.
