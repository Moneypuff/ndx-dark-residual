# Study & dashboard improvement roadmap

All items below are now **implemented** on this branch. Each entry notes where
the implementation lives and — where the data has already spoken — what came of
it. Findings docs: `EARNINGS_DPI_FINDINGS.md`, `INDEX_COMOVEMENT_FINDINGS.md`.

## Sector ETFs: return vs DIX  **[done]**

The *DIX vs Return* tab has a **Sectors** toggle: any of the 23 sector /
industry funds' reconstructed dollar-DIX vs **its own ETF's** 1/2/3-month
forward return — decile bars (overlap-adjusted ±1 SE), scatter with today's-DIX
marker, Pearson/Spearman + block-bootstrap CIs, and a cross-fund summary table
ranked by the D10−D1 1-month spread (click a row to load that fund).
Plumbing: `build_sector_payload` packs `r21/r42/r63` per fund from the ETF's
own adjusted close.

## Earnings DPI study  **[all done]**

1. **Market-excess outcomes** — every horizon now has a `*_xret` twin (the
   name's return minus QQQ's over the identical window); the summary, report
   payload and report headline lead with the excess numbers.
2. **Real-time (expanding) within-name percentiles** — `dpi{5,10}_pct_exp`
   ranks each event only against that name's prior events (min 8); the
   headline spread is reported on both bases.
3. **Season-clustered inference** — `cluster_boot_corr` / `cluster_boot_spread`
   resample calendar quarters (≈32 seasons) instead of trusting ~2,700
   correlated events; cluster CIs and p-values sit next to every headline stat.
4. **Momentum double-sort** — pre-earnings 1-month return terciles × DPI
   terciles, excess returns per cell, DPI spread within each momentum tercile.
5. **Gap-direction split** — post-reaction drift (T+1→T+21) by DPI tercile
   within gap-down / flat / gap-up events; tests the "informed dark
   accumulation" reading directly.
6. **Foreign filers** — `fetch_earnings_edgar.py` falls back to a scored 6-K
   heuristic (quarter-end windows + filing-index contents) for issuers with no
   8-K/2.02 trail (`source=6k` column); recovers ASML, ARM, PDD, CCEP, FER,
   NBIS.
   The study also writes `earnings_dpi_summary.txt` (committed by the refresh
   workflow) so the findings doc can be updated from real runs.

## Cross-index comovement study  **[all done]**

1. **Expanding-window regimes** (`--basis expanding|full|both`) — under
   real-time cutoffs the requested LLH divergence has existed on only 17 live
   days; its 94-day full-sample count was mostly relabeled history.
2. **Two-factor regression** (LEVEL + large-vs-small SPREAD, Newey-West 21
   lags) — level carries nothing anywhere; the spread tilts NDX only
   (+0.66pp/z, t=1.76).
3. **Regime-entry event study** (first day + 21-session cool-down) — the
   "large-cap firm / small-cap Low" family survives (NDX +4.6–6.8%, hit
   86–100% over 28 entries); the requested LLH divergence does not (NDX +0.3%
   at entry).
4. **Block-bootstrap CIs** (21-day moving blocks) on every regime mean, in the
   CSV and the printed table; degenerate small-cell CIs are suppressed.
5. **Sector dark-flow gauge** — defensive-minus-cyclical DIX z-spread vs index
   forward returns: a clean null (|t| < 0.5 everywhere). Recorded so it stays
   found.
6. **Out-of-sample split** (fit <2024, evaluate 2024+) — the two-factor model
   does not generalize (OOS corr ≈ 0 to −0.14); the findings doc now says so.

## Dashboard  **[all done]**

1. **Payload compression** — the JSON payload is embedded deflate+base64 and
   inflated client-side via `DecompressionStream` (inline module script, still
   a single self-contained file that works over file://). `docs/index.html`
   drops ~25 MB → ~12 MB and the browser no longer parses a 25 MB JS literal.
   Python payload readers (`index_comovement_study.py`, `build_comovement.py`)
   accept both the old and new encodings.
2. **Comovement regime badge** — the header's new **Today** strip leads with
   the current N/S/I regime, its day count and historical 1-month mean/hit per
   index, linking to the comovement tab.
3. **Today summary strip** — same strip also shows each gauge's 1-year DIX
   percentile, and per-name D-streaks (5+ consecutive sessions in a name's own
   top/bottom D decile).
4. **What-changed alerts** — sector-DIX P80/P20 band crossings from the last 5
   sessions and a "regime changed today" chip surface signal transitions on
   every load, computed client-side from the payload the nightly refresh
   already ships (no extra workflow state).
5. **Real-time-safe deciles** — a "trailing deciles (no look-ahead)" toggle in
   the grid cell modal and the SP500 decile table ranks each day's D against
   only the name's prior 252 sessions (min 120), matching the event-study
   tab's trailing basis, so "today's decile" is the one that was actually
   knowable.

## Episodes vs streaks tab  **[done]**

The D-streak events tab counts every N-day run as an event and re-fires after a
one-day wobble out of the band, so one ten-day accumulation can be scored several
times with overlapping forward windows. The **Episodes vs streaks** tab studies
the same decile signal with the *episode* as the observation unit and puts the
two side by side:

1. **Episode definition** — enter at ≥D9, stay while ≥D7, exit after 2 days
   below (hysteresis), merge gaps ≤2 days, minimum length 3; every threshold is
   a control, and a "Low D" toggle mirrors the scale for distribution. Signal is
   the raw 1-day ratio (as the streak tab) or its 5-day MA (DPI); deciles use the
   same trailing / full-sample / pooled bases.
2. **One table, six rows** — A. streak trigger (the existing method, on the same
   deciles), B. episode entry, C. episode entry with a refractory rule (no
   overlapping windows within a name), D. exit-anchored with refractory (window
   starts only after the exit is confirmed, so length and intensity are known),
   E. during-episode, F. hold-while-dark. Each row shows obs vs distinct episodes,
   the within-name overlap share, mean ± SE, hit, a date-matched cross-sectional
   baseline, edge, and a circular-shift placebo p on the whole window pattern.
   With hysteresis, merging and confirmation switched off (stay = enter, gap 0,
   confirm 1, min length 1) row B reproduces row A exactly.
3. **Conditioning** — exit-anchored return by episode length bucket, intensity
   tercile and the sign of the during-episode move, plus an OLS of the
   exit-anchored return on ln(L) and mean decile with SEs clustered by exit month.
4. **Drift** — entry/trigger-aligned paths for A and C over a matched baseline,
   and an exit-aligned path for D.
5. **Split-safe returns** — the payload now packs split-adjusted closes
   (`rel.adj`; `build_html(..., adjclose_panel=)`) so every episode return is
   measured on the same basis as `r21/r42/r63`; the tab falls back to raw closes
   on older payloads and says so.
