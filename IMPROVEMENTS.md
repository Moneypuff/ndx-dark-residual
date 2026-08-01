# Study & dashboard improvement roadmap

Prioritized improvements for the three research surfaces in this repo. Items marked
**[done]** were implemented on this branch; the rest are ordered by expected
insight-per-effort within each section.

## Sector ETFs: return vs DIX  **[done]**

The dashboard now plots each sector / industry fund's reconstructed dollar-DIX
against **its own ETF's forward return** — the same treatment the NDX / SPX / IWM
gauges get.

- **Where:** the *DIX vs Return* tab gained a **Sectors** toggle. Pick any of the
  23 funds (8+5 broad sectors, 10 SPDR industry funds); you get the familiar
  1/2/3-month decile bars (±1 SE, overlap-adjusted), the daily scatter with
  today's-DIX marker, and Pearson/Spearman with block-bootstrap CIs.
- **At a glance:** a summary table ranks all funds by the size of the effect
  (D10−D1: mean 1-month forward return in the fund's top DIX decile minus its
  bottom decile), with r at each horizon and which decile today's DIX sits in.
  Click a row to load that fund into the panels.
- **Plumbing:** `build_sector_payload` now packs `r21/r42/r63` per fund from the
  ETF's own adjusted close (fetched alongside the constituent panels); forward
  returns are computed on the full price history before slicing to the plot
  window, exactly like `build_index_payload`.

## Earnings DPI study (`earnings_dpi_study.py`)

1. **Market-excess outcomes.** Forward returns are raw in a mostly-bull
   2018–2026 sample, so part of the high-DPI drift is beta. Add QQQ-excess
   (and, with the sector panels now available, sector-excess) forward returns as
   the headline outcome; the raw numbers can stay as a secondary column.
2. **Real-time (expanding) within-name percentiles.** Event DPI is currently
   ranked against the name's *full* history — a mild look-ahead. Rank each event
   against only prior events/history (expanding window, min N) so terciles are
   tradable as-of the report date; report both cuts.
3. **Season-clustered inference.** Earnings cluster in reporting weeks, so the
   2,663 events are cross-sectionally correlated and pooled p-values overstate
   precision. Cluster the bootstrap/SEs by calendar quarter (earnings season) —
   the honest unit is ~32 seasons, not 2,663 events.
4. **Control for pre-earnings momentum.** High DPI into a report may proxy for
   the preceding run-up. A double sort (DPI tercile × pre-earnings 1-month
   return tercile) would show whether DPI adds anything beyond momentum.
5. **Surprise interaction.** Does high pre-report DPI predict the *reaction to*
   the news (dark accumulation = informed positioning) or drift regardless of
   it? Split post-earnings paths by the T+1 gap direction: high-DPI + negative
   gap that still recovers by T+21 would be the strongest "informed flow" tell.
6. **Complete the universe.** The 6 foreign filers (6-K reporters) can be added
   via their press-release exhibit dates; SPCX/HONA need a fallback source.

## Cross-index comovement study (`index_comovement_study.py`)

1. **Expanding-window regimes.** Deciles are full-sample (flagged in caveats).
   Add `--expanding` cutoffs (min ~250 sessions) and report the headline regimes
   both ways — the divergence result is only tradable if it survives this.
2. **Continuous spread factor instead of 27 cells.** Most cells hold 30–130
   overlapping days ≈ a handful of episodes. Collapse to two continuous
   signals — joint level (mean of the three DIX5 z-scores) and the large-vs-small
   spread ((NDX+SPX)/2 − IWM z) — and regress forward returns on both. Uses every
   observation, kills the arbitrary Low/Mid/High binning, and directly tests the
   study's own conclusion that *divergence, not level, carries the signal*.
3. **Regime-entry event study.** Score only the first day a regime is entered
   (with a cool-down), not every day inside it — cuts the overlap problem and
   answers the practical question "what happens after the setup appears?".
4. **Block-bootstrap CIs on the regime means** (21-day blocks), so the table can
   show which cells are distinguishable from baseline instead of leaning on
   medians-vs-means language.
5. **Add sector gauges to the comovement set.** With per-sector DIX now packed,
   defensive (XLP/XLU/XLV) vs cyclical (XLI/XLF/XLE) dark-flow divergence is a
   natural extension of the same framework.
6. **Out-of-sample split.** Freeze thresholds on pre-2024 data, evaluate 2024+;
   the COVID recovery and 2022 drawdown currently sit inside the sample and
   flatter several cells.

## Dashboard (`ndx_dark_residual.py` → `docs/index.html`)

1. **Payload size / first paint.** `docs/index.html` is ~25 MB with the JSON
   inlined. Move the payload to a sibling `payload.json` fetched on load (GitHub
   Pages gzips JSON ~5×), or embed it deflate+base64 and inflate client-side.
   Biggest single UX win available.
2. **Surface the comovement regime on the dashboard.** The comovement study's
   actionable output is *today's* N/S/I regime; a small header badge (with the
   regime's historical 1-month stats) would connect the study to the daily view.
3. **Cross-linked "today" summary.** One strip aggregating the latest signals —
   index DIX percentiles, sectors at P80/P20 crossings, comovement regime,
   names with active D-streaks — so the tabs become drill-downs from a single
   morning read.
4. **Signal-change alerts from the nightly refresh.** The workflow already knows
   yesterday→today; have it append to a small `alerts.json` (sector band
   crossings, regime changes) rendered as a "what changed" panel, or open a
   GitHub issue on big transitions.
5. **Real-time-safe percentiles everywhere.** The event-study tab already offers
   trailing (no look-ahead) decile cutoffs; extend that option to the grid
   modals and the SP500 decile table so displayed "today's decile" values are
   tradable, not full-sample.
