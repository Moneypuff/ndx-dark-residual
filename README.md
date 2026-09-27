# ndx-dark-residual

A dark-pool-flow research toolkit and dashboard for the **Nasdaq-100** (plus SPX
/ IWM extensions), built **entirely from free public data — no paid API, no
key**.

## The signal

For every constituent the dark-pool indicator **`D`** = 5-day MA of
`ShortVolume / off-exchange TotalVolume`, computed directly from **FINRA's** daily
consolidated off-exchange (CNMSshvol) files — the same construction SqueezeMetrics
uses for DIX. Prices come from **Yahoo Finance** (raw close for dollar weighting,
adjusted close for split-safe forward returns).

Each name's *name-specific* dark flow is isolated by residualizing its `D` against
a reconstructed index dollar-DIX benchmark, two ways:
1. **Simple difference** — `resid = D_i − INDEX_DIX`
2. **Regression residual** — rolling OLS `D_i ~ a + b·INDEX_DIX` → ε (removes both
   the common level and each name's beta to market dark flow).

## Index DIX history (point-in-time, 2009+)

The NDX, SPX and IWM index tabs (and the comovement tab) show a **point-in-time,
dollar-weighted DIX from August 2009**. It is committed as `data/pit_dix_history.csv`
and spliced in front of the live build (`load_pit_dix_history` /
`splice_dix_history`): dates up to the file's last row come from it, later dates from
the nightly live computation.

- **Why committed, not rebuilt in CI:** it needs FINRA Reg SHO files back to 2009 and
  membership history, neither available to the nightly job. Before 2018-08 FINRA only
  publishes per-facility files (FNSQ/FNYX/FNQC/FORF/FNRA), which sum exactly to the
  consolidated file where both exist.
- **Membership:** Nasdaq-100 from contemporaneous Wikipedia snapshots; S&P 500 from a
  point-in-time table; Russell 2000 from IWM's actual holdings in SEC N-Q (2009-2018)
  and N-PORT (2019+) filings. The live tail uses current holdings, so survivorship is
  confined to the days since the file was last refreshed. The build log prints the
  overlap agreement for each index.
- **Validation:** the same construction on the S&P 500 reproduces SqueezeMetrics'
  published DIX at 0.994 (daily level) / 0.996 (5-day MA), 2011-2026.
- **Refresh** (locally, where the history pipeline lives):
  `python tools/export_pit_dix_history.py`, then commit `data/pit_dix_history.csv`.

Dollar weighting pairs FINRA's **as-traded** share volume with the **as-traded** close
(`rawclose`, derived from Yahoo's split events). Yahoo's `close` is split-adjusted,
and pairing it with as-traded volume mis-weights every name around splits, badly for
reverse splits.

## Layout

- **Dashboard**: `build_report.py` + `report_template.html` (D-vs-forward-return
  tabs, comovement, dispersion, vol-surface, etc.). Other `build_*.py` /
  `*_template.html` pairs build the individual tabs.
- **Data plumbing**: `ndx_dark_residual.py` (core builder), `snapshot_option_chains.py`,
  `backfill_optsnap_from_orats.py`, `sync_watchlist.py`, `fetch_earnings_edgar.py`.
- **Studies**: standalone analyses, each with a `*_study.py` and a `*_FINDINGS.md`
  writeup — e.g. `EARNINGS_DPI_FINDINGS.md`, `EXPECTED_MOVE_FINDINGS.md`,
  `INDEX_COMOVEMENT_FINDINGS.md`, `GDX_CHASE_FINDINGS.md`, `GEX_DISPERSION_GUIDE.md`,
  `ETF_PATH_PLAYBOOK.md`, `VOL_TRACKER.md`.
- `tests/`, `pytest.ini`, `requirements.txt` / `requirements-dev.txt`.

## Setup

```bash
pip install -r requirements.txt
python build_report.py        # build the dashboard from freshly fetched data
```

> **Research verdict so far:** across multiple studies here, the reconstructed
> DIX / `D` signal shows **no robust beta-adjusted, name-level predictive edge**.
> Read this as a well-documented negative result, not a live trading signal.
