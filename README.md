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
