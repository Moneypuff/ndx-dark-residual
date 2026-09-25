# ndx-dark-residual

A dark-pool-flow research toolkit and dashboard for the **Nasdaq-100** (plus SPX
/ IWM extensions), built **entirely from free public data — no paid API, no
key**.

## The signal

For every constituent the dark-pool indicator **`D`** = 5-day MA of
`ShortVolume / off-exchange TotalVolume`, computed directly from **FINRA's** daily
consolidated off-exchange (CNMSshvol) files — the same construction SqueezeMetrics
uses for DIX. Prices come from **Yahoo Finance** (split-adjusted close for dollar
weighting — FINRA's as-traded volumes are put on the same split basis with Yahoo's
split events — and adjusted close for forward returns).

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
- **Relative-darkness toolkit**: `dark_flow.py` (name-level dark-flow measures and
  the cross-sectional statistics to test them) + `dark_flow_study.py` — see
  `DARK_FLOW_FINDINGS.md`.
- **Studies**: standalone analyses, each with a `*_study.py` and a `*_FINDINGS.md`
  writeup — e.g. `DARK_FLOW_FINDINGS.md`, `EARNINGS_DPI_FINDINGS.md`, `EXPECTED_MOVE_FINDINGS.md`,
  `INDEX_COMOVEMENT_FINDINGS.md`, `GDX_CHASE_FINDINGS.md`, `GEX_DISPERSION_GUIDE.md`,
  `ETF_PATH_PLAYBOOK.md`, `VOL_TRACKER.md`.
- `tests/`, `pytest.ini`, `requirements.txt` / `requirements-dev.txt`.

## Setup

```bash
pip install -r requirements.txt
python build_report.py        # build the dashboard from freshly fetched data
```

> **Research verdict** (methodology review: `DARK_FLOW_FINDINGS.md`): name-level
> dark flow — the `D`-vs-DIX residual and five better-built "relative darkness"
> measures — shows **no robust directional edge** across the S&P 500, NDX-100 or
> Russell 2000. At the index level the DIX behaves like a stress gauge (nothing
> survives detrending or a realized-vol control), and the one index result that
> looked tradeable (the comovement study's "large-cap firm / small-cap Low" family)
> was an artifact of a split-weighting bug in the dollar-DIX, now fixed. Dark flow
> does carry a little volatility information. Read the dashboard as a descriptive
> monitor, not a live trading signal.
