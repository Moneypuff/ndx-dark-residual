# optsnap-data

All option-chain data for the vol tracker, vol surface and S&P 500 captures. Data only; code
lives on main and on claude/etf-signal-options-playbook. This is the only data branch:
`claude/optsnap-data` was merged in on 2026-10-06; nothing writes to it any more and every commit on it is contained here, so it can be deleted.

## Layout

| Path | What it holds | Written by |
|---|---|---|
| `universe/core.csv` | The 51 stocks and ETFs captured hourly (`symbol,kind`). | edited by hand |
| `universe/sp500.csv` | S&P 500 constituents from SPY's holdings file, as of 2026-10-05 (503 symbols, Yahoo tickers such as BRK-B). Refresh after index changes. | edited by hand |
| `intraday/<date>_<HHMM>ET.csv.gz` | Hourly chains for the core universe, 9:50am to 3:50pm ET. | scheduled task |
| `boards/board_<date>_<HHMM>ET.csv` | Per-symbol, per-tenor summary of each hourly capture. `boards/_uploaded.txt` lists the boards already copied to Google Drive. | scheduled task |
| `optsnap/<date>.csv.gz` | One end-of-day chain per trading day for the core universe: the 3:50pm capture. This is what `refresh.yml`, `build_vol_tracker.py` and `build_vol_surface.py` read, so its path must not change. | scheduled task; `optsnap.yml` fills days it missed |
| `optsnap/_status-<date>.json` | Capture report from the nightly fallback workflow. | `optsnap.yml` |
| `spx/<date>.csv.gz` | One chain per trading day for every S&P 500 constituent, captured at 3:50pm ET. | scheduled task |
| `spx/_status-<date>.json` | Per-symbol report for the S&P 500 capture. | scheduled task |
| `optsnap_alt/` | Captures kept for reference but left out of the daily series (evening captures displaced by a 3:50pm capture, and a Sunday-dated copy of the Aug 7 close). | one-off |

Rows in `optsnap/` from 2025-08-11 through 2026-08-07 were backfilled from ORATS
(`backfill_optsnap_from_orats.py` on main); later rows are Yahoo Finance captures.

## Writers

1. Scheduled task "Option-chain capture (core hourly + S&P 500 daily)", weekdays 9:50am to 3:50pm ET.
   Every run captures the core universe into `intraday/` and `boards/`. The 3:50pm run also
   copies that capture to `optsnap/<date>.csv.gz` and then captures the S&P 500 into `spx/`.
2. `.github/workflows/optsnap.yml` on main, evening ET: captures `optsnap/<date>.csv.gz` only
   when the day has no daily file yet. It uses main's own universe file.

Both writers rebase onto the latest branch tip before pushing.

## Schema

Every chain file: `date,symbol,expiry,right,strike,iv,oi,volume,bid,ask,last,spot`. `oi` is the
previous session's OCC-settled open interest as shown by Yahoo; `iv` is Yahoo's figure, and the
builds recompute it. Contract policy per symbol: nearest 2 expiries, one monthly out to about
9 months, every January expiry; strikes within 25% of spot (65% for January), plus the top 20
open-interest strikes per side.
