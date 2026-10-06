# optsnap-data

Option-chain snapshots for the fixed-strike vol tracker and vol surface. Data only; code lives on main.
This branch is the single home for chain data. `claude/optsnap-data` was merged into it on 2026-10-06
and is no longer written to.

## Layout

- `optsnap/<YYYY-MM-DD>.csv.gz` - one end-of-day chain per trading day (date = ET trading day).
  This is what `refresh.yml`, `build_vol_tracker.py` and `build_vol_surface.py` read.
- `optsnap/_status-<date>.json` - per-symbol capture report from the nightly workflow.
- `intraday/<date>_<HHMM>ET.csv.gz` - hourly chain captures, 9:50am to 3:50pm ET.
- `boards/board_<date>_<HHMM>ET.csv` - distilled boards from each hourly capture;
  `boards/_uploaded.txt` lists the ones already synced to Google Drive.
- `optsnap_alt/` - captures kept for reference but left out of the daily series:
  `<date>_evening.csv.gz` are evening captures displaced by a 3:50pm capture for the same day,
  and `2026-08-09_friday-eod-seed.csv.gz` is a Sunday-dated copy of the Aug 7 close.

## Writers

1. Hourly routine "vol-tracker hourly chain capture" (9:50am to 3:50pm ET): writes `intraday/` and `boards/`.
   Its 3:50pm run also copies that capture to `optsnap/<date>.csv.gz`, the canonical daily mark.
2. `.github/workflows/optsnap.yml` on main (evening ET): captures `optsnap/<date>.csv.gz` only if the
   day has no daily file yet, so it fills in when the 3:50pm run did not happen.

Both writers pull before pushing. Rows from 2025-08-11 through 2026-08-07 were backfilled from ORATS
(`backfill_optsnap_from_orats.py` on main); later rows are Yahoo Finance captures.

## Schema

`date,symbol,expiry,right,strike,iv,oi,volume,bid,ask,last,spot` - `oi` is the previous session's
OCC-settled open interest as shown by Yahoo; `iv` is Yahoo's figure (the builds recompute it).
