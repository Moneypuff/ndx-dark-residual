#!/usr/bin/env python3
"""Throwaway diagnostic: trace WHERE 2026 DPI becomes NaN in the earnings pipeline
using the restored CI cache. Not part of the pipeline; removed after diagnosis."""
import sys
import pandas as pd
import numpy as np

import ndx_dark_residual as N
import earnings_dpi_study as E

CACHE = sys.argv[1] if len(sys.argv) > 1 else ".ndx_dark_cache"
SAMPLE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "WMT", "XEL"]


def yearcov(s):
    s = s.dropna()
    if s.empty:
        return "EMPTY"
    yr = pd.to_datetime(s.index).year
    return {int(y): int((yr == y).sum()) for y in sorted(set(yr))}


print("========== STAGE 1: raw FINRA documents on disk ==========")
for kind, fname in [("TOTAL", "finra_offexch_volume_earn.csv"),
                    ("SHORT", "finra_short_volume_earn.csv")]:
    p = f"{CACHE}/{fname}"
    try:
        doc = pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {kind}: could not read {p}: {e}")
        continue
    yr = doc.index.year
    g26 = doc[yr == 2026]
    print(f"\n{kind} [{p}]  shape={doc.shape}  cols={len(doc.columns)}")
    print(f"  index min/max: {doc.index.min()} .. {doc.index.max()}")
    print(f"  2026 rows: {len(g26)}")
    if len(g26):
        frac_all = g26.isna().mean(axis=1)
        want = [c for c in doc.columns]  # all cols are the earnings universe here
        print(f"  2026 all-col NaN-frac med={frac_all.median():.3f} max={frac_all.max():.3f}")
        for t in SAMPLE:
            if t in doc.columns:
                print(f"    {t:6}: 2026 non-NaN = {int(g26[t].notna().sum())} / {len(g26)}"
                      f"  (sample vals: {list(g26[t].dropna().head(2).round(0))})")
            else:
                print(f"    {t:6}: NOT A COLUMN")
    # how many wanted-style columns fully NaN in 2026
    if len(g26):
        fully = [c for c in doc.columns if g26[c].notna().sum() == 0]
        print(f"  2026 fully-NaN columns: {len(fully)} / {len(doc.columns)}  e.g. {fully[:10]}")

print("\n========== STAGE 2: rebuilt panels via build_universe_panels ==========")
earn = E.load_earnings("earnings_dates_edgar.csv")
syms = sorted(earn.ticker.unique())
start = (earn.report_date.min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
end = (earn.report_date.max() + pd.Timedelta(days=70)).strftime("%Y-%m-%d")
print(f"universe {len(syms)} syms  window {start}..{end}")
panels = N.build_universe_panels(syms, start, end, workers=10, cache_dir=CACHE, ns="earn", label="EARN")
close = panels["close"]; dpi = panels["dpi"]; short = panels["short"]; total = panels["total"]
print("close index max:", close.index.max(), " total rows:", len(close.index))
print("close index 2026 count:", int((close.index.year == 2026).sum()))
print("dpi   index 2026 count:", int((dpi.index.year == 2026).sum()))
print("short cols:", len(short.columns), " total cols:", len(total.columns), " dpi cols:", len(dpi.columns))
for t in SAMPLE:
    if t in dpi.columns:
        print(f"  {t:6}: short 2026-nonNaN={int((short[t][short.index.year==2026]).notna().sum()) if t in short else 'NA'}"
              f"  total 2026-nonNaN={int((total[t][total.index.year==2026]).notna().sum()) if t in total else 'NA'}"
              f"  dpi 2026-nonNaN={int((dpi[t][dpi.index.year==2026]).notna().sum())}")
    else:
        print(f"  {t:6}: not in dpi columns")

print("\n========== STAGE 3: merge_share_classes then event dpi ==========")
panels2, earn2 = E.merge_share_classes(panels, earn)
dpi2 = panels2["dpi"]
print("post-merge dpi cols:", len(dpi2.columns))
for t in SAMPLE:
    if t in dpi2.columns:
        print(f"  {t:6}: dpi 2026-nonNaN={int((dpi2[t][dpi2.index.year==2026]).notna().sum())}")
