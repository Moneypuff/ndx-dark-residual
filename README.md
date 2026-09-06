# optsnap-data

Daily option-chain snapshots for the fixed-strike vol tracker.
Written by .github/workflows/optsnap.yml; consumed by the vol-tracker build.
See VOL_TRACKER.md on main.

## ibit_pdf.py — IBIT risk-neutral probability density

Extracts the risk-neutral (Breeden–Litzenberger) probability density of the
IBIT share price at each monthly expiration from the October 2026 monthly
through the January 2027 monthly.

For each expiry it estimates the implied forward and discount factor from
put-call parity, inverts the out-of-the-money wing to a Black-76 vol smile,
smooths and C¹-extrapolates the smile, re-prices a dense strike grid and takes
the second derivative — `q(K) = e^{rT} d²C/dK²` — then normalises to a density.
The density mean is checked against the forward (`mean/fwd-1%`) as a quality gauge.

```bash
pip install numpy scipy pandas matplotlib requests   # + yfinance is optional
python3 ibit_pdf.py                     # pulls the live chain from Yahoo Finance
python3 ibit_pdf.py --asof 2026-09-06   # pin the valuation date
python3 ibit_pdf.py --csv snapshot.csv  # run offline from an optsnap-schema file
```

Outputs (in `--outdir`, default `ibit_pdf_out/`): a per-expiry price/density/CDF
CSV, `ibit_pdf_summary.csv` (forward, discount, moments and key probabilities),
and `ibit_pdf.png` (density + CDF plots). The optsnap snapshots do not currently
track IBIT, so the default live source is Yahoo; point `--csv` at any snapshot in
the optsnap schema once IBIT is included.
