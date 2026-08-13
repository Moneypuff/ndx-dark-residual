# 3D vol surface viewer — implementation plan (foundations only)

Step-by-step instructions for building `docs/vol_surface.html`: an
interactive, rotatable 3D implied-volatility surface (moneyness ×
time-to-expiry × IV) for every symbol captured by the optsnap pipeline —
i.e. the same universe `build_vol_tracker.py` consumes
(`data/optsnap_universe.csv`: the 24 ETFs + AAPL TSLA GOOGL NVDA AMZN
MSFT META AMD MU CRWV IONQ OWL RKLB ONDS SPCX DRAM), with a date scrubber
to replay how each surface moved over the captured history.

This document is the whole spec. Every decision is already made; do not
re-open them. Follow the phases in order, run the tests after each phase,
and keep every guardrail in "Rules" below.

---

## Rules (read first, apply throughout)

1. **Branch**: all work on `claude/3d-vol-surface-viewer-a6z1g1`. Never
   push to `main` or `optsnap-data`.
2. **No network calls.** The build reads snapshot CSVs from disk only.
   Tests use hand-built frames (copy the style of
   `tests/test_vol_tracker.py`). Never call Yahoo, never import anything
   that fetches.
3. **No new dependencies.** Python: numpy + pandas only (already in
   `requirements.txt`). HTML: zero external scripts — every template in
   this repo is self-contained vanilla JS, and the viewer must be too.
   No plotly, no three.js, no CDN `<script src>`. The 3D math is small
   and is given verbatim in Phase 3 — transcribe it, don't invent it.
4. **Reuse, don't duplicate.** `load_snapshots` and `atm_iv` come from
   `build_vol_tracker.py`; `smile_points` comes from
   `trade_structures.py`. Do not re-implement smile despiking, liveness
   filtering, or snapshot loading. Do not modify those files.
5. **Degrade honestly.** With zero snapshot files the page must still
   render — a stub with the "no capture yet" note, exactly like
   `build_vol_tracker.py` does. One snapshot day = a working surface
   with a one-position slider. The build step must never fail the
   nightly workflow for lack of data.
6. **Repo conventions**: pure computation at the top of the build script
   under a `# Pure computation (unit tested)` banner, CLI `main()` at the
   bottom; payloads injected by replacing `/*__NAME__*/` placeholders in
   the template; the rendered page gets the same
   `<!doctype html>...<body>` wrapper `render_page` uses in
   `build_vol_tracker.py`. Match the comment density and docstring voice
   of the existing scripts.
7. After each phase run `python -m pytest` (offline, seconds). All
   existing tests must stay green.
8. Commit per phase with a descriptive message. Push with
   `git push -u origin claude/3d-vol-surface-viewer-a6z1g1`.

---

## What already exists (read these before writing code)

| File | What you need from it |
|---|---|
| `snapshot_option_chains.py` | Snapshot schema: one `optsnap/YYYY-MM-DD.csv.gz` per day, columns `date, symbol, expiry, right, strike, iv, oi, volume, bid, ask, last, spot`. Strikes cover ±25% moneyness (±65% for January LEAPs) plus top-OI outliers. |
| `build_vol_tracker.py` | `load_snapshots(snap_dir)` (all days concatenated, empty frame when none), `atm_iv(day_rows)` (interpolated ATM IV for one (date, symbol, expiry) group, NaN when thin), and `render_page` (the injection pattern to copy). |
| `trade_structures.py` | `smile_points(expiry_rows, spot)` → `(strikes, ivs)` of the live, despiked OTM composite smile (puts below spot, calls at/above, dead quotes dropped, spikes removed). This is the ONLY sanctioned source of smile points. |
| `vol_tracker_template.html` | The CSS token block (`:root` dark/light/system themes), page scaffolding, `esc`/`num` helpers, and the `/*__CTX__*/` payload pattern. The new template copies this style block verbatim. |
| `.github/workflows/refresh.yml` | Where the nightly build steps live. The optsnap branch is already fetched into `optsnap/` by the "Fetch chain snapshots" step before the vol tracker builds — the new build step slots in right after "Build vol tracker" and reuses that directory. |
| `tests/test_vol_tracker.py` | The testing idiom: `_rows(...)` helper building snapshot rows by hand, no fixtures from disk, `pytest.approx` on the math. |

---

## Architecture at a glance

```
optsnap/*.csv.gz ──> build_vol_surface.py ──> docs/vol_surface.html
   (data branch,        pure functions:            template:
    fetched by           - surface_grid            vol_surface_template.html
    refresh.yml)         - symbol_surface          canvas 3D renderer,
                         - build_payload           symbol picker, date
                                                   scrubber, drag-rotate
```

Per (symbol, snapshot date): each captured expiry contributes one smile,
interpolated onto a **common moneyness grid**; stacking the expiries by
time-to-expiry gives the surface. The browser gets the pre-gridded
numbers as JSON and only projects/draws — no financial math in JS.

### Fixed numerical decisions

| Constant | Value | Why |
|---|---|---|
| `M_GRID` | `np.arange(0.60, 1.4001, 0.025)` → 33 points (60%–140% moneyness) | Covers the ±25% regular capture window fully and the meat of the LEAP window; a fixed grid is what makes surfaces comparable across dates. |
| Min smile points per expiry | 4, and spot strictly inside `[ks[0], ks[-1]]` | Same threshold `atm_iv` uses; thinner smiles are noise. |
| Extrapolation | **None.** Grid cells outside `[ks[0], ks[-1]]` are `NaN` (→ JSON `null`). | A flat-extrapolated wing looks like data and isn't. The renderer simply leaves holes. |
| Max expiries per surface | 14, nearest-first by DTE | Keeps every monthly + the near LEAPs; caps payload and draw cost. |
| `SURF_DAYS` | last **10** snapshot days | The scrubber's history window. Payload stays ~1–2 MB across the full universe; older history remains reachable by rebuilding with `--days N`. |
| IV rounding in payload | 4 decimals (0.0001 = 0.01 vol pt) | Size, without visible loss. |
| DTE axis spacing | plot at `sqrt(DTE)` (computed in JS from the shipped `dte` values) | Linear DTE crushes the front months where all the action is; sqrt keeps LEAPs on-screen without a log axis's distortion. |

---

## Phase 1 — `build_vol_surface.py` (pure functions + CLI)

Create `build_vol_surface.py`. Module docstring: what it builds, the
grid decisions above, and the run line
`python build_vol_surface.py --snap-dir optsnap --docs-out docs/vol_surface.html`.

```python
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import trade_structures as T
from build_vol_tracker import load_snapshots, atm_iv

M_GRID = np.arange(0.60, 1.4001, 0.025)   # 33 moneyness points, 60%..140%
MAX_EXPIRIES = 14
SURF_DAYS = 10
```

### 1a. `surface_grid(expiry_rows, spot)`

One (date, symbol, expiry) group of snapshot rows → the smile sampled on
`M_GRID`.

* Call `T.smile_points(expiry_rows, spot)`.
* If `len(ks) < 4` or not `ks[0] < spot < ks[-1]`: return `None`
  (expiry contributes nothing).
* Else return a float array of `len(M_GRID)`:
  `iv = np.interp(M_GRID * spot, ks, vs)`, then set entries where
  `M_GRID * spot < ks[0]` or `> ks[-1]` to `np.nan` (kill np.interp's
  edge clamping — that's the no-extrapolation rule).

### 1b. `symbol_surface(day_df, symbol, asof)`

One snapshot day, one symbol → the surface dict, or `None` when no
expiry survives.

* Filter to the symbol; group by `expiry`; sort expiries ascending.
* Per expiry: `dte = max((pd.Timestamp(expiry) - pd.Timestamp(asof)).days, 1)`;
  compute `surface_grid`; skip `None` rows; also compute
  `atm_iv` for the expiry group (may be NaN — ship as null).
* Keep the first `MAX_EXPIRIES` surviving expiries (nearest first), but
  **return them sorted by DTE ascending** in the output.
* Return:

```python
{"spot": float, "expiries": [str, ...], "dtes": [int, ...],
 "atm": [float | None, ...],            # per expiry, vol pts as decimals
 "iv": [[float | None, ...], ...]}      # rows = expiries, cols = M_GRID
```

NaN → `None` conversion happens here (use
`[None if not np.isfinite(v) else round(float(v), 4) for v in row]`),
so `json.dumps` never sees NaN.

### 1c. `build_payload(df, days=SURF_DAYS)`

Full snapshot frame → the template payload.

* `dates = sorted(df["date"].unique())[-days:]`.
* For each date, for each symbol present that date: `symbol_surface`.
* Return:

```python
{"m_grid": [round(float(m), 4) for m in M_GRID],
 "dates": dates,                        # ascending
 "symbols": sorted(all symbols that produced >= 1 surface),
 "surfaces": {symbol: {date: surface_dict_or_absent}}}
```

A symbol absent on a given date is simply missing from that date's dict
(the JS handles it); a symbol with zero surfaces anywhere is excluded
from `symbols`.

### 1d. `render_page` + `main()`

* Copy `render_page` from `build_vol_tracker.py`, trimmed to two
  placeholders: `/*__CTX__*/` and `/*__SURF__*/`. Title:
  `Vol Surface`.
* `main()` argparse: `--snap-dir` (default `optsnap`), `--docs-out`
  (default `docs/vol_surface.html`), `--template`
  (default `vol_surface_template.html`), `--days` (int, default
  `SURF_DAYS`).
* Flow: `df = load_snapshots(args.snap_dir)`. If empty → print the
  no-snapshots line and render the stub with
  `ctx = {"built": ..., "days": 0, "note": "No chain capture yet -- the optsnap workflow appends the first snapshot after the next close."}`
  and `SURF = {}` — then return. Otherwise build the payload, print a
  one-line summary (`N days, M symbols, K surfaces`), render with
  `ctx = {"built": ..., "days": len(payload["dates"]), "first": ..., "last": ..., "nSymbols": len(payload["symbols"]), "note": None}`.

Commit: `Vol surface build script: gridded IV surfaces from optsnap history`.

---

## Phase 2 — tests (`tests/test_vol_surface.py`)

Copy the idiom of `tests/test_vol_tracker.py` — hand-built rows, no
disk, no network. Reuse this helper (same as the tracker tests):

```python
def _rows(date, symbol, expiry, spot, quotes):
    """quotes: list of (right, strike, iv, oi)."""
    return [{"date": date, "symbol": symbol, "expiry": expiry, "right": r,
             "strike": k, "iv": iv, "oi": oi, "volume": 0, "bid": 1.0,
             "ask": 1.2, "last": 1.1, "spot": spot}
            for r, k, iv, oi in quotes]
```

Required tests (names are the spec):

1. `test_surface_grid_flat_smile` — 5-point flat 30-vol smile on
   spot=100 (P 80/90, C 100/105/110): every grid cell whose moneyness
   lands inside [0.80, 1.10] is `approx(0.30)`; cells at 0.60 and 1.40
   are NaN (no extrapolation).
2. `test_surface_grid_thin_returns_none` — 3 points, or all strikes on
   one side of spot → `None`.
3. `test_symbol_surface_orders_and_caps_expiries` — build 16 expiries
   with valid smiles at increasing DTE: result has 14, `dtes` strictly
   ascending, `expiries`/`atm`/`iv` all length 14.
4. `test_symbol_surface_skips_dead_expiry` — one good expiry + one
   thin one → only the good one appears.
5. `test_build_payload_shape_and_day_cap` — 12 snapshot days, one
   symbol: `dates` is the last 10, ascending; symbol present in
   `symbols`; `json.dumps(payload)` succeeds (proves no NaN leaked).
6. `test_build_payload_excludes_surfaceless_symbol` — a symbol whose
   every expiry is thin does not appear in `symbols`.

Commit: `Vol surface tests: grid interpolation, expiry cap, payload hygiene`.

---

## Phase 3 — `vol_surface_template.html`

Create the template. Structure mirrors `vol_tracker_template.html`:

1. `<style>` — copy the **entire** `:root`/theme token block and the
   `.page/.wrap/.strip/.card/.note/.foot` rules from
   `vol_tracker_template.html` verbatim, then add viewer-specific rules
   (canvas sizing, control row). The canvas card:
   `canvas{width:100%; height:520px; display:block; cursor:grab; touch-action:none}`.
2. Header strip: link back — 
   `<a href="vol_tracker.html">&larr; Vol tracker</a>` — and the
   `built` span. `<h1>Vol surface</h1>`, a `.sub` paragraph: what the
   surface is (fixed grid, despiked live smiles, no extrapolation —
   holes are honest missing data).
3. Controls row (one `.card`): symbol `<select id="sym">`, date
   `<input type="range" id="day">` with a `mono` label showing the
   selected date, and a right `<label><input type="checkbox" id="spin"> spin</label>`.
4. The canvas card: `<canvas id="surf"></canvas>` plus a `.hint` line:
   "drag to rotate · scroll date slider to replay · color = IV level ·
   gold line = ATM".
5. A second card with a per-expiry readout table (`<table id="terms">`):
   expiry, DTE, ATM IV%, grid coverage (% non-null cells) for the
   selected symbol/date. Reuse the `table()` helper pattern from the
   tracker template.
6. Footer + payload script:

```html
<script>
const CTX = /*__CTX__*/;
const SURF = /*__SURF__*/;
```

### The renderer (transcribe this math; do not redesign it)

Model: the surface lives in unit box coordinates
`x ∈ [-1,1]` (moneyness, linear across `m_grid`),
`y ∈ [-1,1]` (sqrt-DTE: `y = 2*(sqrt(dte)-sqrt(dteMin))/(sqrt(dteMax)-sqrt(dteMin)) - 1`,
with the degenerate one-expiry case pinned to `y=0`),
`z ∈ [0,1]` (IV scaled by the symbol/date's own `[ivMin, ivMax]`,
padded 5% each side; if flat, pin `z=0.5`).

State: `yaw` (init `-0.65`), `pitch` (init `0.42`), updated by pointer
drag (`yaw += dx*0.008; pitch = clamp(pitch + dy*0.008, 0.05, 1.35)`).
Use Pointer Events (`pointerdown/move/up` + `setPointerCapture`) so
touch works. The `spin` checkbox runs
`requestAnimationFrame` adding `0.004` to yaw per frame.

Projection (orthographic — no perspective divide, no camera to get wrong):

```js
function project(x, y, z, W, H) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const rx = x * cy - y * sy;          // rotate about vertical axis
  const ry = x * sy + y * cy;
  const vy = ry * cp + (z - 0.5) * 1.1 * sp;   // tilt: z lifts the point
  const s = Math.min(W, H) * 0.36;
  return [W / 2 + rx * s, H * 0.56 - vy * s - (z - 0.5) * s * 0.55 * cp];
}
```

Draw, per frame:

1. Size the canvas to `devicePixelRatio` (`canvas.width = clientWidth*dpr`),
   scale the context once.
2. Build the quad list: for each adjacent expiry pair `(i, i+1)` and
   adjacent grid pair `(j, j+1)`, if all four `iv` corners are non-null,
   push `{depth, corners, iv}` where `iv` is the 4-corner mean and
   `depth` is the mean of the corners' rotated `ry` (compute `ry` with
   the same yaw math). Sort quads by `depth` descending and draw
   back-to-front (painter's algorithm — this is the whole hidden-surface
   story, no z-buffer).
3. Fill color: interpolate between two theme-agnostic stops by
   normalized IV `t`: low `rgba(70,140,200,a)` → high
   `rgba(212,99,95,a)` with `a = 0.85`; stroke each quad with the CSS
   `--line` color at 1px for the mesh look. (Read `--line`, `--accent`,
   `--ink-dim` via `getComputedStyle(document.documentElement)` once per
   draw so theme switches keep working.)
4. ATM line: for each expiry with non-null `atm`, project the point at
   `x` of moneyness 1.0 and `z` of its ATM IV; connect with a 2px
   `--accent` polyline drawn after the quads.
5. Axis labels (fillText, `--ink-dim`, 11px mono): moneyness `60% … 100% … 140%`
   along the front-left edge, the first/last expiry dates along the
   right edge, and `ivMin`/`ivMax` (as `xx.x%`) on the vertical.
6. A corner legend line (`fillText`): `SYMBOL · date · N expiries`.

Wiring: `sym` change and `day` input both call `draw()`; the date slider
`max` is `CTX.days - 1` and indexes `SURF_dates`; when the selected
symbol has no surface for the selected date, show the note div
("no capture for SYMBOL on DATE") and clear the canvas. Populate the
symbol dropdown from `Object.keys(SURF.surfaces ?? {})` sorted; default
symbol: first entry; default date: the latest. Stub mode
(`CTX.days === 0` or empty `SURF`): show `CTX.note` in the note div and
hide the controls/canvas cards.

Escape all injected strings with the same `esc()` helper the tracker
template uses.

Commit: `Vol surface template: dependency-free canvas 3D renderer`.

---

## Phase 4 — wiring

1. **`refresh.yml`**: after the "Build vol tracker" step (the
   `optsnap/` directory is already fetched two steps earlier), add:

```yaml
      # 3D vol surface viewer -> docs/vol_surface.html. Same snapshot data as the
      # vol tracker; renders a stub until the first capture exists.
      - name: Build vol surface viewer
        run: >
          python build_vol_surface.py
          --snap-dir optsnap
          --docs-out docs/vol_surface.html
```

   (No `--cache-dir` — this build touches no Yahoo panel data.)

2. **Cross-links**: in `vol_tracker_template.html`'s `.strip`, add
   `<a href="vol_surface.html">Vol surface &rarr;</a>` beside the
   regime-log link. The new page already links back (Phase 3).

3. **`VOL_TRACKER.md`**: one short paragraph under "Derived analytics"
   noting the surface viewer page and that it reads the same snapshots.

Commit: `Wire vol surface viewer into nightly build and page strip`.

---

## Phase 5 — local smoke test (no real data exists on this branch)

The `optsnap-data` branch is not checked out locally, so fabricate three
days of synthetic snapshots to eyeball the page (write the generator to
the scratchpad, not the repo):

```python
# scratch/make_fake_snaps.py -- synthetic 3-day capture for smoke testing
import numpy as np, pandas as pd
from pathlib import Path
out = Path("optsnap_fake"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
for i, date in enumerate(["2026-08-10", "2026-08-11", "2026-08-12"]):
    rows = []
    for sym, spot, base in [("GDX", 100.0, 0.30), ("NVDA", 180.0, 0.45)]:
        for expiry, dte in [("2026-09-18", 36), ("2026-12-18", 127),
                            ("2027-01-15", 155), ("2027-06-18", 309)]:
            t = dte / 365.25
            for k in np.arange(0.62, 1.38, 0.02) * spot:
                m = np.log(k / spot)
                iv = base + 0.18 * m * m / t ** 0.5 - 0.10 * m + 0.02 * i \
                     + rng.normal(0, 0.002)
                right = "P" if k < spot else "C"
                rows.append(dict(date=date, symbol=sym, expiry=expiry,
                                 right=right, strike=round(k, 1),
                                 iv=round(iv, 4), oi=1000, volume=10,
                                 bid=1.0, ask=1.1, last=1.05, spot=spot))
    pd.DataFrame(rows).to_csv(out / f"{date}.csv.gz", index=False,
                              compression="gzip")
print("wrote", out)
```

Then:

```
python scratch/make_fake_snaps.py          # writes optsnap_fake/ (gitignored area or scratchpad)
python build_vol_surface.py --snap-dir optsnap_fake --docs-out /tmp/vol_surface.html
python build_vol_surface.py --snap-dir /nonexistent --docs-out /tmp/vol_surface_stub.html
```

Open both outputs in a browser (or at minimum verify: both files exist,
the full one embeds two symbols and three dates in `SURF`, the stub one
shows the note and no `<canvas>` errors in the console). Delete
`optsnap_fake/` afterwards — synthetic chains must never be committed.

---

## Acceptance checklist (all must pass before the final push)

- [ ] `python -m pytest` — entire suite green, including the six new tests.
- [ ] `python build_vol_surface.py --snap-dir /nonexistent --docs-out ...`
      exits 0 and writes a stub page.
- [ ] Smoke-test page renders: surface visible, drag rotates, date
      slider swaps surfaces, ATM polyline drawn, holes (nulls) simply
      absent rather than extrapolated.
- [ ] `docs/vol_surface.html` output contains **no** external
      `<script src>` / `<link href>` — fully self-contained.
- [ ] Payload sanity: `json.dumps(build_payload(df))` contains no `NaN`
      token (test 5 covers this).
- [ ] `refresh.yml` diff is exactly one new build step; no other
      workflow edits; no edits to `optsnap.yml`.
- [ ] No changes to `build_vol_tracker.py`, `trade_structures.py`,
      `snapshot_option_chains.py` logic.
- [ ] No synthetic data files committed; `git status` clean after the
      final commit.
- [ ] Everything pushed to `claude/3d-vol-surface-viewer-a6z1g1`.

## Non-goals (do not build these)

Per-strike hover tooltips, term-structure charts, IV change ("ΔIV since
first capture") coloring modes, WebGL, exporting images, mobile pinch
zoom, and any server/API component. They are follow-ups, not this task.
