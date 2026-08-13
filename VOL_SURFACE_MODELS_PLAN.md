# Vol surface viewer — smile-dynamics model toggle (implementation plan)

Adds **sticky-strike / sticky-delta / hybrid** model views to the 3D vol
surface viewer (`build_vol_surface.py` + `vol_surface_template.html`,
built on `claude/3d-vol-surface-viewer-a6z1g1`, currently at `80c9d38`).
A toggle switches the rendered surface between the observed smile and
each model's **one-day-ahead prediction**, plus a residual view
(observed − predicted) that makes the model's misses visible strike by
strike. The hybrid encodes the empirical finding this feature exists to
display: **wings behave sticky-strike, ATM behaves sticky-delta** — so
the hybrid blends the two by distance-from-ATM in sigma units.

This document is the whole spec. Every decision is already made; do not
re-open them. Follow the phases in order.

---

## Rules (read first, apply throughout)

1. **Branch**: all work on `claude/3d-vol-surface-viewer-a6z1g1`. Never
   push anywhere else. Run `pip install -r requirements-dev.txt` first
   (the container starts bare). `python -m pytest` after every phase —
   the suite is green today (117 tests) and must stay green.
2. **No network, no new dependencies** (Python: numpy+pandas; template:
   vanilla JS, zero CDN scripts). Same as before.
3. **All financial math in Python, unit-tested.** The browser receives
   pre-computed predicted grids and only selects, subtracts (for the
   residual), scales, and draws. Plain arithmetic (a − b, a lerp, a
   mean of |differences| for the table) is allowed in JS; interpolation,
   sigma math, and weight functions are not.
4. **A prediction may only use day t−1's surface plus day t's spot.**
   The sticky models are conditional forecasts: "given the spot move,
   where does each rule put the smile?" Nothing else from day t leaks in.
5. **Script and template change together in one commit** — the payload
   contract changes, so a half-updated pair renders a broken page.
6. **Degrade honestly.** First snapshot day, or a symbol with no capture
   the day before: model views are unavailable → disable the model
   buttons and show the note ("model views need the prior snapshot
   day"). Never fabricate a prediction.
7. Don't touch `build_vol_tracker.py`, `trade_structures.py`,
   `snapshot_option_chains.py`, `optsnap.yml`, or the other page builds.
8. Commit per phase, descriptive messages, push with
   `git push -u origin claude/3d-vol-surface-viewer-a6z1g1`.
   Delete this plan file in the final commit once everything passes.

---

## The three models, precisely (this is the math to implement)

Setup: the payload's grids live on a fixed moneyness grid
`M_GRID` (m = K/S, 33 points, 0.60..1.40). For a (symbol, expiry) pair,
let `G_prev[j]` be day t−1's observed grid row, `S_prev` / `S_t` the two
spots, and `r = S_t / S_prev` the spot ratio. All predictions are rows on
the same grid, aligned to **day t's** expiry list, matched to day t−1 by
**exact expiry string**; an expiry with no match yesterday predicts null.

**Invariant you may rely on** (from `surface_grid`'s no-extrapolation
rule): the non-null cells of any grid row form one contiguous block —
nulls only at the wings, never interior holes.

### 1. Sticky strike
Each absolute strike keeps its IV: `IV_t(K) = IV_{t-1}(K)`. Today's cell
j is strike `K = m_j · S_t`, which sat at moneyness `m' = m_j · r`
yesterday. So:

    pred_strike[j] = interp(G_prev at moneyness m_j * r)

Linear interpolation over the contiguous non-null block of `G_prev`
(x = the M_GRID values of non-null cells, y = their IVs); **null** when
`m_j * r` falls outside the block or the block has < 2 cells. Sanity: a
rally (r > 1) samples yesterday's smile further right, so with a normal
downward skew the fixed-moneyness cells "roll down the skew" — the
classic sticky-strike behavior.

### 2. Sticky delta (moneyness proxy)
The smile rides with spot: `IV_t(m) = IV_{t-1}(m)`. On a moneyness grid
this is yesterday's row verbatim:

    pred_delta[j] = G_prev[j]

We implement **sticky-moneyness as the standard practical proxy** for
sticky delta — true delta-bucketing needs a full BS delta inversion and
adds nothing at daily granularity on this data. Say so in the page hint
("sticky delta (moneyness proxy)") and in the module docstring.

### 3. Hybrid (the researched finding: wings sticky-strike, ATM sticky-delta)
Blend by distance from ATM measured in **sigma-move units**, the repo's
existing convention (`trade_structures.sigma_to_moneyness`: a 1-sigma
strike offset is `atm_iv * sqrt(T)` of fractional moneyness):

    d_j = |m_j - 1| / (atm_prev * sqrt(dte_prev / 365.25))
    w_j = smoothstep of d_j between W_LO and W_HI
          (0 for d <= W_LO; 1 for d >= W_HI; else u = (d - W_LO) / (W_HI - W_LO),
           w = 3u^2 - 2u^3)
    pred_hybrid[j] = w_j * pred_strike[j] + (1 - w_j) * pred_delta[j]

with module constants `W_LO = 0.5`, `W_HI = 1.5` (sigma units) — inside
half a sigma the option is ATM-ish (pure sticky-delta), beyond 1.5 sigma
it is a wing (pure sticky-strike), smooth in between. Comment them as
the calibration knobs for a later phase; do not fit them now.
`atm_prev` / `dte_prev` come from **day t−1's** surface (rule 4). If
`atm_prev` is null, substitute the non-null `G_prev` cell nearest
m = 1.0 (it exists for any valid row). `pred_hybrid[j]` is null when
either input is null.

**Residual** (computed in JS, it's one subtraction):
`residual[j] = observed[j] - pred[j]`, null if either side is null.
Positive = the market re-priced that strike **above** the model.

---

## Payload contract change

`symbol_surface` output for (symbol, date t) gains one optional key:

```python
{"spot": ..., "expiries": [...], "dtes": [...], "atm": [...], "iv": [[...]],
 "models": {                      # ABSENT when day t-1 has no surface
     "strike": [[float|None, ...], ...],   # rows aligned to THIS dict's expiries
     "delta":  [[float|None, ...], ...],
     "hybrid": [[float|None, ...], ...],
 }}
```

Same rounding (4 dp) and NaN→None hygiene as `iv`. Size: roughly 4× the
per-surface grid data → full-universe payload grows from ~1–2 MB toward
~4–7 MB. Acceptable (the repo ships a 13 MB index.html); if it ever
matters, the lever is `--days`, not dropping models.

---

## Phase 1 — Python (`build_vol_surface.py`)

Add near the constants:

```python
W_LO = 0.5    # sigma-distance where the hybrid starts leaving sticky-delta
W_HI = 1.5    # ...and is fully sticky-strike. The calibration knobs.
```

New pure functions (keep them under the "Pure computation" banner,
docstrings in the repo's voice):

* `hybrid_weight(m, atm_iv, dte)` → w in [0,1] per the formula above.
  Vectorized over `m` (an ndarray) or scalar — implementer's choice, but
  the tests below call it with scalars.
* `predict_row_strike(prev_row, ratio)` → list of 33 float/None. Inputs:
  `prev_row` as a list of float/None (payload form), `ratio = S_t/S_prev`.
  Build the contiguous non-null block, `np.interp` at `M_GRID * ratio`,
  null outside the block's moneyness span or when block < 2 cells.
* `model_grids(prev_surf, cur_surf)` → the `"models"` dict or `None`.
  For each expiry in `cur_surf["expiries"]`: find its index in
  `prev_surf["expiries"]` (exact string match; missing → three null
  rows). `ratio = cur_surf["spot"] / prev_surf["spot"]`.
  delta row = prev row verbatim; strike row = `predict_row_strike`;
  hybrid row = the w-blend using `prev_surf["atm"][k]` /
  `prev_surf["dtes"][k]` (nearest-to-ATM fallback for a null atm).
  Return `None` if every row of every model is null (all-miss ⇒ treat
  as no models rather than shipping dead weight).

Wire into `build_payload`: it already iterates dates ascending — keep,
per symbol, the previous date's surface dict as it goes; after building
`surf` for (sym, d), if a previous surface exists, attach
`surf["models"] = model_grids(prev, surf)` (skip the key when that
returns `None`). The first carried date never gets models even if older
snapshots exist on disk — the payload window is the universe; predicting
across the window edge from data the page can't show would be
unverifiable on the page. (One-line comment saying exactly that.)

Commit: `Vol surface models: sticky-strike / sticky-delta / hybrid predictions in the payload`.

---

## Phase 2 — tests (`tests/test_vol_surface.py`, extend in place)

Reuse `_rows` / `FLAT_SMILE`. New tests (names are the spec):

1. `test_hybrid_weight_shape` — w(0 sigma)=0, w(0.5)=0, w(1.0)=0.5,
   w(1.5)=1, w(3)=1; monotone non-decreasing on a sweep. (Call with
   d expressed via m: pick atm_iv=0.30, dte=365, so 1 sigma ≈ 0.2996 of
   moneyness — compute m from d for each case.)
2. `test_predict_row_strike_shift` — prev row = a linear skew in m
   (e.g. `iv = 0.30 - 0.20*(m-1)` on the non-null block), ratio 1.10:
   for an interior cell, prediction equals the analytic value at
   `m*1.10`; cells whose shifted lookup leaves the block are None.
3. `test_predict_row_strike_flat_ratio1` — flat row, ratio 1.0 → equals
   input on the block, None off it.
4. `test_model_grids_delta_is_prev_row` — two hand-built surfaces, same
   expiry: delta rows == prev iv rows; expiry present today only →
   all-None rows in all three models.
5. `test_model_grids_hybrid_interpolates_between` — with a skewed prev
   row and ratio ≠ 1: at the grid cell nearest m=1, hybrid ==
   delta prediction (w=0); at the deepest non-null wing cell with
   d ≥ 1.5 sigma, hybrid == strike prediction (w=1); strictly between
   the two somewhere in the transition band.
6. `test_build_payload_models_presence_and_hygiene` — 3 days, one
   symbol: day 1 surface has no `"models"`, days 2–3 do;
   `json.dumps(payload)` contains no `NaN`.
7. `test_sticky_strike_world_zero_residual` — **the by-construction
   dynamics test.** Build day 1 and day 2 where IV is the SAME pure
   function of absolute strike K both days (e.g. `iv = 0.30 -
   0.001*(K-100)`) while spot moves 100 → 105. Then
   `models["strike"]` row minus day-2 observed row ≈ 0 (1e-6) on every
   cell where both are non-null, and the delta-model residual is
   materially nonzero (≥ 0.001 somewhere).
8. `test_sticky_delta_world_zero_residual` — mirror image: IV the same
   pure function of moneyness both days, spot moves → delta residual
   ≈ 0, strike residual materially nonzero.

Tests 7–8 are the acceptance core: they verify the models against
worlds where the true dynamics are known by construction.

Commit: `Vol surface model tests: weight shape, strike-shift interp, by-construction dynamics`.

---

## Phase 3 — template (`vol_surface_template.html`)

### Controls
In the controls card, add a second row under the existing fields:

```html
<div class="controls" style="margin-top:10px">
  <div class="seg" id="modelSeg">
    <button data-model="obs" class="on">Observed</button>
    <button data-model="strike">Sticky strike</button>
    <button data-model="delta">Sticky delta</button>
    <button data-model="hybrid">Hybrid</button>
  </div>
  <label class="inline" id="residWrap"><input type="checkbox" id="resid"> residual (obs − model)</label>
</div>
```

CSS (matches the page's tokens):

```css
.seg{display:flex; border:1px solid var(--line); border-radius:8px; overflow:hidden}
.seg button{background:var(--surface-2); color:var(--ink-dim); border:0; padding:6px 12px;
  font:inherit; font-size:12.5px; cursor:pointer; border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent-soft); color:var(--accent); font-weight:600}
.seg button:disabled{opacity:.45; cursor:default}
```

### State & wiring
`let model = "obs", residual = false;` Click on a seg button → set
`model`, move the `on` class, call `refresh()`. `resid` change →
`residual = checkbox && model !== "obs"`, `refresh()`. Disable the
checkbox (and uncheck it) while `model === "obs"`.

In `refresh()`, after finding `surf`:
* Pick the active grid: `obs` → `surf.iv`; else `surf.models?.[model]`.
  If the models key is absent (first date / gap day), disable the three
  model buttons, force `model = "obs"` back on, and show the note
  "model views need the prior snapshot day" (re-enable the buttons when
  a models-bearing surface is selected).
* When `residual` is true, hand `computeGeom` the elementwise
  `obs − pred` grid (null if either side is null) and a `diverging`
  flag.

### `computeGeom` changes
Accept `(grid, {diverging})` instead of reading `surf.iv` directly
(`atm`/`dtes`/`expiries` still come from `surf`). For `diverging`:
z-scale symmetrically — `rmax = max(|cell|)` over non-null cells
(fallback 0.005 when everything is ~0 so a perfect model still renders
a flat plane), `z = 0.5 + cell / (2 * rmax)` clamped to [0,1], and set
`geom.rmax` for the axis labels. Non-diverging path unchanged.

### Colors
Keep `lerpColor` for surfaces. Add a diverging ramp for residuals
(neutral at zero-miss):

```js
function divColor(t, a) {            // t in [0,1]; 0.5 = model matched
  t = Math.max(0, Math.min(1, t));
  const lo = [70, 140, 200], mid = [156, 160, 165], hi = [212, 99, 95];
  const mix = (u, v, k) => u.map((x, i) => Math.round(x + (v[i] - x) * k));
  const c = t < 0.5 ? mix(lo, mid, t * 2) : mix(mid, hi, (t - 0.5) * 2);
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}
```

In the primitive-draw loop use `residual ? divColor : lerpColor` (same
alphas). The orphan row-edge segments need no changes — they operate on
whatever `zGrid` holds, and model grids can have *different* null
patterns (the strike shift chops a wing), which is exactly what they're
for.

### Labels, ATM line, terms table
* Vertical axis in residual mode: `±(rmax*100).toFixed(1)` labeled
  `Δvol pts` at z=0 and z=1 (replace the ivMin/ivMax labels there);
  model-surface and observed modes keep the current labels.
* Gold ATM polyline: drawn in `obs` and model-surface modes (observed
  ATM, as a reference); **skip it in residual mode** — it has no meaning
  on a difference surface.
* Terms table: when `model !== "obs"`, add a column
  `miss (vol pts)` = mean of `|obs[j] − pred[j]| * 100` over cells where
  both are non-null ("—" when none). This is the number that shows the
  research finding: sticky-delta's miss is smallest on front/ATM-heavy
  expiries, sticky-strike's in the wings, hybrid ≤ both on mixed rows.
* Canvas hint line: extend with
  `· models predict day t from day t−1's smile + today's spot · sticky delta = moneyness proxy`.
* Corner legend: append the mode, e.g. `· sticky strike (residual)`.

Commit: **same commit as Phase 1** if you prefer atomicity, or a
separate one — but never push Phase 1 without Phase 3 (Rule 5).
Recommended: one commit spanning both after Phase 4 verification, or
commit Phase 1+2 first (payload additions are backward-compatible —
the old template ignores `"models"`) and Phase 3 second. Either is
acceptable; backward compatibility makes the two-commit path safe.

---

## Phase 4 — verify (synthetic worlds where the truth is known)

Regenerate edge data with a generator that builds **by-construction
dynamics** (write it in the scratchpad, run from repo root, delete
`optsnap_model/` after — never commit synthetic chains):

```python
# make_model_snaps.py -- STRK: IV is a fixed function of absolute strike
# (a sticky-strike world), spot 100 -> 105 -> 110. DELT: IV a fixed
# function of moneyness (a sticky-delta world), same spot path. GDX:
# the old mixed-noise symbol as a regression check. No RNG noise on
# STRK/DELT so residuals are limited only by grid interpolation error.
import numpy as np, pandas as pd
from pathlib import Path
out = Path("optsnap_model"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(11)
EXPIRIES = [("2026-09-18", 36), ("2026-12-18", 127), ("2027-06-18", 309)]
def rows_for(date, sym, spot, ivfn, width=0.38):
    rows = []
    for expiry, dte in EXPIRIES:
        for k in np.arange(1 - width, 1 + width, 0.02) * spot:
            rows.append(dict(date=date, symbol=sym, expiry=expiry,
                             right="P" if k < spot else "C",
                             strike=round(k, 2), iv=round(ivfn(k, spot, dte), 4),
                             oi=1000, volume=10, bid=1.0, ask=1.1,
                             last=1.05, spot=spot))
    return rows
strike_world = lambda k, s, dte: 0.30 - 0.0012 * (k - 100) + 0.02 * (365 / (dte + 365))
delta_world  = lambda k, s, dte: 0.30 - 0.25 * np.log(k / s) + 0.02 * (365 / (dte + 365))
gdx_world    = lambda k, s, dte: 0.30 + 0.18 * np.log(k/s)**2 / (dte/365.25)**.5 \
                                  - 0.10 * np.log(k/s) + rng.normal(0, 0.002)
for date, spot in [("2026-08-10", 100.0), ("2026-08-11", 105.0), ("2026-08-12", 110.0)]:
    rows = (rows_for(date, "STRK", spot, strike_world)
            + rows_for(date, "DELT", spot, delta_world)
            + rows_for(date, "GDX", spot, gdx_world))
    pd.DataFrame(rows).to_csv(out / f"{date}.csv.gz", index=False, compression="gzip")
print("wrote", out)
```

Build (`python build_vol_surface.py --snap-dir optsnap_model --docs-out
/tmp/vsm.html`), then verify in headless Chromium (Playwright at
`/opt/node22/lib/node_modules/playwright`, browser
`/opt/pw-browsers/chromium`, 1200×900). The screenshots are the
load-bearing check:

* **STRK, day 3, sticky-strike residual**: a near-uniform neutral-gray
  plane (misses ≈ interpolation error only, well under 0.5 vol pt).
  Same symbol under **sticky-delta residual**: a visibly tilted/colored
  surface (the 5% spot move times the strike-space skew).
* **DELT, day 3**: the mirror image — delta residual neutral, strike
  residual colored.
* **Hybrid residual on both**: neutral near the ATM band on DELT,
  neutral in the wings on STRK — the blend visibly favoring the right
  model in the right region. Terms-table `miss` column: on STRK,
  strike-miss < delta-miss per expiry; on DELT the reverse; hybrid
  between or better.
* **Day 1**: model buttons disabled, note shown, Observed still renders.
* **GDX Observed**: pixel-identical behavior to today's build (the
  observed path must not change — compare against a control build from
  `git stash` / the committed template if in doubt).
* Full sweep (3 symbols × 3 dates × 4 modes × residual on/off): zero
  console/page errors.
* `grep` the built page: still no external `<script src>`/`<link href>`.

Then `rm -rf optsnap_model`, run the full pytest suite, commit, push.

---

## Acceptance checklist (all must pass before the final push)

- [ ] Full suite green including the 8 new tests (tests 7–8 pin the
      dynamics by construction).
- [ ] Stub build (`--snap-dir /nonexistent`) still exits 0.
- [ ] Browser: model toggle + residual verified per Phase 4, zero
      console errors, Observed mode visually unchanged.
- [ ] First-date / gap-day behavior: buttons disabled, honest note.
- [ ] Payload: no `NaN` token; `"models"` absent on first carried date.
- [ ] No edits outside `build_vol_surface.py`,
      `tests/test_vol_surface.py`, `vol_surface_template.html`
      (plus deleting this plan file).
- [ ] No synthetic data committed; tree clean; pushed to
      `claude/3d-vol-surface-viewer-a6z1g1`.
- [ ] This plan file deleted in the final commit.

## Non-goals (do not build)

True delta-bucketed sticky delta (BS inversion), sticky-local-vol or
term-structure stickiness, fitting `W_LO`/`W_HI` from realized data
(that's a later calibration phase — the constants are deliberately
exposed), multi-day-horizon predictions, a model-scoring history page,
and any change to the vol tracker's own `local_repricing` machinery.
