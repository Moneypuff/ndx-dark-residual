# Vol surface viewer — verification report & fix instructions

Agent memory file. Written after an independent verification pass over the
work on `claude/3d-vol-surface-viewer-a6z1g1` (commits `a2d8ea0`..`72112a3`,
per `VOL_SURFACE_VIEWER_PLAN.md`). **Two rendering bugs were found and
confirmed with screenshots; neither is fixed yet.** This file tells the next
agent exactly what to change, how to reproduce, and how to accept the fix.
Delete this file (or fold it into the PR description) once the fixes land.

Work on the same branch. Run `pip install -r requirements-dev.txt` first
(the container starts bare). Run `python -m pytest` before every commit —
the whole suite is green today (117 tests incl. 6 in
`tests/test_vol_surface.py`) and must stay green. Never push anywhere but
`claude/3d-vol-surface-viewer-a6z1g1`.

---

## Verified working (do not re-do)

- `build_vol_surface.py`: grid math, no-extrapolation NaN masking, 14-expiry
  cap, 10-day window, `--days` override, NaN-free JSON, stub page on missing
  snap dir (exit 0). All pinned by tests.
- Template runtime, checked headless (Chromium at `/opt/pw-browsers/chromium`,
  Playwright from `/opt/node22/lib/node_modules/playwright`): zero console/page
  errors across a full symbol×date sweep including edge data; drag-rotate,
  spin toggle, date scrubber, symbol switch, per-expiry table with correct
  coverage %, "no capture for SYM on DATE" note + canvas clear on missing
  (symbol, date), recovery after scrubbing back; dark-mode tokens correct;
  no external `<script src>`/`<link href>` in output.
- Camera framing: no clipping of surface or labels at default or extreme
  rotations (this was already found and fixed in commit `23495ad`).
- Wiring: `refresh.yml` parses, "Build vol surface viewer" sits right after
  "Build vol tracker"; tracker-page strip link renders; guarded files
  (`build_vol_tracker.py`, `trade_structures.py`, `snapshot_option_chains.py`,
  `optsnap.yml`) untouched vs `origin/main`.

---

## Bug 1 (fix required): a sparse expiry erases its neighbors' wing data

**Symptom.** A surface where one expiry has narrow strike coverage (e.g. a
thin chain that only quotes ±12% while its neighbors cover ±38%) renders as
a narrow ribbon: the *full* wings of the adjacent, well-covered expiries
disappear.

**Root cause.** In `vol_surface_template.html`, `draw()` renders **only**
quads, and a quad is dropped unless all four corners (rows *i*, *i+1* ×
cols *j*, *j+1*) are non-null. A row's cell is only ever displayed as the
corner of a quad shared with a neighbor row — so every non-null cell whose
neighbor-row cell is null becomes invisible, even though it is real data.

## Bug 2 (fix required): a single-expiry surface renders nothing

**Symptom.** A symbol whose surface has exactly one surviving expiry (33
grid cells of real smile data) draws a blank canvas: only the gold ATM dot
and axis labels appear.

**Root cause.** Same as Bug 1 taken to the limit: quads need `E >= 2` rows,
so with one row the primitive list is empty.

### The fix (one change covers both)

Add a second primitive — **orphan row-edge segments** — to the same
depth-sorted painter's list. For each adjacent pair of non-null cells inside
one expiry row, if that edge is not already an edge of any drawable quad
(the quad above and the quad below are both missing), draw it as a colored
line segment. Data then always shows at least as a wireframe smile; the
intact interior of the surface is unchanged (no double-drawing).

In `vol_surface_template.html`, inside `draw()`, replace everything from
`// Build quads (painter's algorithm: back-to-front, no z-buffer).` down
through the quad-drawing `for (const q of quads) { ... }` loop with:

```js
    // Primitives: quads where all 4 corners exist, plus "orphan" row-edge
    // segments -- adjacent non-null cells in one expiry row whose edge is
    // part of no drawable quad (sparse neighbor row, or a single captured
    // expiry). Without them a row's real data is invisible wherever its
    // neighbor rows have holes, and a one-expiry surface draws nothing.
    const quadOk = (i, j) => i >= 0 && i < E - 1 && j >= 0 && j < M - 1 &&
      zGrid[i][j] != null && zGrid[i][j + 1] != null &&
      zGrid[i + 1][j] != null && zGrid[i + 1][j + 1] != null;
    const prims = [];
    for (let i = 0; i < E - 1; i++) {
      for (let j = 0; j < M - 1; j++) {
        if (!quadOk(i, j)) continue;
        const z00 = zGrid[i][j], z01 = zGrid[i][j + 1];
        const z11 = zGrid[i + 1][j + 1], z10 = zGrid[i + 1][j];
        const corners = [
          [xs[j], ys[i], z00], [xs[j + 1], ys[i], z01],
          [xs[j + 1], ys[i + 1], z11], [xs[j], ys[i + 1], z10],
        ];
        const depth = corners.reduce((a, c) => a + rotYOnly(c[0], c[1]), 0) / 4;
        const ivMean = (z00 + z01 + z11 + z10) / 4;
        prims.push({ kind: "quad", depth, corners, ivMean });
      }
    }
    for (let i = 0; i < E; i++) {
      for (let j = 0; j < M - 1; j++) {
        if (zGrid[i][j] == null || zGrid[i][j + 1] == null) continue;
        if (quadOk(i - 1, j) || quadOk(i, j)) continue;   // edge already inside a quad
        const p1 = [xs[j], ys[i], zGrid[i][j]];
        const p2 = [xs[j + 1], ys[i], zGrid[i][j + 1]];
        prims.push({ kind: "seg",
                     depth: (rotYOnly(p1[0], p1[1]) + rotYOnly(p2[0], p2[1])) / 2,
                     p1, p2, ivMean: (p1[2] + p2[2]) / 2 });
      }
    }
    prims.sort((a, b) => b.depth - a.depth);

    for (const q of prims) {
      if (q.kind === "quad") {
        const pts = q.corners.map(c => project(c[0], c[1], c[2], W, H));
        ctx2d.beginPath();
        ctx2d.moveTo(pts[0][0], pts[0][1]);
        for (let k = 1; k < pts.length; k++) ctx2d.lineTo(pts[k][0], pts[k][1]);
        ctx2d.closePath();
        ctx2d.fillStyle = lerpColor(q.ivMean, 0.85);
        ctx2d.fill();
        ctx2d.strokeStyle = lineColor;
        ctx2d.lineWidth = 1;
        ctx2d.stroke();
      } else {
        const a = project(q.p1[0], q.p1[1], q.p1[2], W, H);
        const b = project(q.p2[0], q.p2[1], q.p2[2], W, H);
        ctx2d.beginPath();
        ctx2d.moveTo(a[0], a[1]);
        ctx2d.lineTo(b[0], b[1]);
        ctx2d.strokeStyle = lerpColor(q.ivMean, 0.9);
        ctx2d.lineWidth = 1.5;
        ctx2d.stroke();
      }
    }
```

Notes on the edge condition: row *i*'s edge (j, j+1) is the bottom edge of
the quad between rows *i−1*/*i* (`quadOk(i-1, j)`) and the top edge of the
quad between rows *i*/*i+1* (`quadOk(i, j)`). Skip the segment when either
exists. `quadOk` bounds-checks, so `i = 0` and `i = E-1` are safe. Update
the `.hint` line in the canvas card to mention that sparse chains render as
wireframe smile lines.

### Reproduce → verify → accept

The scratchpad from the review session is gone; regenerate the edge data
with this generator (run from the repo root; **delete `optsnap_edge/`
afterwards — never commit synthetic chains**):

```python
# make_edge_snaps.py -- GDX: 4 wide expiries. NVDA: middle expiry only
# +/-12% wide (Bug 1). IONQ: single expiry (Bug 2). SPCX: absent on the
# middle day (note-path regression check).
import numpy as np, pandas as pd
from pathlib import Path
out = Path("optsnap_edge"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(11)
def chain(date, sym, spot, base, expiry, dte, width):
    t = dte / 365.25; rows = []
    for k in np.arange(1 - width, 1 + width, 0.02) * spot:
        m = np.log(k / spot)
        iv = base + 0.18 * m * m / t ** 0.5 - 0.10 * m + rng.normal(0, 0.002)
        rows.append(dict(date=date, symbol=sym, expiry=expiry,
                         right="P" if k < spot else "C", strike=round(k, 1),
                         iv=round(iv, 4), oi=1000, volume=10,
                         bid=1.0, ask=1.1, last=1.05, spot=spot))
    return rows
for i, date in enumerate(["2026-08-10", "2026-08-11", "2026-08-12"]):
    rows = []
    for expiry, dte in [("2026-09-18", 36), ("2026-12-18", 127),
                        ("2027-01-15", 155), ("2027-06-18", 309)]:
        rows += chain(date, "GDX", 100.0, 0.30 + 0.02 * i, expiry, dte, 0.38)
    rows += chain(date, "NVDA", 180.0, 0.45, "2026-09-18", 36, 0.38)
    rows += chain(date, "NVDA", 180.0, 0.45, "2026-12-18", 127, 0.12)
    rows += chain(date, "NVDA", 180.0, 0.45, "2027-06-18", 309, 0.38)
    rows += chain(date, "IONQ", 40.0, 0.80, "2026-12-18", 127, 0.30)
    if date != "2026-08-11":
        rows += chain(date, "SPCX", 25.0, 0.55, "2026-12-18", 127, 0.30)
    pd.DataFrame(rows).to_csv(out / f"{date}.csv.gz", index=False,
                              compression="gzip")
```

Build (`python build_vol_surface.py --snap-dir optsnap_edge --docs-out
/tmp/vs.html` — expect `3 day(s), 4 symbol(s), 11 surface(s)`), load
`file:///tmp/vs.html` in headless Chromium, select each case on the last
date, **screenshot the canvas and look at it** — the screenshot is the
load-bearing check:

- **IONQ**: a single curved smile polyline spanning the moneyness axis (not
  just the gold ATM dot).
- **NVDA**: the ribbon of near-ATM quads as before, **plus** wireframe wing
  curves on the front and back expiries where the sparse middle row killed
  the quads.
- **GDX**: visually unchanged from today (no double-drawn interior edges).
- The full symbol×date sweep still logs zero console/page errors, and the
  SPCX missing-day note still behaves.

Pixel sanity bounds (same setup: 1200×900 viewport, dpr 1, generator above,
last date, default rotation; count = alpha>0 pixels via `getImageData`).
Pre-fix baselines measured during this review: GDX 73,494 · NVDA 21,305 ·
IONQ 1,910. Post-fix expect IONQ ≈ +500–2,500 and NVDA ≈ +400–3,000 over
baseline, GDX within ~1% of baseline. Treat these as sanity bounds — if a
count moves the wrong way or GDX shifts a lot, look at the screenshot
before trusting the number.

Commit the fix to the same branch with a message explaining the orphan-edge
primitive, then push (`git push -u origin claude/3d-vol-surface-viewer-a6z1g1`).

---

## Minor polish (fold into the same fix commit; skip if any conflicts)

1. **Legend grammar**: `1 expiries` → pluralize:
   `` `... · ${expiries.length} ${expiries.length === 1 ? "expiry" : "expiries"}` ``.
2. **Axis-label font**: `draw()` sets `ctx2d.font = "11px " +
   (getComputedStyle(canvas).fontFamily || "monospace")`, which inherits the
   page's *sans* stack; the plan specified mono. Use the token:
   `getComputedStyle(document.querySelector(".page")).getPropertyValue("--mono")`
   (fallback `"monospace"`).
3. **Hardcoded moneyness labels**: `mLabels` pins `0.60/1.00/1.40`; derive
   ends from `SURF.m_grid[0]` / `SURF.m_grid[SURF.m_grid.length - 1]` so a
   future `M_GRID` change in Python can't make the axis lie.

Known-and-accepted (no action): the ATM polyline is always drawn on top of
the surface even at rotations where it should be occluded, and axis labels
anchor to fixed cube corners so they sit "behind" the surface after a 180°
yaw — both cosmetic, inherent to the painter's-algorithm design.
