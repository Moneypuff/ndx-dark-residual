# The path of a high 10-day-average DPI in a downtrend

Request: test the **path** — not just the endpoint — of high 10-day-average-DPI
downtrend names. The falling-knife study measured the downside of the path (MAE)
for the LOW group; this study characterises the whole path for the HIGH side and
for the persistence study's streak buckets: forward **MAE** (worst point of the
next h sessions), **MFE** (best point), **TILT = MFE + MAE** (bounce-vs-drawdown
asymmetry), and the **timing** of trough and peak. Full S&P 500, 2019–2026,
among 3-month-downtrend names; daily cross-sectional aggregation with
block-bootstrap CIs; complete forward windows only. Reproduce:
```
python spx_dpi_path_study.py --start 2019-01-01 --out spx_dpi_path.csv
```

**Short answer: the *level* of the 10-day-average DPI does nothing to the path —
HIGH and LOW downtrend names trace statistically identical paths (same drawdown,
same bounce, same timing). The *streak* (10+ consecutive high-decile days) does
leave one real path signature: its bounce reaches ~0.7pp higher (MFE +0.72
[+0.28, +1.21] at 21d), with no drawdown reduction, and the MFE edge is the only
number in the whole single-stock thread that still clears zero after the
short-term-reversal control (+0.51 [+0.08, +0.96] full-sample). But it is
upside-reach only, ~half a point, borderline in each OOS half, gone by 42d on
the control — a characterization, not a tradeable edge.**

## 1. The HIGH group (the literal question): path indistinguishable from LOW

**21-day path, downtrend names:**

| group | mean MAE | mean MFE | TILT | P(mae<−10) | P(mfe>+10) | t-trough | t-peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOW | −5.22 | +6.93 | +1.71 | 16.3% | 23.6% | 9.9 | 12.0 |
| MID | −5.43 | +6.98 | +1.56 | 17.2% | 23.9% | 10.0 | 12.0 |
| **HIGH** | −5.39 | +7.12 | +1.73 | 16.4% | 24.0% | 10.0 | 12.1 |

HIGH−LOW: MAE −0.19 [−0.43, +0.05] · MFE +0.17 [−0.11, +0.45] · TILT −0.03
[−0.45, +0.40] · t-trough +0.12 [−0.10, +0.32]. Same at 42d. **High dark flow
does not shorten the drawdown, deepen the bounce, or bring the bottom closer.**
In particular it is *not* a bottom-timing signal: the trough arrives at the same
point of the window (~day 10 of 21, ~day 19 of 42) in every DPI group.

## 2. The streak: the one path signature — a higher best print, not a safer path

**21-day, streak buckets among downtrend names:**

| streak | mean MAE | mean MFE | TILT | P(mfe>+10) |
|---|---:|---:|---:|---:|
| 0 | −5.36 | +6.98 | +1.61 | 23.8% |
| 1–4 | −5.39 | +6.88 | +1.49 | 23.2% |
| 5–9 | −5.35 | +7.13 | +1.78 | 23.8% |
| **10+** | −5.50 | **+7.79** | **+2.29** | **26.5%** |

S10+−S0 differences: **MFE +0.72 [+0.28, +1.21]** and **TILT +0.65 [+0.03,
+1.26]** clear zero; **MAE −0.07 [−0.39, +0.25]** is exactly flat. The gradient
is monotone in MFE from streak 1–4 up, echoing the endpoint gradient. OOS: TILT
+0.51 IS / +0.78 OOS (point estimates stable; only OOS individually clears). At
42d MFE +0.78 [+0.17, +1.46] still clears raw; TILT +0.55 [−0.31, +1.42] doesn't.

So the streak's fingerprint on the path is **asymmetric**: identical worst
moment, best moment ~0.7pp higher and a ~2.7pp higher chance of touching +10%
intraperiod. It buys extra reach on the bounce, zero downside protection.

## 3. The reversal control — the MFE edge is the survivor, barely

Same Fama-MacBeth that deflated the endpoint alpha (path metric ~ rback +
streak10, among downtrend names; streak10 coefficient = path effect beyond the
recent-return characteristic):

| metric (21d) | full | IS 2019–22 | OOS 2023–26 |
|---|---|---|---|
| MAE | −0.11 [−0.41, +0.20] | −0.27 | +0.04 |
| **MFE** | **+0.51 [+0.08, +0.96]** | +0.64 [−0.04, +1.37] | +0.39 [−0.11, +0.90] |
| TILT | +0.40 [−0.18, +0.99] | +0.38 | +0.43 |

Recall the endpoint alpha fell from +0.71% to an insignificant +0.15pp under
this control. The **MFE coefficient keeps ~70% of its raw size (+0.51 of +0.72)
and still clears zero in the full sample** — the only post-control CI-positive
number the single-stock investigation has produced. Both halves are positive
with near-matching point estimates but individually borderline; at 42d the
controlled MFE drops to +0.43 [−0.17, +1.06], no longer clearing.

## Verdict

- **On the level — the question as asked — the answer is a clean null.** A high
  10-day-average DPI in a downtrend changes nothing about the forward path:
  drawdown, bounce, asymmetry, and timing are all indistinguishable from
  low-DPI names. This completes the symmetry with the falling-knife study (LOW
  doesn't mark knives; HIGH doesn't mark springs).
- **On the streak, the path finally explains where the endpoint alpha was
  hiding.** The 10+ streak's +0.71% endpoint alpha (which deflated under the
  reversal control) shows up on the path as *upside reach*: the bounce's best
  print sits ~0.5–0.7pp higher, robust to the reversal control at 1 month, with
  no drawdown change. Mechanically that is worth something only to a seller into
  strength (profit-taking limits inside the window), not to a hold-to-horizon
  book — consistent with the endpoint result washing out.
- **Do not trade it.** One full-sample CI clearing zero at 95%, after the dozens
  of comparisons this thread has run, is roughly what false-discovery arithmetic
  predicts; each OOS half is borderline; the effect halves by 42d; and ~0.5pp of
  best-print reach is inside single-stock execution noise. The honest reading:
  the streak is a real but faint *characterization* of how these names bounce,
  not an edge. The thread's throughline stands — dark flow's tradeable signal is
  systematic/index-level, not single-stock.

## Caveats

- Daily closes: intraday extremes are beyond both MAE and MFE, symmetrically
  across groups.
- Complete forward windows only (last h sessions dropped), so MAE here can
  differ a hair from the falling-knife study, which allowed half windows.
- t-trough/t-peak means are window-mechanical (a min over h sessions drifts
  toward the middle); they are compared *across groups*, where the mechanics
  cancel — the flatness across groups is the result.
- Machinery unit-tested: hand-computed MAE/MFE/timing, NaN propagation, planted
  friendly-path detection, and FM isolation of a planted streak-MFE effect
  (`tests/test_dpi_path.py`).
