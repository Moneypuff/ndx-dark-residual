#!/usr/bin/env python3
"""
Figure rendering for the vol-tracker page (build_vol_tracker --docs-out).

Three figures, each rendered twice -- once per page theme, using the same
design tokens as vol_tracker_template.html so the images sit natively on
the card surface in either mode -- and returned as base64 PNGs the page
swaps with the viewer's theme:

  * paths   -- the universe path map: median post-up-break path per ETF,
               the chase risk/reward scatter, capitulation bars, and the
               turn-signal bars (the ETF_PATH_PLAYBOOK view).
  * expmove -- implied ~3M / ~6M straddle moves vs the signal-conditional
               E|move| at 63/126 sessions (EXPECTED_MOVE_FINDINGS view).
  * skew    -- risk reversal at the +/-1 sigma-move strikes and the
               upside-call-positioning quadrant, ~3M tenor, liquid chains.

This module only draws: build_vol_tracker.chart_inputs() assembles the
data. Everything degrades -- a figure whose input is empty is skipped.
"""
import base64
import io

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {"surface": "#ffffff", "ink": "#16202b", "dim": "#54626f",
              "faint": "#8794a1", "grid": "#e3e9ef", "base": "#d3dbe3",
              "blue": "#2a78d6", "blue2": "#8ab4e8", "orange": "#eb6834",
              "aqua": "#1baf7a", "red": "#e34948", "red2": "#f0a09f"},
    "dark": {"surface": "#141b24", "ink": "#e9eef3", "dim": "#96a4b3",
             "faint": "#5d6b7a", "grid": "#243040", "base": "#383f4c",
             "blue": "#3987e5", "blue2": "#6390c4", "orange": "#d95926",
             "aqua": "#2bc492", "red": "#e66767", "red2": "#a95c5c"},
}


def _style(t):
    return {"font.family": "sans-serif", "font.size": 9,
            "text.color": t["ink"], "axes.edgecolor": t["base"],
            "axes.labelcolor": t["dim"], "xtick.color": t["faint"],
            "ytick.color": t["faint"], "axes.titlesize": 10.5,
            "axes.titleweight": "bold"}


def _ax(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)


def _b64(fig, t):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=t["surface"])
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------
def fig_paths(paths, t):
    """2x2 universe path map."""
    med_paths, up, dn, turn = (paths["med_paths"], paths["up"], paths["dn"],
                               paths["turn"])
    if not med_paths:
        return None
    with plt.rc_context(_style(t)):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), facecolor=t["surface"])
        fig.subplots_adjust(hspace=0.46, wspace=0.26, left=0.07, right=0.98,
                            top=0.94, bottom=0.06)

        # A: median up-break paths
        ax = axes[0, 0]
        _ax(ax, t)
        hl = {"QQQ": t["blue"], "SMH": t["aqua"], "GDX": t["orange"]}
        for sym, p in med_paths.items():
            if sym not in hl:
                ax.plot(np.arange(len(p)), p, color=t["faint"], lw=0.8,
                        alpha=0.4, zorder=1)
        for sym, col in hl.items():
            if sym in med_paths:
                p = med_paths[sym]
                ax.plot(np.arange(len(p)), p, color=col, lw=2.0, zorder=3)
                ax.annotate(f"{sym} {p[-1]:+.1f}%", (len(p) - 2, p[-1]),
                            xytext=(-2, 5 if sym != "GDX" else -12),
                            textcoords="offset points", ha="right",
                            fontsize=8, color=col, fontweight="bold")
        ax.axhline(0, color=t["base"], lw=1, zorder=0)
        ax.set_xlim(0, 126)
        ax.set_xticks([0, 21, 42, 63, 84, 105, 126])
        ax.set_title("Median path after an up-break — all ETFs", loc="left", pad=8)
        ax.set_ylabel("% from event close", fontsize=8.5)

        # B: chase risk vs reward
        ax = axes[0, 1]
        _ax(ax, t, grid_axis="both")
        for cls, col in (("CHASER", t["blue"]), ("ROUND-TRIP", t["orange"])):
            xs = [(-v["mae_q25"], v["med63"], s) for s, v in up.items()
                  if v["cls"] == cls]
            if xs:
                ax.scatter([p[0] for p in xs], [p[1] for p in xs], s=40,
                           color=col, zorder=3, label=cls.title(),
                           edgecolors=t["surface"], linewidths=1.1)
                for x, y, s in xs:
                    ax.annotate(s, (x, y), xytext=(4, 3),
                                textcoords="offset points", fontsize=7,
                                color=t["dim"])
        ax.axhline(0, color=t["base"], lw=1, zorder=0)
        ax.set_title("Chase risk vs reward", loc="left", pad=8)
        ax.set_xlabel("q25 max drawdown, % (worse →)", fontsize=8.5)
        ax.set_ylabel("median fwd 63d, %", fontsize=8.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        # C: capitulation bars
        ax = axes[1, 0]
        _ax(ax, t, grid_axis="x")
        order = sorted(dn, key=lambda s: dn[s]["med63"])
        y = np.arange(len(order))
        ax.barh(y, [dn[s]["med63"] for s in order], height=0.62,
                color=t["aqua"], alpha=0.9)
        ax.set_yticks(y, order, fontsize=7)
        for i, s in enumerate(order):
            ax.text(dn[s]["med63"] + 0.12, i,
                    f"{dn[s]['med63']:+.1f} · {dn[s]['hit63']:.0f}%",
                    va="center", fontsize=6.5, color=t["dim"])
        ax.axvline(0, color=t["base"], lw=1)
        ax.set_xlim(0, max(dn[s]["med63"] for s in order) * 1.35)
        ax.set_title("Capitulation works everywhere", loc="left", pad=8)
        ax.set_xlabel("median fwd 63d after down-break, %", fontsize=8.5)

        # D: turn-signal bars
        ax = axes[1, 1]
        _ax(ax, t, grid_axis="x")
        tt = {s: v for s, v in turn.items() if v["n"] >= 8}
        order = sorted(tt, key=lambda s: tt[s]["med63"])
        y = np.arange(len(order))
        vals = [tt[s]["med63"] for s in order]
        ax.barh(y, vals, height=0.5,
                color=[t["blue"] if v > 0 else t["red"] for v in vals], alpha=0.9)
        ax.set_yticks(y, order, fontsize=8)
        for i, s in enumerate(order):
            v = tt[s]["med63"]
            ax.text(v + (0.3 if v > 0 else -0.3), i,
                    f"{v:+.1f} · {tt[s]['hit63']:.0f}% · n={int(tt[s]['n'])}",
                    va="center", ha="left" if v > 0 else "right",
                    fontsize=7, color=t["dim"])
        ax.axvline(0, color=t["base"], lw=1)
        lo, hi = min(vals + [0]), max(vals + [0])
        ax.set_xlim(lo - 8.5, hi + 7.0)
        ax.set_title("Turn signals: themes turn, financials trap", loc="left", pad=8)
        ax.set_xlabel("median own-price fwd 63d after turn, %", fontsize=8.5)
        return _b64(fig, t)


def fig_expmove(rows, t):
    """Implied vs conditional expected-move scatters, ~3M and ~6M."""
    rows = [r for r in rows if np.isfinite(r.get("imp3m", np.nan))]
    if len(rows) < 5:
        return None
    with plt.rc_context(_style(t)):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), facecolor=t["surface"])
        fig.subplots_adjust(wspace=0.26, left=0.07, right=0.98, top=0.9,
                            bottom=0.13)
        for ax, ik, ck, lbl in ((axes[0], "imp3m", "cond63", "~3M vs 63d"),
                                (axes[1], "imp6m", "cond126", "~6M vs 126d")):
            _ax(ax, t, grid_axis="both")
            pts = [(r[ck], r[ik], r["symbol"]) for r in rows
                   if np.isfinite(r.get(ik, np.nan))]
            lim = max(max(p[0] for p in pts), max(p[1] for p in pts)) * 1.12
            ax.plot([0, lim], [0, lim], color=t["base"], lw=1.2,
                    ls=(0, (4, 3)), zorder=1)
            ratio = sorted(pts, key=lambda p: p[1] / p[0] if p[0] else 0)
            label = ({p[2] for p in ratio[:3]} | {p[2] for p in ratio[-4:]} |
                     {"GDX", "QQQ", "SMH"})
            for x, y, s in pts:
                col = t["orange"] if s == "GDX" else t["blue"]
                ax.scatter(x, y, s=36, color=col, zorder=3,
                           edgecolors=t["surface"], linewidths=1.0)
                if s in label:
                    ax.annotate(s, (x, y), xytext=(4, 3),
                                textcoords="offset points", fontsize=7,
                                color=t["dim"])
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_title(f"Implied {lbl} conditional", loc="left", pad=8)
            ax.text(0.97, 0.05, "diagonal = fair\nabove = options rich",
                    transform=ax.transAxes, ha="right", fontsize=7,
                    color=t["faint"])
            ax.set_xlabel("conditional E|move| after up-break, %", fontsize=8.5)
            ax.set_ylabel("implied straddle move, %", fontsize=8.5)
        return _b64(fig, t)


def fig_skew(rows, t):
    """Risk-reversal bars + the positioning quadrant, ~3M tenor."""
    rows = [r for r in rows if np.isfinite(r.get("rr", np.nan))]
    if len(rows) < 5:
        return None
    with plt.rc_context(_style(t)):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), facecolor=t["surface"])
        fig.subplots_adjust(wspace=0.26, left=0.08, right=0.98, top=0.9,
                            bottom=0.13)
        ax = axes[0]
        _ax(ax, t, grid_axis="x")
        order = sorted(rows, key=lambda r: r["rr"])
        y = np.arange(len(order))
        vals = [r["rr"] for r in order]
        ax.barh(y, vals, height=0.6,
                color=[t["red"] if v > 0 else t["blue"] for v in vals], alpha=0.9)
        ax.set_yticks(y, [r["symbol"] for r in order], fontsize=7.5)
        for i, r in enumerate(order):
            v = r["rr"]
            ax.text(v + (0.2 if v > 0 else -0.2), i, f"{v:+.1f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=6.5, color=t["dim"])
        ax.axvline(0, color=t["base"], lw=1)
        ax.set_xlim(min(vals) - 2, max(vals) + 2)
        ax.set_title("Risk reversal at ±1 expected-move strikes", loc="left", pad=8)
        ax.set_xlabel("call wing − put wing, vol pts (red = upside bid)", fontsize=8.5)

        ax = axes[1]
        _ax(ax, t, grid_axis="both")
        for r in rows:
            col = t["orange"] if r["symbol"] == "GDX" else t["blue"]
            ax.scatter(r["rr"], r["upside"], s=40, color=col, zorder=3,
                       edgecolors=t["surface"], linewidths=1.1)
            ax.annotate(r["symbol"], (r["rr"], r["upside"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=t["dim"])
        ax.axvline(0, color=t["base"], lw=1)
        ax.set_title("The positioning quadrant", loc="left", pad=8)
        ax.text(0.97, 0.93, "positioned for upside →", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=t["faint"], style="italic")
        ax.set_xlabel("rr, vol points", fontsize=8.5)
        ax.set_ylabel("call OI share ≥ +0.5σ, %", fontsize=8.5)
        return _b64(fig, t)


def fig_skew_near(rows, t):
    """1M vs 2M risk-reversal term structure: paired bars per ETF (sorted
    by the 1M leg) and the steepening scatter."""
    rows = [r for r in rows if np.isfinite(r.get("rr1", np.nan))
            and np.isfinite(r.get("rr2", np.nan))]
    if len(rows) < 5:
        return None
    d1 = int(np.median([r["dte1"] for r in rows]))
    d2 = int(np.median([r["dte2"] for r in rows]))
    with plt.rc_context(_style(t)):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), facecolor=t["surface"])
        fig.subplots_adjust(wspace=0.26, left=0.08, right=0.98, top=0.9,
                            bottom=0.13)
        ax = axes[0]
        _ax(ax, t, grid_axis="x")
        order = sorted(rows, key=lambda r: r["rr1"])
        y = np.arange(len(order))
        h = 0.36
        for off, key, col, col_pos, lbl in (
                (+h / 2 + 0.02, "rr1", t["blue"], t["red"], f"~1M ({d1}d)"),
                (-h / 2 - 0.02, "rr2", t["blue2"], t["red2"], f"~2M ({d2}d)")):
            vals = [r[key] for r in order]
            ax.barh(y + off, vals, height=h,
                    color=[col_pos if v > 0 else col for v in vals],
                    alpha=0.92, label=lbl)
        ax.set_yticks(y, [r["symbol"] for r in order], fontsize=7.5)
        ax.axvline(0, color=t["base"], lw=1)
        ax.set_title("Risk reversal term: ~1M vs ~2M", loc="left", pad=8)
        ax.set_xlabel("call wing − put wing at ±1σ-move, vol pts "
                      "(warm = upside bid)", fontsize=8.5)
        ax.legend(frameon=False, fontsize=8, loc="lower right")

        ax = axes[1]
        _ax(ax, t, grid_axis="both")
        vals = [v for r in rows for v in (r["rr1"], r["rr2"])]
        lo, hi = min(vals) - 1.5, max(vals) + 1.5
        ax.plot([lo, hi], [lo, hi], color=t["base"], lw=1.2, ls=(0, (4, 3)),
                zorder=1)
        for r in rows:
            col = t["orange"] if r["symbol"] == "GDX" else t["blue"]
            ax.scatter(r["rr1"], r["rr2"], s=40, color=col, zorder=3,
                       edgecolors=t["surface"], linewidths=1.1)
            ax.annotate(r["symbol"], (r["rr1"], r["rr2"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=t["dim"])
        ax.axvline(0, color=t["grid"], lw=1)
        ax.axhline(0, color=t["grid"], lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title("Near-dated vs deferred skew", loc="left", pad=8)
        ax.text(0.03, 0.94, "above diagonal = skew eases into the front\n"
                "below = front-loaded (event risk priced near-dated)",
                transform=ax.transAxes, fontsize=7, color=t["faint"])
        ax.set_xlabel(f"rr ~1M ({d1}d), vol pts", fontsize=8.5)
        ax.set_ylabel(f"rr ~2M ({d2}d), vol pts", fontsize=8.5)
        return _b64(fig, t)


def render_all(paths, exp_rows, skew_rows, near_rows=None):
    """{key: {"light": b64, "dark": b64}} for every figure whose input
    exists."""
    out = {}
    for key, fn, data in (("paths", fig_paths, paths),
                          ("expmove", fig_expmove, exp_rows),
                          ("skew", fig_skew, skew_rows),
                          ("skewnear", fig_skew_near, near_rows or [])):
        imgs = {}
        for theme, tokens in THEMES.items():
            img = fn(data, tokens)
            if img:
                imgs[theme] = img
        if imgs:
            out[key] = imgs
    return out
