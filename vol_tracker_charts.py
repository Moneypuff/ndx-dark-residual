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
  * skewterm -- the 25-delta risk-reversal term structure in one view: a
               tenor x ETF heatmap (~1M/2M/3M/6M, annualized vol points,
               wing strikes solved from each tenor's smile at +/-25 delta)
               plus the ~3M upside-call-positioning quadrant. Liquid only.

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


SKEW_TENORS = (30, 60, 91, 182)


def fig_skew_term(rows, t):
    """The whole skew term structure in one view: a tenor x ETF heatmap of
    the 25-delta risk reversal (25d-call IV minus 25d-put IV, wing strikes
    solved from each tenor's smile; IV differences, so the units are
    ANNUALIZED vol points and every cell measures identical deltas) plus
    the ~3M positioning quadrant."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    rows = [r for r in rows
            if any(np.isfinite(r.get(f"rr{d}", np.nan)) for d in SKEW_TENORS)]
    if len(rows) < 5:
        return None
    dtes = {d: int(np.nanmedian([r.get(f"dte{d}", np.nan) for r in rows]))
            for d in SKEW_TENORS}
    labels = {30: "~1M", 60: "~2M", 91: "~3M", 182: "~6M"}
    rows = sorted(rows, key=lambda r: -np.nanmean(
        [r.get(f"rr{d}", np.nan) for d in SKEW_TENORS]))
    M = np.array([[r.get(f"rr{d}", np.nan) for d in SKEW_TENORS] for r in rows])

    with plt.rc_context(_style(t)):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.1), facecolor=t["surface"],
                                 gridspec_kw={"width_ratios": [1.05, 1]})
        fig.subplots_adjust(wspace=0.24, left=0.07, right=0.98, top=0.9,
                            bottom=0.12)

        # A: tenor x ETF heatmap, diverging blue (put skew) <-> red (upside bid)
        ax = axes[0]
        ax.set_facecolor(t["surface"])
        cmap = LinearSegmentedColormap.from_list(
            "rr", [t["blue"], t["surface"], t["red"]])
        vmax = max(np.nanmax(np.abs(M)), 1.0)
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax),
                       aspect="auto")
        ax.set_xticks(range(len(SKEW_TENORS)),
                      [f"{labels[d]}\n({dtes[d]}d)" for d in SKEW_TENORS],
                      fontsize=8)
        ax.set_yticks(range(len(rows)), [r["symbol"] for r in rows], fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    strong = abs(v) > vmax * 0.55
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            fontsize=7.5,
                            color=t["surface"] if strong else t["ink"])
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7.5,
                            color=t["faint"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title("25Δ risk reversal across the term", loc="left", pad=8)
        ax.text(0, -0.12, "annualized vol pts · 25Δ call IV − 25Δ put IV, wings "
                "solved from each tenor's smile · red = upside bid, blue = put skew",
                transform=ax.transAxes, fontsize=7.5, color=t["faint"])

        # B: positioning quadrant at ~3M
        ax = axes[1]
        _ax(ax, t, grid_axis="both")
        for r in rows:
            rr, up = r.get("rr91", np.nan), r.get("upside", np.nan)
            if not (np.isfinite(rr) and np.isfinite(up)):
                continue
            col = t["orange"] if r["symbol"] == "GDX" else t["blue"]
            ax.scatter(rr, up, s=40, color=col, zorder=3,
                       edgecolors=t["surface"], linewidths=1.1)
            ax.annotate(r["symbol"], (rr, up), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=t["dim"])
        ax.axvline(0, color=t["base"], lw=1)
        ax.set_title("The positioning quadrant (~3M)", loc="left", pad=8)
        ax.text(0.97, 0.93, "positioned for upside →", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=t["faint"], style="italic")
        ax.set_xlabel("25Δ rr ~3M, annualized vol points", fontsize=8.5)
        ax.set_ylabel("call OI share ≥ +0.5σ, %", fontsize=8.5)
        return _b64(fig, t)


def render_all(paths, exp_rows, skew_rows):
    """{key: {"light": b64, "dark": b64}} for every figure whose input
    exists."""
    out = {}
    for key, fn, data in (("paths", fig_paths, paths),
                          ("expmove", fig_expmove, exp_rows),
                          ("skewterm", fig_skew_term, skew_rows)):
        imgs = {}
        for theme, tokens in THEMES.items():
            img = fn(data, tokens)
            if img:
                imgs[theme] = img
        if imgs:
            out[key] = imgs
    return out
