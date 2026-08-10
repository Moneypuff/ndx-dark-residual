#!/usr/bin/env python3
"""
Visualization for ETF expected-move study.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import ndx_dark_residual as N
from build_regime_log import (SECTORS, THEMES, conviction_frame,
                              conviction_events, detect_breaks, spread_frames)
from etf_expected_move_study import (
    HORIZONS, MIN_N, fwd_returns, uncond_returns, dist_stats, fair_premia,
    fetch_chains
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=N.DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--output", default="etf_expected_move_study.png",
                    help="output PNG file path")
    args = ap.parse_args()

    univ = {**SECTORS, **THEMES}
    syms = sorted(set(univ) | {"SPY"})
    panels = N.load_yahoo_panels(syms, "2004-01-01", pd.Timestamp.today().normalize(),
                                 cache_dir=args.cache_dir or None,
                                 refresh=args.refresh, label="EXPMOVE")
    adj = panels["adjclose"].dropna(how="all")
    volp = panels["volume"]
    spy = adj["SPY"].dropna()

    rows, live = [], []
    for t in sorted(univ):
        if t not in adj:
            continue
        c = adj[t].dropna()
        conv = conviction_frame(c, volp[t])
        ev = conviction_events(c, conv, min_score=3, cooldown=21, horizon=63)
        s21, s63, z, _ = spread_frames(c, spy)
        fams = {"up": list(ev[ev["dir"] == 1]["date"]),
                "dn": list(ev[ev["dir"] == -1]["date"]),
                "turn": [d for d, s in detect_breaks(s63, z) if s == -1]}
        base = {h: dist_stats(uncond_returns(c, h)) for h in HORIZONS}
        for fam, dates in fams.items():
            for h in HORIZONS:
                r = fwd_returns(c, dates, h)
                if len(r) < MIN_N:
                    continue
                st = dist_stats(r)
                st.update(fair_premia(r))
                st.update(etf=t, family=fam, horizon=h,
                          base_eabs=base[h]["eabs"],
                          ratio=st["eabs"] / base[h]["eabs"])
                rows.append(st)

    R = pd.DataFrame(rows).set_index(["etf", "family", "horizon"])

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Move ratio comparison (conditional vs unconditional)
    ax1 = fig.add_subplot(gs[0, :2])
    for fam, color in [("up", "green"), ("dn", "red"), ("turn", "blue")]:
        try:
            F = R.xs(fam, level="family")
            data = F.groupby("etf")["ratio"].mean().sort_values(ascending=False)
            ax1.barh(range(len(data)), data.values, alpha=0.7, label=fam)
        except KeyError:
            pass
    ax1.set_yticks(range(len(data)))
    ax1.set_yticklabels(data.index)
    ax1.set_xlabel("Move Ratio (Conditional / Unconditional E|r|)")
    ax1.set_title("Expected Move Amplification by ETF Family")
    ax1.legend()
    ax1.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
    ax1.grid(axis="x", alpha=0.3)

    # 2. Straddle fair values by family & horizon
    ax2 = fig.add_subplot(gs[0, 2])
    straddle_by_fam = {}
    for fam in ["up", "dn", "turn"]:
        try:
            F = R.xs(fam, level="family")
            straddle_by_fam[fam] = F["straddle"].values
        except KeyError:
            pass
    if straddle_by_fam:
        bp = ax2.boxplot(straddle_by_fam.values())
        ax2.set_xticklabels(list(straddle_by_fam.keys()))
    ax2.set_ylabel("Fair Straddle Value (% of spot)")
    ax2.set_title("Straddle Premium Distribution")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Distribution of mean returns by family
    ax3 = fig.add_subplot(gs[1, 0])
    for fam, color in [("up", "green"), ("dn", "red"), ("turn", "blue")]:
        try:
            F = R.xs(fam, level="family")
            ax3.scatter(F["base_eabs"], F["eabs"], alpha=0.6, label=fam, s=100)
        except KeyError:
            pass
    ax3.set_xlabel("Unconditional E|r| (%)")
    ax3.set_ylabel("Conditional E|r| (%)")
    ax3.set_title("Move Size: Conditional vs Unconditional")
    ax3.legend()
    ax3.plot([0, R["base_eabs"].max() * 1.1], [0, R["base_eabs"].max() * 1.1],
             "k--", alpha=0.3)
    ax3.grid(alpha=0.3)

    # 4. Win rate (P>0) by family
    ax4 = fig.add_subplot(gs[1, 1])
    pos_prob_by_fam = []
    labels_fam = []
    for fam in ["up", "dn", "turn"]:
        try:
            F = R.xs(fam, level="family")
            pos_prob_by_fam.append(F["pos"].values)
            labels_fam.append(fam)
        except KeyError:
            pass
    if pos_prob_by_fam:
        bp = ax4.boxplot(pos_prob_by_fam)
        ax4.set_xticklabels(labels_fam)
    ax4.set_ylabel("P(return > 0) %")
    ax4.set_title("Win Rate by Family")
    ax4.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax4.grid(axis="y", alpha=0.3)

    # 5. Range metrics (Q10-Q90)
    ax5 = fig.add_subplot(gs[1, 2])
    for fam, color in [("up", "green"), ("dn", "red"), ("turn", "blue")]:
        try:
            F = R.xs(fam, level="family")
            F_h63 = F.xs(63, level="horizon")
            ranges = F_h63["q90"] - F_h63["q10"]
            ax5.scatter([fam] * len(ranges), ranges, alpha=0.6, s=80, color=color)
        except KeyError:
            pass
    ax5.set_ylabel("Q90 - Q10 Range (%)")
    ax5.set_title("Return Spread (63-session horizon)")
    ax5.grid(axis="y", alpha=0.3)

    # 6. Option payoffs by strike for UP-BREAKS
    ax6 = fig.add_subplot(gs[2, :])
    try:
        F_up = R.xs("up", level="family")
        F_up_h63 = F_up.xs(63, level="horizon")
        x_pos = np.arange(len(F_up_h63))
        width = 0.25
        ax6.bar(x_pos - width, F_up_h63["call"].values, width, label="ATM Call", alpha=0.8)
        ax6.bar(x_pos, F_up_h63["call5"].values, width, label="Call +5%", alpha=0.8)
        ax6.bar(x_pos + width, F_up_h63["straddle"].values, width, label="ATM Straddle", alpha=0.8)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(F_up_h63.index, rotation=45, ha="right")
        ax6.set_ylabel("Fair Value (% of spot)")
        ax6.set_title("UP-BREAKS: Fair Option Premiums (63-session horizon)")
        ax6.legend()
        ax6.grid(axis="y", alpha=0.3)
    except KeyError:
        ax6.text(0.5, 0.5, "No UP-BREAKS data", ha="center", va="center",
                 transform=ax6.transAxes)

    # 7. Loss odds for different strikes
    ax7 = fig.add_subplot(gs[3, 0])
    for fam, color in [("up", "green"), ("dn", "red"), ("turn", "blue")]:
        try:
            F = R.xs(fam, level="family")
            F_h63 = F.xs(63, level="horizon")
            data = F_h63.groupby("etf")[["pbelow5", "pbelow10"]].mean()
            ax7.scatter(data["pbelow5"], data["pbelow10"], alpha=0.6, s=100, label=fam, color=color)
        except KeyError:
            pass
    ax7.set_xlabel("P(r < -5%) %")
    ax7.set_ylabel("P(r < -10%) %")
    ax7.set_title("Downside Loss Probability")
    ax7.legend()
    ax7.grid(alpha=0.3)

    # 8. Ratio distribution by horizon
    ax8 = fig.add_subplot(gs[3, 1])
    for h, color in [(63, "lightblue"), (126, "darkblue")]:
        try:
            ratios = R.xs(h, level="horizon")["ratio"]
            ax8.hist(ratios.values, bins=15, alpha=0.6, label=f"h={h}", color=color)
        except KeyError:
            pass
    ax8.set_xlabel("Move Ratio")
    ax8.set_ylabel("Frequency")
    ax8.set_title("Distribution of Move Ratios")
    ax8.legend()
    ax8.grid(axis="y", alpha=0.3)

    # 9. Sample size distribution
    ax9 = fig.add_subplot(gs[3, 2])
    sample_sizes = []
    sample_labels = []
    for fam in ["up", "dn", "turn"]:
        try:
            F = R.xs(fam, level="family")
            sample_sizes.append(F["n"].values)
            sample_labels.append(fam)
        except KeyError:
            pass
    if sample_sizes:
        bp = ax9.boxplot(sample_sizes)
        ax9.set_xticklabels(sample_labels)
    ax9.set_ylabel("Sample Size (n)")
    ax9.set_title("Signal Frequency by Family")
    ax9.grid(axis="y", alpha=0.3)

    plt.suptitle("ETF Expected-Move Study: Options-Pricing Lens on Post-Signal Path",
                 fontsize=16, fontweight="bold", y=0.995)

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
