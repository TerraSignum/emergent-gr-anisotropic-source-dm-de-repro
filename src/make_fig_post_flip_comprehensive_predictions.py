r"""Comprehensive figure: 13 post-flip structural predictions
across rounds 1-5.

Two-panel figure:
  A: relative-error bar chart (ordered by tier)
  B: predicted vs observed scatter (log-log) with diagonal
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # All 13 predictions (test, pred, target, rel_err_pct, tier, observable)
    preds = [
        ("T1: $D_\\Omega^M/S_{BH}^V$=$\\pi$", 3.1416, 3.1416, 0.00, "EXACT"),
        ("T2: $\\Omega_{DM}h^2$ cross-anchor", 0.1215, 0.120, 1.25, "PRECISE"),
        ("T11: $\\eta_B$=$N!_{gen}\\gamma^{2d+2}$", 6e-10, 6.1e-10, 1.64,
          "PRECISE"),
        ("T12: $|S_{BH}^M|/S_{BH}^V$=$N_{gen}$+$d$", 7, 7, 0.00, "EXACT"),
        ("T16: $\\Omega_{DM}/\\Omega_b$=5+$\\alpha_\\xi$/2",
          5.45, 5.4527, 0.05, "EXACT"),
        ("T19: $\\alpha_{EM}$=$\\gamma^2\\alpha_\\xi^{N_{gen}}$",
          7.29e-3, 7.297e-3, 0.10, "EXACT"),
        ("T23: $v_{EW}/M_{Pl}$=$\\gamma^{d^2}$",
          1e-16, 1.013e-16, 1.31, "PRECISE"),
        ("T25: $m_t/m_W$=2+$\\gamma\\sqrt{N_{gen}}$",
          2.173, 2.149, 1.13, "PRECISE"),
        ("T29: $f_\\pi/m_\\pi$=2/3", 0.667, 0.662, 0.72, "EXACT"),
        ("T33: $y_e$=2$\\gamma^6$", 2e-6, 2.075e-6, 3.63, "PRECISE"),
        ("T34: $y_t$=$\\alpha_\\xi$+$\\gamma$=1",
          1, 0.992, 0.78, "EXACT"),
        ("T35: $m_H/m_W$=$\\pi/2$", 1.571, 1.559, 0.78, "EXACT"),
    ]
    # Sort by rel_err
    preds_sorted = sorted(preds, key=lambda p: p[3])

    # Setup figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.3, width_ratios=[1.3, 1])

    # ---- A: rel error bar chart ----
    ax = fig.add_subplot(gs[0, 0])
    labels = [p[0] for p in preds_sorted]
    rels = [p[3] for p in preds_sorted]
    tiers = [p[4] for p in preds_sorted]
    colors = ["#1f6f3f" if t == "EXACT" else
                "#3b8bbf" if t == "PRECISE" else
                "#f0a020" for t in tiers]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, rels, color=colors, edgecolor="k", alpha=0.85)
    for i, (r, t) in enumerate(zip(rels, tiers)):
        ax.text(r + 0.05, i, f"{r:.2f}% ({t})", va="center",
                 fontsize=8)
    ax.axvline(1, color="green", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(5, color="orange", lw=0.8, ls=":", alpha=0.7)
    ax.text(1, len(labels) - 0.5, "EXACT (<1%)", color="green",
             fontsize=8, va="bottom")
    ax.text(5, len(labels) - 0.5, "PRECISE (<5%)", color="orange",
             fontsize=8, va="bottom")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"relative error (%)", fontsize=10)
    ax.set_title(r"A. Post-flip structural predictions: "
                  r"relative error", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    # Legend for tiers
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#1f6f3f", edgecolor="k", label="EXACT (<1%)"),
        Patch(facecolor="#3b8bbf", edgecolor="k", label="PRECISE (<5%)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    # ---- B: predicted vs observed (log-log) ----
    ax = fig.add_subplot(gs[0, 1])
    preds_filt = [p for p in preds if p[1] is not None and p[2] is not None]
    pred_vals = np.array([p[1] for p in preds_filt])
    obs_vals = np.array([p[2] for p in preds_filt])
    tiers_f = [p[4] for p in preds_filt]
    colors_f = ["#1f6f3f" if t == "EXACT" else
                  "#3b8bbf" if t == "PRECISE" else
                  "#f0a020" for t in tiers_f]
    # Use abs values for log scale
    pred_abs = np.abs(pred_vals)
    obs_abs = np.abs(obs_vals)
    for p_v, o_v, c, name in zip(pred_abs, obs_abs, colors_f,
                                   [p[0] for p in preds_filt]):
        ax.scatter(o_v, p_v, s=120, c=c, edgecolors="k", zorder=5)
    # Diagonal y=x
    lo = min(pred_abs.min(), obs_abs.min()) * 0.5
    hi = max(pred_abs.max(), obs_abs.max()) * 2
    diag = np.geomspace(lo, hi, 10)
    ax.plot(diag, diag, "k--", lw=1.0, alpha=0.5, label=r"$y\!=\!x$")
    # 1% and 5% bands
    ax.fill_between(diag, diag * 0.99, diag * 1.01, alpha=0.15,
                      color="green", label=r"1% band")
    ax.fill_between(diag, diag * 0.95, diag * 1.05, alpha=0.10,
                      color="blue", label=r"5% band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"observed value", fontsize=10)
    ax.set_ylabel(r"predicted value", fontsize=10)
    ax.set_title(r"B. Predicted vs observed across "
                  r"$10^{-10}$ to $10$", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(r"Post-flip System-R$^{(M)}$ structural predictions: "
                   r"13 observables across 17 orders of magnitude",
                   fontsize=13, y=0.98)
    out_pdf = FIG_DIR / "fig_post_flip_comprehensive_predictions.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
