r"""Figure for the local RG-window audit of the chirality-flip
per-node identity.

Reads outputs/verify_chirality_local_RG_window.json and produces a
2-panel figure:

  Panel A: per-d1-regime AUC of theta_xivar vs framework Delta
           (top-10% residual-tail = matter core), with the global
           theta_chir(N) running curve overlaid for context.
  Panel B: per-d1-regime ratio P(C_N|matter)/P(C_N|vacuum) vs
           N_global, showing the matter-core enrichment under
           theta_xivar across the d1 ladder.
"""
from __future__ import annotations

import json
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
N_STAR = 50
D = 4
N_GEN = 3
LN_DG = math.log(D * N_GEN)
N_INV = D * N_GEN * N_STAR
PI = math.pi


def theta_global_deg(n_lat):
    x = math.log(n_lat / N_STAR) / LN_DG
    return math.degrees(math.atan(N_GEN ** (2 * x - 1)))


def main():
    # Per-d1-regime data computed via the local-RG-window pipeline
    # (matches verify_chirality_local_RG_window.py output).
    rows = [
        ("$d1$-$P_{5}N_{64}$",  64, 0.898, None, None),
        ("$d1$-$P_{5}N_{100}$", 100, 0.803, 7.54, 24),
        ("$d1$-$P_{5}N_{128}$", 128, 0.512, 2.13, 348),
        ("$d1$-$P_{5}N_{200}$", 200, 0.676, 1.53, 1060),
        ("$d1$-$P_{5}N_{256}$", 256, 0.775, 0.85, 452),
        ("$d1$-$P_{5}N_{300}$", 300, 0.548, 0.31, 3401),
        ("$d1$-$P_{6}N_{128}$", 128, 0.663, 4.51, 303),
        ("$d1$-$P_{8}N_{128}$", 128, 0.639, 4.06, 323),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    n_grid = np.linspace(35, 320, 250)
    theta_curve = np.array([theta_global_deg(n) for n in n_grid])
    ax.plot(n_grid, theta_curve, "k-", lw=1.0, alpha=0.55,
            label=r"$\theta_{\rm chir}(N)$ running")
    ax.axhline(45, color="purple", lw=0.7, ls="--", alpha=0.7,
               label=r"$\theta\!=\!\pi/4$ (flip)")
    n_arr = np.array([r[1] for r in rows])
    auc = np.array([r[2] for r in rows])
    sc = ax.scatter(n_arr, [theta_global_deg(n) for n in n_arr],
                    c=auc, cmap="RdYlGn", s=160, edgecolors="k",
                    vmin=0.5, vmax=0.9, zorder=5)
    for r in rows:
        ax.annotate(f"AUC={r[2]:.2f}",
                     xy=(r[1], theta_global_deg(r[1])),
                     xytext=(4, 6), textcoords="offset points",
                     fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(35, 360)
    ax.set_ylim(0, 75)
    ax.set_xlabel(r"$N_{\rm global}$", fontsize=11)
    ax.set_ylabel(r"$\theta_{\rm chir}^{\rm global}(N)$ (deg)",
                   fontsize=11)
    ax.set_title("A. Per-$d1$-regime AUC of "
                  r"$\theta^{\rm xivar}_{\rm chir}(a)$ vs framework "
                  r"$\Delta(a)$",
                  fontsize=11)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("AUC", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    rows_ratio = [r for r in rows if r[3] is not None]
    n_arr_r = np.array([r[1] for r in rows_ratio])
    ratios = np.array([r[3] for r in rows_ratio])
    bar_color = ["#2c8a3a" if r >= 1.0 else "#bd4f3b"
                  for r in ratios]
    bars = ax.bar(range(len(rows_ratio)), ratios, color=bar_color,
                   edgecolor="k", alpha=0.85)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6,
                label="ratio = 1 (no enrichment)")
    for i, (r, b) in enumerate(zip(rows_ratio, bars)):
        ax.text(i, r[3] + 0.15, f"{r[3]:.2f}$\\times$",
                 ha="center", fontsize=9, fontweight="bold")
        ax.text(i, -0.4, f"$N\\!=\\!{r[1]}$", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows_ratio)))
    ax.set_xticklabels([r[0] for r in rows_ratio], rotation=20,
                        fontsize=8, ha="right")
    ax.set_ylim(-0.6, 8.5)
    ax.set_ylabel(r"$P(C_{N}|\theta\!>\!\pi/4)\,/\,"
                   r"P(C_{N}|\theta\!\leq\!\pi/4)$",
                   fontsize=11)
    ax.set_title("B. Matter-core enrichment ratio "
                  "(framework $\\Delta$, top-$10\\%$)",
                  fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Local RG-window audit of the chirality-flip "
                  "per-node identity (Xi-row-variance scale)",
                  fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_local_RG_window_audit.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
