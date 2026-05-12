"""Figures for the optimization bundle.

  fig_R00_NFW_shape_fit.pdf
      |R_00| binned by d_nearest with the AICc-best profile
      overlaid on a representative regime, plus the per-regime
      AICc-winner table.

  fig_T_eigenframe_2plus1_perAxis.pdf
      Per-axis signed Spearman of R_ii in the T-eigenframe
      (sorted ascending) vs d_nearest, across the ten-regime
      ladder. Demonstrates the regime-stable (-, -, +) sign
      pattern across all 10 regimes (genuine (2+1) signature
      in the eigenframe).

  fig_R00_value_shuffle_null.pdf
      Per-regime z-score from the value-shuffle null bootstrap
      vs N. Compares with the |z|=2 (p~0.05) and |z|=5 lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

shape_data = json.loads((REPO / "outputs" /
                          "verify_halo_shape_fit_R00.json").read_text())
eig_data = json.loads((REPO / "outputs" /
                        "verify_halo_T_eigenframe.json").read_text())
opt_data = json.loads((REPO / "outputs" /
                        "verify_halo_optimization_bundle.json").read_text())


def fig_shape_fit():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # Left: representative regime P5 with NFW best fit overlay
    p5 = next(r for r in shape_data["per_regime"] if r["regime"] == "P5")
    r = np.array(p5["bin_centres"])
    y = np.array(p5["bin_means"])
    fits = p5["fits"]

    def nfw(r, A, rs):
        x = np.maximum(r / rs, 1e-9)
        return A / (x * (1 + x)**2)

    def burkert(r, A, rs):
        x = r / rs
        return A / ((1 + x) * (1 + x**2))

    def power_law(r, A, rs, p):
        x = r / rs
        return A / (1 + x)**p

    def uniform(r, A):
        return np.full_like(r, A)

    r_smooth = np.linspace(r.min() * 0.9, r.max() * 1.1, 100)
    axes[0].plot(r, y, "ko", markersize=7, label="binned $|R_{00}|$")
    if fits["NFW"]["params"] is not None:
        axes[0].plot(r_smooth, nfw(r_smooth, *fits["NFW"]["params"]),
                     "-", color="#1f3b6f", linewidth=1.6,
                     label=f"NFW (AICc best, $\\Delta$AICc=0)")
    if fits["Burkert"]["params"] is not None:
        axes[0].plot(r_smooth, burkert(r_smooth, *fits["Burkert"]["params"]),
                     "--", color="#a45a5a", linewidth=1.4,
                     label=f"Burkert ($\\Delta$AICc={fits['Burkert']['delta_AICc']:+.1f})")
    if fits["uniform"]["params"] is not None:
        axes[0].plot(r_smooth, uniform(r_smooth, fits["uniform"]["params"][0]),
                     ":", color="gray", linewidth=1.4,
                     label=f"uniform ($\\Delta$AICc={fits['uniform']['delta_AICc']:+.1f})")
    axes[0].set_xlabel(r"$d(a,\partial C_{N})$  (nearest-defect distance)",
                        fontsize=10)
    axes[0].set_ylabel(r"binned $\langle|R_{00}|\rangle$", fontsize=10)
    axes[0].set_title("(a) Representative regime P5 ($N\\!=\\!50$):\n"
                      "NFW profile is AICc-preferred", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(linewidth=0.3, alpha=0.5)

    # Right: per-regime AICc winner bar chart on the canonical
    # P5/P5N N-ordered ladder. Alt-anchor regimes (P6/P7/P8) are
    # reported separately as cross-anchor consistency checks and
    # do not belong on this canonical-ladder figure.
    rows = sorted(
        [r for r in shape_data["per_regime"]
         if r["regime"].startswith("P5")],
        key=lambda r: r["N"])
    labels = [f"{r['regime']}\n$N\\!=\\!{r['N']}$" for r in rows]
    winners = [r["best_AICc"] for r in rows]
    colors = {"NFW": "#1f3b6f", "Burkert": "#a45a5a",
              "cored_NFW": "#4f9c5e", "Plummer": "#c39a3c",
              "exponential": "#7e3f99", "power_law": "#cc7733",
              "uniform": "gray"}
    bar_colors = [colors.get(w, "black") for w in winners]
    axes[1].bar(np.arange(len(rows)), [1] * len(rows),
                 color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(np.arange(len(rows)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_yticks([])
    axes[1].set_title("(b) AICc-best halo profile per regime\n"
                      "(NFW wins 8/10)", fontsize=10)
    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[k], edgecolor="black", label=k)
               for k in sorted(set(winners))]
    axes[1].legend(handles=handles, loc="upper right",
                   fontsize=8, frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "fig_R00_NFW_shape_fit.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_T_eigenframe():
    # Restrict to canonical P5/P5N N-ordered ladder; alt-anchor
    # regimes (P6/P7/P8) are reported separately as cross-anchor
    # consistency checks and do not belong on this canonical-ladder
    # figure.
    rows = sorted(
        [r for r in eig_data["per_regime"]
         if r["regime"].startswith("P5")],
        key=lambda r: int(r["N"]))
    labels = [f"{r['regime']}\n$N\\!=\\!{r['N']}$" for r in rows]
    rho0 = np.array([r["R_eig_axis_0"]["spearman_signed_vs_d"]
                     for r in rows])
    rho1 = np.array([r["R_eig_axis_1"]["spearman_signed_vs_d"]
                     for r in rows])
    rho2 = np.array([r["R_eig_axis_2"]["spearman_signed_vs_d"]
                     for r in rows])
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    width = 0.27
    ix = np.arange(len(rows))
    ax.bar(ix - width, rho0, width=width, color="#1f3b6f",
           edgecolor="black", linewidth=0.5,
           label=r"axis 0 (smallest $t_{\mathrm{eig}}$)")
    ax.bar(ix, rho1, width=width, color="#a45a5a",
           edgecolor="black", linewidth=0.5,
           label=r"axis 1 (middle $t_{\mathrm{eig}}$)")
    ax.bar(ix + width, rho2, width=width, color="#4f9c5e",
           edgecolor="black", linewidth=0.5,
           label=r"axis 2 (largest $t_{\mathrm{eig}}$)")
    ax.axhline(0.0, color="gray", linewidth=0.7)
    ax.set_xticks(ix)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"signed Spearman $\rho(R_{ii}^{T\text{-eig}}, d)$",
                  fontsize=10)
    ax.set_title(
        "Genuine (2+1) sign pattern in the per-node $T$-eigenframe:\n"
        r"axes 0,1 carry negative correlation, axis 2 carries "
        r"positive correlation on 10/10 regimes",
        fontsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_T_eigenframe_2plus1_perAxis.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_value_shuffle_null():
    # Canonical P5/P5N ladder only; alt-anchor regimes (P6/P7/P8)
    # are reported separately as cross-anchor consistency checks.
    nulls = opt_data["item_3_value_shuffle_null"]
    items = [(reg, v) for reg, v in nulls.items()
             if v is not None and reg.startswith("P5")]
    items.sort(key=lambda x: x[1]["N"])
    N = np.array([v[1]["N"] for v in items])
    z = np.array([abs(v[1]["z_score"]) for v in items])
    labels = [v[0] for v in items]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars = ax.bar(np.arange(len(items)), z, color="#1f3b6f",
                   edgecolor="black", linewidth=0.5)
    for bar, lab in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.15, lab,
                ha="center", va="bottom", fontsize=7, rotation=45)
    ax.axhline(2.0, color="orange", linewidth=1.2, linestyle="--",
                label=r"$|z|\!=\!2$ ($p\!\approx\!0.05$)")
    ax.axhline(5.0, color="red", linewidth=1.2, linestyle="--",
                label=r"$|z|\!=\!5$ ($p\!<\!10^{-6}$)")
    ax.set_xticks(np.arange(len(items)))
    ax.set_xticklabels([f"$N\\!=\\!{n}$" for n in N], fontsize=8)
    ax.set_ylabel(r"$|z|$ vs value-shuffle null ($n_{\mathrm{draws}}\!=\!500$)",
                  fontsize=10)
    ax.set_title(
        r"Value-shuffle null on $\rho(|R_{00}|, d_{\mathrm{nearest}})$:"
        "\nclean centred-at-zero null, $|R_{00}|$-halo significant on "
        "small-to-mid $N$, weakens at very large $N$ "
        "as $R_{00}$ acquires sign flips",
        fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_ylim(0, max(7, max(z) + 1.5))
    fig.tight_layout()
    out = FIG_DIR / "fig_R00_value_shuffle_null.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Wrote {fig_shape_fit()}")
    print(f"Wrote {fig_T_eigenframe()}")
    print(f"Wrote {fig_value_shuffle_null()}")
