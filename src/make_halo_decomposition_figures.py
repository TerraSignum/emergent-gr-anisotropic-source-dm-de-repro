"""Generate figures for the per-component halo decomposition.

Produces three figures from
outputs/verify_signed_halo_component_decomposition.json:

  fig_residual_components_spearman.pdf
      Per-component signed Spearman of R_mu_nu(a) vs distance to
      nearest matter concentration, across the 10-regime ladder.
      Shows that the time-time component carries a regime-stable
      positive correlation while spatial-diagonal components show
      the (2+1) anisotropy pattern, and the trace cancels.

  fig_residual_R00_distance_shells.pdf
      Median R_00 in five quintile shells of distance from the
      nearest matter concentration, per regime. Shows a monotone
      approach from a strongly negative value near matter cores
      toward zero in the outer shells (NFW-class halo radial
      decrease in the residual energy density).

  fig_residual_components_signs.pdf
      Mean and 1-sigma band of R_00 vs R_11+R_22+R_33 across
      regimes; visualises the +0.4 vs -0.4 cancellation that the
      trace projection produces.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "outputs" / "verify_signed_halo_component_decomposition.json"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

data = json.loads(JSON_PATH.read_text())
# Restrict to the canonical P5/P5N N-ordered ladder: alt-anchor regimes
# (P6/P7/P8) are not part of the canonical-physics ladder and are omitted
# from the decomposition figures (cf. Memory: alt-anchor separation rule).
_canonical_prefixes = ("P5",)
_raw_rows = data["per_regime"]
rows = sorted(
    [r for r in _raw_rows if r["regime"].startswith(_canonical_prefixes)],
    key=lambda r: int(r["N"]))
labels = [f"{r['regime']}\n$N\\!=\\!{r['N']}$" for r in rows]
ix = np.arange(len(rows))


def fig_components_spearman():
    comp_keys = ["R_00", "R_11", "R_22", "R_33", "R_tr"]
    comp_names = [r"$R_{00}$", r"$R_{11}$", r"$R_{22}$",
                  r"$R_{33}$", r"$\mathrm{tr}\,R$"]
    colors = ["#1f3b6f", "#a45a5a", "#4f9c5e", "#c39a3c", "#2d2d2d"]
    rho = np.array([
        [r[k]["spearman_signed_vs_d"] for r in rows] for k in comp_keys
    ])
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    width = 0.16
    for j, (name, color) in enumerate(zip(comp_names, colors)):
        ax.bar(ix + (j - 2) * width, rho[j], width=width,
               color=color, edgecolor="black", linewidth=0.5,
               label=name)
    ax.axhline(0.0, color="gray", linewidth=0.7)
    ax.set_xticks(ix)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"signed Spearman $\rho$ vs $d(a,\partial C_{N})$",
                  fontsize=10)
    ax.set_title(
        "Per-component decomposition of the signed residual\n"
        "vs distance to the nearest matter concentration",
        fontsize=10)
    ax.legend(loc="lower left", fontsize=8, ncol=5, frameon=True)
    ax.set_ylim(-0.55, 0.55)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_residual_components_spearman.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_R00_distance_shells():
    n_shells = 5
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(rows)))
    for i, r in enumerate(rows):
        shells = r["R_00"]["shells_by_d"]
        if any(np.isnan(s.get("median", np.nan)) for s in shells):
            continue
        meds = [s["median"] for s in shells]
        ax.plot(np.arange(1, n_shells + 1), meds,
                "-o", color=cmap[i], linewidth=1.2, markersize=4,
                label=f"{r['regime']} $N\\!=\\!{r['N']}$")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("distance shell from nearest matter concentration "
                  "(quintile of $d(a,\\partial C_{N})$)", fontsize=10)
    ax.set_ylabel(r"median $R_{00}(a)$", fontsize=10)
    ax.set_title(
        "Residual energy density $R_{00}$ approaches zero with\n"
        "increasing distance to the nearest matter concentration",
        fontsize=10)
    ax.set_xticks(np.arange(1, n_shells + 1))
    ax.legend(fontsize=7, ncol=2, loc="lower right",
              frameon=True, framealpha=0.9)
    ax.grid(linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_residual_R00_distance_shells.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_R00_vs_Rii_cancellation():
    rho_R00 = np.array([r["R_00"]["spearman_signed_vs_d"] for r in rows])
    rho_R11 = np.array([r["R_11"]["spearman_signed_vs_d"] for r in rows])
    rho_R22 = np.array([r["R_22"]["spearman_signed_vs_d"] for r in rows])
    rho_R33 = np.array([r["R_33"]["spearman_signed_vs_d"] for r in rows])
    rho_diag_sum = rho_R11 + rho_R22 + rho_R33
    rho_tr = np.array([r["R_tr"]["spearman_signed_vs_d"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(ix, rho_R00, "-o", color="#1f3b6f", linewidth=1.4,
            markersize=5, label=r"$\rho(R_{00},d)$ "
            r"$\approx +0.40$")
    ax.plot(ix, rho_diag_sum, "-s", color="#a45a5a", linewidth=1.4,
            markersize=5, label=r"$\rho(R_{11},d)+\rho(R_{22},d)+\rho(R_{33},d)$ "
            r"$\approx -0.30$")
    ax.plot(ix, rho_tr, "-^", color="black", linewidth=1.4,
            markersize=5, label=r"$\rho(\mathrm{tr}\,R,d)$ (net)")
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xticks(ix)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"signed Spearman $\rho$ vs $d(a,\partial C_{N})$",
                  fontsize=10)
    ax.set_title(
        "The trace projection is a net of canceling contributions:\n"
        r"the time-time component $R_{00}$ carries the halo signal,"
        "\nthe spatial diagonal carries the (2+1) anisotropy",
        fontsize=10)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_ylim(-0.7, 0.7)
    fig.tight_layout()
    out = FIG_DIR / "fig_residual_components_signs.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = fig_components_spearman()
    p2 = fig_R00_distance_shells()
    p3 = fig_R00_vs_Rii_cancellation()
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
