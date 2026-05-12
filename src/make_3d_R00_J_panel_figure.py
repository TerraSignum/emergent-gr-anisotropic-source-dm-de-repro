"""Combined / multi-N panel for R_00 + heat-current J in 3D.

Replaces the two single-regime figures
(fig_R00_spatial_3d.pdf, fig_heat_current_J_3d.pdf) with one
2x2 panel showing R_00-sign distribution AND J-vector field on
four representative regimes spanning the canonical-physics
ladder:
  Top-left:    P5    N=50
  Top-right:   P7    N=72
  Bottom-left: P5N100  N=100
  Bottom-right:P5N200  N=200

Each panel shows the same single-seed configuration in the
spectral-Laplacian Fiedler frame:
  - Node colour = sign(R_00) (blue: negative, red: positive)
  - Node size proportional to |R_00|
  - Heat-current arrows J_i(a) at every node, scaled by |J|
  - Black crosses = matter cores

This (i) demonstrates regime-stability of the spatial halo
signature visually across N spanning a factor of 4 (50 to 200),
and (ii) integrates two previously-separate visualisations
into one self-consistent figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO.parent
sys.path.insert(0, str(PARENT / "emergent-gr-closure-repro" / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, _name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LAMBDA_T, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CORE_TOP_FRAC = 0.05

PANELS = [
    ("P5",     50,  "(a) $P_{5}$, $N\\!=\\!50$"),
    ("P7",     72,  "(b) $P_{7}$, $N\\!=\\!72$"),
    ("P5N100", 100, "(c) $P_{5}N_{100}$, $N\\!=\\!100$"),
    ("P5N200", 200, "(d) $P_{5}N_{200}$, $N\\!=\\!200$"),
]


def _build(regime, n_lat):
    p = find_snapshot = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat) if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))
    if not seeds:
        return None
    seed = seeds[0]
    xi_mat = np.asarray(seed[0], float).copy()
    psi = np.asarray(seed[1])
    k_field = np.asarray(seed[2]) if len(seed) > 2 else None
    q_field = np.asarray(seed[3]) if len(seed) > 3 else None

    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]; t00 = prep["t00"]
    R_00 = g_00 + LAMBDA_T - t00

    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat, dtype=np.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    _, eigvecs = np.linalg.eigh(l_norm)
    spatial = eigvecs[:, 1:4]

    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    e_alpha = spatial_diff * inv_d[:, :, None]
    delta_t = t00[None, :] - t00[:, None]
    contrib = weight_adj[:, :, None] * delta_t[:, :, None] * e_alpha
    J = contrib.sum(axis=1)

    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    core_idx = np.argsort(np.abs(t00))[-top_n:]
    return spatial, R_00, J, core_idx


def _draw_panel(ax, spatial, R_00, J, core_idx, title, n_lat):
    pos = R_00 > 0
    neg = R_00 < 0
    f_pos = float(pos.mean())
    f_neg = float(neg.mean())
    abs_R = np.abs(R_00)
    abs_R = np.where(np.isfinite(abs_R), abs_R, 0)
    size_norm = np.maximum(abs_R, 1e-12)
    size = 30 + 280 * (size_norm / max(size_norm.max(), 1e-12))

    if neg.sum() > 0:
        ax.scatter(spatial[neg, 0], spatial[neg, 1], spatial[neg, 2],
                   s=size[neg], c="#1f3b6f", alpha=0.65,
                   edgecolors="none", depthshade=True,
                   label=(f"$R_{{00}}<0$ ({neg.sum()} of {n_lat}, "
                          f"{f_neg*100:.1f}%, "
                          f"unmet near-matter density)"))
    if pos.sum() > 0:
        ax.scatter(spatial[pos, 0], spatial[pos, 1], spatial[pos, 2],
                   s=size[pos], c="#a45a5a", alpha=0.65,
                   edgecolors="none", depthshade=True,
                   label=(f"$R_{{00}}>0$ ({pos.sum()}, "
                          f"{f_pos*100:.1f}%)"))
    j_mag = np.linalg.norm(J, axis=1)
    j_mag = np.where(np.isfinite(j_mag), j_mag, 0.0)
    j_mag_max = max(j_mag.max(), 1e-12)
    j_n = np.where(np.isfinite(J), J, 0.0) / j_mag_max * 0.10
    cmap = plt.cm.viridis
    quiver_colors = cmap(j_mag / j_mag_max)
    sub = np.arange(len(R_00))
    ax.quiver(spatial[sub, 0], spatial[sub, 1], spatial[sub, 2],
              j_n[sub, 0], j_n[sub, 1], j_n[sub, 2],
              colors=quiver_colors, length=1.0, normalize=False,
              arrow_length_ratio=0.35, linewidth=1.0)
    ax.scatter(spatial[core_idx, 0], spatial[core_idx, 1],
               spatial[core_idx, 2],
               marker="x", s=180, c="black", linewidths=2.6,
               label="matter cores (top 5% $|T_{00}^{\\Xi}|$)")
    ax.set_xlabel(r"$x^{\,1}_{\mathrm{spec}}$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$x^{\,2}_{\mathrm{spec}}$", fontsize=11, labelpad=6)
    ax.set_zlabel(r"$x^{\,3}_{\mathrm{spec}}$", fontsize=11, labelpad=6)
    ax.set_title(title, fontsize=15, pad=12)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.view_init(elev=22, azim=35)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)


def main():
    panels_data = []
    for regime, n_lat, title in PANELS:
        result = _build(regime, n_lat)
        panels_data.append((regime, n_lat, title, result))

    for regime, n_lat, title, result in panels_data:
        fig = plt.figure(figsize=(10, 8.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if result is None:
            ax.text(0, 0, 0, f"(no data) {regime} N={n_lat}",
                    fontsize=12)
        else:
            spatial, r_00, j_vec, core_idx = result
            _draw_panel(ax, spatial, r_00, j_vec, core_idx, title,
                        n_lat)
        fig.tight_layout()
        out = FIG_DIR / f"fig_R00_J_{regime}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
