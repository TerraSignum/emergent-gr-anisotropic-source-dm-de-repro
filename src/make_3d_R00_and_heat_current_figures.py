"""Two new 3D figures for the per-component halo audit.

  fig_R00_spatial_3d.pdf
      3D scatter of R_00(a) values in the spectral-Laplacian
      Fiedler frame on a representative regime. Replaces the
      legacy fig_3d_halo_around_defect (which used the trace
      projection, now known to be a net-of-canceling
      observable) with the correct energy-density component
      R_00. Matter cores marked with black crosses; nodes
      coloured by sign(R_00) with size proportional to
      |R_00|.

  fig_heat_current_J_3d.pdf
      3D quiver of the heat-current vector field J_i(a) at
      each lattice node, computed as J_i(a) = sum_b
      Xi_ab (T_00(b) - T_00(a)) e_ab^i. Visualises the
      regime-stable rho(|J|, d) = -0.63 (10/10 regimes)
      result of sub-finding (vi). Matter cores marked.
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
REGIME = "P5"
N_LAT = 50


def _load_one_seed():
    p = find_d1_npz(REGIME, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        raise SystemExit(f"snapshot not found for {REGIME} N={N_LAT}")
    seeds = (load_snapshots(p, N_LAT) if "snapshots" in p.name.lower()
             else load_canonical(p, N_LAT))
    if not seeds:
        raise SystemExit("no seeds")
    return seeds[0]


def _build_observables():
    seed = _load_one_seed()
    xi_mat = np.asarray(seed[0], float).copy()
    psi = np.asarray(seed[1])
    k_field = np.asarray(seed[2]) if len(seed) > 2 else None
    q_field = np.asarray(seed[3]) if len(seed) > 3 else None

    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, N_LAT, np)
    g_00 = prep["g_00_h"]
    t00 = prep["t00"]
    R_00 = g_00 + LAMBDA_T - t00

    # Recompute spectral-frame spatial coordinates (matches the per_seed
    # Galerkin construction used internally).
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(N_LAT, dtype=np.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    _, eigvecs = np.linalg.eigh(l_norm)
    spatial = eigvecs[:, 1:4]    # 3-component Fiedler-frame coordinates

    # Heat-current J_i(a) = sum_b Xi_ab adj_ab (T_00(b) - T_00(a)) e_ab_i,
    # with e_ab_i = (spatial[b] - spatial[a])_i / d_ab.
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    e_alpha = spatial_diff * inv_d[:, :, None]
    delta_t = t00[None, :] - t00[:, None]
    contrib = weight_adj[:, :, None] * delta_t[:, :, None] * e_alpha
    J = contrib.sum(axis=1)        # per-node heat-current 3-vector

    # Matter cores: top 5% by |T_00|
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * N_LAT)))
    core_idx = np.argsort(np.abs(t00))[-top_n:]
    return {
        "spatial": spatial, "R_00": R_00, "T_00": t00, "J": J,
        "core_idx": core_idx,
    }


def fig_R00_3d():
    d = _build_observables()
    sp = d["spatial"]; R = d["R_00"]
    core = d["core_idx"]
    pos = R > 0
    neg = R < 0
    abs_R = np.abs(R)
    size = 20 + 200 * (abs_R / max(abs_R.max(), 1e-12))

    fig = plt.figure(figsize=(8.6, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    if neg.sum() > 0:
        ax.scatter(sp[neg, 0], sp[neg, 1], sp[neg, 2],
                   s=size[neg], c="#1f3b6f", alpha=0.55,
                   label=f"$R_{{00}}<0$ ({neg.sum()} of {N_LAT}, "
                          f"unmet near-matter energy density)",
                   edgecolors="none", depthshade=True)
    if pos.sum() > 0:
        ax.scatter(sp[pos, 0], sp[pos, 1], sp[pos, 2],
                   s=size[pos], c="#a45a5a", alpha=0.55,
                   label=f"$R_{{00}}>0$ ({pos.sum()})",
                   edgecolors="none", depthshade=True)
    ax.scatter(sp[core, 0], sp[core, 1], sp[core, 2],
               marker="x", s=140, c="black", linewidths=2.0,
               label=f"matter cores (top {int(CORE_TOP_FRAC*100)}%"
                      f"$|T_{{00}}^\\Xi|$)")
    ax.set_xlabel(r"$x^{\,1}_{\mathrm{spec}}$", fontsize=9)
    ax.set_ylabel(r"$x^{\,2}_{\mathrm{spec}}$", fontsize=9)
    ax.set_zlabel(r"$x^{\,3}_{\mathrm{spec}}$", fontsize=9)
    ax.set_title(
        "Energy-density residual $R_{00}(a)$ in the spectral-frame\n"
        "spatial coordinates on a representative regime "
        f"({REGIME}, $N\\!=\\!{N_LAT}$, single seed)",
        fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92)
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()
    out = FIG_DIR / "fig_R00_spatial_3d.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_J_quiver_3d():
    d = _build_observables()
    sp = d["spatial"]; J = d["J"]
    core = d["core_idx"]
    Jmag = np.linalg.norm(J, axis=1)
    # Normalise arrow lengths for visual clarity
    Jmag_max = max(Jmag.max(), 1e-12)
    arrow_scale = 0.08
    Jn = J / Jmag_max * arrow_scale

    fig = plt.figure(figsize=(8.6, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    # Colour quiver by |J| via cmap: viridis
    cmap = plt.cm.viridis
    norm_mag = Jmag / Jmag_max
    colors = cmap(norm_mag)
    ax.quiver(sp[:, 0], sp[:, 1], sp[:, 2],
              Jn[:, 0], Jn[:, 1], Jn[:, 2],
              colors=colors, length=1.0, normalize=False,
              arrow_length_ratio=0.35, linewidth=0.9)
    ax.scatter(sp[core, 0], sp[core, 1], sp[core, 2],
               marker="x", s=160, c="red", linewidths=2.4,
               label=f"matter cores (top {int(CORE_TOP_FRAC*100)}%"
                     f"$|T_{{00}}^\\Xi|$)")
    ax.set_xlabel(r"$x^{\,1}_{\mathrm{spec}}$", fontsize=9)
    ax.set_ylabel(r"$x^{\,2}_{\mathrm{spec}}$", fontsize=9)
    ax.set_zlabel(r"$x^{\,3}_{\mathrm{spec}}$", fontsize=9)
    ax.set_title(
        r"Heat-current vector field $J_i(a)\!=\!\sum_b\Xi_{ab}"
        r"(T_{00}^{\Xi}(b)-T_{00}^{\Xi}(a))\,\hat e_{ab}^{\,(i)}$"
        "\n" f"on a representative regime ({REGIME}, $N\\!=\\!{N_LAT}$, "
        "single seed); arrow colour by $|J|/|J|_{\\max}$",
        fontsize=10)
    # Add manual proxy for matter-cores legend (quiver doesn't legend)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    # Add a colorbar proxy via a scalar mappable
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=Jmag_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.10)
    cbar.set_label(r"$|J(a)|$", fontsize=9)
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()
    out = FIG_DIR / "fig_heat_current_J_3d.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Wrote {fig_R00_3d()}")
    print(f"Wrote {fig_J_quiver_3d()}")
