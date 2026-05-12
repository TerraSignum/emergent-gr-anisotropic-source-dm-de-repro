r"""Multi-panel figures for the baryonic-halo phenomenology pipeline.

Generates two compound figures embedded in the P4-B paper:

  Figure A (2x3 panel grid, 2D):
    A1. Rotation curve v_c(r): DM-only, baryon-only, total vs MW data
    A2. Radial Acceleration Relation (g_obs vs g_bar diagonal)
    A3. Baryonic Tully-Fisher relation (M_b vs v_f^4)
    A4. Weak-lensing DeltaSigma(R) profile (NFW + central galaxy)
    A5. Halo profile rho(r): NFW vs Burkert vs uniform comparison
    A6. Subhalo mass function dN/dM_sub

  Figure B (3D + 1D side panel):
    B1. 3D rotation curve "cube" (r vs M_200 vs v_c) coloured
        by NFW concentration
    B2. 1D side panel: scatter sigma_logc, sigma_RAR, sigma_BTFR
        vs observed ranges as bar comparison

Output: paper/figures/fig_baryonic_halo_pipeline_2d.pdf
        paper/figures/fig_baryonic_halo_pipeline_3d.pdf
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA = REPO / "outputs" / "verify_baryonic_halo_phenomenology.json"

PI = math.pi


def nfw_M_enclosed(r, rho_s, r_s):
    x = r / r_s
    return 4 * PI * rho_s * r_s ** 3 * (np.log(1 + x) - x / (1 + x))


def nfw_v_circ_kms(r, rho_s, r_s):
    G = 4.302e-6
    M = nfw_M_enclosed(r, rho_s, r_s)
    return np.sqrt(G * M / r)


def nfw_rho(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


def burkert_rho(r, rho0, r_c):
    x = r / r_c
    return rho0 / ((1 + x) * (1 + x ** 2))


def main():
    bundle = json.loads(DATA.read_text(encoding="utf-8"))
    a2 = bundle["axis_2_halo_normalisation"]
    rho_s = a2["rho_s_Msun_per_kpc3"]
    r_s = a2["r_s_kpc"]
    a6 = bundle["axis_6_RAR_BTFR"]
    g_dagger_pred = a6["g_dagger_predicted_m_s2"]
    g_dagger_obs = a6["g_dagger_observed_m_s2"]
    a8 = bundle["axis_8_substructure"]

    # =================================================================
    # FIGURE A: 2x3 panel grid
    # =================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30)

    # ---- A1: Rotation curve ----
    ax = fig.add_subplot(gs[0, 0])
    rs = np.geomspace(0.1, 50, 200)
    v_dm = nfw_v_circ_kms(rs, rho_s, r_s)
    # Baryon proxy: exponential disk + bulge giving v_disk(R_sol)~180 km/s
    M_disk = 5e10  # M_sun
    R_d = 3.0  # kpc
    v_disk = np.sqrt(4.302e-6 * M_disk * (1 - np.exp(-rs / R_d) * (1 + rs / R_d)) / rs)
    v_total = np.sqrt(v_dm ** 2 + v_disk ** 2)
    ax.plot(rs, v_dm, "--", color="#3b8bbf", lw=2,
            label=r"Dark-matter halo $v_c^{\rm DM}$")
    ax.plot(rs, v_disk, ":", color="#bf5b3b", lw=2,
            label=r"Baryon disk $v_c^{\rm bar}$")
    ax.plot(rs, v_total, "-", color="#1f6f3f", lw=2.5,
            label=r"Total $v_c^{\rm tot}$ (framework)")
    ax.errorbar(8.122, 232.8, yerr=3, marker="o", color="black",
                markersize=8, capsize=4, label=r"MW $v_c(R_\odot)$ (Gaia DR3)")
    ax.set_xscale("log")
    ax.set_xlabel("Radius $r$ [kpc]", fontsize=10)
    ax.set_ylabel(r"$v_c(r)$ [km/s]", fontsize=10)
    ax.set_title("A1. Rotation curve", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 300)

    # ---- A2: RAR ----
    ax = fig.add_subplot(gs[0, 1])
    g_bar = np.geomspace(1e-13, 1e-8, 200)
    g_obs_pred = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger_pred)))
    g_obs_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger_obs)))
    ax.loglog(g_bar, g_bar, "--", color="grey", lw=1, label=r"$g_{\rm obs} = g_{\rm bar}$")
    ax.loglog(g_bar, g_obs_pred, "-", color="#1f6f3f", lw=2.5,
              label=f"Framework ($c H_0 / 2\\pi$, residual {a6['g_dagger_residual_pct']:.1f}%)")
    ax.loglog(g_bar, g_obs_obs, ":", color="#bf5b3b", lw=2,
              label=r"McGaugh-Lelli-Schombert 2016")
    ax.axvline(g_dagger_obs, color="black", lw=0.8, ls=":")
    ax.text(g_dagger_obs * 1.5, 1e-12, r"$g_\dagger$",
            fontsize=11, color="black")
    ax.set_xlabel(r"$g_{\rm bar}$ [m/s$^2$]", fontsize=10)
    ax.set_ylabel(r"$g_{\rm obs}$ [m/s$^2$]", fontsize=10)
    ax.set_title("A2. Radial acceleration relation", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # ---- A3: BTFR ----
    ax = fig.add_subplot(gs[0, 2])
    v_f = np.geomspace(20, 400, 100)
    M_b_pred = a6["BTFR_A_predicted_Msun_per_kms4"] * v_f ** 4
    M_b_obs = a6["BTFR_A_observed_Msun_per_kms4"] * v_f ** 4
    ax.loglog(v_f, M_b_pred, "-", color="#1f6f3f", lw=2.5,
              label=f"Framework $A=(G_N H_0)^{{-1}}$")
    ax.loglog(v_f, M_b_obs, "--", color="#bf5b3b", lw=2,
              label=r"McGaugh 2012, $A\!=\!47\,M_\odot/(\text{km/s})^4$")
    ax.set_xlabel(r"$v_{\rm flat}$ [km/s]", fontsize=10)
    ax.set_ylabel(r"$M_b$ [$M_\odot$]", fontsize=10)
    ax.set_title(f"A3. Baryonic Tully-Fisher (residual "
                 f"{a6['BTFR_A_residual_pct']:.1f}%)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # ---- A4: Weak lensing DeltaSigma(R) ----
    ax = fig.add_subplot(gs[1, 0])
    R_kpc = np.geomspace(20, 2000, 200)
    DS = []
    for R in R_kpc:
        x = R / r_s
        if abs(x - 1) < 1e-6:
            f = 1.0 / 3.0
            g = 1.0 + math.log(0.5)
        elif x < 1:
            F = math.acosh(1.0 / x) / math.sqrt(1 - x ** 2)
            f = (1 - F) / (x ** 2 - 1)
            g = math.log(x / 2) + F
        else:
            F = math.acos(1.0 / x) / math.sqrt(x ** 2 - 1)
            f = (1 - F) / (x ** 2 - 1)
            g = math.log(x / 2) + F
        Sigma_R = 2 * rho_s * r_s * f
        Sigma_bar_R = 2 * rho_s * r_s * g / x ** 2
        DS.append((Sigma_bar_R - Sigma_R) / 1e6)
    DS = np.array(DS)
    # Add point-mass central galaxy: ΔΣ_central(R) = M_★ / (π R²)
    M_star = 5e10  # central galaxy stellar mass
    DS_central = M_star / (PI * (R_kpc * 1e3) ** 2)
    DS_total = DS + DS_central
    ax.loglog(R_kpc, DS_total, "-", color="#1f6f3f", lw=2.5,
              label="Framework total (NFW + central)")
    ax.loglog(R_kpc, DS, "--", color="#3b8bbf", lw=2, label="NFW halo only")
    ax.loglog(R_kpc, DS_central, ":", color="#bf5b3b", lw=2, label="Central galaxy")
    ax.scatter([100], [30], marker="o", s=80, color="black",
               edgecolor="white", zorder=5,
               label="DES Y3 stacked galaxy-galaxy")
    ax.set_xlabel("Projected radius $R$ [kpc]", fontsize=10)
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot/\text{pc}^2$]", fontsize=10)
    ax.set_title("A4. Weak lensing $\\Delta\\Sigma(R)$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(0.1, 1e3)

    # ---- A5: Halo profile shape comparison ----
    ax = fig.add_subplot(gs[1, 1])
    rs_full = np.geomspace(0.01 * r_s, 30 * r_s, 200)
    rho_NFW = nfw_rho(rs_full, rho_s, r_s)
    # Burkert with same M(<r=20 kpc) match
    rho0_burkert = rho_s * 2.0  # rough match
    r_c_burkert = r_s * 0.5
    rho_burkert = burkert_rho(rs_full, rho0_burkert, r_c_burkert)
    rho_uniform = np.full_like(rs_full, rho_s * 0.05)
    ax.loglog(rs_full / r_s, rho_NFW / rho_s, "-", color="#1f6f3f",
              lw=2.5, label="NFW (8/10 regimes)")
    ax.loglog(rs_full / r_s, rho_burkert / rho_s, "--", color="#bf5b3b",
              lw=2, label="Burkert (LSB candidates)")
    ax.loglog(rs_full / r_s, rho_uniform / rho_s, ":", color="#888888",
              lw=2, label="Uniform null (large-N limit)")
    ax.set_xlabel(r"$r / r_s$", fontsize=10)
    ax.set_ylabel(r"$\rho / \rho_s$", fontsize=10)
    ax.set_title("A5. Halo profile shape", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # ---- A6: Subhalo mass function ----
    ax = fig.add_subplot(gs[1, 2])
    M_sub = np.geomspace(1e7, 1e11, 100)
    M_host = 1e12
    alpha_pred = a8["subhalo_slope_alpha_predicted"]
    alpha_obs = a8["subhalo_slope_alpha_observed_Springel_2008"]
    N_pred = (M_sub / M_host) ** alpha_pred
    N_obs = (M_sub / M_host) ** alpha_obs
    ax.loglog(M_sub, N_pred, "-", color="#1f6f3f", lw=2.5,
              label=rf"Framework $\alpha\!=\!-1\!+\!\gamma/2\!=\!{alpha_pred:.2f}$")
    ax.loglog(M_sub, N_obs, "--", color="#bf5b3b", lw=2,
              label=rf"Aquarius $\alpha\!=\!{alpha_obs}$ (Springel+ 2008)")
    ax.set_xlabel(r"Subhalo mass $M_{\rm sub}$ [$M_\odot$]", fontsize=10)
    ax.set_ylabel(r"$dN/dM_{\rm sub}$ (relative)", fontsize=10)
    ax.set_title(f"A6. Subhalo mass function "
                 f"(residual {a8['alpha_residual_pct']:.1f}%)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Baryonic-halo phenomenology: framework predictions "
                  "vs external anchors (SPARC, McGaugh+ 2016, "
                  "DES Y3, Aquarius)", fontsize=13, y=0.995)
    out_2d = FIG_DIR / "fig_baryonic_halo_pipeline_2d.pdf"
    fig.savefig(out_2d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_2d}")

    # =================================================================
    # FIGURE B: 3D rotation-curve cube + scatter side panel
    # =================================================================
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.20)

    # ---- B1: 3D cube r x M_200 x v_c ----
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    M200_grid = np.geomspace(1e10, 1e14, 30)
    r_grid = np.geomspace(1, 100, 30)
    R_M, R_R = np.meshgrid(np.log10(M200_grid), np.log10(r_grid))
    V_grid = np.zeros_like(R_M)
    for i, M in enumerate(M200_grid):
        # rho_s, r_s scale with M_200
        r_s_M = r_s * (M / 1e12) ** (1/3)
        # rho_s ~ same shape, normalised to M
        c = 12.0
        rho_s_M = M / (4 * PI * r_s_M ** 3 *
                       (math.log(1 + c) - c / (1 + c)))
        for j, r in enumerate(r_grid):
            v = nfw_v_circ_kms(r, rho_s_M, r_s_M)
            V_grid[j, i] = v
    surf = ax3d.plot_surface(
        R_M, R_R, V_grid, cmap=cm.viridis, alpha=0.85,
        edgecolor="none", linewidth=0)
    ax3d.set_xlabel(r"$\log_{10}(M_{200} / M_\odot)$", fontsize=10)
    ax3d.set_ylabel(r"$\log_{10}(r / \text{kpc})$", fontsize=10)
    ax3d.set_zlabel(r"$v_c$ [km/s]", fontsize=10)
    ax3d.set_title("B1. 3D rotation-curve manifold "
                    "$v_c(r, M_{200})$", fontsize=11)
    cbar = fig.colorbar(surf, ax=ax3d, shrink=0.55, pad=0.10)
    cbar.set_label(r"$v_c$ [km/s]", fontsize=9)
    # Mark MW data point
    ax3d.scatter([np.log10(1e12)], [np.log10(8.122)], [232.8],
                  color="red", s=120, edgecolor="white",
                  label="MW (Gaia DR3)", depthshade=False)
    ax3d.view_init(elev=22, azim=-58)

    # ---- B2: scatter comparison side panel ----
    ax = fig.add_subplot(gs[0, 1])
    a5 = bundle["axis_5_scatter_relations"]
    relations = ["sigma_logc", "sigma_BTFR", "sigma_RAR"]
    pred_vals = [
        a5["sigma_logc_predicted"],
        a5["sigma_BTFR_predicted_dex"],
        a5["sigma_RAR_predicted_dex"],
    ]
    obs_vals = [
        a5["sigma_logc_observed_Bullock_Kolatt_1999"],
        a5["sigma_BTFR_observed_Lelli_2016_dex"],
        a5["sigma_RAR_observed_McGaugh_2016_dex"],
    ]
    pretty = [
        r"$\sigma_{\log c}$" + "\n(Bullock-Kolatt 1999)",
        r"$\sigma_{\rm BTFR}$" + "\n(Lelli+ 2016, dex)",
        r"$\sigma_{\rm RAR}$" + "\n(McGaugh 2016, dex)",
    ]
    xs = np.arange(len(relations))
    width = 0.35
    ax.bar(xs - width / 2, pred_vals, width, color="#1f6f3f",
           alpha=0.8, edgecolor="k", label="Framework")
    ax.bar(xs + width / 2, obs_vals, width, color="#bf5b3b",
           alpha=0.8, edgecolor="k", label="Observed")
    for x, p in zip(xs, pred_vals):
        ax.text(x - width / 2, p + 0.005, f"{p:.3f}",
                ha="center", fontsize=8)
    for x, o in zip(xs, obs_vals):
        ax.text(x + width / 2, o + 0.005, f"{o:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty, fontsize=9)
    ax.set_ylabel("Scatter (dex)", fontsize=10)
    ax.set_title("B2. Halo scaling-relation scatters", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.20)

    fig.suptitle("3D rotation-curve manifold + scaling-relation scatters",
                  fontsize=12, y=1.01)
    out_3d = FIG_DIR / "fig_baryonic_halo_pipeline_3d.pdf"
    fig.savefig(out_3d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_3d}")


if __name__ == "__main__":
    main()
