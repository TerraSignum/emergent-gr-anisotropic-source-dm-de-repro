r"""SPARC + stacked weak-lensing comparison figure for P4-B.

Two-page composite (4 panels):

  A. SPARC chi^2/dof per model across 15 representative
     galaxies (bar groups: framework / Burkert / Einasto / MOND)
  B. Example rotation-curve fit on NGC 3198 (framework NFW
     vs observed v_c(r) + baryon decomposition)
  C. Stacked weak-lensing DeltaSigma(R) profiles for three
     halo-mass bins, framework prediction overlaid with DES
     Y3 / KiDS-1000 / HSC-Y3 anchors
  D. Residual distribution: percentage residual histograms
     for SPARC (per-galaxy chi^2/dof) and lensing
     (per-data-point DeltaSigma residual %)

Output: paper/figures/fig_sparc_lensing_comparison.pdf
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

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SPARC_DATA = REPO / "outputs" / "verify_sparc_galaxy_fits.json"
LENS_DATA = REPO / "outputs" / "verify_stacked_lensing_des_kids_hsc.json"

PI = math.pi
G_ASTRO = 4.302e-6


def nfw_v_kms(r_kpc, rho_s, r_s):
    x = r_kpc / r_s
    M = 4 * PI * rho_s * r_s ** 3 * (np.log(1 + x) - x / (1 + x))
    return np.sqrt(G_ASTRO * M / np.maximum(r_kpc, 0.001))


def nfw_DS_kpc2(R, rho_s, r_s):
    x = R / r_s
    if abs(x - 1) < 1e-4:
        f = 1.0 / 3.0
        g = 1 + math.log(0.5)
    elif x < 1:
        F = math.acosh(1 / x) / math.sqrt(1 - x ** 2)
        f = (1 - F) / (x ** 2 - 1)
        g = math.log(x / 2) + F
    else:
        F = math.acos(1 / x) / math.sqrt(x ** 2 - 1)
        f = (1 - F) / (x ** 2 - 1)
        g = math.log(x / 2) + F
    return 4 * rho_s * r_s * g / x ** 2 - 2 * rho_s * r_s * f


def main():
    sparc = json.loads(SPARC_DATA.read_text(encoding="utf-8"))
    lens = json.loads(LENS_DATA.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # ---- Panel A: SPARC chi^2/dof bars ----
    ax = fig.add_subplot(gs[0, 0])
    rows = sparc["rows"]
    galaxies = [r["galaxy"] for r in rows]
    fw = [r["framework_NFW"]["chi2_per_dof"] for r in rows]
    burk = [r["Burkert"]["chi2_per_dof"] for r in rows]
    ein = [r["Einasto"]["chi2_per_dof"] for r in rows]
    mond = [r["MOND"]["chi2_per_dof"] for r in rows]
    xs = np.arange(len(galaxies))
    width = 0.20
    ax.bar(xs - 1.5 * width, fw, width, color="#1f6f3f", label="Framework NFW (0 free)")
    ax.bar(xs - 0.5 * width, burk, width, color="#3b8bbf", label="Burkert (2 free)")
    ax.bar(xs + 0.5 * width, ein, width, color="#bf5b3b", label="Einasto (3 free)")
    ax.bar(xs + 1.5 * width, np.minimum(mond, 1000),
           width, color="#888888", label="MOND (0 free)")
    ax.set_xticks(xs)
    ax.set_xticklabels([g.replace("_", "\n") for g in galaxies],
                        rotation=0, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\chi^2 / \text{dof}$ (clipped at 1000)", fontsize=10)
    ax.set_title("A. SPARC galaxy-by-galaxy fit quality "
                  "(15 galaxies, dwarf to massive)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")

    # ---- Panel B: Example NGC 3198 rotation curve ----
    ax = fig.add_subplot(gs[0, 1])
    g = next(r for r in rows if r["galaxy"] == "NGC_3198")
    M_b = g["M_b_e9_Msun"] * 1e9
    R_d = 4.4 / 1.68  # R_eff / 1.68
    rs_test = np.linspace(0.5, 30, 100)
    # Synthetic v_obs
    v_obs_pts = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    v_flat = g["v_flat_kms"]
    v_obs = v_flat * (1 - np.exp(-v_obs_pts / 4.4))
    # Baryon
    M_b_e9 = M_b * 1e-9
    v_bar = np.sqrt(G_ASTRO * M_b * (1 - np.exp(-rs_test / R_d) *
                                       (1 + rs_test / R_d)) / rs_test)
    # Framework NFW
    M_200 = 47.0 * v_flat ** 4
    c_BM = 10.0 * (M_200 / 1e12) ** (-0.1)
    R_200 = (3 * M_200 / (4 * PI * 200 * 140)) ** (1 / 3)
    r_s = R_200 / c_BM
    rho_s = M_200 / (4 * PI * r_s ** 3 *
                       (np.log(1 + c_BM) - c_BM / (1 + c_BM)))
    v_DM = nfw_v_kms(rs_test, rho_s, r_s)
    v_total = np.sqrt(v_DM ** 2 + v_bar ** 2)

    ax.plot(rs_test, v_DM, "--", color="#3b8bbf", lw=2,
            label=r"DM halo $v^{\rm DM}_c$ (framework NFW)")
    ax.plot(rs_test, v_bar, ":", color="#bf5b3b", lw=2,
            label=r"Baryon disk $v^{\rm bar}_c$")
    ax.plot(rs_test, v_total, "-", color="#1f6f3f", lw=2.5,
            label=r"Total $v^{\rm tot}_c$")
    ax.errorbar(v_obs_pts, v_obs, yerr=5, fmt="o",
                color="black", markersize=6, capsize=3,
                label="Observed (synthetic SPARC-like)")
    ax.set_xlabel("Radius $r$ [kpc]", fontsize=10)
    ax.set_ylabel(r"$v_c$ [km/s]", fontsize=10)
    ax.set_title(f"B. NGC 3198 rotation curve "
                  f"($v_{{\\rm flat}}\\!=\\!{v_flat}$ km/s)",
                  fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(v_total) * 1.2)

    # ---- Panel C: Stacked lensing DeltaSigma(R) ----
    ax = fig.add_subplot(gs[1, 0])
    surveys_colors = {
        "DES_Y3": "#1f6f3f",
        "KiDS_1000": "#3b8bbf",
        "HSC_Y3": "#bf5b3b",
    }
    # Plot anchor points
    for r in lens["rows"]:
        survey = r["survey_mass_bin"].split("_M_h")[0]
        if survey in surveys_colors:
            color = surveys_colors[survey]
            mh_val = r["M_h_anchor_Msun"]
            marker = "o" if mh_val < 5e12 else ("s" if mh_val < 5e13 else "D")
            ax.scatter(r["R_kpc"], r["DS_anchor_Msun_pc2"],
                       marker=marker, s=70, color=color, edgecolor="white",
                       zorder=4)
            ax.scatter(r["R_kpc"], r["DS_predicted_Msun_pc2"],
                       marker=marker, s=70, facecolor="none",
                       edgecolor=color, linewidth=1.8, zorder=3)
    # Plot continuous framework predictions
    R_grid = np.geomspace(20, 1500, 200)
    for M_h, M_star, lbl in [
        (1e12, 5e10, r"$M_h = 10^{12}\,M_\odot$ (L*)"),
        (1e13, 1.5e11, r"$M_h = 10^{13}\,M_\odot$ (group)"),
        (5e13, 3e11, r"$M_h = 5\times 10^{13}\,M_\odot$ (cluster)"),
    ]:
        c = 10 * (M_h / 1e12) ** (-0.1)
        R_200 = (3 * M_h / (4 * PI * 200 * 140)) ** (1/3)
        r_s = R_200 / c
        rho_s = M_h / (4 * PI * r_s ** 3 * (math.log(1 + c) - c / (1 + c)))
        DS_NFW = np.array([nfw_DS_kpc2(R, rho_s, r_s) for R in R_grid]) / 1e6
        DS_central = M_star / (PI * (R_grid * 1e3) ** 2) * 1e6 / 1e6
        DS_total = DS_NFW + DS_central
        ax.loglog(R_grid, DS_total, "-", color="black", lw=1.0, alpha=0.6)
        ax.text(R_grid[10], DS_total[10] * 1.3, lbl, fontsize=8,
                color="black", alpha=0.9)
    # Legend
    for survey, color in surveys_colors.items():
        ax.scatter([], [], color=color, marker="o", s=60,
                   label=survey.replace("_", " "))
    ax.scatter([], [], facecolor="none", edgecolor="black",
               marker="o", s=60, linewidth=1.5, label="Framework prediction")
    ax.scatter([], [], color="black", marker="o", s=60,
               label="Survey anchor")
    ax.set_xlabel(r"Projected radius $R$ [kpc]", fontsize=10)
    ax.set_ylabel(r"$\Delta\Sigma(R)$ [$M_\odot/\text{pc}^2$]", fontsize=10)
    ax.set_title("C. Stacked weak-lensing $\\Delta\\Sigma(R)$: "
                  "framework vs DES/KiDS/HSC", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(0.3, 500)

    # ---- Panel D: Residual histograms ----
    ax = fig.add_subplot(gs[1, 1])
    sparc_resids = np.array([r["framework_NFW"]["chi2_per_dof"] for r in rows])
    sparc_resids_burkert = np.array([r["Burkert"]["chi2_per_dof"] for r in rows])
    lens_resids = np.array([r["residual_pct"] for r in lens["rows"]])

    ax.hist(np.log10(sparc_resids), bins=10,
            alpha=0.6, color="#1f6f3f", label="SPARC framework NFW")
    ax.hist(np.log10(sparc_resids_burkert), bins=10,
            alpha=0.6, color="#3b8bbf", label="SPARC Burkert (2 free)")
    ax2 = ax.twiny()
    ax2.hist(lens_resids, bins=10, alpha=0.4, color="#bf5b3b",
             label="Stacked lensing residual %")
    ax2.set_xlabel("Lensing residual %", fontsize=9, color="#bf5b3b")
    ax.set_xlabel(r"$\log_{10}(\chi^2 / \text{dof})$", fontsize=10)
    ax.set_ylabel("Counts", fontsize=10)
    ax.set_title("D. Residual distributions: SPARC + lensing",
                  fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Galaxy-by-galaxy SPARC fits + stacked weak-lensing "
                  "comparison vs DES Y3 / KiDS-1000 / HSC-Y3",
                  fontsize=13, y=0.995)
    out_pdf = FIG_DIR / "fig_sparc_lensing_comparison.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
