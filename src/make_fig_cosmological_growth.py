r"""3-panel cosmological-growth figure for P4-B:
  A. f sigma_8(z) framework vs BOSS/eBOSS RSD
  B. S_8 framework vs Planck/DES/KiDS/HSC (S_8 tension)
  C. Halo mass function f(sigma) Tinker+ 2008 reproduction

Output: paper/figures/fig_cosmological_growth.pdf
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
DATA = REPO / "outputs" / "verify_cosmological_growth_S8.json"


def main():
    bundle = json.loads(DATA.read_text(encoding="utf-8"))
    rsd = bundle["f_sigma_8_z_evolution"]
    s8 = bundle["S_8_predictions_vs_surveys"]
    hmf = bundle["halo_mass_function_Tinker2008"]

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    # ---- A: f sigma_8(z) ----
    ax = fig.add_subplot(gs[0, 0])
    zs = [r["z"] for r in rsd]
    fsig_pred = [r["f_sigma_8_predicted"] for r in rsd]
    fsig_obs = [r["f_sigma_8_anchor"] for r in rsd]
    fsig_unc = [r["anchor_unc"] for r in rsd]
    z_grid = np.linspace(0.05, 1.05, 50)
    fsig_grid = []
    Om = bundle["cosmology_inputs"]["Omega_m_Planck_2018"]
    sigma_8_0 = bundle["cosmology_inputs"]["sigma_8_Planck_2018"]
    for z in z_grid:
        # quick D(z) approx via a-grid (cheap)
        Om_z = Om * (1 + z) ** 3 / (Om * (1 + z) ** 3 + (1 - Om))
        f_z = Om_z ** 0.55
        # Inline growth ODE quick-eval (~2000 step)
        a_z = 1 / (1 + z)
        n_steps = 500
        integral = 0.0
        for k in range(n_steps):
            a_p = a_z * (k + 0.5) / n_steps
            z_p = 1 / a_p - 1
            H_p = math.sqrt(Om * (1 + z_p) ** 3 + (1 - Om))
            integral += (a_z / n_steps) / (a_p * H_p) ** 3
        H_z = math.sqrt(Om * (1 + z) ** 3 + (1 - Om))
        D_z = 2.5 * Om * H_z * integral
        # D_0 same approx at z=0
        n0 = 500
        int0 = 0.0
        for k in range(n0):
            a_p = (k + 0.5) / n0
            z_p = 1 / a_p - 1
            H_p = math.sqrt(Om * (1 + z_p) ** 3 + (1 - Om))
            int0 += (1 / n0) / (a_p * H_p) ** 3
        D_0 = 2.5 * Om * 1.0 * int0
        sigma_8_z = sigma_8_0 * D_z / D_0
        fsig_grid.append(f_z * sigma_8_z)
    ax.plot(z_grid, fsig_grid, "-", color="#1f6f3f", lw=2.5,
            label=r"Framework $f\sigma_8(z)$ (LCDM, Planck 18 anchors)")
    ax.errorbar(zs, fsig_obs, yerr=fsig_unc, fmt="o", color="black",
                markersize=8, capsize=4, label="BOSS/eBOSS/6dFGS RSD")
    ax.set_xlabel("Redshift $z$", fontsize=11)
    ax.set_ylabel(r"$f\sigma_8(z)$", fontsize=11)
    ax.set_title(r"A. Linear growth $f\sigma_8(z)$ vs RSD", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.30, 0.55)

    # ---- B: S_8 tension ----
    ax = fig.add_subplot(gs[0, 1])
    surveys_short = {
        "Planck_2018_implied": "Planck 2018",
        "DES_Year_3_3x2pt": "DES Y3 (3x2pt)",
        "KiDS_1000_cosmic_shear": "KiDS-1000",
        "HSC_Year_3": "HSC Y3",
    }
    pretty_names = [surveys_short[r["survey"]] for r in s8]
    s8_vals = [r["S_8_anchor"] for r in s8]
    s8_unc = [r["unc"] for r in s8]
    colors = ["#1f6f3f"] + ["#bf5b3b"] * 3
    s8_pred = s8[0]["S_8_predicted"]
    ys = np.arange(len(pretty_names))
    ax.errorbar(s8_vals, ys, xerr=s8_unc, fmt="o",
                color="black", markersize=10, capsize=5,
                label="Survey anchors")
    for x, y, c in zip(s8_vals, ys, colors):
        ax.scatter(x, y, color=c, s=200, edgecolor="white",
                   linewidth=2, zorder=4)
    ax.axvline(s8_pred, color="#1f6f3f", lw=2.5, ls="-",
               label=f"Framework $S_8 = {s8_pred:.3f}$")
    ax.fill_betweenx([-0.5, 3.5], s8_pred - 0.014, s8_pred + 0.014,
                      color="#1f6f3f", alpha=0.2)
    ax.set_yticks(ys)
    ax.set_yticklabels(pretty_names, fontsize=10)
    ax.set_xlabel(r"$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$", fontsize=11)
    ax.set_title(r"B. $S_8$ tension reproduction", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.72, 0.86)
    # Annotate sigma offsets
    for r, y in zip(s8, ys):
        ax.text(0.728, y, f"{r['sigma_offset']:+.1f}σ",
                fontsize=8, va="center")

    # ---- C: Halo mass function ----
    ax = fig.add_subplot(gs[0, 2])
    Ms = np.geomspace(1e11, 1e16, 100)
    sigma_M_grid = sigma_8_0 * (Ms / 6e14) ** (-0.2)
    f_sigma_grid = []
    for sig in sigma_M_grid:
        A, a, b, c = 0.186, 1.47, 2.57, 1.19
        f_sigma_grid.append(A * ((sig / b) ** (-a) + 1)
                             * np.exp(-c / sig ** 2))
    ax.semilogx(Ms, f_sigma_grid, "-", color="#1f6f3f", lw=2.5,
                label="Framework Tinker $f(\\sigma)$")
    Ms_test = [r["M_Msun"] for r in hmf]
    f_sigma_test = [r["f_sigma_Tinker"] for r in hmf]
    ax.scatter(Ms_test, f_sigma_test, color="#bf5b3b", s=100,
               edgecolor="black", zorder=4, label="Test points")
    ax.set_xlabel(r"Halo mass $M$ [$M_\odot$]", fontsize=11)
    ax.set_ylabel(r"$f(\sigma)$ Tinker+ 2008", fontsize=11)
    ax.set_title("C. Halo mass function $f(\\sigma)$",
                  fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Cosmological growth: $f\\sigma_8(z)$, "
                  "$S_8$-tension, halo mass function "
                  "vs Planck 18 / DES Y3 / KiDS / HSC / BOSS RSD",
                  fontsize=12, y=1.02)
    out_pdf = FIG_DIR / "fig_cosmological_growth.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
