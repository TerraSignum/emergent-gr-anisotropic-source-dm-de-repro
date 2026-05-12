r"""Six-panel chirality-flip phase diagram figure for P4 manuscript.

Panels:
  A: theta_chir(N) running -- 8-regime data + log-fit + power-law fit
     + flip line theta=pi/4 + asymptote arctan(N_gen)
  B: alpha_xi(N) and gamma(N) running, with vacuum-matter flip at
     alpha=gamma=1/2 marked
  C: beta_pi(N) per-regime: observed vs (143/144, 23/48) refined
     mixing form vs (15/16, 1/2) old form
  D: D_Omega(N) per-regime non-monotonic + linear chirality-mixing fit
  E: C2 constraint residual D_Omega - (beta_pi - gamma) per N
     (grows from +0.001 at N=50 to +0.874 at N=300)
  F: alpha_xi/beta_pi power-law N^(-2/5) fit (R^2=0.991)
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
DATA = REPO / "data"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

D = 4
N_GEN = 3
PI = math.pi


def main():
    src = DATA / "causal_wave_per_N_readout.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data["p5_ladder_per_N_readout"]
    Ns = np.array([r["n_lat"] for r in rows])
    alphas = np.array([r["alpha_xi"] for r in rows])
    betas = np.array([r["beta_pi"] for r in rows])
    DOs = np.array([r["D_omega_lattice"] for r in rows])
    gammas = np.array([r["gamma_C1"] for r in rows])
    thetas = np.array([math.acos(math.sqrt(a)) if 0 < a < 1 else
                         float("nan") for a in alphas])
    thetas_deg = np.degrees(thetas)
    tan2s = np.tan(thetas) ** 2
    regime_names = [r["regime"] for r in rows]

    # Power-law fit for tan^2(theta) ~ N^b
    log_N = np.log(Ns)
    log_tan2 = np.log(tan2s)
    pwr_b, pwr_a = np.polyfit(log_N, log_tan2, 1)

    # Log-linear fit for theta_deg vs ln(N)
    th_b, th_a = np.polyfit(log_N, thetas_deg, 1)

    # Predict N_flip and N_inversion
    N_flip = math.exp((-pwr_a) / pwr_b)
    N_inv = math.exp((math.log(N_GEN ** 2) - pwr_a) / pwr_b)

    # Refined and old beta_pi mixing forms
    a_vac_new = (2 ** D * N_GEN ** 2 - 1) / (2 ** D * N_GEN ** 2)
    a_mat_new = (2 * D * N_GEN - 1) / (4 * D * N_GEN)
    bp_pred_new = a_vac_new * alphas + a_mat_new * gammas
    bp_pred_old = (15 / 16) * alphas + 0.5 * gammas

    # alpha/beta power-law
    ratios = alphas / betas
    log_r = np.log(ratios)
    rp_b, rp_a = np.polyfit(log_N, log_r, 1)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # ---- A: theta_chir(N) ----
    ax = fig.add_subplot(gs[0, 0])
    N_smooth = np.geomspace(40, 700, 200)
    theta_smooth_log = th_a + th_b * np.log(N_smooth)
    theta_smooth_pwr = np.degrees(np.arctan(np.sqrt(
        np.exp(pwr_a) * N_smooth ** pwr_b)))
    ax.plot(N_smooth, theta_smooth_log, "g--", lw=1.0, alpha=0.6,
             label=r"$\theta = a + b\log N$ ($R^2\!=\!0.96$)")
    ax.plot(N_smooth, theta_smooth_pwr, "b-", lw=1.5, alpha=0.7,
             label=r"$\tan^2\theta \sim N^{1.63}$ ($R^2\!=\!0.94$)")
    ax.scatter(Ns, thetas_deg, s=80, c="#bf5b3b", edgecolors="k",
                zorder=5, label="lattice data (8 regimes)")
    ax.axhline(45, color="purple", lw=1.0, ls=":")
    ax.axhline(math.degrees(math.atan(N_GEN)), color="green",
                lw=1.0, ls=":")
    ax.axhline(math.degrees(math.atan(1/N_GEN)), color="orange",
                lw=1.0, ls=":")
    ax.text(50, 47, r"$\theta\!=\!\pi/4$ flip",
             color="purple", fontsize=9)
    ax.text(50, 73, r"$\arctan(N_{\rm gen})\!=\!71.6^\circ$ "
             "(inversion)", color="green", fontsize=9)
    ax.text(50, 14, r"$\arctan(1/N_{\rm gen})\!=\!18.4^\circ$ "
             "(canonical)", color="orange", fontsize=9)
    ax.set_xlabel(r"$N$ (lattice resolution)", fontsize=10)
    ax.set_ylabel(r"$\theta_{\rm chir}$ (deg)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(40, 700)
    ax.set_ylim(0, 90)
    ax.set_title(r"A. Chirality angle $\theta_{\rm chir}(N)$",
                  fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- B: alpha_xi and gamma running ----
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Ns, alphas, "o-", color="#bf5b3b", lw=1.5, ms=8,
             label=r"$\alpha_\xi(N)\!=\!\cos^2\theta$")
    ax.plot(Ns, gammas, "s-", color="#3b8bbf", lw=1.5, ms=8,
             label=r"$\gamma(N)\!=\!\sin^2\theta$")
    ax.axhline(0.5, color="purple", lw=1.0, ls=":",
                label=r"$\alpha\!=\!\gamma\!=\!1/2$ flip")
    ax.axvspan(50, 110, alpha=0.1, color="orange",
                label="vacuum phase")
    ax.axvspan(120, 300, alpha=0.1, color="green",
                label="matter phase")
    for N_v, a_v, g_v in zip(Ns, alphas, gammas):
        ax.text(N_v, a_v + 0.02, f"{N_v}", ha="center",
                fontsize=7, color="#bf5b3b")
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel("coefficient", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(40, 400)
    ax.set_ylim(0, 1)
    ax.set_title(r"B. $\alpha_\xi(N)$ and $\gamma(N)$ "
                   r"with vacuum-matter flip", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- C: beta_pi refined vs old mixing ----
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(alphas, betas, s=80, c="#bf5b3b", edgecolors="k",
                zorder=5, label=r"$\beta_\pi(N)$ observed")
    a_smooth = np.linspace(0.1, 1.0, 100)
    g_smooth = 1 - a_smooth
    bp_new_smooth = a_vac_new * a_smooth + a_mat_new * g_smooth
    bp_old_smooth = (15/16) * a_smooth + 0.5 * g_smooth
    ax.plot(a_smooth, bp_new_smooth, "g-", lw=1.7, alpha=0.8,
             label=r"$\frac{143}{144}\cos^2 + \frac{23}{48}\sin^2$ "
             r"(0.6\%)")
    ax.plot(a_smooth, bp_old_smooth, "b--", lw=1.3, alpha=0.7,
             label=r"$\frac{15}{16}\cos^2 + \frac{1}{2}\sin^2$ "
             r"(3.2\%)")
    ax.set_xlabel(r"$\alpha_\xi(N) = \cos^2\theta$", fontsize=10)
    ax.set_ylabel(r"$\beta_\pi(N)$", fontsize=10)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(r"C. $\beta_\pi$ chirality-mixing form",
                  fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- D: D_Omega per-regime + linear fit ----
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(Ns, DOs, s=80, c="#3b8bbf", edgecolors="k", zorder=5,
                label=r"$D_\Omega(N)$ observed")
    # Linear fit in alpha_xi
    DO_fit_b, DO_fit_a = np.polyfit(alphas, DOs, 1)
    DO_pred = DO_fit_a + DO_fit_b * alphas
    ax.plot(Ns, DO_pred, "r--", lw=1.3, alpha=0.7,
             label=f"linear in $\\alpha_\\xi$ "
             f"(slope {DO_fit_b:.2f}, R$^2$=0.65)")
    ax.axhline(67/80, color="purple", lw=1.0, ls=":",
                label=r"canonical $67/80$")
    ax.axhline(PI/4, color="green", lw=1.0, ls=":",
                label=r"Symanzik $\pi/4$")
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel(r"$D_\Omega(N)$", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(40, 400)
    ax.set_ylim(0, 1.1)
    ax.set_title(r"D. $D_\Omega(N)$: non-monotonic, "
                  r"NOT clean chirality-mix",
                  fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # ---- E: C2 constraint residual per N ----
    ax = fig.add_subplot(gs[1, 1])
    c2_residuals = DOs - (betas - gammas)
    bars = ax.bar(range(len(Ns)), c2_residuals, color="#bf5b3b",
                    alpha=0.7, edgecolor="k")
    for i, (n, r) in enumerate(zip(Ns, c2_residuals)):
        c = "#1f6f3f" if abs(r) < 0.05 else \
            "#3b8bbf" if abs(r) < 0.2 else "#bf5b3b"
        bars[i].set_color(c)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns], rotation=0,
                         fontsize=9)
    ax.axhline(0.05, color="green", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel(r"$D_\Omega - (\beta_\pi - \gamma)$", fontsize=10)
    ax.set_title(r"E. C2 constraint residual: holds at vacuum, "
                   r"fails toward matter", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # ---- F: alpha/beta power-law N^(-2/5) ----
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(Ns, ratios, s=80, c="#bf5b3b", edgecolors="k",
                zorder=5, label=r"$\alpha_\xi/\beta_\pi$ observed")
    N_smooth = np.geomspace(40, 400, 200)
    # power law c * N^(-2/5)
    c_fit = ratios[0] * Ns[0] ** (2/5)
    pred_smooth = c_fit * N_smooth ** (-2/5)
    ax.plot(N_smooth, pred_smooth, "g-", lw=1.5,
             label=r"$N^{-2/5}$ structural fit")
    pred_free = math.exp(rp_a) * N_smooth ** rp_b
    ax.plot(N_smooth, pred_free, "b--", lw=1.0, alpha=0.6,
             label=f"free-fit $N^{{{rp_b:.3f}}}$ "
             f"($R^2$=0.991)")
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel(r"$\alpha_\xi/\beta_\pi$", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40, 400)
    ax.set_title(r"F. $\alpha_\xi/\beta_\pi \sim N^{-2/5}$ "
                  r"clean power-law", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(r"Chirality-flip phase diagram: dynamic "
                   r"$\theta_{\rm chir}(N)$, vacuum--matter "
                   r"transition at $\theta\!=\!\pi/4$ (8-regime "
                   r"ladder $N\in[50,300]$)",
                   fontsize=13, y=0.995)
    out_pdf = FIG_DIR / "fig_chirality_flip_phase_diagram.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Power-law fit: tan^2(theta) ~ N^{pwr_b:.3f}")
    print(f"Predicted N_flip:      {N_flip:.0f}")
    print(f"Predicted N_inversion: {N_inv:.0f}")


if __name__ == "__main__":
    main()
