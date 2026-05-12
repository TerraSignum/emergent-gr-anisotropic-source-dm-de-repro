r"""4-Panel Fast-Slow Decomposition Figure for D_Omega(N) running.

Maps to Paper 03 Feldtheorie-Notebook §4.1.1 Fast-Slow-Struktur:
  partial_t Xi = partial_t Xi|fast + epsilon * partial_t Xi|slow

Panels:
  A: theta_chir(N) running with vacuum/inversion endpoints + flip
     point at theta=pi/4
  B: Slow envelope D_Omega^slow(N) = (67/80)cos² + (pi/d)sin²
     transitioning from vacuum 67/80 to matter pi/d
  C: Fast oscillation D_Omega^fast(N) period-d=4 in log_2(N)
     with peak at phi=3 (deepest dip)
  D: Total D_Omega(N) = slow - fast: 8 lattice points + predictions
     at N=256, 512, ..., 4096
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
N_STAR = 50
D_OMEGA_VACUUM = 67/80
D_OMEGA_MATTER = PI / D


def theta_chir(N):
    if N <= 0:
        return 0
    x = math.log(N / N_STAR) / math.log(D * N_GEN)
    return math.atan(N_GEN ** (2 * x - 1))


def D_Omega_slow(N):
    th = theta_chir(N)
    return D_OMEGA_VACUUM * math.cos(th) ** 2 + \
            D_OMEGA_MATTER * math.sin(th) ** 2


def fast_bump(phi, peak_phase=3.0, power=2):
    if phi <= peak_phase:
        f = phi / peak_phase
    else:
        f = (D - phi) / (D - peak_phase)
    f = max(0, min(1, f))
    return f ** power


def D_Omega_fast(N, amplitude=0.55):
    log2_N = math.log2(N)
    phi = log2_N % D
    return amplitude * fast_bump(phi)


def main():
    src = DATA / "causal_wave_per_N_readout.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data["p5_ladder_per_N_readout"]

    Ns = np.array([r["n_lat"] for r in rows])
    DOs = np.array([r["D_omega_lattice"] for r in rows])

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # ---- A: theta_chir(N) running ----
    ax = fig.add_subplot(gs[0, 0])
    N_grid = np.geomspace(40, 5000, 400)
    theta_grid = np.array([math.degrees(theta_chir(N)) for N in N_grid])
    ax.plot(N_grid, theta_grid, "g-", lw=2, label="$\\theta_{\\rm chir}(N)$ (slow)")
    ax.scatter(Ns, [math.degrees(theta_chir(N)) for N in Ns],
                s=80, c="#bf5b3b", edgecolors="k", zorder=5,
                label="lattice data")
    ax.axhline(45, color="purple", lw=0.8, ls=":",
                label="$\\theta\\!=\\!\\pi/4$ (vacuum-matter flip)")
    ax.axhline(math.degrees(math.atan(1/N_GEN)), color="orange",
                lw=0.8, ls=":",
                label="$\\arctan(1/N_{\\rm gen})\\!=\\!18.4^\\circ$ (vacuum)")
    ax.axhline(math.degrees(math.atan(N_GEN)), color="green",
                lw=0.8, ls=":",
                label="$\\arctan(N_{\\rm gen})\\!=\\!71.6^\\circ$ (inversion)")
    ax.axvline(D * N_GEN * N_STAR, color="red", lw=0.8, ls="--",
                alpha=0.5)
    ax.text(D * N_GEN * N_STAR + 50, 30, "$N_{\\rm inv}\\!=\\!600$",
             color="red", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(40, 5000)
    ax.set_ylim(0, 90)
    ax.set_xlabel("$N$", fontsize=10)
    ax.set_ylabel("$\\theta_{\\rm chir}$ (deg)", fontsize=10)
    ax.set_title("A. SLOW chirality angle running $\\theta_{\\rm chir}(N)$",
                  fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # ---- B: Slow envelope ----
    ax = fig.add_subplot(gs[0, 1])
    slow_grid = np.array([D_Omega_slow(N) for N in N_grid])
    ax.plot(N_grid, slow_grid, "b-", lw=2,
             label="$D_\\Omega^{\\rm slow}\\!=\\!\\frac{67}{80}\\cos^2\\theta\\!+\\!\\frac{\\pi}{d}\\sin^2\\theta$")
    ax.scatter(Ns, [D_Omega_slow(N) for N in Ns],
                s=60, c="#3b8bbf", edgecolors="k", zorder=5)
    ax.axhline(D_OMEGA_VACUUM, color="purple", lw=0.8, ls=":",
                label="$67/80\\!=\\!0.838$ (vacuum)")
    ax.axhline(D_OMEGA_MATTER, color="green", lw=0.8, ls=":",
                label="$\\pi/d\\!=\\!0.785$ (matter)")
    ax.axvline(D * N_GEN * N_STAR, color="red", lw=0.8, ls="--",
                alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlim(40, 5000)
    ax.set_ylim(0.77, 0.85)
    ax.set_xlabel("$N$", fontsize=10)
    ax.set_ylabel("$D_\\Omega^{\\rm slow}(N)$", fontsize=10)
    ax.set_title("B. SLOW chirality-mixing envelope", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # ---- C: Fast oscillation ----
    ax = fig.add_subplot(gs[1, 0])
    log2_grid = np.linspace(5.5, 13, 600)
    fast_grid = np.array([0.55 * fast_bump(l % D) for l in log2_grid])
    N_for_log = np.power(2, log2_grid)
    ax.plot(N_for_log, fast_grid, "r-", lw=2,
             label="$D_\\Omega^{\\rm fast}\\!=\\!A\\,f(\\log_2(N)\\! \\mathrm{mod}\\!d)$")
    # Mark period boundaries (mod d = 0)
    for k in [6, 8, 10, 12]:
        N_b = 2 ** k
        ax.axvline(N_b, color="purple", lw=0.6, ls=":", alpha=0.5)
        ax.text(N_b * 1.1, 0.05,
                 f"$N\\!=\\!2^{{{k}}}$\nmod=0", fontsize=7)
    ax.scatter(Ns, [0.55 * fast_bump(math.log2(N) % D) for N in Ns],
                s=60, c="#bf5b3b", edgecolors="k", zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(40, 9000)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("$N$", fontsize=10)
    ax.set_ylabel("$D_\\Omega^{\\rm fast}$ deviation", fontsize=10)
    ax.set_title("C. FAST lattice-harmonic oscillation (period $d\\!=\\!4$)",
                  fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # ---- D: Total D_Omega(N) with predictions ----
    ax = fig.add_subplot(gs[1, 1])
    total_grid = np.array([D_Omega_slow(N) - D_Omega_fast(N)
                              for N in N_grid])
    ax.plot(N_grid, total_grid, "k-", lw=1.5, alpha=0.5,
             label="$D_\\Omega^{\\rm pred}\\!=\\!{\\rm slow}\\!-\\!{\\rm fast}$")
    # Existing 8-point lattice data
    ax.scatter(Ns, DOs, s=100, c="#bf5b3b", edgecolors="k",
                zorder=5, label="lattice data (8 pts)")
    # Predictions at higher N
    pred_Ns = [256, 512, 1024, 2048, 3072, 4096]
    pred_DOs = [D_Omega_slow(N) - D_Omega_fast(N) for N in pred_Ns]
    ax.scatter(pred_Ns, pred_DOs, s=100, marker="*",
                c="#3b8bbf", edgecolors="k", zorder=5,
                label="predictions (untested)")
    for N, do in zip(pred_Ns, pred_DOs):
        ax.text(N, do + 0.04, f"$N\\!=\\!{N}$\n${do:.2f}$",
                 fontsize=7, ha="center")
    ax.axhline(D_OMEGA_VACUUM, color="purple", lw=0.6, ls=":",
                label="vacuum 67/80")
    ax.axhline(D_OMEGA_MATTER, color="green", lw=0.6, ls=":",
                label="matter $\\pi/d$")
    ax.set_xscale("log")
    ax.set_xlim(40, 9000)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("$N$", fontsize=10)
    ax.set_ylabel("$D_\\Omega(N)$", fontsize=10)
    ax.set_title("D. Total $D_\\Omega\\!=\\!{\\rm slow}\\!-\\!{\\rm fast}$ "
                   "+ 6 predictions", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Fast-Slow Decomposition of $D_\\Omega(N)$ "
                   "(Paper 03 §4.1.1 framework)",
                   fontsize=13, y=0.995)
    out_pdf = FIG_DIR / "fig_fast_slow_D_Omega_decomposition.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
