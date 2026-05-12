"""Generate fig_KQ_top5_closure.pdf — matter-localised top-5%
closure of <K>_top5(N), <Q>_top5(N) with the parameter-free
chirality-harmonic + lattice-resonance closure overlay.

Two-panel figure:
  Top: lattice top-5% means +/- per-seed SEM as filled markers
       on the canonical-physics ladder N in [64, 512];
       the closure curve from Eq. (KQ_top5_closure) plotted as
       smooth lines through the eight data points; the
       lattice-resonance contribution v_2(N)/240 shown as a
       separate dashed overlay for Q.
  Bottom: per-regime residuals z_K, z_Q in seed-error units.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "paper" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

GAMMA = 0.1
N_GEN = 3
D_DIM = 4
N_STAR = 50


def theta_chir(n_lat):
    x = np.log(n_lat / N_STAR) / np.log(D_DIM * N_GEN)
    return np.arctan(N_GEN ** (2 * x - 1))


def v2_of_N(n):
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def main():
    bundle_path = ROOT / "outputs" / "verify_KQ_top5_full_structural_closure.json"
    bundle = json.load(open(bundle_path))
    rows = bundle["rows"]
    # Order: canonical P5/P5N family by N first, then alt-anchor
    # (P6/P7/P8) family by N. Never lump alt-anchor regimes into the
    # canonical N-ordered ladder.
    _canonical = sorted(
        [r for r in rows if r["regime"].startswith("P5")],
        key=lambda r: r["N"])
    _alt = sorted(
        [r for r in rows if r["regime"].startswith(("P6", "P7", "P8"))],
        key=lambda r: r["N"])
    rows = _canonical + _alt
    N_arr = np.array([r["N"] for r in rows])
    K_obs = np.array([r["K_mean"] for r in rows])
    K_sem = np.array([r["K_sem"] for r in rows])
    Q_obs = np.array([r["Q_mean"] for r in rows])
    Q_sem = np.array([r["Q_sem"] for r in rows])
    K_pred = np.array([r["K_predicted"] for r in rows])
    Q_pred = np.array([r["Q_predicted"] for r in rows])
    z_K = np.array([r["K_residual_z"] for r in rows])
    z_Q = np.array([r["Q_residual_z"] for r in rows])

    A_K = 5.0 / 4.0
    B_K = 4.0 / 3.0 - GAMMA ** 2
    A_Q = 1.0 / (N_GEN ** 2)
    B_Q = 2.0 / (D_DIM ** 2 - 1)
    S_Q = 2.0 / (D_DIM ** 2)
    V_Q = 1.0 / (D_DIM ** 2 * (D_DIM ** 2 - 1))

    # Smooth curves for closure (without v_2 jitter, evaluated at integer N)
    N_smooth = np.arange(50, 600)
    th_smooth = theta_chir(N_smooth)
    K_smooth = A_K * np.cos(th_smooth) ** 2 + B_K * np.sin(th_smooth) ** 2
    Q_smooth_no_v2 = (A_Q * np.cos(th_smooth) ** 2
                      + B_Q * np.sin(th_smooth) ** 2
                      + S_Q * np.sin(2 * th_smooth))
    # v_2(N) overlay only at integer N
    Q_v2_overlay = V_Q * np.array([v2_of_N(n) for n in N_smooth], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                              gridspec_kw={"height_ratios": [3, 1]},
                              sharex=True)

    ax = axes[0]
    ax.errorbar(N_arr, K_obs, yerr=K_sem, fmt="o", color="C0",
                 ms=8, capsize=3, label=r"$\langle K\rangle_{\rm top5}$ data")
    ax.plot(N_smooth, K_smooth, "-", color="C0", alpha=0.6, lw=1.6,
             label=r"$\frac{5}{4}\cos^{2}\theta + (\frac{4}{3}-\gamma^{2})\sin^{2}\theta$")
    ax.errorbar(N_arr, Q_obs, yerr=Q_sem, fmt="s", color="C3",
                 ms=8, capsize=3, label=r"$\langle Q\rangle_{\rm top5}$ data")
    ax.plot(N_smooth, Q_smooth_no_v2 + Q_v2_overlay, "-", color="C3",
             alpha=0.6, lw=1.6,
             label=r"$\frac{1}{9}\cos^{2}\!+\!\frac{2}{15}\sin^{2}\!+\!\frac{1}{8}\sin 2\theta\!+\!\frac{1}{240}v_{2}(N)$")
    ax.plot(N_smooth, Q_smooth_no_v2, "--", color="C3",
             alpha=0.35, lw=1.0,
             label=r"$Q$ closure without $v_{2}(N)$")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\langle K\rangle_{\rm top5}, \langle Q\rangle_{\rm top5}$")
    ax.set_title(r"Matter-localised top-5\% closure of $\langle K\rangle, \langle Q\rangle$"
                 r" — zero free parameters in $(\gamma, N_{\rm gen}, d)$")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    # Annotate v_2 peak/drop/recovery
    for n_label, v2_label in [(256, "$v_{2}=8$"), (300, "$v_{2}=2$"), (512, "$v_{2}=9$")]:
        idx = int(np.argmin(np.abs(N_arr - n_label)))
        ax.annotate(v2_label, xy=(N_arr[idx], Q_obs[idx]),
                     xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=7, color="C3", alpha=0.8)

    ax2 = axes[1]
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axhspan(-1, 1, color="0.85", alpha=0.5, label=r"$|z|\le 1$")
    ax2.axhspan(-2, -1, color="0.92", alpha=0.5)
    ax2.axhspan(1, 2, color="0.92", alpha=0.5)
    ax2.errorbar(N_arr, z_K, fmt="o", color="C0", ms=7, label=r"$z_{K}$")
    ax2.errorbar(N_arr, z_Q, fmt="s", color="C3", ms=7, label=r"$z_{Q}$")
    ax2.set_xscale("log")
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_xlabel(r"Lattice size $N$")
    ax2.set_ylabel(r"residual $z = (\rm obs - pred)/\sigma$")
    ax2.grid(alpha=0.3)

    chi2_2N = bundle["joint_metric"]["joint_chi2_per_2N_reg"]
    ax.text(0.02, 0.04, f"joint $\\chi^{{2}}/(2N_{{\\rm reg}}) = {chi2_2N:.3f}$",
             transform=ax.transAxes, fontsize=9, color="0.2",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()
    out_pdf = OUT_FIG / "fig_KQ_top5_closure.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
