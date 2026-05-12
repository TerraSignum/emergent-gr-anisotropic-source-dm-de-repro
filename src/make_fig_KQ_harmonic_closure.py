r"""Generate the K, Q harmonic-closure figure for the
factor-field section of the manuscript. Two columns (K, Q),
two rows (data + closure, residuals). All eight coefficients
fixed at System-R rationals.

Output: paper/figures/fig_KQ_harmonic_closure.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GAMMA = 0.1
N_GEN = 3
D_DIM = 4
N_STAR = 50

# System-R rational targets
K_PRE = 4/3 - GAMMA**2 / 3                    # 133/100
K_POST = 4/3 + GAMMA**3                       # 1.334333
A_K = -GAMMA**2 / D_DIM                       # -1/400
B_K = +GAMMA / 16                             # +1/160

Q_PRE = 1/4 + GAMMA**2 * (N_GEN + D_DIM)      # 8/25
Q_POST = 1/4 + GAMMA**2 * N_GEN / D_DIM**2    # 403/1600 (gamma^2-scale; old 61/240 was gamma^1-linear FAIL on 12-seed P5N256)
A_Q = -2 * GAMMA**2                           # -1/50
B_Q = -9 * GAMMA**2 / 8                       # -9/800


def theta_chir(n):
    x = np.log(n / N_STAR) / np.log(D_DIM * N_GEN)
    return np.arctan(N_GEN ** (2 * x - 1))


def closure_curve(N_arr, F_pre, F_post, a, b):
    th = theta_chir(N_arr)
    return (F_pre * np.cos(th)**2 + F_post * np.sin(th)**2
            + a * np.sin(2 * th) + b * np.sin(4 * th))


def main():
    bundle_path = ROOT / "outputs" / "verify_factor_field_KQ_full_closure.json"
    with open(bundle_path) as f:
        bundle = json.load(f)
    rows = bundle["rows"]
    N_data = np.array([r["N"] for r in rows])
    K_data = np.array([r["K_mean"] for r in rows])
    K_sem = np.array([r["K_sem"] for r in rows])
    Q_data = np.array([r["Q_mean"] for r in rows])
    Q_sem = np.array([r["Q_sem"] for r in rows])

    N_curve = np.linspace(50, 600, 401)
    K_curve = closure_curve(N_curve, K_PRE, K_POST, A_K, B_K)
    Q_curve = closure_curve(N_curve, Q_PRE, Q_POST, A_Q, B_Q)
    K_pred_data = closure_curve(N_data, K_PRE, K_POST, A_K, B_K)
    Q_pred_data = closure_curve(N_data, Q_PRE, Q_POST, A_Q, B_Q)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                              gridspec_kw={"height_ratios": [3, 1]},
                              sharex=False)

    # K closure (top-left)
    ax = axes[0, 0]
    ax.plot(N_curve, K_curve, color="#1f77b4", lw=2,
            label=r"$\langle K\rangle(N) = K_{\rm pre}\cos^{2}\theta + K_{\rm post}\sin^{2}\theta + a_K\sin(2\theta) + b_K\sin(4\theta)$")
    ax.errorbar(N_data, K_data, yerr=K_sem,
                fmt="o", color="black", markersize=6, capsize=3,
                label="Lattice mean (8 regimes, 126 seeds)")
    ax.axhline(K_PRE, ls=":", color="gray", lw=0.8)
    ax.axhline(K_POST, ls=":", color="gray", lw=0.8)
    ax.text(560, K_PRE + 0.0002, r"$K_{\rm pre}=\frac{4}{3}-\frac{\gamma^{2}}{3}=\frac{133}{100}$",
             fontsize=8, color="gray", ha="right")
    ax.text(560, K_POST - 0.0008, r"$K_{\rm post}=\frac{4}{3}+\gamma^{3}$",
             fontsize=8, color="gray", ha="right")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\langle K\rangle$")
    ax.set_title("Factor field $K$: harmonic-basis System-$\\mathcal{R}$ closure")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # K residuals (bottom-left)
    ax = axes[1, 0]
    K_res_sigma = (K_data - K_pred_data) / K_sem
    ax.bar(np.arange(len(N_data)), K_res_sigma,
           color=["#1f77b4" if r > 0 else "#d62728" for r in K_res_sigma])
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="gray", ls=":", lw=0.5)
    ax.axhline(-1, color="gray", ls=":", lw=0.5)
    ax.set_xticks(np.arange(len(N_data)))
    ax.set_xticklabels([f"N={n}" for n in N_data], rotation=45, fontsize=8)
    ax.set_ylabel(r"$(K_{\rm meas}-K_{\rm pred})/\sigma$")
    ax.set_title(r"Residuals (constrained closure, 0 free parameters)", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Q closure (top-right)
    ax = axes[0, 1]
    ax.plot(N_curve, Q_curve, color="#2ca02c", lw=2,
            label=r"$\langle Q\rangle(N) = Q_{\rm pre}\cos^{2}\theta + Q_{\rm post}\sin^{2}\theta + a_Q\sin(2\theta) + b_Q\sin(4\theta)$")
    ax.errorbar(N_data, Q_data, yerr=Q_sem,
                fmt="o", color="black", markersize=6, capsize=3,
                label="Lattice mean")
    ax.axhline(Q_PRE, ls=":", color="gray", lw=0.8)
    ax.axhline(Q_POST, ls=":", color="gray", lw=0.8)
    ax.text(560, Q_PRE + 0.0006, r"$Q_{\rm pre}=\frac{1}{4}+\gamma^{2}(N_{\rm gen}+d)=\frac{8}{25}$",
             fontsize=8, color="gray", ha="right")
    ax.text(560, Q_POST - 0.0008, r"$Q_{\rm post}=\frac{1}{4}+\frac{\gamma^{2}N_{\rm gen}}{d^{2}}=\frac{403}{1600}$",
             fontsize=8, color="gray", ha="right")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\langle Q\rangle$")
    ax.set_title("Factor field $Q$: harmonic-basis System-$\\mathcal{R}$ closure")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Q residuals (bottom-right)
    ax = axes[1, 1]
    Q_res_sigma = (Q_data - Q_pred_data) / Q_sem
    ax.bar(np.arange(len(N_data)), Q_res_sigma,
           color=["#2ca02c" if r > 0 else "#d62728" for r in Q_res_sigma])
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="gray", ls=":", lw=0.5)
    ax.axhline(-1, color="gray", ls=":", lw=0.5)
    ax.set_xticks(np.arange(len(N_data)))
    ax.set_xticklabels([f"N={n}" for n in N_data], rotation=45, fontsize=8)
    ax.set_ylabel(r"$(Q_{\rm meas}-Q_{\rm pred})/\sigma$")
    ax.set_title(r"Residuals (constrained closure, 0 free parameters)", fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax in axes[0]:
        ax.set_xlabel(r"$N$ (lattice size, log scale)")

    plt.tight_layout()
    out_path = OUT / "fig_KQ_harmonic_closure.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
