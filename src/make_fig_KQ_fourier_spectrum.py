r"""Figure: Fourier-mode spectrum of K, Q in the chirality angle theta.

Shows the per-mode amplitudes |M_n| = sqrt(c_n^2 + s_n^2) for modes
n = 1, 2, 3, 4 (i.e. 2theta, 4theta, 6theta, 8theta), tested as a
candidate structural identification:

    n=1 (2theta): chirality fundamental
    n=2 (4theta): spacetime / Clifford-frame mode (d = 4)
    n=3 (6theta): generation / matter-side mode (2 N_gen = 6)
    n=4 (8theta): structural integer 2d, partly muted

Output: paper/figures/fig_KQ_fourier_spectrum.pdf
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

GAMMA = 0.1
N_GEN = 3
D_DIM = 4


def main():
    bundle = json.load(open(ROOT / "outputs" / "verify_factor_field_KQ_full_closure.json"))
    rows = bundle["rows"]
    theta = np.array([r["theta_chir"] for r in rows])
    K = np.array([r["K_mean"] for r in rows])
    K_sem = np.array([r["K_sem"] for r in rows])
    Q = np.array([r["Q_mean"] for r in rows])
    Q_sem = np.array([r["Q_sem"] for r in rows])

    # Iteratively fit increasing N_max to extract mode amplitudes and uncertainties
    M_K = np.zeros(4); M_Q = np.zeros(4)
    sM_K = np.zeros(4); sM_Q = np.zeros(4)

    # N_max = 1: extracts M_1 with proper uncertainty
    # N_max = 2: extracts M_2 (re-fit M_1)
    # N_max = 3: extracts M_3
    # N_max = 4: pinv (underdetermined, no uncertainty)
    for N_max in [1, 2, 3]:
        cols = [np.ones_like(theta)]
        for n in range(1, N_max + 1):
            cols.append(np.cos(2 * n * theta))
            cols.append(np.sin(2 * n * theta))
        X = np.column_stack(cols)
        # Fit K and Q
        for tag, y, s in [("K", K, K_sem), ("Q", Q, Q_sem)]:
            W = np.diag(1 / s ** 2)
            cov = np.linalg.inv(X.T @ W @ X)
            beta = cov @ X.T @ W @ y
            sigma = np.sqrt(np.diag(cov))
            n = N_max
            cn = beta[2 * n - 1]; sn = beta[2 * n]
            scn = sigma[2 * n - 1]; ssn = sigma[2 * n]
            amp = np.sqrt(cn ** 2 + sn ** 2)
            # Approx 1-sigma on amplitude via error propagation
            damp_dcn = cn / amp if amp > 0 else 0
            damp_dsn = sn / amp if amp > 0 else 0
            samp = np.sqrt((damp_dcn * scn) ** 2 + (damp_dsn * ssn) ** 2)
            if tag == "K":
                M_K[n - 1] = amp
                sM_K[n - 1] = samp
            else:
                M_Q[n - 1] = amp
                sM_Q[n - 1] = samp

    # N_max = 4 via pinv (no uncertainty)
    cols = [np.ones_like(theta)]
    for n in range(1, 5):
        cols.append(np.cos(2 * n * theta))
        cols.append(np.sin(2 * n * theta))
    X = np.column_stack(cols)
    for tag, y, s in [("K", K, K_sem), ("Q", Q, Q_sem)]:
        Xs = X / s[:, None]
        ys = y / s
        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        cn = beta[7]; sn = beta[8]
        amp = np.sqrt(cn ** 2 + sn ** 2)
        if tag == "K":
            M_K[3] = amp; sM_K[3] = 0
        else:
            M_Q[3] = amp; sM_Q[3] = 0

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    modes = np.array([1, 2, 3, 4])
    integer_labels = [
        "$2\\theta$\nchirality",
        "$4\\theta$\n$d\\!=\\!4$",
        "$6\\theta$\n$2N_{\\rm gen}\\!=\\!6$",
        "$8\\theta$\n$2d\\!=\\!8$",
    ]

    width = 0.35
    ax.bar(modes - width / 2, M_K, width=width, yerr=sM_K, label="K",
           color="#1f77b4", capsize=3)
    ax.bar(modes + width / 2, M_Q, width=width, yerr=sM_Q, label="Q",
           color="#2ca02c", capsize=3)

    # Annotate amplitudes
    for n, mk, mq in zip(modes, M_K, M_Q):
        if n <= 3:
            ax.text(n - width / 2, mk + 0.005, f"{mk:.3f}",
                     ha="center", fontsize=8)
            ax.text(n + width / 2, mq + 0.005, f"{mq:.3f}",
                     ha="center", fontsize=8)

    ax.axvline(3.5, color="gray", ls=":", lw=0.8)
    ax.text(3.5, ax.get_ylim()[1] * 0.95,
             "$\\leftarrow$ statistically determined |  pinv (underdetermined) $\\rightarrow$",
             ha="center", fontsize=8, color="gray")
    ax.set_xticks(modes)
    ax.set_xticklabels(integer_labels)
    ax.set_ylabel("Mode amplitude $|M_n| = \\sqrt{c_n^2+s_n^2}$")
    ax.set_title("Fourier-mode spectrum of $\\langle K\\rangle$, $\\langle Q\\rangle$ in chirality angle $\\theta_{\\rm chir}$")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = OUT / "fig_KQ_fourier_spectrum.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print()
    print("Mode amplitudes (with 1-sigma uncertainties for n<=3):")
    for n in range(4):
        print(f"  n={n+1}: K |M_{n+1}| = {M_K[n]:.4f} ± {sM_K[n]:.4f},  "
              f"Q |M_{n+1}| = {M_Q[n]:.4f} ± {sM_Q[n]:.4f}")


if __name__ == "__main__":
    main()
