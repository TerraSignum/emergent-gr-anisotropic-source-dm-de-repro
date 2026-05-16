r"""Iter-36 figure: first-principles theta_chir(N) running +
higher-N predictions + p95 CIs + D_Omega non-monotonic dips.

4-panel composite:
  A: theta_chir(N) data + first-principles structural form
     tan(theta) = N_gen^(2x-1) extending to N=1000
     with bootstrap p95 CI band
  B: higher-N predictions for alpha_xi(N), beta_pi(N) extending
     to N=2000, marking N_inversion = d*N_gen*50 = 600
  C: D_Omega(N) per-regime with chirality-mix prediction
     overlaid; dips at N=84, 128 highlighted as resonance
  D: alpha/beta ~ N^(-2/5) bootstrap distribution + p95 CI band
"""
from __future__ import annotations

import json
import math
import random
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


def linfit(x_list, y_list):
    n = len(x_list)
    mx = sum(x_list) / n
    my = sum(y_list) / n
    sxy = sum((x_list[i] - mx) * (y_list[i] - my) for i in range(n))
    sxx = sum((x_list[i] - mx) ** 2 for i in range(n))
    if sxx < 1e-30:
        return None
    b = sxy / sxx
    a = my - b * mx
    return a, b


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

    # First-principles ansatz constants
    ln_d_Ngen = math.log(D * N_GEN)
    a_vac = (2 ** D * N_GEN ** 2 - 1) / (2 ** D * N_GEN ** 2)
    a_mat = (2 * D * N_GEN - 1) / (4 * D * N_GEN)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)

    # ---- A: theta_chir running with first-principles ansatz ----
    ax = fig.add_subplot(gs[0, 0])
    N_smooth = np.geomspace(40, 2000, 300)
    x_frac = np.log(N_smooth / 50.0) / ln_d_Ngen
    tan_pred = N_GEN ** (2 * x_frac - 1)
    theta_pred = np.degrees(np.arctan(tan_pred))
    # Bootstrap CI on the running
    rng = random.Random(7)
    log_N = np.log(Ns)
    theta_deg = np.degrees(thetas)
    n_boot = 2000
    theta_boot = []
    for _ in range(n_boot):
        idx = [rng.randint(0, len(Ns) - 1) for _ in range(len(Ns))]
        bN = log_N[idx]
        bT = theta_deg[idx]
        if len(set(bN)) < 2:
            continue
        result = linfit(bN.tolist(), bT.tolist())
        if result is None:
            continue
        a_b, b_b = result
        theta_boot.append([a_b + b_b * math.log(N) for N in N_smooth])
    theta_boot = np.array(theta_boot)
    if len(theta_boot) > 0:
        p2_5 = np.percentile(theta_boot, 2.5, axis=0)
        p97_5 = np.percentile(theta_boot, 97.5, axis=0)
        ax.fill_between(N_smooth, p2_5, p97_5, alpha=0.2,
                          color="green", label="bootstrap p95 CI")
    ax.plot(N_smooth, theta_pred, "g-", lw=1.7,
             label=r"$\tan\theta\!=\!N_{\rm gen}^{2x-1}$ "
             r"(first principles)")
    ax.scatter(Ns, theta_deg, s=80, c="#bf5b3b", edgecolors="k",
                zorder=5, label="lattice data")
    ax.axhline(45, color="purple", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(math.degrees(math.atan(N_GEN)), color="green",
                lw=0.8, ls=":", alpha=0.7)
    ax.axvline(600, color="red", lw=0.8, ls="--", alpha=0.7)
    ax.text(620, 30, r"$N_{\rm inv}\!=\!d\cdot N_{\rm gen}\cdot N_*\!=\!600$",
             color="red", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(40, 2000)
    ax.set_ylim(0, 90)
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel(r"$\theta_{\rm chir}$ (deg)", fontsize=10)
    ax.set_title(r"A. First-principles $\theta_{\rm chir}(N)$ "
                  r"running + p95 CI", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # ---- B: alpha_xi, beta_pi higher-N predictions ----
    ax = fig.add_subplot(gs[0, 1])
    alpha_pred_smooth = np.cos(np.radians(theta_pred)) ** 2
    gamma_pred_smooth = np.sin(np.radians(theta_pred)) ** 2
    beta_pred_smooth = (a_vac * alpha_pred_smooth +
                          a_mat * gamma_pred_smooth)
    ax.plot(N_smooth, alpha_pred_smooth, "r-", lw=1.5,
             label=r"$\alpha_\xi(N)$ predicted")
    ax.plot(N_smooth, beta_pred_smooth, "b-", lw=1.5,
             label=r"$\beta_\pi(N)$ predicted")
    ax.scatter(Ns, alphas, s=70, marker="o", c="#bf5b3b",
                edgecolors="k", zorder=5, label=r"$\alpha_\xi$ data")
    ax.scatter(Ns, betas, s=70, marker="s", c="#3b8bbf",
                edgecolors="k", zorder=5, label=r"$\beta_\pi$ data")
    ax.axhline(0.1, color="r", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(0.5, color="b", lw=0.6, ls=":", alpha=0.6)
    ax.axvline(600, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.text(620, 0.95, r"$N_{\rm inv}\!=\!600$", color="purple",
             fontsize=8)
    ax.text(2050, 0.10, r"$\gamma_{\rm canonical}$", color="r",
             fontsize=8)
    ax.text(2050, 0.49, r"$1/2$ asymp", color="b", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(40, 2200)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel("coefficient", fontsize=10)
    ax.set_title(r"B. Higher-$N$ predictions: $\alpha_\xi$, "
                  r"$\beta_\pi$ to $N\!=\!2000$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # ---- C: D_Omega non-monotonic + chirality-mix overlay ----
    ax = fig.add_subplot(gs[1, 0])
    # Chirality-mix prediction for D_Omega: 67/80*alpha + pi/4*gamma
    DO_chirality_smooth = (67/80) * alpha_pred_smooth + \
                            (PI/4) * gamma_pred_smooth
    ax.plot(N_smooth, DO_chirality_smooth, "g-", lw=1.5,
             label=r"$\frac{67}{80}\cos^2 + \frac{\pi}{4}\sin^2$ "
             r"(naive chirality-mix)")
    ax.scatter(Ns, DOs, s=80, c="#bf5b3b", edgecolors="k",
                zorder=5, label=r"$D_\Omega(N)$ data")
    # Mark the dips
    dip_indices = [i for i, n in enumerate(Ns)
                    if n in (84, 128)]
    for i in dip_indices:
        ax.annotate(f"$N\\!=\\!{Ns[i]}$ dip\n("
                     f"$2^{int(math.log2(Ns[i]&-Ns[i]))}\\cdot$"
                     f"{int(Ns[i]/(Ns[i]&-Ns[i]))})",
                     xy=(Ns[i], DOs[i]),
                     xytext=(Ns[i] * 0.7, DOs[i] - 0.15),
                     fontsize=8,
                     arrowprops={"arrowstyle": "->",
                                  "color": "red",
                                  "lw": 0.8})
    ax.set_xscale("log")
    ax.set_xlim(40, 400)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$N$", fontsize=10)
    ax.set_ylabel(r"$D_\Omega(N)$", fontsize=10)
    ax.set_title(r"C. $D_\Omega(N)$: lattice-resonance dips "
                  r"at $N\!=\!84,\,128\!=\!2^7$", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # ---- D: alpha/beta ratio bootstrap ----
    ax = fig.add_subplot(gs[1, 1])
    ratios = alphas / betas
    log_r = np.log(ratios)
    log_N_arr = np.log(Ns)
    rng2 = random.Random(13)
    boot_b = []
    boot_a = []
    for _ in range(n_boot):
        idx = [rng2.randint(0, len(Ns) - 1) for _ in range(len(Ns))]
        bN = log_N_arr[idx]
        bR = log_r[idx]
        if len(set(bN)) < 2:
            continue
        result = linfit(bN.tolist(), bR.tolist())
        if result is None:
            continue
        boot_a.append(result[0])
        boot_b.append(result[1])
    # Plot bootstrap distribution + p95 + p99
    ax.hist(boot_b, bins=50, density=True, alpha=0.6, color="#3b8bbf",
             edgecolor="k", label="bootstrap dist")
    p2_5 = np.percentile(boot_b, 2.5)
    p97_5 = np.percentile(boot_b, 97.5)
    p0_5 = np.percentile(boot_b, 0.5)
    p99_5 = np.percentile(boot_b, 99.5)
    p50 = np.percentile(boot_b, 50)
    ax.axvline(-0.4, color="g", lw=2.0, ls="-",
                label=r"target $-2/5$")
    ax.axvline(p50, color="b", lw=1.5, ls="--",
                label=f"median ${p50:.3f}$")
    ax.axvspan(p2_5, p97_5, alpha=0.15, color="green",
                label=f"p95 [{p2_5:.3f}, {p97_5:.3f}]")
    ax.axvspan(p0_5, p99_5, alpha=0.07, color="orange",
                label=f"p99 [{p0_5:.3f}, {p99_5:.3f}]")
    target_in_p95 = p2_5 <= -0.4 <= p97_5
    ax.set_xlabel(r"exponent $b$ in $\alpha_\xi/\beta_\pi \sim N^b$",
                    fontsize=10)
    ax.set_ylabel("bootstrap density", fontsize=10)
    ax.set_title(r"D. $\alpha_\xi/\beta_\pi$ exponent bootstrap "
                  r"(target $-2/5$ inside p95: "
                  f"{'YES' if target_in_p95 else 'NO'})",
                  fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(r"Iter-36: first-principles $\theta_{\rm chir}(N)$ "
                   r"running, higher-$N$ predictions, and "
                   r"$D_\Omega$ resonance",
                   fontsize=13, y=0.995)
    out_pdf = FIG_DIR / "fig_chirality_higher_N_predictions.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
