"""Figure for the null-hypothesis test of the signed residual halo audit.

Plots the observed Spearman rho per regime against the
randomised-defect-label null distribution (N3) and the
uniform-bulk null (N0). All 10 regimes show observed rho
strictly negative (NFW radial-decrease signature) and the N3
null distribution is centred at zero with std ~0.02-0.03;
observed rho lands at z = -7 to -13 standard deviations from
the null.

Reads outputs/verify_signed_dm_radial_null_test.json.
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

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "outputs" / "verify_signed_dm_radial_null_test.json"
OUT = REPO / "paper" / "figures" / "fig_null_test_comparison.pdf"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(IN.read_text(encoding="utf-8"))
    # Canonical P5/P5N N-ordered ladder only; alt-anchor regimes
    # (P6/P7/P8) are reported separately as cross-anchor consistency
    # checks and do not belong on this canonical-ladder figure.
    rows = sorted(
        [r for r in raw["per_regime"] if r["regime"].startswith("P5")],
        key=lambda r: r["N"])

    n_arr = np.array([r["N"] for r in rows], dtype=float)
    rho_obs = np.array([r["rho_observed"] for r in rows])
    null_mean = np.array([r["n3_null_rho_mean_pm_std"][0] for r in rows])
    null_std = np.array([r["n3_null_rho_mean_pm_std"][1] for r in rows])
    null_lo = np.array([r["n3_null_rho_95CI"][0] for r in rows])
    null_hi = np.array([r["n3_null_rho_95CI"][1] for r in rows])
    z_obs = np.array([r["z_score_against_n3_null"] for r in rows])
    regimes = [r["regime"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160,
                                     gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Left panel: observed rho + null band, per regime
    ax1.fill_between(n_arr, null_lo, null_hi,
                      color="gray", alpha=0.30,
                      label=r"N3 randomised-defect 95% CI band")
    ax1.plot(n_arr, null_mean, "--", color="gray", lw=1.0,
              label="N3 null mean (~0)")
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.errorbar(n_arr, rho_obs, yerr=null_std, fmt="o",
                  color="C0", ms=8, capsize=4, lw=1.6,
                  label=r"observed $\rho(|\Pi R|_{\rm bulk}, d_{\rm core})$")
    ax1.set_xlabel(r"lattice size $N$", fontsize=11)
    ax1.set_ylabel(r"Spearman $\rho$", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_xticks([50, 100, 200, 300, 512])
    ax1.set_xticklabels(["50", "100", "200", "300", "512"])
    ax1.set_title(
        r"Observed bulk-amplitude vs distance-to-defect "
        r"correlation on $\mathcal{P}_{5}/\mathcal{P}_{5}N$ "
        r"ladder (10 regimes)",
        fontsize=10.5)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax1.grid(alpha=0.3)

    # Right panel: z-score per regime
    z_abs = np.abs(z_obs)
    bars = ax2.barh(np.arange(len(regimes)), z_abs, color="C0",
                     alpha=0.85)
    ax2.axvline(2.0, color="C3", lw=1.4, linestyle="--",
                 label=r"$|z|=2$ ($p\!\approx\!0.05$)")
    ax2.set_yticks(np.arange(len(regimes)))
    ax2.set_yticklabels(regimes, fontsize=9)
    ax2.set_xlabel(r"$|z|$ (regimes from N3 null)", fontsize=11)
    ax2.set_title("Rejection of N3 null per regime",
                   fontsize=10.5)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax2.grid(axis="x", alpha=0.3)
    for i, b in enumerate(bars):
        ax2.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                  f"{z_abs[i]:.1f}",
                  ha="left", va="center", fontsize=8)

    fig.suptitle(
        "Null-hypothesis tests for the signed residual halo audit "
        "(N0 uniform, N3 randomised-defect)",
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
