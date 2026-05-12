"""Generate fig_nec_sec_margins: per-regime NEC and SEC margin distributions
across the canonical 9-regime ladder."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = REPO / "outputs" / "lambda_anisotropy_NEC_SEC_robustness.json"
    if not p.exists():
        p = REPO / "data" / "lambda_anisotropy_NEC_SEC_robustness.json"
    with open(p) as f:
        d = json.load(f)
    pr = d["per_regime"]
    Ns = np.array(pr["N_values"], dtype=float)
    nec_mean = np.array(pr["NEC_margin_mean"], dtype=float)
    nec_std = np.array(pr["NEC_margin_std"], dtype=float)
    sec_mean = np.array(pr["SEC_margin_mean"], dtype=float)
    sec_std = np.array(pr["SEC_margin_std"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.errorbar(Ns, nec_mean, yerr=nec_std, fmt="o", markersize=8,
                  color="#3c6ea7", ecolor="#3c6ea7", capsize=3,
                  markeredgecolor="black", linewidth=0.6,
                  label=r"$\rho+p$ per regime")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(np.mean(nec_mean), color="#d97f4a", linestyle="--",
                 linewidth=1.2, label=f"pooled mean $\\approx{np.mean(nec_mean):.3f}$")
    ax1.set_xlabel("$N$"); ax1.set_ylabel(r"$\rho + p$")
    ax1.set_title("NEC margin: strongly satisfied (non-phantom solid, $1666\\sigma$ above 0)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.errorbar(Ns, sec_mean, yerr=sec_std, fmt="s", markersize=8,
                  color="#5a3010", ecolor="#5a3010", capsize=3,
                  markeredgecolor="black", linewidth=0.6,
                  label=r"$\rho+3p$ per regime")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(np.mean(sec_mean), color="#d97f4a", linestyle="--",
                 linewidth=1.2, label=f"pooled mean $\\approx{np.mean(sec_mean):.3f}$")
    ax2.set_xlabel("$N$"); ax2.set_ylabel(r"$\rho + 3p$")
    ax2.set_title("SEC margin: satisfied with $7.5\\sigma$ above 0")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT / "fig_nec_sec_margins.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
