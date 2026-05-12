r"""4-panel robustness-audit figure for the local RG-window
audit of the chirality-flip per-node identity.

Panels:
  A. Bootstrap 95% CIs for AUC and matter-core enrichment ratio
     across the three matter-core indicators.
  B. Leave-one-regime-out: per-fold out-of-sample test AUC, with
     the row-variance of Xi as the held-in candidate.
  C. Permutation null distribution (within-regime theta shuffle)
     vs observed AUC; z-scores annotated.
  D. Threshold scan: matter-core enrichment ratio
     P(C_N|theta>theta_0)/P(C_N|theta<=theta_0) as a function
     of theta_0; framework-predicted threshold pi/4 highlighted.
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
import sys

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))


def main():
    bundle_path = (REPO / "outputs"
                       / "verify_local_RG_window_robustness_audit.json")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    targets = [
        ("vs_t00fw",   "C_N: top-decile $t_{00}(a)$",  "#1f3b6f"),
        ("vs_delta",   "C_N: top-decile $\\Delta(a)$", "#bf5b3b"),
        ("vs_winding", "C_N: $|w(a)|\\!>\\!1/2$",      "#3b8b5a"),
    ]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # Panel A: Bootstrap CIs
    ax = fig.add_subplot(gs[0, 0])
    cis = bundle["A_bootstrap_CIs"]
    auc_pts = []; auc_los = []; auc_his = []
    rat_pts = []; rat_los = []; rat_his = []
    labels = []
    for i, (tk, lab, col) in enumerate(targets):
        if tk not in cis:
            continue
        auc_pts.append(cis[tk]["AUC_point"])
        auc_los.append(cis[tk]["AUC_CI"][0])
        auc_his.append(cis[tk]["AUC_CI"][1])
        rat_pts.append(cis[tk]["ratio_point"])
        rat_los.append(cis[tk]["ratio_CI"][0])
        rat_his.append(cis[tk]["ratio_CI"][1])
        labels.append(lab)
    xs = np.arange(len(labels))
    auc_pts = np.array(auc_pts)
    auc_lo_err = auc_pts - np.array(auc_los)
    auc_hi_err = np.array(auc_his) - auc_pts
    ax.errorbar(xs - 0.13, auc_pts, yerr=[auc_lo_err, auc_hi_err],
                  fmt="o", color="#1f3b6f", markersize=10, capsize=6,
                  label="AUC")
    ax2 = ax.twinx()
    rat_pts_a = np.array(rat_pts)
    rat_lo_err = rat_pts_a - np.array(rat_los)
    rat_hi_err = np.array(rat_his) - rat_pts_a
    ax2.errorbar(xs + 0.13, rat_pts_a,
                   yerr=[rat_lo_err, rat_hi_err],
                   fmt="s", color="#bf5b3b", markersize=10,
                   capsize=6, label="enrichment ratio")
    ax.axhline(0.5, color="grey", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(0.85, color="black", lw=0.6, ls="--", alpha=0.6,
                label=r"$0.85$ promotion gate")
    ax2.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC", color="#1f3b6f", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#1f3b6f")
    ax2.set_ylabel(r"ratio $P(C_{N}|\theta_{\rm loc}>\theta_{\rm glob})/"
                     r"P(C_{N}|\theta_{\rm loc}\leq\theta_{\rm glob})$",
                     color="#bf5b3b", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#bf5b3b")
    ax.set_ylim(0.4, 0.95)
    ax2.set_ylim(0, 16)
    ax.set_title("A. Bootstrap $95\\%$ CIs (regime-relative "
                  r"$\theta_{\rm glob}(N)$ threshold)",
                  fontsize=11)
    ax.grid(alpha=0.3)

    # Panel B: LORO
    ax = fig.add_subplot(gs[0, 1])
    loro = bundle["B_LORO"]
    width = 0.27
    for i, (tk, lab, col) in enumerate(targets):
        if tk not in loro:
            continue
        folds = loro[tk]["folds"]
        regs = sorted(folds.keys())
        test_aucs = [(folds[r][0]["test_AUC"]
                          if folds[r] and folds[r][0]["test_AUC"]
                          is not None else np.nan) for r in regs]
        x = np.arange(len(regs)) + (i - 1) * width
        ax.bar(x, test_aucs, width=width * 0.9, color=col,
                edgecolor="k", alpha=0.85, label=lab)
    ax.axhline(0.85, color="black", lw=0.6, ls="--", alpha=0.6,
                label=r"$0.85$ gate")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.set_xticks(np.arange(len(regs)))
    ax.set_xticklabels([r.replace("d1_", "") for r in regs],
                        rotation=20, fontsize=8, ha="right")
    ax.set_ylabel("LORO test AUC (held-out regime)",
                   fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(r"B. Leave-one-regime-out (8/8 folds with "
                  r"$\Xi$-row-variance as top candidate"
                  r" for $t_{00}$ and $\Delta$ targets)",
                  fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")

    # Panel C: permutation null
    ax = fig.add_subplot(gs[1, 0])
    nc = bundle["C_negative_controls"]
    aucs_real = []; aucs_null = []; zs = []
    labs2 = []
    for tk, lab, col in targets:
        if tk not in nc:
            continue
        aucs_real.append(nc[tk]["AUC_real"])
        aucs_null.append(nc[tk]["AUC_null_mean"])
        zs.append(nc[tk]["z_score"] if nc[tk]["z_score"] else 0.0)
        labs2.append(lab)
    xs = np.arange(len(labs2))
    width = 0.36
    ax.bar(xs - width / 2, aucs_null, width=width, color="grey",
            alpha=0.7, edgecolor="k",
            label="permutation null mean (n_null$\\!=\\!1000$)")
    ax.bar(xs + width / 2, aucs_real, width=width, color="#bf5b3b",
            alpha=0.85, edgecolor="k", label="observed AUC")
    for i, z in enumerate(zs):
        ax.text(i, max(aucs_real[i], aucs_null[i]) + 0.025,
                 f"$z\\!=\\!{z:.1f}$", ha="center",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs2, fontsize=9)
    ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.5)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.4, 0.85)
    ax.set_title("C. Negative controls: within-regime $\\theta$ "
                  "permutation null", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # Panel D: regime-relative threshold scan
    ax = fig.add_subplot(gs[1, 1])
    ts_rel = bundle.get("D_threshold_scan_relative", {})
    for tk, lab, col in targets:
        if tk not in ts_rel:
            continue
        rows = ts_rel[tk]
        deltas_deg = [r["delta_deg"] for r in rows]
        ratios = [r["ratio"] for r in rows]
        ax.plot(deltas_deg, ratios, color=col, lw=2,
                 marker="o", markersize=3, label=lab)
    ax.axvline(0, color="purple", lw=1.0, ls="--", alpha=0.7,
                label=r"$\Delta\!=\!0$: framework "
                       r"$\theta_{\rm glob}(N)$")
    ax.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel(r"offset $\Delta\theta_{0}$ from "
                   r"regime-running $\theta_{\rm glob}(N)$ (deg)",
                   fontsize=11)
    ax.set_ylabel(r"ratio $P(C_{N}|\theta_{\rm loc}\!>\!\theta_{\rm glob}\!+\!\Delta)/"
                   r"P(C_{N}|\leq)$",
                   fontsize=10)
    ax.set_xlim(-25, 25)
    ax.set_ylim(0.5, 80)
    ax.set_yscale("log")
    ax.set_title("D. Regime-relative threshold scan "
                  r"($\theta_{\rm glob}(N)\!+\!\Delta$)",
                  fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Robustness audit of the local RG-window probe "
                  "(8 regimes, 10976 lattice nodes)",
                  fontsize=13, y=1.0)
    out = FIG_DIR / "fig_local_RG_window_robustness_audit.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
