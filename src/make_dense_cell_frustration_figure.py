"""Three-panel visualisation of the dense-cell graph-frustration
vs.\\ R_00 halo cross-relation across the canonical-physics ladder.

Reads the per-regime audit
\\verb|outputs/verify_dense_cell_frustration_canonical_ladder.json|
and emits the three-panel figure
\\verb|paper/figures/fig_dense_cell_frustration.pdf|:

  (a) f_{delta < 0} (graph M3 over-saturation share) vs N with
      Symanzik 1/N fit and extrapolation to N -> infinity at 1.012.
  (b) f_{R_00 < 0} (energy-density halo node fraction) vs N, with
      the across-regime mean and stability band.
  (c) Scatter of (f_{delta<0}, f_{R_00<0}) per regime with the
      cross-regime Pearson correlation r = -0.68 anticorrelation
      regression line and per-point regime labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "verify_dense_cell_frustration_canonical_ladder.json"
OUT = REPO / "paper" / "figures" / "fig_dense_cell_frustration.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    # Order: canonical P5/P5N family by N first, then alt-anchor
    # (P6/P7/P8) family by N. Never lump alt-anchor regimes into the
    # canonical N-ordered ladder.
    _canonical = sorted(
        [r for r in d["per_regime"] if r["regime"].startswith("P5")],
        key=lambda r: r["N"])
    _alt = sorted(
        [r for r in d["per_regime"]
         if r["regime"].startswith(("P6", "P7", "P8"))],
        key=lambda r: r["N"])
    rows = _canonical + _alt
    s = d["summary"]

    n_vals = np.array([r["N"] for r in rows], dtype=float)
    nts = np.array([r["neg_triangle_share_mean"] for r in rows])
    nts_err = np.array([r["neg_triangle_share_std"] for r in rows])
    fneg = np.array([r["f_neg_R00_mean"] for r in rows])
    fneg_err = np.array([r["f_neg_R00_std"] for r in rows])
    labels = [r["regime"] for r in rows]

    nts_a0 = s["neg_triangle_share_symanzik_intercept"]
    nts_a1 = s["neg_triangle_share_symanzik_slope"]
    fneg_a0 = s["f_neg_R00_symanzik_intercept"]
    fneg_a1 = s["f_neg_R00_symanzik_slope"]

    fig = plt.figure(figsize=(15, 4.4))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    # Panel (a): f_{delta<0} vs N with Symanzik 1/N fit
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(n_vals, nts, yerr=nts_err, fmt="o", ms=7,
                  capsize=3, color="#1f3b6f", label="lattice mean")
    n_grid = np.linspace(40, 1000, 200)
    nts_pred = nts_a0 + nts_a1 / n_grid
    ax_a.plot(n_grid, nts_pred, "--", color="#a45a5a",
              label=f"Symanzik $1/N$ fit (asymp.\\ {nts_a0:.3f})")
    ax_a.axhline(1.0, color="black", linestyle=":", linewidth=1.2,
                 alpha=0.5, label="universal-frustration limit")
    ax_a.set_xlabel("$N$ (lattice size)", fontsize=11)
    ax_a.set_ylabel("$f_{\\delta_{ijk}<0}$", fontsize=11)
    ax_a.set_title("(a) Graph M3 over-saturation share",
                   fontsize=12)
    ax_a.set_ylim(0.78, 1.02)
    ax_a.set_xscale("log")
    ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    for x, y, lab in zip(n_vals, nts, labels):
        ax_a.annotate(lab, (x, y), fontsize=7,
                      xytext=(4, -10), textcoords="offset points",
                      color="#444")

    # Panel (b): f_{R_00<0} vs N
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.errorbar(n_vals, fneg, yerr=fneg_err, fmt="s", ms=7,
                  capsize=3, color="#2e7d32",
                  label="lattice mean")
    fneg_pred = fneg_a0 + fneg_a1 / n_grid
    ax_b.plot(n_grid, fneg_pred, "--", color="#a45a5a",
              label=f"Symanzik $1/N$ fit (asymp.\\ {fneg_a0:.3f})")
    fneg_mean = float(np.mean(fneg))
    ax_b.axhline(fneg_mean, color="black", linestyle=":",
                 linewidth=1.2, alpha=0.5,
                 label=f"across-regime mean {fneg_mean:.3f}")
    ax_b.fill_between([40, 1000],
                      fneg_mean - float(np.std(fneg)),
                      fneg_mean + float(np.std(fneg)),
                      color="grey", alpha=0.10)
    ax_b.set_xlabel("$N$ (lattice size)", fontsize=11)
    ax_b.set_ylabel("$f_{R_{00}<0}$", fontsize=11)
    ax_b.set_title("(b) Per-node energy-density halo fraction",
                   fontsize=12)
    ax_b.set_ylim(0.55, 0.90)
    ax_b.set_xscale("log")
    ax_b.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    for x, y, lab in zip(n_vals, fneg, labels):
        ax_b.annotate(lab, (x, y), fontsize=7,
                      xytext=(4, 6), textcoords="offset points",
                      color="#444")

    # Panel (c): scatter f_{delta<0} vs f_{R_00<0}
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(nts, fneg, s=80, c=n_vals, cmap="viridis",
                 edgecolors="black", linewidths=0.6, zorder=3)
    coef = np.polyfit(nts, fneg, 1)
    x_line = np.linspace(nts.min() - 0.01, nts.max() + 0.01, 50)
    y_line = np.polyval(coef, x_line)
    pearson = float(np.corrcoef(nts, fneg)[0, 1])
    ax_c.plot(x_line, y_line, "--", color="#a45a5a",
              label=(f"linear fit, "
                     f"slope $={coef[0]:+.2f}$\n"
                     f"Pearson $r={pearson:+.3f}$"))
    ax_c.set_xlabel("$f_{\\delta_{ijk}<0}$ (graph frustration)",
                    fontsize=11)
    ax_c.set_ylabel("$f_{R_{00}<0}$ (energy halo)", fontsize=11)
    ax_c.set_title("(c) Cross-regime anticorrelation",
                   fontsize=12)
    ax_c.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    for x, y, lab in zip(nts, fneg, labels):
        ax_c.annotate(lab, (x, y), fontsize=7,
                      xytext=(5, -10), textcoords="offset points",
                      color="#444")
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=n_vals.min(), vmax=n_vals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_c, shrink=0.85, pad=0.02)
    cbar.set_label("$N$", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
