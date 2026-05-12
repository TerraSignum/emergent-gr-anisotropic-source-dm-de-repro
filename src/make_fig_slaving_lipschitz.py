r"""Generate the 3D slaving-Lipschitz figure for P4-B.

Two-panel composite:
  Left  : 3D scatter (degree, Laplacian^2 diag, K) with RF
          predicted surface coloured by residual.
  Right : R^2 comparison bar across the four aggregator
          classes (linear, polynomial-3, RF, spectral-12).

Output: paper/figures/fig_slaving_lipschitz.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_first_snapshot(regime: str, n_lat: int):
    candidates = [
        PARENT / f"results_d1_{regime.lower()}_24seeds" / f"{regime}.snapshots.npz",
        PARENT / f"results_d1_{regime.lower()}_8seeds" / f"{regime}.snapshots.npz",
        PARENT / f"results_d1_{regime.lower()}_12seeds" / f"{regime}.snapshots.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            xi = d["edge_xi_snapshots"][0, -1].astype(float).copy()
            np.fill_diagonal(xi, 1.0)
            ff_k = d.get("ff_K_seed0", np.full((n_lat, n_lat), 0.55))
            return xi, np.asarray(ff_k, dtype=float)
    return None


def base_features(xi):
    n = xi.shape[0]
    xi_off = xi.copy()
    np.fill_diagonal(xi_off, 0.0)
    deg = xi_off.sum(axis=1)
    L = np.diag(deg) - xi_off
    lap_sq = np.diag(L @ L)
    eigvals, eigvecs = np.linalg.eigh(L)
    f1, f2, f3 = eigvecs[:, 1], eigvecs[:, 2], eigvecs[:, 3]
    mean_edge = xi_off.mean(axis=1)
    return np.column_stack([deg, lap_sq, f1, f2, f3, mean_edge])


def fit_rf(features, target):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None, None
    rf = RandomForestRegressor(n_estimators=64, max_depth=8, random_state=0)
    rf.fit(features, target)
    pred = rf.predict(features)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2), pred


def main():
    # Snapshot for 3D scatter (use P5N100 for richer node count)
    snap = load_first_snapshot("P5N100", 100)
    if snap is None:
        snap = load_first_snapshot("P5N84", 84)
    if snap is None:
        print("No snapshot found")
        return
    xi, ff_k = snap
    feats = base_features(xi)
    deg = feats[:, 0]
    lap_sq = feats[:, 1]
    if ff_k.ndim == 2:
        K = ff_k.mean(axis=1)
    else:
        K = ff_k

    # Random forest fit + predictions
    r2_rf, K_pred = fit_rf(feats, K)
    if K_pred is None:
        K_pred = K
        r2_rf = float("nan")
    residual = K - K_pred

    # Cross-regime R^2 panel data (from earlier run, hardcoded summary)
    R2_summary = {
        "linear_6feat": [0.116, 0.089, 0.028, 0.017],
        "polynomial_3": [0.956, 0.696, 0.667, 0.548],
        "random_forest": [0.833, 0.833, 0.795, 0.784],
        "spectral_12": [0.232, 0.199, 0.225, 0.166],
    }
    regime_labels = ["N=64", "N=72", "N=84", "N=100"]

    # Figure
    fig = plt.figure(figsize=(13.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.20)

    # Panel 1: 3D scatter
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    sc = ax3d.scatter(
        deg, lap_sq, K,
        c=residual, cmap="coolwarm",
        s=24, alpha=0.85, edgecolor="k", linewidth=0.3,
    )
    ax3d.set_xlabel(r"node degree $\sum_j \Xi_{ij}$", fontsize=10)
    ax3d.set_ylabel(r"squared Laplacian $(\mathcal{L}^2)_{ii}$",
                     fontsize=10)
    ax3d.set_zlabel(r"factor field $\langle K_{i,\cdot}\rangle$",
                     fontsize=10)
    ax3d.set_title(
        f"Slaving relationship $K = \\mathcal{{F}}(\\Xi)$ (R$^2$={r2_rf:.3f})",
        fontsize=11)
    cbar = fig.colorbar(sc, ax=ax3d, shrink=0.55, pad=0.10)
    cbar.set_label(r"$K - \mathcal{F}(\Xi)$ residual", fontsize=9)
    ax3d.view_init(elev=18, azim=42)

    # Panel 2: R^2 comparison bars
    ax2 = fig.add_subplot(gs[0, 1])
    aggregators = list(R2_summary.keys())
    pretty = {
        "linear_6feat": "Linear\n(6 features)",
        "polynomial_3": "Polynomial\n(degree 3)",
        "random_forest": "Random Forest\n(depth=8)",
        "spectral_12": "Spectral\n(top-12 modes)",
    }
    means = [np.mean(R2_summary[a]) for a in aggregators]
    stds = [np.std(R2_summary[a]) for a in aggregators]
    colours = ["#888888", "#3b8bbf", "#1f6f3f", "#bf5b3b"]
    xs = np.arange(len(aggregators))
    ax2.bar(xs, means, yerr=stds, color=colours, alpha=0.75,
            edgecolor="k", linewidth=0.6, capsize=4)
    ax2.axhline(0.10, color="grey", linestyle="--", linewidth=1,
                label=r"naive sigmoid baseline $\sim\!0.10$")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([pretty[a] for a in aggregators], fontsize=9)
    ax2.set_ylabel(r"average $R^2$ on per-node $K$ across $N\!\in\![64,100]$",
                    fontsize=10)
    ax2.set_title("Slaving aggregator comparison", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for x, m in zip(xs, means):
        ax2.text(x, m + 0.04, f"{m:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        r"Lipschitz slaving of factor field $K$ to $\Xi$-graph topology",
        fontsize=12, y=1.00)
    out_pdf = FIG_DIR / "fig_slaving_lipschitz.pdf"
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
