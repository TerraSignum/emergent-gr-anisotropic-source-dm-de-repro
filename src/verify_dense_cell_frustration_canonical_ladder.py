"""Dense-cell metric-axiom frustration audit on the
canonical-physics ladder.

The audit computes, for every per-node Xi/psi snapshot bundled
in results_d1_p5n*_*seeds/*.snapshots.npz, the share of
distinct (i,j,k) triangles for which the metric-axiom defect

  delta_ijk = -log(Xi_ik) + log(Xi_ij) + log(Xi_jk)

is negative, i.e.\\ the share of triangles for which Xi
over-saturates the M3 submultiplicativity inequality
Xi_ik >= Xi_ij * Xi_jk. The share f_{delta < 0} is a
parameter-free dimensionless measure of the dense-cell
graph-frustration content of Xi, additional to and structurally
distinct from the per-node energy-density halo fraction
f_{R_00 < 0} reported in the same paper. The aim is to test
how f_{delta < 0} scales with the lattice size N along the
canonical-physics ladder (Symanzik 1/N continuum limit) and
to cross-reference with the regime-stable f_{R_00 < 0}
finding on the same configurations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO.parent
sys.path.insert(0, str(PARENT / "emergent-gr-closure-repro" / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, _name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LAMBDA_T, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_dense_cell_frustration_canonical_ladder.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

REGIMES = [
    # Canonical P5/P5N family, N-ordered ascending.
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
    # Alt-anchor (P6/P7/P8) regimes, listed separately after canonical.
    ("P7",     72),
]


def negative_triangle_share(xi, eps=1e-12, epsilon_triangle=0.0):
    """Fraction of distinct ordered triangles (i,j,k) with
    delta_ijk = -log Xi_ik + log Xi_ij + log Xi_jk < epsilon_triangle.
    Faithful re-implementation of
    \\verb|worldformula.core.fast_slow_dynamics.active_triangle_cells|.
    """
    n = xi.shape[0]
    log_xi = np.log(np.clip(xi, eps, 1.0))
    delta = (-log_xi[:, None, :]
             + log_xi[:, :, None]
             + log_xi[None, :, :])
    active = delta + epsilon_triangle < 0.0
    idx = np.arange(n)
    distinct = ((idx[:, None, None] != idx[None, :, None])
                & (idx[:, None, None] != idx[None, None, :])
                & (idx[None, :, None] != idx[None, None, :]))
    active = active & distinct
    n_distinct = int(distinct.sum())
    if n_distinct == 0:
        return float("nan")
    return float(active.sum()) / n_distinct


def cycle_density(xi, threshold=0.13):
    """Per-node closed 3-cycle density on the Xi-thresholded graph.
    A 3-cycle through node a is a triangle (a,b,c) with all three
    edges above threshold. Average count per node.
    """
    n = xi.shape[0]
    adj = (xi >= threshold) & (~np.eye(n, dtype=bool))
    a2 = adj.astype(np.int64) @ adj.astype(np.int64)
    n_triangles = int(np.einsum("ij,ij->", a2, adj.astype(np.int64))) // 6
    return float(n_triangles) / float(n)


def f_neg_R00(xi, psi, k_field, q_field, n_lat):
    """Fraction of nodes with R_00(a) = G_00(a) + LAMBDA_T - T_00(a) < 0.
    Uses the per-seed Galerkin pipeline of P4 / P4-B."""
    prep = per_seed_galerkin(xi, psi, k_field, q_field, n_lat, np)
    r00 = prep["g_00_h"] + LAMBDA_T - prep["t00"]
    return float((r00 < 0).mean())


def _process_regime(regime, n_lat):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat) if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))
    if not seeds:
        return None
    nts_list, cd_list, fneg_list = [], [], []
    for s in seeds:
        xi = np.asarray(s[0], float).copy()
        psi = np.asarray(s[1])
        k = np.asarray(s[2]) if len(s) > 2 else None
        q = np.asarray(s[3]) if len(s) > 3 else None
        nts_list.append(negative_triangle_share(xi))
        cd_list.append(cycle_density(xi))
        try:
            fneg_list.append(f_neg_R00(xi, psi, k, q, n_lat))
        except Exception:  # noqa: BLE001
            fneg_list.append(float("nan"))
    nts = np.array(nts_list, dtype=float)
    cd = np.array(cd_list, dtype=float)
    fneg = np.array(fneg_list, dtype=float)
    return {
        "regime": regime,
        "N": int(n_lat),
        "n_seeds": len(seeds),
        "neg_triangle_share_mean": float(np.mean(nts)),
        "neg_triangle_share_std": float(np.std(nts)),
        "cycle_density_mean": float(np.mean(cd)),
        "cycle_density_std": float(np.std(cd)),
        "f_neg_R00_mean": float(np.nanmean(fneg)),
        "f_neg_R00_std": float(np.nanstd(fneg)),
    }


def main():
    rows = []
    print(f'{"regime":<10}{"N":>5}{"seeds":>6}{"neg_tri":>10}'
          f'{"cycle_d":>10}{"f_neg_R00":>12}')
    for regime, n_lat in REGIMES:
        r = _process_regime(regime, n_lat)
        if r is None:
            continue
        rows.append(r)
        print(f'{r["regime"]:<10}{r["N"]:>5}{r["n_seeds"]:>6}'
              f'{r["neg_triangle_share_mean"]:>10.4f}'
              f'{r["cycle_density_mean"]:>10.2f}'
              f'{r["f_neg_R00_mean"]:>12.4f}')

    if not rows:
        return

    nts_vals = np.array([r["neg_triangle_share_mean"] for r in rows])
    fneg_vals = np.array([r["f_neg_R00_mean"] for r in rows])
    cd_vals = np.array([r["cycle_density_mean"] for r in rows])
    n_vals = np.array([r["N"] for r in rows], dtype=float)

    summary = {
        "n_regimes": len(rows),
        "neg_triangle_share_mean_across_regimes":
            float(np.mean(nts_vals)),
        "neg_triangle_share_min": float(np.min(nts_vals)),
        "neg_triangle_share_max": float(np.max(nts_vals)),
        "f_neg_R00_mean_across_regimes": float(np.mean(fneg_vals)),
        "f_neg_R00_min": float(np.min(fneg_vals)),
        "f_neg_R00_max": float(np.max(fneg_vals)),
        "cycle_density_mean_across_regimes": float(np.mean(cd_vals)),
        "h110_PASS_threshold": 0.50,
        "h110_PASS_on_canonical_ladder": bool(
            np.all(nts_vals >= 0.50)),
    }

    if len(rows) >= 3:
        a = np.column_stack([np.ones_like(n_vals), 1.0 / n_vals])
        c, *_ = np.linalg.lstsq(a, nts_vals, rcond=None)
        summary["neg_triangle_share_symanzik_intercept"] = float(c[0])
        summary["neg_triangle_share_symanzik_slope"] = float(c[1])
        c2, *_ = np.linalg.lstsq(a, fneg_vals, rcond=None)
        summary["f_neg_R00_symanzik_intercept"] = float(c2[0])
        summary["f_neg_R00_symanzik_slope"] = float(c2[1])

    out = {
        "method": ("Negative-triangle defect share + per-node "
                   "f_neg(R_00) on the canonical-physics ladder; "
                   "H110 Big Bang dense-cell universality analogue"),
        "definitions": {
            "negative_triangle_share":
                "fraction of distinct (i,j,k) with "
                "delta_ijk = -log Xi_ik + log Xi_ij + "
                "log Xi_jk < 0 (M3 over-saturation)",
            "f_neg_R00":
                "fraction of nodes with R_00(a) = G_00(a) + "
                "LAMBDA_T - T_00(a) < 0 (P4-B halo signature)",
            "cycle_density":
                "per-node closed-triangle density on the "
                "Xi-thresholded graph (Xi_threshold=0.13)",
        },
        "per_regime": rows,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {OUT}")
    print(f"  neg_triangle_share across {len(rows)} regimes: "
          f"mean={summary['neg_triangle_share_mean_across_regimes']:.4f}, "
          f"min={summary['neg_triangle_share_min']:.4f}, "
          f"max={summary['neg_triangle_share_max']:.4f}")
    print(f"  f_neg(R_00) across {len(rows)} regimes: "
          f"mean={summary['f_neg_R00_mean_across_regimes']:.4f}, "
          f"min={summary['f_neg_R00_min']:.4f}, "
          f"max={summary['f_neg_R00_max']:.4f}")
    print(f"  H110 dense-cell PASS threshold (>= 0.50): "
          f"{summary['h110_PASS_on_canonical_ladder']}")


if __name__ == "__main__":
    main()
