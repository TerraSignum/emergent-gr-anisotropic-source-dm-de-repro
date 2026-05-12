"""Item 7: Per-axis (2+1) re-test in the per-node T-eigenframe.

The original (ii) paragraph reported R_11, R_22, R_33 in the
spectral-Laplacian Fiedler basis and gave a regime-dependent
mixed-sign pattern. The genuine (2+1) sign structure of
Lambda_back lives in the per-node T-eigenframe (sorted
ascending), per memory line 53.

For each node a we diagonalize T_ij(a) to get eigenvalues
t_eigs(a) sorted ascending and the rotation U(a). The geometric
spatial block G_ij(a) is rotated to the same per-node frame and
the residual is

   R_ii_eig(a) = (U^T G U)_ii(a) + Lambda_s_axis(a) - t_eigs_i(a)

where Lambda_s_axis is the per-axis backreaction component. We
report the per-axis signed Spearman of R_ii_eig vs d(a, partial
C_N) to test whether the (2+1) sign pattern is stable in the
T-eigenframe across regimes.

Output: outputs/verify_halo_T_eigenframe.json
"""
from __future__ import annotations

import json
import math
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
    LADDER, LAMBDA_T, LAMBDA_S, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_halo_T_eigenframe.json"
CORE_TOP_FRAC = 0.05


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _bfs(adj, target_idx, d_mat):
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    for t in target_idx:
        dist[t] = 0.0
    visited = np.zeros(n, dtype=bool)
    while True:
        u = -1; best = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]; u = i
        if u < 0 or best == np.inf:
            break
        visited[u] = True
        for v in np.where(adj[u] > 0)[0]:
            if not visited[v]:
                alt = dist[u] + d_mat[u, v]
                if alt < dist[v]:
                    dist[v] = alt
    return dist


def _per_seed_eigenframe(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_ij = prep["g_ij_h"]   # (n, 3, 3)
    t_ij = prep["t_ij"]     # (n, 3, 3)
    t00 = prep["t00"]
    n = g_ij.shape[0]

    # Per-node diagonalize T_ij; eigvals sorted ascending; eigvecs cols U_a
    t_clean = np.where(np.isfinite(t_ij), t_ij, 0.0)
    R_eig = np.full((n, 3), np.nan)
    t_eig = np.full((n, 3), np.nan)
    for a in range(n):
        try:
            w, U = np.linalg.eigh(t_clean[a])
        except Exception:  # noqa: BLE001
            continue
        # Rotate G_ij into T-eigenframe (sort ascending matches w)
        G_rot = U.T @ g_ij[a] @ U
        # Lambda_s in T-eigenframe: assumed isotropic on the diagonal
        # (the framework's Lambda_back-spatial is Lambda_s * eye3 in
        # any spatial basis; the (2+1) sign pattern arises in the
        # T-eigenframe per memory line 53)
        for i in range(3):
            R_eig[a, i] = G_rot[i, i] + LAMBDA_S - w[i]
            t_eig[a, i] = w[i]

    # Distance to nearest matter concentration
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs(adj, top_idx, d_mat_inf)

    return {
        "R_eig": R_eig,        # (n, 3) sorted axes
        "t_eig": t_eig,
        "d_nearest": d_nearest,
        "t00": t00,
    }


def _process(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    pools_R = [[], [], []]; pools_d = []; pools_t = [[], [], []]
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed_eigenframe(np.asarray(xi_mat), np.asarray(psi),
                                       np.asarray(k_field),
                                       np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        m = np.isfinite(r["d_nearest"]) & np.all(np.isfinite(r["R_eig"]),
                                                    axis=1)
        if m.sum() < 30:
            continue
        for i in range(3):
            pools_R[i].append(r["R_eig"][m, i])
            pools_t[i].append(r["t_eig"][m, i])
        pools_d.append(r["d_nearest"][m])
    if not pools_d:
        return None
    R_pool = [np.concatenate(p) for p in pools_R]
    t_pool = [np.concatenate(p) for p in pools_t]
    d_pool = np.concatenate(pools_d)
    out = {
        "regime": regime, "N": int(n_lat),
        "n_pooled": int(len(d_pool)),
    }
    for i in range(3):
        out[f"R_eig_axis_{i}"] = {
            "spearman_signed_vs_d": _spearman(R_pool[i], d_pool),
            "spearman_abs_vs_d":    _spearman(np.abs(R_pool[i]), d_pool),
            "frac_positive":        float((R_pool[i] > 0).mean()),
            "median":               float(np.median(R_pool[i])),
            "mean":                 float(R_pool[i].mean()),
            "t_eig_mean":           float(t_pool[i].mean()),
            "t_eig_median":         float(np.median(t_pool[i])),
        }
    # Detect (2+1) sign pattern: are exactly 2 axes negative-mean
    # and 1 positive-mean (or vice versa)?
    means = [out[f"R_eig_axis_{i}"]["mean"] for i in range(3)]
    n_neg = sum(1 for m in means if m < 0)
    n_pos = sum(1 for m in means if m > 0)
    is_2_plus_1 = (n_neg == 2 and n_pos == 1) or (n_neg == 1 and n_pos == 2)
    out["is_2plus1_pattern"] = bool(is_2_plus_1)
    out["sign_split_n_neg_n_pos"] = [int(n_neg), int(n_pos)]
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d} ...")
        r = _process(regime, n_lat, max_seeds=24)
        if r is None:
            continue
        rows.append(r)
        s0 = r["R_eig_axis_0"]; s1 = r["R_eig_axis_1"]; s2 = r["R_eig_axis_2"]
        n_neg, n_pos = r["sign_split_n_neg_n_pos"]
        print(f"    sorted axes (ascending t_eig):  "
              f"axis0 mean={s0['mean']:+.3f} rho_d={s0['spearman_signed_vs_d']:+.3f}  "
              f"axis1 mean={s1['mean']:+.3f} rho_d={s1['spearman_signed_vs_d']:+.3f}  "
              f"axis2 mean={s2['mean']:+.3f} rho_d={s2['spearman_signed_vs_d']:+.3f}  "
              f"|  ({n_neg}neg/{n_pos}pos) {'2+1' if r['is_2plus1_pattern'] else '!!'}")

    n_2plus1 = sum(1 for r in rows if r["is_2plus1_pattern"])
    print()
    print(f"  Summary across {len(rows)} regimes:")
    print(f"    (2+1) mean-sign pattern in T-eigenframe: {n_2plus1}/{len(rows)}")

    out = {
        "method": "Item 7: per-axis (2+1) re-test in T-eigenframe",
        "core_top_frac": CORE_TOP_FRAC,
        "axes_sorted_ascending_by_t_eig": True,
        "per_regime": rows,
        "summary": {
            "n_regimes": len(rows),
            "n_2plus1_mean_sign_pattern": n_2plus1,
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
