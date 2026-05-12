"""Component-wise decomposition of the signed halo signal.

The user-asked question:
   "ist es -0.5 oder +1 - 1.5 = 0.5"
i.e. is a net negative correlation a uniform signal, or the result of
canceling positive and negative pieces?

For each node a we extract:
   R_00         (energy density component)
   R_11, R_22, R_33  (spatial diagonal, per axis)
   R_12, R_13, R_23  (spatial off-diagonal)
   tr R         (trace, possibly mixed)

For each component we report:
   (a) Sign distribution: % positive, % negative, mean, median.
   (b) Spearman(component, d_nearest)  — signed, no abs().
   (c) Spearman(|component|, d_nearest) — magnitude vs distance.
   (d) Distance-shell decomposition (5 quintile shells of d_nearest):
       median value of the component in each shell, so we see WHERE
       it lives.

This reveals whether the halo signal is:
   (A) uniform across components, all decaying with d
   (B) carried by R_00 only (energy density, expected for a halo)
   (C) net of canceling components (+R_00 - R_ii)
   (D) localized in a specific distance shell

Output: outputs/verify_signed_halo_component_decomposition.json
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

OUT = REPO / "outputs" / "verify_signed_halo_component_decomposition.json"
CORE_TOP_FRAC = 0.05
N_SHELLS = 5


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if den == 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def _bfs(adj, target_idx, d_mat):
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    for t in target_idx:
        dist[t] = 0.0
    visited = np.zeros(n, dtype=bool)
    while True:
        u = -1
        best = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u < 0 or best == np.inf:
            break
        visited[u] = True
        nbrs = np.where(adj[u] > 0)[0]
        for v in nbrs:
            if not visited[v]:
                alt = dist[u] + d_mat[u, v]
                if alt < dist[v]:
                    dist[v] = alt
    return dist


def _per_seed(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = np.eye(3)
    R_00 = g_00 + LAMBDA_T - t00
    R_d = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij  # per-node 3x3
    R_11 = R_d[:, 0, 0]
    R_22 = R_d[:, 1, 1]
    R_33 = R_d[:, 2, 2]
    R_12 = R_d[:, 0, 1]
    R_13 = R_d[:, 0, 2]
    R_23 = R_d[:, 1, 2]
    R_tr = R_00 + R_11 + R_22 + R_33

    # graph-distance from nearest top-T_00 defect
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    deg = adj.sum(axis=1)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.where(adj > 0, d_mat, np.inf)
    weight_grad = np.where(adj > 0,
                            xi_off * adj / np.maximum(d_mat, 1e-9) ** 2,
                            0.0)
    omega_a = weight_grad.sum(axis=1)
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs(adj, top_idx, d_mat)

    return {
        "R_00": R_00, "R_11": R_11, "R_22": R_22, "R_33": R_33,
        "R_12": R_12, "R_13": R_13, "R_23": R_23,
        "R_tr": R_tr, "t00": t00,
        "deg": deg, "omega": omega_a, "d": d_nearest,
    }


def _summary_for_component(name, values, d, t00, deg, omega):
    """For one R-component: sign distribution + corr + shell medians."""
    finite = np.isfinite(values) & np.isfinite(d)
    v = values[finite]
    dn = d[finite]
    n = len(v)
    if n < 30:
        return None
    n_pos = int((v > 0).sum())
    n_neg = int((v < 0).sum())
    out = {
        "n": n,
        "frac_positive": n_pos / n,
        "frac_negative": n_neg / n,
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "spearman_signed_vs_d": _spearman(v, dn),
        "spearman_abs_vs_d": _spearman(np.abs(v), dn),
        "spearman_signed_vs_T00": _spearman(v, t00[finite]),
    }
    # Shell decomposition by d_nearest quintiles
    edges = np.quantile(dn, np.linspace(0, 1, N_SHELLS + 1))
    edges[-1] += 1e-9
    shells = []
    for k in range(N_SHELLS):
        m = (dn >= edges[k]) & (dn < edges[k + 1])
        if m.sum() < 3:
            shells.append({"n": int(m.sum()), "med": float("nan"),
                            "mean": float("nan"), "frac_pos": float("nan")})
            continue
        vs = v[m]
        shells.append({
            "n": int(m.sum()),
            "d_low": float(edges[k]),
            "d_high": float(edges[k + 1]),
            "median": float(np.median(vs)),
            "mean": float(vs.mean()),
            "frac_positive": float((vs > 0).mean()),
        })
    out["shells_by_d"] = shells
    return out


def _process(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    pools = {k: [] for k in (
        "R_00", "R_11", "R_22", "R_33",
        "R_12", "R_13", "R_23", "R_tr", "t00",
        "deg", "omega", "d")}
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed(np.asarray(xi_mat), np.asarray(psi),
                            np.asarray(k_field), np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        m = np.isfinite(r["d"])
        if m.sum() < 30:
            continue
        for k in pools:
            pools[k].append(r[k][m])
    if not pools["R_00"]:
        return None
    cat = {k: np.concatenate(v) for k, v in pools.items()}
    out = {"regime": regime, "N": int(n_lat), "n_pooled": len(cat["d"])}
    for comp in ("R_00", "R_11", "R_22", "R_33",
                  "R_12", "R_13", "R_23", "R_tr"):
        out[comp] = _summary_for_component(
            comp, cat[comp], cat["d"], cat["t00"], cat["deg"], cat["omega"])
    # Reconstruction check: R_tr from sum
    diff = (cat["R_00"] + cat["R_11"] + cat["R_22"] + cat["R_33"]) - cat["R_tr"]
    out["sum_check_max_abs_diff"] = float(np.max(np.abs(diff)))
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        r = _process(regime, n_lat, max_seeds=24)
        if r is None:
            continue
        rows.append(r)
        # Compact line: show signed Spearman + frac_neg per component
        s00 = r["R_00"]; s11 = r["R_11"]; s22 = r["R_22"]; s33 = r["R_33"]
        str_ = r["R_tr"]
        print(f"    signed_spearman_vs_d:  "
              f"R00={s00['spearman_signed_vs_d']:+.3f}({s00['frac_negative']*100:.0f}% neg)  "
              f"R11={s11['spearman_signed_vs_d']:+.3f}({s11['frac_negative']*100:.0f}%)  "
              f"R22={s22['spearman_signed_vs_d']:+.3f}({s22['frac_negative']*100:.0f}%)  "
              f"R33={s33['spearman_signed_vs_d']:+.3f}({s33['frac_negative']*100:.0f}%)  "
              f"tr={str_['spearman_signed_vs_d']:+.3f}")

    if rows:
        OUT.write_text(json.dumps({
            "method": "Component-wise decomposition of signed halo",
            "core_top_frac": CORE_TOP_FRAC,
            "n_shells": N_SHELLS,
            "per_regime": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
