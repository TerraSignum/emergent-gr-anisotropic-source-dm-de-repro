"""Pre-gravity-hypothesis test: radial sign of the heat current.

Hypothesis (user 2026-05-04): before gravity emerges as
attractive, lattice configurations should show effective
repulsion at short defect-defect distances (matter points
must stay separated to maintain topological winding); at
larger distances, attraction dominates. Empirical signature:
the radial component of the heat current

   J_radial(a) = J(a) . hat e_{r,a}, with
   hat e_{r,a} = (x_a - x_{core,a}) / |x_a - x_{core,a}|

(unit vector pointing from the nearest matter core OUTWARD
through node a) should show a sign flip at some critical
distance d_crit. Convention:

   J_radial < 0  ->  J points inward (toward cores) =
                     attractive (standard gravitational accretion)
   J_radial > 0  ->  J points outward (away from cores) =
                     repulsive (pre-gravity dispersion)

If gravity is purely attractive at every scale, J_radial < 0
on every shell. A sign flip in inner shells -> outer shells
or vice versa would be empirical evidence of the hypothesis.

For each canonical-physics regime and each seed, compute:
  - J_i(a) heat current per node
  - For each node a, identify nearest matter core (top-T_00) c(a)
  - Spectral-frame radial unit vector hat e_{r,a} from c(a)
  - J_radial(a) = J(a) . hat e_{r,a}
  - Pool over seeds; compute median J_radial per quintile
    shell of d(a, partial C_N)

Output: outputs/verify_pre_gravity_radial_sign_flip.json
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
    LADDER, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_pre_gravity_radial_sign_flip.json"
CORE_TOP_FRAC = 0.05
N_SHELLS = 5
MAX_SEEDS = 4


def _bfs(adj, target_idx, d_mat):
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    nearest_target = np.full(n, -1, dtype=int)
    for t in target_idx:
        dist[t] = 0.0
        nearest_target[t] = t
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
                    parent[v] = u
                    nearest_target[v] = nearest_target[u]
    return dist, nearest_target


def _per_seed(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    t00 = prep["t00"]

    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12

    # Spectral spatial frame (matches per_seed_galerkin construction)
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat, dtype=np.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    _, eigvecs = np.linalg.eigh(l_norm)
    spatial = eigvecs[:, 1:4]

    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)

    # Heat-current
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    e_alpha = spatial_diff * inv_d[:, :, None]
    delta_t = t00[None, :] - t00[:, None]
    contrib = weight_adj[:, :, None] * delta_t[:, :, None] * e_alpha
    J = contrib.sum(axis=1)

    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()

    # BFS gives both distance AND nearest-core identity per node
    d_nearest, nearest_core = _bfs(adj, top_idx, d_mat_inf)

    # Radial unit vector: from core outward through node a
    e_radial = np.zeros_like(spatial)
    valid = (nearest_core >= 0) & np.isfinite(d_nearest)
    for a in range(n_lat):
        if not valid[a]:
            continue
        c = nearest_core[a]
        if a == c:
            # Node is itself a core; radial direction undefined
            continue
        diff = spatial[a] - spatial[c]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            continue
        e_radial[a] = diff / norm

    # Radial component of J
    J_radial = (J * e_radial).sum(axis=1)
    # Filter: valid radial direction + finite J + d_nearest > 0
    Jmag = np.linalg.norm(J, axis=1)
    radial_valid = (valid
                    & (np.linalg.norm(e_radial, axis=1) > 0.5)
                    & np.isfinite(J_radial)
                    & np.isfinite(Jmag)
                    & (d_nearest > 1e-6))

    return {
        "spatial": spatial, "J": J, "J_radial": J_radial,
        "d_nearest": d_nearest, "core_idx": np.array(top_idx),
        "nearest_core": nearest_core, "valid": radial_valid,
        "t00": t00,
    }


def _process_regime(regime, n_lat, max_seeds=MAX_SEEDS):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    pools_d = []
    pools_jrad = []
    pools_jmag = []
    for seed_tuple in seeds:
        try:
            r = _per_seed(np.asarray(seed_tuple[0]),
                            np.asarray(seed_tuple[1]),
                            np.asarray(seed_tuple[2]) if len(seed_tuple) > 2
                            else None,
                            np.asarray(seed_tuple[3]) if len(seed_tuple) > 3
                            else None,
                            n_lat)
        except Exception as e:  # noqa: BLE001
            print(f"    seed skipped: {type(e).__name__}: {e}")
            continue
        m = r["valid"]
        if m.sum() < 30:
            continue
        pools_d.append(r["d_nearest"][m])
        pools_jrad.append(r["J_radial"][m])
        pools_jmag.append(np.linalg.norm(r["J"][m], axis=1))
    if not pools_d:
        return None
    d_pool = np.concatenate(pools_d)
    jr_pool = np.concatenate(pools_jrad)
    jm_pool = np.concatenate(pools_jmag)

    # Quintile shells of d_nearest
    edges = np.quantile(d_pool, np.linspace(0, 1, N_SHELLS + 1))
    edges[-1] += 1e-9
    shells = []
    for k in range(N_SHELLS):
        sel = (d_pool >= edges[k]) & (d_pool < edges[k + 1])
        if sel.sum() < 5:
            shells.append({"n": int(sel.sum())})
            continue
        jr = jr_pool[sel]
        jm = jm_pool[sel]
        shells.append({
            "n": int(sel.sum()),
            "d_low": float(edges[k]),
            "d_high": float(edges[k + 1]),
            "J_radial_mean": float(jr.mean()),
            "J_radial_median": float(np.median(jr)),
            "J_radial_std": float(jr.std()),
            "frac_J_radial_positive": float((jr > 0).mean()),
            "frac_J_radial_negative": float((jr < 0).mean()),
            "J_magnitude_mean": float(jm.mean()),
            "J_radial_over_J_magnitude_mean":
                float(jr.mean() / jm.mean()) if jm.mean() > 0 else float("nan"),
        })

    # Overall pool stats + correlation
    overall_jrad_mean = float(jr_pool.mean())
    overall_jrad_median = float(np.median(jr_pool))
    frac_pos_overall = float((jr_pool > 0).mean())
    return {
        "regime": regime, "N": int(n_lat),
        "n_pooled": int(len(d_pool)),
        "shells_by_d": shells,
        "overall_J_radial_mean": overall_jrad_mean,
        "overall_J_radial_median": overall_jrad_median,
        "overall_frac_J_radial_positive": frac_pos_overall,
        "sign_flip_inner_to_outer":
            (shells[0].get("J_radial_mean", 0)
             * shells[-1].get("J_radial_mean", 0) < 0
             if shells[0].get("J_radial_mean") is not None
             and shells[-1].get("J_radial_mean") is not None
             else False),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        try:
            r = _process_regime(regime, n_lat)
        except Exception as e:  # noqa: BLE001
            print(f"    skipped ({type(e).__name__}: {e})")
            continue
        if r is None:
            continue
        rows.append(r)
        # Print shells succinctly
        for k, s in enumerate(r["shells_by_d"]):
            if "J_radial_mean" not in s:
                continue
            print(f"    shell {k}  d=[{s['d_low']:.2f}, {s['d_high']:.2f}]  "
                  f"<J_rad>={s['J_radial_mean']:+.5f}  "
                  f"frac_pos={s['frac_J_radial_positive']:.2%}  "
                  f"<|J|>={s['J_magnitude_mean']:.4f}")
        print(f"    => sign_flip_inner_to_outer = {r['sign_flip_inner_to_outer']}")

    if not rows:
        print("no data")
        return

    n = len(rows)
    n_with_flip = sum(1 for r in rows if r["sign_flip_inner_to_outer"])
    n_attr = sum(1 for r in rows if r["overall_J_radial_mean"] < 0)
    n_repul = sum(1 for r in rows if r["overall_J_radial_mean"] > 0)
    out = {
        "method": "Pre-gravity hypothesis: radial-sign test of heat current",
        "convention": "J_radial < 0 => J points inward (attractive, accretion). "
                       "J_radial > 0 => outward (repulsive, dispersion).",
        "core_top_frac": CORE_TOP_FRAC,
        "n_shells": N_SHELLS,
        "per_regime": rows,
        "summary": {
            "n_regimes": n,
            "n_overall_attractive": n_attr,
            "n_overall_repulsive": n_repul,
            "n_inner_outer_sign_flip": n_with_flip,
            "verdict": (
                "ATTRACTIVE_ON_ALL" if n_attr == n
                else "REPULSIVE_ON_ALL" if n_repul == n
                else "MIXED"
            ),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"  Summary across {n} regimes:")
    print(f"    overall attractive (<J_r> < 0): {n_attr}/{n}")
    print(f"    overall repulsive  (<J_r> > 0): {n_repul}/{n}")
    print(f"    inner-to-outer sign flip:        {n_with_flip}/{n}")
    print(f"    verdict: {out['summary']['verdict']}")
    print()
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
