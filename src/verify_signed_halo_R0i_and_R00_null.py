"""Two follow-ups to the per-component halo decomposition.

(1) Time-space residual R_0i via a heat-current operator on the
    relational graph.  On a quasi-static lattice the Bianchi
    identity gives G_0i to leading order zero, so R_0i is
    proportional to the source-side heat current
       J_i(a) = sum_b xi_ab * (T_00(b) - T_00(a)) * e_ab_i,
    where e_ab_i = (x_b - x_a)_i / d_ab is the unit edge vector
    in the spectral-frame spatial coordinate i.  Per-axis
    {J_1,J_2,J_3} and the magnitude |J| are correlated against
    the geodesic distance to the nearest matter concentration to
    probe for an accretion-class signature.

(2) R_00-specific N3 null-test bootstrap.  For each regime, the
    matter-core mask is shuffled across nodes 200 times keeping
    |C_N| constant; for each draw the geodesic distance is
    recomputed and the Spearman correlation
    rho(|R_00|, d_nearest_shuffled) is recorded.  The observed
    correlation is compared against the null distribution to
    obtain a per-regime z-score and p-value.

Output:
   outputs/verify_signed_halo_R0i_and_R00_null.json
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
    XI_THRESH, ELL_0, D_MIN, EPS_D, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_signed_halo_R0i_and_R00_null.json"
CORE_TOP_FRAC = 0.05
N_BOOTSTRAP = 200
RNG_SEED = 20260503


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


def _build_graph_internals(xi_mat, n_lat):
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None]
              * weight_adj * deg_inv_sqrt[None, :])
    _, eigvecs = np.linalg.eigh(l_norm)
    spatial = eigvecs[:, 1:4]  # (n, 3) per-node spectral coords
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)
    return xi_off, adj, spatial, d_mat, d_mat_inf


def _heat_current(t00, xi_off, adj, spatial, d_mat):
    """J_i(a) = sum_b xi_ab * adj_ab * (T_00(b)-T_00(a)) * e_ab_i.

    e_ab_i = (spatial[b] - spatial[a])_i / d_ab.
    """
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]  # (a,b,3)
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]
    weight_adj = xi_off * adj
    delta_t = t00[None, :] - t00[:, None]  # (a,b)
    contrib = weight_adj[:, :, None] * delta_t[:, :, None] * e_alpha
    J = contrib.sum(axis=1)  # (n, 3)
    return J


def _per_seed(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]
    t00 = prep["t00"]
    R_00 = g_00 + LAMBDA_T - t00

    xi_off, adj, spatial, d_mat, d_mat_inf = _build_graph_internals(
        xi_mat.copy(), n_lat)
    J = _heat_current(t00, xi_off, adj, spatial, d_mat)
    J_mag = np.linalg.norm(J, axis=1)

    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs(adj, top_idx, d_mat_inf)

    return {
        "R_00": R_00, "t00": t00,
        "J": J, "J_mag": J_mag,
        "adj": adj, "d_mat_inf": d_mat_inf, "d_nearest": d_nearest,
        "n_lat": n_lat,
    }


def _r0i_audit(seed_results):
    """Pool across seeds: signed Spearman of J_i and |J| vs d_nearest."""
    pools = {"J0": [], "J1": [], "J2": [], "Jmag": [], "d": []}
    for r in seed_results:
        m = np.isfinite(r["d_nearest"])
        if m.sum() < 30:
            continue
        pools["J0"].append(r["J"][m, 0])
        pools["J1"].append(r["J"][m, 1])
        pools["J2"].append(r["J"][m, 2])
        pools["Jmag"].append(r["J_mag"][m])
        pools["d"].append(r["d_nearest"][m])
    if not pools["d"]:
        return None
    cat = {k: np.concatenate(v) for k, v in pools.items()}
    return {
        "n_pooled": int(len(cat["d"])),
        "rho_J0_vs_d": _spearman(cat["J0"], cat["d"]),
        "rho_J1_vs_d": _spearman(cat["J1"], cat["d"]),
        "rho_J2_vs_d": _spearman(cat["J2"], cat["d"]),
        "rho_Jmag_vs_d": _spearman(cat["Jmag"], cat["d"]),
        "rho_absJ0_vs_d": _spearman(np.abs(cat["J0"]), cat["d"]),
        "rho_absJ1_vs_d": _spearman(np.abs(cat["J1"]), cat["d"]),
        "rho_absJ2_vs_d": _spearman(np.abs(cat["J2"]), cat["d"]),
        "frac_pos_Jmag": float((cat["Jmag"] > 0).mean()),
        "median_Jmag": float(np.median(cat["Jmag"])),
    }


def _r00_null_bootstrap(seed_results, n_boot=N_BOOTSTRAP,
                         seed=RNG_SEED):
    """For each seed: observed rho(|R_00|, d_nearest); null draws
    with the matter-core mask shuffled across nodes."""
    rng = np.random.default_rng(seed)
    obs_rhos = []
    null_rhos_all = []
    for r in seed_results:
        m = np.isfinite(r["d_nearest"])
        if m.sum() < 30:
            continue
        absR00_m = np.abs(r["R_00"][m])
        d_obs = r["d_nearest"][m]
        rho_obs = _spearman(absR00_m, d_obs)
        obs_rhos.append(rho_obs)
        # Null: shuffle |t00| (which determines core selection)
        adj = r["adj"]
        d_mat_inf = r["d_mat_inf"]
        n_lat = r["n_lat"]
        top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
        null_for_seed = []
        idx_all = np.arange(n_lat)
        for _ in range(n_boot):
            shuffled_top = rng.choice(idx_all, size=top_n, replace=False)
            d_null = _bfs(adj, shuffled_top.tolist(), d_mat_inf)
            d_null_m = d_null[m]
            null_for_seed.append(_spearman(absR00_m, d_null_m))
        null_rhos_all.append(null_for_seed)
    if not obs_rhos:
        return None
    obs_pool = float(np.median(obs_rhos))
    null_pool = np.array(null_rhos_all).flatten()
    null_pool = null_pool[np.isfinite(null_pool)]
    if len(null_pool) < 10:
        return {"observed_rho_median": obs_pool,
                "null_n": int(len(null_pool))}
    null_mean = float(null_pool.mean())
    null_std = float(null_pool.std())
    z = (obs_pool - null_mean) / (null_std + 1e-12)
    p_two_sided = float(2.0 * (1.0 -
                                  0.5 * (1.0 + math.erf(
                                      abs(z) / math.sqrt(2.0)))))
    return {
        "observed_rho_median": obs_pool,
        "observed_rhos_per_seed": obs_rhos,
        "null_n_total": int(len(null_pool)),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": float(z),
        "p_two_sided": p_two_sided,
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
    seed_results = []
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed(np.asarray(xi_mat), np.asarray(psi),
                            np.asarray(k_field), np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        seed_results.append(r)
    if not seed_results:
        return None
    return {
        "regime": regime, "N": int(n_lat),
        "n_seeds": len(seed_results),
        "R_0i": _r0i_audit(seed_results),
        "R_00_null": _r00_null_bootstrap(seed_results),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        r = _process(regime, n_lat, max_seeds=24)
        if r is None:
            continue
        rows.append(r)
        roi = r["R_0i"] or {}
        nul = r["R_00_null"] or {}
        print(f"    R_0i: |J|->d rho={roi.get('rho_Jmag_vs_d',float('nan')):+.3f}  "
              f"|J0|/|J1|/|J2|->d "
              f"{roi.get('rho_absJ0_vs_d',float('nan')):+.3f}/"
              f"{roi.get('rho_absJ1_vs_d',float('nan')):+.3f}/"
              f"{roi.get('rho_absJ2_vs_d',float('nan')):+.3f}  "
              f"|  R_00 null: rho_obs={nul.get('observed_rho_median',float('nan')):+.3f}  "
              f"z={nul.get('z_score',float('nan')):+.1f}")

    if rows:
        out = {
            "method": "R_0i heat-current + R_00-specific N3 null bootstrap",
            "n_bootstrap": N_BOOTSTRAP,
            "core_top_frac": CORE_TOP_FRAC,
            "per_regime": rows,
            "summary": {
                "n_regimes": len(rows),
                "n_R0i_Jmag_negative":
                    sum(1 for r in rows
                        if r["R_0i"] and
                        r["R_0i"]["rho_Jmag_vs_d"] ==
                        r["R_0i"]["rho_Jmag_vs_d"]
                        and r["R_0i"]["rho_Jmag_vs_d"] < 0),
                "median_R0i_Jmag_rho":
                    float(np.median([
                        r["R_0i"]["rho_Jmag_vs_d"] for r in rows
                        if r["R_0i"] and
                        r["R_0i"]["rho_Jmag_vs_d"] ==
                        r["R_0i"]["rho_Jmag_vs_d"]])),
                "n_R00_null_z_above_5":
                    sum(1 for r in rows
                        if r["R_00_null"] and
                        abs(r["R_00_null"].get("z_score", 0)) > 5),
                "n_R00_null_z_above_10":
                    sum(1 for r in rows
                        if r["R_00_null"] and
                        abs(r["R_00_null"].get("z_score", 0)) > 10),
                "max_R00_null_z":
                    float(max(
                        (abs(r["R_00_null"]["z_score"]) for r in rows
                         if r["R_00_null"]),
                        default=float("nan"))),
            },
        }
        s = out["summary"]
        print()
        print(f"  Summary across {s['n_regimes']} regimes:")
        print(f"    R_0i |J|->d: median rho = {s['median_R0i_Jmag_rho']:+.3f}, "
              f"{s['n_R0i_Jmag_negative']}/{s['n_regimes']} negative")
        print(f"    R_00 null z: {s['n_R00_null_z_above_5']}/{s['n_regimes']} "
              f"with |z|>5, {s['n_R00_null_z_above_10']}/{s['n_regimes']} "
              f"with |z|>10, max |z|={s['max_R00_null_z']:.1f}")
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
