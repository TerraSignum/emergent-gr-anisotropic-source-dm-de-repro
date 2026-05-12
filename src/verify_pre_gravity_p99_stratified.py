"""p99-stratified pre-gravity audit.

The user-corrected approach: instead of aggregating over all
nodes (which mixes matter and dark-energy-background regions),
stratify by the per-node Frobenius residual

   delta_F(a) = ||R_munu(a)||_F / ||T_munu^Xi(a)||_F

and split into:
  - p99 heavy-tail subset = top 1% of delta_F (matter-cluster
                            nodes; the heavy tail of the
                            bulk-percentile spectrum identified
                            in the Stage 6f audit of the parent
                            paper)
  - bulk subset           = bottom 99%

The pre-gravity hypothesis predicts:
  - p99 (matter): J_radial NEGATIVE (attractive, gravitational
                  binding); mass-scaling chain-reaction visible
  - bulk (dark-energy background): J_radial POSITIVE on average
                  (Lambda-repulsion); no mass amplification

If both predictions hold, the matter/dark-energy dichotomy is
empirically sharp.

Output: outputs/verify_pre_gravity_p99_stratified.json
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
    LADDER, LAMBDA_T, LAMBDA_S, load_canonical, load_snapshots,
    per_node_relative_delta)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_pre_gravity_p99_stratified.json"
CORE_TOP_FRAC = 0.05
P99_QUANTILE = 0.95   # top-5% to match matter-core definition
MAX_SEEDS = 4

GAMMA = 1.0 / 10.0
BETA_PI = 15.0 / 16.0
D_OMEGA = BETA_PI - GAMMA
D_CRIT_PRED = 1.0 / D_OMEGA


def _bfs(adj, target_idx, d_mat):
    n = adj.shape[0]
    dist = np.full(n, np.inf)
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
        for v in np.where(adj[u] > 0)[0]:
            if not visited[v]:
                alt = dist[u] + d_mat[u, v]
                if alt < dist[v]:
                    dist[v] = alt
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
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat, dtype=np.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    _, eigvecs = np.linalg.eigh(l_norm)
    spatial = eigvecs[:, 1:4]

    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    e_alpha = spatial_diff * inv_d[:, :, None]
    delta_t = t00[None, :] - t00[:, None]
    contrib = weight_adj[:, :, None] * delta_t[:, :, None] * e_alpha
    J = contrib.sum(axis=1)

    # Per-node Frobenius residual delta_F(a)
    rel = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)
    delta_full = rel["delta_full"]    # ||R||_F / ||T||_F per node

    # Matter cores
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:]
    d_nearest, nearest_core = _bfs(adj, top_idx.tolist(), d_mat_inf)

    # Per-node radial unit vector + J_radial
    n = spatial.shape[0]
    Jr = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    for a in range(n):
        c = nearest_core[a]
        if c < 0 or a == c:
            continue
        diff = spatial[a] - spatial[c]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            continue
        e_r = diff / norm
        Jr[a] = float(J[a] @ e_r)
        valid[a] = (np.isfinite(Jr[a])
                    and np.isfinite(d_nearest[a])
                    and d_nearest[a] > 1e-6
                    and np.isfinite(np.linalg.norm(J[a]))
                    and np.isfinite(delta_full[a]))
    return {
        "delta_full": delta_full,
        "J_radial": Jr,
        "d_nearest": d_nearest,
        "valid": valid,
        "t00": t00,
        "spatial": spatial, "J": J,
        "nearest_core": nearest_core, "top_idx": top_idx,
    }


def _stratified_stats(d, jr, dF, p99_threshold):
    is_p99 = dF >= p99_threshold
    is_bulk = ~is_p99
    out = {}
    for label, sel in [("p99_heavy_tail", is_p99), ("bulk", is_bulk)]:
        if sel.sum() < 5:
            out[label] = None
            continue
        out[label] = {
            "n": int(sel.sum()),
            "frac_of_lattice": float(sel.mean()),
            "J_radial_mean": float(jr[sel].mean()),
            "J_radial_median": float(np.median(jr[sel])),
            "J_radial_std": float(jr[sel].std()),
            "frac_J_radial_negative": float((jr[sel] < 0).mean()),
            "delta_F_mean": float(dF[sel].mean()),
            "delta_F_min": float(dF[sel].min()),
            "delta_F_max": float(dF[sel].max()),
            "d_nearest_mean": float(d[sel].mean()),
            "d_nearest_median": float(np.median(d[sel])),
        }
    return out


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5:
        return float("nan")
    rx = np.argsort(np.argsort(x[finite])).astype(float)
    ry = np.argsort(np.argsort(y[finite])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _process(regime, n_lat, max_seeds=MAX_SEEDS):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    pool_d = []; pool_jr = []; pool_dF = []; pool_t00 = []
    for seed_tuple in seeds:
        try:
            r = _per_seed(np.asarray(seed_tuple[0]),
                            np.asarray(seed_tuple[1]),
                            np.asarray(seed_tuple[2]) if len(seed_tuple) > 2 else None,
                            np.asarray(seed_tuple[3]) if len(seed_tuple) > 3 else None,
                            n_lat)
        except Exception:  # noqa: BLE001
            continue
        m = r["valid"]
        if m.sum() < 30:
            continue
        pool_d.append(r["d_nearest"][m])
        pool_jr.append(r["J_radial"][m])
        pool_dF.append(r["delta_full"][m])
        pool_t00.append(np.abs(r["t00"][m]))
    if not pool_d:
        return None
    d = np.concatenate(pool_d)
    jr = np.concatenate(pool_jr)
    dF = np.concatenate(pool_dF)
    t00abs = np.concatenate(pool_t00)

    p99_thr = float(np.quantile(dF, P99_QUANTILE))
    strat = _stratified_stats(d, jr, dF, p99_thr)

    is_p99 = dF >= p99_thr
    is_bulk = ~is_p99

    # Spearman within each subset
    sp_p99_M_Jr = _spearman(t00abs[is_p99], jr[is_p99]) if is_p99.sum() >= 5 else float("nan")
    sp_bulk_M_Jr = _spearman(t00abs[is_bulk], jr[is_bulk]) if is_bulk.sum() >= 5 else float("nan")
    sp_p99_d_Jr = _spearman(d[is_p99], jr[is_p99]) if is_p99.sum() >= 5 else float("nan")
    sp_bulk_d_Jr = _spearman(d[is_bulk], jr[is_bulk]) if is_bulk.sum() >= 5 else float("nan")

    return {
        "regime": regime, "N": int(n_lat),
        "n_pooled": int(len(d)),
        "p99_threshold_delta_F": p99_thr,
        "stratified_stats": strat,
        "spearman_M_vs_Jr_p99": sp_p99_M_Jr,
        "spearman_M_vs_Jr_bulk": sp_bulk_M_Jr,
        "spearman_d_vs_Jr_p99": sp_p99_d_Jr,
        "spearman_d_vs_Jr_bulk": sp_bulk_d_Jr,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        try:
            r = _process(regime, n_lat)
        except Exception as e:  # noqa: BLE001
            print(f"    skipped ({type(e).__name__}: {e})")
            continue
        if r is None:
            print("    no data")
            continue
        rows.append(r)
        s = r["stratified_stats"]
        p99 = s["p99_heavy_tail"]; bulk = s["bulk"]
        if p99 and bulk:
            print(f"    p99 (n={p99['n']:>4d}, {p99['frac_of_lattice']*100:.1f}%)  "
                  f"<J_r>={p99['J_radial_mean']:+.5f}  "
                  f"frac_neg={p99['frac_J_radial_negative']:.2%}  "
                  f"<d>={p99['d_nearest_mean']:.3f}")
            print(f"    bulk (n={bulk['n']:>4d})        "
                  f"<J_r>={bulk['J_radial_mean']:+.5f}  "
                  f"frac_neg={bulk['frac_J_radial_negative']:.2%}  "
                  f"<d>={bulk['d_nearest_mean']:.3f}")
            print(f"    Spearman(M, J_r) p99: "
                  f"{r['spearman_M_vs_Jr_p99']:+.3f}  bulk: "
                  f"{r['spearman_M_vs_Jr_bulk']:+.3f}")
            print(f"    Spearman(d, J_r) p99: "
                  f"{r['spearman_d_vs_Jr_p99']:+.3f}  bulk: "
                  f"{r['spearman_d_vs_Jr_bulk']:+.3f}")

    out = {
        "method": "p99-stratified pre-gravity audit",
        "p99_quantile": P99_QUANTILE,
        "core_top_frac": CORE_TOP_FRAC,
        "framework_constants": {
            "alpha_xi": 9/10, "gamma": GAMMA,
            "beta_pi": BETA_PI, "D_Omega": D_OMEGA,
            "d_crit_predicted": D_CRIT_PRED,
        },
        "per_regime": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if rows:
        print()
        print("=== Cross-regime stratified summary ===")
        # p99 vs bulk J_r mean and frac_neg
        p99_means = [r["stratified_stats"]["p99_heavy_tail"]["J_radial_mean"]
                      for r in rows
                      if r["stratified_stats"]["p99_heavy_tail"]]
        bulk_means = [r["stratified_stats"]["bulk"]["J_radial_mean"]
                       for r in rows
                       if r["stratified_stats"]["bulk"]]
        p99_neg_frac = [r["stratified_stats"]["p99_heavy_tail"]["frac_J_radial_negative"]
                         for r in rows if r["stratified_stats"]["p99_heavy_tail"]]
        bulk_neg_frac = [r["stratified_stats"]["bulk"]["frac_J_radial_negative"]
                          for r in rows if r["stratified_stats"]["bulk"]]
        sp_p99_M = [r["spearman_M_vs_Jr_p99"] for r in rows
                     if np.isfinite(r["spearman_M_vs_Jr_p99"])]
        sp_bulk_M = [r["spearman_M_vs_Jr_bulk"] for r in rows
                      if np.isfinite(r["spearman_M_vs_Jr_bulk"])]
        print(f"  p99  <J_r> median across regimes: "
              f"{np.median(p99_means):+.5f}  "
              f"(range [{np.min(p99_means):+.5f}, {np.max(p99_means):+.5f}])")
        print(f"  bulk <J_r> median across regimes: "
              f"{np.median(bulk_means):+.5f}  "
              f"(range [{np.min(bulk_means):+.5f}, {np.max(bulk_means):+.5f}])")
        print(f"  p99  frac_J_r_neg median: {np.median(p99_neg_frac):.2%}")
        print(f"  bulk frac_J_r_neg median: {np.median(bulk_neg_frac):.2%}")
        print(f"  p99  Spearman(M, J_r) median: {np.median(sp_p99_M):+.3f}")
        print(f"  bulk Spearman(M, J_r) median: {np.median(sp_bulk_M):+.3f}")
        n_p99_attr = sum(1 for x in p99_means if x < 0)
        n_bulk_repul = sum(1 for x in bulk_means if x > 0)
        print(f"  p99 attractive (J_r<0):  {n_p99_attr}/{len(p99_means)} regimes")
        print(f"  bulk repulsive (J_r>0):  {n_bulk_repul}/{len(bulk_means)} regimes")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
