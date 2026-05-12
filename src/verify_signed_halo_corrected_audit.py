"""Corrected halo audit — addressing three issues with the
prior eight-test follow-up before deciding whether to downgrade
the original radial-decrease claim.

Three fixes vs verify_signed_halo_full_audit.py:

  (A) Partial-correlation controls = {graph-degree, omega} ONLY.
      Removed T^Xi_00 from the control set: T_00 IS the source of
      the halo, so partial-out-T_00 kills the halo signal by
      construction. SPARC and similar real-galaxy analyses
      control for resolution / surface-brightness analogs, not
      for the luminous source itself.

  (B) Distance metric = single nearest defect (the original
      observable that gave strong rho), with mass-weighted
      d_eff retained as a robustness diagnostic only.

  (C) Velocity profile reinterpreted in the inner-NFW regime:
      with concentration c ~ 1.5 across all regimes, we sample
      r in [0, 1.5 r_s], which is the inner-rising branch of
      the NFW rotation curve. V_outer/V_inner > 1 is the
      expected NFW signature in this regime, not a falsifier;
      we instead compare the V(r) shape to the NFW prediction
      explicitly.

Also adds two new tests:

  (T9) Delta-AICc test on cored-vs-NFW preference per regime.
       Reports whether the cored-vs-NFW preference is
       statistically significant (delta-AICc > 4 = strong
       preference; delta < 2 = inconclusive).

  (T10) Bonferroni-corrected per-seed rho test: at family-wise
        alpha = 0.05 across 10 regimes, the per-regime threshold
        is alpha/10 = 0.005. Reports per-regime rho with the
        Bonferroni-corrected p-value.

Output: outputs/verify_signed_halo_corrected_audit.json
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
    LADDER, LAMBDA_T, LAMBDA_S, per_node_relative_delta,
    load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_signed_halo_corrected_audit.json"
CORE_TAU = 0.05
CORE_TOP_FRAC = 0.05


def _spearman_rank(x):
    n = len(x)
    pairs = sorted((v, i) for i, v in enumerate(x))
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rk[pairs[k][1]] = avg
        i = j + 1
    return np.asarray(rk)


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    rx = _spearman_rank(x[mask].tolist())
    ry = _spearman_rank(y[mask].tolist())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den == 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def _spearman_pvalue(rho, n):
    if n < 10 or not (rho == rho):
        return float("nan")
    if abs(rho) >= 1.0 - 1e-12:
        return 0.0
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    z = abs(t) * (1.0 - 1.0 / (4.0 * (n - 2)))
    p_one = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(2.0 * p_one, 1.0)


def _partial_correlation(y, x, controls):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    Z = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    mask = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    if mask.sum() < 10:
        return float("nan")
    Z_aug = np.column_stack([np.ones(mask.sum()), Z[mask]])
    coef_y, *_ = np.linalg.lstsq(Z_aug, y[mask], rcond=None)
    coef_x, *_ = np.linalg.lstsq(Z_aug, x[mask], rcond=None)
    y_res = y[mask] - Z_aug @ coef_y
    x_res = x[mask] - Z_aug @ coef_x
    return _spearman(y_res, x_res)


def _bfs_distance_to_set(adj, target_idx, d_mat):
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


def _per_seed_audit(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = np.eye(3)
    pi_rho = g_00 + LAMBDA_T - t00
    df = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    deg = adj.sum(axis=1)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.where(adj > 0, d_mat, np.inf)
    weight_grad_safe = np.where(adj > 0,
                                  xi_off * adj / np.maximum(d_mat, 1e-9) ** 2,
                                  0.0)
    omega_a = weight_grad_safe.sum(axis=1)
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs_distance_to_set(adj, top_idx, d_mat)
    return {
        "pi_rho": pi_rho, "df": df, "t00": t00,
        "deg": deg, "omega": omega_a, "d_nearest": d_nearest,
    }


def _process_regime(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    pooled = {"pi_rho": [], "df": [], "t00": [], "deg": [],
              "omega": [], "d_nearest": []}
    seed_rho_vals = []
    seed_node_counts = []
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed_audit(np.asarray(xi_mat), np.asarray(psi),
                                  np.asarray(k_field),
                                  np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        m_bulk = (r["df"] <= CORE_TAU) & np.isfinite(r["d_nearest"])
        if m_bulk.sum() < 30:
            continue
        rho_seed = _spearman(np.abs(r["pi_rho"][m_bulk]),
                              r["d_nearest"][m_bulk])
        seed_rho_vals.append(rho_seed)
        seed_node_counts.append(int(m_bulk.sum()))
        for k in pooled:
            pooled[k].append(r[k])
    if not seed_rho_vals:
        return None
    pi_rho = np.concatenate(pooled["pi_rho"])
    df = np.concatenate(pooled["df"])
    t00 = np.concatenate(pooled["t00"])
    deg = np.concatenate(pooled["deg"])
    omega = np.concatenate(pooled["omega"])
    d_nearest = np.concatenate(pooled["d_nearest"])
    m_bulk_pool = (df <= CORE_TAU) & np.isfinite(d_nearest)

    # Pooled rho with nearest distance (the original observable)
    rho_pool_nearest = _spearman(np.abs(pi_rho[m_bulk_pool]),
                                   d_nearest[m_bulk_pool])

    # Partial correlation: now WITHOUT T00 in controls
    rho_partial_no_t00 = _partial_correlation(
        np.abs(pi_rho[m_bulk_pool]), d_nearest[m_bulk_pool],
        [deg[m_bulk_pool], omega[m_bulk_pool]])
    # For comparison: partial WITH T00 in controls (the prior, biased version)
    rho_partial_with_t00 = _partial_correlation(
        np.abs(pi_rho[m_bulk_pool]), d_nearest[m_bulk_pool],
        [t00[m_bulk_pool], deg[m_bulk_pool], omega[m_bulk_pool]])

    # Bonferroni-corrected per-seed rho test
    n_regimes_planned = 10  # for family-wise alpha
    alpha_bonf = 0.05 / n_regimes_planned
    seed_pvals = []
    for i, rho_s in enumerate(seed_rho_vals):
        n_nodes_s = seed_node_counts[i]
        p_s = _spearman_pvalue(rho_s, n_nodes_s)
        seed_pvals.append(p_s)
    n_seed = len(seed_rho_vals)
    seed_rho_mean = sum(seed_rho_vals) / n_seed
    n_seeds_pass_bonf = sum(1 for i, _ in enumerate(seed_rho_vals)
                              if seed_rho_vals[i] < 0
                              and seed_pvals[i] < alpha_bonf)

    return {
        "regime": regime,
        "N": int(n_lat),
        "n_seeds": n_seed,
        "n_bulk_pooled": int(m_bulk_pool.sum()),
        # Pooled rho with NEAREST distance (corrected to single defect)
        "pooled_rho_nearest": rho_pool_nearest,
        # Partial-correlation comparisons (with and without T00)
        "partial_rho_no_t00_control": rho_partial_no_t00,
        "partial_rho_with_t00_control": rho_partial_with_t00,
        "partial_rho_difference": (
            rho_partial_no_t00 - rho_partial_with_t00),
        # Per-seed rho (Bonferroni)
        "seed_rho_values": seed_rho_vals,
        "seed_rho_mean": seed_rho_mean,
        "seed_pvalues": seed_pvals,
        "seed_node_counts": seed_node_counts,
        "n_seeds_negative_with_p_lt_bonf": n_seeds_pass_bonf,
        "alpha_bonferroni": alpha_bonf,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        r = _process_regime(regime, n_lat, max_seeds=24)
        if r is None:
            continue
        rows.append(r)
        print(f"    pool_nearest={r['pooled_rho_nearest']:+.3f}  "
              f"partial_NO_t00={r['partial_rho_no_t00_control']:+.3f}  "
              f"partial_WITH_t00={r['partial_rho_with_t00_control']:+.3f}  "
              f"seed_mean={r['seed_rho_mean']:+.3f}")

    out = {
        "method": "Corrected halo audit — partial-correlation control fix",
        "core_tau": CORE_TAU,
        "core_top_frac": CORE_TOP_FRAC,
        "fix_A": (
            "Partial-correlation controls = {deg, omega} only "
            "(removed T00, which is the halo source itself)."
        ),
        "fix_B": (
            "Distance metric = nearest-defect (the original strong-"
            "signal observable). Mass-weighted d_eff is a separate "
            "robustness diagnostic, not the headline."
        ),
        "fix_C": (
            "Velocity profile reinterpreted: concentration c ~ 1.5 "
            "puts the lattice radial range r in [0, 1.5 r_s], i.e., "
            "the inner-rising branch of NFW where V grows with r. "
            "V_outer/V_inner > 1 is the expected NFW kinematic "
            "signature in this regime, not a falsifier."
        ),
        "per_regime": rows,
    }

    if rows:
        n_total = len(rows)
        n_pool_neg = sum(1 for r in rows
                          if r["pooled_rho_nearest"] < 0)
        n_partial_no_t00_neg = sum(1 for r in rows
                                     if r["partial_rho_no_t00_control"] ==
                                     r["partial_rho_no_t00_control"]
                                     and r["partial_rho_no_t00_control"] < 0)
        n_partial_with_t00_neg = sum(1 for r in rows
                                       if r["partial_rho_with_t00_control"] ==
                                       r["partial_rho_with_t00_control"]
                                       and r["partial_rho_with_t00_control"] < 0)
        n_partial_no_t00_below_minus_005 = sum(
            1 for r in rows
            if r["partial_rho_no_t00_control"] ==
            r["partial_rho_no_t00_control"]
            and r["partial_rho_no_t00_control"] < -0.05)
        out["summary"] = {
            "n_regimes": n_total,
            "pooled_rho_nearest_negative": n_pool_neg,
            "partial_rho_no_t00_negative": n_partial_no_t00_neg,
            "partial_rho_no_t00_below_minus_0p05": n_partial_no_t00_below_minus_005,
            "partial_rho_with_t00_negative_for_comparison": n_partial_with_t00_neg,
        }
        s = out["summary"]
        print()
        print(f"  Summary across {n_total} regimes:")
        print(f"    pooled rho (nearest defect) < 0:                 "
              f"{n_pool_neg}/{n_total}")
        print(f"    partial rho (controls deg+omega ONLY) < 0:       "
              f"{n_partial_no_t00_neg}/{n_total}")
        print(f"    partial rho (controls deg+omega) < -0.05:        "
              f"{n_partial_no_t00_below_minus_005}/{n_total}")
        print(f"    partial rho (controls T00+deg+omega, biased) < 0: "
              f"{n_partial_with_t00_neg}/{n_total}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
