"""Three-projection halo audit.

Tests the radial halo signature under three explicit projections,
none mixed:

  Pi_rho R(a) = u^mu u^nu R_munu(a) = R_00(a)        (energy density)
  Pi_F   R(a) = ||R(a)||_F / ||T^Xi(a)||_F           (norm ratio = Delta_N)
  Pi_tr  R(a) = tr R(a) = R_00 + R_11 + R_22 + R_33   (trace, mixed)

Each projection is regressed against the nearest-defect distance,
with controls = {deg, omega} (NOT T_00 — that is the source of the
halo). All three are reported per regime; we check which (if any)
gives a robust radial signal.

Output: outputs/verify_signed_halo_three_projections.json
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

OUT = REPO / "outputs" / "verify_signed_halo_three_projections.json"
CORE_TAU = 0.05
CORE_TOP_FRAC = 0.05


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


def _partial_corr(y, x, controls):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    Z = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    mask = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    if mask.sum() < 10:
        return float("nan")
    Z_aug = np.column_stack([np.ones(mask.sum()), Z[mask]])
    cy, *_ = np.linalg.lstsq(Z_aug, y[mask], rcond=None)
    cx, *_ = np.linalg.lstsq(Z_aug, x[mask], rcond=None)
    return _spearman(y[mask] - Z_aug @ cy, x[mask] - Z_aug @ cx)


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
    res_00 = g_00 + LAMBDA_T - t00
    res_d = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij
    pi_rho = res_00
    pi_tr = res_00 + res_d[:, 0, 0] + res_d[:, 1, 1] + res_d[:, 2, 2]
    pi_F = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]

    df = pi_F  # delta_full is the relative Frobenius residual
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
    return pi_rho, pi_tr, pi_F, df, deg, omega_a, d_nearest


def _process(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    pools = {k: [] for k in ("rho", "tr", "F", "df", "deg", "omg", "d")}
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed(np.asarray(xi_mat), np.asarray(psi),
                            np.asarray(k_field), np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        pi_rho, pi_tr, pi_F, df, deg, omg, d = r
        m = (df <= CORE_TAU) & np.isfinite(d)
        if m.sum() < 30:
            continue
        pools["rho"].append(pi_rho[m])
        pools["tr"].append(pi_tr[m])
        pools["F"].append(pi_F[m])
        pools["df"].append(df[m])
        pools["deg"].append(deg[m])
        pools["omg"].append(omg[m])
        pools["d"].append(d[m])
    if not pools["rho"]:
        return None
    pi_rho_b = np.concatenate(pools["rho"])
    pi_tr_b = np.concatenate(pools["tr"])
    pi_F_b = np.concatenate(pools["F"])
    deg_b = np.concatenate(pools["deg"])
    omg_b = np.concatenate(pools["omg"])
    d_b = np.concatenate(pools["d"])
    n_b = len(pi_rho_b)
    out = {
        "regime": regime, "N": int(n_lat),
        "n_bulk_pooled": n_b,
        # Plain Spearman per projection
        "rho_spearman_pi_rho_vs_d": _spearman(np.abs(pi_rho_b), d_b),
        "rho_spearman_pi_tr_vs_d": _spearman(np.abs(pi_tr_b), d_b),
        "rho_spearman_pi_F_vs_d": _spearman(pi_F_b, d_b),
        # Partial correlation, controls = deg + omega only
        "partial_pi_rho_vs_d": _partial_corr(
            np.abs(pi_rho_b), d_b, [deg_b, omg_b]),
        "partial_pi_tr_vs_d": _partial_corr(
            np.abs(pi_tr_b), d_b, [deg_b, omg_b]),
        "partial_pi_F_vs_d": _partial_corr(
            pi_F_b, d_b, [deg_b, omg_b]),
    }
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
        print(f"    rho_spear: rho={r['rho_spearman_pi_rho_vs_d']:+.3f}  "
              f"tr={r['rho_spearman_pi_tr_vs_d']:+.3f}  "
              f"F={r['rho_spearman_pi_F_vs_d']:+.3f}  |  "
              f"partial: rho={r['partial_pi_rho_vs_d']:+.3f}  "
              f"tr={r['partial_pi_tr_vs_d']:+.3f}  "
              f"F={r['partial_pi_F_vs_d']:+.3f}")

    if rows:
        n_total = len(rows)
        out = {
            "method": "Three-projection halo audit",
            "core_tau": CORE_TAU,
            "core_top_frac": CORE_TOP_FRAC,
            "controls_for_partial_correlation": [
                "graph_degree", "weight_density_omega"],
            "T00_intentionally_NOT_in_controls": (
                "T_00 is the source of the halo; controlling for it "
                "would kill the halo signal by construction."),
            "per_regime": rows,
            "summary": {
                "n_regimes": n_total,
                "n_pi_rho_spearman_negative":
                    sum(1 for r in rows if r["rho_spearman_pi_rho_vs_d"] < 0),
                "n_pi_tr_spearman_negative":
                    sum(1 for r in rows if r["rho_spearman_pi_tr_vs_d"] < 0),
                "n_pi_F_spearman_negative":
                    sum(1 for r in rows if r["rho_spearman_pi_F_vs_d"] < 0),
                "n_pi_rho_partial_negative":
                    sum(1 for r in rows
                        if r["partial_pi_rho_vs_d"] ==
                        r["partial_pi_rho_vs_d"]
                        and r["partial_pi_rho_vs_d"] < 0),
                "n_pi_tr_partial_negative":
                    sum(1 for r in rows
                        if r["partial_pi_tr_vs_d"] ==
                        r["partial_pi_tr_vs_d"]
                        and r["partial_pi_tr_vs_d"] < 0),
                "n_pi_F_partial_negative":
                    sum(1 for r in rows
                        if r["partial_pi_F_vs_d"] ==
                        r["partial_pi_F_vs_d"]
                        and r["partial_pi_F_vs_d"] < 0),
                "n_pi_F_partial_below_minus_0p10":
                    sum(1 for r in rows
                        if r["partial_pi_F_vs_d"] ==
                        r["partial_pi_F_vs_d"]
                        and r["partial_pi_F_vs_d"] < -0.10),
            },
        }
        s = out["summary"]
        print()
        print(f"  Summary across {n_total} regimes:")
        print(f"    Plain Spearman < 0: rho={s['n_pi_rho_spearman_negative']}  "
              f"tr={s['n_pi_tr_spearman_negative']}  "
              f"F={s['n_pi_F_spearman_negative']}")
        print(f"    Partial (deg,omg) < 0: rho={s['n_pi_rho_partial_negative']}  "
              f"tr={s['n_pi_tr_partial_negative']}  "
              f"F={s['n_pi_F_partial_negative']}")
        print(f"    Partial F < -0.10:        "
              f"{s['n_pi_F_partial_below_minus_0p10']}/{n_total}")
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
