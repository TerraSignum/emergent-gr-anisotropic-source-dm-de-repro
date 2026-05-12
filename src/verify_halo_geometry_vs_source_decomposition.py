"""Variance decomposition of the halo signal.

The per-node residual is, by construction,
   R_00(a) = G_00(a) + Lambda_t - T_00(a) = X(a) - T_00(a),
with X(a) = G_00(a) + Lambda_t the geometry-side amplitude.
The observed cross-correlation rho(R_00, T_00) ~= -0.94 is
trivially close to -1 if Var(T_00) dominates Var(X), since
then R_00 ~= -T_00 modulo small fluctuations.

This audit decomposes the halo signal into the parts that
come from the geometry side X and the parts that come from
the source side T_00:

  (a) Var ratio:  Var(X) vs Var(T_00) per regime, and what
      fraction of Var(R_00) is from each side and from their
      cross-covariance:
         Var(R_00) = Var(X) + Var(T_00) - 2 Cov(X, T_00).
  (b) rho(X, d_nearest)        — does geometry alone show
                                   the halo radial pattern?
  (c) rho(T_00, d_nearest)     — and does the source alone?
  (d) rho(X, T_00)             — do geometry and source
                                   actually track each other
                                   per node?
  (e) Per-shell median of X(a) — same shell decomposition as
                                   for R_00 but on the
                                   geometry side alone.

If X has a regime-stable rho with d_nearest of similar sign
and magnitude as R_00, the halo is structural on the
geometry side and not just an arithmetic flip of T_00.

Output: outputs/verify_halo_geometry_vs_source_decomposition.json
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
    LADDER, LAMBDA_T, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_halo_geometry_vs_source_decomposition.json"
CORE_TOP_FRAC = 0.05
N_SHELLS = 5


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xs = x[mask]; ys = y[mask]
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
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
        u = -1; best = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]; u = i
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
    t00 = prep["t00"]
    X = g_00 + LAMBDA_T          # geometry-side amplitude
    R_00 = X - t00               # residual
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.where(adj > 0, d_mat, np.inf)
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs(adj, top_idx, d_mat)
    return X, t00, R_00, d_nearest


def _shells(values, d, n_shells=N_SHELLS):
    finite = np.isfinite(values) & np.isfinite(d)
    v = values[finite]; dn = d[finite]
    if len(v) < 30:
        return None
    edges = np.quantile(dn, np.linspace(0, 1, n_shells + 1))
    edges[-1] += 1e-9
    out = []
    for k in range(n_shells):
        m = (dn >= edges[k]) & (dn < edges[k + 1])
        if m.sum() < 3:
            out.append({"n": int(m.sum()), "median": float("nan")})
            continue
        out.append({
            "n": int(m.sum()),
            "d_low": float(edges[k]),
            "d_high": float(edges[k + 1]),
            "median": float(np.median(v[m])),
            "mean": float(v[m].mean()),
        })
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
    pools = {k: [] for k in ("X", "T", "R", "d")}
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            X, T, R, d = _per_seed(np.asarray(xi_mat), np.asarray(psi),
                                       np.asarray(k_field),
                                       np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        m = np.isfinite(d)
        if m.sum() < 30:
            continue
        pools["X"].append(X[m])
        pools["T"].append(T[m])
        pools["R"].append(R[m])
        pools["d"].append(d[m])
    if not pools["X"]:
        return None
    X = np.concatenate(pools["X"])
    T = np.concatenate(pools["T"])
    R = np.concatenate(pools["R"])
    d = np.concatenate(pools["d"])

    var_X = float(X.var())
    var_T = float(T.var())
    var_R = float(R.var())
    cov_XT = float(np.cov(X, T)[0, 1])
    # Var(R) = Var(X) + Var(T) - 2 Cov(X,T)  [since R = X - T]
    decomp_check = var_X + var_T - 2 * cov_XT
    return {
        "regime": regime, "N": int(n_lat), "n_pooled": int(len(d)),
        "var_X_geom": var_X,
        "var_T_source": var_T,
        "var_R_residual": var_R,
        "cov_X_T": cov_XT,
        "var_decomposition_check": decomp_check,
        "var_ratio_X_over_T": var_X / max(var_T, 1e-30),
        # Halo correlations: geometry alone, source alone, residual
        "rho_X_vs_d": _spearman(X, d),
        "rho_absX_vs_d": _spearman(np.abs(X), d),
        "rho_T_vs_d": _spearman(T, d),
        "rho_absT_vs_d": _spearman(np.abs(T), d),
        "rho_R_vs_d": _spearman(R, d),
        "rho_absR_vs_d": _spearman(np.abs(R), d),
        # Geometry vs source coupling per node
        "rho_X_vs_T": _spearman(X, T),
        "rho_R_vs_T": _spearman(R, T),
        # Mean values
        "mean_X": float(X.mean()),
        "mean_T": float(T.mean()),
        "mean_R": float(R.mean()),
        # Shell decomposition for X (geometry side)
        "X_shells_by_d": _shells(X, d),
        "T_shells_by_d": _shells(T, d),
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
        print(f"    Var(X)/Var(T) = {r['var_ratio_X_over_T']:.4f}  "
              f"(X = G_00+Lambda_t)")
        print(f"    rho(X,d) = {r['rho_X_vs_d']:+.3f}  "
              f"rho(T,d) = {r['rho_T_vs_d']:+.3f}  "
              f"rho(R,d) = {r['rho_R_vs_d']:+.3f}")
        print(f"    rho(X,T) = {r['rho_X_vs_T']:+.3f}  "
              f"(geometry vs source per-node)")

    if rows:
        med_var_ratio = float(np.median(
            [r["var_ratio_X_over_T"] for r in rows]))
        med_rho_X_d = float(np.median([r["rho_X_vs_d"] for r in rows]))
        med_rho_T_d = float(np.median([r["rho_T_vs_d"] for r in rows]))
        med_rho_R_d = float(np.median([r["rho_R_vs_d"] for r in rows]))
        med_rho_XT = float(np.median([r["rho_X_vs_T"] for r in rows]))
        n_X_pos = sum(1 for r in rows if r["rho_X_vs_d"] > 0.10)
        n_X_neg = sum(1 for r in rows if r["rho_X_vs_d"] < -0.10)
        out = {
            "method": "Geometry vs source decomposition of the halo signal",
            "definition": ("R_00 = X - T with X = G_00 + Lambda_t. "
                            "If Var(T) >> Var(X), R_00 ~= -T trivially."),
            "core_top_frac": CORE_TOP_FRAC,
            "per_regime": rows,
            "summary": {
                "n_regimes": len(rows),
                "median_var_ratio_X_over_T": med_var_ratio,
                "median_rho_geometry_vs_d": med_rho_X_d,
                "median_rho_source_vs_d": med_rho_T_d,
                "median_rho_residual_vs_d": med_rho_R_d,
                "median_rho_geometry_vs_source": med_rho_XT,
                "n_geometry_halo_positive": n_X_pos,
                "n_geometry_halo_negative": n_X_neg,
            },
        }
        s = out["summary"]
        print()
        print(f"  Summary across {s['n_regimes']} regimes:")
        print(f"    median Var(X)/Var(T) = {s['median_var_ratio_X_over_T']:.4f}")
        print(f"    median rho(X,d)        = {s['median_rho_geometry_vs_d']:+.3f}  "
              f"(geometry-only halo signal)")
        print(f"    median rho(T,d)        = {s['median_rho_source_vs_d']:+.3f}  "
              f"(source-only halo signal)")
        print(f"    median rho(R,d)        = {s['median_rho_residual_vs_d']:+.3f}  "
              f"(residual halo signal)")
        print(f"    median rho(X,T)        = {s['median_rho_geometry_vs_source']:+.3f}  "
              f"(per-node geometry-vs-source coupling)")
        print(f"    rho(X,d) regimes: {n_X_pos} positive (>0.10), "
              f"{n_X_neg} negative (<-0.10)")
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
