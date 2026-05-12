"""Optimization bundle covering items 1-4 of the per-component
halo audit follow-up:

  (1) AICc model comparison: Symanzik 1/N vs 1/N^2 fits to
      rho(X,T), rho(X,d), rho(T,d) on the within-P5 ladder.

  (2) Bootstrap-CI on the Symanzik continuum-limit asymptotes:
      5000 within-P5 seed resamples per regime, refit, retain
      median + CI95 of y_inf for each (correlation, model) pair.

  (3) R_00 *value-shuffle* null (replaces the weak mask-shuffle
      null): with d_actual fixed for each regime, shuffle |R_00|
      across nodes 500 times and compute the null Spearman
      distribution. The null is centred at zero and the observed
      |rho| is then a clean significance statement.

  (4) Geometry-side variance decomposition: X = G_00 + Lambda_t
      with Lambda_t a constant, so by construction the
      Spearman of X against any node-indexed quantity equals
      the Spearman of G_00 against the same quantity. The audit
      verifies this numerically (it would expose a bug in the
      reduction otherwise) and reports rho(G_00, d), rho(G_00, T)
      separately so the geometry-side halo is unambiguously
      pinned to the Hessian-Ricci tensor and not to the
      back-reaction constant.

Reads:
  outputs/verify_halo_geometry_vs_source_decomposition.json
  + per-seed lattice data via the existing per_seed_galerkin
  pipeline.

Output: outputs/verify_halo_optimization_bundle.json
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

OUT = REPO / "outputs" / "verify_halo_optimization_bundle.json"
GEOM_JSON = REPO / "outputs" / "verify_halo_geometry_vs_source_decomposition.json"
P5_REGIMES = ("P5", "P5N64", "P5N72", "P5N84", "P5N100", "P5N128", "P5N200", "P5N256", "P5N300", "P5N512")
N_BOOTSTRAP_FIT = 5000
N_NULL_DRAWS = 500
RNG_SEED = 20260503
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


def _symanzik_fit(N, y, order):
    """order=1: y ~ y_inf + a/N. order=2: y ~ y_inf + a/N^2."""
    x = 1.0 / N**order
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    rss = float(((y - pred) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    R2 = 1 - rss / tss if tss > 0 else float("nan")
    return float(coef[0]), float(coef[1]), R2, rss


def _aicc(rss, n, k):
    """Corrected AIC for small samples. k = number of parameters
    (intercept + 1 slope = 2)."""
    if n - k - 1 <= 0:
        return float("nan")
    sigma2 = rss / n
    if sigma2 <= 0:
        return float("-inf")
    return float(n * math.log(sigma2) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def _bootstrap_symanzik(N, y_per_regime_seed_lists, order, n_boot, rng):
    """y_per_regime_seed_lists: list of arrays, one per regime; each
    holds the per-seed correlation values for that regime. We resample
    seeds with replacement within each regime, take the seed-mean
    per regime, refit Symanzik, and store y_inf across draws."""
    boots = []
    for _ in range(n_boot):
        y_resampled = np.empty(len(N))
        for j, vals in enumerate(y_per_regime_seed_lists):
            if len(vals) == 0:
                y_resampled[j] = np.nan
            else:
                draw = rng.choice(vals, size=len(vals), replace=True)
                y_resampled[j] = float(np.mean(draw))
        if not np.all(np.isfinite(y_resampled)):
            continue
        y_inf, _, _, _ = _symanzik_fit(N, y_resampled, order)
        boots.append(y_inf)
    boots = np.asarray(boots)
    if len(boots) < 10:
        return None
    return {
        "n_draws": int(len(boots)),
        "median": float(np.median(boots)),
        "ci95_lo": float(np.quantile(boots, 0.025)),
        "ci95_hi": float(np.quantile(boots, 0.975)),
        "std": float(boots.std()),
    }


def _per_seed_correlations(xi_mat, psi, k_field, q_field, n_lat):
    """Return per-seed: rho(X,T), rho(X,d), rho(T,d), rho(G_00,d),
    rho(G_00,T), and pooled raw arrays |R_00| and d_nearest for the
    value-shuffle null."""
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]
    t00 = prep["t00"]
    X = g_00 + LAMBDA_T
    R_00 = X - t00

    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs(adj, top_idx, d_mat_inf)
    finite = np.isfinite(d_nearest)
    return {
        "rho_X_T": _spearman(X[finite], t00[finite]),
        "rho_X_d": _spearman(X[finite], d_nearest[finite]),
        "rho_T_d": _spearman(t00[finite], d_nearest[finite]),
        "rho_G00_d": _spearman(g_00[finite], d_nearest[finite]),
        "rho_G00_T": _spearman(g_00[finite], t00[finite]),
        "abs_R_00_pooled": np.abs(R_00[finite]),
        "d_nearest_pooled": d_nearest[finite],
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
    per_seed = []
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed_correlations(np.asarray(xi_mat), np.asarray(psi),
                                          np.asarray(k_field),
                                          np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        per_seed.append(r)
    if not per_seed:
        return None
    return {"regime": regime, "N": int(n_lat), "per_seed": per_seed}


def _value_shuffle_null(per_regime, n_draws, rng):
    """For each regime, pool |R_00| and d_nearest across seeds. Observed
    rho = Spearman(|R_00|, d). Null: shuffle |R_00| across all pooled
    nodes n_draws times, recompute rho. Return per-regime z-score."""
    out = {}
    for r in per_regime:
        regime = r["regime"]
        absR = np.concatenate([s["abs_R_00_pooled"] for s in r["per_seed"]])
        d = np.concatenate([s["d_nearest_pooled"] for s in r["per_seed"]])
        rho_obs = _spearman(absR, d)
        if not np.isfinite(rho_obs) or len(absR) < 30:
            out[regime] = None
            continue
        nulls = []
        for _ in range(n_draws):
            shuf = rng.permutation(absR)
            nulls.append(_spearman(shuf, d))
        nulls = np.asarray([x for x in nulls if np.isfinite(x)])
        if len(nulls) < 10:
            out[regime] = None
            continue
        mu = float(nulls.mean())
        sig = float(nulls.std())
        z = (rho_obs - mu) / (sig + 1e-12)
        # Two-sided p via standard normal approximation
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        out[regime] = {
            "N": int(r["N"]),
            "rho_observed": rho_obs,
            "null_n": int(len(nulls)),
            "null_mean": mu,
            "null_std": sig,
            "z_score": float(z),
            "p_two_sided": float(p),
            "n_pooled": int(len(absR)),
        }
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    print("(*) Re-running per-seed correlations for items 1-4 ...")
    per_regime = []
    for regime, n_lat in LADDER:
        print(f"   {regime:>10s}  N={n_lat:>4d} ...")
        r = _process_regime(regime, n_lat, max_seeds=24)
        if r is None:
            continue
        per_regime.append(r)

    # ----- Item 1 + 2: Symanzik on within-P5 with AICc + bootstrap -----
    print()
    print("(*) Item 1+2: AICc Symanzik 1/N vs 1/N^2 + bootstrap CI ...")
    p5 = [r for r in per_regime if r["regime"] in P5_REGIMES]
    p5.sort(key=lambda x: x["N"])
    N_arr = np.array([x["N"] for x in p5], float)

    rho_keys = [("rho_X_T",  "rho(X,T)"),
                ("rho_X_d",  "rho(X,d)"),
                ("rho_T_d",  "rho(T,d)"),
                ("rho_G00_d", "rho(G_00,d)"),
                ("rho_G00_T", "rho(G_00,T)")]
    fit_results = {}
    for key, label in rho_keys:
        per_seed_lists = [
            np.array([s[key] for s in r["per_seed"]
                       if np.isfinite(s[key])])
            for r in p5
        ]
        seed_means = np.array([
            np.mean(v) if len(v) > 0 else np.nan
            for v in per_seed_lists
        ])
        valid = np.isfinite(seed_means)
        if valid.sum() < 4:
            continue
        Nv = N_arr[valid]; yv = seed_means[valid]
        psl_v = [per_seed_lists[i] for i in range(len(N_arr)) if valid[i]]

        y_inf1, a1, R2_1, rss1 = _symanzik_fit(Nv, yv, 1)
        y_inf2, a2, R2_2, rss2 = _symanzik_fit(Nv, yv, 2)
        n_pts = int(valid.sum())
        aicc1 = _aicc(rss1, n_pts, 2)
        aicc2 = _aicc(rss2, n_pts, 2)
        delta = aicc2 - aicc1  # positive => 1/N preferred
        boot1 = _bootstrap_symanzik(Nv, psl_v, 1, N_BOOTSTRAP_FIT, rng)
        boot2 = _bootstrap_symanzik(Nv, psl_v, 2, N_BOOTSTRAP_FIT, rng)
        fit_results[key] = {
            "label": label,
            "n_points": n_pts,
            "symanzik_1_over_N":  {
                "y_inf": y_inf1, "a": a1, "R2": R2_1, "AICc": aicc1,
                "bootstrap_y_inf": boot1,
            },
            "symanzik_1_over_N2": {
                "y_inf": y_inf2, "a": a2, "R2": R2_2, "AICc": aicc2,
                "bootstrap_y_inf": boot2,
            },
            "delta_AICc_1over_N2_minus_1over_N": delta,
            "model_preferred": ("1/N" if delta > 2 else
                                ("1/N^2" if delta < -2 else
                                 "indistinguishable")),
        }
        prefer = fit_results[key]["model_preferred"]
        print(f"   {label:>14s}: 1/N y_inf={y_inf1:+.3f} R2={R2_1:.2f} AICc={aicc1:.2f}  "
              f"|  1/N^2 y_inf={y_inf2:+.3f} R2={R2_2:.2f} AICc={aicc2:.2f}  "
              f"|  delta_AICc={delta:+.2f}  preferred={prefer}")

    # ----- Item 3: R_00 value-shuffle null -----
    print()
    print("(*) Item 3: R_00 value-shuffle null bootstrap ...")
    null_results = _value_shuffle_null(per_regime, N_NULL_DRAWS, rng)
    n_strong = 0
    for reg, v in null_results.items():
        if v is None:
            continue
        z = v["z_score"]
        if abs(z) > 5:
            n_strong += 1
        print(f"   {reg:>10s}  N={v['N']:>4d}  rho_obs={v['rho_observed']:+.3f}  "
              f"null_mu={v['null_mean']:+.4f}  null_std={v['null_std']:.4f}  "
              f"|z|={abs(z):.1f}  p={v['p_two_sided']:.2e}")
    print(f"   ===> {n_strong}/{len(null_results)} regimes with |z|>5")

    # ----- Item 4: G_00 alone vs Lambda_t -----
    print()
    print("(*) Item 4: G_00 vs Lambda_t decomposition ...")
    print("   X = G_00 + Lambda_t, with Lambda_t = const = alpha_xi^2 = 0.81")
    print("   Therefore Spearman(X, *) == Spearman(G_00, *) by definition")
    item4 = []
    for r in per_regime:
        means = {}
        for key in ("rho_X_d", "rho_X_T", "rho_G00_d", "rho_G00_T"):
            vals = [s[key] for s in r["per_seed"] if np.isfinite(s[key])]
            means[key] = float(np.mean(vals)) if vals else float("nan")
        diff_d = means["rho_X_d"] - means["rho_G00_d"]
        diff_T = means["rho_X_T"] - means["rho_G00_T"]
        item4.append({
            "regime": r["regime"], "N": r["N"],
            "rho_X_d_mean": means["rho_X_d"],
            "rho_G00_d_mean": means["rho_G00_d"],
            "rho_X_T_mean": means["rho_X_T"],
            "rho_G00_T_mean": means["rho_G00_T"],
            "abs_diff_X_minus_G00_d": abs(diff_d),
            "abs_diff_X_minus_G00_T": abs(diff_T),
        })
        print(f"   {r['regime']:>10s} N={r['N']:>4d}  "
              f"rho(X,d)={means['rho_X_d']:+.3f} == "
              f"rho(G_00,d)={means['rho_G00_d']:+.3f}? "
              f"diff={diff_d:+.5f}")

    out_dict = {
        "method": "Optimization bundle items 1-4",
        "n_bootstrap_fit": N_BOOTSTRAP_FIT,
        "n_null_draws": N_NULL_DRAWS,
        "p5_within_ladder": list(P5_REGIMES),
        "core_top_frac": CORE_TOP_FRAC,
        "item_1_2_AICc_bootstrap_symanzik": fit_results,
        "item_3_value_shuffle_null": null_results,
        "item_4_X_vs_G00_decomposition": item4,
    }
    OUT.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
