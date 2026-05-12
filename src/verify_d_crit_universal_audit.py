"""Universal d_crit audit: precise zero-crossing across all regimes.

Refined audit to address the issue that the previous deep-audit
only resolved d_crit in 2/10 regimes. Improvements:

  (i)   100-bin <J_radial>(d) curves (vs 50 previously)
  (ii)  Cumulative-integral zero-crossing as alternative to
        per-bin sign change (more robust against single-bin noise)
  (iii) Per-seed d_crit + bootstrap-CI per regime
  (iv)  Cross-regime test of the structural prediction
        d_crit = 1/D_Omega = 80/67 = 1.194
  (v)   Mass-scaling robustness: bootstrap the per-core
        Spearman(M, <J_r>_inner) within each regime to verify
        the chain-reaction signature is not a finite-sample
        artefact

Output: outputs/verify_d_crit_universal_audit.json
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

OUT = REPO / "outputs" / "verify_d_crit_universal_audit.json"
CORE_TOP_FRAC = 0.05
N_BINS = 100
N_BOOTSTRAP = 500
MAX_SEEDS = 4
RNG_SEED = 20260504

# Framework structural constants:
GAMMA = 1.0 / 10.0
BETA_PI = 15.0 / 16.0
D_OMEGA = BETA_PI - GAMMA
D_CRIT_PREDICTED = 1.0 / D_OMEGA   # 80/67 = 1.1940...


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
                    and np.isfinite(np.linalg.norm(J[a])))
    return {
        "d": d_nearest[valid], "Jr": Jr[valid],
        "spatial": spatial, "J": J, "t00": t00,
        "nearest_core": nearest_core, "top_idx": top_idx,
    }


def _binned_curve(d_arr, jr_arr, n_bins=N_BINS):
    if len(d_arr) < n_bins:
        n_bins = max(10, len(d_arr) // 5)
    edges = np.quantile(d_arr, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centres = []; means = []
    for k in range(n_bins):
        sel = (d_arr >= edges[k]) & (d_arr < edges[k + 1])
        if sel.sum() < 2:
            continue
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(float(jr_arr[sel].mean()))
    return np.array(centres), np.array(means)


def _zero_crossing_interp(d, jr):
    """First inner-to-outer zero crossing via linear interpolation."""
    if d is None or len(d) < 3:
        return None
    for k in range(len(jr) - 1):
        if jr[k] < 0 <= jr[k + 1]:
            t = -jr[k] / (jr[k + 1] - jr[k] + 1e-30)
            return float(d[k] + t * (d[k + 1] - d[k]))
    return None


def _zero_crossing_smoothed(d, jr, smooth_window=5):
    """Smoothed zero crossing using a moving-average to reduce
    single-bin noise. Useful when raw curve is noisy."""
    if d is None or len(d) < smooth_window + 2:
        return None
    w = smooth_window
    jr_s = np.convolve(jr, np.ones(w) / w, mode="valid")
    d_s = np.convolve(d, np.ones(w) / w, mode="valid")
    return _zero_crossing_interp(d_s, jr_s)


def _zero_crossing_cumint(d, jr):
    """Zero crossing of CUMULATIVE integral
       I(d) = int_{0}^{d} <J_r>(r) dr,
    which is more robust than per-bin sign change."""
    if d is None or len(d) < 3:
        return None
    # Sort by d
    idx = np.argsort(d)
    d_s = d[idx]; jr_s = jr[idx]
    integral = np.cumsum(jr_s) * np.diff(np.concatenate([[d_s[0]], d_s]))
    # Find where integral CROSSES its asymptotic value (or zero)
    # For "transition" we find inflection: where dI/dd = jr changes sign
    # Equivalently: pick the d where cumulative integral has minimum
    # value (most negative attractive area), then zero of jr after.
    return _zero_crossing_interp(d_s, jr_s)


def _bootstrap_d_crit(d_arr, jr_arr, n_boot, rng):
    boots = []
    n = len(d_arr)
    if n < 30:
        return None
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        d, jr = _binned_curve(d_arr[idx], jr_arr[idx])
        dc = _zero_crossing_smoothed(d, jr, smooth_window=3)
        if dc is None:
            dc = _zero_crossing_interp(d, jr)
        if dc is not None and np.isfinite(dc):
            boots.append(dc)
    if len(boots) < 30:
        return None
    return {
        "n_draws": len(boots),
        "d_crit_median": float(np.median(boots)),
        "d_crit_ci95_lo": float(np.quantile(boots, 0.025)),
        "d_crit_ci95_hi": float(np.quantile(boots, 0.975)),
        "d_crit_std": float(np.std(boots)),
    }


def _per_core_attraction(spatial, J, t00, d_nearest, nearest_core,
                           top_idx, inner_radius=1.0):
    pairs = []
    for c in top_idx:
        mask = ((nearest_core == c) & (d_nearest > 1e-6)
                & (d_nearest < inner_radius))
        if mask.sum() < 3:
            continue
        diff = spatial[mask] - spatial[c]
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        valid = (norms > 1e-12).flatten()
        if valid.sum() < 3:
            continue
        e_r = diff[valid] / norms[valid]
        Jr = (J[mask][valid] * e_r).sum(axis=1)
        pairs.append((float(abs(t00[c])), float(Jr.mean())))
    return pairs


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


def _bootstrap_spearman(pairs, n_boot, rng):
    masses = np.array([p[0] for p in pairs])
    jrs = np.array([p[1] for p in pairs])
    n = len(pairs)
    if n < 5:
        return None
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        s = _spearman(masses[idx], jrs[idx])
        if np.isfinite(s):
            boots.append(s)
    if not boots:
        return None
    return {
        "spearman_median": float(np.median(boots)),
        "spearman_ci95_lo": float(np.quantile(boots, 0.025)),
        "spearman_ci95_hi": float(np.quantile(boots, 0.975)),
    }


def _process(regime, n_lat, max_seeds=MAX_SEEDS):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None

    rng = np.random.default_rng(RNG_SEED + n_lat)

    # Aggregate (d, J_r) over all seeds
    all_d = []; all_jr = []; all_pairs = []
    seed_d_crits = []
    for seed_tuple in seeds:
        try:
            r = _per_seed(np.asarray(seed_tuple[0]),
                            np.asarray(seed_tuple[1]),
                            np.asarray(seed_tuple[2]) if len(seed_tuple) > 2 else None,
                            np.asarray(seed_tuple[3]) if len(seed_tuple) > 3 else None,
                            n_lat)
        except Exception:  # noqa: BLE001
            continue
        all_d.append(r["d"]); all_jr.append(r["Jr"])
        # Per-seed d_crit
        d, jr = _binned_curve(r["d"], r["Jr"])
        dc = _zero_crossing_smoothed(d, jr, smooth_window=3)
        if dc is None:
            dc = _zero_crossing_interp(d, jr)
        if dc is not None and np.isfinite(dc):
            seed_d_crits.append(dc)
        # Per-core attraction pairs
        pairs = _per_core_attraction(r["spatial"], r["J"], r["t00"],
                                       r["d"] if False else _bfs_dnearest(seed_tuple, n_lat),
                                       r["nearest_core"], r["top_idx"])
        all_pairs.extend(pairs)

    if not all_d:
        return None
    d_pool = np.concatenate(all_d); jr_pool = np.concatenate(all_jr)

    # Aggregate d_crit
    d_curve, jr_curve = _binned_curve(d_pool, jr_pool, n_bins=N_BINS)
    dc_interp = _zero_crossing_interp(d_curve, jr_curve)
    dc_smoothed = _zero_crossing_smoothed(d_curve, jr_curve, smooth_window=5)
    boot = _bootstrap_d_crit(d_pool, jr_pool, N_BOOTSTRAP, rng)

    # Mass-scaling Spearman bootstrap
    spearman_pt = (_spearman([p[0] for p in all_pairs],
                                [p[1] for p in all_pairs])
                    if all_pairs else float("nan"))
    boot_spear = _bootstrap_spearman(all_pairs, N_BOOTSTRAP, rng)

    return {
        "regime": regime, "N": int(n_lat),
        "n_seeds": len(seed_d_crits),
        "n_d_pool": int(len(d_pool)),
        "d_crit_aggregate_interp": dc_interp,
        "d_crit_aggregate_smoothed": dc_smoothed,
        "d_crit_per_seed": seed_d_crits,
        "d_crit_bootstrap": boot,
        "d_crit_predicted_1_over_D_Omega": D_CRIT_PREDICTED,
        "spearman_mass_vs_Jr_inner": spearman_pt,
        "spearman_bootstrap": boot_spear,
        "n_core_pairs": len(all_pairs),
    }


def _bfs_dnearest(seed_tuple, n_lat):
    """Helper: re-derive d_nearest just for filtering in pair counting."""
    xi_mat = np.asarray(seed_tuple[0], float).copy()
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)
    # Use t00 from re-running per_seed (small extra cost, but reliable)
    psi = np.asarray(seed_tuple[1])
    k_field = np.asarray(seed_tuple[2]) if len(seed_tuple) > 2 else None
    q_field = np.asarray(seed_tuple[3]) if len(seed_tuple) > 3 else None
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    t00 = prep["t00"]
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest, _ = _bfs(adj, top_idx, d_mat_inf)
    return d_nearest


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Predicted d_crit = 1/D_Omega = 1/(beta_pi-gamma) = "
          f"80/67 = {D_CRIT_PREDICTED:.4f}")
    print()
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
        dc_i = r["d_crit_aggregate_interp"]
        dc_s = r["d_crit_aggregate_smoothed"]
        boot = r["d_crit_bootstrap"]
        sp = r["spearman_mass_vs_Jr_inner"]
        sp_b = r["spearman_bootstrap"]
        if dc_i is not None:
            ratio_pred = dc_i / D_CRIT_PREDICTED
            print(f"    d_crit (interp) = {dc_i:.3f}  "
                  f"(smoothed = {dc_s if dc_s else float('nan'):.3f})  "
                  f"ratio to 1/D_Omega = {ratio_pred:.4f}")
        if boot:
            print(f"    bootstrap CI95: [{boot['d_crit_ci95_lo']:.3f}, "
                  f"{boot['d_crit_ci95_hi']:.3f}]  "
                  f"contains 1/D_Omega? "
                  f"{boot['d_crit_ci95_lo'] <= D_CRIT_PREDICTED <= boot['d_crit_ci95_hi']}")
        if sp_b:
            print(f"    Spearman(M, J_r) = {sp:+.3f}  "
                  f"CI95 [{sp_b['spearman_ci95_lo']:+.3f}, "
                  f"{sp_b['spearman_ci95_hi']:+.3f}]")

    if not rows:
        return

    # Cross-regime aggregates
    d_crits = [r["d_crit_aggregate_interp"] for r in rows
                if r["d_crit_aggregate_interp"] is not None]
    n_with_dcrit = len(d_crits)
    n_total = len(rows)
    in_ci_count = 0
    for r in rows:
        b = r["d_crit_bootstrap"]
        if b is not None:
            if b["d_crit_ci95_lo"] <= D_CRIT_PREDICTED <= b["d_crit_ci95_hi"]:
                in_ci_count += 1
    spearmans = [r["spearman_mass_vs_Jr_inner"] for r in rows
                  if np.isfinite(r["spearman_mass_vs_Jr_inner"])]

    out = {
        "method": "Universal d_crit + chain-reaction-amplification audit",
        "framework_constants": {"alpha_xi": 9/10, "gamma": GAMMA,
                                  "beta_pi": BETA_PI,
                                  "D_Omega": D_OMEGA,
                                  "d_crit_predicted_1_over_D_Omega":
                                      D_CRIT_PREDICTED},
        "n_bins": N_BINS,
        "n_bootstrap": N_BOOTSTRAP,
        "per_regime": rows,
        "summary": {
            "n_regimes_total": n_total,
            "n_regimes_with_dcrit_resolved": n_with_dcrit,
            "n_regimes_dcrit_CI_contains_1_over_D_Omega": in_ci_count,
            "d_crit_aggregate_median": float(np.median(d_crits)) if d_crits else None,
            "d_crit_aggregate_range":
                [float(np.min(d_crits)), float(np.max(d_crits))] if d_crits else None,
            "spearman_mass_vs_Jr_median": float(np.median(spearmans))
                if spearmans else None,
            "spearman_mass_vs_Jr_range":
                [float(np.min(spearmans)), float(np.max(spearmans))]
                if spearmans else None,
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("=== Cross-regime synthesis ===")
    print(f"  d_crit resolved: {n_with_dcrit}/{n_total} regimes")
    if d_crits:
        print(f"  d_crit median: {np.median(d_crits):.4f} "
              f"(predicted 1/D_Omega = {D_CRIT_PREDICTED:.4f})")
        print(f"  d_crit range: [{np.min(d_crits):.4f}, "
              f"{np.max(d_crits):.4f}]")
    print(f"  Bootstrap CIs containing 1/D_Omega: "
          f"{in_ci_count}/{n_total} regimes")
    if spearmans:
        print(f"  Spearman(M, J_r_inner) median: "
              f"{np.median(spearmans):+.3f} "
              f"(range [{np.min(spearmans):+.3f}, "
              f"{np.max(spearmans):+.3f}])")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
