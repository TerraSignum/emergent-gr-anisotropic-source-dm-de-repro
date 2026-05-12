"""Deep follow-up audit on the pre-gravity sign-flip hypothesis.

Tests A through E:

  A. d_crit per regime: precise distance at which the radial heat
     current J_radial transitions from inward (attractive) to
     outward (repulsive). Computed via interpolation on a
     50-bin <J_radial>(d) curve.

  B. d_crit vs cluster mass: weight each core by T_00; does the
     attraction-domain radius scale with local mass concentration?
     Gravitational signature: d_crit should grow with sqrt(M_local)
     or some monotone function of cluster mass.

  C. Multi-scale hierarchy: do clusters cluster? Compute the
     pairwise distance distribution among cores in spectral-frame
     coordinates. Look for multi-modal peaks (= nested hierarchy).

  D. Defect-defect direct interaction: for each pair of cores
     (c_i, c_j), compute the heat-current component along the
     line connecting them at the midpoint between them.
     Net inward (toward midpoint) = repulsive, outward = attractive.

  E. Chain-reaction amplification: does the attractive
     J_radial in shell 0 scale with cluster mass M_local?
     Gravitational signature: attraction proportional to mass.
     Saturated chain-reaction: would saturate at high mass.

Output: outputs/verify_pre_gravity_deep_audit.json
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

OUT = REPO / "outputs" / "verify_pre_gravity_deep_audit.json"
CORE_TOP_FRAC = 0.05
N_BINS_DCRIT = 50
MAX_SEEDS = 4


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


def _per_seed_observables(xi_mat, psi, k_field, q_field, n_lat):
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

    return {
        "spatial": spatial, "J": J, "t00": t00,
        "d_nearest": d_nearest, "nearest_core": nearest_core,
        "top_idx": top_idx, "n_lat": n_lat,
    }


def _radial_jr_curve(spatial, J, d_nearest, nearest_core, n_bins):
    """Return (d_centres, J_r_mean) for fine-binned J_radial(d)."""
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
    d_v = d_nearest[valid]; jr_v = Jr[valid]
    if len(d_v) < n_bins:
        return None, None
    edges = np.quantile(d_v, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centres = []; jrm = []
    for k in range(n_bins):
        sel = (d_v >= edges[k]) & (d_v < edges[k + 1])
        if sel.sum() < 3:
            continue
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        jrm.append(float(jr_v[sel].mean()))
    return np.array(centres), np.array(jrm)


def _find_zero_crossing(d, jr):
    """Linear-interp d_crit where jr crosses zero (first inner-to-outer flip)."""
    if d is None or len(d) < 3:
        return None
    for k in range(len(jr) - 1):
        if jr[k] < 0 <= jr[k + 1]:
            # linear interpolation
            t = -jr[k] / (jr[k + 1] - jr[k] + 1e-30)
            return float(d[k] + t * (d[k + 1] - d[k]))
    return None


def _test_A_d_crit(seeds_obs):
    """A: per-regime d_crit (median across seeds)."""
    d_crits = []
    for r in seeds_obs:
        d, jr = _radial_jr_curve(r["spatial"], r["J"], r["d_nearest"],
                                    r["nearest_core"], N_BINS_DCRIT)
        dc = _find_zero_crossing(d, jr)
        if dc is not None:
            d_crits.append(dc)
    if not d_crits:
        return None
    return {
        "n_seeds": len(d_crits),
        "d_crit_median": float(np.median(d_crits)),
        "d_crit_mean": float(np.mean(d_crits)),
        "d_crit_std": float(np.std(d_crits)),
        "d_crit_per_seed": [float(x) for x in d_crits],
    }


def _test_BE_mass_scaling(seeds_obs):
    """B+E: per-core attraction strength vs core mass.

    For each core c_i, compute mean J_radial of nodes a with
    nearest_core(a) = c_i AND d_nearest(a) in inner-shell range
    (e.g. d <= 1.0 lattice units). Plot vs T_00(c_i)."""
    pairs = []
    for r in seeds_obs:
        spatial = r["spatial"]; J = r["J"]; t00 = r["t00"]
        d_nearest = r["d_nearest"]; nearest_core = r["nearest_core"]
        for c in r["top_idx"]:
            mask = ((nearest_core == c) & (d_nearest > 1e-6)
                    & (d_nearest < 1.0))
            if mask.sum() < 3:
                continue
            diff = spatial[mask] - spatial[c]
            norms = np.linalg.norm(diff, axis=1, keepdims=True)
            valid = (norms > 1e-12).flatten()
            if valid.sum() < 3:
                continue
            e_r = diff[valid] / norms[valid]
            Jr = (J[mask][valid] * e_r).sum(axis=1)
            mean_Jr = float(Jr.mean())
            pairs.append({
                "core_T00": float(abs(t00[c])),
                "core_id": int(c),
                "mean_J_radial_inner": mean_Jr,
                "n_inner_nodes": int(valid.sum()),
            })
    if len(pairs) < 5:
        return None
    masses = np.array([p["core_T00"] for p in pairs])
    jr_inner = np.array([p["mean_J_radial_inner"] for p in pairs])
    finite = np.isfinite(masses) & np.isfinite(jr_inner)
    masses = masses[finite]; jr_inner = jr_inner[finite]
    # Spearman rank correlation
    rx = np.argsort(np.argsort(masses)).astype(float)
    ry = np.argsort(np.argsort(jr_inner)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    spearman = float((rx * ry).sum() / den) if den > 0 else float("nan")
    # Linear-log fit: |J_r_inner| vs |T_00|
    abs_jr = np.abs(jr_inner)
    pos = abs_jr > 1e-12
    if pos.sum() >= 3:
        slope_loglog = float(np.polyfit(np.log(masses[pos]),
                                          np.log(abs_jr[pos]), 1)[0])
    else:
        slope_loglog = float("nan")
    return {
        "n_cores": int(finite.sum()),
        "spearman_mass_vs_J_radial_inner": spearman,
        "loglog_slope_abs_Jr_vs_M": slope_loglog,
        "interpretation": (
            "spearman < 0 (more negative for higher mass): consistent "
            "with mass-amplified attraction (chain-reaction signature). "
            "loglog slope ~ 1: linear gravitational scaling. "
            "loglog slope ~ 0: saturated."),
        "core_mass_J_pairs": pairs[:30],
    }


def _test_C_hierarchy(seeds_obs):
    """C: pairwise core-distance distribution; multi-modal => hierarchy."""
    if not seeds_obs:
        return None
    all_pair_dists = []
    for r in seeds_obs:
        sp = r["spatial"]
        cores = r["top_idx"]
        for i in range(len(cores)):
            for j in range(i + 1, len(cores)):
                d = float(np.linalg.norm(sp[cores[i]] - sp[cores[j]]))
                all_pair_dists.append(d)
    if len(all_pair_dists) < 5:
        return None
    arr = np.array(all_pair_dists)
    # Test for multi-modality via: ratio of std to range
    return {
        "n_pairs": int(len(arr)),
        "dist_min": float(arr.min()),
        "dist_max": float(arr.max()),
        "dist_median": float(np.median(arr)),
        "dist_mean": float(arr.mean()),
        "dist_std": float(arr.std()),
        "ratio_std_over_mean": float(arr.std() / arr.mean()),
        "histogram_edges": np.linspace(arr.min(), arr.max(), 11).tolist(),
        "histogram_counts": np.histogram(
            arr, bins=10)[0].tolist(),
    }


def _test_D_defect_defect(seeds_obs):
    """D: net heat-current FLUX through midpoints between core pairs."""
    out_pairs = []
    for r in seeds_obs:
        sp = r["spatial"]; J = r["J"]
        cores = r["top_idx"]
        for i in range(len(cores)):
            for j in range(i + 1, len(cores)):
                ci, cj = cores[i], cores[j]
                xi = sp[ci]; xj = sp[cj]
                line_vec = xj - xi
                line_dist = np.linalg.norm(line_vec)
                if line_dist < 1e-12:
                    continue
                line_dir = line_vec / line_dist
                midpoint = 0.5 * (xi + xj)
                # Find nearest node to midpoint
                d_to_mid = np.linalg.norm(sp - midpoint, axis=1)
                m_idx = int(np.argmin(d_to_mid))
                Jm = J[m_idx]
                # Skip if J at midpoint is non-finite
                if not np.all(np.isfinite(Jm)):
                    continue
                Jm_along = float(Jm @ line_dir)
                Jm_perp = float(np.linalg.norm(
                    Jm - (Jm @ line_dir) * line_dir))
                # Sign-symmetric reading: |Jm_along| is the
                # along-axis flow magnitude (independent of which
                # core is "first").
                jm_along_abs = abs(Jm_along)
                out_pairs.append({
                    "ci": int(ci), "cj": int(cj),
                    "line_dist": float(line_dist),
                    "Jm_along_abs": jm_along_abs,
                    "Jm_perp_line": Jm_perp,
                    "Jm_magnitude": float(np.linalg.norm(Jm)),
                })
    if not out_pairs:
        return None
    along_abs = np.array([p["Jm_along_abs"] for p in out_pairs])
    perp = np.array([p["Jm_perp_line"] for p in out_pairs])
    along_abs = along_abs[np.isfinite(along_abs)]
    perp = perp[np.isfinite(perp)]
    if len(along_abs) == 0:
        return None
    return {
        "n_pairs": len(out_pairs),
        "Jm_along_abs_mean": float(along_abs.mean()),
        "Jm_along_abs_median": float(np.median(along_abs)),
        "Jm_perp_mean": float(perp.mean()) if len(perp) > 0 else float("nan"),
        "Jm_perp_median": float(np.median(perp)) if len(perp) > 0 else float("nan"),
        "ratio_along_to_perp": (
            float(along_abs.mean() / perp.mean())
            if len(perp) > 0 and perp.mean() > 0
            else float("nan")),
        "interpretation": (
            "Jm_along/perp == flow at midpoint along/perpendicular to "
            "core-core axis. Sign-symmetric: along should be ~0 mean by "
            "exchange symmetry. Perp dominance: J flows orthogonal to "
            "core-pair line at midpoint, consistent with each core "
            "attracting its own halo independently rather than direct "
            "core-core gravitational pull."),
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
    obs = []
    for seed_tuple in seeds:
        try:
            r = _per_seed_observables(
                np.asarray(seed_tuple[0]),
                np.asarray(seed_tuple[1]),
                np.asarray(seed_tuple[2]) if len(seed_tuple) > 2 else None,
                np.asarray(seed_tuple[3]) if len(seed_tuple) > 3 else None,
                n_lat)
        except Exception:  # noqa: BLE001
            continue
        obs.append(r)
    if not obs:
        return None
    return {
        "regime": regime, "N": int(n_lat),
        "n_seeds": len(obs),
        "test_A_d_crit": _test_A_d_crit(obs),
        "test_BE_mass_scaling": _test_BE_mass_scaling(obs),
        "test_C_hierarchy": _test_C_hierarchy(obs),
        "test_D_defect_defect": _test_D_defect_defect(obs),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        r = _process(regime, n_lat)
        if r is None:
            print("    skipped (no data)")
            continue
        rows.append(r)
        a = r["test_A_d_crit"]
        b = r["test_BE_mass_scaling"]
        c = r["test_C_hierarchy"]
        d = r["test_D_defect_defect"]
        if a:
            print(f"    A) d_crit median = {a['d_crit_median']:.3f}  +- "
                  f"{a['d_crit_std']:.3f}")
        if b:
            print(f"    B+E) Spearman(M, <J_r>_inner) = "
                  f"{b['spearman_mass_vs_J_radial_inner']:+.3f}  "
                  f"loglog slope = {b['loglog_slope_abs_Jr_vs_M']:+.3f}")
        if c:
            print(f"    C) pair-dist spread = "
                  f"{c['ratio_std_over_mean']:.3f}  "
                  f"(min={c['dist_min']:.3f}, max={c['dist_max']:.3f})")
        if d:
            print(f"    D) <|Jm_along|> = {d['Jm_along_abs_mean']:.5f}  "
                  f"<Jm_perp> = {d['Jm_perp_mean']:.5f}  "
                  f"ratio along/perp = {d['ratio_along_to_perp']:.3f}")

    out = {
        "method": "Pre-gravity hypothesis: deep audit (A-E)",
        "core_top_frac": CORE_TOP_FRAC,
        "n_bins_dcrit": N_BINS_DCRIT,
        "per_regime": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    if rows:
        # Aggregates
        d_crits_all = [r["test_A_d_crit"]["d_crit_median"]
                        for r in rows if r["test_A_d_crit"]]
        spearmans = [r["test_BE_mass_scaling"]["spearman_mass_vs_J_radial_inner"]
                      for r in rows if r["test_BE_mass_scaling"]]
        slopes = [r["test_BE_mass_scaling"]["loglog_slope_abs_Jr_vs_M"]
                   for r in rows if r["test_BE_mass_scaling"]
                   and np.isfinite(r["test_BE_mass_scaling"]["loglog_slope_abs_Jr_vs_M"])]
        spreads = [r["test_C_hierarchy"]["ratio_std_over_mean"]
                    for r in rows if r["test_C_hierarchy"]]
        print("Aggregates:")
        print(f"  d_crit (median over regimes): {np.median(d_crits_all):.3f}")
        print(f"  d_crit range: [{np.min(d_crits_all):.3f}, "
              f"{np.max(d_crits_all):.3f}]")
        print(f"  Spearman(M, J_r_inner) median: "
              f"{np.median(spearmans):+.3f}")
        print(f"  loglog slope |J_r| vs M median: "
              f"{np.median(slopes):+.3f}")
        print(f"  pair-dist spread (std/mean): {np.median(spreads):.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
