"""SPARC-standard signed-residual halo audit.

Full eight-test follow-up to the bulk-core sign-balance:

  (1) Halo profile family fit on cumulative mass M(<r), AICc/BIC
      ranked across {NFW, cored-NFW, Burkert, Einasto,
      pseudo-isothermal, uniform, exponential, power-law}.
  (2) Mass-weighted distance to all defects
      d_eff(a) = [sum_b T00(b) d(a,b)^-2]^(-1/2)
      replacing the nearest-defect distance.
  (3) Cumulative mass profile M(<r) per seed.
  (4) Concentration parameter c = r_vir / r_s per regime.
  (5) Per-seed Spearman bootstrap of rho.
  (6) Partial-correlation control: residualise the
      bulk-amplitude on T00, graph-degree and omega before
      computing rho with d_eff.
  (7) SIDM cross-check: small-r slope alpha_inner of the
      cumulative mass profile (cuspy NFW alpha_inner = 2,
      cored Burkert alpha_inner = 3).
  (8) Velocity profile V(r) = sqrt(M(<r) / r); look for the
      Keplerian -> flat transition characteristic of
      galactic-DM rotation curves.

The energy-density projection Pi_rho R = u^mu u^nu R_munu is
the headline observable; the trace |tr R| is reported as a
robustness check.

Output: outputs/verify_signed_halo_full_audit.json
"""
from __future__ import annotations

import json
import math
import random
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

OUT = REPO / "outputs" / "verify_signed_halo_full_audit.json"
CORE_TAU = 0.05
CORE_TOP_FRAC = 0.05  # top 5% of T00 nodes are 'defect cores' for d_eff


# ============================================================
# Halo profile models
# ============================================================
def _nfw_cum(r, rho_0, r_s):
    """Cumulative mass for NFW: M(<r) = 4 pi rho_0 r_s^3 [ln(1+x) - x/(1+x)], x=r/r_s."""
    x = np.maximum(r / r_s, 1e-9)
    return 4 * math.pi * rho_0 * r_s ** 3 * (np.log1p(x) - x / (1.0 + x))


def _burkert_cum(r, rho_0, r_c):
    """Cumulative mass for Burkert: M(<r) = pi rho_0 r_c^3 [ln(1+x^2) + 2 ln(1+x) - 2 atan(x)]."""
    x = np.maximum(r / r_c, 1e-9)
    return (math.pi * rho_0 * r_c ** 3
            * (np.log1p(x ** 2) + 2 * np.log1p(x) - 2 * np.arctan(x)))


def _isothermal_cum(r, rho_0, r_c):
    """Pseudo-isothermal: M(<r) = 4 pi rho_0 r_c^3 [x - atan(x)], x=r/r_c."""
    x = np.maximum(r / r_c, 1e-9)
    return 4 * math.pi * rho_0 * r_c ** 3 * (x - np.arctan(x))


def _einasto_cum(r, rho_0, r_s, alpha=0.18):
    """Einasto cumulative mass (alpha fixed at 0.18, the cosmologically preferred value)."""
    x = np.maximum(r / r_s, 1e-9)
    # Einasto: rho(r) = rho_0 exp(-2/alpha (x^alpha - 1)); cumulative is via
    # incomplete gamma. We use a numerical-integration-free closed form
    # via mpmath would be ideal but sticking to numpy:
    # M(<r) ~ 4 pi rho_0 r_s^3 (alpha/2)^(3/alpha-1) exp(2/alpha) gamma(3/alpha, 2 x^alpha / alpha)
    # we use a finite numerical sum.
    n_pts = 400
    r_arr = np.linspace(1e-6, r_s * 12.0, n_pts)
    rho = rho_0 * np.exp(-2.0 / alpha * (r_arr / r_s) ** alpha + 2.0 / alpha)
    M_arr = np.cumsum(4 * math.pi * r_arr ** 2 * rho * np.gradient(r_arr))
    M_at_r = np.interp(r, r_arr, M_arr)
    return M_at_r


def _cored_nfw_cum(r, rho_0, r_s, r_c):
    """Cored-NFW (regularised) approximate cumulative mass."""
    x = np.maximum((r + r_c) / r_s, 1e-9)
    return 4 * math.pi * rho_0 * r_s ** 3 * (np.log1p(x) - x / (1.0 + x))


def _uniform_cum(r, rho_0):
    return (4.0 * math.pi / 3.0) * rho_0 * r ** 3


def _exponential_cum(r, rho_0, r_e):
    """Exponential disc analog: M(<r) = 4 pi rho_0 [2 r_e^3 - exp(-r/r_e)(2 r_e^3 + 2 r_e^2 r + r_e r^2)]."""
    x = r / r_e
    return (4.0 * math.pi * rho_0 * r_e ** 3
            * (2.0 - np.exp(-x) * (2.0 + 2.0 * x + x * x)))


def _powerlaw_cum(r, rho_0, alpha_pl):
    """Power-law density rho(r) ~ r^-alpha gives M(<r) ~ r^(3-alpha) for alpha < 3."""
    return rho_0 * np.maximum(r, 1e-9) ** (3.0 - alpha_pl)


# ============================================================
# Fit + AICc/BIC
# ============================================================
def _fit_least_squares(model_fn, xs, ys, p0, bounds=None, maxiter=120):
    """Minimal least-squares fit (no scipy needed)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    p = np.array(p0, dtype=float)
    if bounds is None:
        bounds = [(1e-12, 1e+12)] * len(p)
    bounds = [(b[0], b[1]) for b in bounds]
    lr = 1e-3
    for _ in range(maxiter):
        try:
            yhat = model_fn(xs, *p)
            r = ys - yhat
            ss = float(np.sum(r * r))
            grad = np.zeros_like(p)
            eps = 1e-4
            for i in range(len(p)):
                p_pert = p.copy()
                p_pert[i] += eps * max(abs(p[i]), 1e-9)
                yhat_p = model_fn(xs, *p_pert)
                r_p = ys - yhat_p
                ss_p = float(np.sum(r_p * r_p))
                grad[i] = (ss_p - ss) / (eps * max(abs(p[i]), 1e-9))
            p_new = p - lr * grad
            for i in range(len(p)):
                lo, hi = bounds[i]
                p_new[i] = max(lo, min(hi, p_new[i]))
            yhat_new = model_fn(xs, *p_new)
            r_new = ys - yhat_new
            ss_new = float(np.sum(r_new * r_new))
            if ss_new < ss:
                p = p_new
                lr = min(lr * 1.5, 1.0)
            else:
                lr *= 0.5
                if lr < 1e-9:
                    break
        except Exception:  # noqa: BLE001
            lr *= 0.5
            if lr < 1e-9:
                break
    yhat = model_fn(xs, *p)
    rss = float(np.sum((ys - yhat) ** 2))
    return p, rss


def _aicc_bic(rss, n, k):
    if n <= k + 1 or rss <= 0:
        return float("inf"), float("inf")
    sigma2 = rss / n
    aic = n * math.log(sigma2) + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1)
    bic = n * math.log(sigma2) + k * math.log(n)
    return aicc, bic


# ============================================================
# Spearman & partial correlation
# ============================================================
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


def _partial_correlation(y, x, controls):
    """Spearman-style partial correlation: residualise both y and x
    against controls (linear regression on ranks), then compute
    correlation of residuals."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    Z = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    mask = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    if mask.sum() < 10:
        return float("nan")
    y_m = y[mask]
    x_m = x[mask]
    Z_m = Z[mask]
    # linear regression to residualise
    Z_aug = np.column_stack([np.ones(len(Z_m)), Z_m])
    coef_y, *_ = np.linalg.lstsq(Z_aug, y_m, rcond=None)
    coef_x, *_ = np.linalg.lstsq(Z_aug, x_m, rcond=None)
    y_res = y_m - Z_aug @ coef_y
    x_res = x_m - Z_aug @ coef_x
    return _spearman(y_res, x_res)


# ============================================================
# Distance fields on the lattice graph
# ============================================================
def _bfs_distances_to_set(adj, target_idx_set, d_mat):
    """Multi-source Dijkstra from target_idx_set; returns
    distance to nearest target node per source node."""
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    for t in target_idx_set:
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


def _mass_weighted_distance_field(t00, adj, d_mat, top_idx, dist_to_set):
    """d_eff(a) = [sum_b T00(b) / d(a,b)^2]^(-1/2)
    over b in top_idx. We use the per-source distance from the
    multi-source field as a proxy: if the nearest defect is at
    distance d, we approximate the sum by T00_max / d^2 (single-
    source approximation) and refine with two more nearest defects
    if available."""
    n = adj.shape[0]
    # For computational simplicity: use the multi-source nearest
    # distance from each node to the defect set (top_idx).
    # The mass-weighted version is a smoother proxy.
    # We compute per-defect distance using individual BFS, then
    # combine.
    inv_sq = np.zeros(n)
    if len(top_idx) == 0:
        return np.full(n, np.inf)
    # Use up to 8 strongest defects to keep cost bounded
    sorted_top = sorted(top_idx, key=lambda i: -t00[i])[:8]
    for t in sorted_top:
        d_t = _bfs_distances_to_set(adj, [int(t)], d_mat)
        d_safe = np.where(np.isfinite(d_t), np.maximum(d_t, 1e-3), np.inf)
        inv_sq += t00[t] / d_safe ** 2
    inv_sq = np.where(inv_sq > 0, inv_sq, 1e-12)
    return 1.0 / np.sqrt(inv_sq)


# ============================================================
# Per-seed audit
# ============================================================
def _per_seed_audit(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = np.eye(3)
    # R^{mu nu} components
    r_00 = g_00 + LAMBDA_T - t00
    r_ii_diag = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij
    # Energy-density projection: in the spectral frame u^mu = (1,0,0,0),
    # so u^mu u^nu R_{mu nu} = R_00 = g_{00} R^{00} approximately = R^00
    pi_rho = r_00
    pi_tr = r_00 + r_ii_diag[:, 0, 0] + r_ii_diag[:, 1, 1] + r_ii_diag[:, 2, 2]
    df = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]
    # Adjacency + distance matrix
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = adj.sum(axis=1)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.where(adj > 0, d_mat, np.inf)
    # Defect cores: top CORE_TOP_FRAC of |T00|
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    # Distance fields
    d_nearest = _bfs_distances_to_set(adj, top_idx, d_mat)
    d_eff = _mass_weighted_distance_field(t00, adj, d_mat, top_idx, d_nearest)
    # Omega field (graph weight density)
    weight_grad_safe = np.where(adj > 0,
                                  weight_adj / np.maximum(d_mat, 1e-9) ** 2,
                                  0.0)
    omega_a = weight_grad_safe.sum(axis=1)
    return {
        "pi_rho": pi_rho,
        "pi_tr": pi_tr,
        "df": df,
        "t00": t00,
        "deg": deg,
        "omega": omega_a,
        "d_nearest": d_nearest,
        "d_eff": d_eff,
        "n_lat": n_lat,
    }


# ============================================================
# Cumulative mass profile + family fit
# ============================================================
def _fit_halo_family(r_vals, M_cum):
    """Fit eight halo models to cumulative mass M(<r). Return per-model
    (params, RSS, AICc, BIC)."""
    r_vals = np.asarray(r_vals, dtype=float)
    M_cum = np.asarray(M_cum, dtype=float)
    n = len(r_vals)
    # robust normalisation: rescale so M_cum(max) = 1, r_vals(max) = 1
    M_max = max(np.abs(M_cum).max(), 1e-12)
    r_max = max(r_vals.max(), 1e-12)
    ys = M_cum / M_max
    xs = r_vals / r_max
    out = {}
    # NFW: 2 params
    p, rss = _fit_least_squares(_nfw_cum, xs, ys, p0=[1.0, 0.5],
                                  bounds=[(1e-6, 1e6), (1e-3, 5.0)])
    a, b_ic = _aicc_bic(rss, n, 2)
    out["NFW"] = {"k": 2, "params": p.tolist(), "rss": rss,
                   "aicc": a, "bic": b_ic, "r_s_norm": float(p[1])}
    # Burkert: 2 params
    p, rss = _fit_least_squares(_burkert_cum, xs, ys, p0=[1.0, 0.3],
                                  bounds=[(1e-6, 1e6), (1e-3, 5.0)])
    a, b_ic = _aicc_bic(rss, n, 2)
    out["Burkert"] = {"k": 2, "params": p.tolist(), "rss": rss,
                       "aicc": a, "bic": b_ic, "r_c_norm": float(p[1])}
    # Pseudo-isothermal: 2 params
    p, rss = _fit_least_squares(_isothermal_cum, xs, ys, p0=[1.0, 0.3],
                                  bounds=[(1e-6, 1e6), (1e-3, 5.0)])
    a, b_ic = _aicc_bic(rss, n, 2)
    out["pseudo-isothermal"] = {"k": 2, "params": p.tolist(),
                                  "rss": rss, "aicc": a, "bic": b_ic,
                                  "r_c_norm": float(p[1])}
    # Einasto: 2 params (rho_0, r_s) with alpha fixed
    p, rss = _fit_least_squares(_einasto_cum, xs, ys, p0=[1.0, 0.5],
                                  bounds=[(1e-6, 1e6), (1e-3, 5.0)])
    a, b_ic = _aicc_bic(rss, n, 2)
    out["Einasto"] = {"k": 2, "params": p.tolist(), "rss": rss,
                       "aicc": a, "bic": b_ic, "r_s_norm": float(p[1])}
    # Cored-NFW: 3 params
    p, rss = _fit_least_squares(_cored_nfw_cum, xs, ys,
                                  p0=[1.0, 0.5, 0.1],
                                  bounds=[(1e-6, 1e6), (1e-3, 5.0),
                                          (1e-3, 1.0)])
    a, b_ic = _aicc_bic(rss, n, 3)
    out["cored-NFW"] = {"k": 3, "params": p.tolist(), "rss": rss,
                         "aicc": a, "bic": b_ic,
                         "r_s_norm": float(p[1]), "r_c_norm": float(p[2])}
    # Uniform: 1 param
    p, rss = _fit_least_squares(_uniform_cum, xs, ys, p0=[1.0],
                                  bounds=[(1e-6, 1e6)])
    a, b_ic = _aicc_bic(rss, n, 1)
    out["uniform"] = {"k": 1, "params": p.tolist(), "rss": rss,
                       "aicc": a, "bic": b_ic}
    # Exponential: 2 params
    p, rss = _fit_least_squares(_exponential_cum, xs, ys, p0=[1.0, 0.3],
                                  bounds=[(1e-6, 1e6), (1e-3, 5.0)])
    a, b_ic = _aicc_bic(rss, n, 2)
    out["exponential"] = {"k": 2, "params": p.tolist(), "rss": rss,
                           "aicc": a, "bic": b_ic}
    # Power-law: 2 params (rho_0, alpha)
    p, rss = _fit_least_squares(_powerlaw_cum, xs, ys, p0=[1.0, 1.5],
                                  bounds=[(1e-6, 1e6), (0.1, 2.9)])
    a, b_ic = _aicc_bic(rss, n, 2)
    out["power-law"] = {"k": 2, "params": p.tolist(), "rss": rss,
                         "aicc": a, "bic": b_ic,
                         "alpha": float(p[1])}
    # Best by AICc
    best = min(out.items(), key=lambda kv: kv[1]["aicc"])
    return out, best[0], best[1]["aicc"]


# ============================================================
# Per-regime audit
# ============================================================
def _process_regime(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))
    seeds = seeds[:max_seeds]
    if not seeds:
        return None
    # Pool per-seed audits
    pooled_pi_rho = []
    pooled_pi_tr = []
    pooled_df = []
    pooled_t00 = []
    pooled_deg = []
    pooled_omega = []
    pooled_d_nearest = []
    pooled_d_eff = []
    seed_rho_pi_rho_d_eff = []
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            r = _per_seed_audit(np.asarray(xi_mat), np.asarray(psi),
                                  np.asarray(k_field), np.asarray(q_field),
                                  n_lat)
        except Exception as exc:  # noqa: BLE001
            continue
        # bulk-fringe mask: nodes NOT in matter-core (df > tau)
        mask_bulk = (r["df"] <= CORE_TAU) & np.isfinite(r["d_eff"])
        if mask_bulk.sum() < 30:
            continue
        # per-seed Spearman of |Pi_rho R| vs d_eff in bulk
        rho_seed = _spearman(np.abs(r["pi_rho"][mask_bulk]),
                              r["d_eff"][mask_bulk])
        seed_rho_pi_rho_d_eff.append(rho_seed)
        pooled_pi_rho.append(r["pi_rho"])
        pooled_pi_tr.append(r["pi_tr"])
        pooled_df.append(r["df"])
        pooled_t00.append(r["t00"])
        pooled_deg.append(r["deg"])
        pooled_omega.append(r["omega"])
        pooled_d_nearest.append(r["d_nearest"])
        pooled_d_eff.append(r["d_eff"])
    if not seed_rho_pi_rho_d_eff:
        return None
    pi_rho = np.concatenate(pooled_pi_rho)
    pi_tr = np.concatenate(pooled_pi_tr)
    df = np.concatenate(pooled_df)
    t00 = np.concatenate(pooled_t00)
    deg = np.concatenate(pooled_deg)
    omega = np.concatenate(pooled_omega)
    d_nearest = np.concatenate(pooled_d_nearest)
    d_eff = np.concatenate(pooled_d_eff)
    mask_bulk = (df <= CORE_TAU) & np.isfinite(d_eff)
    # Pooled tests
    rho_pi_rho_nearest = _spearman(np.abs(pi_rho[mask_bulk]),
                                     d_nearest[mask_bulk])
    rho_pi_rho_eff = _spearman(np.abs(pi_rho[mask_bulk]),
                                 d_eff[mask_bulk])
    rho_pi_tr_eff = _spearman(np.abs(pi_tr[mask_bulk]),
                                d_eff[mask_bulk])
    # Partial correlation: residualise on T00, deg, omega
    rho_partial = _partial_correlation(
        np.abs(pi_rho[mask_bulk]), d_eff[mask_bulk],
        [t00[mask_bulk], deg[mask_bulk], omega[mask_bulk]])
    # Per-seed bootstrap
    n_seed = len(seed_rho_pi_rho_d_eff)
    rng = random.Random(0xCAFE)
    boot_means = []
    for _ in range(2000):
        sample = [seed_rho_pi_rho_d_eff[rng.randrange(n_seed)]
                  for _ in range(n_seed)]
        boot_means.append(sum(sample) / n_seed)
    boot_means.sort()
    seed_rho_mean = sum(seed_rho_pi_rho_d_eff) / n_seed
    seed_rho_lo = boot_means[int(0.025 * len(boot_means))]
    seed_rho_hi = boot_means[int(0.975 * len(boot_means))]
    p_rho_lt_zero = sum(1 for r in seed_rho_pi_rho_d_eff if r < 0) / n_seed

    # Cumulative mass profile + family fit
    bulk_amp = np.abs(pi_rho[mask_bulk])
    bulk_d = d_eff[mask_bulk]
    sort_idx = np.argsort(bulk_d)
    r_sorted = bulk_d[sort_idx]
    amp_sorted = bulk_amp[sort_idx]
    M_cum = np.cumsum(amp_sorted)
    # SIDM/inner-slope check: log-slope of M(<r) at small r
    inner_mask = r_sorted < np.percentile(r_sorted, 10)
    if inner_mask.sum() >= 5:
        log_r = np.log(np.maximum(r_sorted[inner_mask], 1e-9))
        log_M = np.log(np.maximum(M_cum[inner_mask], 1e-12))
        coef = np.polyfit(log_r, log_M, 1)
        alpha_inner = float(coef[0])
    else:
        alpha_inner = float("nan")
    # Family fit
    fits, best_model, best_aicc = _fit_halo_family(r_sorted, M_cum)
    # Concentration parameter c = r_max / r_s (best-fit model)
    if best_model == "NFW":
        rs_norm = fits["NFW"]["r_s_norm"]
        c = 1.0 / rs_norm if rs_norm > 0 else float("nan")
    elif best_model == "cored-NFW":
        rs_norm = fits["cored-NFW"]["r_s_norm"]
        c = 1.0 / rs_norm if rs_norm > 0 else float("nan")
    else:
        # use NFW-fit r_s as concentration proxy
        c = (1.0 / fits["NFW"]["r_s_norm"]
             if fits["NFW"]["r_s_norm"] > 0 else float("nan"))

    # Velocity profile V(r) = sqrt(M(<r) / r); look for plateau
    v_profile = np.sqrt(M_cum / np.maximum(r_sorted, 1e-9))
    inner_v = np.median(v_profile[inner_mask]) if inner_mask.sum() >= 5 else float("nan")
    outer_mask = r_sorted > np.percentile(r_sorted, 60)
    outer_v = (np.median(v_profile[outer_mask])
               if outer_mask.sum() >= 5 else float("nan"))
    v_ratio = (outer_v / inner_v
               if inner_v > 0 and inner_v == inner_v else float("nan"))
    # Keplerian (V ~ r^-0.5) gives v_ratio < 1 if no halo;
    # flat rotation curve (DM halo) gives v_ratio ~ 1 (or > 1)

    return {
        "regime": regime,
        "N": int(n_lat),
        "n_seeds": int(n_seed),
        "n_bulk_nodes_pooled": int(mask_bulk.sum()),
        # Test 5: per-seed bootstrap
        "seed_rho_mean": seed_rho_mean,
        "seed_rho_95CI": [seed_rho_lo, seed_rho_hi],
        "seed_rho_P_lt_zero": p_rho_lt_zero,
        # Test 2: pooled rho with nearest vs effective distance
        "rho_pi_rho_nearest_pooled": rho_pi_rho_nearest,
        "rho_pi_rho_eff_pooled": rho_pi_rho_eff,
        "rho_pi_tr_eff_pooled": rho_pi_tr_eff,
        # Test 6: partial correlation
        "rho_partial_pi_rho_eff_given_t00_deg_omega": rho_partial,
        # Tests 1+3: family fit
        "halo_family_fit_aicc_ranked": [
            (k, v["aicc"], v["bic"]) for k, v in
            sorted(fits.items(), key=lambda kv: kv[1]["aicc"])],
        "halo_family_best_model": best_model,
        "halo_family_best_aicc": best_aicc,
        "halo_family_full_fits": fits,
        # Test 4: concentration
        "concentration_c_proxy": c,
        # Test 7: SIDM cross-check (inner-slope)
        "alpha_inner_log_slope": alpha_inner,
        # Test 8: velocity profile
        "v_inner_median": inner_v,
        "v_outer_median": outer_v,
        "v_outer_over_inner": v_ratio,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d}  ...")
        r = _process_regime(regime, n_lat, max_seeds=24)
        if r is None:
            print(f"    SKIP")
            continue
        rows.append(r)
        print(f"    seed_rho={r['seed_rho_mean']:+.3f} "
              f"95CI=[{r['seed_rho_95CI'][0]:+.3f},{r['seed_rho_95CI'][1]:+.3f}]  "
              f"partial_rho={r['rho_partial_pi_rho_eff_given_t00_deg_omega']:+.3f}  "
              f"best={r['halo_family_best_model']}  "
              f"c={r['concentration_c_proxy']:.2f}  "
              f"alpha_inner={r['alpha_inner_log_slope']:+.2f}  "
              f"V_out/V_in={r['v_outer_over_inner']:.2f}")
    out = {
        "method": "SPARC-standard signed-residual halo audit (8-test follow-up)",
        "core_tau": CORE_TAU,
        "core_top_frac": CORE_TOP_FRAC,
        "models_compared": [
            "NFW", "Burkert", "pseudo-isothermal", "Einasto",
            "cored-NFW", "uniform", "exponential", "power-law"],
        "per_regime": rows,
    }
    if rows:
        # Aggregate the headline verdicts
        n_total = len(rows)
        n_seed_lt0 = sum(1 for r in rows if r["seed_rho_mean"] < 0)
        n_partial_lt0 = sum(
            1 for r in rows
            if (r["rho_partial_pi_rho_eff_given_t00_deg_omega"] ==
                r["rho_partial_pi_rho_eff_given_t00_deg_omega"] and
                r["rho_partial_pi_rho_eff_given_t00_deg_omega"] < 0))
        cored_wins = sum(1 for r in rows
                          if r["halo_family_best_model"] in
                          ("Burkert", "cored-NFW",
                           "Einasto", "pseudo-isothermal"))
        nfw_wins = sum(1 for r in rows
                        if r["halo_family_best_model"] == "NFW")
        v_ratio_meds = [r["v_outer_over_inner"] for r in rows
                         if r["v_outer_over_inner"] ==
                         r["v_outer_over_inner"]]
        out["summary"] = {
            "n_regimes": n_total,
            "test5_seed_rho_negative_n": n_seed_lt0,
            "test6_partial_rho_negative_after_controls_n": n_partial_lt0,
            "test1_best_family_cored_n": cored_wins,
            "test1_best_family_nfw_n": nfw_wins,
            "test8_velocity_outer_over_inner_median": (
                float(np.median(v_ratio_meds)) if v_ratio_meds else None),
        }
        print()
        print(f"  Summary across {n_total} regimes:")
        print(f"    Test 5 (per-seed rho < 0):                    "
              f"{n_seed_lt0}/{n_total}")
        print(f"    Test 6 (partial-correlation < 0 after T00/deg/omega "
              f"controls): {n_partial_lt0}/{n_total}")
        print(f"    Test 1 (cored profile wins by AICc):          "
              f"{cored_wins}/{n_total}")
        print(f"    Test 1 (NFW wins by AICc):                    "
              f"{nfw_wins}/{n_total}")
        if v_ratio_meds:
            print(f"    Test 8 (V_outer/V_inner median):              "
                  f"{out['summary']['test8_velocity_outer_over_inner_median']:.2f}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
