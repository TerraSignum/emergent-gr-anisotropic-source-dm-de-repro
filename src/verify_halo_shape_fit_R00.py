"""Item 5: Halo-shape model comparison directly on |R_00|.

Replaces the original NFW fit (which was applied to the trace
projection) with the same family of one-parameter density
profiles applied to the energy-density component:

  NFW          rho(r) ~ A / (r/r_s) (1 + r/r_s)^2
  Burkert      rho(r) ~ A / ((1 + r/r_s)(1 + (r/r_s)^2))
  cored-NFW    rho(r) ~ A / ((1 + r/r_s)^2 (1 + (r/r_s)^3))
  Plummer      rho(r) ~ A / (1 + (r/r_s)^2)^(5/2)
  Exponential  rho(r) ~ A exp(-r/r_s)
  Power-law    rho(r) ~ A / (r/r_s + 1)^p   (p free)
  Uniform      rho(r) ~ A   (null model)

Per regime, the bulk-pooled |R_00(a)| values are binned by
geodesic distance d(a, partial C_N) into 10 equal-count bins.
Each profile is fit by non-linear least squares on the binned
means. Models are ranked by AICc and BIC and a model-averaged
inner-slope estimate is reported.

Output: outputs/verify_halo_shape_fit_R00.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

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

OUT = REPO / "outputs" / "verify_halo_shape_fit_R00.json"
N_BINS = 10
CORE_TOP_FRAC = 0.05


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


def nfw(r, A, rs):
    x = np.maximum(r / rs, 1e-9)
    return A / (x * (1 + x)**2)


def burkert(r, A, rs):
    x = r / rs
    return A / ((1 + x) * (1 + x**2))


def cored_nfw(r, A, rs):
    x = r / rs
    return A / ((1 + x)**2 * (1 + x**3))


def plummer(r, A, rs):
    x = r / rs
    return A / (1 + x**2)**2.5


def exponential(r, A, rs):
    return A * np.exp(-r / rs)


def power_law(r, A, rs, p):
    x = r / rs
    return A / (1 + x)**p


def uniform(r, A):
    return np.full_like(r, A)


PROFILES = {
    "NFW":         (nfw,         2),
    "Burkert":     (burkert,     2),
    "cored_NFW":   (cored_nfw,   2),
    "Plummer":     (plummer,     2),
    "exponential": (exponential, 2),
    "power_law":   (power_law,   3),
    "uniform":     (uniform,     1),
}


def _aicc(rss, n, k):
    if n - k - 1 <= 0:
        return float("nan")
    sig2 = rss / n
    if sig2 <= 0:
        return float("-inf")
    return float(n * math.log(sig2) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def _bic(rss, n, k):
    if rss <= 0:
        return float("-inf")
    return float(n * math.log(rss / n) + k * math.log(n))


def _fit_profile(name, fn, k, r, y):
    try:
        if name == "uniform":
            popt, _ = curve_fit(fn, r, y, p0=[max(y.mean(), 1e-9)],
                                  maxfev=5000)
        elif name == "power_law":
            popt, _ = curve_fit(fn, r, y,
                                  p0=[max(y[0], 1e-9),
                                      max(r.mean(), 1e-9), 2.0],
                                  maxfev=5000,
                                  bounds=([0, 1e-6, 0.1],
                                          [np.inf, 1e3, 10]))
        else:
            popt, _ = curve_fit(fn, r, y,
                                  p0=[max(y[0], 1e-9), max(r.mean(), 1e-9)],
                                  maxfev=5000,
                                  bounds=([0, 1e-6], [np.inf, 1e3]))
        pred = fn(r, *popt)
        rss = float(np.sum((y - pred)**2))
        n = len(y)
        return {
            "params": [float(p) for p in popt],
            "rss": rss,
            "AICc": _aicc(rss, n, k),
            "BIC": _bic(rss, n, k),
            "n_params": k,
            "fit_status": "ok",
        }
    except Exception as e:  # noqa: BLE001
        return {"params": None, "rss": float("nan"),
                 "AICc": float("nan"), "BIC": float("nan"),
                 "n_params": k, "fit_status": f"error: {type(e).__name__}"}


def _per_seed_R00_d(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    g_00 = prep["g_00_h"]
    t00 = prep["t00"]
    R_00 = g_00 + LAMBDA_T - t00
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat_inf = np.where(adj > 0, d_mat, np.inf)
    top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
    top_idx = np.argsort(np.abs(t00))[-top_n:].tolist()
    d_nearest = _bfs(adj, top_idx, d_mat_inf)
    finite = np.isfinite(d_nearest)
    return np.abs(R_00[finite]), d_nearest[finite]


def _process(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    abs_R, d_n = [], []
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            a, dd = _per_seed_R00_d(np.asarray(xi_mat), np.asarray(psi),
                                       np.asarray(k_field),
                                       np.asarray(q_field), n_lat)
        except Exception:  # noqa: BLE001
            continue
        abs_R.append(a); d_n.append(dd)
    if not abs_R:
        return None
    A = np.concatenate(abs_R); D = np.concatenate(d_n)
    edges = np.quantile(D, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    bin_centres = []
    bin_means = []
    bin_n = []
    for k in range(N_BINS):
        m = (D >= edges[k]) & (D < edges[k + 1])
        if m.sum() < 5:
            continue
        bin_centres.append(0.5 * (edges[k] + edges[k + 1]))
        bin_means.append(float(A[m].mean()))
        bin_n.append(int(m.sum()))
    if len(bin_means) < 5:
        return None
    r = np.array(bin_centres); y = np.array(bin_means)
    fits = {}
    for name, (fn, k) in PROFILES.items():
        fits[name] = _fit_profile(name, fn, k, r, y)
    aiccs = {n: f["AICc"] for n, f in fits.items()
             if np.isfinite(f["AICc"])}
    if aiccs:
        best = min(aiccs, key=aiccs.get)
        a_min = aiccs[best]
        for n, f in fits.items():
            if np.isfinite(f["AICc"]):
                f["delta_AICc"] = f["AICc"] - a_min
    return {
        "regime": regime, "N": int(n_lat),
        "n_pooled": int(len(A)),
        "bin_edges": edges.tolist(),
        "bin_centres": [float(x) for x in r],
        "bin_means": [float(x) for x in y],
        "bin_counts": bin_n,
        "fits": fits,
        "best_AICc": best if aiccs else None,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        print(f"  {regime:>10s}  N={n_lat:>4d} ...")
        r = _process(regime, n_lat, max_seeds=24)
        if r is None:
            continue
        rows.append(r)
        bests = [(name, f["delta_AICc"])
                 for name, f in r["fits"].items()
                 if "delta_AICc" in f]
        bests.sort(key=lambda x: x[1])
        line = ", ".join(f"{n}({d:+.1f})" for n, d in bests[:4])
        print(f"    AICc top4: {line}   best={r['best_AICc']}")

    win_counts = {}
    for r in rows:
        b = r["best_AICc"]
        if b is None:
            continue
        win_counts[b] = win_counts.get(b, 0) + 1
    print()
    print("AICc-best per regime:")
    for k, v in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:>14s}: {v}/{len(rows)}")

    out = {
        "method": "Item 5: shape-fit on |R_00| binned by d_nearest",
        "n_bins": N_BINS,
        "models": list(PROFILES.keys()),
        "per_regime": rows,
        "summary": {
            "n_regimes": len(rows),
            "AICc_best_counts": win_counts,
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
