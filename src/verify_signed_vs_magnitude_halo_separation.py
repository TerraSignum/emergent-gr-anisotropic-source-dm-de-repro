"""Clean separation of signed-halo vs magnitude-halo diagnostics.

The two halo readings on the per-node R_00(a) field are
\\emph{distinct} observables and require separate quantification:

  * SIGNED halo = signed Spearman of R_00(a) vs distance to
    matter-cluster boundary, rho(R_00, d). This captures the
    \\emph{polarity} of the energy-density residual: does R_00
    go from negative (near cluster) to positive (far from
    cluster) as distance grows? On all canonical-physics
    regimes the empirical answer is "yes" with rho in
    [+0.31, +0.47] on 10/10 regimes (regime-stable).

  * MAGNITUDE halo = signed Spearman of |R_00|(a) vs distance,
    rho(|R_00|, d). Together with NFW shape-fit goodness, this
    captures the \\emph{radial magnitude profile} of the
    residual: does the absolute residual decrease with
    distance like an NFW halo? The empirical answer varies
    with N: rho(|R_00|, d) is strongly negative at moderate N
    but weakens at high N (P5N200, P5N300) because more
    sign-flip nodes occupy the cluster-far transition zone
    where R_00 crosses zero, dragging |R_00| down at
    intermediate distances and degrading the NFW shape match.

This audit decouples the two diagnostics, computes a
"polarity-violation rate" that is the structural cause of the
magnitude-halo degradation, and reports them separately. Output
JSON has explicit per-regime entries for signed Spearman,
magnitude Spearman, polarity-violation rate (fraction of nodes
whose R_00 sign disagrees with the cluster-boundary expectation
given their distance), and NFW shape-fit AICc.
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
    LAMBDA_T, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "verify_signed_vs_magnitude_halo_separation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

REGIMES = [
    # Canonical P5/P5N N-ordered ladder.
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
    # Alt-anchor cross-checks (separated; not part of canonical ladder).
    ("P6",     60),
    ("P7",     72),
    ("P8",     84),
]
CORE_TOP_FRAC = 0.05


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    s = float(np.sum(rx * ry))
    norm = math.sqrt(float(np.sum(rx ** 2)) * float(np.sum(ry ** 2)))
    return s / norm if norm > 0 else float("nan")


def fit_nfw_log(d_norm, abs_R_norm):
    """NFW |R(d)| ~ rho_0 / [(d/r_s) (1 + d/r_s)^2] in log-log
    space. Two parameters (rho_0, r_s); return AICc on log
    residuals."""
    mask = (d_norm > 1e-6) & (abs_R_norm > 1e-9)
    if mask.sum() < 6:
        return float("nan")
    d = d_norm[mask]
    y = abs_R_norm[mask]
    log_y = np.log(y)
    best_aicc = float("inf")
    for rs in np.linspace(0.05, 1.0, 25):
        x = d / rs
        log_pred = -np.log(x * (1 + x) ** 2)
        # Fit log_rho_0 by mean offset
        log_rho0 = float(np.mean(log_y - log_pred))
        log_pred_full = log_pred + log_rho0
        ss_res = float(np.sum((log_y - log_pred_full) ** 2))
        n = len(log_y)
        if ss_res > 0:
            nll = 0.5 * n * math.log(ss_res / n)
            aicc = 4.0 + 2 * nll
            if n - 3 > 0:
                aicc += 12.0 / (n - 3)
            if aicc < best_aicc:
                best_aicc = aicc
    return best_aicc


def fit_uniform_log(d_norm, abs_R_norm):
    mask = (d_norm > 1e-6) & (abs_R_norm > 1e-9)
    if mask.sum() < 6:
        return float("nan")
    log_y = np.log(abs_R_norm[mask])
    log_const = float(np.mean(log_y))
    ss_res = float(np.sum((log_y - log_const) ** 2))
    n = len(log_y)
    nll = 0.5 * n * math.log(ss_res / n) if ss_res > 0 else -1e9
    aicc = 2.0 + 2 * nll
    if n - 2 > 0:
        aicc += 4.0 / (n - 2)
    return aicc


def _process_regime(regime, n_lat):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat) if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))
    if not seeds:
        return None

    R_00_pool = []
    d_pool = []
    T00_pool = []
    n_signs_neg = 0
    n_signs_pos = 0
    for s in seeds:
        xi_mat = np.asarray(s[0], float).copy()
        psi = np.asarray(s[1])
        k = np.asarray(s[2]) if len(s) > 2 else None
        q = np.asarray(s[3]) if len(s) > 3 else None
        prep = per_seed_galerkin(xi_mat, psi, k, q, n_lat, np)
        r00 = prep["g_00_h"] + LAMBDA_T - prep["t00"]
        t00 = prep["t00"]

        # Distance from cluster boundary: nearest neighbour of any
        # core node (top 5% |T_00|)
        top_n = max(1, int(np.ceil(CORE_TOP_FRAC * n_lat)))
        core_idx = np.argsort(np.abs(t00))[-top_n:]
        # Edge-distance metric: -log(Xi)
        xi_off = xi_mat.copy()
        np.fill_diagonal(xi_off, 0.0)
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_to_core = d_mat[:, core_idx].min(axis=1)

        bulk = np.array([i for i in range(n_lat) if i not in core_idx])
        R_00_pool.extend(r00[bulk].tolist())
        d_pool.extend(d_to_core[bulk].tolist())
        T00_pool.extend(np.abs(t00[bulk]).tolist())
        n_signs_neg += int((r00[bulk] < 0).sum())
        n_signs_pos += int((r00[bulk] > 0).sum())

    R_00 = np.array(R_00_pool, float)
    d = np.array(d_pool, float)
    T00 = np.array(T00_pool, float)

    rho_signed = spearman(R_00, d)
    rho_magnitude = spearman(np.abs(R_00), d)
    rho_T = spearman(T00, d)

    f_neg = n_signs_neg / max(n_signs_neg + n_signs_pos, 1)

    # Polarity-violation rate: fraction of nodes whose R_00 sign
    # disagrees with the dominant halo polarity (negative near
    # cluster, positive far from cluster). The halo prediction:
    # nodes with d < median(d) should have R_00 < 0; nodes with
    # d >= median(d) should have R_00 > 0. A "violator" is a
    # node breaking that pattern.
    d_med = float(np.median(d))
    near = d < d_med
    far = ~near
    n_violations = int(((near) & (R_00 > 0)).sum()
                       + ((far) & (R_00 < 0)).sum())
    polarity_viol_rate = n_violations / max(len(R_00), 1)

    # NFW vs uniform shape fit on |R_00| in normalized
    # (d / d_max) coordinates
    d_norm = d / max(d.max(), 1e-12)
    abs_R = np.abs(R_00)
    abs_R_norm = abs_R / max(abs_R.max(), 1e-12)
    aicc_nfw = fit_nfw_log(d_norm, abs_R_norm)
    aicc_unif = fit_uniform_log(d_norm, abs_R_norm)
    delta_aicc_nfw_vs_unif = (aicc_nfw - aicc_unif
                               if math.isfinite(aicc_nfw)
                               and math.isfinite(aicc_unif)
                               else float("nan"))

    return {
        "regime":           regime,
        "N":                int(n_lat),
        "n_seeds":          len(seeds),
        "n_bulk_pooled":    int(len(R_00)),
        "f_R00_negative":   f_neg,
        "rho_signed_R00_vs_d":     rho_signed,
        "rho_magnitude_absR_vs_d": rho_magnitude,
        "rho_T00_vs_d":            rho_T,
        "polarity_violation_rate": polarity_viol_rate,
        "aicc_NFW_log":           aicc_nfw,
        "aicc_uniform_log":        aicc_unif,
        "delta_aicc_NFW_vs_uniform": delta_aicc_nfw_vs_unif,
    }


def main():
    rows = []
    for regime, n_lat in REGIMES:
        r = _process_regime(regime, n_lat)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No data; output skipped.")
        return

    print(f"{'regime':<8}{'N':>5}{'seeds':>6}"
          f"{'rho_signed':>12}{'rho_mag':>10}"
          f"{'pol_viol':>10}{'dAICc(NFW-uni)':>16}")
    print("-" * 68)
    for r in rows:
        print(f"{r['regime']:<8}{r['N']:>5}{r['n_seeds']:>6}"
              f"{r['rho_signed_R00_vs_d']:>+12.4f}"
              f"{r['rho_magnitude_absR_vs_d']:>+10.4f}"
              f"{r['polarity_violation_rate']:>10.4f}"
              f"{r['delta_aicc_NFW_vs_uniform']:>+16.2f}")

    # Cross-regime correlations
    rho_signed_arr = np.array([r["rho_signed_R00_vs_d"] for r in rows])
    rho_mag_arr = np.array([r["rho_magnitude_absR_vs_d"] for r in rows])
    pol_viol_arr = np.array([r["polarity_violation_rate"] for r in rows])
    n_arr = np.array([r["N"] for r in rows], float)

    cor_pol_mag = float(np.corrcoef(pol_viol_arr, rho_mag_arr)[0, 1])
    cor_pol_signed = float(np.corrcoef(pol_viol_arr, rho_signed_arr)[0, 1])
    cor_N_mag = float(np.corrcoef(n_arr, rho_mag_arr)[0, 1])
    cor_N_signed = float(np.corrcoef(n_arr, rho_signed_arr)[0, 1])

    summary = {
        "n_regimes": len(rows),
        "rho_signed_mean":     float(rho_signed_arr.mean()),
        "rho_signed_min":      float(rho_signed_arr.min()),
        "rho_signed_max":      float(rho_signed_arr.max()),
        "rho_signed_all_positive_n_of_N":
            f"{int((rho_signed_arr > 0).sum())}/{len(rows)}",
        "rho_magnitude_mean":  float(rho_mag_arr.mean()),
        "rho_magnitude_min":   float(rho_mag_arr.min()),
        "rho_magnitude_max":   float(rho_mag_arr.max()),
        "rho_magnitude_all_negative_n_of_N":
            f"{int((rho_mag_arr < 0).sum())}/{len(rows)}",
        "polarity_violation_mean": float(pol_viol_arr.mean()),
        "corr_polarity_violation_vs_magnitude_rho":  cor_pol_mag,
        "corr_polarity_violation_vs_signed_rho":     cor_pol_signed,
        "corr_N_vs_magnitude_rho":                    cor_N_mag,
        "corr_N_vs_signed_rho":                       cor_N_signed,
    }

    out = {
        "method": ("Signed-halo (rho(R_00, d_core)) vs magnitude-halo "
                   "(rho(|R_00|, d_core); NFW vs uniform shape) "
                   "separation audit on canonical-physics ladder"),
        "definitions": {
            "signed_halo":    "rho_Spearman(R_00(a), d(a, core))",
            "magnitude_halo": "rho_Spearman(|R_00(a)|, d(a, core))",
            "polarity_violation_rate":
                "fraction of bulk nodes whose R_00 sign disagrees "
                "with the dominant halo polarity (R_00 < 0 near "
                "cluster, R_00 > 0 far from cluster, split at "
                "median d)",
            "delta_aicc_NFW_vs_uniform":
                "AICc(NFW) - AICc(uniform) on log |R_00|(d). "
                "Negative => NFW preferred over uniform; "
                "positive => uniform preferred (no radial profile)",
        },
        "per_regime": rows,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print("=== Cross-regime summary ===")
    print(f"  signed   rho mean = {summary['rho_signed_mean']:+.4f}, "
          f"range = [{summary['rho_signed_min']:+.4f}, "
          f"{summary['rho_signed_max']:+.4f}], "
          f"all positive: {summary['rho_signed_all_positive_n_of_N']}")
    print(f"  magnit.  rho mean = {summary['rho_magnitude_mean']:+.4f}, "
          f"range = [{summary['rho_magnitude_min']:+.4f}, "
          f"{summary['rho_magnitude_max']:+.4f}], "
          f"all negative: {summary['rho_magnitude_all_negative_n_of_N']}")
    print(f"  Pearson(polarity_viol, magnitude rho)  "
          f"= {cor_pol_mag:+.4f}")
    print(f"  Pearson(polarity_viol, signed rho)     "
          f"= {cor_pol_signed:+.4f}")
    print(f"  Pearson(N, magnitude rho)              "
          f"= {cor_N_mag:+.4f}")
    print(f"  Pearson(N, signed rho)                 "
          f"= {cor_N_signed:+.4f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
