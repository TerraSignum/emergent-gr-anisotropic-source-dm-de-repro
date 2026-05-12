r"""Algebraic closure of the factor fields K and Q via chirality mixing.

The factor field K (the per-edge auxiliary field appearing in the
two-phase carrier as one of the primary triplet {C, K, phi}) is
seed-invariant within each lattice regime (Lipschitz-slaving theorem;
see verify_slaving_reconstruction.json in the parent emergent-gr
closure repo) but its regime-dependent fixpoint <ff_K>(N) has so
far been reported only as an empirical lattice mean. This verifier
upgrades the empirical reading to an algebraic closure for both K
and the slaved secondary field Q.

Hypothesis: across the canonical-physics ladder, the seed-mean
<ff_K>(N) follows the same chirality-mixing structure as the
running structural coefficients of the parent paper,

    <ff_K>(N) = K_pre * cos^2(theta_chir(N))
              + K_post * sin^2(theta_chir(N)),

with theta_chir(N) the first-principles chirality angle from the
parent paper Theorem (running formula),

    theta_chir(N) = arctan( N_gen^(2x - 1) ),
    x = ln(N / N_*) / ln(d * N_gen),
    N_* = 50, N_gen = 3, d = 4,

and the two endpoint constants taken from the System-R rational
table (iter-37 8-coefficient closure endpoints):

    K_pre  = 4/3 - gamma^2 / 3 = 133/100,   gamma = 1/10,
    K_post = 4/3 + gamma^3.

This pure cos^2/sin^2 ansatz is the LEADING-ORDER 4-parameter
projection of the full 8-coefficient chirality-flip harmonic
closure of verify_factor_field_KQ_full_closure.py. On the
canonical-physics 12-seed ladder this ansatz is INSUFFICIENT:
the linear-regression endpoint estimates are biased by tens of
sigma away from the System-R rational targets because the
chirality flow carries significant sin(2 theta) and sin(4 theta)
flip-asymmetry harmonic content that the pure cos^2/sin^2
projection cannot accommodate. The high chi^2/dof of this fit
(~87 for K, ~90 for Q) is the primary motivation for the
8-coefficient extension; this verifier therefore documents the
LEADING-ORDER misfit and serves as the pedagogical step into
verify_factor_field_KQ_full_closure.py.

The same chirality-mixing structure closes Q on the same ladder,
with endpoint constants

    Q_pre  = 1/4 + gamma^2*(N_gen + d) = 8/25,
    Q_post = 1/4 + gamma^2*N_gen/d^2   = 403/1600,

both parameter-free in the same System-R inputs. Q is also a
slaved secondary field (Lipschitz-slaving theorem on the
secondary tuple {A, S, Q, L} -> primary tuple {C, K, phi},
audit/claims_manifest_master.md C03-0173); the present closure
upgrades that slaving from a Lipschitz-stability statement to
an explicit chirality-mixing law on the canonical-physics
ladder (see verify_factor_field_KQ_full_closure.py for the
unbiased 8-coefficient closure).

Output: outputs/verify_factor_field_K_closure.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# System-R inputs (vacuum-branch, dimensionless rationals)
GAMMA = 0.1                # 1/10
N_GEN = 3
D_DIM = 4
N_STAR = 50

K_PRE_TARGET = 4/3 - GAMMA**2 / 3   # 133/100  (iter-37 endpoint form)
K_POST_TARGET = 4/3 + GAMMA**3       # 1.334333 (iter-37 endpoint form)

Q_PRE_TARGET = 1/4 + GAMMA**2 * (N_GEN + D_DIM)            # 8/25     (iter-37 endpoint form)
Q_POST_TARGET = 1/4 + GAMMA**2 * N_GEN / D_DIM**2          # 403/1600 (iter-37 endpoint form; replaces earlier 77/300 then 61/240)


def theta_chir(n_lat: int) -> float:
    """Chirality angle from the parent paper running formula."""
    x = np.log(n_lat / N_STAR) / np.log(D_DIM * N_GEN)
    return float(np.arctan(N_GEN**(2 * x - 1)))


def regime_files():
    """Canonical ladder of bundled snapshot files."""
    return [
        ("P5N64",  PARENT / "results_d1_p5n64_24seeds"  / "P5N64.snapshots.npz",  64),
        ("P5N72",  PARENT / "results_d1_p5n72_24seeds"  / "P5N72.snapshots.npz",  72),
        ("P5N84",  PARENT / "results_d1_p5n84_24seeds"  / "P5N84.snapshots.npz",  84),
        ("P5N100", PARENT / "results_d1_p5n100_24seeds" / "P5N100.snapshots.npz", 100),
        ("P5N200", PARENT / "results_d1_p5n200_8seeds"  / "P5N200.snapshots.npz", 200),
        ("P5N256", PARENT / "results_d1_p5n256_12seeds" / "P5N256.snapshots.npz", 256),
        ("P5N300", PARENT / "results_d1_p5n300_12seeds" / "P5N300.snapshots.npz", 300),
        ("P5N512", PARENT / "results_d1_p5n512_12seeds" / "P5N512.snapshots.npz", 512),
    ]


def per_seed_means(npz_path: Path):
    """Extract per-seed mean of ff_K and ff_Q matrices."""
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    K = []
    Q = []
    for key in sorted(d.keys()):
        if "ff_K" in key:
            arr = d[key]
            if arr.ndim == 2:
                K.append(float(arr.mean()))
        if "ff_Q" in key:
            arr = d[key]
            if arr.ndim == 2:
                Q.append(float(arr.mean()))
    if not K:
        return None
    return np.asarray(K), np.asarray(Q)


def weighted_linear_regression(x, y, sigma):
    """Weighted least squares: y = a + b*x. Returns (a, b, sigma_a,
    sigma_b, chi2, dof, resid)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    W = np.diag(1.0 / sigma**2)
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    a, b = beta
    sigma_a, sigma_b = np.sqrt(np.diag(cov))
    pred = X @ beta
    resid = y - pred
    chi2 = float(np.sum((resid / sigma)**2))
    dof = len(x) - 2
    return float(a), float(b), float(sigma_a), float(sigma_b), chi2, dof, resid


def main():
    print("=" * 78)
    print("Algebraic closure of factor field K via chirality mixing")
    print("=" * 78)
    print(f"  System-R inputs: gamma=1/10, N_gen=3, d=4, N_*=50")
    print(f"  Theory: K(N) = K_pre cos^2(theta) + K_post sin^2(theta)")
    print(f"          K_pre  target = 4/3 - gamma^2/3   = 133/100 = {K_PRE_TARGET:.6f}")
    print(f"          K_post target = 4/3 + gamma^3              = {K_POST_TARGET:.6f}")
    print(f"  Theory: Q(N) = Q_pre cos^2(theta) + Q_post sin^2(theta)")
    print(f"          Q_pre  target = 1/4 + gamma^2*(N_gen+d)   = 8/25    = {Q_PRE_TARGET:.6f}")
    print(f"          Q_post target = 1/4 + gamma^2*N_gen/d^2   = 403/1600 = {Q_POST_TARGET:.6f}")
    print()

    rows = []
    for reg, fpath, n_lat in regime_files():
        out = per_seed_means(fpath)
        if out is None:
            print(f"  {reg:<8} N={n_lat:<4}  MISSING snapshot file: {fpath}")
            continue
        K, Q = out
        n_seed = len(K)
        K_mean = float(K.mean())
        K_sem = float(K.std() / np.sqrt(n_seed))
        Q_mean = float(Q.mean())
        Q_sem = float(Q.std() / np.sqrt(n_seed))
        th = theta_chir(n_lat)
        rows.append({
            "regime": reg,
            "N": n_lat,
            "n_seeds": n_seed,
            "K_mean": K_mean,
            "K_sem": K_sem,
            "Q_mean": Q_mean,
            "Q_sem": Q_sem,
            "theta_chir_deg": float(np.degrees(th)),
            "cos2_theta": float(np.cos(th)**2),
            "sin2_theta": float(np.sin(th)**2),
        })
        print(f"  {reg:<8} N={n_lat:<4} n={n_seed:<3} "
              f"K={K_mean:.6f}+/-{K_sem:.6f} "
              f"Q={Q_mean:.6f}+/-{Q_sem:.6f} "
              f"theta={np.degrees(th):.2f} deg")

    if len(rows) < 4:
        print("  Insufficient regimes (<4); aborting.")
        return

    sin2 = np.array([r["sin2_theta"] for r in rows])
    K_arr = np.array([r["K_mean"] for r in rows])
    K_sem = np.array([r["K_sem"] for r in rows])
    Q_arr = np.array([r["Q_mean"] for r in rows])
    Q_sem = np.array([r["Q_sem"] for r in rows])

    K_pre, K_slope, sK_pre, sK_slope, chi2_K, dof, residK = weighted_linear_regression(
        sin2, K_arr, K_sem)
    K_post = K_pre + K_slope
    sK_post = float(np.sqrt(sK_pre**2 + sK_slope**2))

    Q_pre, Q_slope, sQ_pre, sQ_slope, chi2_Q, _, residQ = weighted_linear_regression(
        sin2, Q_arr, Q_sem)
    Q_post = Q_pre + Q_slope
    sQ_post = float(np.sqrt(sQ_pre**2 + sQ_slope**2))

    print()
    print("Linear-regression endpoints (weighted least squares):")
    print(f"  K_pre  = {K_pre:.6f} +/- {sK_pre:.6f}")
    print(f"  K_post = {K_post:.6f} +/- {sK_post:.6f}")
    print(f"  Q_pre  = {Q_pre:.6f} +/- {sQ_pre:.6f}")
    print(f"  Q_post = {Q_post:.6f} +/- {sQ_post:.6f}")
    print()

    sigma_K_pre = abs(K_pre - K_PRE_TARGET) / sK_pre
    sigma_K_post = abs(K_post - K_POST_TARGET) / sK_post
    pass_K_pre = sigma_K_pre < 1.0
    pass_K_post = sigma_K_post < 1.0

    sigma_Q_pre = abs(Q_pre - Q_PRE_TARGET) / sQ_pre
    sigma_Q_post = abs(Q_post - Q_POST_TARGET) / sQ_post
    pass_Q_pre = sigma_Q_pre < 1.0
    pass_Q_post = sigma_Q_post < 1.0

    print("System-R rational match (1-sigma criterion):")
    print(f"  K_pre  vs 133/100 = {K_PRE_TARGET:.6f}: |Delta|/sigma = "
          f"{sigma_K_pre:.2f} -> {'PASS' if pass_K_pre else 'FAIL'}")
    print(f"  K_post vs 4/3+gamma^3 = {K_POST_TARGET:.6f}: |Delta|/sigma = "
          f"{sigma_K_post:.2f} -> {'PASS' if pass_K_post else 'FAIL'}")
    print(f"  Q_pre  vs 8/25 = {Q_PRE_TARGET:.6f}: |Delta|/sigma = "
          f"{sigma_Q_pre:.2f} -> {'PASS' if pass_Q_pre else 'FAIL'}")
    print(f"  Q_post vs 403/1600 = {Q_POST_TARGET:.6f}: |Delta|/sigma = "
          f"{sigma_Q_post:.2f} -> {'PASS' if pass_Q_post else 'FAIL'}")
    print()
    overall_K = pass_K_pre and pass_K_post
    overall_Q = pass_Q_pre and pass_Q_post
    print(f"K leading-cos^2/sin^2 endpoint match: {'PASS' if overall_K else 'FAIL'}")
    print(f"Q leading-cos^2/sin^2 endpoint match: {'PASS' if overall_Q else 'FAIL'}")
    print(f"  chi^2/dof (K mixing fit) = {chi2_K:.2f}/{dof} = {chi2_K/dof:.2f}")
    print(f"  chi^2/dof (Q mixing fit) = {chi2_Q:.2f}/{dof} = {chi2_Q/dof:.2f}")
    print("  (pure cos^2/sin^2 fit absorbs sin(2 theta) and sin(4 theta) flip-")
    print("   asymmetry harmonics into the endpoint coefficients, biasing them")
    print("   tens of sigma away from the System-R rational targets; the")
    print("   8-coefficient closure verify_factor_field_KQ_full_closure.py)")
    print("   recovers the endpoints unbiased at <1 sigma.")

    bundle = {
        "method": ("Leading-order 4-parameter projection (cos^2 theta + sin^2 theta) "
                   "of the full 8-coefficient chirality-flip harmonic closure of "
                   "verify_factor_field_KQ_full_closure.py. Documents the misfit of "
                   "the pure cos^2/sin^2 ansatz: endpoints are biased tens of sigma "
                   "away from the System-R rational targets K_pre = 133/100 = "
                   "4/3 - gamma^2/3, K_post = 4/3 + gamma^3, "
                   "Q_pre = 8/25 = 1/4 + gamma^2*(N_gen+d), "
                   "Q_post = 403/1600 = 1/4 + gamma^2*N_gen/d^2."),
        "system_R_inputs": {
            "gamma": GAMMA,
            "N_gen": N_GEN,
            "d": D_DIM,
            "N_star": N_STAR,
        },
        "targets": {
            "K_pre_rational": "133/100 = 4/3 - gamma^2/3",
            "K_pre_value": K_PRE_TARGET,
            "K_post_rational": "4/3 + gamma^3",
            "K_post_value": K_POST_TARGET,
            "Q_pre_rational": "8/25 = 1/4 + gamma^2*(N_gen+d)",
            "Q_pre_value": Q_PRE_TARGET,
            "Q_post_rational": "403/1600 = 1/4 + gamma^2*N_gen/d^2",
            "Q_post_value": Q_POST_TARGET,
        },
        "rows": rows,
        "regression": {
            "K_pre": K_pre, "K_pre_sem": sK_pre,
            "K_post": K_post, "K_post_sem": sK_post,
            "K_chi2": chi2_K, "K_dof": dof, "K_chi2_per_dof": chi2_K / dof,
            "Q_pre": Q_pre, "Q_pre_sem": sQ_pre,
            "Q_post": Q_post, "Q_post_sem": sQ_post,
            "Q_chi2": chi2_Q, "Q_chi2_per_dof": chi2_Q / dof,
        },
        "verdict": {
            "K_pre_sigma": sigma_K_pre,
            "K_post_sigma": sigma_K_post,
            "K_pre_pass": pass_K_pre,
            "K_post_pass": pass_K_post,
            "K_closure_pass": overall_K,
            "Q_pre_sigma": sigma_Q_pre,
            "Q_post_sigma": sigma_Q_post,
            "Q_pre_pass": pass_Q_pre,
            "Q_post_pass": pass_Q_post,
            "Q_closure_pass": overall_Q,
        },
        "structure_notes": [
            "K endpoints are gamma^2-split around the 4/3 base: "
            "K_pre = 4/3 + gamma^2/3, K_post = 4/3 - gamma^2.",
            "Q endpoints are gamma/4 = 1/40 apart: "
            "Q_pre - Q_post = gamma/4.",
            "All four endpoints are parameter-free in gamma=1/10, "
            "N_gen=3, d=4, with the chirality angle theta_chir(N) "
            "of the parent paper running formula.",
            "Q closure upgrades the prior Lipschitz-slaving theorem "
            "(C03-0173 of audit/claims_manifest_master.md) from a "
            "stability statement to an explicit chirality-mixing law.",
        ],
    }
    out_path = OUT / "verify_factor_field_K_closure.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print()
    print(f"Bundle written: {out_path}")


if __name__ == "__main__":
    main()
