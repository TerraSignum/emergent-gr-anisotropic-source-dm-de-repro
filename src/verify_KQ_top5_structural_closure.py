r"""Matter-localised top-5% closure of the lattice fixpoints
<K>_top5(N) and <Q>_top5(N) on the eight-regime canonical-physics
ladder, with all six coefficients fixed at System-R rational
targets in (gamma, N_gen, d) and the lattice-resonance v_2(N)
indicator added to Q.

Closure form (zero free parameters):

    <K>_top5(N) = (5/4) cos^2(theta) + (4/3 - gamma^2) sin^2(theta)
    <Q>_top5(N) = (1/N_gen^2) cos^2(theta)
                + (2/(d^2 - 1)) sin^2(theta)
                + (2/d^2) sin(2 theta)
                + (1/(d^2 (d^2 - 1))) v_2(N)

with theta = theta_chir(N) the running chirality angle of
Eq. (theta_running_first_principles) and v_2(N) := the largest
integer k such that 2^k divides N (binary-subdivision content
of the lattice).

The structural origins:
    A_K = 5/4               (K_pre vacuum endpoint)
    B_K = 4/3 - gamma^2     (K_post matter endpoint, leading-gamma
                             correction to the universal 4/3 template)
    A_Q = 1/N_gen^2 = 1/9   (Q_pre vacuum endpoint)
    B_Q = 2/(d^2 - 1) = 2/15 (Q_post matter endpoint)
    S_Q = 2/d^2 = 1/8       (chirality-mixing sin(2 theta) amplitude)
    V_Q = 1/(d^2 (d^2-1))   (matter-resonator factor; equivalently
        = gamma^2 (2 N_gen - 1) / (4 N_gen) on the same (2k-1)/(4k)
        projector family as the refined beta_pi sin^2 coefficient
        23/48 = (2 d N_gen - 1)/(4 d N_gen))

Volume-mean K, Q on the canonical 12-seed ladder give chi^2/dof
= 19.6 (K) and a non-trivial value for Q dominated by the borderline
flip-asymmetry harmonics a_Q, b_Q at 1.29, 1.44 sigma (eight-
coefficient harmonic closure, Eq. KQ_full_closure in the manuscript);
the matter-localised top-5% restriction reduces this to chi^2/(2 N_reg)
~ 0.5 with zero free parameters.

Output: outputs/verify_KQ_top5_full_structural_closure.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

GAMMA = 0.1
N_GEN = 3
D_DIM = 4
N_STAR = 50


def theta_chir(n_lat):
    x = np.log(n_lat / N_STAR) / np.log(D_DIM * N_GEN)
    return np.arctan(N_GEN ** (2 * x - 1))


def v2_of_N(n):
    """v_2(n) = largest k such that 2^k divides n."""
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def regime_files():
    base = PARENT
    return [
        ("P5N64",  base / "results_d1_p5n64_24seeds"  / "P5N64.snapshots.npz",  64),
        ("P5N72",  base / "results_d1_p5n72_24seeds"  / "P5N72.snapshots.npz",  72),
        ("P5N84",  base / "results_d1_p5n84_24seeds"  / "P5N84.snapshots.npz",  84),
        ("P5N100", base / "results_d1_p5n100_24seeds" / "P5N100.snapshots.npz", 100),
        ("P5N200", base / "results_d1_p5n200_8seeds"  / "P5N200.snapshots.npz", 200),
        ("P5N256", base / "results_d1_p5n256_12seeds" / "P5N256.snapshots.npz", 256),
        ("P5N300", base / "results_d1_p5n300_12seeds" / "P5N300.snapshots.npz", 300),
        ("P5N512", base / "results_d1_p5n512_12seeds" / "P5N512.snapshots.npz", 512),
    ]


def top5_means(K_mat, Q_mat, xi_mat, top_frac=0.05):
    """Restrict K, Q to the top fraction of nodes by Xi row-sum.

    Returns (K_top_mean, Q_top_mean): mean of K, Q sub-matrix on
    the selected matter-localised node block.
    """
    np.fill_diagonal(xi_mat, 0.0)
    deg = xi_mat.sum(axis=1)
    n = len(deg)
    n_top = max(1, int(np.ceil(top_frac * n)))
    top_idx = np.argsort(-deg)[:n_top]
    K_block = K_mat[np.ix_(top_idx, top_idx)]
    Q_block = Q_mat[np.ix_(top_idx, top_idx)]
    return float(K_block.mean()), float(Q_block.mean())


def main():
    print("=" * 78)
    print("K, Q matter-localised top-5% structural closure")
    print("Zero free parameters in (gamma, N_gen, d)")
    print("=" * 78)

    rows = []
    for reg, fpath, N in regime_files():
        if not fpath.exists():
            continue
        d = np.load(fpath, allow_pickle=True)
        edge_snap = d["edge_xi_snapshots"]
        K_keys = sorted([k for k in d.keys() if "ff_K" in k])
        Q_keys = sorted([k for k in d.keys() if "ff_Q" in k])
        K_vals, Q_vals = [], []
        for s, (kk, qk) in enumerate(zip(K_keys, Q_keys)):
            xi = edge_snap[s, -1].astype(float).copy()
            np.fill_diagonal(xi, 1.0)
            K_mat = np.asarray(d[kk], dtype=float)
            Q_mat = np.asarray(d[qk], dtype=float)
            if K_mat.ndim != 2:
                continue
            Kv, Qv = top5_means(K_mat, Q_mat, xi.copy(), top_frac=0.05)
            K_vals.append(Kv)
            Q_vals.append(Qv)
        K_vals = np.array(K_vals)
        Q_vals = np.array(Q_vals)
        n_s = len(K_vals)
        rows.append({
            "regime": reg, "N": N, "n_seeds": n_s,
            "v_2_N": v2_of_N(N),
            "theta_chir_rad": float(theta_chir(N)),
            "theta_chir_deg": float(np.degrees(theta_chir(N))),
            "K_mean": float(K_vals.mean()),
            "K_sem": float(K_vals.std() / np.sqrt(max(n_s, 1))),
            "Q_mean": float(Q_vals.mean()),
            "Q_sem": float(Q_vals.std() / np.sqrt(max(n_s, 1))),
        })

    # Predict with zero-free-parameter closure
    A_K = 5.0 / 4.0
    B_K = 4.0 / 3.0 - GAMMA ** 2
    A_Q = 1.0 / (N_GEN ** 2)
    B_Q = 2.0 / (D_DIM ** 2 - 1)
    S_Q = 2.0 / (D_DIM ** 2)
    V_Q = 1.0 / (D_DIM ** 2 * (D_DIM ** 2 - 1))

    chi2_K = 0.0
    chi2_Q = 0.0
    print(f"\n{'reg':<8}{'N':>5} {'theta':>8} {'v_2':>4}  {'K_obs':>9} {'K_pred':>9}"
          f" {'z_K':>6}   {'Q_obs':>9} {'Q_pred':>9} {'z_Q':>6}")
    for r in rows:
        t = r["theta_chir_rad"]
        c2 = np.cos(t) ** 2
        s2 = np.sin(t) ** 2
        s2t = np.sin(2 * t)
        v2 = r["v_2_N"]
        K_pred = A_K * c2 + B_K * s2
        Q_pred = A_Q * c2 + B_Q * s2 + S_Q * s2t + V_Q * v2
        z_K = (r["K_mean"] - K_pred) / r["K_sem"]
        z_Q = (r["Q_mean"] - Q_pred) / r["Q_sem"]
        chi2_K += z_K ** 2
        chi2_Q += z_Q ** 2
        r["K_predicted"] = float(K_pred)
        r["Q_predicted"] = float(Q_pred)
        r["K_residual_z"] = float(z_K)
        r["Q_residual_z"] = float(z_Q)
        print(f"{r['regime']:<8}{r['N']:>5} {r['theta_chir_deg']:>6.2f}deg {v2:>4}"
              f"  {r['K_mean']:>9.5f} {K_pred:>9.5f} {z_K:>+6.2f}"
              f"   {r['Q_mean']:>9.5f} {Q_pred:>9.5f} {z_Q:>+6.2f}")

    n_reg = len(rows)
    print(f"\nK chi^2/N_reg = {chi2_K / n_reg:.3f}  (parameter-free, n={n_reg})")
    print(f"Q chi^2/N_reg = {chi2_Q / n_reg:.3f}  (parameter-free, n={n_reg})")
    print(f"Joint chi^2/(2 N_reg) = {(chi2_K + chi2_Q) / (2 * n_reg):.3f}")
    print(f"PASS chi^2 < 2: {(chi2_K + chi2_Q) / (2 * n_reg) < 2}")

    bundle = {
        "stage": "K_Q_top_5pct_chirality_lattice_resonance_full_closure",
        "summary": ("K and Q top-5%-of-nodes (matter-localized cluster cores) "
                    "closure across 8 canonical-physics regimes "
                    "(P5N64..P5N512) with ZERO free parameters: all 6 "
                    "coefficients are exact System-R rationals in "
                    "(d, N_gen, gamma=1/10). Joint chi^2/(2 N_reg) = "
                    f"{(chi2_K + chi2_Q) / (2 * n_reg):.3f}."),
        "closure_forms": {
            "K": {
                "form": "K(theta) = (5/4) cos^2(theta) + (4/3 - gamma^2) sin^2(theta)",
                "A_K": A_K, "B_K": B_K,
                "chi2_per_N_reg": chi2_K / n_reg,
            },
            "Q": {
                "form": ("Q(theta, N) = (1/N_gen^2) cos^2(theta) + "
                         "(2/(d^2-1)) sin^2(theta) + (2/d^2) sin(2 theta) "
                         "+ (1/(d^2 (d^2-1))) v_2(N)"),
                "A_Q": A_Q, "B_Q": B_Q, "S_Q": S_Q, "V_Q": V_Q,
                "V_Q_alternative": ("gamma^2 (2 N_gen - 1) / (4 N_gen) "
                                     f"= {GAMMA**2 * (2*N_GEN - 1) / (4*N_GEN)}"),
                "chi2_per_N_reg": chi2_Q / n_reg,
            },
        },
        "running_chirality_angle": {
            "form": "tan(theta_chir(N)) = N_gen^(2x - 1), x = ln(N/N_*) / ln(d N_gen)",
            "constants": {"N_star": N_STAR, "N_gen": N_GEN, "d": D_DIM},
        },
        "joint_metric": {
            "K_chi2_per_N_reg": chi2_K / n_reg,
            "Q_chi2_per_N_reg": chi2_Q / n_reg,
            "joint_chi2_per_2N_reg": (chi2_K + chi2_Q) / (2 * n_reg),
            "PASS_chi2_lt_2": bool((chi2_K + chi2_Q) / (2 * n_reg) < 2),
            "n_regimes": n_reg,
            "free_parameters": 0,
        },
        "rows": rows,
    }
    out_path = OUT / "verify_KQ_top5_full_structural_closure.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nBundle: {out_path}")


if __name__ == "__main__":
    main()
