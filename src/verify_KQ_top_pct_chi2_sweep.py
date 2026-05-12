"""K, Q matter-localised closure chi^2 sweep across top-X%
percentiles.

The bundled top-5% closure of K, Q (verify_KQ_top5_full_structural_closure.json)
reports the chirality-mixing form

  <K>_top5 = (5/4) cos^2(theta) + (4/3 - gamma^2) sin^2(theta)
  <Q>_top5 = (1/N_gen^2) cos^2(theta) + (2/(d^2 - 1)) sin^2(theta)
           + (2/d^2) sin(2 theta) + v_2(N)/(d^2*(d^2-1)) sin(4 theta)

with chi^2 / (2 N_reg) = 0.53 on the eight-regime ladder with
zero free parameters (v_2(N) integer per regime).

This sweep measures <K>_topX, <Q>_topX at multiple top-X%
percentiles to see whether the chi^2 tightens (improves toward
zero) or loosens at smaller / larger sub-graph fractions. The
hypothesis to test (user 2026-05-07): can we get the chi^2
closer to zero at top-3% or top-2% than at top-5%?

Output: outputs/verify_KQ_top_pct_chi2_sweep.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_module(self, name, _path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO.parent

GAMMA = 0.1
ALPHA_XI = 0.9
N_GEN = 3
D = 4
N_STAR = 50

# Closure prediction coefficients (from bundled top5 audit)
# Match exactly the original verify_KQ_top5_structural_closure.py
# convention:
#   K_pred = (5/4) cos^2 + (4/3 - gamma^2) sin^2
#   Q_pred = (1/N_gen^2) cos^2 + (2/(d^2 - 1)) sin^2
#          + (2/d^2) sin(2 theta)
#          + (1/(d^2 (d^2 - 1))) * v_2(N)         [CONSTANT, no sin]
# with v_2(N) = 2-adic valuation of N (largest k s.t. 2^k | N).
K_PRE = 5.0 / 4.0                            # 5/4 cos^2
K_POST = 4.0 / 3.0 - GAMMA ** 2              # (4/3 - gamma^2) sin^2
Q_PRE = 1.0 / N_GEN ** 2                     # 1/9 cos^2
Q_POST = 2.0 / (D ** 2 - 1)                  # 2/15 sin^2
Q_A = 2.0 / D ** 2                           # 2/16 = 1/8 sin(2 theta)
V_Q = 1.0 / (D ** 2 * (D ** 2 - 1))          # 1/240 (constant times v_2)


def v2_of_n(n: int) -> int:
    """v_2(n) = 2-adic valuation: largest k s.t. 2^k divides n."""
    k = 0
    while n % 2 == 0 and n > 0:
        n //= 2
        k += 1
    return k

LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
]

PERCENTILES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def chirality_angle(n_lat: int) -> float:
    x = math.log(n_lat / N_STAR) / math.log(D * N_GEN)
    return math.atan(N_GEN ** (2 * x - 1))


def kq_top_pct_per_seed(xi_mat, psi, k_field, q_field, pct: float):
    """Mean K, Q on the top-X% Xi-degree sub-graph (the same
    matter-localised selector used in the original top-5% closure
    of verify_KQ_top5_structural_closure.py). Returns the
    full-submatrix mean (including diagonal), since that is the
    convention of the published closure that gives
    chi^2/(2 N_reg) = 0.53 at top-5%."""
    n = xi_mat.shape[0]
    n_top = max(int(math.ceil(pct / 100.0 * n)), 4)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    deg = xi_off.sum(axis=1)
    idx = np.argsort(-deg)[:n_top]
    idx_sorted = np.sort(idx)
    k_sub = k_field[np.ix_(idx_sorted, idx_sorted)]
    q_sub = q_field[np.ix_(idx_sorted, idx_sorted)]
    return float(k_sub.mean()), float(q_sub.mean()), n_top


def predict_k(theta: float) -> float:
    return K_PRE * math.cos(theta) ** 2 + K_POST * math.sin(theta) ** 2


def predict_q(theta: float, v2: int) -> float:
    return (Q_PRE * math.cos(theta) ** 2
            + Q_POST * math.sin(theta) ** 2
            + Q_A * math.sin(2.0 * theta)
            + V_Q * v2)


def main() -> int:
    rows_per_pct = {p: [] for p in PERCENTILES}
    for label, n_lat, sub in LADDER:
        path = REPO_ROOT / sub
        if not path.exists():
            print(f"  skip {label}: {path} missing")
            continue
        z = np.load(path, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_re = z["psi_real_snapshots"]
        psi_im = z["psi_imag_snapshots"]
        last_idx = snaps.shape[1] - 1
        n_seeds = int(snaps.shape[0])
        # Use ff_K_seed{s} / ff_Q_seed{s} to exactly match the
        # original verify_KQ_top5_structural_closure.py convention
        # (per-seed final state K, Q matrix).
        k_keys = sorted([k for k in z.files if k.startswith("ff_K_seed")])
        q_keys = sorted([k for k in z.files if k.startswith("ff_Q_seed")])
        if not k_keys or not q_keys:
            print(f"  skip {label}: no ff_K/ff_Q keys in NPZ")
            continue
        theta = chirality_angle(n_lat)
        v2 = v2_of_n(n_lat)
        for pct in PERCENTILES:
            ks = []
            qs = []
            n_top_list = []
            n_pairs = min(n_seeds, len(k_keys), len(q_keys))
            for s in range(n_pairs):
                xi = np.asarray(snaps[s, last_idx], dtype=float)
                xi = 0.5 * (xi + xi.T)
                psi = (np.asarray(psi_re[s, last_idx],
                                    dtype=float)
                        + 1j * np.asarray(psi_im[s, last_idx],
                                           dtype=float))
                kf = np.asarray(z[k_keys[s]], dtype=float)
                qf = np.asarray(z[q_keys[s]], dtype=float)
                k_mean, q_mean, n_top = kq_top_pct_per_seed(
                    xi, psi, kf, qf, pct)
                ks.append(k_mean)
                qs.append(q_mean)
                n_top_list.append(n_top)
            k_arr = np.array(ks)
            q_arr = np.array(qs)
            k_mean_seed = float(k_arr.mean())
            q_mean_seed = float(q_arr.mean())
            k_sem = float(k_arr.std(ddof=1)
                          / math.sqrt(len(k_arr)))
            q_sem = float(q_arr.std(ddof=1)
                          / math.sqrt(len(q_arr)))
            k_pred = predict_k(theta)
            q_pred = predict_q(theta, v2)
            k_z = ((k_mean_seed - k_pred)
                    / max(k_sem, 1e-9))
            q_z = ((q_mean_seed - q_pred)
                    / max(q_sem, 1e-9))
            rows_per_pct[pct].append({
                "regime": label,
                "N": n_lat,
                "theta_chir_deg": math.degrees(theta),
                "v_2": v2,
                "n_top_median": int(np.median(n_top_list)),
                "K_mean":  k_mean_seed,
                "Q_mean":  q_mean_seed,
                "K_sem":   k_sem,
                "Q_sem":   q_sem,
                "K_pred":  k_pred,
                "Q_pred":  q_pred,
                "K_z":     k_z,
                "Q_z":     q_z,
            })
            print(f"  {label} top-{pct:>4.1f}%  "
                  f"K={k_mean_seed:.4f} Q={q_mean_seed:.4f}  "
                  f"K_pred={k_pred:.4f} Q_pred={q_pred:.4f}  "
                  f"z_K={k_z:+.2f} z_Q={q_z:+.2f}")

    summary_per_pct = {}
    for pct, rows in rows_per_pct.items():
        if not rows:
            continue
        z_k = np.array([r["K_z"] for r in rows])
        z_q = np.array([r["Q_z"] for r in rows])
        chi2_k = float(np.sum(z_k ** 2))
        chi2_q = float(np.sum(z_q ** 2))
        n_reg = len(rows)
        summary_per_pct[f"top_{pct:g}"] = {
            "percentile":     pct,
            "n_regimes":      n_reg,
            "chi2_K":         chi2_k,
            "chi2_K_per_dof": chi2_k / max(n_reg, 1),
            "chi2_Q":         chi2_q,
            "chi2_Q_per_dof": chi2_q / max(n_reg, 1),
            "joint_chi2_per_2N":
                (chi2_k + chi2_q) / max(2 * n_reg, 1),
            "rows": rows,
        }

    out = (REPO / "outputs"
           / "verify_KQ_top_pct_chi2_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "verify_KQ_top_pct_chi2_sweep",
        "schema_version": "1.0.0",
        "framework_constants": {
            "gamma": GAMMA, "alpha_xi": ALPHA_XI,
            "N_gen": N_GEN, "d": D,
        },
        "closure_coefficients": {
            "K_pre": K_PRE, "K_post": K_POST,
            "Q_pre": Q_PRE, "Q_post": Q_POST,
            "Q_a":   Q_A,
            "V_Q_constant_factor_per_unit_v2_2adic": V_Q,
        },
        "v_2_per_regime_via_2_adic_valuation":
            {f"P5N{n}": v2_of_n(n)
             for _, n, _ in LADDER},
        "percentiles_tested": PERCENTILES,
        "summary_per_pct": summary_per_pct,
    }, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("Cross-percentile summary: chi^2 / (2 N_reg)")
    print("=" * 78)
    for pct in PERCENTILES:
        key = f"top_{pct:g}"
        if key not in summary_per_pct:
            continue
        s = summary_per_pct[key]
        print(f"  top-{pct:>4.1f}%  N_reg={s['n_regimes']:>2d}  "
              f"chi^2/N_K={s['chi2_K_per_dof']:.3f}  "
              f"chi^2/N_Q={s['chi2_Q_per_dof']:.3f}  "
              f"joint chi^2/(2N)={s['joint_chi2_per_2N']:.3f}")
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
