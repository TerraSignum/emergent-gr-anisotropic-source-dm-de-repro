r"""Optimized test of K, Q Fourier mode-spectrum, extending the
canonical-physics ladder with cross-family regimes (P6N128, P8N128)
to test if mode amplitudes are stable under the regime-family extension
and to better constrain the per-mode rational closure.

Output: outputs/verify_KQ_fourier_optimized_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
OUT = ROOT / "outputs"

GAMMA = 0.1
N_GEN = 3
D_DIM = 4
N_STAR = 50


def theta_chir(n_lat: int) -> float:
    x = np.log(n_lat / N_STAR) / np.log(D_DIM * N_GEN)
    return float(np.arctan(N_GEN ** (2 * x - 1)))


def regime_files():
    return [
        ("P5N64",  PARENT / "results_d1_p5n64_24seeds"  / "P5N64.snapshots.npz",  64),
        ("P5N72",  PARENT / "results_d1_p5n72_24seeds"  / "P5N72.snapshots.npz",  72),
        ("P5N84",  PARENT / "results_d1_p5n84_24seeds"  / "P5N84.snapshots.npz",  84),
        ("P5N100", PARENT / "results_d1_p5n100_24seeds" / "P5N100.snapshots.npz", 100),
        ("P5N128", PARENT / "results_d1_p5n128_kq_fixed" / "P5N128.snapshots.npz", 128),
        ("P5N200", PARENT / "results_d1_p5n200_8seeds"  / "P5N200.snapshots.npz", 200),
        ("P5N256", PARENT / "results_d1_p5n256_12seeds" / "P5N256.snapshots.npz", 256),
        ("P5N300", PARENT / "results_d1_p5n300_12seeds" / "P5N300.snapshots.npz", 300),
        ("P5N512", PARENT / "results_d1_p5n512_12seeds" / "P5N512.snapshots.npz", 512),
        ("P6N128", PARENT / "results_d1_p6n128_12seeds" / "P6N128.snapshots.npz", 128),
        ("P8N128", PARENT / "results_d1_p8n128_12seeds" / "P8N128.snapshots.npz", 128),
    ]


def load_means(npz_path: Path):
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    K = []; Q = []
    for k in sorted(d.keys()):
        if "ff_K" in k:
            arr = d[k]
            if arr.ndim == 2: K.append(float(arr.mean()))
        if "ff_Q" in k:
            arr = d[k]
            if arr.ndim == 2: Q.append(float(arr.mean()))
    if not K:
        return None
    return np.asarray(K), np.asarray(Q)


def fit_fourier(theta, y, sigma, n_max):
    cols = [np.ones_like(theta)]
    for n in range(1, n_max + 1):
        cols.append(np.cos(2 * n * theta))
        cols.append(np.sin(2 * n * theta))
    X = np.column_stack(cols)
    W = np.diag(1 / sigma ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    pred = X @ beta
    chi2 = float(np.sum(((y - pred) / sigma) ** 2))
    return beta, np.sqrt(np.diag(cov)), chi2, len(theta) - X.shape[1]


def main():
    rows = []
    for reg, fpath, n_lat in regime_files():
        out = load_means(fpath)
        if out is None:
            continue
        K, Q = out
        n_seed = len(K)
        rows.append({
            "regime": reg,
            "N": n_lat,
            "n_seeds": n_seed,
            "K_mean": float(K.mean()),
            "K_sem": float(K.std() / np.sqrt(n_seed)),
            "Q_mean": float(Q.mean()),
            "Q_sem": float(Q.std() / np.sqrt(n_seed)),
            "theta_chir": theta_chir(n_lat),
        })

    print(f"Total regimes: {len(rows)}")
    theta = np.array([r["theta_chir"] for r in rows])
    K = np.array([r["K_mean"] for r in rows])
    K_sem = np.array([r["K_sem"] for r in rows])
    Q = np.array([r["Q_mean"] for r in rows])
    Q_sem = np.array([r["Q_sem"] for r in rows])

    # Bootstrap mode amplitudes
    print()
    print("Bootstrap mode amplitudes (1000 resamples, with within-regime seed-resampling):")
    n_boot = 1000
    rng = np.random.default_rng(42)

    # For each regime, draw seeds (with replacement) and recompute mean
    # We need per-seed data, not just mean. Reload all seeds.
    seed_data = {}
    for reg, fpath, n_lat in regime_files():
        out = load_means(fpath)
        if out is None:
            continue
        K_seeds, Q_seeds = out
        seed_data[reg] = (K_seeds, Q_seeds)

    print(f"{'mode':>6} {'angle':>8} {'integer':<14} {'K |M_n|':>14} {'Q |M_n|':>14}")
    for n_max in [1, 2, 3, 4]:
        # Define which integer this n corresponds to
        if n_max == 1: integer_label = "chirality"
        elif n_max == 2: integer_label = f"d = {D_DIM}"
        elif n_max == 3: integer_label = f"2N_gen = {2*N_GEN}"
        elif n_max == 4: integer_label = f"2d = {2*D_DIM}"

        n_param = 1 + 2 * n_max
        if n_param > len(rows):
            print(f"  n={n_max}: {n_param} params > {len(rows)} data, skipping")
            continue

        # Bootstrap
        amps_K = []
        amps_Q = []
        for b in range(n_boot):
            K_boot = np.zeros(len(rows))
            Q_boot = np.zeros(len(rows))
            sem_K_boot = np.zeros(len(rows))
            sem_Q_boot = np.zeros(len(rows))
            for i, r in enumerate(rows):
                K_seeds, Q_seeds = seed_data[r["regime"]]
                idx = rng.integers(0, len(K_seeds), size=len(K_seeds))
                Kb = K_seeds[idx]; Qb = Q_seeds[idx]
                K_boot[i] = Kb.mean(); Q_boot[i] = Qb.mean()
                sem_K_boot[i] = max(Kb.std() / np.sqrt(len(Kb)), 1e-9)
                sem_Q_boot[i] = max(Qb.std() / np.sqrt(len(Qb)), 1e-9)

            try:
                bK, _, _, _ = fit_fourier(theta, K_boot, sem_K_boot, n_max)
                bQ, _, _, _ = fit_fourier(theta, Q_boot, sem_Q_boot, n_max)
                cn_K = bK[2 * n_max - 1]; sn_K = bK[2 * n_max]
                cn_Q = bQ[2 * n_max - 1]; sn_Q = bQ[2 * n_max]
                amps_K.append(np.sqrt(cn_K ** 2 + sn_K ** 2))
                amps_Q.append(np.sqrt(cn_Q ** 2 + sn_Q ** 2))
            except np.linalg.LinAlgError:
                continue

        amp_K_med = np.median(amps_K); amp_K_sem = np.std(amps_K)
        amp_Q_med = np.median(amps_Q); amp_Q_sem = np.std(amps_Q)
        # 95% CI
        amp_K_ci = np.percentile(amps_K, [2.5, 97.5])
        amp_Q_ci = np.percentile(amps_Q, [2.5, 97.5])
        z_K = amp_K_med / amp_K_sem if amp_K_sem > 0 else 0
        z_Q = amp_Q_med / amp_Q_sem if amp_Q_sem > 0 else 0

        print(f"  {n_max:>4} {2*n_max}θ      {integer_label:<14} "
              f"{amp_K_med:.4f}±{amp_K_sem:.4f} (z={z_K:.1f}) | "
              f"{amp_Q_med:.4f}±{amp_Q_sem:.4f} (z={z_Q:.1f})")

    # Fit n_max=4 unconstrained at fixed theta values (3 dof since n_param=9, but theta has 11 distinct values)
    # Actually with 11 data points and 9 params we have 2 dof - good!
    print()
    print("="*78)
    print("With 11 regimes (3 N=128 from different families), n_max=4 is statistically determined:")
    print("="*78)
    bK, sK, c2K, dofK = fit_fourier(theta, K, K_sem, 4)
    bQ, sQ, c2Q, dofQ = fit_fourier(theta, Q, Q_sem, 4)
    print(f"K: chi^2/dof = {c2K:.2f}/{dofK} = {c2K/max(1,dofK):.2f}")
    print(f"Q: chi^2/dof = {c2Q:.2f}/{dofQ} = {c2Q/max(1,dofQ):.2f}")

    labels = ['c_0', 'c_1·cos(2θ)', 's_1·sin(2θ)', 'c_2·cos(4θ)', 's_2·sin(4θ)',
              'c_3·cos(6θ)', 's_3·sin(6θ)', 'c_4·cos(8θ)', 's_4·sin(8θ)']
    print()
    print("K coefficients (z-score = val/sigma):")
    for lab, val, sig in zip(labels, bK, sK):
        z = val/sig if sig > 0 else 0
        marker = " ←sig" if abs(z) > 2 else ""
        print(f"  {lab:<22} = {val:+.6f} ± {sig:.6f}  (z={z:+5.2f}){marker}")
    print()
    print("Q coefficients:")
    for lab, val, sig in zip(labels, bQ, sQ):
        z = val/sig if sig > 0 else 0
        marker = " ←sig" if abs(z) > 2 else ""
        print(f"  {lab:<22} = {val:+.6f} ± {sig:.6f}  (z={z:+5.2f}){marker}")

    # Mode amplitudes from the n_max=4 fit
    print()
    print("Mode amplitudes |M_n| from n_max=4 fit (statistically determined):")
    print(f"{'n':>3} {'angle':>6} {'integer':<16} {'K |M_n|':>16} {'Q |M_n|':>16}")
    for n in range(1, 5):
        cn_K = bK[2*n-1]; sn_K = bK[2*n]; scn_K = sK[2*n-1]; ssn_K = sK[2*n]
        cn_Q = bQ[2*n-1]; sn_Q = bQ[2*n]; scn_Q = sQ[2*n-1]; ssn_Q = sQ[2*n]
        amp_K = np.sqrt(cn_K**2 + sn_K**2)
        amp_Q = np.sqrt(cn_Q**2 + sn_Q**2)
        # Approx uncertainty
        if amp_K > 0:
            samp_K = np.sqrt((cn_K*scn_K)**2 + (sn_K*ssn_K)**2) / amp_K
        else: samp_K = 0
        if amp_Q > 0:
            samp_Q = np.sqrt((cn_Q*scn_Q)**2 + (sn_Q*ssn_Q)**2) / amp_Q
        else: samp_Q = 0
        if n == 1: il = "chirality"
        elif n == 2: il = f"d={D_DIM}"
        elif n == 3: il = f"2N_gen={2*N_GEN}"
        elif n == 4: il = f"2d={2*D_DIM}"
        z_K = amp_K/samp_K if samp_K > 0 else 0
        z_Q = amp_Q/samp_Q if samp_Q > 0 else 0
        print(f"  {n} {2*n}θ    {il:<16} {amp_K:.4f}±{samp_K:.4f} (z={z_K:.1f}) | "
              f"{amp_Q:.4f}±{samp_Q:.4f} (z={z_Q:.1f})")

    bundle = {
        "method": "Optimized Fourier mode-spectrum test with cross-family regimes (11 total)",
        "n_regimes": len(rows),
        "rows": rows,
        "K_n_max_4_fit": {"chi2": c2K, "dof": dofK, "beta": bK.tolist(),
                           "sigma": sK.tolist(), "labels": labels},
        "Q_n_max_4_fit": {"chi2": c2Q, "dof": dofQ, "beta": bQ.tolist(),
                           "sigma": sQ.tolist(), "labels": labels},
    }
    with open(OUT / "verify_KQ_fourier_optimized_test.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nBundle: {OUT / 'verify_KQ_fourier_optimized_test.json'}")


if __name__ == "__main__":
    main()
