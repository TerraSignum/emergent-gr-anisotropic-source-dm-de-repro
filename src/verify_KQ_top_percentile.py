"""K, Q closure restricted to the top-X% of nodes (matter-localized
heavy-tail concentration), where the chirality-flip / branch-
dependent physics is most pronounced.

For each lattice snapshot we identify the top-percentile of nodes
by a matter-localization indicator (here: per-node row-sum of Xi,
which tracks neighbor-connectivity / cluster-core density). The
K and Q sub-matrix restricted to these nodes is then averaged to
give a percentile-restricted regime estimator.

We test percentiles {5%, 10%, 20%, 50% (volume-mean reference)}
and report chi^2/dof of the harmonic closure for each.
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


def theta_chir(n_lat):
    x = np.log(n_lat / N_STAR) / np.log(D_DIM * N_GEN)
    return np.arctan(N_GEN ** (2 * x - 1))


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


def top_percentile_means(K_mat, Q_mat, xi_mat, top_frac):
    """Restrict K, Q to the top fraction of nodes by Xi row-sum.

    Returns (K_top_mean, Q_top_mean): mean of K, Q sub-matrix on the
    selected node block.
    """
    np.fill_diagonal(xi_mat, 0.0)
    deg = xi_mat.sum(axis=1)
    n = len(deg)
    n_top = max(1, int(np.ceil(top_frac * n)))
    top_idx = np.argsort(-deg)[:n_top]
    K_block = K_mat[np.ix_(top_idx, top_idx)]
    Q_block = Q_mat[np.ix_(top_idx, top_idx)]
    return float(K_block.mean()), float(Q_block.mean())


def fit_wls(features, y, sigma):
    X = np.asarray(features, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    w_diag = 1.0 / sigma ** 2
    XtWX = (X * w_diag[:, None]).T @ X
    XtWy = (X * w_diag[:, None]).T @ y
    cov = np.linalg.inv(XtWX)
    beta = cov @ XtWy
    pred = X @ beta
    chi2 = float(np.sum(((y - pred) / sigma) ** 2))
    return beta, np.sqrt(np.diag(cov)), chi2


def fourier_design(theta, n_max):
    cols = [np.ones_like(theta)]
    for n in range(1, n_max + 1):
        cols.append(np.cos(2 * n * theta))
        cols.append(np.sin(2 * n * theta))
    return np.column_stack(cols)


def main():
    print("=" * 78)
    print("K, Q top-X% closure: harmonic fit on percentile-restricted regime values")
    print("=" * 78)

    fractions = [0.05, 0.10, 0.20, 0.50, 1.0]

    by_fraction = {}
    for frac in fractions:
        rows = []
        for reg, fpath, N in regime_files():
            if not fpath.exists():
                continue
            d = np.load(fpath, allow_pickle=True)
            edge_snap = d["edge_xi_snapshots"]
            n_seeds = int(edge_snap.shape[0])
            K_keys = sorted([k for k in d.keys() if "ff_K" in k])
            Q_keys = sorted([k for k in d.keys() if "ff_Q" in k])
            K_vals = []; Q_vals = []
            for s, (kk, qk) in enumerate(zip(K_keys, Q_keys)):
                xi = edge_snap[s, -1].astype(float).copy()
                np.fill_diagonal(xi, 1.0)
                K_mat = np.asarray(d[kk], dtype=float)
                Q_mat = np.asarray(d[qk], dtype=float)
                if K_mat.ndim != 2:
                    continue
                Kv, Qv = top_percentile_means(K_mat, Q_mat, xi.copy(), frac)
                K_vals.append(Kv); Q_vals.append(Qv)
            K_vals = np.array(K_vals); Q_vals = np.array(Q_vals)
            n_s = len(K_vals)
            rows.append({
                "regime": reg, "N": N, "n_seeds": n_s,
                "K_mean": float(K_vals.mean()),
                "K_sem": float(K_vals.std()/np.sqrt(n_s)),
                "Q_mean": float(Q_vals.mean()),
                "Q_sem": float(Q_vals.std()/np.sqrt(n_s)),
                "theta_chir": float(theta_chir(N)),
            })
        # Fit harmonic closure
        if not rows:
            continue
        theta = np.array([r["theta_chir"] for r in rows])
        K = np.array([r["K_mean"] for r in rows])
        K_sem = np.array([r["K_sem"] for r in rows])
        Q = np.array([r["Q_mean"] for r in rows])
        Q_sem = np.array([r["Q_sem"] for r in rows])
        n_data = len(rows)

        results = {}
        for n_max in [1, 2, 3]:
            X = fourier_design(theta, n_max)
            k = X.shape[1]
            dof = n_data - k
            if dof <= 0:
                results[n_max] = None
                continue
            bK, sK, c2K = fit_wls(X, K, K_sem)
            bQ, sQ, c2Q = fit_wls(X, Q, Q_sem)
            results[n_max] = {"k": int(k), "dof": int(dof),
                                "K_chi2_per_dof": c2K/dof,
                                "Q_chi2_per_dof": c2Q/dof,
                                "K_beta": bK.tolist(),
                                "Q_beta": bQ.tolist()}

        by_fraction[frac] = {"rows": rows, "fits": results}
        print(f"\n--- top {int(frac*100)}% of nodes ---")
        print(f"{'reg':<8} {'N':>4} {'theta':>8} {'K_top':>10} {'Q_top':>10}")
        for r in rows:
            print(f"  {r['regime']:<6} {r['N']:>4} {np.degrees(r['theta_chir']):>7.2f}° "
                  f"{r['K_mean']:>10.5f} {r['Q_mean']:>10.5f}")
        print(f"  Fourier nested fits:")
        for nm in [1, 2, 3]:
            f = results[nm]
            if f is None: continue
            print(f"    n_max={nm} (k={f['k']}, dof={f['dof']}): "
                  f"K chi²/dof={f['K_chi2_per_dof']:.3f}, "
                  f"Q chi²/dof={f['Q_chi2_per_dof']:.3f}")

    # Summary table
    print()
    print("=" * 78)
    print(f"SUMMARY: harmonic-closure chi²/dof by node-percentile restriction")
    print("=" * 78)
    print(f"{'Top%':>6} {'n_max=1':>16} {'n_max=2':>16} {'n_max=3':>16}")
    print(f"{'':>6} {'K  /  Q':>16} {'K  /  Q':>16} {'K  /  Q':>16}")
    for frac in fractions:
        if frac not in by_fraction: continue
        line = f"{int(frac*100):>5}%"
        for nm in [1, 2, 3]:
            f = by_fraction[frac]["fits"].get(nm)
            if f is None:
                line += f" {'NA':>16}"
            else:
                line += f"  {f['K_chi2_per_dof']:>5.2f} / {f['Q_chi2_per_dof']:>5.2f}".rjust(16)
        print(line)

    bundle = {
        "method": ("K, Q closure on top-X% of nodes by Xi row-sum (matter-"
                   "localized heavy tail). Per-fraction harmonic Fourier "
                   "fits at n_max=1..3 on 8 canonical-physics regimes."),
        "fractions": fractions,
        "by_fraction": {str(f): by_fraction[f] for f in by_fraction},
    }
    out_path = OUT / "verify_KQ_top_percentile.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nBundle: {out_path}")


if __name__ == "__main__":
    main()
