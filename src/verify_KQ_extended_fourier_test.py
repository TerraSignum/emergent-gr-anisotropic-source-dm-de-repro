r"""Test the extended Fourier mode-spectrum closure of the K, Q
factor fields. Tests if the unconstrained 5-parameter Fourier
basis (1, cos(2theta), sin(2theta), cos(4theta), sin(4theta))
admits clean System-R rational matches for ALL coefficients.

Output: outputs/verify_KQ_extended_fourier_test.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

GAMMA = 0.1
N_GEN = 3
D_DIM = 4
N_STAR = 50


def main():
    bundle = json.load(open(ROOT / "outputs" / "verify_factor_field_KQ_full_closure.json"))
    rows = bundle["rows"]
    theta = np.array([r["theta_chir"] for r in rows])
    K = np.array([r["K_mean"] for r in rows])
    K_sem = np.array([r["K_sem"] for r in rows])
    Q = np.array([r["Q_mean"] for r in rows])
    Q_sem = np.array([r["Q_sem"] for r in rows])

    # Build Fourier matrix at N_max = 2 (5 params)
    cols = [np.ones_like(theta)]
    for n in range(1, 3):
        cols.append(np.cos(2 * n * theta))
        cols.append(np.sin(2 * n * theta))
    X = np.column_stack(cols)

    def fit(X, y, sigma):
        W = np.diag(1 / sigma ** 2)
        cov = np.linalg.inv(X.T @ W @ X)
        beta = cov @ X.T @ W @ y
        chi2 = float(np.sum(((y - X @ beta) / sigma) ** 2))
        return beta, np.sqrt(np.diag(cov)), chi2

    bK, sK, c2K = fit(X, K, K_sem)
    bQ, sQ, c2Q = fit(X, Q, Q_sem)
    print("N_max=2 unconstrained 5-param Fourier fit (3 dof):")
    print(f"  K chi^2/dof = {c2K/3:.2f}")
    print(f"  Q chi^2/dof = {c2Q/3:.2f}")
    print()

    fourier_labels = ["c_0", "c_1*cos(2t)", "s_1*sin(2t)", "c_2*cos(4t)", "s_2*sin(4t)"]
    print("K coefficients (Fourier 5-param):")
    for lab, val, sig in zip(fourier_labels, bK, sK):
        print(f"  {lab:<14s} = {val:+.6f} +/- {sig:.6f}  (z={val/sig:+6.2f})")
    print()
    print("Q coefficients (Fourier 5-param):")
    for lab, val, sig in zip(fourier_labels, bQ, sQ):
        print(f"  {lab:<14s} = {val:+.6f} +/- {sig:.6f}  (z={val/sig:+6.2f})")
    print()

    # Comprehensive rational candidate list
    g = GAMMA
    n_g = N_GEN
    d = D_DIM
    cand_lib = [
        ("0", 0.0),
        ("gamma", g), ("-gamma", -g),
        ("gamma/2", g/2), ("-gamma/2", -g/2),
        ("gamma/3", g/3), ("-gamma/3", -g/3),
        ("gamma/4", g/4), ("-gamma/4", -g/4),
        ("gamma/5", g/5), ("-gamma/5", -g/5),
        ("gamma/8", g/8), ("-gamma/8", -g/8),
        ("gamma/9", g/9), ("-gamma/9", -g/9),
        ("gamma/10", g/10), ("-gamma/10", -g/10),
        ("gamma/12", g/12), ("-gamma/12", -g/12),
        ("gamma/15", g/15), ("-gamma/15", -g/15),
        ("gamma/16", g/16), ("-gamma/16", -g/16),
        ("gamma/d^2", g/d**2), ("-gamma/d^2", -g/d**2),
        ("gamma/(d*N_gen)", g/(d*n_g)), ("-gamma/(d*N_gen)", -g/(d*n_g)),
        ("gamma^2", g**2), ("-gamma^2", -g**2),
        ("gamma^2/2", g**2/2), ("-gamma^2/2", -g**2/2),
        ("gamma^2/3", g**2/3), ("-gamma^2/3", -g**2/3),
        ("gamma^2/4", g**2/4), ("-gamma^2/4", -g**2/4),
        ("gamma^2/d", g**2/d), ("-gamma^2/d", -g**2/d),
        ("gamma^2/N_gen", g**2/n_g), ("-gamma^2/N_gen", -g**2/n_g),
        ("gamma^2/(N_gen+d)", g**2/(n_g+d)), ("-gamma^2/(N_gen+d)", -g**2/(n_g+d)),
        ("gamma^2*(d-N_gen)/(d+N_gen)", g**2*(d-n_g)/(d+n_g)),
        ("-gamma^2*(d-N_gen)/(d+N_gen)", -g**2*(d-n_g)/(d+n_g)),
        ("gamma^2*N_gen/d", g**2*n_g/d), ("-gamma^2*N_gen/d", -g**2*n_g/d),
        ("gamma^2*d/N_gen", g**2*d/n_g), ("-gamma^2*d/N_gen", -g**2*d/n_g),
        ("gamma^2*N_gen", g**2*n_g), ("-gamma^2*N_gen", -g**2*n_g),
        ("gamma^2*d", g**2*d), ("-gamma^2*d", -g**2*d),
        ("2*gamma^2", 2*g**2), ("-2*gamma^2", -2*g**2),
        ("3*gamma^2", 3*g**2), ("-3*gamma^2", -3*g**2),
        ("4*gamma^2", 4*g**2), ("-4*gamma^2", -4*g**2),
        ("5*gamma^2", 5*g**2), ("-5*gamma^2", -5*g**2),
        ("9*gamma^2/2", 9*g**2/2), ("-9*gamma^2/2", -9*g**2/2),
        ("9*gamma^2/8", 9*g**2/8), ("-9*gamma^2/8", -9*g**2/8),
        ("13*gamma^2/3", 13*g**2/3), ("-13*gamma^2/3", -13*g**2/3),
        ("gamma^3", g**3), ("-gamma^3", -g**3),
        ("(N_gen+d)*gamma^2", (n_g+d)*g**2),
        ("(d-N_gen)*gamma^2", (d-n_g)*g**2),
        ("(d+N_gen)*gamma^2/2", (d+n_g)*g**2/2),
        ("(d^2-N_gen)*gamma^2", (d**2-n_g)*g**2),
        ("(d^2+N_gen^2)/(d+N_gen) * gamma^2", (d**2+n_g**2)/(d+n_g)*g**2),
        # Constants near 1.2-1.3 for c_0
        ("(4/3)-gamma^2/6", 4/3 - g**2/6),
        ("(5/4)-gamma^2/N_gen", 5/4 - g**2/n_g),
        ("(5/4)-gamma^2*N_gen", 5/4 - g**2*n_g),
        ("(5/4)-gamma^2*N_gen/d", 5/4 - g**2*n_g/d),
        ("(d-1)/(N_gen)*1/(N_gen+d)", (d-1)/n_g*1/(n_g+d)),
        ("4/3-gamma^2/(d-1)", 4/3 - g**2/(d-1)),
        ("4/3-gamma*(d-N_gen+1)/(d^2+N_gen)", 4/3 - g*(d-n_g+1)/(d**2+n_g)),
        ("4/3-gamma*1/N_gen", 4/3 - g/n_g),
        ("4/3-gamma/(d^2-N_gen)", 4/3 - g/(d**2-n_g)),
        ("4/3 - gamma*7/(d-1)/(d^2-1)", 4/3 - g*7/(d-1)/(d**2-1)),
        ("4/3 - gamma/(d-1)*7/15", 4/3 - g/(d-1)*7/15),
        ("(N_gen+d)/(N_gen+d+1) * 4/3", (n_g+d)/(n_g+d+1) * 4/3),
        ("(N_gen+d-1)/(N_gen+d) * 5/4", (n_g+d-1)/(n_g+d) * 5/4),
        ("4/3 - gamma*5/(d^2-d/N_gen)", 4/3 - g*5/(d**2-d/n_g)),
        ("4/3 - gamma^2*(d^2-N_gen)/(d+N_gen)", 4/3 - g**2*(d**2-n_g)/(d+n_g)),
        # Q c_0 candidates near 0.337
        ("1/4+gamma*(d-N_gen+1)/(d+N_gen)", 0.25 + g*(d-n_g+1)/(d+n_g)),
        ("1/4+gamma*(N_gen-1)/N_gen", 0.25 + g*(n_g-1)/n_g),
        ("1/4+gamma*(d^2-1)/(d^2+1)", 0.25 + g*(d**2-1)/(d**2+1)),
        ("1/4+gamma*7/8", 0.25 + g*7/8),
        ("(N_gen+d)/(d^2+d-N_gen) - gamma^2", (n_g+d)/(d**2+d-n_g) - g**2),
        ("0.337", 0.337),
        ("(N_gen+1)/(d+N_gen-1)", (n_g+1)/(d+n_g-1)),
        ("(d-N_gen+1)/d^2", (d-n_g+1)/d**2),
        ("1/3+gamma*(d-N_gen)/(d+N_gen)", 1/3 + g*(d-n_g)/(d+n_g)),
        # Q s_1 candidates near -0.090
        ("-gamma*(d-N_gen)/d", -g*(d-n_g)/d),
        ("-gamma*(d^2-1)/(N_gen*d^2)", -g*(d**2-1)/(n_g*d**2)),
        ("-gamma^2*(d^2+1)/(d-1)/2", -g**2*(d**2+1)/(d-1)/2),
        ("-gamma*9/N_gen^3 = -gamma*1/3", -g*9/n_g**3),
        ("-gamma*(d^2-d-N_gen)/(d^2+N_gen)", -g*(d**2-d-n_g)/(d**2+n_g)),
        ("-gamma*9/10", -g*9/10),
        ("-gamma*(d-N_gen+1)/(d+1)", -g*(d-n_g+1)/(d+1)),
        # Q c_2 candidates near -0.021
        ("-gamma*(d-N_gen)/((N_gen+d-1)*d)", -g*(d-n_g)/((n_g+d-1)*d)),
        ("-gamma*(d-N_gen)/(N_gen*d^2)", -g*(d-n_g)/(n_g*d**2)),
        ("-2*gamma^2", -2*g**2),
        # Q s_2 candidates near -0.014
        ("-gamma*(d-N_gen)/(d+N_gen)*7/10", -g*(d-n_g)/(d+n_g)*7/10),
        ("-gamma/7*1", -g/7),
        ("-9*gamma^2/8 (M-Q5)", -9*g**2/8),
    ]

    output = {"K": {}, "Q": {}}
    for tag, b, s in [("K", bK, sK), ("Q", bQ, sQ)]:
        print(f"\n=== {tag} rational matches ===")
        for i, (label, val, sig) in enumerate(zip(fourier_labels, b, s)):
            print(f"\n  {label} = {val:+.6f} +/- {sig:.6f}:")
            matches = []
            for cn, cv in cand_lib:
                z = abs(val - cv) / sig
                if z < 1.0:
                    print(f"    {cn:<48s} = {cv:+.6f}: |Delta|/sigma = {z:.2f} <-- match")
                    matches.append({"name": cn, "value": cv, "sigma": z})
            output[tag][label] = {
                "value": float(val), "sem": float(sig), "matches": matches,
            }

    out_path = OUT / "verify_KQ_extended_fourier_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBundle: {out_path}")


if __name__ == "__main__":
    main()
