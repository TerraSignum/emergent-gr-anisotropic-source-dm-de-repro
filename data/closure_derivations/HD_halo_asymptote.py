"""Closure-derivation H-D: halo correlation asymptote rho(X, d) -> ?

The halo audit reports per-regime Spearman rank correlation
of the geometry-side amplitude X = G_00 + Lambda_t and the
nearest-defect distance d(a, dC_N), with empirical values
clustering around |rho| ~ 0.4. The user's standing question:
does the asymptote land on -alpha_xi^2 / 2 = -81/200 = -0.405,
or on a related System-R rational?

Hypotheses tested (no fitting):
  A. rho_inf = -alpha_xi^2 / 2 = -81/200 = -0.405
  B. rho_inf = -alpha_xi / 2 = -9/20 = -0.450
  C. rho_inf = -alpha_xi^2 = -81/100 = -0.810
  D. rho_inf = -2/5 = -0.4 (= -gamma N_gen + ...)
  E. rho_inf = -gamma alpha_xi = -9/100 = -0.090

Symanzik 1/N + 1/N^2 fit on the canonical 10-regime ladder
(P5/P5N N-ordered) gives a parameter-free asymptote estimate
plus 95% bootstrap CI; we then test which rational is in CI
and which has lowest |z|.

Reads outputs/verify_halo_geometry_vs_source_decomposition.json
Writes peer_reviews/HD_halo_asymptote.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SRC = (ROOT / "emergent-gr-anisotropic-source-dm-de-repro" /
       "outputs" / "verify_halo_geometry_vs_source_decomposition.json")
OUT = REPO / "data" / "closure_derivations" / "HD_halo_asymptote.json"

GAMMA = 1.0 / 10.0
ALPHA_XI = 9.0 / 10.0
N_GEN = 3

CANONICAL_REGIMES = {"P5", "P5N64", "P5N72", "P5N84", "P5N100",
                     "P5N128", "P5N200", "P5N256", "P5N300", "P5N512"}


def fit_symanzik(Ns, ys, n_boot=2000, seed=0):
    """Symanzik 2-term fit y = y_inf + a/N + b/N^2; bootstrap CI."""
    rng = np.random.default_rng(seed)
    Ns = np.asarray(Ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    X = np.column_stack([np.ones_like(Ns), 1.0/Ns, 1.0/(Ns**2)])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    y_inf = float(coef[0])

    boot = []
    n = len(Ns)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xs = X[idx]
        ys_s = ys[idx]
        try:
            c, *_ = np.linalg.lstsq(Xs, ys_s, rcond=None)
            boot.append(c[0])
        except np.linalg.LinAlgError:
            continue
    boot = np.array(boot)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return y_inf, lo, hi, boot


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    canonical = [r for r in d["per_regime"]
                 if r["regime"] in CANONICAL_REGIMES]
    canonical.sort(key=lambda r: r["N"])
    Ns = [r["N"] for r in canonical]
    rho_X = [r["rho_X_vs_d"] for r in canonical]
    rho_T = [r["rho_T_vs_d"] for r in canonical]
    print(f"=== Canonical 10-regime ladder ===")
    print(f"  Ns = {Ns}")
    print(f"  rho(X, d) = {[f'{x:.4f}' for x in rho_X]}")
    print(f"  rho(T, d) = {[f'{x:.4f}' for x in rho_T]}")
    print()

    # Symanzik 2-term fit
    y_inf_X, X_lo, X_hi, _ = fit_symanzik(Ns, rho_X)
    y_inf_T, T_lo, T_hi, _ = fit_symanzik(Ns, rho_T)
    print(f"=== Symanzik 2-term fit (y_inf + a/N + b/N^2) ===")
    print(f"  rho(X, d) y_inf = {y_inf_X:+.4f}, "
          f"95% CI = [{X_lo:+.4f}, {X_hi:+.4f}]")
    print(f"  rho(T, d) y_inf = {y_inf_T:+.4f}, "
          f"95% CI = [{T_lo:+.4f}, {T_hi:+.4f}]")
    print()

    # Pure 1/N fit (linear), for comparison
    Xlin = np.column_stack([np.ones(len(Ns)), 1.0/np.array(Ns, dtype=float)])
    cX, *_ = np.linalg.lstsq(Xlin, np.array(rho_X), rcond=None)
    cT, *_ = np.linalg.lstsq(Xlin, np.array(rho_T), rcond=None)
    print(f"=== 1/N fit (single-term) ===")
    print(f"  rho(X, d) y_inf = {cX[0]:+.4f}")
    print(f"  rho(T, d) y_inf = {cT[0]:+.4f}")
    print()

    candidates = [
        ("-alpha_xi^2/2 = -81/200", -ALPHA_XI**2 / 2),
        ("-alpha_xi/2 = -9/20", -ALPHA_XI / 2),
        ("-alpha_xi^2 = -81/100", -ALPHA_XI**2),
        ("-2/5", -0.4),
        ("-gamma alpha_xi = -9/100", -GAMMA * ALPHA_XI),
        ("-3/8", -0.375),
        ("-1/2", -0.5),
        ("-gamma N_gen = -3/10", -GAMMA * N_GEN),
        ("-gamma alpha_xi^2 = -81/1000", -GAMMA * ALPHA_XI**2),
        ("-1/3", -1.0/3.0),
        ("-gamma^2 N_gen = -3/100", -GAMMA**2 * N_GEN),
    ]

    print(f"=== Candidates vs rho(X, d) y_inf = {y_inf_X:+.4f} ===")
    cand_X = []
    for label, val in candidates:
        in_ci = bool(X_lo <= val <= X_hi)
        rel = abs(val - y_inf_X) / abs(y_inf_X)
        cand_X.append({"label": label, "value": float(val),
                       "in_CI95": in_ci, "rel_err": float(rel)})
        flag = "IN-CI" if in_ci else "out  "
        print(f"  {label:<28s} = {val:+.4f}  {flag}  rel-err = {100*rel:.2f}%")

    print(f"\n=== Candidates vs rho(T, d) y_inf = {y_inf_T:+.4f} ===")
    cand_T = []
    for label, val in candidates:
        in_ci = bool(T_lo <= val <= T_hi)
        rel = abs(val - y_inf_T) / abs(y_inf_T)
        cand_T.append({"label": label, "value": float(val),
                       "in_CI95": in_ci, "rel_err": float(rel)})
        flag = "IN-CI" if in_ci else "out  "
        print(f"  {label:<28s} = {val:+.4f}  {flag}  rel-err = {100*rel:.2f}%")

    cand_X.sort(key=lambda c: c["rel_err"])
    cand_T.sort(key=lambda c: c["rel_err"])
    print()
    print(f"=== Best matches ===")
    print(f"  rho(X, d): {cand_X[0]['label']} (rel-err {100*cand_X[0]['rel_err']:.2f}%)")
    print(f"  rho(T, d): {cand_T[0]['label']} (rel-err {100*cand_T[0]['rel_err']:.2f}%)")

    bundle = {
        "method": "HD_halo_asymptote",
        "framework_constants": {"gamma": GAMMA, "alpha_xi": ALPHA_XI},
        "canonical_ladder_Ns": Ns,
        "rho_X_vs_d_per_regime": rho_X,
        "rho_T_vs_d_per_regime": rho_T,
        "symanzik_2term_fit": {
            "rho_X_d_yinf": y_inf_X,
            "rho_X_d_CI95": [X_lo, X_hi],
            "rho_T_d_yinf": y_inf_T,
            "rho_T_d_CI95": [T_lo, T_hi],
        },
        "linear_1_over_N_fit": {
            "rho_X_d_yinf": float(cX[0]),
            "rho_T_d_yinf": float(cT[0]),
        },
        "candidates_X": cand_X,
        "candidates_T": cand_T,
        "best_match_X": cand_X[0],
        "best_match_T": cand_T[0],
        "verdict": _verdict(cand_X, cand_T),
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {OUT}")
    print(f"Verdict: {bundle['verdict']}")
    return 0


def _verdict(cand_X, cand_T):
    bx = cand_X[0]
    bt = cand_T[0]
    if (bx["in_CI95"] and bt["in_CI95"]
        and bx["rel_err"] < 0.10 and bt["rel_err"] < 0.10):
        return (f"STRUCTURAL_MATCH: rho(X,d) -> {bx['label']}, "
                f"rho(T,d) -> {bt['label']}, both in 95% CI <10% rel")
    if bx["in_CI95"] or bt["in_CI95"]:
        return (f"PARTIAL: best matches in CI but rel-err >= 10%; "
                f"rho(X,d) {bx['label']} ({100*bx['rel_err']:.1f}%), "
                f"rho(T,d) {bt['label']} ({100*bt['rel_err']:.1f}%)")
    return "NO_CLEAN_RATIONAL: no candidate within 95% CI"


if __name__ == "__main__":
    raise SystemExit(main())
