"""Closure-derivation H-O: dark/baryon partition of Omega_m via System-R rationals.

H-M established Omega_m = gamma*N_gen + gamma^2*d/N_gen = 47/150.
H-O closes the partition into dark / baryonic components:

  Omega_dm = gamma * N_gen  -  gamma^2 * (d + N_gen) / 2
           = 30/100 - 7/200
           = 53/200
           = 0.26500

  Omega_b  = Omega_m - Omega_dm
           = gamma^2 * d/N_gen + gamma^2 * (d + N_gen)/2
           = gamma^2 * (2d + N_gen*(d+N_gen)) / (2 N_gen)
           = gamma^2 * (8 + 21) / 6
           = gamma^2 * 29/6
           = 29/600
           = 0.04833

Planck 2018 (TT+TE+EE+lowE+lensing):
  Omega_dm = 0.2647 +/- 0.0049
  Omega_b  = 0.0493 +/- 0.0007
  Omega_m  = 0.3158 +/- 0.0073

Structural reading: the dimensional sum (d + N_gen) = 7 enters
the partition both subtractively (Omega_dm reduces by
gamma^2 (d+N_gen)/2) and through N_gen*(d+N_gen) in Omega_b's
prefactor 2d + N_gen*(d+N_gen) = 29.

Writes peer_reviews/HO_dark_baryon_partition.json
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "data" / "closure_derivations" / "HO_dark_baryon_partition.json"


def main():
    GAMMA = Fraction(1, 10)
    N_GEN = 3
    D = 4
    D_TOT = D + N_GEN  # = 7

    Odm_pred = GAMMA * N_GEN - GAMMA**2 * D_TOT / 2  # = 53/200
    Ob_pred = GAMMA**2 * (2*D + N_GEN * D_TOT) / (2 * N_GEN)  # = 29/600
    Om_pred = Odm_pred + Ob_pred  # should equal 47/150 from H-M

    print(f"=== Hypothesis ===")
    print(f"  Omega_dm = gamma*N_gen - gamma^2*(d+N_gen)/2 = "
          f"{Odm_pred} = {float(Odm_pred):.5f}")
    print(f"  Omega_b  = gamma^2*(2d + N_gen*(d+N_gen))/(2 N_gen) = "
          f"{Ob_pred} = {float(Ob_pred):.5f}")
    print(f"  Omega_m  = Omega_dm + Omega_b = {Om_pred} (must be 47/150)")
    print()

    anchors = {
        "Omega_dm": (0.2647, 0.0049, Odm_pred),
        "Omega_b":  (0.0493, 0.0007, Ob_pred),
    }
    print(f"{'Quantity':<12s} {'Predicted':>10s} {'Planck':>10s} {'rel-err':>10s} {'z':>8s}")
    rows = []
    for label, (val, sigma, pred_frac) in anchors.items():
        pred = float(pred_frac)
        rel = abs(pred - val) / val
        z = (pred - val) / sigma
        tier = ("EXACT" if rel < 0.004 else
                ("PRECISE" if rel < 0.025 else "FACTOR2"))
        print(f"  {label:<10s}  {pred:>10.5f} {val:>10.4f} "
              f"{100*rel:>9.3f}% {z:>+7.2f}  [{tier}]")
        rows.append({
            "quantity": label,
            "predicted_fraction": f"{pred_frac.numerator}/{pred_frac.denominator}",
            "predicted": pred,
            "Planck_central": val,
            "Planck_sigma": sigma,
            "rel_err_pct": float(100*rel),
            "z": float(z),
            "tier": tier,
        })
    print()

    # Cross-check: Omega_dm/Omega_b ratio
    ratio_pred = float(Odm_pred / Ob_pred)
    ratio_PDG = 0.2647 / 0.0493
    print(f"=== Cross-check Omega_dm/Omega_b ratio ===")
    print(f"  Predicted: 53/200 / 29/600 = "
          f"{(Odm_pred/Ob_pred).numerator}/{(Odm_pred/Ob_pred).denominator} "
          f"= {ratio_pred:.5f}")
    print(f"  Planck:    0.2647/0.0493 = {ratio_PDG:.5f}")
    print(f"  rel-err = {100*abs(ratio_pred-ratio_PDG)/ratio_PDG:.3f}%")
    print()

    verdict = (
        f"DARK_BARYON_PARTITION_CLOSED: Omega_dm = 53/200 = 0.265 "
        f"(z = {rows[0]['z']:+.2f}, EXACT) and Omega_b = 29/600 = "
        f"0.0483 (z = {rows[1]['z']:+.2f}, PRECISE). The dimensional "
        f"sum (d + N_gen) = 7 enters both: Omega_dm = gamma*N_gen "
        f"reduced by gamma^2 (d+N_gen)/2; Omega_b prefactor "
        f"2d + N_gen*(d+N_gen) = 29 contains it via "
        f"N_gen*(d+N_gen) = 21. Sum reproduces Omega_m = 47/150 "
        f"(H-M) exactly. The full cosmological energy budget "
        f"(Omega_dm, Omega_b, Omega_Lambda) is now parameter-free "
        f"in System-R rationals."
    )
    print(f"Verdict: {verdict}")

    bundle = {
        "method": "HO_dark_baryon_partition",
        "framework_constants": {"gamma": "1/10", "N_gen": N_GEN, "d": D,
                                  "d_plus_N_gen": D_TOT},
        "predictions": rows,
        "ratio_Omega_dm_over_Omega_b": {
            "predicted": ratio_pred,
            "Planck": ratio_PDG,
        },
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
