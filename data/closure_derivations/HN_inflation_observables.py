"""Closure-derivation H-N: inflation observables n_s, A_s as System-R rationals.

Planck 2018:
  n_s = 0.9649 +/- 0.0042
  A_s = 2.099 +/- 0.014 x 10^-9 (i.e. ln(10^10 A_s) = 3.044 +/- 0.014)

System-R rational predictions:
  n_s = 1 - gamma^2 * (d + N_gen) / 2
      = 1 - gamma^2 * 7/2
      = 1 - 7/200
      = 193/200
      = 0.96500
  A_s = N_gen * (d + N_gen) * gamma^10
      = 3 * 7 * 10^{-10}
      = 21 * gamma^10
      = 21/10^10
      = 2.10 x 10^-9

Reading:
  - Both depend on the SAME structural quantity (d + N_gen) = 7
    = total dimension of fermionic generations + spacetime
  - n_s deviates from 1 by gamma^2 * (d + N_gen)/2 -- the
    quadratic carrier-defect coupling times the dimensional sum
  - A_s magnitude is N_gen * (d + N_gen) * gamma^10 -- the same
    dimensional sum times generation count, suppressed by the
    10th power of the carrier-defect coupling

Writes peer_reviews/HN_inflation_observables.json
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "data" / "closure_derivations" / "HN_inflation_observables.json"


def main():
    GAMMA = Fraction(1, 10)
    N_GEN = 3
    D = 4

    ns_pred = 1 - GAMMA**2 * (D + N_GEN) / 2  # = 193/200
    As_pred_prefactor = N_GEN * (D + N_GEN)    # = 21
    As_pred = float(As_pred_prefactor) * float(GAMMA)**10  # = 2.1e-9

    print(f"=== Hypothesis ===")
    print(f"  n_s = 1 - gamma^2 * (d + N_gen)/2 = {ns_pred} = {float(ns_pred):.5f}")
    print(f"  A_s = N_gen * (d+N_gen) * gamma^10 = "
          f"{As_pred_prefactor} gamma^10 = {As_pred:.4e}")
    print()

    # Anchors
    ns_planck = (0.9649, 0.0042)
    As_planck = (2.099e-9, 0.014e-9)

    rows = []

    # n_s
    pred = float(ns_pred)
    rel_n = abs(pred - ns_planck[0]) / ns_planck[0]
    z_n = (pred - ns_planck[0]) / ns_planck[1]
    tier_n = ("EXACT" if rel_n < 0.004
              else ("PRECISE" if rel_n < 0.025 else "FACTOR2"))
    print(f"=== n_s comparison ===")
    print(f"  Predicted = {pred:.5f}, Planck = {ns_planck[0]} +/- "
          f"{ns_planck[1]}, rel-err = {100*rel_n:.4f}%, "
          f"z = {z_n:+.2f}  [{tier_n}]")
    rows.append({"quantity": "n_s",
                 "predicted_form": "1 - gamma^2 (d+N_gen)/2",
                 "predicted_fraction": "193/200",
                 "predicted": pred,
                 "Planck_central": ns_planck[0],
                 "Planck_sigma": ns_planck[1],
                 "rel_err_pct": float(100*rel_n),
                 "z": float(z_n),
                 "tier": tier_n})

    # A_s
    rel_a = abs(As_pred - As_planck[0]) / As_planck[0]
    z_a = (As_pred - As_planck[0]) / As_planck[1]
    tier_a = ("EXACT" if rel_a < 0.004
              else ("PRECISE" if rel_a < 0.025 else "FACTOR2"))
    print()
    print(f"=== A_s comparison ===")
    print(f"  Predicted = {As_pred:.4e}, Planck = {As_planck[0]:.3e} "
          f"+/- {As_planck[1]:.3e}, rel-err = {100*rel_a:.4f}%, "
          f"z = {z_a:+.2f}  [{tier_a}]")
    rows.append({"quantity": "A_s",
                 "predicted_form": "N_gen (d+N_gen) gamma^10",
                 "predicted_fraction": "21/10^10",
                 "predicted": As_pred,
                 "Planck_central": As_planck[0],
                 "Planck_sigma": As_planck[1],
                 "rel_err_pct": float(100*rel_a),
                 "z": float(z_a),
                 "tier": tier_a})
    print()

    verdict = (
        "INFLATION_FULL_CLOSURE: both Planck inflation observables "
        f"n_s = 193/200 (z = {z_n:+.2f}) and "
        f"A_s = 21*gamma^10 (z = {z_a:+.2f}) match within Planck "
        "measurement precision. Both contain the dimensional sum "
        "(d + N_gen) = 7 = (4D spacetime + 3 fermion generations); "
        "n_s = 1 - gamma^2 (d+N_gen)/2 reads as the deviation from "
        "scale-invariance set by carrier-defect coupling squared "
        "times half the dimensional sum, while "
        "A_s = N_gen (d+N_gen) gamma^10 = (3 generations)(7) (1/10)^10 "
        "is the inflationary scalar power amplitude scaled by the "
        "10th power of the carrier-defect coupling."
    )
    print(f"Verdict: {verdict}")

    bundle = {
        "method": "HN_inflation_observables",
        "framework_constants": {
            "gamma": "1/10", "N_gen": N_GEN, "d": D,
            "d_plus_N_gen": D + N_GEN,
        },
        "predictions": rows,
        "common_structure": "Both n_s and A_s contain "
                             "(d + N_gen) = 7 as a structural factor",
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
