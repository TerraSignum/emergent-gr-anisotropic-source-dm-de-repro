"""Closure-derivation H-L: cosmological clustering amplitude
sigma_8 = alpha_xi^2 = 81/100 = 0.81.

Planck 2018 TT+TE+EE+lowE+lensing: sigma_8 = 0.8111 +/- 0.0060
DES Y3 weak lensing: S_8 = sigma_8 sqrt(Omega_m/0.3) = 0.776 +/- 0.017
KiDS-1000: S_8 = 0.766 +/- 0.020 (the S_8 tension)

Hypothesis: sigma_8 is structurally identical to Lambda_t (the
back-channel projection asymptote) = alpha_xi^2 = 81/100.
This would be a deep cosmological identification:
  - Lambda_t = alpha_xi^2 (back-channel matter asymptote, lattice)
  - sigma_8 = alpha_xi^2 (matter clustering amplitude, cosmology)
both being the same chirality-projected mass-energy amplitude
expressed at different scales.

Writes peer_reviews/HL_sigma8.json
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "data" / "closure_derivations" / "HL_sigma8.json"


def main():
    GAMMA = Fraction(1, 10)
    ALPHA_XI = Fraction(9, 10)

    pred = ALPHA_XI ** 2  # 81/100
    pred_f = float(pred)

    anchors = {
        "Planck_2018_TTTEEE_lowE_lensing": (0.8111, 0.0060),
        "Planck_2018_TT_lowP_lensing": (0.811, 0.0080),
        "DES_Y3_weak_lensing_S8": (0.776, 0.017),  # tension
        "KiDS_1000": (0.766, 0.020),  # tension
        "Planck_2015_TT_lowP": (0.8159, 0.0086),
    }

    print(f"=== Hypothesis: sigma_8 = alpha_xi^2 = 81/100 = {pred_f} ===")
    print()
    print(f"{'Anchor':<40s} {'Central':>10s} {'sigma':>10s} "
          f"{'rel-err':>10s} {'z-score':>10s}")
    rows = []
    for label, (val, sigma) in anchors.items():
        rel_err = abs(pred_f - val) / val
        z = (pred_f - val) / sigma
        rows.append({
            "anchor": label, "central": val, "sigma": sigma,
            "rel_err_pct": float(100*rel_err),
            "z_score": float(z),
        })
        flag = "PRECISE" if abs(z) < 1 else ("FACTOR2" if abs(z) < 2 else "TENSION")
        print(f"  {label:<38s} {val:>10.4f} {sigma:>10.4f} "
              f"{100*rel_err:>9.3f}% {z:>+9.2f} [{flag}]")
    print()

    # Verdict
    in_planck_1sig = (abs(pred_f - 0.8111) / 0.0060) < 1
    if in_planck_1sig:
        verdict = (
            "STRUCTURAL_MATCH: sigma_8 = alpha_xi^2 = 81/100 = 0.81 "
            "is within Planck 2018 1-sigma band (0.8111 +/- 0.0060). "
            "This identifies the cosmological matter-clustering "
            "amplitude with the same back-channel projection squared "
            "that gives Lambda_t in P4 (back-channel matter asymptote). "
            "Different scales, same chirality-projected mass-energy "
            "amplitude. The S_8 tension between Planck (~0.81) and "
            "weak-lensing surveys (~0.77) sits with Planck closer to "
            "the System-R prediction; weak-lensing surveys are 2-3 "
            "sigma below."
        )
    else:
        verdict = (
            f"NOT_LOCKED: sigma_8 prediction 81/100 differs from "
            f"Planck central {0.8111} by "
            f"{100*abs(pred_f-0.8111)/0.8111:.2f}%."
        )
    print(f"Verdict: {verdict}")

    bundle = {
        "method": "HL_sigma8_alpha_xi_squared",
        "schema_version": "1.0.0",
        "framework_constants": {"alpha_xi": "9/10",
                                  "alpha_xi_squared": "81/100"},
        "structural_prediction": {
            "form": "sigma_8 = alpha_xi^2 = Lambda_t",
            "fraction": "81/100",
            "decimal": pred_f,
            "interpretation": (
                "Cosmological matter clustering amplitude is the "
                "back-channel chirality projection squared, the same "
                "quantity that gives the asymptote of the matter "
                "stress-energy fluctuation Lambda_t in the discrete "
                "relational lattice (P4)."
            ),
        },
        "anchor_comparisons": rows,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
