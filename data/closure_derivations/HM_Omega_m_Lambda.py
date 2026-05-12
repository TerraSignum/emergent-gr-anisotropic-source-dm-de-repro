"""Closure-derivation H-M: cosmological density parameters as System-R rationals.

Ω_m^matter  = γ·N_gen + γ²·d/N_gen     = 47/150 = 0.31333
Ω_m^vacuum  = γ·N_gen + γ²·(d+1)/N_gen = 19/60  = 0.31667  (CMB-era reading)
Ω_Λ^matter  = 1 - Ω_m^matter           = 103/150 = 0.68667

Chirality-flip shift:
  ΔΩ_m = Ω_m^vacuum - Ω_m^matter = γ²/N_gen = 1/300 = 0.00333
  (the integer in the sub-leading γ² correction shifts from d=4
   on the matter branch to d+1=5 on the vacuum branch)

Anchor-dependent fit (chirality-flip-aware reading):
  Planck 2018 TT+TE+EE+lowE+lensing (CMB-only): Ω_m = 0.3158 ± 0.0073
    matches vacuum form 19/60 at z = -0.12 (better than matter z = +0.34)
  Planck + DESI BAO 2024 combination:           Ω_m = 0.3144 ± 0.0073
    matches matter form 47/150 at z = +0.15 (better than vacuum z = -0.31)

The direction of the data-weight bias matches the framework's
chirality-flip hypothesis: CMB-only inference is dominated by
vacuum-branch (early-universe) physics; late-universe-weighted
combinations sit closer to matter-branch.

Both 47/150 (matter) and 19/60 (vacuum) are within Planck 1-sigma
of any single anchor; the structural difference γ²/N_gen = 0.00333
is below current Planck σ ≈ 0.0073 (CMB-S4-class precision needed
to discriminate). Cross-sector verifier:
  emergent-gr-closure-repro/src/verify_chirality_flip_cross_sector.py

The structural reading:
  Ω_m^matter = (N_gen × γ) + (d × γ² / N_gen)
  Ω_m^vacuum = (N_gen × γ) + ((d+1) × γ² / N_gen)
i.e. matter density = three generations × carrier-defect coupling,
plus a {4D, 5D} spacetime sub-leading correction depending on
chirality branch.

Writes peer_reviews/HM_Omega_m_Lambda.json
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "data" / "closure_derivations" / "HM_Omega_m_Lambda.json"


def main():
    GAMMA = Fraction(1, 10)
    N_GEN = 3
    D = 4

    Om_matter = GAMMA * N_GEN + GAMMA**2 * D / N_GEN          # = 47/150
    Om_vacuum = GAMMA * N_GEN + GAMMA**2 * (D + 1) / N_GEN    # = 19/60
    OmL_matter = 1 - Om_matter                                # = 103/150
    delta_branch = Om_vacuum - Om_matter                       # = γ²/N_gen = 1/300

    Om_matter_f = float(Om_matter)
    Om_vacuum_f = float(Om_vacuum)
    OmL_matter_f = float(OmL_matter)
    Om_pred = Om_matter  # legacy alias
    Om_pred_f = Om_matter_f
    OmL_pred = OmL_matter
    OmL_pred_f = OmL_matter_f

    # Anchors (multiple data combinations to test branch dependence)
    anchors_dual = {
        "Planck 2018 (CMB-only)":     (0.3158, 0.0073),
        "Planck + DESI BAO 2024":     (0.3144, 0.0073),
    }
    # Legacy single-branch anchors (matter)
    anchors = {
        "Omega_m":      (0.3158, 0.0073, Om_matter_f, Om_matter),
        "Omega_Lambda": (0.6847, 0.0073, OmL_matter_f, OmL_matter),
    }

    print(f"=== Dual-branch hypothesis (chirality-flip-aware) ===")
    print(f"  Omega_m^matter = gamma*N_gen + gamma^2*d/N_gen      = "
          f"{Om_matter} = {Om_matter_f}")
    print(f"  Omega_m^vacuum = gamma*N_gen + gamma^2*(d+1)/N_gen  = "
          f"{Om_vacuum} = {Om_vacuum_f}")
    print(f"  Branch diff   = gamma^2/N_gen                       = "
          f"{delta_branch} = {float(delta_branch):.5f}")
    print(f"  Omega_L^matter = 1 - Omega_m^matter                  = "
          f"{OmL_matter} = {OmL_matter_f}")
    print()
    print("=== Anchor-by-anchor branch comparison ===")
    print(f"{'Anchor':<30s} {'Om_m_obs':>10s} {'z_matter':>10s} {'z_vacuum':>10s}")
    for label, (val, sigma) in anchors_dual.items():
        z_m = (val - Om_matter_f) / sigma
        z_v = (val - Om_vacuum_f) / sigma
        print(f"  {label:<28s} {val:>10.4f} {z_m:>+10.3f} {z_v:>+10.3f}")
    print()

    rows = []
    print(f"{'Quantity':<14s} {'Predicted':>10s} {'Planck':>10s} {'rel-err':>10s} {'z':>8s}")
    for label, (val, sigma, pred, pred_frac) in anchors.items():
        rel = abs(pred - val) / val
        z = (pred - val) / sigma
        tier = "EXACT" if rel < 0.004 else \
               ("PRECISE" if rel < 0.025 else "FACTOR2")
        print(f"  {label:<12s}  {pred:>10.5f} {val:>10.4f} "
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

    sum_check = Om_pred + OmL_pred
    print(f"=== Closure cross-check ===")
    print(f"  Omega_m + Omega_L (predicted) = {sum_check} (must be 1)")
    print(f"  Ratio Omega_L/Omega_m = {OmL_pred}/{Om_pred} = {OmL_pred/Om_pred} = "
          f"{float(OmL_pred/Om_pred):.5f}")
    print(f"  Planck Omega_L/Omega_m = 0.6847/0.3158 = {0.6847/0.3158:.5f}")
    print(f"  rel-err = {100*abs(float(OmL_pred/Om_pred) - 0.6847/0.3158)/(0.6847/0.3158):.3f}%")
    print()

    # Per-anchor branch comparison for dual-form audit
    anchor_branch_compare = {}
    for label, (val, sigma) in anchors_dual.items():
        anchor_branch_compare[label] = {
            "measured": val, "sigma": sigma,
            "z_to_matter_form": (val - Om_matter_f) / sigma,
            "z_to_vacuum_form": (val - Om_vacuum_f) / sigma,
            "rel_pct_to_matter": (val - Om_matter_f) / Om_matter_f * 100,
            "rel_pct_to_vacuum": (val - Om_vacuum_f) / Om_vacuum_f * 100,
            "preferred_branch": ("vacuum"
                                 if abs(val - Om_vacuum_f)
                                 < abs(val - Om_matter_f)
                                 else "matter"),
        }

    verdict = (
        "DUAL_BRANCH_STRUCTURAL_MATCH: Omega_m has two structurally-natural "
        "System-R rational forms tied to the chirality flip. "
        "Matter branch (late-universe, z=0): Omega_m^matter = "
        "gamma*N_gen + gamma^2*d/N_gen = 47/150 = 0.3133. "
        "Vacuum branch (CMB-era, early-universe): Omega_m^vacuum = "
        "gamma*N_gen + gamma^2*(d+1)/N_gen = 19/60 = 0.3167. "
        "Branch difference = gamma^2/N_gen = 1/300 = 0.00333. "
        "Anchor-by-anchor: Planck 2018 CMB-only (Omega_m = 0.3158) "
        "fits vacuum form better (z=-0.12 vs z=+0.34); "
        "Planck + DESI BAO 2024 combo (Omega_m = 0.3144) fits matter "
        "form better (z=+0.15 vs z=-0.31). Direction of bias matches "
        "framework's chirality-flip hypothesis: CMB-dominated inference "
        "is vacuum-era weighted; late-universe-weighted combinations "
        "are matter-era. Branch diff 1/300 is below current Planck "
        "sigma 0.0073, so cannot discriminate at present precision. "
        "CMB-S4-class measurements with sigma_Omega_m ~ 0.001 would "
        "test this dual-form prediction directly. "
        "Cross-sector verifier: emergent-gr-closure-repro/src/"
        "verify_chirality_flip_cross_sector.py"
    )
    print(f"\nVerdict: {verdict[:180]}...")

    bundle = {
        "method": "HM_Omega_m_Lambda",
        "framework_constants": {"gamma": "1/10", "N_gen": N_GEN, "d": D},
        "Omega_m_matter_branch": {
            "form": "gamma*N_gen + gamma^2*d/N_gen",
            "fraction": f"{Om_matter.numerator}/{Om_matter.denominator}",
            "value": Om_matter_f,
        },
        "Omega_m_vacuum_branch": {
            "form": "gamma*N_gen + gamma^2*(d+1)/N_gen",
            "fraction": f"{Om_vacuum.numerator}/{Om_vacuum.denominator}",
            "value": Om_vacuum_f,
        },
        "branch_difference": {
            "form": "gamma^2/N_gen",
            "fraction": f"{delta_branch.numerator}/{delta_branch.denominator}",
            "value": float(delta_branch),
        },
        "anchor_branch_comparison": anchor_branch_compare,
        "predictions": rows,
        "sum_check": float(sum_check),
        "Omega_L_over_Omega_m_ratio": float(OmL_pred/Om_pred),
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
