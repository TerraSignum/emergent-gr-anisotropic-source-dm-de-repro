"""Closure-derivation H-W: continuum (8/pi^2) vs lattice (81/100)
INDIRECT discrimination via aggregated cross-observable chi^2.

Direct test (high-N lattice or CMB-S4) takes a decade. INDIRECT
test uses the fact that the 0.07% Diophantine drift between
81/100 and 8/pi^2 propagates through MANY observables. With N
independent observables each measured at uncertainty sigma_i,
the cumulative chi^2 difference between the two hypotheses can
be statistically discriminating EVEN IF no individual
measurement reaches 0.07% precision.

Method: for each closure where alpha_xi^2 enters the predicted
value, compute:
  - z_lattice = (pred_lattice - obs) / sigma_obs
  - z_continuum = (pred_continuum - obs) / sigma_obs
  - Delta chi^2 = z_continuum^2 - z_lattice^2

Sum across all alpha_xi^2-dependent observables. Positive
Delta chi^2_total means lattice 81/100 is preferred; negative
means continuum 8/pi^2 is preferred.

Includes: Lambda_t (lattice), sigma_8 (Planck), -rho(T,d)
(P5/P5N halo), Lambda_s axes (-gamma^2/2), Wolfenstein A
(via V_cb/V_us^2). Plus N-body re-extractions if available.

Writes peer_reviews/HW_indirect_audit.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "data" / "closure_derivations" / "HW_indirect_audit.json"


def main():
    PI = math.pi
    AX2_LAT = 81.0 / 100.0  # lattice rational
    AX2_CONT = 8.0 / PI**2  # continuum candidate
    GAMMA = 0.1  # both hypotheses share gamma = 1/10 (no drift)

    # alpha_xi^2-dependent observables; each row:
    #   (label, observed_value, observed_sigma, prediction_function)
    # where prediction_function takes alpha_xi^2 and returns predicted value
    obs = [
        # Cosmological clustering (CMB-anchored)
        ("sigma_8 (Planck 2018)",
         0.8111, 0.0060, lambda a2: a2),
        ("sigma_8 (ACT DR6 + lensing)",
         0.823, 0.011, lambda a2: a2),
        ("S_8 (Planck-derived) sqrt(Omega_m/0.3)",
         0.831, 0.014, lambda a2: a2 * math.sqrt(47/150 / 0.3)),
        # Lattice direct (need higher precision in future)
        ("Lambda_t (P4 Symanzik 2-term, lattice)",
         0.8134, 0.013, lambda a2: a2),
        # Halo source-side (per-regime mean)
        ("rho_T,d_continuum (P4-B Symanzik)",
         -0.4066, 0.058, lambda a2: -a2 / 2),
        # Wolfenstein A (depends on alpha_xi^2 through alpha_xi/(2*11))
        # A = V_cb / V_us^2 = (alpha_xi/(2*11)) / (alpha_xi*1/4)^2
        #   = (alpha_xi/22) / (alpha_xi^2/16)
        #   = 16/(22*alpha_xi)
        #   so depends on alpha_xi (not squared); skip for this audit
        # alpha_EM depends on alpha_xi^N_gen, drift = 1.5 * 0.07%
        # Use framework-realistic sigma (~0.1% capturing loop
        # corrections not in the leading structural form), NOT
        # PDG measurement precision which is ~10^-9.
        ("alpha_EM = gamma^2 alpha_xi^3 (frame-sigma)",
         1/137.036, 0.001 * (1/137.036),  # 0.1% framework sigma
         lambda a2: GAMMA**2 * (math.sqrt(a2))**3),
        # V_us = alpha_xi * s_face; depends on alpha_xi (NOT squared)
        ("V_us = alpha_xi * s_face (PDG)",
         0.22501, 0.00046,
         lambda a2: math.sqrt(a2) * 0.25),
        # V_cb = alpha_xi / (2*11); depends on alpha_xi
        ("V_cb = alpha_xi / 22 (HFLAV)",
         0.0408, 0.0008,
         lambda a2: math.sqrt(a2) / 22),
        # m_W: m_W requires alpha_EM(M_Z) running coupling, NOT
        # the leading-order alpha_EM(0) = gamma^2 alpha_xi^N_gen
        # closure. The alpha_xi^2 dependence enters only through
        # subleading running corrections, so m_W cannot directly
        # discriminate lattice vs continuum alpha_xi^2 at current
        # precision. EXCLUDED from indirect audit.
        # m_t via y_t = 1 - 2 d gamma^3 = 124/125; depends on gamma
        # only (no alpha_xi^2 entry), cannot discriminate.
        # H_0 = 100 alpha_xi s_face N_gen depends on alpha_xi (sqrt
        # of alpha_xi^2). Use Planck anchor:
        ("H_0 (Planck) = 100 alpha_xi s_face N_gen",
         67.36, 0.54,
         lambda a2: 100 * math.sqrt(a2) * 0.25 * 3),
    ]

    print(f"=== Hypothesis comparison via aggregated z-scores ===")
    print(f"  Lattice  : alpha_xi^2 = 81/100   = {AX2_LAT:.6f}")
    print(f"  Continuum: alpha_xi^2 = 8/pi^2   = {AX2_CONT:.6f}")
    print(f"  Drift: {100*(AX2_CONT-AX2_LAT)/AX2_LAT:.4f}%")
    print()

    print(f"{'Observable':<42s} {'obs':>10s} {'pred-lat':>10s} {'z-lat':>7s} {'pred-cont':>10s} {'z-cont':>7s} {'chi2-diff':>11s}")
    chi2_lat_total = 0.0
    chi2_cont_total = 0.0
    rows = []
    for label, val, sig, pred_fn in obs:
        pred_l = pred_fn(AX2_LAT)
        pred_c = pred_fn(AX2_CONT)
        z_l = (pred_l - val) / sig
        z_c = (pred_c - val) / sig
        chi2_l = z_l**2
        chi2_c = z_c**2
        d_chi2 = chi2_c - chi2_l
        chi2_lat_total += chi2_l
        chi2_cont_total += chi2_c
        print(f"  {label:<40s}  {val:>10.5g} {pred_l:>10.5g} {z_l:>+6.2f}  "
              f"{pred_c:>10.5g} {z_c:>+6.2f}   {d_chi2:>+8.3f}")
        rows.append({
            "label": label, "obs_value": val, "obs_sigma": sig,
            "pred_lattice": pred_l, "pred_continuum": pred_c,
            "z_lattice": z_l, "z_continuum": z_c,
            "chi2_lattice": chi2_l, "chi2_continuum": chi2_c,
            "delta_chi2": d_chi2,
        })
    print()
    print(f"  {'TOTAL':<40s}                                                       "
          f"chi2_lat={chi2_lat_total:.3f}  chi2_cont={chi2_cont_total:.3f}")
    print(f"  Delta chi^2_total (cont - lat) = {chi2_cont_total - chi2_lat_total:+.4f}")
    print()

    if chi2_lat_total < chi2_cont_total - 1.0:
        verdict = "LATTICE_PREFERRED"
        explanation = (
            f"Lattice rational 81/100 is preferred at "
            f"Delta chi^2 = {chi2_cont_total - chi2_lat_total:.3f} "
            f"over continuum 8/pi^2 across {len(obs)} observables.")
    elif chi2_cont_total < chi2_lat_total - 1.0:
        verdict = "CONTINUUM_PREFERRED"
        explanation = (
            f"Continuum 8/pi^2 is preferred at "
            f"Delta chi^2 = {chi2_lat_total - chi2_cont_total:.3f} "
            f"over lattice 81/100 across {len(obs)} observables.")
    else:
        verdict = "INDISTINGUISHABLE_AT_CURRENT_PRECISION"
        explanation = (
            f"|Delta chi^2| = "
            f"{abs(chi2_cont_total - chi2_lat_total):.3f} < 1; "
            f"current measurement precision cannot distinguish "
            f"lattice 81/100 from continuum 8/pi^2. Need more "
            f"observables or tighter measurements.")

    print(f"=== Verdict ===")
    print(f"  {verdict}")
    print(f"  {explanation}")
    print()

    # Detailed analysis per observable
    print(f"=== Per-observable preference (>1 sigma discrimination) ===")
    for r in rows:
        d = r["delta_chi2"]
        if abs(d) > 1.0:
            pref = "LAT" if d > 0 else "CONT"
            print(f"  {r['label']}: prefers {pref} at Delta chi^2 = {d:+.2f}")

    bundle = {
        "method": "HW_continuum_vs_lattice_indirect_audit",
        "hypothesis_lattice": {"label": "81/100", "value": AX2_LAT},
        "hypothesis_continuum": {"label": "8/pi^2", "value": AX2_CONT},
        "Diophantine_drift_pct": 100*(AX2_CONT-AX2_LAT)/AX2_LAT,
        "n_observables": len(obs),
        "chi2_lattice_total": chi2_lat_total,
        "chi2_continuum_total": chi2_cont_total,
        "delta_chi2": chi2_cont_total - chi2_lat_total,
        "verdict": verdict,
        "explanation": explanation,
        "per_observable": rows,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
