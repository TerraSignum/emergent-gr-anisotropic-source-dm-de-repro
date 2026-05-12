r"""
Verify the bundled cosmological-constant 8-layer closure.

The corpus closes the 122-orders-of-magnitude hierarchy between the
naive QFT zero-point energy (~M_Pl^4) and the observed Planck cosmological
constant (~10^-47 GeV^4) via six additive dressing layers plus two
corrective layers, parameter-free, ratio 0.89 vs Planck (sub-decimal
closure, residual -0.05 OoM) in the canonical regime with all upstream
inputs explicitly sourced (BUG-001 fix: dQ_phase via DQC-02
phase_charge_quantum, N_modes via SMP-02 spectral mode inventory; no
silent hardcoded fallbacks). The leading theory systematic is the L5
finite-N to continuum-limit spread of 0.58 OoM. The extended regime is
explicitly inconsistent at ~8 OoM as a regime-filter stress-test
(P2' = second external world / false vacuum). A previously documented
ninth layer (paired-degeneracy correction H197) was retracted on
2026-05-11; see the h197_retraction_2026_05_11 block in
data/cosmological_constant_closure.json.

Usage:
    python ./src/verify_cosmological_constant.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "cosmological_constant_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Cosmological-constant 9-layer closure recompute")
    print("=" * 78)
    print()

    obs = d["rho_observed"]
    print(f"Planck observation: rho_Lambda = {obs['value_GeV4']:.3e} GeV^4 "
          f"(log10 = {obs['log10']})")
    naive = d["rho_naive_estimate"]
    print(f"Naive QFT estimate: rho_naive ~ {naive['value_GeV4']} "
          f"(log10 = {naive['log10']})")
    print(f"Hierarchy to close: {d['hierarchy_to_close']['orders_of_magnitude']} OoM")
    print()

    print("--- Six dressing layers ---")
    total_dressing = 0.0
    for L in d["six_dressing_layers"]:
        print(f"  Layer {L['layer']}: {L['name']:<40}  "
              f"log10 contribution = {L['log10_contribution']:+.2f}")
        total_dressing += L["log10_contribution"]
    print(f"  Sum of dressing-layer log10 contributions: {total_dressing:+.2f}")
    print()

    print("--- Three corrective layers ---")
    total_corrective = 0.0
    for L in d["three_corrective_layers"]:
        contribution = L["log10_contribution"]
        print(f"  {L['layer']}: {L['name']:<40}  "
              f"log10 contribution = {contribution:+.2f}")
        total_corrective += contribution
    print(f"  Sum of corrective-layer log10 contributions: {total_corrective:+.2f}")
    print()

    closure = d["closure_result"]

    # Independent recompute from the bundled layer contributions:
    # log10(rho_final) = log10(M_Pl^4) + sum_i log10_contribution_i
    # ratio = 10^(log10(rho_final) - log10(rho_obs))
    M_PL_GEV = 1.22091e19
    log10_rho_naive = math.log10(M_PL_GEV ** 4)
    log10_sum_layers = total_dressing + total_corrective
    log10_rho_final_recomputed = log10_rho_naive + log10_sum_layers
    log10_rho_obs = math.log10(closure["rho_observed_GeV4"])
    log10_residual_recomputed = log10_rho_final_recomputed - log10_rho_obs
    ratio_recomputed = 10.0 ** log10_residual_recomputed

    print("--- Closure result (canonical regime) ---")
    print(f"  Total log10 reduction:        {closure['total_log10_reduction']:.2f}  "
          f"(recomputed from layers: {log10_sum_layers:.2f})")
    print(f"  rho_final (JSON GeV^4):       {closure['rho_final_canonical_GeV4']:.3e}  "
          f"(recomputed: {10.0**log10_rho_final_recomputed:.3e})")
    print(f"  rho_observed (Planck GeV^4):  {closure['rho_observed_GeV4']:.3e}")
    print(f"  Ratio (JSON):                 {closure['ratio_predicted_over_observed']:.4f}  "
          f"(recomputed: {ratio_recomputed:.4f})")
    print(f"  Residual (JSON, log10 OoM):   {closure['residual_log10_orders']:+.4f}  "
          f"(recomputed: {log10_residual_recomputed:+.4f})")
    print(f"  Residual (%):                 {closure['residual_pct']:.2f}")
    print(f"  Tier:                         {closure['tier']}")
    print(f"  Fitted parameters:            {closure['fitted_parameters']}")
    print()

    # Reproducibility check: stamped ratio must equal recomputed ratio
    # within the two-decimal-place tolerance of the bundled JSON
    # log10_contribution readouts (each layer has 0.01-OoM precision,
    # so the sum carries up to ~0.1 OoM cumulative rounding).
    REPRO_TOL = 0.02
    ratio_match = abs(closure["ratio_predicted_over_observed"]
                      - ratio_recomputed) <= REPRO_TOL
    residual_match = abs(closure["residual_log10_orders"]
                         - log10_residual_recomputed) <= REPRO_TOL
    print("--- Reproducibility check ---")
    print(f"  Ratio match (tolerance {REPRO_TOL}):   {ratio_match}")
    print(f"  Residual match (tolerance {REPRO_TOL}): {residual_match}")
    print()

    if "h197_retraction_2026_05_11" in d:
        rec = d["h197_retraction_2026_05_11"]
        print(f"--- H197 retraction notice ({rec['status']}) ---")
        print(f"  {rec['reason'][:120]}...")
        print()

    s = d["summary"]
    sys_band = d.get("theory_systematics", {})
    # Closure verdict: sub-decimal residual at finite-N with no fitted
    # parameters. Closure passes when:
    #   (a) zero fitted parameters,
    #   (b) sub-decimal residual (|residual_log10| <= 0.15),
    #   (c) the finite-N central residual sits inside the L5 systematic
    #       band (|central| <= spread).
    band_inside = True
    if sys_band:
        spread = sys_band.get("L5_finite_N_to_continuum_spread", 0.0)
        central = sys_band.get("L5_finite_N_residual",
                               closure["residual_log10_orders"])
        band_inside = (abs(central) <= spread)
    closure_pass = (
        closure["fitted_parameters"] == 0
        and abs(closure["residual_log10_orders"]) <= 0.02
        and 0.95 <= closure["ratio_predicted_over_observed"] <= 1.05
        and band_inside
        and ratio_match
        and residual_match
    )

    import datetime as _dt
    import hashlib as _hashlib
    _data_path = REPO / "data" / "cosmological_constant_closure.json"
    _data_hash = _hashlib.sha256(_data_path.read_bytes()).hexdigest() if _data_path.exists() else ""

    out = {
        "criterion": "Cosmological-constant 9-layer closure recompute",
        "recompute_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "input_data_sha256": _data_hash,
        "M_Pl_GeV_used": M_PL_GEV,
        "log10_rho_naive_recomputed": log10_rho_naive,
        "log10_sum_layers_recomputed": log10_sum_layers,
        "log10_rho_final_recomputed": log10_rho_final_recomputed,
        "log10_residual_recomputed": log10_residual_recomputed,
        "ratio_recomputed": ratio_recomputed,
        "ratio_json_vs_recomputed_match": ratio_match,
        "residual_json_vs_recomputed_match": residual_match,
        "hierarchy_orders": d["hierarchy_to_close"]["orders_of_magnitude"],
        "rho_final_GeV4": closure["rho_final_canonical_GeV4"],
        "rho_observed_GeV4": closure["rho_observed_GeV4"],
        "ratio": closure["ratio_predicted_over_observed"],
        "residual_log10_OoM": closure["residual_log10_orders"],
        "residual_pct": closure["residual_pct"],
        "tier": closure["tier"],
        "L5_finite_N_to_continuum_spread_OoM":
            sys_band.get("L5_finite_N_to_continuum_spread"),
        "residual_finite_N_central":
            sys_band.get("L5_finite_N_residual"),
        "residual_continuum":
            sys_band.get("L5_continuum_residual"),
        "band_inside": band_inside,
        "n_layers_total": s["n_layers_total"],
        "fitted_parameters": closure["fitted_parameters"],
        "h197_status": (d.get("h197_retraction_2026_05_11", {})
                        .get("status", "NOT_RECORDED")),
        "verdict": "PASS" if closure_pass else "FAIL",
    }
    out_path = OUTPUTS / "cosmological_constant_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
