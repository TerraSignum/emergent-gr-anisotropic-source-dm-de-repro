"""Tests for the cosmological-constant 9-layer closure (122-OoM hierarchy)."""

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_cosmological_constant as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "cosmological_constant_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_122_orders_of_magnitude_hierarchy(bundle):
    h = bundle["hierarchy_to_close"]
    assert h["orders_of_magnitude"] == 122


def test_six_dressing_layers_present(bundle):
    layers = bundle["six_dressing_layers"]
    assert len(layers) == 6
    names = [L["name"] for L in layers]
    assert "EW-hierarchy suppression" in names
    assert "Gamow vacuum sequestering" in names
    assert "Spectral-dimension IR screening" in names


def test_three_corrective_layers_present(bundle):
    cor = bundle["three_corrective_layers"]
    assert len(cor) == 3
    layer_ids = [L["layer"] for L in cor]
    assert layer_ids == ["H195", "H196", "H_sync"]


def test_h_sync_derived_layer(bundle):
    """H_sync is the derived 9th layer with log10-multiplier eps_sync^2
    = gamma/2 = 1/20 from P2 §5 C_3 fluctuation-dissipation symmetry.
    """
    cor = bundle["three_corrective_layers"]
    h_sync = next(L for L in cor if L["layer"] == "H_sync")
    assert h_sync["log10_contribution"] == pytest.approx(0.05, rel=1e-6)
    assert "h_sync_promotion_2026_05_11" in bundle
    promotion = bundle["h_sync_promotion_2026_05_11"]
    assert promotion["status"] == "PROMOTED_TO_CLOSED_LAYER"


def test_h197_retracted(bundle):
    """H197 was retracted on 2026-05-11; H_sync is a structurally
    distinct mechanism that replaces it as the 9th layer.
    """
    assert "h197_retraction_2026_05_11" in bundle
    rec = bundle["h197_retraction_2026_05_11"]
    assert rec["status"] == "RETRACTED"


def test_closure_within_exact_band(bundle):
    """Residual must be inside +/-0.02 OoM (EXACT tier under any of
    the three corpus EXACT cuts).

    The nine upstream layers (L1..L6 + H195 + H196 + H_sync) sum to
    -122.94 OoM, exceeding the required 122.95 OoM by within rounding.
    The canonical-regime closure is therefore at residual ~ 0 OoM
    (ratio 0.992, 0.8% below the Planck observation, EXACT).
    """
    c = bundle["closure_result"]
    assert abs(c["residual_log10_orders"]) <= 0.02
    assert 0.95 <= c["ratio_predicted_over_observed"] <= 1.05
    assert c["tier"] in ("EXACT", "SUB_DECIMAL_PRECISE", "PRECISE", "PRECISE_FACTOR_2")


def test_zero_fitted_parameters(bundle):
    assert bundle["closure_result"]["fitted_parameters"] == 0
    assert bundle["summary"]["fitted_parameters"] == 0


def test_planck_anchor_value(bundle):
    """Planck rho_Lambda is 2.49e-47 GeV^4 (cosmological-constant value)."""
    obs = bundle["rho_observed"]
    assert obs["value_GeV4"] == pytest.approx(2.49e-47, rel=1e-2)


def test_canonical_residual_inside_l5_systematic(bundle):
    """The L5 finite-N -> continuum theory-systematic band of 0.58 OoM
    is the leading uncertainty on the closure tier; the canonical
    residual sits inside this band.
    """
    sys_band = bundle["theory_systematics"]
    spread = sys_band["L5_finite_N_to_continuum_spread"]
    central = sys_band["L5_finite_N_residual"]
    assert abs(central) <= spread, (
        f"finite-N residual {central} must lie inside L5 spread {spread}")


def test_extended_regime_explicitly_inconsistent(bundle):
    """The extended regime is a stress-test (P2' = second external world
    / false vacuum) that should fail at multiple OoM.
    """
    e = bundle["extended_regime_residual"]
    assert e["residual_log10_orders"] > 5.0  # explicitly large, regime-filter signal


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["hierarchy_orders"] == 122
    assert output["fitted_parameters"] == 0
    assert output["n_layers_total"] == 9


def test_repro_match(output):
    """The verifier recomputes ratio and residual from per-layer
    log10 contributions plus M_Pl^4 and asserts both match the
    JSON-stated headline numbers within 0.02 OoM. This guarantees
    the closure_result fields are not hardcoded but reproducible.
    """
    assert output["ratio_json_vs_recomputed_match"] is True, (
        f"Ratio mismatch: JSON {output['ratio']:.4f} vs recomputed "
        f"{output['ratio_recomputed']:.4f}")
    assert output["residual_json_vs_recomputed_match"] is True, (
        f"Residual mismatch: JSON {output['residual_log10_OoM']:.4f} "
        f"vs recomputed {output['log10_residual_recomputed']:.4f}")
    # The recomputed ratio must itself be EXACT under all three corpus cuts
    # (i.e., within +/-1% of unity).
    assert 0.99 <= output["ratio_recomputed"] <= 1.01, (
        f"Recomputed ratio {output['ratio_recomputed']:.4f} outside "
        f"EXACT band [0.99, 1.01]")
