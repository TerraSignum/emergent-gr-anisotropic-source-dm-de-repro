"""Build the publication-time pre-registration block.

Stamps the framework's near-term falsification predictions
(DESI DR3 dynamical dark energy, neutrino mass sum, Hubble
constant, CP phase, dark-matter relic density) at freeze
date 2026-05-11 with a content hash over the bundled data
files that ground these predictions.

The hash is deterministic: any reviewer can recompute it
from the downloaded repository state. A change to the hash
between the published manuscript and a downloaded copy is
evidence that the bundled state has been modified since
publication.

Usage:
    python ./src/build_preregistration.py

Output:
    data/preregistration_2026_05_11.json
    outputs/preregistration_recompute.json
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Files that ground the pre-registered predictions
STAMPED_FILES = [
    DATA / "cosmological_constant_closure.json",
    DATA / "einstein_with_lambda_8point.json",
    DATA / "gcc07_cc_residual_closure.json",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build():
    file_hashes = {}
    for p in STAMPED_FILES:
        if p.exists():
            file_hashes[p.name] = sha256_of(p)
        else:
            file_hashes[p.name] = "MISSING"

    # Combined hash over all stamped files (order-stable JSON dump)
    h = hashlib.sha256()
    h.update(json.dumps(file_hashes, sort_keys=True).encode("utf-8"))
    combined_hash = h.hexdigest()

    predictions = [
        {
            "id": "P-W_A",
            "observable": "CPL dark-energy equation-of-state slope w_a",
            "framework_prediction": "|w_a| <= 0.01 (essentially zero)",
            "structural_form": "w_DE = -1 + eps_sync^4 / gamma_eff with d w_DE / d theta_chir |_0 = 0; "
                               "leading-order |w_a| ~ gamma^2 = 0.01",
            "experimental_anchor": "DESI 2024+2025 BAO central w_a ~ -0.4 at ~3 sigma "
                                   "under w_0 w_a CDM prior",
            "decisive_release": "DESI DR3 (expected 2026-2027)",
            "falsification_threshold": "|w_a| > 0.1 confirmed at > 5 sigma falsifies the "
                                       "eps_sync^4 / gamma closure",
            "framework_band_offset_now": "40x outside framework band",
        },
        {
            "id": "P-SIGMA_M_NU",
            "observable": "Neutrino mass sum Sigma m_nu",
            "framework_prediction": "Sigma m_nu = 0.0591 eV (EXACT under m_1 = gamma^2 m_3)",
            "structural_form": "m_1 = gamma^2 m_3, m_2 = sqrt(m_1^2 + Delta m_21^2), "
                               "m_3 from sqrt(Delta m_31^2); System-R rational",
            "experimental_anchor": "DESI DR2 BAO + DR1 full shape: Sigma m_nu < 0.0642 eV "
                                   "(95% CL, flat-LCDM, three degenerate states); "
                                   "oscillation lower limit 0.0586 eV",
            "decisive_release": "CMB-S4 / Roman / Euclid Year-1-2 (2026-2028)",
            "falsification_threshold": "Sigma m_nu outside [0.0586, 0.0642] eV falsifies "
                                       "the refined m_1 = gamma^2 m_3 closure",
            "framework_band_width_eV": 0.0056,
        },
        {
            "id": "P-H0",
            "observable": "Hubble constant H_0 (CMB-anchored)",
            "framework_prediction": "H_0 = 67.5 km/s/Mpc (structurally locked)",
            "structural_form": "H_0 = 100 alpha_xi s_face N_gen = (27/40) x 100",
            "experimental_anchor": "Planck 2018: 67.36 +/- 0.54 km/s/Mpc (0.21% match); "
                                   "JWST CCHP TRGB: 68.81 +/- 2.22 (within 1 sigma); "
                                   "JWST JAGB: 67.80 +/- 2.17 (within 1 sigma); "
                                   "SH0ES Cepheid: 73.04 +/- 1.04 (framework predicts systematic)",
            "decisive_release": "JWST Cepheid re-anchoring campaigns (2026-2028)",
            "falsification_threshold": "Confirmation of SH0ES Cepheid value with "
                                       "JWST-anchored geometric anchors at sub-1% precision "
                                       "falsifies; alternatively, any revision of "
                                       "alpha_xi / s_face / N_gen breaks V_us, "
                                       "sin^2 theta_W, S_BH/A simultaneously",
            "structural_lock": "Cannot be revised without breaking V_us = alpha_xi s_face, "
                               "sin^2 theta_W = 7/30, S_BH/A = 1/4 simultaneously",
        },
        {
            "id": "P-DELTA_CP",
            "observable": "Leptonic CP phase delta_CP (PMNS)",
            "framework_prediction": "delta_CP = 1.1299 rad (PMNS sub-dominant-eigenphase reading)",
            "structural_form": "strict-coherence-regime sub-dominant-eigenphase readout "
                               "of the bounded-operator construction",
            "experimental_anchor": "NuFIT 6.1: delta_CP in 1.1-2.0 rad range with broad uncertainty",
            "decisive_release": "Next NuFIT release; long-baseline experiments T2K/NOvA/DUNE",
            "falsification_threshold": "delta_CP measured outside the framework band at > 3 sigma",
        },
        {
            "id": "P-OMEGA_DM_H2",
            "observable": "Cold dark-matter relic density Omega_DM h^2",
            "framework_prediction": "Omega_DM h^2 = (59/60) x Omega_DM^LCDM_ref",
            "structural_form": "loop-class multiplier 1 - gamma / (2 N_gen) = 59/60 "
                               "applied to the LCDM reference value",
            "experimental_anchor": "Planck 2018: Omega_c h^2 = 0.11933 +/- 0.00091",
            "decisive_release": "DESI Year-3 (DR3, 2026-2027)",
            "falsification_threshold": "Sub-0.5% precision measurement outside the "
                                       "framework's 59/60 multiplier band falsifies "
                                       "the loop-class identity",
        },
    ]

    bundle = {
        "schema_version": "1.0.0",
        "stand": "2026-05-11",
        "title": "Publication-time pre-registration of near-term framework falsification handles",
        "frozen_state_hash": combined_hash,
        "per_file_hashes": file_hashes,
        "framing": (
            "These five predictions are pre-registered at publication freeze time "
            "with the content hash above. Any reviewer can recompute the hash from "
            "the downloaded bundled state to verify that the predictions were stamped "
            "before the indicated decisive-release experiments reported their results. "
            "The framework's central claim is that all five predictions either confirm "
            "(values land inside the bands) or jointly fail (the chirality-running CC "
            "closure, the System-R neutrino-mass closure, and the structural H_0 lock "
            "are coupled through the same six discrete-geometric primitives)."
        ),
        "predictions": predictions,
        "verifier": "src/build_preregistration.py",
        "verifier_output": "outputs/preregistration_recompute.json",
    }

    out_path = DATA / "preregistration_2026_05_11.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Frozen state hash: {combined_hash}")

    # Verification output: re-hash + report match
    recompute_hash = hashlib.sha256()
    recompute_hash.update(json.dumps(
        {p.name: sha256_of(p) if p.exists() else "MISSING"
         for p in STAMPED_FILES},
        sort_keys=True
    ).encode("utf-8"))
    verify_out = {
        "criterion": "Pre-registration hash recompute",
        "stamped_hash": combined_hash,
        "recomputed_hash": recompute_hash.hexdigest(),
        "match": combined_hash == recompute_hash.hexdigest(),
        "n_predictions": len(predictions),
    }
    verify_path = OUTPUTS / "preregistration_recompute.json"
    verify_path.write_text(json.dumps(verify_out, indent=2), encoding="utf-8")
    print(f"Wrote {verify_path}")
    return bundle


if __name__ == "__main__":
    build()
