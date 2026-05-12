"""Per-regime extension status of the cosmological-constant eight-layer
dressing closure.

Reads the broader-corpus inputs and reports, per regime, whether the
GCC eight-layer dressing pipeline CAN be evaluated on that regime.

The pipeline depends on upstream HBR / PG-02 / CSP-03 / TAO-05 / DQC
outputs that produce regime-suffixed inputs ('v_EW_predicted_GeV_p<reg>',
'S_gamow_p<reg>', 'spectral_gap_p<reg>', 'd_spectral_eff_p<reg>',
'E_vac_regularized_p<reg>'). The upstream HBR-EXT-01 audit
(outputs_hbr_audit/hbr_ext_01_cross_regime_inventory.md, 2026-04-24)
documents that those upstream outputs structurally exist only for P1
and P2' (status STRUCTURAL_TWO_REGIME_LOCK); the upstream chain has
not been run on the additional regimes.

This script therefore:
 - reports availability of the per-regime D1 inputs (E_geom, E_res,
   fixpoint stability) on the canonical regimes P3..P8 and the
   lattice-N variants of P5;
 - reports the structural lock on the upstream HBR/PG-02/CSP-03/TAO-05
   outputs;
 - emits a per-regime extension manifest that lists, for each regime,
   exactly which upstream pipelines would have to run before the GCC
   dressing closure could be evaluated on that regime.

Output: outputs/cc_dressing_per_regime_extended.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent
OUT = REPO / "outputs" / "cc_dressing_per_regime_extended.json"

CANONICAL_REGIMES = ("p3", "p4", "p5", "p6", "p7", "p8")
LATTICE_VARIANTS = (
    ("p5n64",  ROOT / "results_d1_p5n64" / "d1_p5n64.json"),
    ("p5n100", ROOT / "results_d1_p5n100" / "d1_p5n100.json"),
    ("p5n128", ROOT / "results_d1_p5n128_kq_fixed" / "d1_p5n128.json"),
    ("p5n200", ROOT / "results_d1_p5n200_8seeds" / "d1_p5n200.json"),
    ("p5n256", ROOT / "results_d1_p5n256" / "d1_p5n256.json"),
    ("p5n300", ROOT / "results_d1_p5n300" / "d1_p5n300.json"),
)

UPSTREAM_PIPELINES = (
    ("HBR",     "outputs_hbr_audit / hierarchy_bridge_bundle",
                "v_EW_predicted_GeV, S_gamow"),
    ("PG-02",   "outputs / stationary_core_physics_bundle",
                "core_mass_proxy, born_hbar_eff"),
    ("CSP-03",  "outputs / cross_section_phenomenology_bundle",
                "scattering_length, V_depth"),
    ("TAO-05",  "outputs / thermal_averaged_observables_bundle",
                "eta_sommerfeld"),
    ("DQC",     "outputs / dqc bundle",
                "spectral_gap, d_spectral_eff, E_vac_regularized"),
)

D1_CANDIDATES_TEMPLATE = (
    "results_d1_fix17/d1_{reg}.json",
    "results_d1_fix16/d1_{reg}.json",
    "results_d1_fix16/{reg}/d1_{reg}.json",
    "results_d1_fix14/d1_{reg}.json",
    "results_d1_fix14/{reg}/d1_{reg}.json",
)


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def d1_status_for(reg, candidates_template=D1_CANDIDATES_TEMPLATE,
                   override_path=None):
    """Return (has_E_geom_E_res, source_path_or_None, mtime, n_keys)."""
    if override_path is not None:
        candidates = [override_path]
    else:
        candidates = [ROOT / t.format(reg=reg) for t in candidates_template]
    for p in candidates:
        if not p.exists():
            continue
        d = _load(p)
        has = (d.get("E_geom") is not None) and (d.get("E_res") is not None)
        mtime = os.path.getmtime(p)
        return {
            "has_E_geom_E_res": has,
            "source_path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "mtime_unix": mtime,
            "n_keys": len(d),
            "E_geom": d.get("E_geom"),
            "E_res": d.get("E_res"),
        }
    return {
        "has_E_geom_E_res": False,
        "source_path": None,
        "mtime_unix": None,
        "n_keys": None,
    }


def _gravitational_fraction_log10(rec):
    """Layer 5 reduction = log10(E_geom / E_res)."""
    import math
    eg = rec.get("E_geom")
    er = rec.get("E_res")
    if eg is None or er is None or er == 0 or eg == 0:
        return None
    return math.log10(eg / er)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canonical = []
    for reg in CANONICAL_REGIMES:
        rec = {"regime": reg, **d1_status_for(reg)}
        rec["layer5_gravitational_fraction_log10"] = (
            _gravitational_fraction_log10(rec))
        canonical.append(rec)
    lattice = []
    for reg, override in LATTICE_VARIANTS:
        rec = {"regime": reg, **d1_status_for(reg, override_path=override)}
        rec["layer5_gravitational_fraction_log10"] = (
            _gravitational_fraction_log10(rec))
        lattice.append(rec)

    out = {
        "method": "Per-regime extension status of CC eight-layer dressing",
        "structural_two_regime_lock": {
            "documented_at": "outputs_hbr_audit/hbr_ext_01_cross_regime_inventory.md",
            "audit_date": "2026-04-24",
            "verdict": "STRUCTURAL_TWO_REGIME_LOCK",
            "summary": (
                "The CC eight-layer dressing pipeline (gap_closure_cosmology) "
                "depends on regime-suffixed outputs of the upstream HBR, "
                "PG-02, CSP-03, TAO-05 and DQC pipelines. Those upstream "
                "pipelines have only been run on P1 and P2'. Extension to "
                "any additional regime requires running the upstream chain "
                "on that regime first; there is no 'pure arithmetic' "
                "extension from existing P1/P2' outputs."
            ),
            "upstream_pipelines_required": [
                {"name": n, "bundle": b, "outputs": o}
                for (n, b, o) in UPSTREAM_PIPELINES
            ],
        },
        "canonical_regimes_p3_p8_d1_status": canonical,
        "lattice_n_variants_p5_d1_status": lattice,
        "what_can_be_done_now": [
            "Algebraic identity Lambda_lat^row-mean = 19/15 is "
            "regime-independent by construction; no per-regime "
            "extension required.",
            "D1 layer (Layer 5: gravitational fraction E_geom/E_res) is "
            "computable on regimes whose d1.json carries E_geom and "
            "E_res; the canonical-regime ladder P0..P8 and the lattice-N "
            "variants P5N64, P5N100 carry these keys (older 1178-key "
            "format). Layer 5 alone explains roughly 8 of the 122 "
            "orders.",
            "Full eight-layer per-regime closure on P3..P8 and the "
            "lattice-N variants is blocked at the upstream HBR / PG-02 / "
            "CSP-03 / TAO-05 / DQC level; promoting them to a per-regime "
            "loop is upstream work outside the scope of this repro.",
        ],
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print("=== Structural two-regime lock ===")
    print(out["structural_two_regime_lock"]["verdict"])
    print(f"  documented at: {out['structural_two_regime_lock']['documented_at']}")
    print()
    print("=== Canonical-regime D1 status + Layer-5 dressing (log10) ===")
    for r in canonical:
        flag = "OK" if r["has_E_geom_E_res"] else "no E_geom/E_res"
        l5 = r["layer5_gravitational_fraction_log10"]
        l5s = f"{l5:+.2f}" if l5 is not None else "-"
        print(f"  {r['regime']}: {flag:>16s}  L5_log10 = {l5s}")
    print()
    print("=== Lattice-N variants D1 status + Layer-5 dressing (log10) ===")
    for r in lattice:
        flag = "OK" if r["has_E_geom_E_res"] else "no E_geom/E_res"
        l5 = r["layer5_gravitational_fraction_log10"]
        l5s = f"{l5:+.2f}" if l5 is not None else "-"
        print(f"  {r['regime']}: {flag:>16s}  L5_log10 = {l5s}")


if __name__ == "__main__":
    main()
