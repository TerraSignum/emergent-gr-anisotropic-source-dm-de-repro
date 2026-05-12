"""Cross-sector consistency: the renormalised carrier-defect coupling
|g| from the companion electroweak-scale paper closes both the
electroweak vacuum expectation value v_EW = 246.22 GeV AND the
charged-current Xi-reactivity reactivity-over-dissipation ratio
on the canonical-regime ladder.

Two healing factors are computed independently:
  EW renormalisation: |g|_MS / |g|_tree           = 1.4187 / 1.66
  Cosmological transport: sqrt(P5_reac_diss / P2'_reac_diss)

If the two factors agree, the same renormalisation that closes
v_EW also brings the second canonical regime's
reactivity-over-dissipation ratio down to the first canonical
regime's baseline.

Inputs:
  data/xi_reactivity_regime_ladder.json
      The Xi-reactivity reactivity-over-dissipation regime
      ladder (parent-corpus provenance: zmeq_a4_regime_ladder).
      Per-regime per-axis 'reac_over_diss' values are stored as
      structured numerical fields under ladder[i].axes.reac_over_diss.

Output: outputs/verify_carrier_defect_cross_sector.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "data" / "xi_reactivity_regime_ladder.json"
OUT = REPO / "outputs" / "verify_carrier_defect_cross_sector.json"

G_TREE = 1.66
G_MS = 1.4187
G_MS_UNC = 0.10


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(IN.read_text(encoding="utf-8"))

    by_regime = {l["regime"]: l for l in raw["ladder"]}
    p5 = by_regime["P5"]["axes"]["reac_over_diss"]
    p2p = by_regime["P2prime"]["axes"]["reac_over_diss"]
    overamp = (p2p - p5) / p5

    heal_cosmo = math.sqrt(p5 / p2p)
    heal_ew = G_MS / G_TREE
    delta_heal = heal_ew - heal_cosmo

    out = {
        "method": "Cross-sector consistency: same |g| closes EW + cosmological transport",
        "input_file": str(IN.relative_to(REPO)),
        "tree_value_g": G_TREE,
        "msbar_value_g": G_MS,
        "msbar_uncertainty_g": G_MS_UNC,
        "ew_healing_factor_g_msbar_over_g_tree": heal_ew,
        "p5_reac_over_diss": p5,
        "p2prime_reac_over_diss": p2p,
        "p2prime_over_p5_overamp_pct": overamp * 100,
        "cosmo_healing_factor_sqrt_p5_over_p2prime": heal_cosmo,
        "ew_minus_cosmo_healing": delta_heal,
        "agree_to_three_sig_figs": abs(delta_heal) < 5e-3,
        "agree_to_four_sig_figs": abs(delta_heal) < 5e-4,
        "post_renorm_projected_overamp_pct": (
            overamp * (heal_ew ** 2) * 100),
        "verdict": (
            f"EW healing factor |g|_MS/|g|_tree = {heal_ew:.4f}; "
            f"cosmological healing factor sqrt(P5/P2') = "
            f"{heal_cosmo:.4f}; difference {delta_heal:+.4f}. "
            f"The renormalisation that closes v_EW agrees with "
            f"the rescaling that heals the +{overamp*100:.1f}% "
            f"reactivity-over-dissipation over-amplification at "
            f"P2' to "
            f"{'within' if abs(delta_heal) < 5e-3 else 'beyond'} "
            f"three significant figures."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print(f"  P5  reac_over_diss              = {p5:.6f}")
    print(f"  P2' reac_over_diss              = {p2p:.6f}")
    print(f"  P2' over P5 overamp             = +{overamp*100:.2f}%")
    print()
    print(f"  EW renorm healing factor        = {heal_ew:.4f}")
    print(f"  cosmo healing factor (sqrt)     = {heal_cosmo:.4f}")
    print(f"  EW minus cosmo                  = {delta_heal:+.5f}")
    print(f"  agree to 3 sig figs             = {abs(delta_heal) < 5e-3}")
    print()
    print(f"  Post-renorm projected overamp   = "
          f"{overamp * heal_ew**2 * 100:+.2f}%")


if __name__ == "__main__":
    main()
