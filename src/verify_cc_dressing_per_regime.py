"""Recompute the cosmological-constant eight-layer dressing closure
from the bundled upstream GCC-07 output and report a clean
per-regime table. Standalone: reads only data/gcc07_cc_residual_closure.json.

Stand 2026-05-11: H197 retracted (see closure JSON
h197_retraction_2026_05_11 block); the chain is L1..L6 + H195 + H196.

For each canonical regime the audit returns
  rho_final         : final dressed lattice vacuum-energy density [GeV^4]
  rho_observed      : Planck-2018 observed value [GeV^4]
  ratio             : rho_final / rho_observed
  log10_residual    : log10(ratio) (closure residual in orders of magnitude)
  total_orders      : orders-of-magnitude reduction relative to naive M_Pl^4
  closed            : True iff |log10_residual| < 0.5

This script does NOT recompute the layer-by-layer pipeline; the
upstream pipeline is at
src/worldformula/experiments/run_gap_closure_cosmology.py
(parent corpus). It re-derives the closure verdict from the
bundled output so the verdict is independently checkable from
the data file alone.

Output: outputs/cc_dressing_per_regime.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "data" / "gcc07_cc_residual_closure.json"
OUT = REPO / "outputs" / "cc_dressing_per_regime.json"

CLOSURE_THRESHOLD_LOG10 = 0.5  # |log10(rho_final/rho_obs)| < 0.5 = within factor ~3


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(IN.read_text(encoding="utf-8"))
    rho_obs = raw.get("pg4_rho_observed_GeV4")

    regimes = []
    for k in raw:
        if k.startswith("pg4_rho_final_GeV4_"):
            reg = k.removeprefix("pg4_rho_final_GeV4_")
            regimes.append(reg)
    regimes.sort()

    rows = []
    for reg in regimes:
        rho_f = raw.get(f"pg4_rho_final_GeV4_{reg}")
        ratio = rho_f / rho_obs if (rho_f and rho_obs) else float("nan")
        log10_res = math.log10(ratio) if ratio > 0 else float("nan")
        orders = raw.get(f"pg4_total_orders_explained_{reg}")
        closed_flag = abs(log10_res) < CLOSURE_THRESHOLD_LOG10
        rows.append({
            "regime": reg,
            "rho_final_GeV4": rho_f,
            "rho_observed_GeV4": rho_obs,
            "ratio_final_over_observed": ratio,
            "log10_residual_orders": log10_res,
            "total_orders_explained": orders,
            "closed_within_factor_~3": closed_flag,
            "absolute_relative_deviation_pct": (
                abs(ratio - 1.0) * 100 if ratio == ratio else float("nan")),
        })

    out = {
        "method": "cosmological-constant eight-layer dressing per-regime closure",
        "input_data_file": str(IN.relative_to(REPO)),
        "rho_observed_GeV4_planck2018": rho_obs,
        "closure_threshold_log10": CLOSURE_THRESHOLD_LOG10,
        "per_regime": rows,
        "summary": {
            "n_regimes_in_pipeline_output": len(rows),
            "n_closed_within_factor_~3": sum(1 for r in rows
                                              if r["closed_within_factor_~3"]),
            "n_open": sum(1 for r in rows
                          if not r["closed_within_factor_~3"]),
        },
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print(f"{'regime':>10s}  {'rho_final [GeV^4]':>18s}  {'ratio':>10s}  "
          f"{'log10 res':>10s}  {'orders':>8s}  closed?")
    print("-" * 80)
    for r in rows:
        print(f"{r['regime']:>10s}  "
              f"{r['rho_final_GeV4']:>18.4e}  "
              f"{r['ratio_final_over_observed']:>10.4g}  "
              f"{r['log10_residual_orders']:>+10.3f}  "
              f"{r['total_orders_explained']!s:>8s}  "
              f"{r['closed_within_factor_~3']!s}")


if __name__ == "__main__":
    main()
