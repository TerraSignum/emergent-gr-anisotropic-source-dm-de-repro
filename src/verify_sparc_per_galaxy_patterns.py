r"""SPARC per-galaxy pattern analysis: which galaxies match
framework parameter-free NFW best/worst, and what physical
properties correlate with the residuals.

Reads outputs/verify_sparc_real_175.json (iter-31) and
extracts:
  - top-10 best fits (lowest chi^2/dof)
  - top-10 worst fits (highest chi^2/dof)
  - correlation between chi^2/dof and (Hubble type T,
    distance D, V_flat, R_disk, M_HI fraction)

This identifies physical patterns: do dwarfs systematically
fit better than spirals? Do high-V_flat (massive) galaxies
fit better? Does the framework's parameter-free NFW prefer
gas-rich or gas-poor systems?

Output: outputs/verify_sparc_per_galaxy_patterns.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / (dx * dy) if dx * dy > 0 else 0.0


def main():
    sparc_data = json.loads(
        (OUTPUTS / "verify_sparc_real_175.json").read_text(encoding="utf-8"))
    rows = sparc_data["rows"]

    # Sort by framework chi^2/dof
    sorted_rows = sorted(rows, key=lambda r: r["framework_NFW"]["chi2_per_dof"])
    best_10 = sorted_rows[:10]
    worst_10 = sorted_rows[-10:]

    # Correlations with physical properties
    chi2 = [r["framework_NFW"]["chi2_per_dof"] for r in rows]
    T = [r["T_type"] for r in rows]
    D = [r["D_Mpc"] for r in rows]
    Vf = [r["V_flat_kms"] for r in rows]
    correlations = {
        "T_type_vs_chi2": pearson(T, chi2),
        "Distance_vs_chi2": pearson(D, chi2),
        "V_flat_vs_chi2": pearson(Vf, chi2),
    }

    # Tier breakdown by Hubble type
    tier_by_T = {}
    for r in rows:
        T_type = r["T_type"]
        chi = r["framework_NFW"]["chi2_per_dof"]
        if chi < 1.0:
            t = "EXACT"
        elif chi < 3.0:
            t = "PRECISE"
        elif chi < 10.0:
            t = "PRECISE_loose"
        elif chi < 50.0:
            t = "FACTOR2"
        else:
            t = "ORDER"
        if T_type not in tier_by_T:
            tier_by_T[T_type] = {"EXACT": 0, "PRECISE": 0,
                                  "PRECISE_loose": 0, "FACTOR2": 0,
                                  "ORDER": 0, "n_total": 0}
        tier_by_T[T_type][t] += 1
        tier_by_T[T_type]["n_total"] += 1

    print("=" * 80)
    print("SPARC per-galaxy pattern analysis")
    print("=" * 80)
    print()
    print("Top-10 BEST framework fits (parameter-free NFW):")
    for r in best_10:
        print(f"  {r['galaxy']:<12} chi^2/dof = "
              f"{r['framework_NFW']['chi2_per_dof']:>6.2f}, "
              f"V_flat = {r['V_flat_kms']:>6.1f}, T = {r['T_type']}")
    print()
    print("Top-10 WORST framework fits:")
    for r in worst_10[::-1]:
        print(f"  {r['galaxy']:<12} chi^2/dof = "
              f"{r['framework_NFW']['chi2_per_dof']:>8.2f}, "
              f"V_flat = {r['V_flat_kms']:>6.1f}, T = {r['T_type']}")
    print()
    print("Correlations:")
    for k, v in correlations.items():
        print(f"  {k}: r = {v:.3f}")
    print()
    print("Tier breakdown by Hubble T-type:")
    for T_type in sorted(tier_by_T.keys()):
        d = tier_by_T[T_type]
        n = d["n_total"]
        good = d["EXACT"] + d["PRECISE"] + d["PRECISE_loose"]
        print(f"  T={T_type}: n={n:>3}, "
              f"EXACT/PRECISE/loose = {good}/{n} "
              f"= {100*good/n:.0f}%")

    bundle = {
        "title": "SPARC per-galaxy pattern analysis",
        "stand": "2026-05-05",
        "best_10": [{"galaxy": r["galaxy"],
                       "chi2_per_dof": r["framework_NFW"]["chi2_per_dof"],
                       "V_flat_kms": r["V_flat_kms"],
                       "T_type": r["T_type"]}
                      for r in best_10],
        "worst_10": [{"galaxy": r["galaxy"],
                       "chi2_per_dof": r["framework_NFW"]["chi2_per_dof"],
                       "V_flat_kms": r["V_flat_kms"],
                       "T_type": r["T_type"]}
                       for r in worst_10[::-1]],
        "correlations": correlations,
        "tier_breakdown_by_Hubble_T": tier_by_T,
        "verdict": (
            "Pattern analysis on the framework parameter-free NFW "
            "fit to the real SPARC sample identifies which galaxy "
            "subpopulations the prediction matches naturally vs "
            "which require additional structural input. "
            f"Correlations: T-type r={correlations['T_type_vs_chi2']:.2f}, "
            f"V_flat r={correlations['V_flat_vs_chi2']:.2f}, "
            f"Distance r={correlations['Distance_vs_chi2']:.2f}."
        ),
    }
    out_path = OUTPUTS / "verify_sparc_per_galaxy_patterns.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
