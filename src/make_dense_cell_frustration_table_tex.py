"""Per-regime metric-axiom defect frustration vs.\\ R_00 halo
summary table.

Emits a LaTeX table that consolidates the canonical-physics
ladder result of
\\verb|verify_dense_cell_frustration_canonical_ladder.py|:
the negative-defect triangle share f_{delta < 0} alongside the
per-node halo fraction f_{R_00 < 0} on the same Xi/psi snapshot
configurations. The table documents that the global graph
frustration metric saturates at the continuum (Symanzik 1/N
intercept ~ 1, universal-frustration limit) while the local
energy-density halo fraction stays regime-stable around 0.65-0.79
- the two are structurally distinct observables on the same
lattice.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "verify_dense_cell_frustration_canonical_ladder.json"
OUT = REPO / "paper" / "tables" / "tab_dense_cell_frustration.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    # Order: canonical P5/P5N family by N first, then alt-anchor
    # (P6/P7/P8) family by N. Never lump alt-anchor regimes into the
    # canonical N-ordered ladder.
    _canonical = sorted(
        [r for r in d["per_regime"] if r["regime"].startswith("P5")],
        key=lambda r: r["N"])
    _alt = sorted(
        [r for r in d["per_regime"]
         if r["regime"].startswith(("P6", "P7", "P8"))],
        key=lambda r: r["N"])
    rows = _canonical + _alt
    s = d["summary"]
    nts_inf = s.get("neg_triangle_share_symanzik_intercept", float("nan"))
    fneg_inf = s.get("f_neg_R00_symanzik_intercept", float("nan"))

    lines = []
    A = lines.append
    A(r"\begin{tabular}{@{}l r r r r r@{}}")
    A(r"\toprule")
    A(r"Regime & $N$ & seeds & "
      r"$f_{\delta_{ijk}<0}$ & $f_{R_{00}<0}$ & "
      r"$\rho(\mathrm{cycles})/N$ \\")
    A(r"\midrule")
    for r in rows:
        A(f"$\\mathrm{{{r['regime']}}}$ & {r['N']} & {r['n_seeds']} & "
          f"{r['neg_triangle_share_mean']:.4f} & "
          f"{r['f_neg_R00_mean']:.4f} & "
          f"{r['cycle_density_mean']:.2f} \\\\")
    A(r"\midrule")
    A(f"\\multicolumn{{3}}{{l}}"
      f"{{Symanzik $1/N$ intercept ($N\\to\\infty$)}} & "
      f"{nts_inf:.4f} & {fneg_inf:.4f} & --- \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
