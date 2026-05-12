"""Generate LaTeX AICc-comparison table from shape-fit JSON."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSON = REPO / "outputs" / "verify_halo_shape_fit_R00.json"
OUT = REPO / "paper" / "tables" / "tab_R00_shape_fit_aicc.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)

d = json.loads(JSON.read_text())
profiles = ["NFW", "Burkert", "cored_NFW", "Plummer",
             "exponential", "power_law", "uniform"]
prof_short = {"NFW": "NFW", "Burkert": "Burkert", "cored_NFW": "cored",
               "Plummer": "Plummer", "exponential": "exp.",
               "power_law": "p-law", "uniform": "unif."}

lines = []
A = lines.append
A(r"\begin{tabular}{l r r r r r r r r}")
A(r"\toprule")
header = ["Regime", "$N$"] + [prof_short[p] for p in profiles]
A(" & ".join(header) + r" \\")
A(r"\midrule")
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
for r in _canonical + _alt:
    name = r["regime"].replace("_", r"\_")
    row = [name, f"${r['N']}$"]
    best = r["best_AICc"]
    for prof in profiles:
        f = r["fits"].get(prof, {})
        if "delta_AICc" in f:
            v = f["delta_AICc"]
            if prof == best:
                cell = r"$\mathbf{0.0}$"
            else:
                cell = f"${v:.1f}$"
        else:
            cell = "---"
        row.append(cell)
    A(" & ".join(row) + r" \\")
A(r"\bottomrule")
A(r"\end{tabular}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
