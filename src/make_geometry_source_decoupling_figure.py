"""Figure: rho(X,T), rho(X,d), rho(T,d) vs N on the within-P5 ladder
with Symanzik 1/N^2 continuum-limit fits overlaid.

Reads: outputs/verify_halo_geometry_vs_source_decomposition.json
Writes: paper/figures/fig_geometry_source_decoupling_vs_N.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
JSON = REPO / "outputs" / "verify_halo_geometry_vs_source_decomposition.json"
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG_DIR / "fig_geometry_source_decoupling_vs_N.pdf"

P5_REGIMES = ("P5", "P5N64", "P5N72", "P5N84", "P5N100", "P5N128", "P5N200", "P5N256", "P5N300", "P5N512")
ALT_PREFIXES = ("P6", "P7", "P8")

data = json.loads(JSON.read_text())
P5 = sorted([r for r in data["per_regime"] if r["regime"] in P5_REGIMES],
            key=lambda r: r["N"])
# Order full row list: canonical P5/P5N family by N first, then alt-anchor
# (P6/P7/P8) family by N — never lump alt-anchor regimes into the
# canonical ladder.
_canonical = sorted(
    [r for r in data["per_regime"] if r["regime"] in P5_REGIMES],
    key=lambda r: r["N"])
_alt = sorted(
    [r for r in data["per_regime"]
     if r["regime"].startswith(ALT_PREFIXES)],
    key=lambda r: r["N"])
all_rows = _canonical + _alt

N_p5 = np.array([r["N"] for r in P5], dtype=float)
rXT_p5 = np.array([r["rho_X_vs_T"] for r in P5])
rXd_p5 = np.array([r["rho_X_vs_d"] for r in P5])
rTd_p5 = np.array([r["rho_T_vs_d"] for r in P5])

N_all = np.array([r["N"] for r in all_rows], dtype=float)
rXT_all = np.array([r["rho_X_vs_T"] for r in all_rows])
rXd_all = np.array([r["rho_X_vs_d"] for r in all_rows])
rTd_all = np.array([r["rho_T_vs_d"] for r in all_rows])
labels_all = [r["regime"] for r in all_rows]
mask_p5 = np.array([lab in P5_REGIMES for lab in labels_all])


def symanzik2(N, y):
    x = 1.0 / N**2
    A = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    ss_res = ((y - pred)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    return c[0], c[1], 1 - ss_res / ss_tot


y_XT, a_XT, R2_XT = symanzik2(N_p5, rXT_p5)
y_Xd, a_Xd, R2_Xd = symanzik2(N_p5, rXd_p5)
y_Td, a_Td, R2_Td = symanzik2(N_p5, rTd_p5)

N_smooth = np.linspace(40, 540, 200)
fit_XT = y_XT + a_XT / N_smooth**2
fit_Xd = y_Xd + a_Xd / N_smooth**2
fit_Td = y_Td + a_Td / N_smooth**2


fig, ax = plt.subplots(figsize=(8.6, 4.6))

# Canonical P5/P5N ladder (filled), alt-anchor cross-checks (open)
ax.plot(N_all[mask_p5], rXT_all[mask_p5], "o", color="#1f3b6f",
        markersize=7,
        label=r"$\rho(X,T^{\Xi}_{00})$ on $\mathcal{P}_{5}/\mathcal{P}_{5}N$",
        zorder=4)
ax.plot(N_all[~mask_p5], rXT_all[~mask_p5], "o", color="#1f3b6f",
        markersize=7, markerfacecolor="white", zorder=3,
        label=r"$\rho(X,T^{\Xi}_{00})$ alt-anchor (open)")

ax.plot(N_all[mask_p5], rXd_all[mask_p5], "s", color="#a45a5a",
        markersize=7,
        label=r"$\rho(X,d)$ on $\mathcal{P}_{5}/\mathcal{P}_{5}N$",
        zorder=4)
ax.plot(N_all[~mask_p5], rXd_all[~mask_p5], "s", color="#a45a5a",
        markersize=7, markerfacecolor="white", zorder=3)

ax.plot(N_all[mask_p5], rTd_all[mask_p5], "^", color="#4f9c5e",
        markersize=7,
        label=r"$\rho(T^{\Xi}_{00},d)$ on $\mathcal{P}_{5}/\mathcal{P}_{5}N$",
        zorder=4)
ax.plot(N_all[~mask_p5], rTd_all[~mask_p5], "^", color="#4f9c5e",
        markersize=7, markerfacecolor="white", zorder=3)

ax.plot(N_smooth, fit_XT, "-", color="#1f3b6f", linewidth=1.4,
        alpha=0.7,
        label=(r"Symanzik fit: $\rho(X,T^{\Xi}_{00})_{\infty}"
               f"={y_XT:+.3f}$ ($R^{{2}}={R2_XT:.2f}$)"))
ax.plot(N_smooth, fit_Xd, "--", color="#a45a5a", linewidth=1.4,
        alpha=0.7,
        label=(r"Symanzik fit: $\rho(X,d)_{\infty}"
               f"={y_Xd:+.3f}$ ($R^{{2}}={R2_Xd:.2f}$)"))
ax.plot(N_smooth, fit_Td, ":", color="#4f9c5e", linewidth=1.6,
        alpha=0.7,
        label=(r"Symanzik fit: $\rho(T^{\Xi}_{00},d)_{\infty}"
               f"={y_Td:+.3f}$ ($R^{{2}}={R2_Td:.2f}$)"))

ax.axhline(0.0, color="gray", linewidth=0.6)
ax.set_xlabel(r"lattice size $N$", fontsize=10)
ax.set_ylabel(r"signed Spearman $\rho$", fontsize=10)
ax.set_title(
    "Geometry-source coupling halves toward the continuum;\n"
    r"halo signature on both $X\!=\!G_{00}\!+\!\Lambda_{t}$ and "
    r"$T^{\Xi}_{00}$ survives at $N\!\to\!\infty$",
    fontsize=10)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
          fontsize=8, frameon=True, framealpha=0.9)
ax.grid(linewidth=0.3, alpha=0.5)
ax.set_xscale("log")
ax.set_xlim(40, 600)
ax.set_xticks([50, 100, 200, 300, 512])
ax.set_xticklabels(["50", "100", "200", "300", "512"])
ax.set_ylim(-0.6, 0.85)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT}")
print(f"Symanzik 1/N^2 within-P5 limits:")
print(f"  rho(X,T)_inf  = {y_XT:+.4f}  (R^2 = {R2_XT:.3f})")
print(f"  rho(X,d)_inf  = {y_Xd:+.4f}  (R^2 = {R2_Xd:.3f})")
print(f"  rho(T,d)_inf  = {y_Td:+.4f}  (R^2 = {R2_Td:.3f})")
