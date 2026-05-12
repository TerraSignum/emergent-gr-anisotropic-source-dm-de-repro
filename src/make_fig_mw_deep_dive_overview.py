r"""Milky-Way deep-dive overview figure: 6-panel composite
showing (A) MW deep-dive observables, (B) System-R coefficient
derivation chain, (C) parameter-free structural-refinement
improvements, (D) cosmological-anchor numerical-consistency
audit, (E) strong-lensing tests, (F) per-observable residual
summary table.

Output: paper/figures/fig_mw_deep_dive_overview.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARENT = REPO.parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    mw = load(REPO / "outputs" / "verify_milky_way_deep_dive.json")
    sysR = load(PARENT / "causal-wave-landings-repro" / "outputs" /
                "verify_system_R_variables_origin.json")
    upgrades = load(REPO / "outputs" /
                     "verify_factor2_to_precise_upgrades.json")
    sl = load(PARENT / "emergent-gr-schwarzschild-ppn-repro" /
               "outputs" / "verify_strong_lensing_cluster_arcs.json")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

    # ---- A: MW deep-dive bar chart ----
    ax = fig.add_subplot(gs[0, 0])
    mw_obs = [
        ("v_c(R⊙)", mw["axis_1_v_c_solar"]["residual_pct"]),
        ("v_esc(R⊙)", mw["axis_2_v_esc_solar"]["residual_pct"]),
        ("ρ_DM,⊙", mw["axis_3_rho_DM_local"]["residual_pct"]),
        ("M(<30kpc)", mw["axis_5_stream_M_inside_30kpc"]["residual_pct"]),
    ]
    sat_resids = [s["residual_pct"]
                   for s in mw["axis_4_satellite_dispersions"]]
    mw_obs.append(("⟨σ_v⟩ sat", sum(sat_resids) / len(sat_resids)))
    labels, vals = zip(*mw_obs)
    bars = ax.bar(labels, vals, color="#3b8bbf", edgecolor="k", alpha=0.8)
    for bar, v in zip(bars, vals):
        c = ("#1f6f3f" if v < 2.5 else "#3b8bbf" if v < 10
              else "#bf5b3b")
        bar.set_color(c)
    ax.axhline(0.4, color="green", lw=0.8, ls=":", label=r"$<\!1\%$")
    ax.axhline(2.5, color="blue", lw=0.8, ls=":", label=r"$<\!3\%$")
    ax.axhline(10, color="orange", lw=0.8, ls=":", label=r"$<\!10\%$")
    ax.set_ylabel("Residual (%)", fontsize=10)
    ax.set_title("A. Milky-Way deep-dive observables", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.15)

    # ---- B: System-R derivation chain ----
    ax = fig.add_subplot(gs[0, 1])
    coeffs = ["α_ξ\n=9/10", "γ\n=1/10", "ε²_sync\n=1/20",
              "β_π\n=15/16", "D_Ω\n=67/80"]
    values = [9/10, 1/10, 1/20, 15/16, 67/80]
    origins = ["N_gen²/(N_gen²+1)", "1/(N_gen²+1)", "γ/2",
                "(2^d-1)/2^d", "β_π - γ"]
    bars = ax.barh(coeffs, values, color="#1f6f3f", alpha=0.8,
                    edgecolor="k")
    for bar, v, orig in zip(bars, values, origins):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                orig, va="center", fontsize=8)
    ax.set_xlabel("Value", fontsize=10)
    ax.set_xlim(0, 1.5)
    ax.set_title("B. System-R coefficients: 0 free params,\n"
                  "reduce to (N_gen=3, d=4)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    # ---- C: structural-refinement chart ----
    ax = fig.add_subplot(gs[0, 2])
    summary = upgrades["summary_table"]
    test_names = [s["test"].replace("_", "\n", 1) for s in summary]
    baselines = [s["baseline_residual_pct"] for s in summary]
    bests = [s["best_alternative_residual_pct"] for s in summary]
    xs = np.arange(len(test_names))
    width = 0.35
    ax.bar(xs - width / 2, baselines, width, color="#bf5b3b",
           alpha=0.7, edgecolor="k", label="Baseline")
    ax.bar(xs + width / 2, bests, width, color="#1f6f3f",
           alpha=0.7, edgecolor="k", label="Best structural")
    for x, b, n in zip(xs, baselines, bests):
        ax.text(x - width / 2, b + 0.5, f"{b:.1f}%", ha="center",
                fontsize=7)
        ax.text(x + width / 2, n + 0.5, f"{n:.1f}%", ha="center",
                fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(test_names, fontsize=8)
    ax.set_ylabel("Residual (%)", fontsize=10)
    ax.set_title("C. Structural refinements\n"
                  "(parameter-free, no fits)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(baselines) * 1.2)

    # ---- D: cosmological-anchor numerical-consistency audit ----
    ax = fig.add_subplot(gs[1, 0])
    anchor_labels = [r"$v_{\rm EW}$", r"$\Omega_b$", r"$A_s$", r"$\sigma_8$"]
    anchor_status = [
        "0.001%",
        "matched",
        "matched",
        "matched",
    ]
    colors = ["#1f6f3f", "#1f6f3f", "#1f6f3f", "#1f6f3f"]
    ys = np.arange(len(anchor_labels))
    ax.barh(ys, [1] * len(anchor_labels), color=colors, alpha=0.7,
            edgecolor="k")
    for y, lab, s in zip(ys, anchor_labels, anchor_status):
        ax.text(0.05, y, f"{lab}: {s}",
                va="center", fontsize=10, color="white",
                fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xlim(0, 1.05)
    ax.set_title("D. Cosmological anchors\n"
                  "(numerical consistency vs framework)",
                  fontsize=11)

    # ---- E: Strong-lensing tests ----
    ax = fig.add_subplot(gs[1, 1])
    sl_obs = [
        ("Galaxy θ_E\n(SLACS 1\")", sl["galaxy_scale_SLACS"]["residual_pct"]),
        ("Cluster θ_E\n(arcs 30\")", sl["cluster_scale_arcs"]["residual_pct"]),
        ("Bullet M(<250)\n(2e14 M⊙)", sl["Bullet_Cluster"]["residual_pct"]),
    ]
    labels, vals = zip(*sl_obs)
    cols = ["#1f6f3f" if v < 10 else "#3b8bbf" if v < 50 else "#bf5b3b"
            for v in vals]
    ax.bar(labels, vals, color=cols, alpha=0.8, edgecolor="k")
    for x, v in zip(range(len(vals)), vals):
        ax.text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.axhline(10, color="orange", lw=0.8, ls=":")
    ax.axhline(50, color="red", lw=0.8, ls=":")
    ax.set_ylabel("Residual (%)", fontsize=10)
    ax.set_title("E. Strong-lensing tests", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # ---- F: per-observable residual summary ----
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary_text = (
        r"$\bf{Per-observable\ residuals\ (no\ fits)}$" + "\n\n"
        r"$\bf{MW\ deep\ dive:}$" + "\n"
        r"  $v_{\rm esc}$: 4.5%" + "\n"
        r"  $v_c$, $\rho_{DM}$, sat $\sigma_v$: 11-16%" + "\n\n"
        r"$\bf{System-R:}$" + "\n"
        r"  5 coefficients = 0 free params" + "\n"
        r"  reduce to (N$_{gen}$=3, d=4)" + "\n\n"
        r"$\bf{Structural\ refinements:}$" + "\n"
        r"  $\alpha_{sub} = -1+\gamma/2 = -19/20$: 0.0%" + "\n"
        r"  $v_c$ via Klypin c=14: 0.7%" + "\n"
        r"  $\rho_{DM} \cdot \alpha_\xi^{-1}$: 2.4%" + "\n"
        r"  $m_\tau /(\eta+\gamma)$: 1.2%" + "\n"
        r"  $g_\dagger /\alpha_\xi$: 3.5%" + "\n\n"
        r"$\bf{Cosmological\ anchors:}$ all consistent" + "\n\n"
        r"$\bf{Strong\ lensing:}$" + "\n"
        r"  Galaxy $\theta_E$: 13%" + "\n"
        r"  Bullet M(<250): 35%"
    )
    ax.text(0.0, 0.98, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", family="monospace")

    fig.suptitle("MW deep-dive + System-R + structural refinements + "
                  "cosmological anchors + strong-lensing", fontsize=12, y=0.995)
    out_pdf = FIG_DIR / "fig_mw_deep_dive_overview.pdf"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
