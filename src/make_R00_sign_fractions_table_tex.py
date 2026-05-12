"""Across-seed mean R_00 sign fractions for the four panel regimes.

Documents quantitatively that the visual impression of "much more
red at N=200" in fig:R00_J_multi_N is a sample-size effect: the
positive fraction varies only modestly across the regimes
(approximately 24%-33%), but the absolute count of positive nodes
scales linearly with N.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO.parent
sys.path.insert(0, str(PARENT / "emergent-gr-closure-repro" / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, _name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LAMBDA_T, load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "paper" / "tables" / "tab_R00_sign_fractions.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("P5",     50,  r"$\mathcal{P}_{5}$",       "(a)"),
    ("P5N64",  64,  r"$\mathcal{P}_{5}N_{64}$", "(b)"),
    ("P5N72",  72,  r"$\mathcal{P}_{5}N_{72}$", "(c)"),
    ("P5N84",  84,  r"$\mathcal{P}_{5}N_{84}$", "(d)"),
    ("P5N100", 100, r"$\mathcal{P}_{5}N_{100}$", "(e)"),
    ("P5N128", 128, r"$\mathcal{P}_{5}N_{128}$", "(f)"),
    ("P5N200", 200, r"$\mathcal{P}_{5}N_{200}$", "(g)"),
    ("P5N256", 256, r"$\mathcal{P}_{5}N_{256}$", "(h)"),
    ("P5N300", 300, r"$\mathcal{P}_{5}N_{300}$", "(i)"),
    ("P5N512", 512, r"$\mathcal{P}_{5}N_{512}$", "(j)"),
]


def main():
    rows = []
    for regime, n_lat, label_tex, panel_tag in PANELS:
        p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
        if p is None or not p.exists():
            rows.append((panel_tag, label_tex, n_lat, 0,
                         float("nan"), float("nan"), float("nan"), 0))
            continue
        seeds = (load_snapshots(p, n_lat)
                 if "snapshots" in p.name.lower()
                 else load_canonical(p, n_lat))
        if not seeds:
            rows.append((panel_tag, label_tex, n_lat, 0,
                         float("nan"), float("nan"), float("nan"), 0))
            continue
        f_neg_list, f_pos_list, med_list = [], [], []
        for s in seeds:
            xi_mat = np.asarray(s[0], float).copy()
            psi = np.asarray(s[1])
            k = np.asarray(s[2]) if len(s) > 2 else None
            q = np.asarray(s[3]) if len(s) > 3 else None
            prep = per_seed_galerkin(xi_mat, psi, k, q, n_lat, np)
            r00 = prep["g_00_h"] + LAMBDA_T - prep["t00"]
            f_neg_list.append(float((r00 < 0).mean()))
            f_pos_list.append(float((r00 > 0).mean()))
            med_list.append(float(np.median(r00)))
        f_neg = float(np.mean(f_neg_list))
        f_pos = float(np.mean(f_pos_list))
        med = float(np.mean(med_list))
        n_pos_abs = int(round(f_pos * n_lat))
        rows.append((panel_tag, label_tex, n_lat, len(seeds),
                     f_neg, f_pos, med, n_pos_abs))

    lines = []
    A = lines.append
    A(r"\begin{tabular}{l l r r r r r r}")
    A(r"\toprule")
    A(r"Panel & Regime & $N$ & seeds & "
      r"$f_{R_{00}<0}$ & $f_{R_{00}>0}$ & "
      r"$\mathrm{median}\,R_{00}$ & $f_{>0}\cdot N$ \\")
    A(r"\midrule")
    for panel_tag, label_tex, n_lat, n_seeds, f_neg, f_pos, med, n_pos in rows:
        if not np.isfinite(f_neg):
            A(f"{panel_tag} & {label_tex} & {n_lat} & --- & "
              f"--- & --- & --- & --- \\\\")
            continue
        A(f"{panel_tag} & {label_tex} & {n_lat} & {n_seeds} & "
          f"{f_neg:.3f} & {f_pos:.3f} & "
          f"{med:+.4f} & {n_pos} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
