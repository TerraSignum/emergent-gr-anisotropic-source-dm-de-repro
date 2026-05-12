"""Compute Layer 5 (gravitational fraction E_geom/E_res) from
snapshots.npz for canonical-physics regimes whose d1.json carries
the reduced 33-key summary only and therefore lacks the 1178-key
E_geom and E_res entries used by the cosmological-constant
nine-layer dressing pipeline.

The energy components are computed using the same definitions as
the reference world-formula pipeline
(\\verb|src/worldformula/residual/residual_energy.py|):

  E_cons(a) = sum_b xi_ab |psi_a - psi_b|^2 / (d_ab^2 + 1e-6)
  E_geom(a) = (curv(a) - median(curv))^2,  curv = rho - L*rho/<d>
  E_rec(a)  = a_K mean(K_a) + a_Q (1 - mean(Q_a)),  a_K=1.0, a_Q=0.5
  E_res(a)  = E_cons(a) + E_geom(a) + E_rec(a)
  d_ab      = -ell_0 log(clip(xi_ab, eps, 1)) with ell_0 = 1, eps = 1e-6

Per-regime aggregates follow the same C4-summary convention used in
\\verb|outputs_multi_priority_campaign| (median of mean-per-seed):

  E_geom_regime = median_seed(mean_a E_geom(a))
  E_res_regime  = median_seed(mean_a E_res(a))
  L5            = log10(E_geom / E_res)

Output: outputs/layer5_extended_from_snapshots.json
"""
from __future__ import annotations

import json
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
    load_canonical, load_snapshots)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "layer5_extended_from_snapshots.json"

# Canonical P5/P5N N-ordered ladder (alt-anchor regimes are reported
# separately as cross-anchor consistency checks and are not part of
# this canonical-ladder layer-5 audit).
LADDER = [
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
]

ELL_0 = 1.0
DIST_EPS = 1.0e-6
DIV_EPS = 1.0e-6
A_K = 1.0
A_Q = 0.5


def _distance_matrix(xi):
    d = -ELL_0 * np.log(np.clip(xi, DIST_EPS, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def _xi_curvature(xi):
    rho = np.mean(xi, axis=1)
    deg = np.sum(xi, axis=1)
    L = np.diag(deg) - xi
    lap_rho = L @ rho
    distances = _distance_matrix(xi)
    n = xi.shape[0]
    scale = float(np.mean(distances + np.eye(n)))
    return rho - lap_rho / max(scale, 1.0e-12)


def _e_cons(xi, psi):
    distances = _distance_matrix(xi)
    psi_arr = np.asarray(psi, dtype=complex)
    if psi_arr.ndim == 1:
        diff_sq = np.abs(psi_arr[:, None] - psi_arr[None, :]) ** 2
    else:
        # For multi-component psi (rare in this codebase), reduce per node
        diff = psi_arr[:, None, :] - psi_arr[None, :, :]
        diff_sq = np.sum(np.abs(diff) ** 2, axis=-1)
    return np.sum(xi * diff_sq / (distances ** 2 + DIV_EPS), axis=1)


def _e_geom(xi):
    curv = _xi_curvature(xi)
    return (curv - np.median(curv)) ** 2


def _e_rec(k_field, q_field):
    k_loc = np.mean(np.asarray(k_field, dtype=float), axis=1)
    q_loc = np.mean(np.asarray(q_field, dtype=float), axis=1)
    return A_K * k_loc + A_Q * (1.0 - q_loc)


def _process_regime(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, PARENT / "emergent-gr-closure-repro")
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat) if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))[:max_seeds]
    if not seeds:
        return None
    eg_means, er_means = [], []
    eg_per_seed, er_per_seed = [], []
    for xi_mat, psi, k_field, q_field in seeds:
        xi = np.asarray(xi_mat, dtype=float)
        np.fill_diagonal(xi, 1.0)
        try:
            eg_a = _e_geom(xi)
            ec_a = _e_cons(xi, psi)
            er_a = _e_rec(k_field, q_field)
            er_total = ec_a + eg_a + er_a
        except Exception as exc:  # noqa: BLE001
            print(f"  seed skipped: {exc}")
            continue
        eg_means.append(float(np.mean(eg_a)))
        er_means.append(float(np.mean(er_total)))
        eg_per_seed.append({
            "E_cons_mean": float(np.mean(ec_a)),
            "E_geom_mean": float(np.mean(eg_a)),
            "E_rec_mean":  float(np.mean(er_a)),
            "E_res_mean":  float(np.mean(er_total)),
        })
    if not eg_means:
        return None
    eg = float(np.median(eg_means))
    er = float(np.median(er_means))
    return {
        "regime": regime,
        "N": int(n_lat),
        "n_seeds": len(eg_means),
        "E_geom": eg,
        "E_res": er,
        "layer5_log10_E_geom_over_E_res": (
            float(np.log10(eg / er))
            if eg > 0 and er > 0 else None),
        "per_seed": eg_per_seed,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f'{"regime":<8} {"N":>4} {"seeds":>5} {"E_geom":>14}'
          f' {"E_res":>14} {"L5_log10":>10}')
    for regime, n_lat in LADDER:
        r = _process_regime(regime, n_lat)
        if r is None:
            print(f'{regime:<8} {n_lat:>4} {"--":>5} {"--":>14} '
                  f'{"--":>14} {"missing":>10}')
            continue
        rows.append(r)
        l5 = r["layer5_log10_E_geom_over_E_res"]
        l5_str = f'{l5:+.3f}' if l5 is not None else "--"
        print(f'{r["regime"]:<8} {r["N"]:>4} {r["n_seeds"]:>5} '
              f'{r["E_geom"]:>14.3e} {r["E_res"]:>14.3e} '
              f'{l5_str:>10}')

    out = {
        "method": ("Layer 5 (E_geom/E_res) computed directly from "
                   "snapshots.npz using residual_energy_components "
                   "definitions of worldformula.residual.residual_energy"),
        "definitions": {
            "E_cons":
                "sum_b xi_ab |psi_a - psi_b|^2 / (d_ab^2 + 1e-6)",
            "E_geom":
                "(curv(a) - median(curv))^2 with curv = rho - L*rho/<d>",
            "E_rec":
                "a_K mean(K_a) + a_Q (1 - mean(Q_a)),  a_K=1.0, a_Q=0.5",
            "E_res":
                "E_cons + E_geom + E_rec",
            "regime_aggregate":
                "median over seeds of mean-per-node",
        },
        "ladder": [(r["regime"], r["N"]) for r in rows],
        "per_regime": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
