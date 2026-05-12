#!/usr/bin/env python3
"""
Verify the framework-level w(z) chirality-running prediction for the
dark-energy equation of state.

Leading-order System-R closure:
  w_DE(theta_chir) = -1 + eps_sync^4 / gamma_eff(theta_chir)
  gamma_eff(theta) = gamma * cos^2(theta) + alpha_xi * sin^2(theta)

Three benchmark angles:
  theta = 0      (canonical pre-flip):     w = -39/40    = -0.975
  theta = pi/4   (chirality critical):     w = -199/200  = -0.995
  theta = pi/2   (projection-flipped):     w = -1+1/360 ~= -0.99722

CPL w_a prediction:
  At theta = 0:  dw/dtheta = 0   (leading order vanishes)
  At leading O(theta_chir^2):
    Delta w ~ -(eps_sync^4 / gamma^2) * (alpha_xi - gamma) * theta_chir^2
  With theta_chir <~ gamma at present epoch:
    |w_a^framework| <= gamma^2 = 0.01
  Central System-R expectation: w_a ~= 0

Falsifier: a confirmed |w_a| > 0.1 at >5 sigma over the BAO redshift
range would falsify the eps_sync^4/gamma closure together with the
chirality-running structure of the dark-energy equation of state.

Output bundle: outputs/verify_wDE_chirality_running.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "verify_wDE_chirality_running.json"

# System-R primitives
d = 4
N_gen = 3
gamma = 0.1
alpha_xi = 0.9
eps_sync2 = 1 / 20
eps_sync4 = eps_sync2 ** 2  # = 1/400


def gamma_eff(theta: float) -> float:
    return gamma * math.cos(theta) ** 2 + alpha_xi * math.sin(theta) ** 2


def w_DE(theta: float) -> float:
    return -1.0 + eps_sync4 / gamma_eff(theta)


def main() -> int:
    benchmarks = {
        "theta_pre_flip_canonical": {
            "theta": 0.0,
            "w_DE_predicted": w_DE(0.0),
            "w_DE_rational": "-39/40",
            "w_DE_decimal": -39 / 40,
        },
        "theta_chirality_critical_pi_over_4": {
            "theta": math.pi / 4,
            "w_DE_predicted": w_DE(math.pi / 4),
            "w_DE_rational": "-199/200",
            "w_DE_decimal": -199 / 200,
        },
        "theta_projection_flipped_pi_over_2": {
            "theta": math.pi / 2,
            "w_DE_predicted": w_DE(math.pi / 2),
            "w_DE_rational": "-1 + 1/360",
            "w_DE_decimal": -1 + 1 / 360,
        },
    }

    # Leading-order w_a bound
    # Delta w ~ -(eps^4/g^2)*(a-g)*theta_now^2, theta_now <~ gamma
    theta_now_upper = gamma
    delta_w_upper = -(eps_sync4 / gamma ** 2) * (alpha_xi - gamma) * theta_now_upper ** 2
    wa_bound = abs(delta_w_upper / 1.0)  # CPL window |da| ~ 1 over observable past
    # Dimensional upper bound from gamma^2
    wa_dim_bound = gamma ** 2

    # DESI 2024-2025 BAO central (under w0waCDM prior)
    wa_DESI_central = -0.4
    wa_DESI_sigma = 0.1  # rough representative 3-sigma scale on central
    ratio = abs(wa_DESI_central / wa_dim_bound)

    bundle = {
        "criterion": "w(z) chirality-running framework prediction "
                     "and falsifiability against DESI 2024-2025 BAO",
        "systemR_primitives": {
            "d": d, "N_gen": N_gen,
            "gamma": gamma, "alpha_xi": alpha_xi,
            "eps_sync2": eps_sync2, "eps_sync4": eps_sync4,
        },
        "interpolator": "gamma_eff(theta) = gamma cos^2 theta + alpha_xi sin^2 theta",
        "running_formula": "w_DE(theta) = -1 + eps_sync^4 / gamma_eff(theta)",
        "benchmarks": benchmarks,
        "w_a_prediction": {
            "central_systemR_expectation": 0.0,
            "leading_correction_at_theta_gamma": delta_w_upper,
            "dimensional_upper_bound": -wa_dim_bound,
            "dimensional_window": [-wa_dim_bound, +wa_dim_bound],
            "interpretation": "Leading dw/dtheta vanishes at theta=0; "
                              "O(theta_chir^2) correction is bounded by gamma^2.",
        },
        "DESI_2024_2025_BAO_comparison": {
            "central_w_a": wa_DESI_central,
            "sigma_w_a": wa_DESI_sigma,
            "framework_dim_bound": wa_dim_bound,
            "ratio_DESI_to_framework_bound": ratio,
            "framework_predicts_signal_outcomes": [
                "(i) BAO Alcock-Paczynski / fiducial-cosmology systematic",
                "(ii) redshift-window selection effect at z~1 tomographic boundary",
                "(iii) genuine framework falsification if signal >5 sigma in DESI-DR3",
            ],
        },
        "falsifier": "Confirmed |w_a| > 0.1 at >5 sigma over the BAO "
                     "redshift range would falsify eps_sync^4/gamma closure + "
                     "chirality-running structure of w_DE.",
        "headline": "WDE_CHIRALITY_RUNNING_PREDICTION_REGISTERED",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(bundle, fh, indent=2)

    print("w(z) chirality-running framework prediction")
    print("=" * 70)
    for name, b in benchmarks.items():
        print(f"  {name}:")
        print(f"    theta = {b['theta']:.5f}")
        print(f"    w_DE  = {b['w_DE_predicted']:+.6f}  ({b['w_DE_rational']})")
    print("=" * 70)
    print(f"w_a System-R central expectation: 0")
    print(f"w_a framework dim bound:          |w_a| <= gamma^2 = {wa_dim_bound:.4f}")
    print(f"DESI 2024-2025 central w_a:       {wa_DESI_central}")
    print(f"Ratio DESI / framework bound:     {ratio:.1f}x")
    print(f"Headline: {bundle['headline']}")
    print(f"Output: {OUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
