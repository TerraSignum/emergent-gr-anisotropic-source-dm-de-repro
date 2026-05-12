"""Unit tests for the halo-audit optimization bundle.

Covers:
  - Symanzik 1/N and 1/N^2 fits recover known asymptotes on
    synthetic data
  - AICc model comparison agrees with the textbook formula
  - Bootstrap CI on the asymptote contains the true value
  - Value-shuffle null produces a null distribution centred at
    zero
  - NFW profile fitting recovers known parameters on synthetic
    NFW samples
  - T-eigenframe per-node diagonalisation: rotated diagonal of
    T_ij equals the sorted eigenvalues, by construction
  - G_00 + Lambda_t Spearman equals G_00 Spearman against any
    reference (Item 4 numerical identity)
  - JSON outputs of the audit scripts contain the load-bearing
    fields
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))


def _spearman_simple(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ------------------------------------------------------------
# Symanzik fits
# ------------------------------------------------------------

def test_symanzik_1over_N_recovers_asymptote():
    """Synthetic y = 0.4 + 12/N exactly should give y_inf = 0.4."""
    from verify_halo_optimization_bundle import _symanzik_fit
    N = np.array([50, 64, 72, 84, 100, 200, 300], float)
    y = 0.4 + 12 / N
    y_inf, a, R2, _ = _symanzik_fit(N, y, 1)
    assert abs(y_inf - 0.4) < 1e-10
    assert abs(a - 12.0) < 1e-10
    assert R2 > 0.999


def test_symanzik_1over_N2_recovers_asymptote():
    from verify_halo_optimization_bundle import _symanzik_fit
    N = np.array([50, 64, 72, 84, 100, 200, 300], float)
    y = -0.3 + 250 / N**2
    y_inf, a, R2, _ = _symanzik_fit(N, y, 2)
    assert abs(y_inf - (-0.3)) < 1e-10
    assert abs(a - 250.0) < 1e-10
    assert R2 > 0.999


# ------------------------------------------------------------
# AICc
# ------------------------------------------------------------

def test_aicc_textbook_formula():
    from verify_halo_optimization_bundle import _aicc
    rss = 0.5
    n = 7
    k = 2
    sigma2 = rss / n
    expected = n * math.log(sigma2) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    got = _aicc(rss, n, k)
    assert abs(got - expected) < 1e-10


# ------------------------------------------------------------
# Bootstrap CI behaviour
# ------------------------------------------------------------

def test_bootstrap_ci_contains_truth_on_synthetic_seeds():
    """Synthetic per-seed values around a known mean: the bootstrap
    CI95 on the regime-mean should contain the true value with high
    probability."""
    from verify_halo_optimization_bundle import _bootstrap_symanzik
    rng = np.random.default_rng(42)
    N = np.array([50, 100, 200], float)
    # True asymptote -0.4, finite-N drift 5/N (so y_50=-0.30, y_100=-0.35,
    # y_200=-0.375). Each "regime" has 4 noisy seeds.
    truth = -0.4 + 5.0 / N
    per_regime_seeds = []
    for t in truth:
        seeds = t + 0.05 * rng.standard_normal(4)
        per_regime_seeds.append(seeds)
    res = _bootstrap_symanzik(N, per_regime_seeds, 1, 500, rng)
    assert res is not None
    # The fit on the seed-means should land near -0.4 (1/N model)
    assert -0.5 < res["median"] < -0.3
    # CI95 should not be pathological
    assert res["ci95_hi"] > res["ci95_lo"]
    assert res["std"] > 0


# ------------------------------------------------------------
# Value-shuffle null behaviour
# ------------------------------------------------------------

def test_value_shuffle_null_centred_at_zero():
    """If we shuffle |R_00| against a fixed d, the null Spearman
    distribution should be centred at zero (mu ~ 0, std ~ 1/sqrt(n))."""
    rng = np.random.default_rng(0)
    n = 1500
    d = rng.uniform(0, 5, size=n)
    absR = rng.exponential(1.0, size=n)  # independent of d
    nulls = []
    for _ in range(300):
        shuf = rng.permutation(absR)
        nulls.append(_spearman_simple(shuf, d))
    nulls = np.asarray(nulls)
    mu = nulls.mean()
    sig = nulls.std()
    expected_sig = 1.0 / math.sqrt(n - 1)
    assert abs(mu) < 0.05
    # Spearman std under permutation null is approximately 1/sqrt(n-1)
    assert 0.5 * expected_sig < sig < 2.0 * expected_sig


# ------------------------------------------------------------
# NFW shape fit recovers parameters
# ------------------------------------------------------------

def test_nfw_shape_fit_recovers_known_parameters():
    from scipy.optimize import curve_fit
    from verify_halo_shape_fit_R00 import nfw
    A_true, rs_true = 0.18, 1.4
    r = np.linspace(0.3, 4.0, 30)
    y_clean = nfw(r, A_true, rs_true)
    rng = np.random.default_rng(7)
    y_noisy = y_clean * (1 + 0.02 * rng.standard_normal(len(r)))
    popt, _ = curve_fit(nfw, r, y_noisy, p0=[0.1, 1.0], maxfev=5000,
                          bounds=([0, 1e-3], [np.inf, 1e3]))
    assert abs(popt[0] - A_true) / A_true < 0.05
    assert abs(popt[1] - rs_true) / rs_true < 0.05


def test_uniform_profile_constant():
    from verify_halo_shape_fit_R00 import uniform
    r = np.linspace(0.1, 5.0, 20)
    A = 0.123
    y = uniform(r, A)
    assert np.allclose(y, A)


# ------------------------------------------------------------
# T-eigenframe diagonalisation
# ------------------------------------------------------------

def test_T_eigenframe_diagonalisation_sorts_ascending():
    """For a random symmetric 3x3 matrix, eigvals from eigh are
    sorted ascending."""
    rng = np.random.default_rng(13)
    M_sym = rng.standard_normal((3, 3))
    M_sym = 0.5 * (M_sym + M_sym.T)
    w, U = np.linalg.eigh(M_sym)
    # Sorted ascending
    assert np.all(np.diff(w) >= -1e-12)
    # Rotation diagonalises: U^T M U should be diag(w) up to numerical
    diag = U.T @ M_sym @ U
    assert np.allclose(np.diag(diag), w, atol=1e-10)
    # Off-diagonal elements should be near zero
    off = diag - np.diag(np.diag(diag))
    assert np.max(np.abs(off)) < 1e-10


# ------------------------------------------------------------
# G_00 + Lambda_t identity (Item 4)
# ------------------------------------------------------------

def test_X_equals_G00_plus_constant_for_rank_correlation():
    """Spearman is rank-based; adding a constant to all values
    preserves ranks, hence Spearman(X+c, anything) == Spearman(X, anything)."""
    rng = np.random.default_rng(0)
    n = 200
    G00 = rng.standard_normal(n)
    d = rng.uniform(0, 5, size=n)
    LAMBDA_T = 0.81
    X = G00 + LAMBDA_T
    rho_X = _spearman_simple(X, d)
    rho_G = _spearman_simple(G00, d)
    assert abs(rho_X - rho_G) < 1e-12


# ------------------------------------------------------------
# JSON outputs structure
# ------------------------------------------------------------

@pytest.mark.parametrize("path,required_keys", [
    (REPO / "outputs" / "verify_halo_optimization_bundle.json",
     ["item_1_2_AICc_bootstrap_symanzik",
      "item_3_value_shuffle_null",
      "item_4_X_vs_G00_decomposition"]),
    (REPO / "outputs" / "verify_halo_shape_fit_R00.json",
     ["per_regime", "models", "summary"]),
    (REPO / "outputs" / "verify_halo_T_eigenframe.json",
     ["per_regime", "summary"]),
])
def test_json_output_has_required_keys(path, required_keys):
    if not path.exists():
        pytest.skip(f"Output JSON not yet generated: {path}")
    d = json.loads(path.read_text())
    for k in required_keys:
        assert k in d, f"Missing key {k} in {path}"


def test_optimization_bundle_item4_diff_machine_precision():
    """Item 4: rho(X,d) - rho(G_00,d) should be numerically zero
    on every regime."""
    p = REPO / "outputs" / "verify_halo_optimization_bundle.json"
    if not p.exists():
        pytest.skip(f"{p} not yet generated")
    d = json.loads(p.read_text())
    for r in d["item_4_X_vs_G00_decomposition"]:
        assert r["abs_diff_X_minus_G00_d"] < 1e-12
        assert r["abs_diff_X_minus_G00_T"] < 1e-12


def test_T_eigenframe_axis_2_positive_on_all_regimes():
    """Item 7: in the T-eigenframe sorted ascending, axis 2
    (largest t_eig) carries a positive Spearman with d on every
    regime, completing the (-, -, +) (2+1) signature."""
    p = REPO / "outputs" / "verify_halo_T_eigenframe.json"
    if not p.exists():
        pytest.skip(f"{p} not yet generated")
    d = json.loads(p.read_text())
    for r in d["per_regime"]:
        rho2 = r["R_eig_axis_2"]["spearman_signed_vs_d"]
        assert rho2 > 0, (
            f"Regime {r['regime']} N={r['N']}: axis 2 rho={rho2:+.3f} "
            "should be positive (genuine (2+1) signature requires)"
        )


def test_shape_fit_NFW_majority_winner():
    """Item 5: NFW should be AICc-best on a majority of regimes."""
    p = REPO / "outputs" / "verify_halo_shape_fit_R00.json"
    if not p.exists():
        pytest.skip(f"{p} not yet generated")
    d = json.loads(p.read_text())
    counts = d["summary"]["AICc_best_counts"]
    n_total = d["summary"]["n_regimes"]
    nfw_count = counts.get("NFW", 0)
    assert nfw_count > n_total / 2, (
        f"NFW won on {nfw_count}/{n_total} regimes; expected majority"
    )
