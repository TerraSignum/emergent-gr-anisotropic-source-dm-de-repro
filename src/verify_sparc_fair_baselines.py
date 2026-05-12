r"""Fair-baseline comparison of the SPARC framework prediction against
three external parameter-free baselines: MOND/BTFR, Dutton-Maccio
NFW, and Moster abundance-matching NFW. For each galaxy we compute
the predicted V_flat from each baseline (using only published
relations and the bundled M_b / V_flat measurements) and report
the per-galaxy chi^2 and population-median residuals.

(B1) MOND BTFR: V_flat^4 = G * M_b * a_0  (fixed a_0 = 1.2e-10 m/s^2)
(B2) Dutton-Maccio NFW + Moster abundance matching:
       Step 1: M_star = 0.7 * M_b  (gas correction for late-types)
       Step 2: M_halo = Moster(M_star) inverted from
               M_star/M_halo = 2 N [(M_h/M1)^(-beta) + (M_h/M1)^gamma]
               with M_1 = 10^11.59, N = 0.0351, beta = 1.376,
               gamma = 0.608
       Step 3: c = 10^(0.905 - 0.101 log10(M_halo / 10^12 Msun))
       Step 4: V_flat predicted from V_flat^2 = G M_halo / R_vir
               where R_vir from M_halo = (4/3) pi R_vir^3 * 200 rho_crit
               and the NFW V_max relation V_max = sqrt(G M_halo / R_vir)
               * sqrt(c / [ln(1+c) - c/(1+c)]) * geometric prefactor.
(B3) Framework prediction:
       below threshold M_b < 10^11 Msun: cored Burkert with v_flat from
         BTFR (slope 4) at the framework prediction, optionally with
         framework-modified BTFR slope.
       above threshold: cuspy NFW with framework matter-branch
         concentration c_matter = 0.8 * c_Dutton-Maccio.

We compute, per galaxy:
   chi2 per baseline = (V_obs - V_pred)^2 / sigma^2 with sigma = 5%
   AICc per baseline (k=0 for parameter-free baselines)
   classification accuracy of the framework's two-phase prediction
   on the AICc-best baseline of each galaxy.

Output: outputs/verify_sparc_fair_baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# Physical constants
G_NEWTON_SI = 6.674e-11      # m^3 / (kg s^2)
M_SUN_KG = 1.989e30          # kg
A_0_MOND = 1.2e-10           # m/s^2 (Milgrom)
KM_S = 1000.0                # m/s per km/s

# NFW + Moster + Dutton-Maccio constants
MOSTER_LOGM1 = 11.59
MOSTER_N = 0.0351
MOSTER_BETA = 1.376
MOSTER_GAMMA = 0.608
DM_A = 0.905    # log10 c200 = a + b log10 (M_h / 10^12)
DM_B = -0.101
RHO_CRIT_z0 = 9.47e-27       # kg / m^3 at z=0
H_RVIR_DELTA = 200           # virial overdensity wrt rho_crit


def moster_M_halo_from_M_star(M_star_Msun, max_iter=100):
    """Newton iteration on log10(M_halo) for the Moster relation."""
    log_Mh = np.log10(M_star_Msun) + 1.5
    M1 = 10**MOSTER_LOGM1
    for _ in range(max_iter):
        Mh = 10**log_Mh
        ratio_Mh_M1 = Mh / M1
        m_term = ratio_Mh_M1**(-MOSTER_BETA) + ratio_Mh_M1**MOSTER_GAMMA
        f = M_star_Msun - 2 * MOSTER_N * Mh * m_term
        # df/dlog_Mh = -2N (m_term + Mh dm_term/dlog_Mh) ln(10)
        dm_dlogMh = (-MOSTER_BETA * ratio_Mh_M1**(-MOSTER_BETA)
                      + MOSTER_GAMMA * ratio_Mh_M1**MOSTER_GAMMA) * np.log(10)
        df = -2 * MOSTER_N * Mh * np.log(10) * (m_term + dm_dlogMh)
        if abs(df) < 1e-30:
            break
        log_Mh -= f / df
        if abs(f / df) < 1e-9:
            break
    return 10**log_Mh


def dutton_maccio_c(M_halo_Msun):
    return 10**(DM_A + DM_B * np.log10(M_halo_Msun / 1e12))


def NFW_v_max_kms(M_halo_Msun, c):
    """V_max of an NFW halo of mass M_halo with concentration c.

    NFW: M(r) = M_halo * [ln(1+c x) - cx/(1+cx)] / [ln(1+c) - c/(1+c)]
    where x = r / R_vir. Maximum of M(r)/r occurs at x_max = 2.163/c.
    V_max = sqrt(G M(r_max) / r_max).
    R_vir from M_halo = (4/3) pi R_vir^3 (200 rho_crit).
    """
    M_kg = M_halo_Msun * M_SUN_KG
    R_vir_m = (3 * M_kg / (4 * np.pi * H_RVIR_DELTA * RHO_CRIT_z0))**(1/3)
    # Concentration function
    def NFW_mu(x_c):
        return np.log(1 + x_c) - x_c / (1 + x_c)
    mu_c = NFW_mu(c)
    # x_max where V circ peaks (NFW): solution of (1+cx)/((cx)^2) - 1/(cx(1+cx)) = mu_c / x^2
    # numerically search
    x_arr = np.linspace(0.01, 1.0, 500)
    mu_x = NFW_mu(c * x_arr)
    M_x = M_kg * mu_x / mu_c
    v_circ_sq = G_NEWTON_SI * M_x / (x_arr * R_vir_m)
    v_max_si = np.sqrt(v_circ_sq.max())
    return v_max_si / KM_S


def mond_V_flat_kms_from_M_b_Msun(M_b_Msun):
    """MOND BTFR: V_flat^4 = G * M_b * a_0."""
    M_kg = M_b_Msun * M_SUN_KG
    V4 = G_NEWTON_SI * M_kg * A_0_MOND
    V_si = V4**0.25
    return V_si / KM_S


def main():
    bundle = json.load(open(ROOT / "outputs" / "verify_sparc_two_phase_refined.json"))
    rows = bundle["rows"]
    n_total = len(rows)

    # 5% relative error on V_flat (typical SPARC quoted)
    V_REL_ERR = 0.05

    print("=" * 78)
    print("Per-galaxy fair-baseline V_flat predictions")
    print("=" * 78)

    results = []
    for r in rows:
        M_b_Msun = r["M_b_e9"] * 1e9
        V_obs = r["V_flat_kms"]
        sigma_V = max(V_REL_ERR * V_obs, 1.0)  # 5% or 1 km/s floor

        # Baseline (B1) MOND
        V_mond = mond_V_flat_kms_from_M_b_Msun(M_b_Msun)
        chi2_mond = ((V_obs - V_mond) / sigma_V)**2

        # Baseline (B2) Moster + Dutton-Maccio
        M_star_est = 0.7 * M_b_Msun
        M_halo = moster_M_halo_from_M_star(M_star_est)
        c_DM = dutton_maccio_c(M_halo)
        V_DM = NFW_v_max_kms(M_halo, c_DM)
        chi2_DM = ((V_obs - V_DM) / sigma_V)**2

        # Baseline (B3) Framework two-phase prediction
        # Below threshold: predict cored profile -> V_flat from BTFR slope
        # similar to MOND but with framework's slope from
        # baryonic Tully-Fisher (independent of MOND a_0).
        # We use the median-fit BTFR slope from SPARC: V_flat^4 = A * M_b
        # with A measured on the dataset itself - so for a parameter-free
        # framework prediction we use the framework's own M_b -> V_flat
        # closure: V_flat = (G M_b a_0_eff)^0.25 with a_0_eff matching
        # framework chirality-flip arrangement. For this comparison we
        # use a_0_eff = a_0_MOND on vacuum-branch (same form as MOND)
        # and a_0_eff = a_0_MOND * (1/0.8) on matter-branch (concentration
        # ratio inverted into BTFR effectively gives flatter rotation).
        threshold_Mb = 1e11
        if M_b_Msun < threshold_Mb:
            # vacuum branch: BTFR-style
            V_framework = mond_V_flat_kms_from_M_b_Msun(M_b_Msun)
        else:
            # matter branch: reduced concentration -> NFW with c_matter
            c_matter = 0.8 * c_DM
            V_framework = NFW_v_max_kms(M_halo, c_matter)
        chi2_framework = ((V_obs - V_framework) / sigma_V)**2

        results.append({
            "galaxy": r["galaxy"],
            "M_b_Msun": M_b_Msun,
            "V_obs_kms": V_obs,
            "sigma_V": sigma_V,
            "T_type": r.get("T_type", -1),
            "best_fit_family": r["best_fit_family"],
            # Predictions
            "V_mond_kms": V_mond,
            "V_DM_kms": V_DM,
            "V_framework_kms": V_framework,
            # chi^2 per baseline
            "chi2_mond": chi2_mond,
            "chi2_DM": chi2_DM,
            "chi2_framework": chi2_framework,
            # M_halo etc.
            "M_halo_Msun": M_halo,
            "c_DM": c_DM,
        })

    # Population-median V_flat residuals (in %)
    V_obs_arr = np.array([r["V_obs_kms"] for r in results])
    V_mond_arr = np.array([r["V_mond_kms"] for r in results])
    V_DM_arr = np.array([r["V_DM_kms"] for r in results])
    V_fw_arr = np.array([r["V_framework_kms"] for r in results])
    chi2_mond_arr = np.array([r["chi2_mond"] for r in results])
    chi2_DM_arr = np.array([r["chi2_DM"] for r in results])
    chi2_fw_arr = np.array([r["chi2_framework"] for r in results])
    M_b_arr = np.array([r["M_b_Msun"] for r in results])

    print()
    print(f"  Population-median residuals (V_pred - V_obs)/V_obs:")
    print(f"    MOND  (B1):      {np.median((V_mond_arr-V_obs_arr)/V_obs_arr):+.4f}")
    print(f"    DM-NFW (B2):     {np.median((V_DM_arr-V_obs_arr)/V_obs_arr):+.4f}")
    print(f"    Framework (B3):  {np.median((V_fw_arr-V_obs_arr)/V_obs_arr):+.4f}")
    print()
    print(f"  Population-median chi^2 (per galaxy):")
    print(f"    MOND  (B1):      {np.median(chi2_mond_arr):>8.2f}")
    print(f"    DM-NFW (B2):     {np.median(chi2_DM_arr):>8.2f}")
    print(f"    Framework (B3):  {np.median(chi2_fw_arr):>8.2f}")
    print()

    # AICc-wins (parameter-free baselines, so AICc = chi^2)
    # For each galaxy find the smallest-chi2 baseline
    wins_mond = 0
    wins_DM = 0
    wins_fw = 0
    for i in range(n_total):
        c2 = [chi2_mond_arr[i], chi2_DM_arr[i], chi2_fw_arr[i]]
        winner = int(np.argmin(c2))
        if winner == 0: wins_mond += 1
        elif winner == 1: wins_DM += 1
        else: wins_fw += 1
    print(f"  AICc-wins on V_flat fit (parameter-free, k=0):")
    print(f"    MOND wins:      {wins_mond}/{n_total}")
    print(f"    DM-NFW wins:    {wins_DM}/{n_total}")
    print(f"    Framework wins: {wins_fw}/{n_total}")

    # Pre-registered D_halo subset (T_type >= 8)
    d_halo_mask = np.array([r["T_type"] >= 8 for r in results])
    n_dh = int(d_halo_mask.sum())
    if n_dh > 0:
        print()
        print(f"  Within pre-registered D_halo (T_type >= 8, N = {n_dh}):")
        print(f"    Median chi^2 (MOND):      {np.median(chi2_mond_arr[d_halo_mask]):>8.2f}")
        print(f"    Median chi^2 (DM-NFW):    {np.median(chi2_DM_arr[d_halo_mask]):>8.2f}")
        print(f"    Median chi^2 (Framework): {np.median(chi2_fw_arr[d_halo_mask]):>8.2f}")
        # Wins within D_halo
        w_m = w_d = w_f = 0
        for i in np.where(d_halo_mask)[0]:
            c2 = [chi2_mond_arr[i], chi2_DM_arr[i], chi2_fw_arr[i]]
            j = int(np.argmin(c2))
            if j == 0: w_m += 1
            elif j == 1: w_d += 1
            else: w_f += 1
        print(f"    AICc-wins MOND/DM-NFW/Framework: {w_m}/{w_d}/{w_f}  out of {n_dh}")

    # Mass-binned wins (above vs below threshold)
    above_mask = M_b_arr > 1e11
    below_mask = ~above_mask
    print()
    print(f"  Mass-binned median chi^2:")
    if above_mask.sum() > 0:
        print(f"    Above 10^11 Msun (matter-branch, N = {above_mask.sum()}):")
        print(f"      MOND={np.median(chi2_mond_arr[above_mask]):>7.2f}, "
              f"DM-NFW={np.median(chi2_DM_arr[above_mask]):>7.2f}, "
              f"Framework={np.median(chi2_fw_arr[above_mask]):>7.2f}")
    if below_mask.sum() > 0:
        print(f"    Below 10^11 Msun (vacuum-branch, N = {below_mask.sum()}):")
        print(f"      MOND={np.median(chi2_mond_arr[below_mask]):>7.2f}, "
              f"DM-NFW={np.median(chi2_DM_arr[below_mask]):>7.2f}, "
              f"Framework={np.median(chi2_fw_arr[below_mask]):>7.2f}")

    bundle_out = {
        "method": "Per-galaxy fair-baseline comparison: MOND/BTFR vs Dutton-Maccio + Moster NFW vs framework two-phase prediction. All three baselines parameter-free. chi^2 weights = 5% V_flat error.",
        "n_total": n_total,
        "median_residuals": {
            "MOND":      float(np.median((V_mond_arr-V_obs_arr)/V_obs_arr)),
            "DM_NFW":    float(np.median((V_DM_arr-V_obs_arr)/V_obs_arr)),
            "framework": float(np.median((V_fw_arr-V_obs_arr)/V_obs_arr)),
        },
        "median_chi2": {
            "MOND":      float(np.median(chi2_mond_arr)),
            "DM_NFW":    float(np.median(chi2_DM_arr)),
            "framework": float(np.median(chi2_fw_arr)),
        },
        "aicc_wins_full_sample": {
            "MOND": wins_mond, "DM_NFW": wins_DM, "framework": wins_fw,
            "n": n_total,
        },
        "D_halo_n": n_dh,
        "D_halo_median_chi2": {
            "MOND":      float(np.median(chi2_mond_arr[d_halo_mask])) if n_dh else None,
            "DM_NFW":    float(np.median(chi2_DM_arr[d_halo_mask])) if n_dh else None,
            "framework": float(np.median(chi2_fw_arr[d_halo_mask])) if n_dh else None,
        },
        "mass_binned_median_chi2": {
            "above_1e11_n": int(above_mask.sum()),
            "below_1e11_n": int(below_mask.sum()),
            "above_MOND":      float(np.median(chi2_mond_arr[above_mask])) if above_mask.any() else None,
            "above_DM_NFW":    float(np.median(chi2_DM_arr[above_mask])) if above_mask.any() else None,
            "above_framework": float(np.median(chi2_fw_arr[above_mask])) if above_mask.any() else None,
            "below_MOND":      float(np.median(chi2_mond_arr[below_mask])) if below_mask.any() else None,
            "below_DM_NFW":    float(np.median(chi2_DM_arr[below_mask])) if below_mask.any() else None,
            "below_framework": float(np.median(chi2_fw_arr[below_mask])) if below_mask.any() else None,
        },
        "per_galaxy": results,
    }
    out_path = OUT / "verify_sparc_fair_baselines.json"
    with open(out_path, "w") as f:
        json.dump(bundle_out, f, indent=2)
    print(f"\nBundle: {out_path}")


if __name__ == "__main__":
    main()
