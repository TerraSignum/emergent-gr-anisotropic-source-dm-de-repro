r"""Milky Way deep-dive: v_c, v_esc, rho_DM_local, satellite
dispersions, stream constraints, M_200_MW.

External anchors (Gaia DR3, RAVE, APOGEE, literature):
  R_solar       = 8.122 +/- 0.031 kpc      (Gravity Coll. 2019)
  v_c(R_sol)    = 232.8 +/- 3 km/s          (Eilers+ 2019, Mroz+ 2019)
  v_esc(R_sol)  = 528 +/- 25 km/s            (Piffl+ 2014 RAVE)
  rho_DM_local  = 0.40 +/- 0.05 GeV/cm^3    (Read 2014; Pato-Iocco 2015)
                = 0.0107 +/- 0.0014 M_sun/pc^3
  M_200_MW      = (1.0 +/- 0.3) x 10^12 M_sun (Callingham+ 2019)
  c_200_MW      = 12 +/- 2                   (Bullock-Kolatt + Gaia)
  R_200_MW      = ~200 kpc

Satellite dwarf-galaxy velocity dispersions:
  Sculptor      sigma_los ~ 9 km/s     (Battaglia+ 2008)
  Draco         sigma_los ~ 9 km/s
  Carina        sigma_los ~ 7 km/s
  Fornax        sigma_los ~ 11 km/s
  Sextans       sigma_los ~ 7 km/s

Tidal streams (Sgr, GD-1, Pal 5):
  M_dyn(<r_apo) inferred constraint on enclosed MW mass

Output: outputs/verify_milky_way_deep_dive.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

PI = math.pi
G_ASTRO = 4.302e-6  # (km/s)^2 kpc / M_sun
RHO_CRIT_MSUN_KPC3 = 140.0


def nfw_M_enclosed(r, rho_s, r_s):
    x = r / r_s
    return 4 * PI * rho_s * r_s ** 3 * (math.log(1 + x) - x / (1 + x))


def nfw_rho(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


def nfw_v_c(r, rho_s, r_s):
    return math.sqrt(G_ASTRO * nfw_M_enclosed(r, rho_s, r_s) / r)


def nfw_phi(r, rho_s, r_s):
    """NFW gravitational potential phi(r) = -4 pi G rho_s r_s^3 / r * log(1+r/r_s)."""
    return -4 * PI * G_ASTRO * rho_s * r_s ** 3 * math.log(1 + r / r_s) / r


def setup_mw_halo(M_200=1.0e12, c=12.0):
    R_200 = (3 * M_200 / (4 * PI * 200 * RHO_CRIT_MSUN_KPC3)) ** (1/3)
    r_s = R_200 / c
    rho_s = M_200 / (4 * PI * r_s ** 3 * (math.log(1 + c) - c / (1 + c)))
    return rho_s, r_s, R_200


def predict_v_c_solar(rho_s, r_s, R_solar=8.122, M_disk=5e10, R_d=3.0):
    """v_c(R_sol) including DM + exponential disk."""
    v_DM = nfw_v_c(R_solar, rho_s, r_s)
    x = R_solar / R_d
    M_disk_in = M_disk * (1 - math.exp(-x) * (1 + x))
    v_disk = math.sqrt(G_ASTRO * M_disk_in / R_solar)
    return v_DM, v_disk, math.sqrt(v_DM ** 2 + v_disk ** 2)


def predict_v_esc(rho_s, r_s, R_solar=8.122, M_disk=5e10, R_d=3.0):
    """v_esc^2 = -2 phi(R_sol). Use NFW phi + disk approx."""
    phi_NFW = nfw_phi(R_solar, rho_s, r_s)
    # Disk contribution: -G M_disk / R_sol approx
    phi_disk = -G_ASTRO * M_disk / R_solar
    phi_total = phi_NFW + phi_disk
    return math.sqrt(-2 * phi_total)


def predict_rho_DM_local(rho_s, r_s, R_solar=8.122):
    """rho_DM at R_sol from NFW profile."""
    return nfw_rho(R_solar, rho_s, r_s)  # M_sun / kpc^3


def predict_satellite_sigma(M_dwarf, R_dwarf_pc=300):
    """Satellite velocity dispersion via Jeans equation, isothermal
    isotropic approx: sigma^2 = G M_dyn / (3 R_eff)."""
    # M_dwarf assumed total dynamical mass within R_dwarf
    R_kpc = R_dwarf_pc / 1e3
    sigma2 = G_ASTRO * M_dwarf / (3 * R_kpc)
    return math.sqrt(sigma2)


def main():
    out_path = OUTPUTS / "verify_milky_way_deep_dive.json"
    print("=" * 90)
    print("Milky Way deep-dive: v_c, v_esc, rho_DM_local, satellites, M_200")
    print("=" * 90)
    print()

    # Setup framework MW halo
    M_200_pred = 1.0e12
    c_pred = 12.0
    rho_s, r_s, R_200 = setup_mw_halo(M_200_pred, c_pred)
    print(f"NFW halo: M_200 = {M_200_pred:.2e} M_sun, c = {c_pred}, "
          f"r_s = {r_s:.2f} kpc, R_200 = {R_200:.1f} kpc")

    R_sol = 8.122
    M_disk = 5e10
    R_d = 3.0

    # 1. v_c(R_sol)
    v_DM, v_disk, v_total = predict_v_c_solar(rho_s, r_s, R_sol, M_disk, R_d)
    v_c_anchor = 232.8
    res_v_c = abs(v_total - v_c_anchor) / v_c_anchor * 100
    print(f"\n1. Rotation speed v_c(R_sol):")
    print(f"   v_DM = {v_DM:.1f} km/s, v_disk = {v_disk:.1f} km/s, "
          f"v_total = {v_total:.1f} km/s")
    print(f"   anchor (Gaia DR3) = {v_c_anchor} +/- 3 km/s, "
          f"residual = {res_v_c:.2f}% [{tier(res_v_c)}]")

    # 2. v_esc(R_sol)
    v_esc_pred = predict_v_esc(rho_s, r_s, R_sol, M_disk, R_d)
    v_esc_anchor = 528.0
    res_esc = abs(v_esc_pred - v_esc_anchor) / v_esc_anchor * 100
    print(f"\n2. Escape speed v_esc(R_sol):")
    print(f"   predicted = {v_esc_pred:.1f} km/s")
    print(f"   anchor (RAVE/Gaia) = {v_esc_anchor} +/- 25 km/s, "
          f"residual = {res_esc:.2f}% [{tier(res_esc)}]")

    # 3. rho_DM(R_sol)
    rho_DM_kpc3 = predict_rho_DM_local(rho_s, r_s, R_sol)
    rho_DM_pc3 = rho_DM_kpc3 / 1e9
    rho_DM_anchor_pc3 = 0.0107
    res_rho = abs(rho_DM_pc3 - rho_DM_anchor_pc3) / rho_DM_anchor_pc3 * 100
    print(f"\n3. Local DM density:")
    print(f"   predicted = {rho_DM_pc3:.4f} M_sun/pc^3 "
          f"= {rho_DM_pc3 * 37.96:.3f} GeV/cm^3")
    print(f"   anchor = 0.40 +/- 0.05 GeV/cm^3, "
          f"residual = {res_rho:.2f}% [{tier(res_rho)}]")

    # 4. Satellite dispersions
    satellites = [
        ("Sculptor", 1e7, 280, 9.2),
        ("Draco", 2e7, 200, 9.1),
        ("Carina", 1e7, 250, 6.6),
        ("Fornax", 5e7, 700, 11.7),
        ("Sextans", 1e7, 700, 7.9),
    ]
    print(f"\n4. Satellite velocity dispersions (Jeans-equation prediction):")
    sat_rows = []
    for name, M_dyn, R_eff, sigma_obs in satellites:
        sigma_pred = predict_satellite_sigma(M_dyn, R_eff)
        res = abs(sigma_pred - sigma_obs) / sigma_obs * 100
        sat_rows.append({
            "name": name,
            "M_dyn_Msun_assumed": M_dyn,
            "R_eff_pc": R_eff,
            "sigma_predicted_kms": sigma_pred,
            "sigma_observed_kms": sigma_obs,
            "residual_pct": res,
            "tier": tier(res),
        })
        print(f"   {name:<10}: pred={sigma_pred:5.1f}, obs={sigma_obs:5.1f}, "
              f"residual={res:.1f}% [{tier(res)}]")

    # 5. Tidal stream constraint: enclosed mass at apogalacticon ~30 kpc
    r_apo = 30.0
    M_inside_pred = nfw_M_enclosed(r_apo, rho_s, r_s) + M_disk
    M_inside_anchor = 2.5e11  # Sgr stream constraint
    res_stream = abs(M_inside_pred - M_inside_anchor) / M_inside_anchor * 100
    print(f"\n5. Stream constraint M_MW(<{r_apo} kpc):")
    print(f"   predicted = {M_inside_pred:.2e} M_sun")
    print(f"   anchor (Sgr stream) = {M_inside_anchor:.2e} M_sun, "
          f"residual = {res_stream:.2f}% [{tier(res_stream)}]")

    bundle = {
        "title": "Milky Way deep-dive: rotation, escape, local density, satellites, streams",
        "stand": "2026-05-05",
        "literature": [
            "Gravity Collaboration 2019 (R_solar)",
            "Eilers+ 2019, Mroz+ 2019 (v_c)",
            "Piffl+ 2014 RAVE (v_esc)",
            "Read 2014 (rho_DM_local)",
            "Callingham+ 2019 (M_200_MW)",
            "Battaglia+ 2008 (satellite dispersions)",
            "Belokurov+ 2014 (Sgr stream constraint)",
        ],
        "framework_NFW_halo": {
            "M_200_Msun": M_200_pred, "c_200": c_pred,
            "r_s_kpc": r_s, "R_200_kpc": R_200,
            "rho_s_Msun_kpc3": rho_s,
        },
        "axis_1_v_c_solar": {
            "v_DM_kms": v_DM, "v_disk_kms": v_disk,
            "v_total_kms": v_total,
            "v_c_anchor_kms": v_c_anchor,
            "residual_pct": res_v_c, "tier": tier(res_v_c),
        },
        "axis_2_v_esc_solar": {
            "v_esc_predicted_kms": v_esc_pred,
            "v_esc_anchor_kms": v_esc_anchor,
            "residual_pct": res_esc, "tier": tier(res_esc),
        },
        "axis_3_rho_DM_local": {
            "rho_predicted_Msun_pc3": rho_DM_pc3,
            "rho_predicted_GeV_cm3": rho_DM_pc3 * 37.96,
            "rho_anchor_Msun_pc3": rho_DM_anchor_pc3,
            "residual_pct": res_rho, "tier": tier(res_rho),
        },
        "axis_4_satellite_dispersions": sat_rows,
        "axis_5_stream_M_inside_30kpc": {
            "M_inside_predicted": M_inside_pred,
            "M_inside_Sgr_anchor": M_inside_anchor,
            "residual_pct": res_stream, "tier": tier(res_stream),
        },
    }
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


def tier(r):
    return ("EXACT" if r < 0.4 else "PRECISE" if r < 2.5 else
            "PRECISE_loose" if r < 10 else "FACTOR2" if r < 50 else "ORDER")


if __name__ == "__main__":
    main()
