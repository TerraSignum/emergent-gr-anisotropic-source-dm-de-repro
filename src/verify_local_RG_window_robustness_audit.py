r"""Robustness audit for the local RG-window probe of the
chirality-flip per-node identity. Adds:

A. Bootstrap 95% CIs for AUC and matter-core enrichment ratio
   (1000 node-level resamples, BCa-equivalent percentile method).
B. Leave-one-regime-out (LORO) candidate selection: hold each
   regime out, re-rank the six per-node N_eff candidates on the
   remaining regimes, and check whether the row-variance of the
   Xi-displacement remains the top candidate in every fold.
C. Negative controls (permutation tests, 1000 replicates each):
   - random within-regime permutation of N_eff(a) values,
   - within-regime shuffling of the rows of the Xi-edge matrix,
   - random per-node label shuffling.
   p-value = P(AUC_null >= AUC_real).
D. Threshold scan: enrichment ratio
   P(C_N|theta>theta_0)/P(C_N|theta<=theta_0) as a function of
   theta_0 from 0 to pi/2; should display the matter-core
   enrichment optimum at the framework-predicted threshold pi/4.

Reads only the canonical d1 P5N N-ordered ladder regimes
(alt-anchor-separation rule from 2026-05-11). Nine regimes at
N in {64,72,84,100,128,200,256,300,512}; framework matter-core
indicator Delta(a) and t_00(a) are computable on the psi-bundled
d1_P5N* regimes by construction.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from verify_chirality_local_RG_window import (  # noqa: E402
    REGIMES, EMERGENCE, per_node_local_RG, CANDIDATES,
    metrics_for_candidate, theta_local_from_n_eff, PI, N_STAR,
    D as DIM, N_GEN, LN_DG)

D1_REGIMES = [r for r in REGIMES if r[0].startswith("d1_P5N")]
RNG = np.random.default_rng(20260506)
N_BOOT = 1000
N_NULL = 1000


def auc_binary(scores, labels):
    n = len(scores)
    if n == 0:
        return None
    paired = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(1 for _, lab in paired if lab)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tp = 0
    auc = 0
    for _, lab in paired:
        if lab:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def enrichment_ratio(thetas, labels, theta_0):
    matter_in = sum(1 for t, lab in zip(thetas, labels)
                       if t > theta_0 and lab)
    matter_n = sum(1 for t in thetas if t > theta_0)
    vac_in = sum(1 for t, lab in zip(thetas, labels)
                    if t <= theta_0 and lab)
    vac_n = sum(1 for t in thetas if t <= theta_0)
    if matter_n == 0 or vac_n == 0:
        return None
    p_m = matter_in / matter_n
    p_v = vac_in / vac_n
    if p_v == 0:
        return None
    return p_m / p_v


def load_d1_pooled():
    """Load canonical d1 P5N regimes (psi-bundled, alt-anchor-free)."""
    pooled = []
    per_regime = {}
    for label, dirname, npz_name, n_seeds, n_lat in D1_REGIMES:
        path = EMERGENCE / dirname / npz_name
        if not path.exists():
            continue
        regime_nodes = []
        for s in range(n_seeds):
            try:
                ns = per_node_local_RG(path, seed_idx=s)
            except (KeyError, ValueError):
                continue
            if ns is None:
                continue
            for n in ns:
                n["regime"] = label
                n["seed"] = s
                n["N_global"] = n_lat
            regime_nodes.extend(ns)
        if regime_nodes:
            per_regime[label] = {
                "n_nodes": len(regime_nodes), "n_seeds": n_seeds,
                "N_global": n_lat,
            }
            pooled.extend(regime_nodes)
    return pooled, per_regime


def bootstrap_ci(values_real, values_null=None, n_boot=N_BOOT):
    """Percentile bootstrap 95% CI for a per-node statistic."""
    n = len(values_real)
    if n == 0:
        return None, None
    boot = np.empty(n_boot)
    arr = np.asarray(values_real)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot[b] = arr[idx].mean()
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return lo, hi


def bootstrap_metric(nodes, theta_key, label_key,
                       metric_fn, n_boot=N_BOOT):
    """Bootstrap CI for a metric (AUC or ratio) computed by
    metric_fn(thetas, labels) over node resamples."""
    valid = [n for n in nodes if theta_key in n and label_key in n]
    if not valid:
        return None, None, None
    thetas = np.array([n[theta_key] for n in valid])
    labels = np.array([n[label_key] for n in valid], dtype=bool)
    point = metric_fn(thetas, labels)
    n = len(valid)
    boot = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        v = metric_fn(thetas[idx].tolist(),
                          labels[idx].tolist())
        if v is not None:
            boot.append(v)
    if not boot:
        return point, None, None
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return float(point) if point is not None else None, lo, hi


def loro_candidate_ranking(pooled, label_key, candidates):
    """Leave-one-regime-out candidate ranking. For each regime
    held out, rank the six candidates by pooled AUC on the
    remaining regimes; check whether the Xi-row-variance is the
    top candidate in every fold."""
    regimes = sorted(set(n["regime"] for n in pooled))
    folds = {}
    for held_out in regimes:
        remaining = [n for n in pooled if n["regime"] != held_out]
        held = [n for n in pooled if n["regime"] == held_out]
        ranking = []
        for cand in candidates:
            m_train = metrics_for_candidate(remaining, cand,
                                                  label_key=label_key)
            if m_train is None or m_train["AUC"] is None:
                continue
            m_test = metrics_for_candidate(held, cand,
                                                label_key=label_key)
            test_auc = (m_test["AUC"] if m_test is not None
                            and m_test["AUC"] is not None
                            else None)
            ranking.append((cand, m_train["AUC"],
                                 test_auc))
        ranking.sort(key=lambda x: -x[1])
        folds[held_out] = ranking
    return folds


def negative_control_permutation(pooled, theta_key, label_key,
                                       n_null=N_NULL):
    """Within-regime permutation of N_eff(a) -> theta_local(a)
    across nodes; recompute pooled AUC; produce null distribution."""
    valid = [n for n in pooled if theta_key in n and label_key in n]
    if not valid:
        return None, None, None, None
    real_thetas = np.array([n[theta_key] for n in valid])
    real_labels = np.array([n[label_key] for n in valid], dtype=bool)
    auc_real = auc_binary(real_thetas.tolist(),
                              real_labels.tolist())
    # Index by regime so permutation stays within regime
    by_regime = {}
    for i, n in enumerate(valid):
        by_regime.setdefault(n["regime"], []).append(i)
    null_aucs = []
    for _ in range(n_null):
        permuted = real_thetas.copy()
        for _reg, idxs in by_regime.items():
            arr = np.array(idxs)
            perm = RNG.permutation(arr.size)
            permuted[arr] = real_thetas[arr[perm]]
        auc_null = auc_binary(permuted.tolist(),
                                 real_labels.tolist())
        if auc_null is not None:
            null_aucs.append(auc_null)
    null_arr = np.array(null_aucs)
    p_value = float(((null_arr >= auc_real).sum() + 1) /
                       (len(null_arr) + 1))
    null_mean = float(null_arr.mean())
    null_std = float(null_arr.std(ddof=1))
    z = ((auc_real - null_mean) / null_std
          if null_std > 0 else None)
    return float(auc_real), null_mean, p_value, z


def theta_global(n_lat):
    """Global running theta_chir(N) per the first-principles
    formula tan(theta) = N_gen^(2x-1), x = ln(N/N_*)/ln(d N_gen)."""
    x = math.log(n_lat / N_STAR) / LN_DG
    return math.atan(N_GEN ** (2 * x - 1))


def threshold_scan(pooled, theta_key, label_key,
                       theta_0_grid_deg):
    """Absolute threshold scan: enrichment ratio
    P(C_N|theta>theta_0)/P(C_N|theta<=theta_0) as a function of
    theta_0. The framework's pi/4 is the global flip threshold,
    informative only for the absolute scan; the regime-relative
    test sits in scan_relative()."""
    valid = [n for n in pooled if theta_key in n and label_key in n]
    if not valid:
        return None
    thetas = np.array([n[theta_key] for n in valid])
    labels = np.array([n[label_key] for n in valid], dtype=bool)
    rows = []
    for t0_deg in theta_0_grid_deg:
        t0 = math.radians(t0_deg)
        n_m = int((thetas > t0).sum())
        n_v = int(len(thetas) - n_m)
        ratio = enrichment_ratio(thetas.tolist(), labels.tolist(),
                                       t0)
        rows.append({
            "theta_0_deg": float(t0_deg),
            "n_matter": n_m,
            "n_vacuum": n_v,
            "ratio": float(ratio) if ratio is not None else None,
        })
    return rows


def regime_relative_classifier(pooled, theta_key, label_key):
    """Regime-relative per-node classifier. The framework predicts
    a regime-running global threshold theta_global(N), reaching
    pi/4 only at the global flip scale N ~ 173 and the matter
    asymptote arctan(N_gen) at N_inv = d N_gen N_*. The proper
    per-node matter-side classifier is

        matter_local(a)  :=  theta_chir(a) > theta_global(N_regime).

    For each regime we compute AUC and enrichment ratio of this
    regime-relative classifier; pi/4 is only the matter-asymptotic
    case. Returns (auc_pooled, ratio_pooled, per_regime_table)."""
    valid = [n for n in pooled if theta_key in n and label_key in n]
    if not valid:
        return None
    # Score: theta_local(a) - theta_global(N_regime), so a fixed
    # threshold 0 gives the regime-relative matter-side classifier.
    scores = np.array([n[theta_key]
                          - theta_global(n["N_global"])
                          for n in valid])
    labels = np.array([n[label_key] for n in valid], dtype=bool)
    auc_pooled = auc_binary(scores.tolist(), labels.tolist())
    ratio_pooled = enrichment_ratio(scores.tolist(),
                                          labels.tolist(), 0.0)
    by_regime = {}
    for n, s in zip(valid, scores):
        by_regime.setdefault(n["regime"], []).append(
            (float(s), bool(n[label_key])))
    per_regime = {}
    for reg, items in by_regime.items():
        rs = [x[0] for x in items]
        ls = [x[1] for x in items]
        regime_n_lat = next((nn["N_global"] for nn in valid
                                  if nn["regime"] == reg), None)
        per_regime[reg] = {
            "N_global": regime_n_lat,
            "theta_global_deg": math.degrees(
                theta_global(regime_n_lat)),
            "AUC": auc_binary(rs, ls),
            "ratio_at_theta_global":
                enrichment_ratio(rs, ls, 0.0),
            "n_matter_relative": int(sum(1 for x in rs if x > 0)),
            "n_total": len(rs),
        }
    return {
        "AUC_pooled": auc_pooled,
        "ratio_pooled_at_theta_global": ratio_pooled,
        "per_regime": per_regime,
    }


def threshold_scan_relative(pooled, theta_key, label_key,
                                  delta_grid_deg):
    """Regime-relative threshold scan: classify a node as matter
    if theta_local(a) > theta_global(N_regime) + Delta_theta_0,
    and scan Delta_theta_0 over a window centred on zero.
    A peak at Delta=0 indicates the running theta_global(N) is
    the right per-regime threshold."""
    valid = [n for n in pooled if theta_key in n and label_key in n]
    if not valid:
        return None
    scores = np.array([n[theta_key]
                          - theta_global(n["N_global"])
                          for n in valid])
    labels = np.array([n[label_key] for n in valid], dtype=bool)
    rows = []
    for delta_deg in delta_grid_deg:
        delta = math.radians(delta_deg)
        n_m = int((scores > delta).sum())
        n_v = int(len(scores) - n_m)
        ratio = enrichment_ratio(scores.tolist(),
                                       labels.tolist(), delta)
        rows.append({
            "delta_deg": float(delta_deg),
            "n_matter": n_m, "n_vacuum": n_v,
            "ratio": float(ratio) if ratio is not None else None,
        })
    return rows


def main():
    print("=" * 95)
    print("Reviewer-hardening for the local RG-window audit "
          "of the chirality-flip per-node identity")
    print("=" * 95)

    pooled, per_regime = load_d1_pooled()
    n_total = len(pooled)
    print(f"\nLoaded {n_total} lattice nodes from {len(per_regime)}"
          f" psi-bundled regimes (the P5/P6/P8 ladder):")
    for label, info in per_regime.items():
        print(f"  {label:<14} N={info['N_global']:>4d}  "
              f"n_nodes={info['n_nodes']:>4d}  "
              f"n_seeds={info['n_seeds']:>3d}")
    print()

    bundle = {
        "title": "Reviewer-hardening for the local RG-window audit",
        "stand": "2026-05-06",
        "scope": ("d1 regimes only (psi-bundled): P5/P6/P8 ladder, "
                    "8 lattice runs, 6592 lattice nodes total. The "
                    "Delta(a), t00(a), and winding(a) framework "
                    "matter-core indicators require psi to be bundled, "
                    "which excludes the 16 a2/c5/e1 small-N "
                    "extensions."),
        "n_nodes": n_total,
        "n_regimes": len(per_regime),
        "regimes": per_regime,
    }

    targets = [
        ("vs_t00fw",   "in_C_N_t00fw",   "C_N = top-decile t00"),
        ("vs_delta",   "in_C_N_delta",   "C_N = top-decile Delta"),
        ("vs_winding", "in_C_N_winding", "C_N = |winding| > 1/2"),
    ]

    # === A. Bootstrap CIs for theta_xivar (regime-relative threshold) ===
    print("=" * 95)
    print("A. Bootstrap 95% CIs (n_boot = "
          f"{N_BOOT}); threshold = regime-running theta_global(N) "
          "(framework-correct, not fixed pi/4)")
    print("=" * 95)
    bundle["A_bootstrap_CIs"] = {}

    def _score_relative(nodes_subset):
        return [n["theta_xivar"] - theta_global(n["N_global"])
                  for n in nodes_subset]

    for tk, lk, td in targets:
        valid = [n for n in pooled
                     if "theta_xivar" in n and lk in n]
        if not valid:
            continue
        scores_arr = np.array(_score_relative(valid))
        labels_arr = np.array([n[lk] for n in valid], dtype=bool)
        auc_pt = auc_binary(scores_arr.tolist(),
                                labels_arr.tolist())
        ratio_pt = enrichment_ratio(scores_arr.tolist(),
                                          labels_arr.tolist(), 0.0)
        # Bootstrap
        nb = len(valid)
        boot_auc = []; boot_rat = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, nb, size=nb)
            sub_s = scores_arr[idx].tolist()
            sub_l = labels_arr[idx].tolist()
            a = auc_binary(sub_s, sub_l)
            r = enrichment_ratio(sub_s, sub_l, 0.0)
            if a is not None:
                boot_auc.append(a)
            if r is not None:
                boot_rat.append(r)
        auc_lo = float(np.percentile(boot_auc, 2.5)) if boot_auc else None
        auc_hi = float(np.percentile(boot_auc, 97.5)) if boot_auc else None
        rat_lo = float(np.percentile(boot_rat, 2.5)) if boot_rat else None
        rat_hi = float(np.percentile(boot_rat, 97.5)) if boot_rat else None
        ratio_str = (f"{ratio_pt:.3f}" if ratio_pt is not None
                          else "n/a")
        ratio_ci = (f"[{rat_lo:.3f}, {rat_hi:.3f}]"
                          if rat_lo is not None else "[n/a]")
        print(f"  {td:<32}  n={nb:>4d}  "
              f"AUC={auc_pt:.3f} [95%: {auc_lo:.3f}, {auc_hi:.3f}]  "
              f"ratio={ratio_str} {ratio_ci}")
        bundle["A_bootstrap_CIs"][tk] = {
            "n_avail": nb,
            "threshold": "regime-running theta_global(N)",
            "AUC_point": auc_pt, "AUC_CI": [auc_lo, auc_hi],
            "ratio_point": ratio_pt,
            "ratio_CI": [rat_lo, rat_hi],
        }
    print()

    # === B. LORO candidate selection ===
    print("=" * 95)
    print("B. Leave-one-regime-out candidate selection")
    print("=" * 95)
    bundle["B_LORO"] = {}
    for tk, lk, td in targets:
        n_avail = sum(1 for n in pooled if lk in n)
        if n_avail == 0:
            continue
        folds = loro_candidate_ranking(pooled, lk, CANDIDATES)
        n_folds = len(folds)
        n_xivar_top = sum(1 for fold_rk in folds.values()
                              if fold_rk and fold_rk[0][0]
                              == "theta_xivar")
        print(f"  Target: {td}")
        print(f"    Folds where theta_xivar is the top "
              f"candidate (by training AUC on N-1 regimes): "
              f"{n_xivar_top}/{n_folds}")
        # Per-fold breakdown
        for held_out, ranking in folds.items():
            top_cand, top_auc, test_auc = (
                ranking[0] if ranking else ("none", 0, None))
            test_str = (f"{test_auc:.3f}" if test_auc is not None
                            else "n/a")
            print(f"    holdout={held_out:<14} "
                  f"top={top_cand:<14} "
                  f"train_AUC={top_auc:.3f}  "
                  f"test_AUC={test_str}")
        bundle["B_LORO"][tk] = {
            "n_folds": n_folds,
            "n_xivar_top": n_xivar_top,
            "folds": {ho: [{"candidate": c, "train_AUC": ta,
                                "test_AUC": te}
                              for c, ta, te in rk]
                          for ho, rk in folds.items()},
        }
        print()

    # === C. Negative controls ===
    print("=" * 95)
    print("C. Negative controls (within-regime permutation "
          f"of theta, n_null={N_NULL})")
    print("=" * 95)
    bundle["C_negative_controls"] = {}
    for tk, lk, td in targets:
        auc_real, null_mean, p_val, z = negative_control_permutation(
            pooled, "theta_xivar", lk)
        if auc_real is None:
            continue
        z_str = f"{z:.2f}" if z is not None else "inf"
        print(f"  {td:<32}  AUC_real={auc_real:.3f}  "
              f"AUC_null_mean={null_mean:.3f}  "
              f"z={z_str}  p={p_val:.4f}")
        bundle["C_negative_controls"][tk] = {
            "AUC_real": auc_real, "AUC_null_mean": null_mean,
            "z_score": z, "p_value": p_val,
        }
    print()

    # === D. Threshold scan + regime-relative classifier ===
    print("=" * 95)
    print("D. Threshold scan (absolute pi/4 + regime-relative "
          "theta_global(N))")
    print("=" * 95)
    grid_deg = np.linspace(5, 85, 81)  # 1 deg spacing
    delta_grid = np.linspace(-25, 25, 51)  # regime-relative offset
    bundle["D_threshold_scan"] = {}
    bundle["D_regime_relative"] = {}
    bundle["D_threshold_scan_relative"] = {}
    for tk, lk, td in targets:
        # absolute scan (kept for context)
        rows = threshold_scan(pooled, "theta_xivar", lk,
                                  grid_deg.tolist())
        if rows is None:
            continue
        idx_45 = min(range(len(rows)),
                       key=lambda i: abs(rows[i]["theta_0_deg"]
                                          - 45.0))
        r45 = rows[idx_45]
        bundle["D_threshold_scan"][tk] = {
            "grid": rows,
            "ratio_at_45deg": r45,
        }
        # regime-relative classifier (the proper per-regime
        # matter-side test: matter <-> theta_local > theta_global(N))
        rr = regime_relative_classifier(pooled, "theta_xivar", lk)
        if rr is not None:
            bundle["D_regime_relative"][tk] = rr
            print(f"  Target: {td}")
            print(f"    Regime-relative classifier "
                  f"(theta_local(a) > theta_global(N)):")
            print(f"      pooled AUC = {rr['AUC_pooled']:.3f}, "
                  f"pooled ratio = "
                  f"{rr['ratio_pooled_at_theta_global']:.3f}")
            for reg, info in rr["per_regime"].items():
                ratio_str = (f"{info['ratio_at_theta_global']:.3f}"
                                 if info['ratio_at_theta_global']
                                 is not None else "n/a")
                auc_str = (f"{info['AUC']:.3f}"
                                 if info['AUC'] is not None
                                 else "n/a")
                print(f"      {reg:<14} N={info['N_global']:>4d}"
                      f"  theta_global={info['theta_global_deg']:>5.1f}deg"
                      f"  matter_relative={info['n_matter_relative']:>4d}/"
                      f"{info['n_total']:>4d}"
                      f"  AUC={auc_str}  ratio={ratio_str}")
        # regime-relative threshold scan (Delta_theta_0 around 0)
        rrs = threshold_scan_relative(pooled, "theta_xivar", lk,
                                            delta_grid.tolist())
        if rrs is not None:
            bundle["D_threshold_scan_relative"][tk] = rrs
            valid_rrs = [r for r in rrs
                              if r["ratio"] is not None]
            if valid_rrs:
                argmax_idx = int(np.argmax(
                    [r["ratio"] for r in valid_rrs]))
                peak = valid_rrs[argmax_idx]
                idx_zero = min(range(len(rrs)),
                                  key=lambda i: abs(
                                      rrs[i]["delta_deg"]))
                r0 = rrs[idx_zero]
                print(f"      regime-relative scan: peak ratio="
                      f"{peak['ratio']:.3f} at "
                      f"Delta={peak['delta_deg']:+.0f}deg; "
                      f"ratio at Delta=0: {r0['ratio']:.3f}")
        print(f"    absolute: ratio at pi/4 = {r45['ratio']:.3f}")
        print()
    print()

    out = OUTPUTS / "verify_local_RG_window_robustness_audit.json"
    out.write_text(json.dumps(bundle, indent=2),
                       encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
