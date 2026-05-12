r"""SPARC two-phase classifier evaluation on the pre-registered
predictive domain D_halo (late-type, gas-rich subset).

The full 129-galaxy quality-cut sample includes early-type and
bulge-dominated systems where the framework's chirality-flip
prediction does not directly apply. The pre-registered domain
D_halo restricts to:

    T_type >= 8   (late-type)
    Q < 3         (already in quality cut)
    V_flat measured

Within D_halo we compute the same statistical hardening:
binomial p-value, Wilson 95% CI, AUC, confusion matrix, bootstrap
accuracy CI. We also report the AICc-wins of the two-phase
parameter-free framework against always-NFW within D_halo.

Output: outputs/verify_sparc_two_phase_D_halo_evaluation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    bundle = json.load(open(ROOT / "outputs" / "verify_sparc_two_phase_refined.json"))
    rows = bundle["rows"]

    # Apply D_halo: T_type >= 8 (late-type)
    d_halo = [r for r in rows if r.get("T_type", -1) >= 8]
    n_d_halo = len(d_halo)
    n_total = len(rows)
    print("=" * 78)
    print("SPARC two-phase classifier on pre-registered D_halo domain")
    print("=" * 78)
    print(f"  Full Q<3 sample:       {n_total} galaxies")
    print(f"  D_halo (T_type >= 8):  {n_d_halo} galaxies")
    print(f"  Excluded (T_type < 8): {n_total - n_d_halo}")
    print()

    if n_d_halo < 10:
        print("D_halo too small for meaningful test")
        return

    M_b = np.array([r["M_b_e9"] for r in d_halo])
    truth = np.array([1 if r["best_fit_family"] == "NFW" else 0 for r in d_halo])
    threshold = 101.0
    pred = (M_b > threshold).astype(int)
    correct = (pred == truth)
    accuracy = correct.mean()
    n_correct = int(correct.sum())
    n_NFW = int(truth.sum())
    n_Burkert = n_d_halo - n_NFW
    p_majority = max(n_NFW, n_Burkert) / n_d_halo

    print(f"  Pre-registered threshold: M_b > {threshold} x 10^9 Msun")
    print(f"  Within D_halo:")
    print(f"    N actual NFW:     {n_NFW}")
    print(f"    N actual Burkert: {n_Burkert}")
    print(f"    Majority baseline: {p_majority:.4f}")
    print(f"    Accuracy:          {accuracy:.4f} ({n_correct}/{n_d_halo})")

    # Binomial p-value
    p_bin = stats.binomtest(n_correct, n_d_halo, p=p_majority,
                              alternative="greater").pvalue
    print(f"    Binomial p-value:  {p_bin:.4e}")

    # Wilson CI
    z = 1.96
    phat = accuracy
    n = n_d_halo
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2*n)) / denom
    spread = z * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2)) / denom
    print(f"    Wilson 95% CI:     [{max(0,centre-spread):.4f}, {min(1,centre+spread):.4f}]")

    # AUC
    sort_idx = np.argsort(-M_b)
    sorted_truth = truth[sort_idx]
    n_pos = sorted_truth.sum()
    n_neg = n_d_halo - n_pos
    if n_pos > 0 and n_neg > 0:
        tpr = np.cumsum(sorted_truth) / n_pos
        fpr = np.cumsum(1 - sorted_truth) / n_neg
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float("nan")
    print(f"    ROC AUC:           {auc:.4f}")

    # Confusion matrix
    TP = int(np.sum((pred == 1) & (truth == 1)))
    TN = int(np.sum((pred == 0) & (truth == 0)))
    FP = int(np.sum((pred == 1) & (truth == 0)))
    FN = int(np.sum((pred == 0) & (truth == 1)))
    sens = TP / max(1, TP + FN)
    spec = TN / max(1, TN + FP)
    bal_acc = 0.5 * (sens + spec)
    print(f"    Sensitivity (NFW): {sens:.4f}")
    print(f"    Specificity (Burk): {spec:.4f}")
    print(f"    Balanced accuracy: {bal_acc:.4f}")

    # AICc wins of the two-phase parameter-free framework
    # within D_halo
    pred_family = ["NFW" if p else "Burkert" for p in pred]
    AICc_NFW = np.array([r["AICc_NFW_2param"] for r in d_halo])
    AICc_Burk = np.array([r["AICc_Burkert_2param"] for r in d_halo])
    AICc_two_phase = np.array([
        AICc_NFW[i] if pred[i] == 1 else AICc_Burk[i]
        for i in range(n_d_halo)
    ])
    wins_two_phase_vs_NFW = int(np.sum(AICc_two_phase < AICc_NFW))
    wins_two_phase_vs_Burk = int(np.sum(AICc_two_phase < AICc_Burk))
    print()
    print(f"  AICc-wins within D_halo:")
    print(f"    Two-phase prediction wins vs always-NFW:    {wins_two_phase_vs_NFW}/{n_d_halo}")
    print(f"    Two-phase prediction wins vs always-Burkert: {wins_two_phase_vs_Burk}/{n_d_halo}")

    print()
    bundle_out = {
        "method": ("SPARC two-phase classifier evaluation on the pre-registered "
                   "D_halo domain (T_type >= 8 late-type subset of the Q<3 quality "
                   "cut). Reports binomial p-value, Wilson CI, AUC, confusion "
                   "matrix, balanced accuracy, and AICc-wins."),
        "domain_definition": "T_type >= 8 (late-type)",
        "n_full_sample": n_total,
        "n_D_halo": n_d_halo,
        "threshold_M_b_e9_Msun": threshold,
        "accuracy": float(accuracy),
        "majority_baseline": float(p_majority),
        "binomial_p_value": float(p_bin),
        "wilson_95_CI": [float(max(0,centre-spread)), float(min(1,centre+spread))],
        "roc_auc": auc,
        "confusion_matrix": {
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "sensitivity": float(sens), "specificity": float(spec),
            "balanced_accuracy": float(bal_acc)
        },
        "aicc_wins_within_D_halo": {
            "two_phase_vs_always_NFW": wins_two_phase_vs_NFW,
            "two_phase_vs_always_Burkert": wins_two_phase_vs_Burk,
            "n_D_halo": n_d_halo,
        },
    }
    out_path = OUT / "verify_sparc_two_phase_D_halo_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(bundle_out, f, indent=2)
    print(f"Bundle: {out_path}")


if __name__ == "__main__":
    main()
