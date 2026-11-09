"""
Generates `AMIGOS_Comprehensive_Results.md` aggregating all executed experiments
(Domain-Shift Diagnostics D0-D4, Shallow Baselines, Deep Baselines, HCAN-CIB variants,
paired Wilcoxon + Holm comparisons, and Stage B Regional Invariance Gates).
"""

import argparse
import glob
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_models import compare
from eeg_common import boot_ci


def _load_folds(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _metric_row(name: str, folds):
    aurocs = np.array([f.get("auroc", np.nan) for f in folds], dtype=float)
    fin_auc = aurocs[np.isfinite(aurocs)]
    lo, hi = boot_ci(fin_auc)
    se = float(fin_auc.std(ddof=1) / np.sqrt(len(fin_auc))) if len(fin_auc) > 1 else float("nan")
    accs = np.array([f.get("accuracy", np.nan) for f in folds], dtype=float)
    deltas = np.array([f.get("delta_trivial", np.nan) for f in folds], dtype=float)
    baccs = np.array(
        [f.get("val_thr_metrics", f).get("balanced_acc", np.nan) for f in folds], dtype=float
    )
    f1s = np.array(
        [f.get("val_thr_metrics", f).get("macro_f1", np.nan) for f in folds], dtype=float
    )
    min_rec = np.array(
        [f.get("val_thr_metrics", f).get("recall_minority", np.nan) for f in folds], dtype=float
    )
    mu_auc = float(np.mean(fin_auc)) if len(fin_auc) > 0 else float("nan")
    return (
        f"| {name} | {len(folds)} | {mu_auc:.4f} +/- {se:.4f} [{lo:.3f}, {hi:.3f}] | "
        f"{np.nanmean(accs):.4f} | {np.nanmean(deltas):+.4f} | {np.nanmean(baccs):.4f} | "
        f"{np.nanmean(f1s):.4f} | {np.nanmean(min_rec):.4f} |"
    )


def build_report(results_root: str = "results", target: str = "valence", out_path: str = None):
    out_path = out_path or os.path.join(results_root, f"AMIGOS_Comprehensive_Results_{target}.md")
    lines = [
        f"# AMIGOS Benchmark Comprehensive Results ({target.capitalize()})",
        "",
        "Generated automatically by `experiments/generate_summary_report.py`.",
        "",
    ]

    # 1. Domain-Shift Diagnostics
    lines.append("## 1. Domain-Shift Diagnostics (D0-D4)")
    lines.append("")
    diag_files = sorted(glob.glob(os.path.join(results_root, f"diag_{target}_*", "diagnostics.json")))
    if diag_files:
        for dp in diag_files:
            feat = os.path.basename(os.path.dirname(dp)).split("_")[-1]
            with open(dp) as f:
                R = json.load(f)
            d0 = R["D0_label_priors"]
            d1 = R["D1"]
            d2 = R["D2"]
            d3 = R["D3"]
            d4 = R["D4"]
            lines.extend([
                f"### Feature Block: `{feat}` ({R['n_subjects']} subjects, {R['n_trials']} trials, {R['n_features']} dims)",
                f"- **D0 Label Priors:** pos_rate mean={d0['pos_rate_mean']:.3f} (range {d0['pos_rate_min']:.3f}..{d0['pos_rate_max']:.3f}), mean trivial acc={d0['trivial_acc_mean']:.3f}, subjects >=0.70 majority={d0['n_subjects_majority_ge_0.7']}/{d0['n_subjects']}",
                f"- **D1 Within-Subject vs LOSO:** blocked CV={d1['within_blocked']['mean_auroc']:.3f}, shuffled CV={d1['within_shuffled']['mean_auroc']:.3f}, LOSO raw={d1['loso_raw']['mean_auroc']:.3f}, LOSO zscore={d1['loso_zscore']['mean_auroc']:.3f}",
                f"- **D2 Subject-ID Decodability:** raw={d2['none']:.3f} (x{d2['none']/d2['chance']:.1f} chance), center={d2['center']:.3f}, zscore={d2['zscore']:.3f} (chance={d2['chance']:.3f})",
                f"- **D3 Variance Share ($\\eta^2$):** subject={d3['eta2_subject_mean']:.3f}, emotion label|subject={d3['eta2_label_within_subject_mean']:.4f} (ratio={d3['ratio_subject_to_label']:.1f}x)",
                f"- **D4 Normalization Effect (val-tuned LR):** none={d4['none']['mean_auroc']:.3f}, center={d4['center']['mean_auroc']:.3f} (diff={d4['center']['paired_diff_vs_none']:+.3f}, p={d4['center']['paired_p']:.3f}), zscore={d4['zscore']['mean_auroc']:.3f} (diff={d4['zscore']['paired_diff_vs_none']:+.3f}, p={d4['zscore']['paired_p']:.3f})",
                "",
            ])
    else:
        lines.append("*No diagnostics JSON files found yet.*")
        lines.append("")

    # 2. Shallow Baselines
    lines.append("## 2. Shallow Baselines (Identical Validation-Subject LOSO Split)")
    lines.append("")
    for norm_mode in ("none", "zscore"):
        md_file = os.path.join(results_root, f"shallow_{target}_{norm_mode}", f"summary_{norm_mode}.md")
        lines.append(f"### Normalization: `{norm_mode}`" + (" *(transductive)*" if norm_mode == "zscore" else " *(raw / zero-leakage)*"))
        if os.path.exists(md_file):
            with open(md_file, encoding="utf-8") as f:
                lines.append(f.read().strip())
        else:
            lines.append(f"*No shallow summary at `{md_file}`.*")
        lines.append("")

    # 3. Deep Models Unified Summary Table
    lines.append("## 3. Deep Models (Baselines & HCAN-CIB Variants)")
    lines.append("")
    lines.append("| Model Variant | Folds | AUROC mean +/- SE [95% CI] | Acc (argmax) | Delta Trivial | Bal Acc (thr) | Macro F1 (thr) | Min Recall (thr) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    model_specs = []
    # Scan baseline_* and hcan_* folders matching target
    for pattern, json_name in [
        (os.path.join(results_root, f"baseline_*_{target}"), "baseline_fold_results.json"),
        (os.path.join(results_root, f"hcan_*_{target}*"), "fold_results.json"),
    ]:
        for folder in sorted(glob.glob(pattern)):
            jp = os.path.join(folder, json_name)
            folds = _load_folds(jp)
            if folds:
                name = os.path.basename(folder)
                lines.append(_metric_row(name, folds))
                model_specs.append(f"{name}={jp}")
    lines.append("")

    # 4. Paired Statistical Comparison (Wilcoxon + Holm)
    comp_specs = list(model_specs)
    shallow_dir = os.path.join(results_root, f"shallow_{target}_none")
    for s_tag in ("T0_none", "LR_eeg_none", "LR_periph_none", "LR_concat_none", "SVC_concat_none", "LR_late_none"):
        sp = os.path.join(shallow_dir, f"fold_results_{s_tag}.json")
        if os.path.exists(sp):
            comp_specs.append(f"shallow_{s_tag}={sp}")
    sp_zs = os.path.join(results_root, f"shallow_{target}_zscore", "fold_results_LR_concat_zscore.json")
    if os.path.exists(sp_zs):
        comp_specs.append(f"shallow_LR_concat_zscore={sp_zs}")

    if len(comp_specs) >= 2:
        lines.append("## 4. Paired Statistical Comparison (Wilcoxon + Holm Correction)")
        lines.append("")
        ref_spec = next(
            (m.split("=")[0] for m in comp_specs if f"hcan_none_none_{target}" == m.split("=")[0]),
            comp_specs[0].split("=")[0],
        )
        comp_table = compare(comp_specs, metric="auroc", ref=ref_spec, chance=0.5)
        lines.append(comp_table)
        lines.append("")

    # 5. Stage B Region Invariance Gates
    lines.append("## 5. Stage B Mean Region Invariance Gates (14-Ch Emotiv EPOC)")
    lines.append("")
    region_names = ["Frontal", "Temporal", "Parietal", "Occipital"]
    gate_rows = []
    for folder in sorted(glob.glob(os.path.join(results_root, f"hcan_*_{target}*"))):
        if "nogate" in folder:
            continue
        folds = _load_folds(os.path.join(folder, "fold_results.json"))
        if folds and "gate_values" in folds[0]:
            arr = np.asarray([f["gate_values"] for f in folds if "gate_values" in f], dtype=float)
            means = arr.mean(0)
            stds = arr.std(0)
            vals_str = " | ".join(f"{means[i]:.3f} +/- {stds[i]:.3f}" for i in range(min(4, arr.shape[1])))
            gate_rows.append(f"| {os.path.basename(folder)} | {vals_str} |")
    if gate_rows:
        lines.append(f"| Variant | {' | '.join(region_names)} |")
        lines.append("|---|---|---|---|---|")
        lines.extend(gate_rows)
    else:
        lines.append("*No HCAN-CIB gate results found yet.*")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    report = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] Written to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--target", default="valence", choices=["valence", "arousal", "dominance"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build_report(args.results_root, args.target, args.out)


if __name__ == "__main__":
    main()
