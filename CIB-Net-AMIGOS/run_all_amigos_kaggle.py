"""
Master Kaggle / CLI Experiment Runner for AMIGOS Replication of the DEAP & DREAMER Suite.

Runs cleanly in resumable phases:
  Phase 1 (`stage1`):
    - Extracts 4-band EEG DE (56-D) + Peripheral (18-D) features cleanly (`raw`)
    - Runs Domain-Shift Diagnostics D0-D4 (`eeg`, `periph`, `concat`) with plots
    - Runs Shallow Baselines (`T0, LR_*, SVC_*, LR_late`) under `none` and `zscore`
  Phase 2 (`baselines` / `hcan` / `stage2`):
    - Deep EEG-Only Baselines (`PlainCNN`, `EEGNet`, `ACRNN`) on raw (`none`) EEG
    - Fair Z-scored Baseline (`EEGNet` + `zscore`, plus optional `PlainCNN`/`ACRNN` zscore)
    - HCAN-CIB Suite:
        1) HCAN-CIB (base / raw: normalize=none, imbalance=none)
        2) HCAN-CIB (Z-score: normalize=zscore, imbalance=none) [transductive]
        3) HCAN-CIB (V-SMOTE: normalize=none, imbalance=v_smote)
        4) HCAN-CIB (GAMO: normalize=none, imbalance=gamo)
        5) HCAN-CIB (DeepSMOTE: normalize=none, imbalance=deepsmote)
        6) Optional extra variants: Class-Weight, Logit-Adj, Z-score + V-SMOTE/GAMO/DeepSMOTE, No-Gating Ablation
  Phase 3 (`summary`):
    - Collates paired Wilcoxon + Holm stats and generates `AMIGOS_Comprehensive_Results_{target}.md`
"""

import argparse
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _run(cmd):
    if cmd and cmd[0] == sys.executable and (len(cmd) < 2 or cmd[1] != "-u"):
        cmd = [cmd[0], "-u", *cmd[1:]]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("\n>>> RUNNING:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/amigos_config.yaml")
    ap.add_argument("--data_path", default="")
    ap.add_argument("--target", default="valence", choices=["valence", "arousal", "dominance"])
    ap.add_argument("--binarize", default="fixed", choices=["fixed", "median"])
    ap.add_argument("--video_subset", default="short", choices=["short", "all"])
    ap.add_argument("--results_root", default="results")
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "stage1", "baselines", "hcan", "stage2", "summary"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--min_epochs", type=int, default=None)
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--also_zscore_all", action="store_true", help="Also run zscore variants of PlainCNN/ACRNN & oversamplers")
    ap.add_argument("--include_ablations", action="store_true", help="Also run class_weight, logit_adj, and no_gating ablation")
    ap.add_argument("--resume", action="store_true", help="Skip steps whose fold_results JSON/npz already exists")
    ap.add_argument("--subjects", nargs="*", type=int, default=None)
    args = ap.parse_args()

    py = sys.executable
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    os.makedirs(args.results_root, exist_ok=True)

    common_flags = ["--config", args.config, "--target", args.target, "--binarize", args.binarize, "--video_subset", args.video_subset]
    if args.data_path:
        common_flags += ["--data_path", args.data_path]
    if args.subjects:
        common_flags += ["--subjects", *[str(s) for s in args.subjects]]

    deep_flags = list(common_flags) + ["--seed", str(args.seed)]
    if args.epochs is not None:
        deep_flags += ["--epochs", str(args.epochs)]
    if args.min_epochs is not None:
        deep_flags += ["--min_epochs", str(args.min_epochs)]
    if args.max_folds is not None:
        deep_flags += ["--max_folds", str(args.max_folds)]

    feat_dir = os.path.join(args.results_root, "features")
    trial_npz = os.path.join(feat_dir, f"feats_amigos_{args.target}_{args.binarize}_trial.npz")

    # ---------------- Phase 1: DE Extraction + Diagnostics + Shallow Baselines ----------------
    if args.phase in ("all", "stage1"):
        if not (args.resume and os.path.exists(trial_npz)):
            _run([py, "extract_features_amigos.py", *common_flags, "--out_dir", feat_dir])
        else:
            print(f"[Resume] Skipping DE extraction; found {trial_npz}")

        for fblock in ("eeg", "periph", "concat"):
            diag_dir = os.path.join(args.results_root, f"diag_{args.target}_{fblock}")
            if not (args.resume and os.path.exists(os.path.join(diag_dir, "diagnostics.json"))):
                _run([
                    py,
                    "domain_shift_diagnostics.py",
                    "--data",
                    trial_npz,
                    "--features",
                    fblock,
                    "--out",
                    diag_dir,
                    "--seed",
                    str(args.seed),
                    "--plot",
                ])

        for norm in ("none", "zscore"):
            sh_dir = os.path.join(args.results_root, f"shallow_{args.target}_{norm}")
            if not (args.resume and os.path.exists(os.path.join(sh_dir, f"summary_{norm}.md"))):
                _run([
                    py,
                    "shallow_baselines.py",
                    "--data",
                    trial_npz,
                    "--out",
                    sh_dir,
                    "--norm",
                    norm,
                    "--seed",
                    str(args.seed),
                ])

    # ---------------- Phase 2A: Deep EEG-only Baselines ----------------
    if args.phase in ("all", "stage2", "baselines"):
        baseline_runs = [
            ("cnn", "none"),
            ("eegnet", "none"),
            ("acrnn", "none"),
            ("eegnet", "zscore"),
        ]
        if args.also_zscore_all:
            baseline_runs += [("cnn", "zscore"), ("acrnn", "zscore")]

        for mname, norm in baseline_runs:
            out_dir = os.path.join(args.results_root, f"baseline_{mname}_{norm}_{args.target}")
            if args.resume and os.path.exists(os.path.join(out_dir, "baseline_fold_results.json")):
                print(f"[Resume] Skipping {out_dir}")
                continue
            _run([
                py,
                os.path.join("experiments", "run_baselines.py"),
                "--model",
                mname,
                "--normalize",
                norm,
                "--results_dir",
                out_dir,
                *deep_flags,
            ])

    # ---------------- Phase 2B: HCAN-CIB Variants ----------------
    if args.phase in ("all", "stage2", "hcan"):
        hcan_runs = [
            ("none", "none", []),
            ("none", "zscore", []),
            ("v_smote", "none", []),
            ("gamo", "none", []),
            ("deepsmote", "none", []),
        ]
        if args.also_zscore_all:
            hcan_runs += [
                ("v_smote", "zscore", []),
                ("gamo", "zscore", []),
                ("deepsmote", "zscore", []),
            ]
        if args.include_ablations:
            hcan_runs += [
                ("class_weight", "none", []),
                ("logit_adj", "none", []),
                ("none", "none", ["--no_gating"]),
            ]

        for imb, norm, extra_flags in hcan_runs:
            tag = f"hcan_{imb}_{norm}_{args.target}"
            if "--no_gating" in extra_flags:
                tag += "_nogate"
            out_dir = os.path.join(args.results_root, tag)
            if args.resume and os.path.exists(os.path.join(out_dir, "fold_results.json")):
                print(f"[Resume] Skipping {out_dir}")
                continue
            _run([
                py,
                os.path.join("experiments", "run_hcan_cib.py"),
                "--imbalance",
                imb,
                "--normalize",
                norm,
                "--results_dir",
                out_dir,
                *deep_flags,
                *extra_flags,
            ])

    # ---------------- Phase 3: Summary Report ----------------
    if args.phase in ("all", "stage1", "stage2", "baselines", "hcan", "summary"):
        _run([
            py,
            os.path.join("experiments", "generate_summary_report.py"),
            "--results_root",
            args.results_root,
            "--target",
            args.target,
        ])


if __name__ == "__main__":
    main()

