import argparse
import os
import sys
import yaml

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.amigos_loader import WindowedAMIGOSDataset
from models.hcan_cib_net import HCANCIBNet
from training.loso_evaluator import run_hcan_cib_loso


def main():
    parser = argparse.ArgumentParser(description="Run HCAN-CIB LOSO Benchmark on AMIGOS")
    parser.add_argument("--config", type=str, default="config/amigos_config.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None, choices=["valence", "arousal", "dominance"])
    parser.add_argument("--binarize", type=str, default=None, choices=["fixed", "median"])
    parser.add_argument("--video_subset", type=str, default=None, choices=["short", "all"])
    parser.add_argument("--normalize", type=str, default=None, choices=["none", "zscore"])
    parser.add_argument(
        "--imbalance",
        type=str,
        default=None,
        choices=["none", "class_weight", "logit_adj", "smote", "v_smote", "deepsmote", "gamo"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min_epochs", type=int, default=None)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--no_gating", action="store_true", help="Ablation: disable Stage B gating (gamma_r=1)")
    parser.add_argument("--no_sd", action="store_true", help="Ablation: disable Spectral Decoupling (lambda_sd=0)")
    parser.add_argument("--ablate_modality", type=str, default=None, choices=["eeg", "periph"])
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    override = {"imbalance": dict(config.get("imbalance", {})), "training": dict(config.get("training", {}))}
    if args.target is not None:
        config["target"] = args.target
        override["target"] = args.target
    if args.binarize is not None:
        config["binarize"] = args.binarize
        override["binarize"] = args.binarize
    if args.video_subset is not None:
        config["video_subset"] = args.video_subset
        override["video_subset"] = args.video_subset
    if args.normalize is not None:
        config["normalize"] = args.normalize
        override["normalize"] = args.normalize
    if args.imbalance is not None:
        override["imbalance"]["method"] = args.imbalance
    if args.epochs is not None:
        override["training"]["epochs"] = args.epochs
    if args.min_epochs is not None:
        override["training"]["min_epochs"] = args.min_epochs

    dpath = args.data_path if args.data_path is not None else config.get("data_path", "")
    norm = override.get("normalize", config.get("normalize", "none"))
    imb = override["imbalance"].get("method", "none")

    dataset = WindowedAMIGOSDataset(
        data_path=dpath,
        target=config["target"],
        binarize=config["binarize"],
        threshold=float(config.get("threshold", 5.0)),
        video_subset=config.get("video_subset", "short"),
        max_short_duration_sec=float(config.get("max_short_duration_sec", 240.0)),
        window_sec=float(config.get("window_sec", 4.0)),
        overlap_sec=float(config.get("overlap_sec", 2.0)),
        fs=int(config.get("sample_rate", 128)),
        baseline_sec=float(config.get("baseline_sec", 5.0)),
        bandpass_low=float(config.get("bandpass_low", 4.0)),
        bandpass_high=float(config.get("bandpass_high", 45.0)),
        normalize=norm,
        subjects=args.subjects,
    )

    tag = f"hcan_{imb}_{norm}_{config['target']}"
    if args.no_gating:
        tag += "_nogate"
    if args.no_sd:
        tag += "_nosd"
    if args.ablate_modality:
        tag += f"_no{args.ablate_modality}"

    results_dir = args.results_dir or os.path.join("results", tag)

    run_hcan_cib_loso(
        dataset=dataset,
        model_class=HCANCIBNet,
        config_path=args.config,
        seed=args.seed,
        results_dir=results_dir,
        config_override=override,
        use_gating=not args.no_gating,
        use_sd=not args.no_sd,
        ablate_modality=args.ablate_modality,
        max_folds=args.max_folds,
    )


if __name__ == "__main__":
    main()

