import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.amigos_loader import WindowedAMIGOSDataset
from eeg_common import resolve_n_val
from evaluation.metrics import compute_metrics, print_metrics
from evaluation.plots import plot_roc_pr_curves
from models.baselines.acrnn import ACRNN
from models.baselines.eegnet import EEGNet
from models.baselines.plain_cnn import PlainCNN
from training.loso_evaluator import pooled_val_auroc, to_py, tune_threshold
from training.subject_sampler import SubjectBalancedSampler


@torch.no_grad()
def predict_trials_baseline(model, dataloader, device):
    model.eval()
    logits, labels = {}, {}
    for x_eeg, _, y, subj_ids, trial_ids in tqdm(dataloader, desc="Evaluating", leave=False):
        x_eeg = x_eeg.to(device)
        final_logits = model(x_eeg).float().cpu().numpy()
        final_logits = np.nan_to_num(final_logits, nan=0.0)
        for i in range(len(y)):
            key = (int(subj_ids[i]), int(trial_ids[i]))
            logits.setdefault(key, []).append(final_logits[i])
            labels[key] = int(y[i])

    keys = sorted(logits.keys())
    mean_logits = np.stack([np.mean(logits[k], axis=0) for k in keys])
    z = mean_logits - mean_logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return {
        "subj": np.array([k[0] for k in keys]),
        "trial": np.array([k[1] for k in keys]),
        "y": np.array([labels[k] for k in keys]),
        "prob1": probs[:, 1],
        "pred": mean_logits.argmax(axis=1),
    }


def train_epoch_baseline(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for x_eeg, _, y, _, _ in tqdm(dataloader, desc="Training", leave=False):
        x_eeg, y = x_eeg.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        final_logits = model(x_eeg)
        loss = criterion(final_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        all_preds.extend(torch.argmax(final_logits.detach(), dim=1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    return total_loss / max(n_batches, 1), compute_metrics(all_targets, all_preds)


def _build_baseline(model_key: str, config: dict):
    n_channels = int(config["n_channels"])
    n_classes = int(config["n_classes"])
    fs = int(config.get("sample_rate", 128))
    win_sec = float(config.get("window_sec", 4.0))
    if model_key == "cnn":
        return PlainCNN(n_channels=n_channels, n_classes=n_classes)
    if model_key == "eegnet":
        return EEGNet(n_channels=n_channels, n_classes=n_classes, sample_rate=fs, window_sec=win_sec)
    if model_key == "acrnn":
        return ACRNN(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(model_key)


def run_baseline_loso(
    dataset,
    model_key: str,
    config: dict,
    seed: int = 0,
    results_dir: str = "results/baseline",
    max_folds: int = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = config["training"]
    subject_ids = np.asarray(dataset.subject_ids)
    subjects = np.unique(subject_ids)
    n_val = resolve_n_val(len(subjects), int(tcfg.get("n_val_subjects", 4)))
    patience = int(tcfg.get("patience", 8))
    min_epochs = int(tcfg.get("min_epochs", 10))
    bs = int(tcfg["batch_size"])

    print(
        f"--- Baseline {model_key.upper()} LOSO on {device} | seed={seed} | n_subjects={len(subjects)} "
        f"| n_val={n_val} | normalize={config.get('normalize', 'none')} ---"
    )

    os.makedirs(f"{results_dir}/weights_baseline", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)

    fold_metrics, fold_preds = [], []

    for fold, test_subject in enumerate(subjects):
        if max_folds is not None and fold >= max_folds:
            break

        test_idx = np.where(subject_ids == test_subject)[0]
        pool_idx = np.where(subject_ids != test_subject)[0]
        rng = np.random.RandomState(seed * 1000 + fold)
        val_subjects = rng.choice(subjects[subjects != test_subject], n_val, replace=False)
        is_val = np.isin(subject_ids[pool_idx], val_subjects)
        val_idx, fit_idx = pool_idx[is_val], pool_idx[~is_val]

        print(
            f"\nFold {fold + 1}/{len(subjects)} | test=P{int(test_subject):02d} "
            f"| val={sorted(map(int, val_subjects))}"
        )

        eff_bs = min(bs, max(8, len(fit_idx)))
        train_sampler = SubjectBalancedSampler(dataset, fit_idx, eff_bs)
        train_loader = DataLoader(
            dataset, batch_size=eff_bs, sampler=train_sampler, drop_last=len(fit_idx) >= eff_bs, num_workers=0
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=eff_bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=eff_bs, shuffle=False, num_workers=0)

        torch.manual_seed(seed * 1000 + fold)
        model = _build_baseline(model_key, config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
        )

        best_val, best_epoch, bad = -np.inf, -1, 0
        best_ckpt = None

        for epoch in range(int(tcfg["epochs"])):
            _, train_m = train_epoch_baseline(model, train_loader, optimizer, device)
            val_pred = predict_trials_baseline(model, val_loader, device)
            val_auc = pooled_val_auroc(val_pred)
            print(
                f"  Epoch {epoch + 1:02d}/{tcfg['epochs']} | Train Acc: {train_m['accuracy']:.4f} "
                f"| Val AUROC: {val_auc:.4f}"
            )

            if np.isfinite(val_auc) and val_auc > best_val + 1e-4:
                best_val, best_epoch, bad = val_auc, epoch + 1, 0
                best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if epoch + 1 >= min_epochs and bad >= patience:
                    print(f"  Early stop at epoch {epoch + 1} (best {best_epoch}, val AUROC {best_val:.4f})")
                    break

        if best_ckpt is None:
            best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(tcfg["epochs"])

        model.load_state_dict(best_ckpt)
        torch.save(best_ckpt, f"{results_dir}/weights_baseline/best_model_fold_{fold + 1}.pth")

        val_pred = predict_trials_baseline(model, val_loader, device)
        thr = tune_threshold(val_pred["y"], val_pred["prob1"])
        te = predict_trials_baseline(model, test_loader, device)

        m = compute_metrics(te["y"], te["pred"], y_prob=te["prob1"])
        m_thr = compute_metrics(te["y"], (te["prob1"] >= thr).astype(int), y_prob=te["prob1"])
        m = to_py(m)
        m["val_thr_metrics"] = to_py(m_thr)
        m["val_threshold"] = thr
        m["best_epoch"] = int(best_epoch)
        m["val_auroc"] = float(best_val)
        m["test_subject"] = int(test_subject)
        m["val_subjects"] = [int(s) for s in val_subjects]

        print_metrics(m, fold_name=f"Test P{int(test_subject):02d} (epoch {best_epoch})")
        fold_metrics.append(m)
        fold_preds.append(te)

    print(f"\n=== FINAL {model_key.upper()} BASELINE RESULTS (AMIGOS) ===")
    for key in ["accuracy", "balanced_acc", "auroc", "macro_f1", "delta_trivial"]:
        v = np.array([fm[key] for fm in fold_metrics], dtype=float)
        print(f"{key}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    with open(f"{results_dir}/baseline_fold_results.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)
    np.savez(
        f"{results_dir}/trial_predictions_seed{seed}.npz",
        **{f"fold{i + 1}_{k}": v for i, p in enumerate(fold_preds) for k, v in p.items()},
    )
    try:
        plot_roc_pr_curves(
            [p["y"] for p in fold_preds],
            [p["prob1"] for p in fold_preds],
            save_dir=f"{results_dir}/plots",
        )
    except Exception as e:
        print(f"Plot skipped: {e}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "eegnet", "acrnn"])
    parser.add_argument("--config", type=str, default="config/amigos_config.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None, choices=["valence", "arousal", "dominance"])
    parser.add_argument("--binarize", type=str, default=None, choices=["fixed", "median"])
    parser.add_argument("--video_subset", type=str, default=None, choices=["short", "all"])
    parser.add_argument("--normalize", type=str, default=None, choices=["none", "zscore"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min_epochs", type=int, default=None)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.target is not None:
        config["target"] = args.target
    if args.binarize is not None:
        config["binarize"] = args.binarize
    if args.video_subset is not None:
        config["video_subset"] = args.video_subset
    if args.normalize is not None:
        config["normalize"] = args.normalize
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.min_epochs is not None:
        config["training"]["min_epochs"] = args.min_epochs

    dpath = args.data_path if args.data_path is not None else config.get("data_path", "")
    norm = config.get("normalize", "none")

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

    out_dir = args.results_dir or f"results/baseline_{args.model}_{norm}_{config['target']}"
    run_baseline_loso(
        dataset=dataset,
        model_key=args.model,
        config=config,
        seed=args.seed,
        results_dir=out_dir,
        max_folds=args.max_folds,
    )


if __name__ == "__main__":
    main()

