import sys
import os
import yaml
import argparse
import numpy as np
import torch
import json
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.windowed_loader import WindowedDEAPDataset
from models.baselines.plain_cnn import PlainCNN
from models.baselines.eegnet import EEGNet
from models.baselines.acrnn import ACRNN
from training.subject_sampler import SubjectBalancedSampler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from evaluation.metrics import compute_metrics, print_metrics

from training.loso_evaluator import pooled_val_auroc, tune_threshold, to_py

@torch.no_grad()
def predict_trials_baseline(model, dataloader, device):
    """
    Average window logits per trial for baselines (which only take EEG, no periph).
    """
    model.eval()
    logits, labels = {}, {}

    for x_eeg, _, y, subj_ids, trial_ids in tqdm(dataloader, desc="Evaluating", leave=False):
        x_eeg = x_eeg.to(device)
        
        # Baselines don't return region logits
        final_logits = model(x_eeg)
        final_logits = final_logits.float().cpu().numpy()
        
        for i in range(len(y)):
            key = (int(subj_ids[i]), int(trial_ids[i]))
            logits.setdefault(key, []).append(final_logits[i])
            labels[key] = int(y[i])

    keys = sorted(logits.keys())
    mean_logits = np.stack([np.mean(logits[k], axis=0) for k in keys])
    mean_logits = np.nan_to_num(mean_logits, nan=0.0) # Protect against NaN explosions
    z = mean_logits - mean_logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return {
        "subj": np.array([k[0] for k in keys]),
        "y": np.array([labels[k] for k in keys]),
        "prob1": probs[:, 1],
        "pred": mean_logits.argmax(axis=1),
    }

def train_epoch_baseline(model, dataloader, optimizer, scaler, device):
    model.train()
    use_amp = False # Baselines are small, disable AMP to prevent NaN overflows
    criterion = nn.CrossEntropyLoss()

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for x_eeg, _, y, _, _ in tqdm(dataloader, desc="Training", leave=False):
        x_eeg, y = x_eeg.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            final_logits = model(x_eeg)
            loss = criterion(final_logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(torch.argmax(final_logits.detach(), dim=1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    return total_loss / max(n_batches, 1), compute_metrics(all_targets, all_preds)


def run_baseline_loso(dataset, model_class, config, seed=0, results_dir="results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = config["training"]
    n_val = tcfg.get("n_val_subjects", 4)
    patience = tcfg.get("patience", 8)
    min_epochs = tcfg.get("min_epochs", 10)
    bs = tcfg["batch_size"]
    
    print(f"--- Baseline LOSO (val-subject selection) on {device} | seed={seed} ---")

    os.makedirs(f"{results_dir}/weights_baseline", exist_ok=True)
    
    subject_ids = np.asarray(dataset.subject_ids)
    subjects = np.unique(subject_ids)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    fold_metrics, fold_preds = [], []

    for fold, test_subject in enumerate(subjects):
        test_idx = np.where(subject_ids == test_subject)[0]
        pool_idx = np.where(subject_ids != test_subject)[0]
        rng = np.random.RandomState(seed * 1000 + fold)
        val_subjects = rng.choice(subjects[subjects != test_subject], n_val, replace=False)
        is_val = np.isin(subject_ids[pool_idx], val_subjects)
        val_idx, fit_idx = pool_idx[is_val], pool_idx[~is_val]

        print(f"\nFold {fold + 1}/{len(subjects)} | test={test_subject} | val={sorted(map(int, val_subjects))}")

        train_sampler = SubjectBalancedSampler(dataset, fit_idx, bs)
        train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler, drop_last=True, num_workers=0)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=bs, shuffle=False, num_workers=0)

        torch.manual_seed(seed * 1000 + fold)
        
        # Baselines only take channels and classes
        model = model_class(
            n_channels=config["n_channels"], 
            n_classes=config["n_classes"]
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

        best_val, best_epoch, bad = -np.inf, -1, 0
        best_ckpt = None
        
        for epoch in range(tcfg["epochs"]):
            _, train_m = train_epoch_baseline(model, train_loader, optimizer, scaler, device)
            val_pred = predict_trials_baseline(model, val_loader, device)
            val_auc = pooled_val_auroc(val_pred)
            
            print(f"  Epoch {epoch + 1:02d}/{tcfg['epochs']} | Train Acc: {train_m['accuracy']:.4f} | Val AUROC: {val_auc:.4f}")

            if np.isfinite(val_auc) and val_auc > best_val + 1e-4:
                best_val, best_epoch, bad = val_auc, epoch + 1, 0
                best_ckpt = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if epoch + 1 >= min_epochs and bad >= patience:
                    print(f"  Early stop at epoch {epoch + 1} (best {best_epoch}, val AUROC {best_val:.4f})")
                    break

        if best_ckpt is None:  
            best_ckpt = model.state_dict()
            best_epoch = epoch + 1
            
        model.load_state_dict(best_ckpt)
        torch.save(best_ckpt, f"{results_dir}/weights_baseline/best_model_fold_{fold + 1}.pth")

        # Evaluate once on test
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

        print_metrics(m, fold_name=f"Test (epoch {best_epoch})")
        fold_metrics.append(m)
        fold_preds.append(te)

    # Aggregation
    print("\n=== FINAL BASELINE RESULTS ===")
    for key in ["accuracy", "balanced_acc", "auroc", "macro_f1"]:
        v = np.array([fm[key] for fm in fold_metrics], dtype=float)
        print(f"{key}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    with open(f"{results_dir}/baseline_fold_results.json", "w") as f:
        json.dump(fold_metrics, f, indent=4)
        
    return fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'eegnet', 'acrnn'])
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'hcan_cib_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = WindowedDEAPDataset(
        data_path=config['data_path'],
        target=config['target'],
        binarize=config['binarize'],
        window_sec=config['window_sec'],
        overlap_sec=config['overlap_sec'],
        fs=config['sample_rate'],
        baseline_sec=config['baseline_sec']
    )
    
    model_map = {'cnn': PlainCNN, 'eegnet': EEGNet, 'acrnn': ACRNN}
    print(f"\nEvaluating Baseline: {args.model.upper()}")
    
    run_baseline_loso(
        dataset=dataset,
        model_class=model_map[args.model],
        config=config,
        results_dir=f"results_{args.model}"
    )

if __name__ == '__main__':
    main()
