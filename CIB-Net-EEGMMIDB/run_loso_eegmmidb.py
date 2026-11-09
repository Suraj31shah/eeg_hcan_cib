import os
import sys
import json
import time
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.eegmmidb_loader import EEGMMIDBDataset
from models.cib_net_motor import CIBNetMotor
from models.stage_b_invariance import InvarianceGate
from training.losses import SpectralDecouplingLoss
from evaluation.plots import (
    plot_training_history, plot_confusion_matrix,
    plot_invariance_gates, plot_tsne_embeddings
)

def compute_fold_metrics(y_true, y_pred, y_probs=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Delta trivial: accuracy over majority baseline
    unique, counts = np.unique(y_true, return_counts=True)
    majority_rate = np.max(counts) / len(y_true) if len(y_true) > 0 else 0.5
    delta_trivial = acc - majority_rate
    
    # Minority class stats (class 1)
    minority_pred_rate = np.mean(y_pred == 1)
    recall_minority = np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).any() else 0.0
    
    if y_probs is not None and len(np.unique(y_true)) == 2:
        try:
            auroc = roc_auc_score(y_true, y_probs[:, 1])
        except Exception:
            auroc = 0.50
    else:
        auroc = 0.50
        
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "auroc": float(auroc),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "delta_trivial": float(delta_trivial),
        "recall_minority": float(recall_minority),
        "minority_pred_rate": float(minority_pred_rate)
    }

def run_loso(max_folds=None, epochs=25, batch_size=64, lr=0.0005, n_val_subjects=3):
    cache_file = "E:/kagglehub_cache/eegmmidb_cache_all109.pt"
    config_path = "CIB-Net-EEGMMIDB/config/config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    results_dir = "CIB-Net-EEGMMIDB/results"
    weights_dir = os.path.join(results_dir, "checkpoints")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"================================================================================")
    print(f" CIB-NET ZERO-LEAKAGE LOSO BENCHMARK: EEGMMIDB (64 Channels, 109 Subjects)")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f" Epochs per Fold: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f" Validation Subjects Held-Out per Fold: {n_val_subjects}")
    print(f"================================================================================\n")
    
    # Load from preprocessed cache
    dataset = EEGMMIDBDataset(
        data_root=config["dataset"]["raw_data_path"],
        task="imagery_left_right",
        cache_path=cache_file
    )
    
    subject_ids = dataset.subject_ids.numpy()
    unique_subjects = np.unique(subject_ids)
    n_total_subjects = len(unique_subjects)
    
    if max_folds is not None and max_folds < n_total_subjects:
        subjects_to_eval = unique_subjects[:max_folds]
        print(f"[LOSO] Evaluating first {len(subjects_to_eval)} folds of {n_total_subjects} subjects.")
    else:
        subjects_to_eval = unique_subjects
        print(f"[LOSO] Evaluating all {len(subjects_to_eval)} Leave-One-Subject-Out folds.")
        
    fold_results = []
    all_fold_metrics = []
    fold_gates = []
    
    start_total_time = time.time()
    
    for fold_idx, test_subj in enumerate(subjects_to_eval):
        fold_num = fold_idx + 1
        t0 = time.time()
        
        # 1. Strictly isolate test subject
        test_indices = np.where(subject_ids == test_subj)[0]
        train_pool_subjs = unique_subjects[unique_subjects != test_subj]
        
        # 2. Pick validation subjects from training pool (zero leakage into test subject)
        rng = np.random.RandomState(42 * 1000 + fold_idx)
        val_subjs = rng.choice(train_pool_subjs, n_val_subjects, replace=False)
        fit_subjs = np.array([s for s in train_pool_subjs if s not in val_subjs])
        
        val_indices = np.where(np.isin(subject_ids, val_subjs))[0]
        train_indices = np.where(np.isin(subject_ids, fit_subjs))[0]
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_indices),
            batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, val_indices),
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_indices),
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
        )
        
        # Model, Gate, Optimizer
        torch.manual_seed(42 * 1000 + fold_idx)
        model = CIBNetMotor(
            n_channels=64, n_classes=2, d_embed=64,
            shared_weights=True, region_dropout_p=0.2
        ).to(device)
        
        invariance_gate = InvarianceGate(n_regions=5, decay=0.9, tau=1.0).to(device)
        loss_fn = SpectralDecouplingLoss(lambda_0=config["loss"]["lambda_0"], beta_aux=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        
        best_val_score = -1.0
        best_model_weights = None
        best_gates = None
        
        # Training loop for this fold
        for ep in range(1, epochs + 1):
            model.train()
            invariance_gate.train()
            
            for x, y, subj, _ in train_loader:
                x, y = x.to(device), y.to(device)
                subj = subj.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", enabled=(device=="cuda")):
                    final_logits, group_logits, group_feats, _ = model(x)
                    invariance_gate.update(group_logits, y, subj)
                    inv_gates = invariance_gate.get_gates()
                    loss, _, _, _ = loss_fn(final_logits, y, group_logits, inv_gates, group_feats)
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                
            # Validation evaluation
            model.eval()
            val_preds, val_targets, val_probs = [], [], []
            with torch.no_grad():
                for x, y, _, _ in val_loader:
                    x = x.to(device)
                    logits, _, _, _ = model(x)
                    probs = torch.softmax(logits, dim=-1)
                    val_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
                    val_targets.extend(y.numpy())
                    val_probs.extend(probs.cpu().numpy())
                    
            val_metric = compute_fold_metrics(val_targets, val_preds, np.array(val_probs))
            
            if val_metric["balanced_accuracy"] > best_val_score:
                best_val_score = val_metric["balanced_accuracy"]
                best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_gates = invariance_gate.get_gates().detach().cpu().numpy()
                
        # 3. Final evaluation on unseen test subject
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
        model.eval()
        test_preds, test_targets, test_probs = [], [], []
        with torch.no_grad():
            for x, y, _, _ in test_loader:
                x = x.to(device)
                logits, _, _, _ = model(x)
                probs = torch.softmax(logits, dim=-1)
                test_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
                test_targets.extend(y.numpy())
                test_probs.extend(probs.cpu().numpy())
                
        test_metric = compute_fold_metrics(test_targets, test_preds, np.array(test_probs))
        all_fold_metrics.append(test_metric)
        fold_gates.append(best_gates)
        
        # Save per-fold weights
        fold_ckpt = os.path.join(weights_dir, f"fold_{fold_num:03d}_test_S{test_subj:03d}.pth")
        torch.save({
            "fold": fold_num, "test_subject": int(test_subj),
            "metrics": test_metric, "gates": best_gates
        }, fold_ckpt)
        
        fold_duration = time.time() - t0
        print(f"[Fold {fold_num:03d}/{len(subjects_to_eval):03d}] Test Subject: S{test_subj:03d} | "
              f"Acc: {test_metric['accuracy']*100:.2f}% | BalAcc: {test_metric['balanced_accuracy']*100:.2f}% | "
              f"AUROC: {test_metric['auroc']:.4f} | Macro F1: {test_metric['macro_f1']:.4f} | "
              f"MCC: {test_metric['mcc']:.4f} | Pred Rate: {test_metric['minority_pred_rate']:.3f} | "
              f"Time: {fold_duration:.1f}s")
              
        fold_results.append({
            "fold": fold_num,
            "test_subject": int(test_subj),
            "metrics": test_metric,
            "gates": best_gates.tolist(),
            "time_sec": fold_duration
        })
        
        # Intermediate dump to JSON so results are never lost
        with open(os.path.join(results_dir, "fold_results_eegmmidb.json"), "w") as jf:
            json.dump(fold_results, jf, indent=2)
            
    total_time = time.time() - start_total_time
    
    # Compute aggregate Mean +/- Std
    metric_keys = ["accuracy", "balanced_accuracy", "auroc", "macro_f1", "mcc", "delta_trivial", "recall_minority", "minority_pred_rate"]
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in all_fold_metrics]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }
        
    mean_gates = np.mean(np.array(fold_gates), axis=0).tolist()
    
    print("\n" + "="*80)
    print("                     FINAL CIB-NET LOSO BENCHMARK RESULTS")
    print("="*80)
    print(f"Total Folds Evaluated: {len(all_fold_metrics)} | Elapsed Time: {total_time/60:.1f} minutes")
    print(f"Accuracy:          {summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}%")
    print(f"Balanced Accuracy: {summary['balanced_accuracy']['mean']*100:.2f}% ± {summary['balanced_accuracy']['std']*100:.2f}%")
    print(f"AUROC:             {summary['auroc']['mean']:.4f} ± {summary['auroc']['std']:.4f}")
    print(f"Macro F1:          {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}")
    print(f"MCC:               {summary['mcc']['mean']:.4f} ± {summary['mcc']['std']:.4f}")
    print(f"Delta Trivial:     {summary['delta_trivial']['mean']:+.4f} ± {summary['delta_trivial']['std']:.4f}")
    print(f"Minority Pred Rate:{summary['minority_pred_rate']['mean']:.4f} ± {summary['minority_pred_rate']['std']:.4f}")
    print(f"Mean Cortical Gates:{[round(g, 3) for g in mean_gates]}")
    print("="*80)
    
    # Save final report txt
    report_txt = os.path.join(results_dir, "fold_results_eegmmidb.txt")
    with open(report_txt, "w") as rf:
        rf.write("CIB-NET EEGMMIDB LOSO BENCHMARK REPORT\n")
        rf.write(f"Folds: {len(all_fold_metrics)} | Total Time: {total_time/60:.1f} min\n")
        rf.write("="*60 + "\n")
        for k in metric_keys:
            rf.write(f"{k:20s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}\n")
        rf.write("="*60 + "\n")
        
    # Plot final aggregate invariance gates
    region_names = list(config["regions"].keys())
    plot_invariance_gates(mean_gates, region_names, save_path=os.path.join(plots_dir, "gate_heatmap_eegmmidb.png"))
    print(f"[LOSO] All results and plots written to {results_dir}")

if __name__ == "__main__":
    # Run full Leave-One-Subject-Out evaluation
    run_loso(max_folds=None, epochs=25, batch_size=64, lr=0.0005, n_val_subjects=3)

