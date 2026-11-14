import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.eegmmidb_loader import EEGMMIDBDataset
from models.baselines.eegnet import EEGNet
from models.baselines.acrnn import ACRNN
from models.baselines.plain_cnn import PlainCNN
from models.cib_net_motor import CIBNetMotor
from models.stage_b_invariance import InvarianceGate
from training.losses import SpectralDecouplingLoss
from evaluation.metrics import compute_all_metrics

def plot_model_artifacts(model_name, y_true, y_pred, y_probs, embeddings, subjects, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Left Fist', 'Right Fist'],
                yticklabels=['Left Fist', 'Right Fist'],
                annot_kws={'size': 13, 'weight': 'bold'})
    plt.title(f'Confusion Matrix: {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{clean_name}.png'), dpi=300)
    plt.close()
    
    # 2. t-SNE Plot
    if embeddings is not None and len(embeddings) > 0:
        embeddings = np.array(embeddings)
        subjects = np.array(subjects)
        y_true = np.array(y_true)
        
        # Standardize features per subject to eliminate between-subject DC offsets
        # This prevents artificial subject-island clustering and shows true class distribution
        embs_norm = np.zeros_like(embeddings)
        for s in np.unique(subjects):
            mask = (subjects == s)
            s_embs = embeddings[mask]
            mu = np.mean(s_embs, axis=0, keepdims=True)
            sd = np.std(s_embs, axis=0, keepdims=True) + 1e-8
            embs_norm[mask] = (s_embs - mu) / sd
            
        n_samples = min(600, len(embs_norm))
        sub_idx = np.random.choice(len(embs_norm), n_samples, replace=False)
        tsne = TSNE(n_components=2, perplexity=35, random_state=42)
        emb_2d = tsne.fit_transform(embs_norm[sub_idx])
        
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true[sub_idx], cmap='coolwarm', alpha=0.75, s=35, edgecolors='none')
        plt.title(f't-SNE Latent Space: {model_name}', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Left Fist', 'Right Fist'])
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'tsne_{clean_name}.png'), dpi=300)
        plt.close()

def apply_subject_zscore(dataset):
    """
    Applies per-subject z-score normalization across all channels and time samples.
    """
    import copy
    norm_ds = copy.deepcopy(dataset)
    subjs = norm_ds.subject_ids.numpy()
    x = norm_ds.x.numpy().copy()
    for s in np.unique(subjs):
        m = (subjs == s)
        s_data = x[m]
        mu = np.mean(s_data, axis=(0, 2), keepdims=True)
        sd = np.std(s_data, axis=(0, 2), keepdims=True) + 1e-8
        x[m] = (s_data - mu) / sd
    norm_ds.x = torch.from_numpy(x).float()
    return norm_ds

def evaluate_model_pipeline(model_type, dataset, n_folds=23, epochs=20, batch_size=64, lr=0.0005, device="cuda", normalize="none"):
    """
    Evaluates one model pipeline across n_folds Leave-One-Subject-Out subjects.
    model_type: 'ACRNN' | 'PlainCNN' | 'EEGNet' | 'CIB-Net' | 'CIB-Net (V-SMOTE)' | 'CIB-Net (GAMO)' | 'CIB-Net (Z-score)' | 'CIB-Net (DeepSMOTE)'
    """
    if normalize == "zscore" or "Z-score" in model_type:
        print("[PREPROCESS] Applying per-subject transductive Z-score standardization...")
        dataset = apply_subject_zscore(dataset)
    subject_ids = dataset.subject_ids.numpy()
    unique_subjects = np.unique(subject_ids)
    test_subjects = unique_subjects[:n_folds]
    
    print(f"\n" + "="*80)
    print(f" STARTING BENCHMARK: {model_type} ({n_folds} Folds, {epochs} Epochs/Fold)")
    print(f"="*80)
    
    results_dir = "CIB-Net-EEGMMIDB/results"
    clean_name = model_type.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    partial_file = os.path.join(results_dir, f"partial_{clean_name}.json")
    
    partial_data = {}
    if os.path.exists(partial_file):
        try:
            with open(partial_file, "r") as f:
                partial_data = json.load(f)
        except Exception:
            partial_data = {}

    fold_metrics = []
    all_test_targets = []
    all_test_preds = []
    all_test_probs = []
    all_test_embs = []
    all_test_subjs = []
    
    start_time = time.time()
    
    for fold_idx, test_subj in enumerate(test_subjects):
        t0 = time.time()
        
        # Check if fold was already evaluated in partial cache
        if str(fold_idx) in partial_data:
            p_fold = partial_data[str(fold_idx)]
            fold_metrics.append(p_fold["metric"])
            all_test_targets.extend(p_fold["targets"])
            all_test_preds.extend(p_fold["preds"])
            all_test_probs.extend(p_fold["probs"])
            all_test_embs.extend(p_fold["embs"])
            all_test_subjs.extend(p_fold["subjs"])
            print(f"[{model_type} Fold {fold_idx+1:02d}/{n_folds:02d}] Subj: S{test_subj:03d} (Resumed from partial cache)")
            continue
            
        test_indices = np.where(subject_ids == test_subj)[0]
        train_pool = unique_subjects[unique_subjects != test_subj]
        
        # Pick 3 validation subjects for checkpoint selection
        rng = np.random.RandomState(42 * 1000 + fold_idx)
        val_subjs = rng.choice(train_pool, 3, replace=False)
        fit_subjs = np.array([s for s in train_pool if s not in val_subjs])
        
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
        
        torch.manual_seed(42 * 1000 + fold_idx)
        
        # Instantiate Model
        is_cib = "CIB-Net" in model_type
        imbalance_method = "none"
        if "V-SMOTE" in model_type:
            imbalance_method = "v_smote"
        elif "GAMO" in model_type:
            imbalance_method = "gamo"
        elif "DeepSMOTE" in model_type:
            imbalance_method = "deepsmote"
            
        if model_type == "ACRNN":
            model = ACRNN(n_channels=64, n_classes=2).to(device)
            criterion = nn.CrossEntropyLoss()
        elif model_type == "PlainCNN":
            model = PlainCNN(n_channels=64, n_classes=2).to(device)
            criterion = nn.CrossEntropyLoss()
        elif model_type == "EEGNet":
            model = EEGNet(n_channels=64, n_classes=2, sample_rate=160).to(device)
            criterion = nn.CrossEntropyLoss()
        else: # CIB-Net variants
            model = CIBNetMotor(n_channels=64, n_classes=2, d_embed=64).to(device)
            invariance_gate = InvarianceGate(n_regions=5, decay=0.9, tau=1.0).to(device)
            sd_loss_fn = SpectralDecouplingLoss(lambda_0=0.001, beta_aux=0.2)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        
        best_val_ba = -1.0
        best_weights = None
        
        for ep in range(1, epochs + 1):
            model.train()
            if is_cib:
                invariance_gate.train()
                
            for x, y, subj, _ in train_loader:
                x, y, subj = x.to(device), y.to(device), subj.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", enabled=(device=="cuda")):
                    if is_cib:
                        final_logits, group_logits, group_feats, _, y_aug, subj_aug, _, aux_loss = model(
                            x, y=y, subj_ids=subj, smote_k=5, imbalance_method=imbalance_method
                        )
                        invariance_gate.update(group_logits, y_aug, subj_aug)
                        inv_gates = invariance_gate.get_gates()
                        loss, _, _, _ = sd_loss_fn(final_logits, y_aug, group_logits, inv_gates, group_feats)
                        if imbalance_method == "deepsmote":
                            loss = loss + 0.1 * aux_loss
                    else:
                        out = model(x)
                        loss = criterion(out, y)
                        
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                
            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for x, y, _, _ in val_loader:
                    x = x.to(device)
                    if is_cib:
                        out = model(x)
                        logits = out[0]
                    else:
                        logits = model(x)
                    val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    val_targets.extend(y.numpy())
                    
            from sklearn.metrics import balanced_accuracy_score
            val_ba = balanced_accuracy_score(val_targets, val_preds)
            if val_ba > best_val_ba:
                best_val_ba = val_ba
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        # Test Evaluation
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model.eval()
        t_preds, t_targets, t_probs = [], [], []
        with torch.no_grad():
            for x, y, subj, _ in test_loader:
                x = x.to(device)
                if is_cib:
                    out = model(x)
                    logits = out[0]
                    bottleneck = out[6] # [B, 64]
                    all_test_embs.extend(bottleneck.cpu().numpy())
                else:
                    logits, feat = model(x, return_feat=True)
                    all_test_embs.extend(feat.cpu().numpy())
                    
                probs = torch.softmax(logits, dim=-1)
                t_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
                t_targets.extend(y.numpy())
                t_probs.extend(probs.cpu().numpy())
                all_test_subjs.extend(subj.numpy())
                
        t_metric = compute_all_metrics(t_targets, t_preds, np.array(t_probs))
        fold_metrics.append(t_metric)
        all_test_targets.extend(t_targets)
        all_test_preds.extend(t_preds)
        all_test_probs.extend(t_probs)
        
        # Save partial progress after each fold
        fold_embs = all_test_embs[-len(t_targets):]
        fold_subjs = all_test_subjs[-len(t_targets):]
        partial_data[str(fold_idx)] = {
            "metric": t_metric,
            "targets": [int(v) for v in t_targets],
            "preds": [int(v) for v in t_preds],
            "probs": [p.tolist() if hasattr(p, 'tolist') else p for p in t_probs],
            "embs": [e.tolist() if hasattr(e, 'tolist') else e for e in fold_embs],
            "subjs": [int(s) for s in fold_subjs]
        }
        with open(partial_file, "w") as f:
            json.dump(partial_data, f)
        
        print(f"[{model_type} Fold {fold_idx+1:02d}/{n_folds:02d}] Subj: S{test_subj:03d} | "
              f"Acc: {t_metric['accuracy']*100:.2f}% | BalAcc: {t_metric['balanced_acc']*100:.2f}% | "
              f"AUROC: {t_metric['auroc']:.4f} | F1: {t_metric['macro_f1']:.4f} | Time: {time.time()-t0:.1f}s")
              
    elapsed = time.time() - start_time
    if os.path.exists(partial_file):
        try:
            os.remove(partial_file)
        except Exception:
            pass
    
    # Aggregate Mean +/- Std
    keys = ["accuracy", "balanced_acc", "auroc", "gini_auroc", "pr_auc", "macro_f1", "mcc", "delta_trivial", "recall_minority", "ece", "minority_pred_rate"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        
    print(f"\n--- {model_type} Summary ({n_folds} Folds, Elapsed: {elapsed/60:.1f} min) ---")
    print(f"Accuracy:        {agg['accuracy']['mean']*100:.2f}% ± {agg['accuracy']['std']*100:.2f}%")
    print(f"Balanced Acc:    {agg['balanced_acc']['mean']*100:.2f}% ± {agg['balanced_acc']['std']*100:.2f}%")
    print(f"AUROC:           {agg['auroc']['mean']:.4f} ± {agg['auroc']['std']:.4f}")
    print(f"Gini AUROC:      {agg['gini_auroc']['mean']:+.4f} ± {agg['gini_auroc']['std']:.4f}")
    print(f"PR AUC:          {agg['pr_auc']['mean']:.4f} ± {agg['pr_auc']['std']:.4f}")
    print(f"Macro F1:        {agg['macro_f1']['mean']:.4f} ± {agg['macro_f1']['std']:.4f}")
    print(f"MCC:             {agg['mcc']['mean']:+.4f} ± {agg['mcc']['std']:.4f}")
    print(f"Delta Trivial:   {agg['delta_trivial']['mean']:+.4f} ± {agg['delta_trivial']['std']:.4f}")
    print(f"Recall Minority: {agg['recall_minority']['mean']:.4f} ± {agg['recall_minority']['std']:.4f}")
    print(f"ECE:             {agg['ece']['mean']:.4f} ± {agg['ece']['std']:.4f}")
    print(f"Minority Pred:   {agg['minority_pred_rate']['mean']:.4f} ± {agg['minority_pred_rate']['std']:.4f}")
    
    # Save artifacts & plots
    results_dir = "CIB-Net-EEGMMIDB/results"
    plots_dir = os.path.join(results_dir, "plots")
    clean_name = model_type.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    
    with open(os.path.join(results_dir, f"fold_results_eegmmidb_{clean_name}.json"), "w") as f:
        json.dump({"summary": agg, "folds": fold_metrics}, f, indent=2)
        
    plot_model_artifacts(
        model_type,
        np.array(all_test_targets),
        np.array(all_test_preds),
        np.array(all_test_probs),
        np.array(all_test_embs),
        np.array(all_test_subjs),
        plots_dir
    )
    
    return agg

