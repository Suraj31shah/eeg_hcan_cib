"""
LOSO evaluator with leakage-free model selection.

What changed vs. the original loso_evaluator.py
-----------------------------------------------
1. Epoch selection uses held-out VALIDATION SUBJECTS (taken from the training
   subjects), never the test subject. The test subject is scored exactly once,
   with the checkpoint chosen on validation AUROC.
2. Trials are keyed by (subject_id, trial_id), so trial ids that repeat across
   subjects (0..17 per subject) cannot merge in the multi-subject validation set.
3. The decision threshold is tuned on validation subjects (balanced accuracy)
   and reported next to the default argmax result. This addresses the
   constant-prediction collapse separately from AUROC.
4. Gate values are snapshotted at the selected epoch (not the last epoch), and
   the gate state is saved in the checkpoint.
5. GradScaler is created once per run, not once per epoch.
6. `os` / `json` imports moved to module level. In the pasted file `import os`
   sat inside the function after its first use, which raises UnboundLocalError.
7. n_regions comes from config (DREAMER = 4, DEAP = 5), not hardcoded.
8. Per-trial predictions are saved so Wilcoxon / bootstrap / pooled AUROC can be
   computed offline across models.
"""
import os
import json
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from evaluation.metrics import compute_metrics, print_metrics
from training.subject_sampler import SubjectBalancedSampler
from models.stage_b_invariance import InvarianceGate
from training.losses import SpectralDecouplingLoss


# --------------------------------------------------------------------------- #
# Training / evaluation primitives
# --------------------------------------------------------------------------- #
def train_epoch(model, dataloader, sd_loss_fn, invariance_gate, optimizer, scaler, device):
    model.train()
    invariance_gate.train()
    use_amp = device == "cuda"

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for x_eeg, x_periph, y, subj_ids, trial_ids in tqdm(dataloader, desc="Training", leave=False):
        x_eeg, x_periph, y = x_eeg.to(device), x_periph.to(device), y.to(device)
        subj_ids = subj_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            final_logits, group_logits, z_eeg, z_periph = model(x_eeg, x_periph)
            invariance_gate.update(group_logits, y, subj_ids)
            inv_scores = invariance_gate.get_gates()
            loss = sd_loss_fn(final_logits, y, group_logits, inv_scores, z_eeg, z_periph)

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


@torch.no_grad()
def predict_trials(model, dataloader, device):
    """
    Average window logits per trial. Trials are keyed by (subject, trial_id).
    Returns arrays: subj, y, prob1 (softmax of mean logits), pred (argmax).
    """
    model.eval()
    logits, labels = {}, {}

    for x_eeg, x_periph, y, subj_ids, trial_ids in tqdm(dataloader, desc="Evaluating", leave=False):
        x_eeg, x_periph = x_eeg.to(device), x_periph.to(device)
        final_logits, _, _, _ = model(x_eeg, x_periph)
        final_logits = final_logits.float().cpu().numpy()
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
        "y": np.array([labels[k] for k in keys]),
        "prob1": probs[:, 1],
        "pred": mean_logits.argmax(axis=1),
    }


def safe_auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def pooled_val_auroc(pred_dict):
    """AUROC over all validation trials pooled. Stable with ~70 trials, but it mixes
    subject offsets. Mean per-subject AUROC is the alternative; it is noisier."""
    return safe_auroc(pred_dict["y"], pred_dict["prob1"])


def tune_threshold(y, p):
    """Threshold on P(class 1) that maximizes balanced accuracy on validation trials."""
    cands = np.unique(np.concatenate([np.quantile(p, np.linspace(0.05, 0.95, 37)), [0.5]]))
    best_t, best_ba = 0.5, -1.0
    for t in cands:
        pred = (p >= t).astype(int)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba + 1e-12:
            best_ba, best_t = ba, float(t)
    return best_t


def to_py(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# Main LOSO loop
# --------------------------------------------------------------------------- #
def run_hcan_cib_loso(dataset, model_class, config_path, seed=0, results_dir="results"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = config["training"]
    n_val = tcfg.get("n_val_subjects", 4)
    patience = tcfg.get("patience", 8)
    min_epochs = tcfg.get("min_epochs", 10)
    n_regions = config.get("n_regions", 4)
    bs = tcfg["batch_size"]
    print(f"--- HCAN-CIB LOSO (val-subject selection) on {device} | seed={seed} | "
          f"n_regions={n_regions} | n_val_subjects={n_val} ---")

    os.makedirs(f"{results_dir}/weights", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)

    subject_ids = np.asarray(dataset.subject_ids)
    subjects = np.unique(subject_ids)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    fold_metrics, fold_preds = [], []

    for fold, test_subject in enumerate(subjects):
        # ---- split: test subject / validation subjects / fit subjects ----
        test_idx = np.where(subject_ids == test_subject)[0]
        pool_idx = np.where(subject_ids != test_subject)[0]
        rng = np.random.RandomState(seed * 1000 + fold)
        val_subjects = rng.choice(subjects[subjects != test_subject], n_val, replace=False)
        is_val = np.isin(subject_ids[pool_idx], val_subjects)
        val_idx, fit_idx = pool_idx[is_val], pool_idx[~is_val]

        s_fit, s_val, s_te = (set(subject_ids[fit_idx]), set(subject_ids[val_idx]),
                              set(subject_ids[test_idx]))
        assert not (s_fit & s_val) and not (s_fit & s_te) and not (s_val & s_te), "subject leakage"
        print(f"\nFold {fold + 1}/{len(subjects)} | test={test_subject} | val={sorted(map(int, val_subjects))}")

        train_sampler = SubjectBalancedSampler(dataset, fit_idx, bs)
        train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler,
                                  drop_last=True, num_workers=0, pin_memory=(device == "cuda"))
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=bs, shuffle=False, num_workers=0)

        torch.manual_seed(seed * 1000 + fold)
        model = model_class(
            n_channels=config["n_channels"], n_classes=config["n_classes"],
            d_embed=config["d_embed"], n_periph_features=config.get("n_periph_features", 24),
            shared_weights=True, region_dropout_p=0.2,
        ).to(device)
        invariance_gate = InvarianceGate(n_regions=n_regions, decay=0.9, tau=1.0).to(device)
        sd_loss_fn = SpectralDecouplingLoss(
            lambda_sd=config["lambda_sd"], alpha_modality=config["alpha_modality"],
            beta_region=config["beta_region"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

        # ---- train with validation-based model selection ----
        best_val, best_epoch, bad = -np.inf, -1, 0
        best_ckpt = None
        for epoch in range(tcfg["epochs"]):
            _, train_m = train_epoch(model, train_loader, sd_loss_fn, invariance_gate,
                                     optimizer, scaler, device)
            val_pred = predict_trials(model, val_loader, device)
            val_auc = pooled_val_auroc(val_pred)
            print(f"  Epoch {epoch + 1:02d}/{tcfg['epochs']} | Train Acc: {train_m['accuracy']:.4f} "
                  f"| Val AUROC (pooled): {val_auc:.4f}")

            if np.isfinite(val_auc) and val_auc > best_val + 1e-4:
                best_val, best_epoch, bad = val_auc, epoch + 1, 0
                best_ckpt = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "gate": {k: v.detach().cpu().clone() for k, v in invariance_gate.state_dict().items()},
                }
            else:
                bad += 1
                if epoch + 1 >= min_epochs and bad >= patience:
                    print(f"  Early stop at epoch {epoch + 1} (best {best_epoch}, val AUROC {best_val:.4f})")
                    break

        if best_ckpt is None:  # val AUROC never finite; fall back to final weights
            best_ckpt = {"model": model.state_dict(), "gate": invariance_gate.state_dict()}
            best_epoch = epoch + 1
        model.load_state_dict(best_ckpt["model"])
        invariance_gate.load_state_dict(best_ckpt["gate"])
        torch.save(best_ckpt, f"{results_dir}/weights/best_model_fold_{fold + 1}.pth")

        # ---- threshold from validation, then ONE test evaluation ----
        val_pred = predict_trials(model, val_loader, device)
        thr = tune_threshold(val_pred["y"], val_pred["prob1"])
        te = predict_trials(model, test_loader, device)

        m = compute_metrics(te["y"], te["pred"], y_prob=te["prob1"])
        m_thr = compute_metrics(te["y"], (te["prob1"] >= thr).astype(int), y_prob=te["prob1"])
        m = to_py(m)
        m["val_thr_metrics"] = to_py(m_thr)
        m["val_threshold"] = thr
        m["best_epoch"] = int(best_epoch)
        m["val_auroc"] = float(best_val)
        m["test_subject"] = int(test_subject)
        m["val_subjects"] = [int(s) for s in val_subjects]
        m["gate_values"] = invariance_gate.get_gates().detach().cpu().numpy().tolist()

        print_metrics(m, fold_name=f"Test (val-selected epoch {best_epoch})")
        print(f"  [val-threshold {thr:.3f}] bal_acc={m_thr.get('balanced_acc', float('nan')):.4f} "
              f"| macro_f1={m_thr.get('macro_f1', float('nan')):.4f} "
              f"| minority_pred_rate={m_thr.get('minority_pred_rate', float('nan')):.2f}")
        fold_metrics.append(m)
        fold_preds.append(te)

    # ---- aggregate ----
    print("\n=== FINAL AGGREGATED RESULTS (validation-selected, test scored once) ===")
    for key in ["accuracy", "balanced_acc", "auroc", "gini_auroc", "pr_auc", "macro_f1", "mcc",
                "delta_trivial", "recall_minority", "ece"]:
        if key in fold_metrics[0]:
            v = np.array([fm[key] for fm in fold_metrics], dtype=float)
            print(f"{key}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")
    print("-- with validation-tuned threshold --")
    for key in ["balanced_acc", "macro_f1", "mcc", "recall_minority", "precision_minority", "minority_pred_rate"]:
        if key in fold_metrics[0]["val_thr_metrics"]:
            v = np.array([fm["val_thr_metrics"][key] for fm in fold_metrics], dtype=float)
            print(f"{key}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    # pooled AUROC and subject-level bootstrap CI of the mean AUROC
    aurocs = np.array([fm["auroc"] for fm in fold_metrics], dtype=float)
    boots = np.random.RandomState(0).choice(aurocs[np.isfinite(aurocs)], (5000, np.isfinite(aurocs).sum())).mean(1)
    print(f"mean per-subject AUROC {np.nanmean(aurocs):.4f}  95% bootstrap CI "
          f"[{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]  (0.5 = chance)")

    with open(f"{results_dir}/fold_results.json", "w") as f:
        json.dump(fold_metrics, f, indent=4)
    np.savez(f"{results_dir}/trial_predictions_seed{seed}.npz",
             **{f"fold{i + 1}_{k}": v for i, p in enumerate(fold_preds) for k, v in p.items()})
    print(f"\nSaved fold results and per-trial predictions to '{results_dir}/'")

    # ---- plots (best-epoch model of the last fold) ----
    try:
        from evaluation.plots import plot_region_gate_heatmap, plot_tsne
        plot_region_gate_heatmap(fold_metrics, save_path=f"{results_dir}/plots/gate_heatmap.png")
        model.eval()
        embeds, labels = [], []
        with torch.no_grad():
            for x_eeg, x_periph, y, _, _ in test_loader:
                _, _, z_eeg, _ = model(x_eeg.to(device), x_periph.to(device))
                embeds.append(z_eeg.float().cpu().numpy())
                labels.append(y.numpy())
        plot_tsne(np.concatenate(embeds), np.concatenate(labels),
                  save_path=f"{results_dir}/plots/tsne_brain_embeddings.png")
    except Exception as e:  # plotting must never kill a finished run
        print(f"Plotting skipped: {e}")

    return fold_metrics
