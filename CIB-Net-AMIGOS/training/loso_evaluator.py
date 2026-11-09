"""
Leakage-free LOSO evaluator for HCAN-CIB on AMIGOS.

Replicates the validation-subject LOSO protocol of loso_evaluator_fixed.py:
  1. Test subject held out cleanly (never participates in epoch selection, threshold tuning, or weights).
  2. Validation subjects drawn deterministically (`RandomState(seed*1000 + fold)`).
  3. Trials keyed by `(subject_id, trial_id)` so trial IDs across validation subjects cannot collide.
  4. Threshold tuned on validation probabilities (balanced accuracy).
  5. Supports all imbalance methods (`none`, `class_weight`, `logit_adj`, `smote`, `v_smote`, `deepsmote`, `gamo`)
     plus ablation flags (`use_gating`, `use_sd`, `ablate_modality`).
  6. Saves `fold_results.json`, `trial_predictions_seed{seed}.npz`, fold checkpoints, and diagnostic plots.
"""

import json
import os
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eeg_common import resolve_n_val
from evaluation.metrics import compute_metrics, print_metrics
from models.stage_b_invariance import InvarianceGate
from training.imbalance import LogitAdjustedLoss, get_class_weights
from training.losses import SpectralDecouplingLoss
from training.subject_sampler import SubjectBalancedSampler


def safe_auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def pooled_val_auroc(pred_dict) -> float:
    return safe_auroc(pred_dict["y"], pred_dict["prob1"])


def tune_threshold(y, p) -> float:
    if len(p) == 0:
        return 0.5
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


def train_epoch(
    model,
    dataloader,
    sd_loss_fn,
    invariance_gate,
    optimizer,
    scaler,
    device,
    smote_k: int = 0,
    use_gating: bool = True,
):
    model.train()
    invariance_gate.train()
    use_amp = device == "cuda"

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for x_eeg, x_periph, y, subj_ids, _ in tqdm(dataloader, desc="Training", leave=False):
        x_eeg, x_periph, y = x_eeg.to(device), x_periph.to(device), y.to(device)
        subj_ids = subj_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            final_logits, group_logits, z_eeg, z_periph, y_aug, subj_aug = model(
                x_eeg, x_periph, y=y, subj_ids=subj_ids, smote_k=smote_k
            )
            invariance_gate.update(group_logits, y_aug, subj_aug)
            if use_gating:
                inv_scores = invariance_gate.get_gates()
            else:
                inv_scores = torch.ones(invariance_gate.n_regions, device=device)

            loss = sd_loss_fn(final_logits, y_aug, group_logits, inv_scores, z_eeg, z_periph)
            if getattr(model, "last_aux_loss", None) is not None:
                loss = loss + model.last_aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1
        all_preds.extend(torch.argmax(final_logits.detach(), dim=1).cpu().numpy())
        all_targets.extend(y_aug.detach().cpu().numpy())

    return total_loss / max(n_batches, 1), compute_metrics(all_targets, all_preds)


@torch.no_grad()
def predict_trials(model, dataloader, device):
    """Average window logits per `(subject_id, trial_id)`."""
    model.eval()
    logits, labels = {}, {}

    for x_eeg, x_periph, y, subj_ids, trial_ids in tqdm(dataloader, desc="Evaluating", leave=False):
        x_eeg, x_periph = x_eeg.to(device), x_periph.to(device)
        final_logits, _, _, _, _, _ = model(x_eeg, x_periph)
        final_logits = np.nan_to_num(final_logits.float().cpu().numpy(), nan=0.0)
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


def run_hcan_cib_loso(
    dataset,
    model_class,
    config_path: str,
    seed: int = 0,
    results_dir: str = "results/hcan_cib",
    config_override: dict = None,
    use_gating: bool = True,
    use_sd: bool = True,
    ablate_modality: str = None,
    max_folds: int = None,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_override:
        for k, v in config_override.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = config["training"]
    subject_ids = np.asarray(dataset.subject_ids)
    subjects = np.unique(subject_ids)

    n_val = resolve_n_val(len(subjects), int(tcfg.get("n_val_subjects", 4)))
    patience = int(tcfg.get("patience", 8))
    min_epochs = int(tcfg.get("min_epochs", 10))
    n_regions = int(config.get("n_regions", 4))
    max_subjects = int(config.get("max_subjects", max(64, int(subjects.max()) + 1)))
    bs = int(tcfg["batch_size"])
    imb_cfg = config.get("imbalance", {})
    imb_method = str(imb_cfg.get("method", "none")).lower()
    smote_k = int(imb_cfg.get("smote_k", 5)) if imb_method in ("v_smote", "smote", "deepsmote") else 0

    print(
        f"--- HCAN-CIB LOSO (AMIGOS) on {device} | seed={seed} | n_subjects={len(subjects)} "
        f"| n_val={n_val} | normalize={config.get('normalize', 'none')} | imbalance={imb_method} "
        f"| gating={use_gating} | sd={use_sd} | ablate={ablate_modality} ---"
    )

    os.makedirs(f"{results_dir}/weights", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
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

        s_fit, s_val, s_te = (
            set(subject_ids[fit_idx]),
            set(subject_ids[val_idx]),
            set(subject_ids[test_idx]),
        )
        assert not (s_fit & s_val) and not (s_fit & s_te) and not (s_val & s_te), "subject leakage"
        print(
            f"\nFold {fold + 1}/{len(subjects)} | test=P{int(test_subject):02d} "
            f"| val={sorted(map(int, val_subjects))}"
        )

        eff_bs = min(bs, max(8, len(fit_idx)))
        train_sampler = SubjectBalancedSampler(dataset, fit_idx, eff_bs)
        train_loader = DataLoader(
            dataset,
            batch_size=eff_bs,
            sampler=train_sampler,
            drop_last=len(fit_idx) >= eff_bs,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=eff_bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=eff_bs, shuffle=False, num_workers=0)

        torch.manual_seed(seed * 1000 + fold)
        model = model_class(
            n_channels=int(config["n_channels"]),
            n_classes=int(config["n_classes"]),
            d_embed=int(config["d_embed"]),
            n_periph_features=int(config.get("n_periph_features", 18)),
            shared_weights=True,
            region_dropout_p=float(config.get("region_dropout_p", 0.2)),
            imbalance_method=imb_method,
            ablate_modality=ablate_modality,
        ).to(device)

        invariance_gate = InvarianceGate(
            n_regions=n_regions,
            decay=float(config.get("decay_gate", 0.9)),
            tau=float(config.get("tau_gate", 1.0)),
            max_subjects=max_subjects,
        ).to(device)

        y_fit = np.asarray(dataset.labels)[fit_idx]
        class_weights_tensor = None
        base_criterion = None
        if imb_method == "class_weight":
            class_weights_tensor = get_class_weights(y_fit).to(device)
        elif imb_method == "logit_adj":
            base_criterion = LogitAdjustedLoss(y_fit, tau=float(imb_cfg.get("logit_adj_tau", 1.0))).to(device)

        lambda_sd = float(config["lambda_sd"]) if use_sd else 0.0
        sd_loss_fn = SpectralDecouplingLoss(
            lambda_sd=lambda_sd,
            alpha_modality=float(config["alpha_modality"]),
            beta_region=float(config["beta_region"]),
            weight=class_weights_tensor,
            base_criterion=base_criterion,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
        )

        best_val, best_epoch, bad = -np.inf, -1, 0
        best_ckpt = None
        for epoch in range(int(tcfg["epochs"])):
            _, train_m = train_epoch(
                model,
                train_loader,
                sd_loss_fn,
                invariance_gate,
                optimizer,
                scaler,
                device,
                smote_k=smote_k,
                use_gating=use_gating,
            )
            val_pred = predict_trials(model, val_loader, device)
            val_auc = pooled_val_auroc(val_pred)
            print(
                f"  Epoch {epoch + 1:02d}/{tcfg['epochs']} | Train Acc: {train_m['accuracy']:.4f} "
                f"| Val AUROC (pooled): {val_auc:.4f}"
            )

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

        if best_ckpt is None:
            best_ckpt = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "gate": {k: v.detach().cpu().clone() for k, v in invariance_gate.state_dict().items()},
            }
            best_epoch = int(tcfg["epochs"])

        model.load_state_dict(best_ckpt["model"])
        invariance_gate.load_state_dict(best_ckpt["gate"])
        torch.save(best_ckpt, f"{results_dir}/weights/best_model_fold_{fold + 1}.pth")

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
        m["gate_values"] = (
            invariance_gate.get_gates().detach().cpu().numpy().tolist()
            if use_gating
            else [1.0] * n_regions
        )

        print_metrics(m, fold_name=f"Test P{int(test_subject):02d} (epoch {best_epoch})")
        print(
            f"  [val-thr {thr:.3f}] bal_acc={m_thr.get('balanced_acc', float('nan')):.4f} "
            f"| macro_f1={m_thr.get('macro_f1', float('nan')):.4f} "
            f"| min_pred_rate={m_thr.get('minority_pred_rate', float('nan')):.2f}"
        )
        fold_metrics.append(m)
        fold_preds.append(te)

    print("\n=== FINAL AGGREGATED RESULTS (AMIGOS, validation-selected, test scored once) ===")
    for key in [
        "accuracy",
        "balanced_acc",
        "auroc",
        "gini_auroc",
        "pr_auc",
        "macro_f1",
        "mcc",
        "delta_trivial",
        "recall_minority",
        "ece",
    ]:
        if fold_metrics and key in fold_metrics[0]:
            v = np.array([fm[key] for fm in fold_metrics], dtype=float)
            print(f"{key}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")
    print("-- with validation-tuned threshold --")
    for key in [
        "balanced_acc",
        "macro_f1",
        "mcc",
        "recall_minority",
        "precision_minority",
        "minority_pred_rate",
    ]:
        if fold_metrics and key in fold_metrics[0]["val_thr_metrics"]:
            v = np.array([fm["val_thr_metrics"][key] for fm in fold_metrics], dtype=float)
            print(f"{key}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    aurocs = np.array([fm["auroc"] for fm in fold_metrics], dtype=float)
    valid_auc = aurocs[np.isfinite(aurocs)]
    if len(valid_auc) > 0:
        boots = np.random.RandomState(0).choice(valid_auc, (5000, len(valid_auc))).mean(1)
        print(
            f"mean per-subject AUROC {np.mean(valid_auc):.4f}  95% bootstrap CI "
            f"[{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]"
        )

    with open(f"{results_dir}/fold_results.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)
    np.savez(
        f"{results_dir}/trial_predictions_seed{seed}.npz",
        **{f"fold{i + 1}_{k}": v for i, p in enumerate(fold_preds) for k, v in p.items()},
    )
    print(f"\nSaved fold_results.json and trial_predictions_seed{seed}.npz to '{results_dir}/'")

    # Diagnostic plots using best checkpoint of last fold + full gate heatmap
    try:
        from evaluation.plots import plot_region_gate_heatmap, plot_roc_pr_curves, plot_tsne

        plot_region_gate_heatmap(fold_metrics, save_path=f"{results_dir}/plots/gate_heatmap.png")
        plot_roc_pr_curves(
            [p["y"] for p in fold_preds],
            [p["prob1"] for p in fold_preds],
            save_dir=f"{results_dir}/plots",
        )

        # Extract embeddings on held-out test fold + subsample of dataset for class & subject t-SNE
        model.eval()
        all_idx = np.arange(len(dataset))
        if len(all_idx) > 2000:
            all_idx = np.random.RandomState(42).choice(all_idx, 2000, replace=False)
        emb_loader = DataLoader(Subset(dataset, all_idx), batch_size=eff_bs, shuffle=False, num_workers=0)
        embeds, lbls, subjs = [], [], []
        with torch.no_grad():
            for x_eeg, x_periph, y, s_id, _ in emb_loader:
                ch_emb = model.eeg_encoder(x_eeg.to(device))
                group_feats, _ = model.region_tokenizer.extract_region_tokens(ch_emb)
                pooled_eeg = torch.mean(torch.stack(group_feats, dim=1), dim=1)
                embeds.append(pooled_eeg.float().cpu().numpy())
                lbls.append(y.numpy())
                subjs.append(s_id.numpy())
        plot_tsne(
            np.concatenate(embeds),
            np.concatenate(lbls),
            subjects=np.concatenate(subjs),
            save_path=f"{results_dir}/plots/tsne_brain_embeddings.png",
        )
    except Exception as e:
        import traceback
        print(f"Plotting skipped due to error: {e}")
        traceback.print_exc()

    return fold_metrics

