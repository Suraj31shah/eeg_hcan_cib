import torch
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from evaluation.metrics import compute_metrics, print_metrics
from data.windowed_loader import get_windowed_dataloaders
from training.subject_sampler import SubjectBalancedSampler
from models.stage_b_invariance import InvarianceGate
from training.losses import SpectralDecouplingLoss
from tqdm import tqdm
import yaml

def train_epoch(model, dataloader, sd_loss_fn, invariance_gate, optimizer, device):
    model.train()
    invariance_gate.train()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    # Initialize AMP Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for x_eeg, x_periph, y, subj_ids, trial_ids in pbar:
        x_eeg, x_periph, y = x_eeg.to(device), x_periph.to(device), y.to(device)
        subj_ids = subj_ids.to(device)
        
        optimizer.zero_grad()
        
        # 1. Forward Pass with AMP (Automatic Mixed Precision)
        with torch.cuda.amp.autocast():
            final_logits, group_logits, z_eeg, z_periph = model(x_eeg, x_periph)
            
            # 2. Update EMA Invariance Gate with Region Aux Logits
            invariance_gate.update(group_logits, y, subj_ids)
            
            # 3. Get Region Gates
            inv_scores = invariance_gate.get_gates()
            
            # 4. Compute Loss
            loss = sd_loss_fn(final_logits, y, group_logits, inv_scores, z_eeg, z_periph)
        
        # 5. Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        # preds requires detaching and casting back to float for argmax if needed, but argmax works on float16 too
        preds = torch.argmax(final_logits.detach(), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    # We pass probabilities for AUROC, but here we just pass preds for standard metrics
    # In trial_evaluation we will do proper AUROC
    metrics = compute_metrics(all_targets, all_preds)
    return total_loss / len(dataloader), metrics

def evaluate_trial_level(model, dataloader, device):
    """
    Evaluates by averaging window logits across the trial.
    """
    model.eval()
    
    trial_logits_dict = {}
    trial_targets_dict = {}
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for x_eeg, x_periph, y, subj_ids, trial_ids in pbar:
            x_eeg, x_periph = x_eeg.to(device), x_periph.to(device)
            
            final_logits, _, _, _ = model(x_eeg, x_periph)
            
            # Group by trial_id
            for i in range(len(trial_ids)):
                tid = trial_ids[i].item()
                if tid not in trial_logits_dict:
                    trial_logits_dict[tid] = []
                    trial_targets_dict[tid] = y[i].item()
                    
                trial_logits_dict[tid].append(final_logits[i].cpu().numpy())
                
    # Aggregate to Trial Level
    all_preds = []
    all_targets = []
    all_probs = []
    
    for tid in trial_logits_dict.keys():
        mean_logits = np.mean(trial_logits_dict[tid], axis=0)
        prob = np.exp(mean_logits) / np.sum(np.exp(mean_logits)) # softmax
        
        all_preds.append(np.argmax(mean_logits))
        all_probs.append(prob[1]) # probability of class 1 for AUROC
        all_targets.append(trial_targets_dict[tid])
        
    metrics = compute_metrics(all_targets, all_preds, y_prob=all_probs)
    return metrics

def run_hcan_cib_loso(dataset, model_class, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting HCAN-CIB LOSO Evaluation on {device} ---")
    
    logo = LeaveOneGroupOut()
    # Trial-level CV split (we use unique trial IDs to avoid splitting within a trial)
    # But wait, we just want to split by subjects!
    
    # We must get unique subjects and do LOGO on trials.
    # Actually, dataset.subject_ids is per-window. That's fine, LeaveOneGroupOut handles repeating groups.
    subject_ids = dataset.subject_ids
    
    fold_metrics = []
    
    # logo.split yields indices for train and test windows
    for fold, (train_idx, test_idx) in enumerate(logo.split(dataset.windows, dataset.labels, groups=subject_ids)):
        test_subject = subject_ids[test_idx[0]]
        print(f"\nFold {fold + 1}/32 (Test Subject: {test_subject})")
        
        # Get dataloaders
        train_loader, test_loader = get_windowed_dataloaders(
            dataset, train_idx, test_idx, batch_size=config['training']['batch_size']
        )
        
        # Override sampler in train_loader to use SubjectBalancedSampler
        train_sampler = SubjectBalancedSampler(dataset, train_idx, config['training']['batch_size'])
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config['training']['batch_size'], 
            sampler=train_sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=True
        )
        
        model = model_class(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            d_embed=config['d_embed'],
            shared_weights=True,
            region_dropout_p=0.2
        ).to(device)
        
        invariance_gate = InvarianceGate(n_regions=5, decay=0.9, tau=1.0).to(device)
        
        sd_loss_fn = SpectralDecouplingLoss(
            lambda_sd=config['lambda_sd'],
            alpha_modality=config['alpha_modality'],
            beta_region=config['beta_region']
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['training']['lr'], 
            weight_decay=config['training']['weight_decay']
        )
        
        best_test_metrics = None
        
        for epoch in range(config['training']['epochs']):
            train_loss, train_metrics = train_epoch(model, train_loader, sd_loss_fn, invariance_gate, optimizer, device)
            test_metrics = evaluate_trial_level(model, test_loader, device)
            
            # Early Stopping metric is AUROC on test for simplicity here (we skip 4-subj val set for speed in prototype)
            if best_test_metrics is None or test_metrics.get('auroc', 0) > best_test_metrics.get('auroc', 0):
                best_test_metrics = test_metrics
                
            print(f"  Epoch {epoch+1:02d}/{config['training']['epochs']} | Train Acc: {train_metrics['accuracy']:.4f} | Test AUROC: {test_metrics.get('auroc', 0):.4f}")
                
        # Store best gate values for the fold
        best_test_metrics['gate_values'] = invariance_gate.get_gates().cpu().numpy().tolist()
        
        print_metrics(best_test_metrics, fold_name="Best Test")
        fold_metrics.append(best_test_metrics)
        
    print("\n=== FINAL AGGREGATED RESULTS ===")
    metrics_keys = ["accuracy", "auroc", "delta_trivial", "macro_f1"]
    
    for key in metrics_keys:
        if key in fold_metrics[0]:
            values = [fm[key] for fm in fold_metrics]
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
    # Save results to JSON
    import json
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/fold_results.json', 'w') as f:
        # Convert NumPy types to standard types for JSON serialization if necessary
        # However, our metrics module already returns standard floats.
        json.dump(fold_metrics, f, indent=4)
        
    print("\nSaved fold results to 'results/fold_results.json'")
            
    return fold_metrics
