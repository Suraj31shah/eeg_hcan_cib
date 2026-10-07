import sys
import os
import yaml
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.deap_loader import DEAPDataset, get_dataloaders
from data.preprocessing import preprocess_eeg
from models.cib_net import CIBNet
from training.losses import SpectralDecouplingLoss
from evaluation.metrics import compute_metrics, print_metrics

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_ablation(config, ablation_type, device):
    print(f"\n--- Running Ablation: {ablation_type} ---")
    
    dataset = DEAPDataset(
        data_path=config['data_path'],
        target=config['target'],
        preprocess_fn=lambda x: preprocess_eeg(x, fs=config['sample_rate'], baseline_sec=config['baseline_sec'])
    )
    
    logo = LeaveOneGroupOut()
    subject_ids = dataset.subject_ids
    
    # Pre-loaded invariance scores (mocked/loaded depending on ablation)
    # In a real run, this would be loaded from Stage B outputs
    invariance_scores = torch.ones(5) # Default: no gating (uniform penalty)
    if "gated" in ablation_type:
        print("Using Stage B invariance scores to gate Spectral Decoupling")
        # Would load from results/per_subject/invariance_scores.npy
        # For now, placeholder simulating invariant channels:
        invariance_scores = torch.tensor([0.9, 0.4, 0.5, 0.2, 0.1]) 
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(dataset.x, dataset.y, groups=subject_ids)):
        test_subject = subject_ids[test_idx[0]]
        print(f"\nFold {fold + 1} (Test Subject: {test_subject})")
        
        train_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx, batch_size=config['training']['batch_size'])
        
        model = CIBNet(n_channels=config['n_channels'], n_classes=config['n_classes']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
        sd_loss_fn = SpectralDecouplingLoss(lambda_base=config['stage_c']['lambda_base'])
        
        for epoch in range(10): # Short epochs for demonstration
            model.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                final_logits, group_logits = model(x)
                
                if "no_sd" in ablation_type:
                    loss = nn.CrossEntropyLoss()(final_logits, y)
                else:
                    loss = sd_loss_fn(final_logits, y, group_logits, invariance_scores.to(device))
                    
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device)
                final_logits, _ = model(x)
                preds = torch.argmax(final_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.numpy())
                
        metrics = compute_metrics(all_targets, all_preds)
        print_metrics(metrics, fold_name="Test")
        break # Break early just to verify pipeline works

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='A+C_no_sd', choices=['A+C_no_sd', 'A+C_sd_uniform', 'A+B+C_sd_gated'])
    args = parser.parse_args()
    
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_ablation(config, args.type, device)
