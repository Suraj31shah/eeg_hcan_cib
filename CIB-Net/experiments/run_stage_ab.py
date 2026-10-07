import sys
import os
import yaml
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.deap_loader import DEAPDataset, get_dataloaders
from data.preprocessing import preprocess_eeg
from models.stage_a_encoder import StageAModel
from evaluation.channel_analysis import extract_embeddings, probe_single_channel_accuracy, plot_channel_dominance

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_encoder_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

def main():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Dataset for Stage A+B Diagnostics...")
    dataset = DEAPDataset(
        data_path=config['data_path'],
        target=config['target'],
        preprocess_fn=lambda x: preprocess_eeg(x, fs=config['sample_rate'], baseline_sec=config['baseline_sec'])
    )
    
    logo = LeaveOneGroupOut()
    subject_ids = dataset.subject_ids
    
    all_subject_channel_accuracies = []
    
    print("\n--- Running Single-Channel Probes across LOSO Folds ---")
    
    # We will train the shared encoder on N-1 subjects, then probe channel representations on Test subject.
    for fold, (train_idx, test_idx) in enumerate(logo.split(dataset.x, dataset.y, groups=subject_ids)):
        test_subject = subject_ids[test_idx[0]]
        print(f"\nFold {fold + 1}/{len(np.unique(subject_ids))} (Test Subject: {test_subject})")
        
        train_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx, batch_size=config['training']['batch_size'])
        
        model = StageAModel(
            n_channels=config['n_channels'], 
            n_classes=config['n_classes'],
            d_embed=config.get('encoder', {}).get('d_embed', 128),
            shared_weights=config.get('encoder', {}).get('shared_weights', True)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
        criterion = nn.CrossEntropyLoss()
        
        # Train the encoder for a few epochs just to learn generic embeddings
        # For diagnostics, we don't need full convergence, just a representative feature space.
        epochs_for_diagnostic = 20
        for epoch in range(epochs_for_diagnostic):
            train_encoder_epoch(model, train_loader, criterion, optimizer, device)
            
        # Extract embeddings
        train_emb, train_targets = extract_embeddings(model, train_loader, device)
        test_emb, test_targets = extract_embeddings(model, test_loader, device)
        
        # Probe each channel
        channel_accs = probe_single_channel_accuracy(train_emb, train_targets, test_emb, test_targets)
        all_subject_channel_accuracies.append(channel_accs)
        
        print(f"Max Channel Acc: {max(channel_accs):.4f} | Min Channel Acc: {min(channel_accs):.4f}")
    
    # Convert to numpy array: [32 subjects, 32 channels]
    acc_matrix = np.array(all_subject_channel_accuracies)
    
    # Save the raw diagnostic array
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "per_subject"), exist_ok=True)
    np.save(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "per_subject", "channel_probe_accuracies.npy"), acc_matrix)
    
    # Generate Figure 1
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
    plot_channel_dominance(acc_matrix, save_dir=plot_dir)
    print("\nStage A+B Diagnostic complete! Figure 1 generated.")

if __name__ == '__main__':
    main()
