import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from data.windowed_loader import WindowedDEAPDataset
from models.hcan_cib_net import HCANCIBNet
from evaluation.plots import plot_tsne

def generate_tsne(config_path="config/hcan_cib_config.yaml", checkpoint_path=None, test_subject=32, output_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path
    
    with open(full_config_path) as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check dataset path
    data_path = config['data_path']
    if not os.path.exists(data_path):
        # Fallback to local dataset directory if present
        local_deap = os.path.join(os.path.dirname(base_dir), "deap-dataset", "data_preprocessed_python")
        if os.path.exists(local_deap):
            data_path = local_deap
            print(f"Using local DEAP path: {data_path}")
        else:
            raise FileNotFoundError(f"Dataset path not found: {data_path}")

    print(f"Loading DEAP Dataset for test subject {test_subject}...")
    dataset = WindowedDEAPDataset(
        data_path=data_path,
        target=config.get('target', 'valence'),
        binarize=config.get('binarize', 'median'),
        window_sec=config.get('window_sec', 4.0),
        overlap_sec=config.get('overlap_sec', 2.0),
        fs=config.get('sample_rate', 128),
        baseline_sec=config.get('baseline_sec', 3),
        subjects=[test_subject]
    )
    print(f"Test subject {test_subject}: {len(dataset)} windows prepared.")
    
    test_loader = DataLoader(
        dataset,
        batch_size=config.get('training', {}).get('batch_size', 64),
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = HCANCIBNet(
        n_channels=config.get('n_channels', 32),
        d_embed=config.get('d_embed', 64),
        n_classes=config.get('n_classes', 2)
    ).to(device)
    
    # Candidate checkpoint locations
    if checkpoint_path is None:
        candidates = [
            os.path.join(base_dir, "best_model_fold_32.zip"),
            os.path.join(base_dir, "results", "weights", f"best_model_fold_{test_subject}.pth"),
            os.path.join(base_dir, f"best_model_fold_{test_subject}.pth"),
            os.path.join(base_dir, "best_model_fold_32")
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break
            
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint for fold {test_subject} not found in any candidate location!")
        
    model.eval()
    embeds, labels = [], []
    with torch.no_grad():
        for x_eeg, x_periph, y, _, _ in test_loader:
            out = model(x_eeg.to(device), x_periph.to(device))
            z_eeg = out[2]  # z_eeg is index 2
            embeds.append(z_eeg.float().cpu().numpy())
            labels.append(y.numpy())
            
    all_embeds = np.concatenate(embeds)
    all_labels = np.concatenate(labels)
    
    imb_method = config.get("imbalance", {}).get("method", "deepsmote")
    if not output_path:
        output_dir = os.path.join(base_dir, "results", "plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"tsne_brain_embeddings_deap_{imb_method}.png")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_tsne(all_embeds, all_labels, save_path=output_path)
    print(f"Successfully generated t-SNE plot at: {output_path}")

if __name__ == "__main__":
    generate_tsne()
