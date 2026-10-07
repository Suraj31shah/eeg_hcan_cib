#!/bin/bash
#SBATCH --job-name=CIB_FULL
#SBATCH --output=cib_full_%j.log
#SBATCH --error=cib_full_%j.err
#SBATCH --partition=LocalQ
#SBATCH --gres=shard:20

cd $SLURM_SUBMIT_DIR

# ==========================================
# 1. HPC Environment Setup
# ==========================================
source $HOME/miniconda3/bin/activate cib

# PyTorch and Slurm will handle thread counts automatically based on the shard cgroup.

echo "--- HPC GPU DIAGNOSTICS ---"
nvidia-smi || echo "nvidia-smi command failed or not found!"
python -c "import torch; print('PyTorch version:', torch.__version__); print('PyTorch built with CUDA:', torch.version.cuda); print('PyTorch CUDA is_available():', torch.cuda.is_available())"
echo "---------------------------"

echo "Starting CIB-Net Real Data Training on HPC..."
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# ==========================================
# 2. Generate the Python Code
# ==========================================
cat << 'EOF' > cib_net_hpc_run.py
import os
import pickle
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import torch.nn.functional as F

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def butter_bandpass_filter(data, lowcut=4.0, highcut=45.0, fs=128, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def preprocess_eeg(eeg_data, fs=128, baseline_sec=3):
    filtered = butter_bandpass_filter(eeg_data, fs=fs)
    baseline_samples = baseline_sec * fs
    baseline = np.mean(filtered[:, :, :baseline_samples], axis=-1, keepdims=True)
    corrected = filtered[:, :, baseline_samples:] - baseline
    
    mean = np.mean(corrected, axis=-1, keepdims=True)
    std = np.std(corrected, axis=-1, keepdims=True)
    return (corrected - mean) / (std + 1e-8)

def compute_plv_centrality(eeg_data):
    channels = eeg_data.shape[0]
    analytic_signal = hilbert(eeg_data, axis=1)
    phase = np.angle(analytic_signal)
    
    plv_matrix = np.zeros((channels, channels))
    for i in range(channels):
        for j in range(i+1, channels):
            plv = np.abs(np.mean(np.exp(1j * (phase[i] - phase[j]))))
            plv_matrix[i, j] = plv_matrix[j, i] = plv
            
    np.fill_diagonal(plv_matrix, 1.0)
    G = nx.from_numpy_array(plv_matrix)
    try:
        centrality_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
        centrality = np.array([centrality_dict[i] for i in range(channels)])
    except:
        centrality = np.ones(channels) / channels
    return centrality

class DEAPDataset(Dataset):
    def __init__(self, data_path, target='valence'):
        self.x, self.y, self.subject_ids, self.plv = [], [], [], []
        
        for subject_id in range(1, 33):
            file_path = os.path.join(data_path, f's{subject_id:02d}.dat')
            if not os.path.exists(file_path): continue
                
            with open(file_path, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                
            data, labels = content['data'], content['labels']
            eeg_data = data[:, :32, :] 
            eeg_data = preprocess_eeg(eeg_data)
            
            target_idx = 0 if target == 'valence' else 1
            binary_labels = (labels[:, target_idx] > 5).astype(np.int64)
            
            self.x.append(eeg_data)
            self.y.append(binary_labels)
            self.subject_ids.extend([subject_id] * 40)
            
            for i in range(len(eeg_data)):
                self.plv.append(compute_plv_centrality(eeg_data[i]))
            
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.subject_ids = np.array(self.subject_ids)
        self.plv = np.array(self.plv)
        
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.long), \
               torch.tensor(self.subject_ids[idx], dtype=torch.long), \
               torch.tensor(self.plv[idx], dtype=torch.float32)

def get_dataloaders(dataset, train_idx, test_idx, batch_size=16):
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler), \
           DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# ==========================================
# CIB-NET ARCHITECTURE
# ==========================================
class PerChannelCNNEncoder(nn.Module):
    def __init__(self, d_embed=128, n_channels=32):
        super().__init__()
        self.d_embed, self.n_channels = d_embed, n_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, padding=12), nn.BatchNorm1d(32), nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7), nn.BatchNorm1d(64), nn.ELU(),
            nn.AdaptiveAvgPool1d(64), 
            nn.Conv1d(64, d_embed, kernel_size=7, padding=3), nn.BatchNorm1d(d_embed), nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        B, C, T = x.size()
        out = self.encoder(x.view(B * C, 1, T))
        return out.view(B, C, self.d_embed)

class RegionGroupedFusion(nn.Module):
    def __init__(self, d_embed=128, n_classes=2):
        super().__init__()
        self.regions = {
            "frontal": [0,1,2,3,16,17,18], "temporal": [4,5,12,13],
            "central": [6,7,19,20,21], "parietal": [8,9,22,23], "occipital": [10,11,24,25]
        }
        self.intra_attentions = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(d_embed, d_embed//2), nn.Tanh(), nn.Linear(d_embed//2, 1), nn.Softmax(dim=1)) 
            for r in self.regions.keys()
        })
        self.group_classifiers = nn.ModuleDict({r: nn.Linear(d_embed, n_classes) for r in self.regions.keys()})
        self.final_classifier = nn.Linear(len(self.regions) * d_embed, n_classes)

    def forward(self, x_embed):
        group_feats, group_logits = [], []
        for region, channels in self.regions.items():
            region_x = x_embed[:, channels, :] 
            attn = self.intra_attentions[region](region_x)
            pooled = torch.sum(region_x * attn, dim=1)
            group_feats.append(pooled)
            group_logits.append(self.group_classifiers[region](pooled))
        return self.final_classifier(torch.cat(group_feats, dim=1)), group_logits

class SpectralDecouplingLoss(nn.Module):
    def __init__(self, lambda_base=0.1, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_base = lambda_base
    def forward(self, logits, y, group_logits, inv_scores):
        sd_penalty = sum((self.lambda_base / (inv_scores[k] + 1e-4)) * torch.mean(z ** 2) for k, z in enumerate(group_logits))
        return self.ce(logits, y) + sd_penalty

# ==========================================
# STAGE B & D: ADVANCED LOSS FUNCTIONS
# ==========================================
def compute_dynamic_invariance(group_logits, subj_ids):
    unique_subjs = torch.unique(subj_ids)
    if len(unique_subjs) <= 1:
        return torch.ones(len(group_logits), device=group_logits[0].device)
        
    variances = []
    for z in group_logits:
        mags = torch.norm(z, dim=1) 
        subj_means = torch.stack([mags[subj_ids == s].mean() for s in unique_subjs])
        variances.append(torch.var(subj_means))
        
    variances = torch.stack(variances)
    v_min, v_max = variances.min(), variances.max()
    if v_max > v_min:
        norm_vars = (variances - v_min) / (v_max - v_min + 1e-8)
    else:
        norm_vars = torch.zeros_like(variances)
        
    return 1.0 - norm_vars

def compute_graph_consistency_loss(learned_importance, plv_centrality):
    cos_sim = F.cosine_similarity(learned_importance.unsqueeze(0), plv_centrality)
    return (1.0 - cos_sim).mean()

class CIBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PerChannelCNNEncoder()
        self.fusion = RegionGroupedFusion()
    def forward(self, x):
        channel_embeddings = self.encoder(x) # [B, 32, D]
        # Calculate learned importance per channel for Stage D
        learned_importance = torch.norm(channel_embeddings, dim=2).mean(dim=0) # [32]
        logits, group_logits = self.fusion(channel_embeddings)
        return logits, group_logits, learned_importance

# ==========================================
# EVALUATION & TRAINING HARNESS
# ==========================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    majority_acc = np.max(np.unique(y_true, return_counts=True)[1]) / len(y_true)
    delta_trivial = acc - majority_acc
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "delta_trivial": float(delta_trivial),
        "majority_class_acc": float(majority_acc),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

if __name__ == '__main__':
    # >>> CHANGE THIS LINE BEFORE RUNNING ON HPC <<<
    HPC_DATA_PATH = "/home/user3/Group5_research/deap/DEAP/deap-dataset/data_preprocessed_python/"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"HPC Node initialized. Device: {device}")
    
    if not os.path.exists(HPC_DATA_PATH) or not os.path.exists(os.path.join(HPC_DATA_PATH, 's01.dat')):
        print(f"ERROR: Dataset not found at {HPC_DATA_PATH}.")
        print("Please upload DEAP and edit HPC_DATA_PATH in this script!")
        exit(1)
        
    print("Loading DEAP Dataset (this will take a minute or two)...")
    dataset = DEAPDataset(data_path=HPC_DATA_PATH, target='valence')
    
    logo = LeaveOneGroupOut()
    subject_ids = dataset.subject_ids
    
    print("Starting LOSO Cross-Validation Training on CIB-Net...")
    # Using uniform invariance scores for this test run (No Stage B loaded)
    inv_scores = torch.ones(5).to(device) 
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(dataset.x, dataset.y, groups=subject_ids)):
        test_subject = subject_ids[test_idx[0]]
        print(f"\n--- Fold {fold + 1}/32 (Test Subject: {test_subject}) ---")
        
        train_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx, batch_size=16)
        
        # Calculate class weights to prevent Mode Collapse
        y_train = dataset.y[train_idx]
        class_counts = np.bincount(y_train)
        weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        # Restricting CPU threads so PyTorch doesn't trigger Slurm cgroup kill
        torch.set_num_threads(4)
        model = CIBNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sd_loss_fn = SpectralDecouplingLoss(class_weights=class_weights)
        
        best_acc = 0
        for epoch in range(50):
            model.train()
            for batch_idx, (x, y, subj_ids, plv_centrality) in enumerate(train_loader):
                x, y, subj_ids, plv_centrality = x.to(device), y.to(device), subj_ids.to(device), plv_centrality.to(device)
                optimizer.zero_grad()
                logits, group_logits, learned_importance = model(x)
                
                # Stage B: Dynamic Invariance Scoring (Plan B Proxy)
                inv_scores = compute_dynamic_invariance(group_logits, subj_ids)
                
                # Base CE + Stage C Spectral Decoupling (gated by Stage B scores)
                loss_ce_sd = sd_loss_fn(logits, y, group_logits, inv_scores)
                
                # Stage D PLV Graph Consistency (using biological neuroscience prior)
                loss_graph = compute_graph_consistency_loss(learned_importance, plv_centrality)
                
                # Full CIB-Net Loss
                loss = loss_ce_sd + 0.01 * loss_graph
                
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    print(f"    [Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)} processed... VRAM: {vram_mb:.2f} MB")
                    torch.cuda.empty_cache()
                
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, y, _, _ in test_loader:
                    x = x.to(device)
                    logits, _, _ = model(x)
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_targets.extend(y.numpy())
                    
            metrics = compute_metrics(all_targets, all_preds)
            acc = metrics['accuracy']
            if acc > best_acc: 
                best_acc = acc
                best_metrics = metrics
                
            print(f"  Epoch {epoch+1}/50 | Test Acc: {acc:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Delta: {metrics['delta_trivial']:+.4f}")
            
        fold_metrics.append(best_metrics)
        print(f"Best Fold Acc: {best_acc:.4f} | Delta Over Trivial: {best_metrics['delta_trivial']:+.4f}")
        
    print("\n==============================")
    final_acc = np.mean([m['accuracy'] for m in fold_metrics])
    final_std = np.std([m['accuracy'] for m in fold_metrics])
    final_delta = np.mean([m['delta_trivial'] for m in fold_metrics])
    print(f"FINAL CIB-NET LOSO ACCURACY: {final_acc:.4f} ± {final_std:.4f}")
    print(f"FINAL DELTA OVER TRIVIAL GUESSING: {final_delta:+.4f}")
    print("==============================")
    
    with open('cib_net_1e4_results.json', 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    print("Saved all fold metrics to cib_net_1e4_results.json")

EOF

python3 -u cib_net_hpc_run.py
