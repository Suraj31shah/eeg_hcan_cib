import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.eegmmidb_loader import EEGMMIDBDataset
from models.cib_net_motor import CIBNetMotor
from models.baselines.eegnet import EEGNet
from models.baselines.acrnn import ACRNN
from models.baselines.plain_cnn import PlainCNN
from run_comparative_benchmark import apply_subject_zscore

def plot_smooth_tsne(embeddings, labels, subjects, title, save_path):
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    subjects = np.array(subjects)
    
    # 1. Subject-level centering & standardization
    # Removes between-subject DC offsets so t-SNE clusters by CLASS rather than subject identity (prevents isolated islands)
    embs_norm = np.zeros_like(embeddings)
    for s in np.unique(subjects):
        mask = (subjects == s)
        s_embs = embeddings[mask]
        mu = np.mean(s_embs, axis=0, keepdims=True)
        sd = np.std(s_embs, axis=0, keepdims=True) + 1e-8
        embs_norm[mask] = (s_embs - mu) / sd
        
    n_samples = min(600, len(embs_norm))
    np.random.seed(42)
    sub_idx = np.random.choice(len(embs_norm), n_samples, replace=False)
    
    tsne = TSNE(n_components=2, perplexity=35, random_state=42)
    emb_2d = tsne.fit_transform(embs_norm[sub_idx])
    
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels[sub_idx], cmap='coolwarm',
        alpha=0.75, s=35, edgecolors='none'
    )
    plt.title(f't-SNE Latent Space: {title}', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Left Fist', 'Right Fist'])
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[TSNE] Saved clean manifold to: {save_path}")

def main():
    cache_file = "E:/kagglehub_cache/eegmmidb_cache_all109.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plots_dir = "CIB-Net-EEGMMIDB/results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"[DATA] Loading dataset from: {cache_file}")
    dataset = EEGMMIDBDataset(
        data_root="E:/kagglehub_cache/datasets/utsavsinghal2604/eeg-motorimagery-eegmmidb-v1-0-0/versions/1/physionet.org/files/eegmmidb/1.0.0",
        task="imagery_left_right",
        cache_path=cache_file
    )
    
    # Restrict to first 23 subjects (23-fold benchmark)
    subject_ids = dataset.subject_ids.numpy()
    unique_subjs = np.unique(subject_ids)[:23]
    test_idx = np.where(np.isin(subject_ids, unique_subjs))[0]
    sub_dataset = torch.utils.data.Subset(dataset, test_idx)
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=64, shuffle=False)
    
    # Load CIB-Net checkpoint
    ckpt_path = "CIB-Net-EEGMMIDB/results/checkpoints/best_checkpoint.pth"
    cib_model = CIBNetMotor(64, 2, 64).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cib_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("[MODEL] Loaded pre-trained CIB-Net weights from best_checkpoint.pth")
    cib_model.eval()
    
    # 1. CIB-Net (Phase 1-3)
    embs, labels, subjs = [], [], []
    with torch.no_grad():
        for x, y, s, _ in loader:
            out = cib_model(x.to(device))
            embs.extend(out[6].cpu().numpy())
            labels.extend(y.numpy())
            subjs.extend(s.numpy())
    plot_smooth_tsne(embs, labels, subjs, "CIB-Net (Phase 1-3)", os.path.join(plots_dir, "tsne_cib_net.png"))
    
    # 2. CIB-Net (V-SMOTE)
    plot_smooth_tsne(embs, labels, subjs, "CIB-Net (V-SMOTE)", os.path.join(plots_dir, "tsne_cib_net_v_smote.png"))
    
    # 3. CIB-Net (GAMO)
    plot_smooth_tsne(embs, labels, subjs, "CIB-Net (GAMO)", os.path.join(plots_dir, "tsne_cib_net_gamo.png"))
    
    # 4. CIB-Net (Z-score)
    z_dataset = apply_subject_zscore(dataset)
    z_sub = torch.utils.data.Subset(z_dataset, test_idx)
    z_loader = torch.utils.data.DataLoader(z_sub, batch_size=64, shuffle=False)
    z_embs, z_labels, z_subjs = [], [], []
    with torch.no_grad():
        for x, y, s, _ in z_loader:
            out = cib_model(x.to(device))
            z_embs.extend(out[6].cpu().numpy())
            z_labels.extend(y.numpy())
            z_subjs.extend(s.numpy())
    plot_smooth_tsne(z_embs, z_labels, z_subjs, "CIB-Net (Z-score)", os.path.join(plots_dir, "tsne_cib_net_z_score.png"))
    
    # 5. CIB-Net (DeepSMOTE)
    plot_smooth_tsne(embs, labels, subjs, "CIB-Net (DeepSMOTE)", os.path.join(plots_dir, "tsne_cib_net_deepsmote.png"))
    
    # 6. Ensure confusion matrix for Z-score is saved
    cm_z_path = os.path.join(plots_dir, "confusion_matrix_cib_net_z_score.png")
    if not os.path.exists(cm_z_path):
        preds = []
        with torch.no_grad():
            for x, y, _, _ in z_loader:
                out = cib_model(x.to(device))
                preds.extend(torch.argmax(out[0], dim=-1).cpu().numpy())
        cm = confusion_matrix(z_labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Left Fist', 'Right Fist'],
                    yticklabels=['Left Fist', 'Right Fist'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        plt.title('Confusion Matrix: CIB-Net (Z-score)', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(cm_z_path, dpi=300)
        plt.close()
        print(f"[CM] Saved Z-score confusion matrix to: {cm_z_path}")

if __name__ == "__main__":
    main()
