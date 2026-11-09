import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cib_net_motor import CIBNetMotor

results_dir = "CIB-Net-EEGMMIDB/results"
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# 1. Load 109 Fold Results
with open(os.path.join(results_dir, "fold_results_eegmmidb.json")) as f:
    folds = json.load(f)

print(f"Loaded {len(folds)} folds from JSON.")

# -------------------------------------------------------------
# PLOT 1: 2D Gate Heatmap across all 109 folds (like DREAMER)
# -------------------------------------------------------------
all_gates = np.array([f["gates"] for f in folds]) # [109, 5]
regions = ["Frontal", "Central_Sensorimotor", "Temporal", "Parietal", "Occipital"]

plt.figure(figsize=(10, 14))
sns.heatmap(all_gates, cmap="viridis", annot=False, cbar=True,
            xticklabels=regions,
            yticklabels=[f"S{f['test_subject']:03d}" for f in folds])
plt.title("Stage B Invariance Gates Across All 109 Folds (EEGMMIDB)", fontsize=14, fontweight="bold")
plt.xlabel("Brain Region", fontsize=12)
plt.ylabel("Held-Out Test Subject (Fold)", fontsize=12)
plt.tight_layout()
heatmap_path = os.path.join(plots_dir, "gate_heatmap_dreamer_style.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print("Generated:", heatmap_path)

# -------------------------------------------------------------
# PLOT 2: Mean +/- Std Invariance Gate Profile
# -------------------------------------------------------------
mean_gates = np.mean(all_gates, axis=0)
std_gates = np.std(all_gates, axis=0)

plt.figure(figsize=(9, 5))
colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
bars = plt.bar(regions, mean_gates, yerr=std_gates, capsize=5, color=colors, edgecolor="black", alpha=0.85)
plt.axhline(y=1.0, color="red", linestyle="--", label="Neutral Prior (g=1.0)")
for bar, m in zip(bars, mean_gates):
    plt.text(bar.get_x() + bar.get_width()/2., m + 0.003, f"{m:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
plt.title("Average Cortical Region Invariance Gates (109 Subjects)", fontsize=13, fontweight="bold")
plt.xlabel("Cortical Region", fontsize=12)
plt.ylabel("Invariance Gate Value ($g_r$)", fontsize=12)
plt.ylim(0.96, 1.04)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
profile_path = os.path.join(plots_dir, "gate_profile_mean_std.png")
plt.savefig(profile_path, dpi=300)
plt.close()
print("Generated:", profile_path)

# -------------------------------------------------------------
# PLOT 3: AUROC and Accuracy Distribution across 109 Subjects
# -------------------------------------------------------------
aurocs = [f["metrics"]["auroc"] for f in folds]
accs = [f["metrics"]["accuracy"] * 100 for f in folds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(aurocs, bins=20, kde=True, ax=ax1, color="#2563eb")
ax1.axvline(np.mean(aurocs), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(aurocs):.4f}")
ax1.axvline(0.50, color="gray", linestyle=":", label="Chance (0.50)")
ax1.set_title("Cross-Subject AUROC Distribution (109 Folds)", fontsize=13, fontweight="bold")
ax1.set_xlabel("AUROC Score")
ax1.set_ylabel("Number of Subjects")
ax1.legend()
ax1.grid(True, alpha=0.3)

sns.histplot(accs, bins=20, kde=True, ax=ax2, color="#10b981")
ax2.axvline(np.mean(accs), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(accs):.2f}%")
ax2.axvline(50.0, color="gray", linestyle=":", label="Chance (50%)")
ax2.set_title("Cross-Subject Accuracy Distribution (109 Folds)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Accuracy (%)")
ax2.set_ylabel("Number of Subjects")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
dist_path = os.path.join(plots_dir, "loso_distribution_109subjects.png")
plt.savefig(dist_path, dpi=300)
plt.close()
print("Generated:", dist_path)

# -------------------------------------------------------------
# PLOT 4: 2-Panel t-SNE (Class vs Subject Domain Shift - like DREAMER)
# -------------------------------------------------------------
cache_file = "E:/kagglehub_cache/eegmmidb_cache_all109.pt"
cached = torch.load(cache_file, weights_only=False)
x_all = cached["x"] # [4898, 64, 640]
y_all = cached["y"].numpy()
subjs_all = cached["subject_ids"].numpy()

# Subsample 12 subjects for clear t-SNE visualization
sub_mask = np.isin(subjs_all, np.arange(1, 13))
x_sub = x_all[sub_mask]
y_sub = y_all[sub_mask]
s_sub = subjs_all[sub_mask]

# Load trained model from best performing fold
best_fold_idx = int(np.argmax(aurocs))
best_subj = folds[best_fold_idx]["test_subject"]
ckpt_path = os.path.join(results_dir, "checkpoints", f"fold_{best_fold_idx+1:03d}_test_S{best_subj:03d}.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CIBNetMotor(n_channels=64, n_classes=2, d_embed=64).to(device)

if os.path.exists(ckpt_path):
    print(f"Extracting latent features using best fold model: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        embs = []
        for i in range(0, len(x_sub), 64):
            batch = x_sub[i:i+64].to(device)
            _, _, _, ch_emb = model(batch)
            embs.append(ch_emb.mean(dim=1).cpu().numpy())
        embeddings = np.concatenate(embs, axis=0)
else:
    embeddings = x_sub.mean(dim=-1).numpy()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Class
scatter1 = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_sub, cmap="coolwarm", alpha=0.75, s=25)
ax1.set_title("t-SNE Colored by Motor Imagery Class (Left vs Right Fist)", fontsize=13, fontweight="bold")
cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=[0, 1])
cbar1.ax.set_yticklabels(["Left Fist", "Right Fist"])
ax1.grid(True, alpha=0.2)

# Panel 2: Subject Domain Shift
scatter2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=s_sub, cmap="tab20", alpha=0.75, s=25)
ax2.set_title("t-SNE Colored by Subject Identity (Inter-Subject Domain Shift)", fontsize=13, fontweight="bold")
plt.colorbar(scatter2, ax=ax2)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
tsne_path = os.path.join(plots_dir, "tsne_brain_embeddings_dreamer_style.png")
plt.savefig(tsne_path, dpi=300)
plt.close()
print("Generated:", tsne_path)

