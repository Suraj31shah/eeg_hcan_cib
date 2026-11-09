import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_training_history(history, save_path=None):
    """
    Plots training loss and validation accuracy curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Losses
    axes[0].plot(history['train_loss'], label='Total Loss', color='#2563eb', lw=2)
    if 'train_main_loss' in history:
        axes[0].plot(history['train_main_loss'], label='Main CE Loss', color='#10b981', ls='--')
    if 'train_sd_loss' in history:
        axes[0].plot(history['train_sd_loss'], label='Spectral Decoupling Loss', color='#f59e0b', ls=':')
    axes[0].set_title('Training Losses Across Epochs', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss Value')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Accuracy
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='#8b5cf6', lw=2)
    if 'val_balanced_acc' in history:
        axes[1].plot(history['val_balanced_acc'], label='Val Balanced Acc', color='#ec4899', ls='--')
    axes[1].axhline(y=0.5, color='gray', linestyle=':', label='Chance (50%)')
    axes[1].set_title('Validation Performance', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None):
    """
    Plots confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    cm = np.array(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names or ['Class 0', 'Class 1'],
                yticklabels=class_names or ['Class 0', 'Class 1'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_invariance_gates(gate_values, region_names, save_path=None):
    """
    Plots Stage B Invariance Gate distribution across brain lobes.
    """
    plt.figure(figsize=(8, 4.5))
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
    bars = plt.bar(region_names, gate_values, color=colors[:len(region_names)], edgecolor='black', alpha=0.85)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Neutral Baseline (g=1.0)')
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.03, f"{h:.2f}",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
                 
    plt.title('Stage B Invariance Gates Across Cortical Lobes', fontsize=13, fontweight='bold')
    plt.xlabel('Cortical Region')
    plt.ylabel('Invariance Gate Value ($g_r$)')
    plt.ylim(0, max(gate_values) * 1.25)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne_embeddings(embeddings, labels, class_names=None, save_path=None):
    """
    Visualizes latent feature space via 2D t-SNE.
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = ['#2563eb', '#ef4444', '#10b981', '#f59e0b']
    
    for i, lab in enumerate(unique_labels):
        mask = (labels == lab)
        name = class_names[lab] if class_names and lab < len(class_names) else f"Class {lab}"
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    label=name, color=colors[i % len(colors)], alpha=0.7, edgecolors='none', s=35)
                    
    plt.title('t-SNE Projection of CIB-Net Learned Latent Manifold', fontsize=13, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

