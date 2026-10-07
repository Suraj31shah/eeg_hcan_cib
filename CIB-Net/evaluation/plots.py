import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve

def load_results(json_path='results/fold_results.json'):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_roc_pr_curves(y_true_list, y_prob_list, save_dir='results/plots'):
    """
    Plots average ROC and PR curves with confidence bands across folds.
    y_true_list: list of arrays (one per fold)
    y_prob_list: list of arrays (one per fold)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Simple aggregated plot for now
    all_y = np.concatenate(y_true_list)
    all_prob = np.concatenate(y_prob_list)
    
    fpr, tpr, _ = roc_curve(all_y, all_prob)
    prec, rec, _ = precision_recall_curve(all_y, all_prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Aggregated ROC Curve')
    
    ax2.plot(rec, prec, color='green', lw=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Aggregated PR Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_pr_curves.png'), dpi=300)
    plt.close()

def plot_tsne(embeddings, labels, subjects=None, save_path='results/plots/tsne.png'):
    """
    Plots t-SNE of 64-d embeddings.
    If subjects is provided, plots a second panel colored by subject ID.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeds_2d = tsne.fit_transform(embeddings)
    
    if subjects is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        scatter1 = ax1.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=10)
        ax1.set_title('t-SNE Colored by Class')
        plt.colorbar(scatter1, ax=ax1)
        
        scatter2 = ax2.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=subjects, cmap='tab20', alpha=0.6, s=10)
        ax2.set_title('t-SNE Colored by Subject')
        
    else:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=10)
        plt.title('t-SNE Colored by Class')
        plt.colorbar(scatter)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_region_gate_heatmap(fold_results, save_path='results/plots/gate_heatmap.png'):
    """
    Plots a heatmap of the Stage B invariance gate values across the 32 folds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract gate values
    gates = [fold['gate_values'] for fold in fold_results if 'gate_values' in fold]
    if not gates:
        return
        
    gates_array = np.array(gates) # [32 folds, 5 regions]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(gates_array, cmap='viridis', annot=True, fmt=".2f",
                xticklabels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occipital'],
                yticklabels=[f'Fold {i+1}' for i in range(len(gates))])
    plt.title('Region Invariance Gates Across Folds')
    plt.xlabel('Brain Region')
    plt.ylabel('Held-out Subject (Fold)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Plotting suite ready. Run this after full LOSO evaluation is complete.")
    # Example usage:
    # results = load_results()
    # plot_region_gate_heatmap(results)
