import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os
import matplotlib.pyplot as plt

def extract_embeddings(model, dataloader, device):
    """
    Extracts embeddings from the Stage A encoder for a given dataloader.
    """
    model.eval()
    all_embeddings = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            # encoder output: [B, C, D]
            embeddings = model.encoder(x)
            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(y.numpy())
            
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_targets, axis=0)

def probe_single_channel_accuracy(train_embeddings, train_targets, test_embeddings, test_targets):
    """
    Trains a linear probe for EACH channel independently to see how predictive it is alone.
    Embeddings shape: [Samples, Channels, D]
    Returns list of test accuracies for each channel.
    """
    n_channels = train_embeddings.shape[1]
    channel_accuracies = []
    
    for c in range(n_channels):
        X_train_c = train_embeddings[:, c, :]
        X_test_c = test_embeddings[:, c, :]
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_c, train_targets)
        
        preds = clf.predict(X_test_c)
        acc = accuracy_score(test_targets, preds)
        channel_accuracies.append(acc)
        
    return channel_accuracies

def plot_channel_dominance(channel_accuracies_across_subjects, save_dir="results/figures"):
    """
    Generates 'Figure 1': Channel Contribution Variance (Dominance Exists)
    channel_accuracies_across_subjects: numpy array [n_subjects, n_channels]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    means = np.mean(channel_accuracies_across_subjects, axis=0)
    stds = np.std(channel_accuracies_across_subjects, axis=0)
    
    # Sort channels by mean accuracy for better visualization
    sorted_idx = np.argsort(means)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(means)), means[sorted_idx], yerr=stds[sorted_idx], capsize=5, alpha=0.8, color='steelblue')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.xticks(range(len(means)), [f"Ch{i}" for i in sorted_idx], rotation=45)
    plt.xlabel('EEG Channel (Sorted by Predictive Power)')
    plt.ylabel('Single-Channel Probe Accuracy (LOSO)')
    plt.title('Stage B Diagnostic: Variance of Channel Contributions Across Subjects\n(High std dev indicates spurious dominance)')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "channel_dominance_variance.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved dominance plot to {save_path}")
    
    return means, stds
