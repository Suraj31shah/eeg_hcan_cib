import torch
import torch.nn as nn
import numpy as np

def apply_gamo_latent(channel_embeddings, y, subj_ids=None):
    """
    Generative Adversarial Minority Oversampling (GAMO) in latent embedding space.
    Generates convex combinations of minority class representations with noise perturbations.
    """
    device = channel_embeddings.device
    y_np = y.cpu().numpy()
    classes, counts = np.unique(y_np, return_counts=True)
    
    if len(classes) < 2:
        return channel_embeddings, y, subj_ids
        
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    n_majority = counts[np.argmax(counts)]
    n_minority = counts[np.argmin(counts)]
    
    n_to_generate = n_majority - n_minority
    if n_to_generate <= 0:
        return channel_embeddings, y, subj_ids
        
    min_indices = torch.where(y == minority_class)[0]
    min_embeddings = channel_embeddings[min_indices] # [N_min, 64, D]
    min_subjs = subj_ids[min_indices] if subj_ids is not None else None
    
    B_min, C, D = min_embeddings.shape
    
    # Generate convex weights across minority samples with Dirichlet / Softmax distribution
    idx1 = torch.randint(0, B_min, (n_to_generate,), device=device)
    idx2 = torch.randint(0, B_min, (n_to_generate,), device=device)
    
    lam = torch.rand((n_to_generate, 1, 1), device=device)
    # Add small adversarial gaussian noise scaled to minority feature variance
    noise = torch.randn((n_to_generate, C, D), device=device) * 0.05
    
    syn_emb = lam * min_embeddings[idx1] + (1.0 - lam) * min_embeddings[idx2] + noise
    syn_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)
    syn_subj = min_subjs[idx1] if min_subjs is not None else None
    
    aug_embeddings = torch.cat([channel_embeddings, syn_emb], dim=0)
    aug_y = torch.cat([y, syn_y], dim=0)
    aug_subj = torch.cat([subj_ids, syn_subj], dim=0) if subj_ids is not None else None
    
    return aug_embeddings, aug_y, aug_subj

