import torch
import numpy as np

def apply_v_smote_latent(channel_embeddings, y, subj_ids=None, k_neighbors=5, alpha=1.0):
    """
    Variance-Preserving Latent SMOTE (V-SMOTE) for 64-channel EEG embeddings.
    Oversamples the minority class in the latent embedding space [B, 64, d_embed].
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
    flat_emb = min_embeddings.view(B_min, -1)
    
    dist_matrix = torch.cdist(flat_emb, flat_emb)
    k = min(k_neighbors, B_min - 1)
    
    if k <= 0:
        gen_indices = torch.randint(0, B_min, (n_to_generate,), device=device)
        syn_emb = min_embeddings[gen_indices]
        syn_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)
        syn_subj = min_subjs[gen_indices] if min_subjs is not None else None
    else:
        _, nn_indices = torch.topk(dist_matrix, k=k+1, largest=False)
        nn_indices = nn_indices[:, 1:] # [B_min, k]
        
        gen_emb = []
        gen_subjs = []
        global_var = torch.var(flat_emb, dim=0, keepdim=True).view(1, C, D)
        norm_var = global_var / (torch.max(global_var) + 1e-8)
        
        for _ in range(n_to_generate):
            base_idx = torch.randint(0, B_min, (1,)).item()
            neighbor_idx = nn_indices[base_idx, torch.randint(0, k, (1,)).item()]
            
            # Variance-scaled jump
            weight = torch.rand((1, C, D), device=device) * alpha * norm_var
            new_emb = min_embeddings[base_idx] + weight * (min_embeddings[neighbor_idx] - min_embeddings[base_idx])
            gen_emb.append(new_emb)
            if min_subjs is not None:
                gen_subjs.append(min_subjs[base_idx])
                
        syn_emb = torch.cat(gen_emb, dim=0)
        syn_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)
        syn_subj = torch.stack(gen_subjs) if min_subjs is not None else None
        
    aug_embeddings = torch.cat([channel_embeddings, syn_emb], dim=0)
    aug_y = torch.cat([y, syn_y], dim=0)
    aug_subj = torch.cat([subj_ids, syn_subj], dim=0) if subj_ids is not None else None
    
    return aug_embeddings, aug_y, aug_subj

