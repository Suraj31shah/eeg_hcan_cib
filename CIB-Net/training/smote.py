import torch
import numpy as np

def apply_v_smote(channel_embeddings, x_periph, y, subj_ids, k_neighbors=5, alpha=1.0):
    """
    Variance-Preserving SMOTE (V-SMOTE) applied in the latent embedding space.
    Oversamples the minority class to match the majority class size within the current batch.
    
    Args:
        channel_embeddings: [B, n_channels, d_embed]
        x_periph: [B, n_periph_features]
        y: [B]
        subj_ids: [B]
    Returns:
        Augmented tensors with balanced classes for this batch.
    """
    device = channel_embeddings.device
    
    # Identify majority and minority classes
    y_np = y.cpu().numpy()
    classes, counts = np.unique(y_np, return_counts=True)
    
    if len(classes) < 2:
        return channel_embeddings, x_periph, y, subj_ids # Cannot balance a single-class batch
        
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    
    n_majority = counts[np.argmax(counts)]
    n_minority = counts[np.argmin(counts)]
    
    n_to_generate = n_majority - n_minority
    if n_to_generate <= 0:
        return channel_embeddings, x_periph, y, subj_ids
        
    # Get minority samples
    min_indices = torch.where(y == minority_class)[0]
    min_embeddings = channel_embeddings[min_indices] # [N_min, C, D]
    min_periph = x_periph[min_indices]
    min_subjs = subj_ids[min_indices]
    
    # Flatten embeddings to find nearest neighbors
    # [N_min, C * D]
    flat_emb = min_embeddings.view(n_minority, -1)
    
    # Compute pairwise Euclidean distances
    dist_matrix = torch.cdist(flat_emb, flat_emb)
    
    # Get k nearest neighbors (excluding self, so start from index 1)
    k = min(k_neighbors, n_minority - 1)
    if k <= 0:
        # Not enough minority samples to interpolate, just duplicate
        gen_indices = torch.randint(0, n_minority, (n_to_generate,), device=device)
        return (
            torch.cat([channel_embeddings, min_embeddings[gen_indices]], dim=0),
            torch.cat([x_periph, min_periph[gen_indices]], dim=0),
            torch.cat([y, torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)], dim=0),
            torch.cat([subj_ids, min_subjs[gen_indices]], dim=0)
        )
        
    _, nn_indices = torch.topk(dist_matrix, k=k+1, largest=False)
    nn_indices = nn_indices[:, 1:] # [N_min, k]
    
    # Generate new samples
    gen_emb = []
    gen_periph = []
    gen_subjs = []
    
    for _ in range(n_to_generate):
        # Pick a random minority sample
        base_idx = torch.randint(0, n_minority, (1,)).item()
        # Pick a random neighbor of that sample
        neighbor_idx = nn_indices[base_idx, torch.randint(0, k, (1,)).item()]
        
        # V-SMOTE Interpolation (incorporating variance preservation alpha)
        # weight = random value scaled by alpha
        weight = torch.rand(1, device=device) * alpha
        
        new_emb = min_embeddings[base_idx] + weight * (min_embeddings[neighbor_idx] - min_embeddings[base_idx])
        new_periph = min_periph[base_idx] + weight * (min_periph[neighbor_idx] - min_periph[base_idx])
        
        gen_emb.append(new_emb)
        gen_periph.append(new_periph)
        gen_subjs.append(min_subjs[base_idx]) # Inherit subject ID of base sample
        
    # Stack generated samples
    gen_emb = torch.stack(gen_emb, dim=0)
    gen_periph = torch.stack(gen_periph, dim=0)
    gen_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)
    gen_subjs = torch.stack(gen_subjs, dim=0)
    
    # Concatenate with original batch
    new_embeddings = torch.cat([channel_embeddings, gen_emb], dim=0)
    new_periph = torch.cat([x_periph, gen_periph], dim=0)
    new_y = torch.cat([y, gen_y], dim=0)
    new_subjs = torch.cat([subj_ids, gen_subjs], dim=0)
    
    # Shuffle the augmented batch
    perm = torch.randperm(new_embeddings.size(0))
    
    return new_embeddings[perm], new_periph[perm], new_y[perm], new_subjs[perm]
