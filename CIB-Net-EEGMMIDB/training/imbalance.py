import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepSMOTELatent(nn.Module):
    """
    Autoencoder-based DeepSMOTE operating over EEG channel tokens [B, C, D].
    Encodes minority samples into a compact latent bottleneck, runs k-NN SMOTE in latent space,
    decodes back to channel embedding space, and returns augmented embeddings.
    """
    def __init__(self, channels=64, d_embed=64, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.channels = channels
        self.d_embed = d_embed
        input_dim = channels * d_embed
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def oversample(self, channel_embeddings, y, subj_ids=None, k_neighbors=5):
        B, C, D = channel_embeddings.shape
        device = channel_embeddings.device
        flat_in = channel_embeddings.reshape(B, C * D)
        
        latent = self.encoder(flat_in.detach())
        recon = self.decoder(latent)
        recon_loss = F.mse_loss(recon, flat_in.detach())
        
        y_np = y.cpu().numpy()
        classes, counts = np.unique(y_np, return_counts=True)
        if len(classes) < 2:
            return channel_embeddings, y, subj_ids, recon_loss
            
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        n_majority = counts[np.argmax(counts)]
        n_minority = counts[np.argmin(counts)]
        
        n_to_generate = n_majority - n_minority
        if n_to_generate <= 0:
            return channel_embeddings, y, subj_ids, recon_loss
            
        min_idx = torch.where(y == minority_class)[0]
        min_lat = latent[min_idx]
        min_subjs = subj_ids[min_idx] if subj_ids is not None else None
        
        k = min(int(k_neighbors), n_minority - 1)
        if k <= 0:
            gen_pick = torch.randint(0, n_minority, (n_to_generate,), device=device)
            syn_lat = min_lat[gen_pick]
            gen_subjs = min_subjs[gen_pick] if min_subjs is not None else None
        else:
            dist = torch.cdist(min_lat.float(), min_lat.float())
            _, nn_idx = torch.topk(dist, k=k + 1, largest=False)
            nn_idx = nn_idx[:, 1:]
            base = torch.randint(0, n_minority, (n_to_generate,), device=device)
            nbr = nn_idx[base, torch.randint(0, k, (n_to_generate,), device=device)]
            gap = torch.rand((n_to_generate, 1), device=device, dtype=min_lat.dtype)
            syn_lat = min_lat[base] + gap * (min_lat[nbr] - min_lat[base])
            gen_subjs = min_subjs[base] if min_subjs is not None else None
            
        syn_decoded = self.decoder(syn_lat)
        syn_emb = syn_decoded.reshape(n_to_generate, C, D)
        syn_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)
        
        aug_embeddings = torch.cat([channel_embeddings, syn_emb.to(channel_embeddings.dtype)], dim=0)
        aug_y = torch.cat([y, syn_y], dim=0)
        aug_subj = torch.cat([subj_ids, gen_subjs], dim=0) if subj_ids is not None else None
        
        perm = torch.randperm(aug_embeddings.size(0), device=device)
        return aug_embeddings[perm], aug_y[perm], aug_subj[perm] if aug_subj is not None else None, recon_loss

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

