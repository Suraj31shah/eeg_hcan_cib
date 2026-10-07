import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. Classical / Loss-based Methods
# ==========================================

def get_class_weights(y):
    """Computes standard inverse frequency class weights."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {u: total / (len(unique) * c) for u, c in zip(unique, counts)}
    # Convert to tensor sorted by class index (0, 1)
    w_tensor = torch.tensor([weights.get(i, 1.0) for i in range(max(unique)+1)], dtype=torch.float32)
    return w_tensor

class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjusted Loss (Menon et al., 2020)
    Adjusts the logits based on the prior probability of each class.
    """
    def __init__(self, y_train, tau=1.0):
        super().__init__()
        unique, counts = np.unique(y_train, return_counts=True)
        pi = counts / len(y_train)
        self.tau = tau
        # adjustment = tau * log(pi)
        self.adjustment = torch.tensor(tau * np.log(pi + 1e-8), dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Move adjustment to same device as logits
        adj = self.adjustment.to(logits.device)
        adjusted_logits = logits + adj
        return self.ce(adjusted_logits, targets)


# ==========================================
# 2. SMOTE Variants (Operating on Embeddings)
# ==========================================

def apply_smote(X, y, k=5):
    """
    Standard SMOTE in embedding space.
    X: [N, D] numpy array of embeddings
    y: [N] numpy array of labels
    """
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    
    n_minority = counts[np.argmin(counts)]
    n_majority = counts[np.argmax(counts)]
    n_to_generate = n_majority - n_minority
    
    if n_to_generate <= 0:
        return X, y
        
    X_min = X[y == minority_class]
    
    # Fit kNN
    # Handle edge case where n_minority is very small
    k_actual = min(k, n_minority - 1)
    if k_actual < 1:
        # Cannot do SMOTE with < 2 minority samples, just duplicate
        indices = np.random.choice(n_minority, n_to_generate, replace=True)
        X_syn = X_min[indices]
    else:
        neigh = NearestNeighbors(n_neighbors=k_actual + 1)
        neigh.fit(X_min)
        distances, indices = neigh.kneighbors(X_min)
        
        # Generate samples
        X_syn = np.zeros((n_to_generate, X.shape[1]))
        
        for i in range(n_to_generate):
            # Pick a random minority sample
            idx = np.random.randint(0, n_minority)
            # Pick a random neighbor (excluding self which is at index 0)
            nn_idx = indices[idx, np.random.randint(1, k_actual + 1)]
            
            # Interpolation step
            diff = X_min[nn_idx] - X_min[idx]
            gap = np.random.rand()
            X_syn[i] = X_min[idx] + gap * diff
            
    X_resampled = np.vstack([X, X_syn])
    y_resampled = np.hstack([y, np.full(n_to_generate, minority_class)])
    
    return X_resampled, y_resampled


def apply_v_smote(X, y, k=5):
    """
    Variance-Scaled SMOTE.
    Instead of uniform random interpolation [0,1], the interpolation 
    is constrained by the local feature variance.
    """
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    n_minority = counts[np.argmin(counts)]
    n_to_generate = counts[np.argmax(counts)] - n_minority
    
    if n_to_generate <= 0:
        return X, y
        
    X_min = X[y == minority_class]
    
    # Global minority variance per feature
    global_var = np.var(X_min, axis=0) + 1e-8
    global_var = global_var / np.max(global_var) # Scale to [0,1]
    
    k_actual = min(k, n_minority - 1)
    if k_actual < 1:
        indices = np.random.choice(n_minority, n_to_generate, replace=True)
        return np.vstack([X, X_min[indices]]), np.hstack([y, np.full(n_to_generate, minority_class)])
        
    neigh = NearestNeighbors(n_neighbors=k_actual + 1)
    neigh.fit(X_min)
    distances, indices = neigh.kneighbors(X_min)
    
    X_syn = np.zeros((n_to_generate, X.shape[1]))
    
    for i in range(n_to_generate):
        idx = np.random.randint(0, n_minority)
        nn_idx = indices[idx, np.random.randint(1, k_actual + 1)]
        
        diff = X_min[nn_idx] - X_min[idx]
        
        # Variance-scaled gap: if a feature has high variance, we allow larger jumps.
        # gap is a vector now, not a scalar.
        gap = np.random.rand(X.shape[1]) * global_var
        
        X_syn[i] = X_min[idx] + gap * diff
        
    return np.vstack([X, X_syn]), np.hstack([y, np.full(n_to_generate, minority_class)])


# ==========================================
# 3. Deep Oversampling Methods
# ==========================================

class DeepSMOTE(nn.Module):
    """
    Simplified DeepSMOTE: Autoencoder where SMOTE happens in the latent space.
    """
    def __init__(self, input_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def generate_samples(self, X_min, n_to_generate, k=5, device='cpu'):
        """Runs SMOTE in latent space and decodes."""
        self.eval()
        with torch.no_grad():
            latent_min = self.encoder(torch.FloatTensor(X_min).to(device)).cpu().numpy()
            
        # Apply standard SMOTE in latent space
        # We pass a dummy y array since we know all are minority
        y_dummy = np.ones(len(latent_min))
        # Add one majority class point far away just to trick apply_smote into oversampling the minority
        latent_maj = np.zeros((1, latent_min.shape[1])) + 999 
        y_maj = np.zeros(1)
        
        X_smote_in = np.vstack([latent_min, latent_maj])
        y_smote_in = np.hstack([y_dummy, y_maj])
        
        # We want exactly n_to_generate. apply_smote balances exactly.
        # So we adjust the majority dummy count to be len(latent_min) + n_to_generate
        dummy_maj = np.zeros((len(latent_min) + n_to_generate, latent_min.shape[1]))
        y_dummy_maj = np.zeros(len(latent_min) + n_to_generate)
        
        X_smote_in = np.vstack([latent_min, dummy_maj])
        y_smote_in = np.hstack([y_dummy, y_dummy_maj])
        
        X_res, y_res = apply_smote(X_smote_in, y_smote_in, k=k)
        
        # Extract just the synthetic minority points
        # The first len(latent_min) are the original. The newly generated ones are at the end.
        synthetic_latents = X_res[y_res == 1][len(latent_min):]
        
        with torch.no_grad():
            synthetic_X = self.decoder(torch.FloatTensor(synthetic_latents).to(device)).cpu().numpy()
            
        return synthetic_X


class GAMO(nn.Module):
    """
    Generative Adversarial Minority Oversampling (GAMO)
    Simplified to work on embeddings.
    """
    def __init__(self, input_dim=64):
        super().__init__()
        # Generator takes noise and minority samples and produces new minority samples
        self.generator = nn.Sequential(
            nn.Linear(input_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
        # Discriminator tries to distinguish real vs fake minority samples
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def generate_samples(self, X_min, n_to_generate, device='cpu'):
        self.eval()
        with torch.no_grad():
            # Randomly select base minority samples
            indices = np.random.choice(len(X_min), n_to_generate, replace=True)
            base_samples = torch.FloatTensor(X_min[indices]).to(device)
            
            # Add noise
            noise = torch.randn(n_to_generate, 16).to(device)
            gen_input = torch.cat([base_samples, noise], dim=1)
            
            # Generate
            synthetic_X = self.generator(gen_input).cpu().numpy()
            
        return synthetic_X
