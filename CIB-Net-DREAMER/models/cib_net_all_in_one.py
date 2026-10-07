import numpy as np
import networkx as nx
from scipy.signal import hilbert
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# STAGE A: Per-Channel Temporal CNN Encoder
# =====================================================================
class PerChannelCNNEncoder(nn.Module):
    def __init__(self, d_embed=128, shared_weights=True, n_channels=32):
        super(PerChannelCNNEncoder, self).__init__()
        self.shared_weights = shared_weights
        self.n_channels = n_channels
        self.d_embed = d_embed
        
        if shared_weights:
            self.encoder = self._build_encoder()
        else:
            self.encoders = nn.ModuleList([self._build_encoder() for _ in range(n_channels)])

    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(64), 
            nn.Conv1d(64, self.d_embed, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.d_embed),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        """ x: [B, N_channels, T] -> Returns: [B, N_channels, D_embed] """
        B, C, T = x.size()
        embeddings = []
        if self.shared_weights:
            x_reshaped = x.view(B * C, 1, T)
            out = self.encoder(x_reshaped)
            return out.view(B, C, self.d_embed)
        else:
            for i in range(self.n_channels):
                xi = x[:, i, :].unsqueeze(1)
                ei = self.encoders[i](xi)
                embeddings.append(ei.squeeze(-1))
            return torch.stack(embeddings, dim=1)


# =====================================================================
# STAGE B: Invariance Scoring (Diagnostic Math)
# =====================================================================
def compute_variance_proxy_invariance(channel_accuracies):
    """ Plan B: Variance of Contribution Proxy """
    variances = np.var(channel_accuracies, axis=0)
    v_min, v_max = variances.min(), variances.max()
    norm_vars = (variances - v_min) / (v_max - v_min + 1e-8)
    invariance = 1.0 - norm_vars
    return torch.tensor(invariance, dtype=torch.float32)

def compute_irm_penalty(logits, y):
    """ Plan A: Full Causal IRMv1 gradient penalty """
    scale = torch.tensor(1.).requires_grad_()
    loss = nn.CrossEntropyLoss()(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# =====================================================================
# STAGE C: Region-Grouped Fusion
# =====================================================================
class RegionGroupedFusion(nn.Module):
    def __init__(self, d_embed=128, n_classes=2):
        super(RegionGroupedFusion, self).__init__()
        
        self.regions = {
            "frontal": [0,1,2,3,16,17,18],
            "temporal": [4,5,12,13],
            "central": [6,7,19,20,21],
            "parietal": [8,9,22,23],
            "occipital": [10,11,24,25],
        }
        
        self.intra_attentions = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(d_embed, d_embed // 2),
                nn.Tanh(),
                nn.Linear(d_embed // 2, 1),
                nn.Softmax(dim=1)
            ) for region in self.regions.keys()
        })
        
        self.group_classifiers = nn.ModuleDict({
            region: nn.Linear(d_embed, n_classes) for region in self.regions.keys()
        })
        
        self.final_classifier = nn.Linear(len(self.regions) * d_embed, n_classes)

    def forward(self, x_embed):
        group_feats = []
        group_logits = []
        for region, channels in self.regions.items():
            region_x = x_embed[:, channels, :] 
            attn = self.intra_attentions[region](region_x)
            pooled = torch.sum(region_x * attn, dim=1)
            group_feats.append(pooled)
            g_logit = self.group_classifiers[region](pooled)
            group_logits.append(g_logit)
            
        fused = torch.cat(group_feats, dim=1)
        final_logits = self.final_classifier(fused)
        return final_logits, group_logits


# =====================================================================
# STAGE C: Spectral Decoupling Loss
# =====================================================================
class SpectralDecouplingLoss(nn.Module):
    def __init__(self, lambda_base=0.1):
        super(SpectralDecouplingLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_base = lambda_base

    def forward(self, logits, y, group_logits, invariance_scores):
        loss_ce = self.ce(logits, y)
        sd_penalty = 0.0
        for k, z_k in enumerate(group_logits):
            inv_k = invariance_scores[k] + 1e-4
            lambda_k = self.lambda_base / inv_k
            sd_penalty += lambda_k * torch.mean(z_k ** 2)
        return loss_ce + sd_penalty


# =====================================================================
# STAGE D: Neuroscience PLV Graph Consistency
# =====================================================================
def compute_plv_centrality(eeg_data):
    channels = eeg_data.shape[0]
    analytic_signal = hilbert(eeg_data, axis=1)
    phase = np.angle(analytic_signal)
    
    plv_matrix = np.zeros((channels, channels))
    for i in range(channels):
        for j in range(i+1, channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
            
    plv_matrix += plv_matrix.T
    np.fill_diagonal(plv_matrix, 1.0)
    
    G = nx.from_numpy_array(plv_matrix)
    try:
        centrality_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
        centrality = np.array([centrality_dict[i] for i in range(channels)])
    except:
        centrality = np.ones(channels) / channels
    return centrality

def compute_graph_consistency_loss(learned_importance, plv_centrality):
    cos_sim = F.cosine_similarity(learned_importance.unsqueeze(0), plv_centrality.unsqueeze(0))
    loss = 1.0 - cos_sim
    return loss.mean()


# =====================================================================
# FULL ASSEMBLED CIB-NET
# =====================================================================
class CIBNet(nn.Module):
    def __init__(self, n_channels=32, n_classes=2, d_embed=128, shared_weights=True):
        super(CIBNet, self).__init__()
        self.encoder = PerChannelCNNEncoder(d_embed, shared_weights, n_channels)
        self.fusion = RegionGroupedFusion(d_embed, n_classes)
        
    def forward(self, x):
        embeddings = self.encoder(x)
        final_logits, group_logits = self.fusion(embeddings)
        return final_logits, group_logits
