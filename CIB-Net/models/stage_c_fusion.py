import torch
import torch.nn as nn

class RegionGroupedFusion(nn.Module):
    def __init__(self, d_embed=64, n_classes=2, region_dropout_p=0.2):
        super(RegionGroupedFusion, self).__init__()
        
        self.region_dropout_p = region_dropout_p
        
        # 10-20 system 32-channel mapping for DEAP
        self.regions = {
            "frontal": [0,1,2,3,16,17,18],  # Fp1, Fp2, F3, F4, F7, F8, Fz
            "temporal": [4,5,12,13],        # T7, T8, FC5, FC6
            "central": [6,7,19,20,21],      # C3, C4, Cz, CP1, CP2
            "parietal": [8,9,22,23],        # P3, P4, P7, P8
            "occipital": [10,11,24,25],     # O1, O2, PO3, PO4
        }
        
        # Intra-group attention (simple self-attention per region to aggregate channels)
        self.intra_attentions = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(d_embed, d_embed // 2),
                nn.Tanh(),
                nn.Linear(d_embed // 2, 1),
                nn.Softmax(dim=1)
            ) for region in self.regions.keys()
        })
        
        # Independent classifiers per region to extract z_k (logit contribution)
        self.group_classifiers = nn.ModuleDict({
            region: nn.Linear(d_embed, n_classes) for region in self.regions.keys()
        })
        
        # Final joint classifier
        self.final_classifier = nn.Linear(len(self.regions) * d_embed, n_classes)

    def forward(self, x_embed):
        """
        x_embed: [B, 32, d_embed] from Stage A Encoder
        """
        group_feats = []
        group_logits = []
        
        # Create region dropout mask (shape: [B, n_regions])
        # If region_dropout_p > 0 and training, we drop regions with probability p
        B = x_embed.size(0)
        n_regions = len(self.regions)
        if self.training and self.region_dropout_p > 0:
            mask = torch.empty(B, n_regions, device=x_embed.device).bernoulli_(1 - self.region_dropout_p)
            # Scale to maintain expected value
            mask = mask / (1 - self.region_dropout_p)
        else:
            mask = torch.ones(B, n_regions, device=x_embed.device)
        
        for idx, (region, channels) in enumerate(self.regions.items()):
            # Extract channels for this region: [B, num_channels, d_embed]
            region_x = x_embed[:, channels, :] 
            
            # Intra-region attention weights: [B, num_channels, 1]
            attn = self.intra_attentions[region](region_x)
            
            # Weighted sum: [B, d_embed]
            pooled = torch.sum(region_x * attn, dim=1)
            
            # Apply structured region dropout
            pooled = pooled * mask[:, idx:idx+1]
            
            group_feats.append(pooled)
            
            # Compute z_k logit for Spectral Decoupling loss
            g_logit = self.group_classifiers[region](pooled)
            group_logits.append(g_logit)
            
        # Concat all region features
        fused = torch.cat(group_feats, dim=1) # [B, n_regions * d_embed]
        final_logits = self.final_classifier(fused)
        
        return final_logits, group_logits
