import torch
import torch.nn as nn

class IntraRegionAttention(nn.Module):
    def __init__(self, d_embed=64):
        super(IntraRegionAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_embed, d_embed // 2),
            nn.Tanh(),
            nn.Linear(d_embed // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        x: [B, n_ch_in_region, d_embed]
        returns attention weights: [B, n_ch_in_region, 1]
        """
        return self.attn(x)

class RegionGroupedFusion(nn.Module):
    """
    Stage C: Region-Grouped Fusion for 64-Channel Motor Imagery.
    Partitions 64 EEG channels into 5 physiological cortical lobes.
    Applies intra-region attention, structured region dropout, and inter-region self-attention.
    """
    def __init__(self, d_embed=64, n_classes=2, region_dropout_p=0.2, regions_dict=None):
        super(RegionGroupedFusion, self).__init__()
        self.d_embed = d_embed
        self.n_classes = n_classes
        self.region_dropout_p = region_dropout_p
        
        # Default 64-channel 10-10 system anatomical grouping
        if regions_dict is None:
            self.regions = {
                'Frontal': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
                'Central_Sensorimotor': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                'Temporal': [38, 39, 40, 41, 42, 43, 44, 45],
                'Parietal': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                'Occipital': [60, 61, 62, 63]
            }
        else:
            self.regions = regions_dict
            
        self.intra_attentions = nn.ModuleDict({
            r: IntraRegionAttention(d_embed) for r in self.regions.keys()
        })
        
        # Auxiliary regional classifiers (for Stage B Invariance Gate tracking)
        self.group_classifiers = nn.ModuleDict({
            r: nn.Linear(d_embed, n_classes) for r in self.regions.keys()
        })
        
        # Inter-region Transformer Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=4, dim_feedforward=d_embed * 2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(len(self.regions) * d_embed, d_embed),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_embed, n_classes)
        )
        
    def forward(self, channel_embeddings):
        """
        channel_embeddings: [B, 64, d_embed]
        Returns:
            final_logits: [B, n_classes]
            group_logits: list of 5 [B, n_classes]
            group_feats: list of 5 [B, d_embed]
        """
        B = channel_embeddings.size(0)
        group_feats = []
        group_logits = []
        
        # Structured region dropout mask
        n_reg = len(self.regions)
        if self.training and self.region_dropout_p > 0:
            mask = torch.empty(B, n_reg, device=channel_embeddings.device).bernoulli_(1 - self.region_dropout_p)
            mask = mask / (1 - self.region_dropout_p)
        else:
            mask = torch.ones(B, n_reg, device=channel_embeddings.device)
            
        for idx, (region, channels) in enumerate(self.regions.items()):
            region_x = channel_embeddings[:, channels, :] # [B, n_ch, d_embed]
            attn = self.intra_attentions[region](region_x) # [B, n_ch, 1]
            pooled = torch.sum(region_x * attn, dim=1) # [B, d_embed]
            pooled = pooled * mask[:, idx:idx+1]
            
            group_feats.append(pooled)
            group_logits.append(self.group_classifiers[region](pooled))
            
        # Inter-region Transformer fusion
        region_tokens = torch.stack(group_feats, dim=1) # [B, 5, d_embed]
        transformed_tokens = self.transformer(region_tokens) # [B, 5, d_embed]
        
        flat = transformed_tokens.reshape(B, -1) # [B, 5 * d_embed]
        final_logits = self.classifier(flat) # [B, n_classes]
        
        return final_logits, group_logits, group_feats

