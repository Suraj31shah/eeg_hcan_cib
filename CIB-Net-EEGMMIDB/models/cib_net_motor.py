import torch
import torch.nn as nn
from models.stage_a_encoder import PerChannelCNNEncoder
from models.stage_c_fusion import RegionGroupedFusion

class CIBNetMotor(nn.Module):
    """
    Channel-Invariance Balancing Network (CIB-Net) for 64-Channel EEG Motor Imagery.
    
    Architecture:
      1. Stage A: Per-Channel 1D-CNN Temporal Encoder (prevents Pezeshki spatial gradient starvation).
      2. Stage C: Region-Grouped Cortical Fusion (groups into 5 brain lobes with intra-region attention
                  and inter-region Transformer fusion).
      3. Stage B interaction: Gated by Subject Invariance Gates during training via Spectral Decoupling Loss.
    """
    def __init__(self, n_channels=64, n_classes=2, d_embed=64, shared_weights=True, region_dropout_p=0.2, regions_dict=None):
        super(CIBNetMotor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_embed = d_embed
        
        self.encoder = PerChannelCNNEncoder(n_channels=n_channels, d_embed=d_embed, shared_weights=shared_weights)
        self.fusion = RegionGroupedFusion(d_embed=d_embed, n_classes=n_classes, region_dropout_p=region_dropout_p, regions_dict=regions_dict)
        
    def forward(self, x):
        """
        x: [B, 64, T]
        returns:
            final_logits: [B, n_classes]
            group_logits: list of 5 [B, n_classes]
            group_feats: list of 5 [B, d_embed]
            channel_embeddings: [B, 64, d_embed]
        """
        channel_embeddings = self.encoder(x) # [B, 64, d_embed]
        final_logits, group_logits, group_feats = self.fusion(channel_embeddings)
        return final_logits, group_logits, group_feats, channel_embeddings

