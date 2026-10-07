import torch
import torch.nn as nn
from models.stage_a_encoder import PerChannelCNNEncoder
from models.stage_c_fusion import RegionGroupedFusion

class CIBNet(nn.Module):
    def __init__(self, n_channels=32, n_classes=2, d_embed=128, shared_weights=True):
        super(CIBNet, self).__init__()
        
        # Stage A
        self.encoder = PerChannelCNNEncoder(d_embed, shared_weights, n_channels)
        
        # Stage C
        self.fusion = RegionGroupedFusion(d_embed, n_classes)
        
    def forward(self, x):
        """
        x: [B, 32, T]
        """
        # [B, 32, d_embed]
        embeddings = self.encoder(x)
        
        # final_logits: [B, 2], group_logits: list of 5 tensors [B, 2]
        final_logits, group_logits = self.fusion(embeddings)
        
        return final_logits, group_logits
