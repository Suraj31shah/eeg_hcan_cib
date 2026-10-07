import torch
import torch.nn as nn

from models.stage_a_encoder import PerChannelCNNEncoder
from models.stage_c_fusion import RegionGroupedFusion
from models.peripheral_encoder import PeripheralEncoder
from models.hcan_fusion import HCANCrossAttentionFusion

class HCANCIBNet(nn.Module):
    def __init__(self, n_channels=32, n_classes=2, d_embed=64, n_periph_features=24, shared_weights=True, region_dropout_p=0.2):
        """
        Unified Architecture combining HCAN multimodal fusion and CIB-Net region starvation correction.
        """
        super(HCANCIBNet, self).__init__()
        
        # 1. EEG Backbone (Stage A)
        self.eeg_encoder = PerChannelCNNEncoder(d_embed=d_embed, shared_weights=shared_weights, n_channels=n_channels)
        
        # 2. Region Tokenizer (Stage C)
        # Note: We override final_classifier behavior inside RegionGroupedFusion because we don't use 
        # its final_logits anymore (HCAN fusion does the final prediction). 
        # But we still need it to extract region tokens (group_feats).
        self.region_tokenizer = RegionGroupedFusion(d_embed=d_embed, n_classes=n_classes, region_dropout_p=region_dropout_p)
        
        # 3. Peripheral Encoder
        self.periph_encoder = PeripheralEncoder(in_features=n_periph_features, d_embed=d_embed)
        
        # 4. Multimodal Fusion (HCAN + Shallow Prior)
        self.fusion = HCANCrossAttentionFusion(d_embed=d_embed, n_heads=4, n_classes=n_classes)
        
    def forward(self, x_eeg, x_periph):
        """
        x_eeg: [B, 32, 512]
        x_periph: [B, 24]
        Returns:
            final_logits: [B, 2]
            group_logits: List of 5 region aux logits [B, 2] (for Region SD)
            z_eeg: EEG modality aux logit [B, 2] (for Modality SD)
            z_periph: Periph modality aux logit [B, 2] (for Modality SD)
        """
        # 1. Encode EEG Channels -> [B, 32, d_embed]
        channel_embeddings = self.eeg_encoder(x_eeg)
        
        # 2. Tokenize Regions -> group_feats: list of 5 [B, d_embed], group_logits: list of 5 [B, 2]
        # We manually call the inner loop of RegionGroupedFusion to avoid the unused final_classifier
        group_feats = []
        group_logits = []
        
        B = channel_embeddings.size(0)
        n_regions = len(self.region_tokenizer.regions)
        if self.training and self.region_tokenizer.region_dropout_p > 0:
            mask = torch.empty(B, n_regions, device=channel_embeddings.device).bernoulli_(1 - self.region_tokenizer.region_dropout_p)
            mask = mask / (1 - self.region_tokenizer.region_dropout_p)
        else:
            mask = torch.ones(B, n_regions, device=channel_embeddings.device)
            
        for idx, (region, channels) in enumerate(self.region_tokenizer.regions.items()):
            region_x = channel_embeddings[:, channels, :] 
            attn = self.region_tokenizer.intra_attentions[region](region_x)
            pooled = torch.sum(region_x * attn, dim=1)
            pooled = pooled * mask[:, idx:idx+1]
            group_feats.append(pooled)
            group_logits.append(self.region_tokenizer.group_classifiers[region](pooled))
            
        # Stack region tokens for Cross-Attention -> [B, 5, d_embed]
        region_tokens = torch.stack(group_feats, dim=1)
        
        # 3. Encode Peripheral -> [B, 1, d_embed], [B, 2]
        periph_token, z_periph = self.periph_encoder(x_periph)
        
        # 4. HCAN Fusion
        final_logits, z_eeg = self.fusion(periph_token, region_tokens)
        
        return final_logits, group_logits, z_eeg, z_periph
