import torch
import torch.nn as nn

class HCANCrossAttentionFusion(nn.Module):
    def __init__(self, d_embed=64, n_heads=4, n_classes=2):
        super(HCANCrossAttentionFusion, self).__init__()
        
        # Cross-Attention: Q=Peripheral, K=V=EEG Regions
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_embed)
        
        # Deep Fusion Classifier
        self.deep_classifier = nn.Linear(d_embed, n_classes)
        
        # Shallow Prior (Logistic Regression equivalent on Peripheral token)
        self.shallow_classifier = nn.Linear(d_embed, n_classes)
        
        # Learned scalar weight w for the shallow prior (initialized to 0)
        self.w_shallow = nn.Parameter(torch.zeros(1))
        
        # We also need an EEG-level aux head for Modality Spectral Decoupling (z_eeg)
        self.eeg_aux_head = nn.Linear(d_embed, n_classes)

    def forward(self, periph_token, region_tokens):
        """
        periph_token: [B, 1, d_embed] (Query)
        region_tokens: [B, 5, d_embed] from the 5 brain regions
        Returns:
            final_logits: [B, 2]
            z_eeg: [B, 2] (EEG-modality aux logit)
        """
        # 1. Create pooled EEG token (mean of region tokens) -> [B, 1, d_embed]
        pooled_eeg = torch.mean(region_tokens, dim=1, keepdim=True)
        
        # Calculate EEG Aux Logit (z_eeg)
        z_eeg = self.eeg_aux_head(pooled_eeg.squeeze(1))
        
        # 2. Key/Value = 5 Region Tokens + 1 Pooled EEG Token = 6 Tokens
        kv = torch.cat([region_tokens, pooled_eeg], dim=1) # [B, 6, d_embed]
        
        # 3. Cross Attention
        # Q: [B, 1, d_embed], K,V: [B, 6, d_embed]
        attn_out, _ = self.cross_attn(periph_token, kv, kv)
        
        # 4. Residual + LayerNorm
        fused_token = self.layer_norm(periph_token + attn_out).squeeze(1) # [B, d_embed]
        
        # 5. Deep Logits
        logit_deep = self.deep_classifier(fused_token)
        
        # 6. Shallow Prior
        # The shallow classifier acts purely on the raw peripheral token
        logit_shallow = self.shallow_classifier(periph_token.squeeze(1))
        
        # 7. Final Output
        final_logits = logit_deep + self.w_shallow * logit_shallow
        
        return final_logits, z_eeg
