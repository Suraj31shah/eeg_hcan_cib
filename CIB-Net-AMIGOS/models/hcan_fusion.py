import torch
import torch.nn as nn


class HCANCrossAttentionFusion(nn.Module):
    """
    HCAN Cross-Attention Fusion: Peripheral Query attends over EEG region + pooled tokens,
    combined with a learned weighted shallow peripheral prior.
    """

    def __init__(self, d_embed: int = 64, n_heads: int = 4, n_classes: int = 2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_embed)

        self.deep_classifier = nn.Linear(d_embed, n_classes)
        self.shallow_classifier = nn.Linear(d_embed, n_classes)
        self.w_shallow = nn.Parameter(torch.zeros(1))
        self.eeg_aux_head = nn.Linear(d_embed, n_classes)

    def forward(self, periph_token: torch.Tensor, region_tokens: torch.Tensor):
        """
        periph_token: `[B, 1, d_embed]`
        region_tokens: `[B, n_regions, d_embed]`
        Returns: `(final_logits [B, n_classes], z_eeg [B, n_classes])`
        """
        pooled_eeg = torch.mean(region_tokens, dim=1, keepdim=True)
        z_eeg = self.eeg_aux_head(pooled_eeg.squeeze(1))

        kv = torch.cat([region_tokens, pooled_eeg], dim=1)
        attn_out, _ = self.cross_attn(periph_token, kv, kv)
        fused_token = self.layer_norm(periph_token + attn_out).squeeze(1)

        logit_deep = self.deep_classifier(fused_token)
        logit_shallow = self.shallow_classifier(periph_token.squeeze(1))
        final_logits = logit_deep + self.w_shallow * logit_shallow
        return final_logits, z_eeg

