import torch
import torch.nn as nn

class SpectralDecouplingLoss(nn.Module):
    """
    Invariance-Gated Spectral Decoupling Loss.
    Combines:
      1. Primary Cross-Entropy on final prediction logits.
      2. Regional auxiliary Cross-Entropy on regional tokens.
      3. Gated Spectral Decoupling penalty: sum_r (lambda_0 / g_r) * ||z_r||^2.
    """
    def __init__(self, lambda_0=1e-3, beta_aux=0.2):
        super(SpectralDecouplingLoss, self).__init__()
        self.lambda_0 = lambda_0
        self.beta_aux = beta_aux
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, final_logits, targets, group_logits, inv_gates, group_feats):
        """
        final_logits: [B, n_classes]
        targets: [B]
        group_logits: list of 5 [B, n_classes]
        inv_gates: [5] tensor of invariance scores
        group_feats: list of 5 [B, d_embed]
        """
        loss_main = self.ce_loss(final_logits, targets)
        
        loss_aux = 0.0
        loss_sd = 0.0
        
        for r, (logits_r, feat_r) in enumerate(zip(group_logits, group_feats)):
            loss_aux += self.ce_loss(logits_r, targets)
            
            # Gated L2 norm penalty on regional embeddings:
            # If region r is unstable/noisy, inv_gates[r] is small -> penalty is large!
            g_r = torch.clamp(inv_gates[r], min=1e-3)
            penalty_coeff = self.lambda_0 / g_r
            l2_norm = torch.mean(torch.sum(feat_r ** 2, dim=-1))
            loss_sd += penalty_coeff * l2_norm
            
        loss_aux = loss_aux / len(group_logits)
        
        total_loss = loss_main + (self.beta_aux * loss_aux) + loss_sd
        return total_loss, loss_main.item(), loss_aux.item(), loss_sd.item()

