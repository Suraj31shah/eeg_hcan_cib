import torch
import torch.nn as nn

class SpectralDecouplingLoss(nn.Module):
    def __init__(self, lambda_sd=0.001, alpha_modality=0.3, beta_region=0.2):
        super(SpectralDecouplingLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_sd = lambda_sd
        self.alpha_modality = alpha_modality
        self.beta_region = beta_region

    def forward(self, logits, y, group_logits, invariance_scores, z_eeg=None, z_periph=None):
        """
        logits: final joint model output [B, 2]
        y: target labels [B]
        group_logits: list of logits per region group
        invariance_scores: tensor of length n_groups for region gating
        z_eeg: aux logits from the EEG modality [B, 2]
        z_periph: aux logits from the peripheral modality [B, 2]
        """
        # Joint Cross Entropy
        loss = self.ce(logits, y)
        
        # Modality-level Aux CE + SD
        if z_eeg is not None:
            loss += self.alpha_modality * self.ce(z_eeg, y)
            loss += self.lambda_sd * torch.mean(z_eeg ** 2)
            
        if z_periph is not None:
            loss += self.alpha_modality * self.ce(z_periph, y)
            loss += self.lambda_sd * torch.mean(z_periph ** 2)
        
        # Region-level Aux CE + Gated SD
        if group_logits and invariance_scores is not None:
            for k, z_k in enumerate(group_logits):
                # Region Aux CE
                loss += self.beta_region * self.ce(z_k, y)
                
                # Region Gated SD
                # lambda_k = lambda_0 * g_k (g_k is the invariance gate, directly computed now)
                # Wait, the plan says: g_r = R * softmax(...) -> scales SD strength. 
                # So lambda_k = lambda_sd / (invariance_scores[k] + 1e-4) or directly * gate if gate is computed properly.
                # Let's stick to the invariance_scores logic for gating.
                inv_k = invariance_scores[k] + 1e-4
                lambda_k = self.lambda_sd / inv_k
                loss += lambda_k * torch.mean(z_k ** 2)
                
        return loss
