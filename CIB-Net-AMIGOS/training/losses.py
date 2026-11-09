import torch
import torch.nn as nn


class SpectralDecouplingLoss(nn.Module):
    def __init__(
        self,
        lambda_sd: float = 0.001,
        alpha_modality: float = 0.3,
        beta_region: float = 0.2,
        weight: torch.Tensor = None,
        base_criterion: nn.Module = None,
    ):
        super().__init__()
        self.ce = base_criterion if base_criterion is not None else nn.CrossEntropyLoss(weight=weight)
        self.lambda_sd = float(lambda_sd)
        self.alpha_modality = float(alpha_modality)
        self.beta_region = float(beta_region)

    def forward(self, logits, y, group_logits, invariance_scores, z_eeg=None, z_periph=None):
        loss = self.ce(logits, y)

        if z_eeg is not None:
            loss = loss + self.alpha_modality * self.ce(z_eeg, y)
            loss = loss + self.lambda_sd * torch.mean(z_eeg ** 2)

        if z_periph is not None:
            loss = loss + self.alpha_modality * self.ce(z_periph, y)
            loss = loss + self.lambda_sd * torch.mean(z_periph ** 2)

        if group_logits and invariance_scores is not None:
            for k, z_k in enumerate(group_logits):
                loss = loss + self.beta_region * self.ce(z_k, y)
                inv_k = invariance_scores[k] + 1e-4
                lambda_k = self.lambda_sd / inv_k
                loss = loss + lambda_k * torch.mean(z_k ** 2)

        return loss

