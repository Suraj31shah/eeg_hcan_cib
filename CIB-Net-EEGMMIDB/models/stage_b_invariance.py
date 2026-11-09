import torch
import torch.nn as nn
import torch.nn.functional as F

class InvarianceGate(nn.Module):
    """
    Stage B: Subject-Aware Invariance Gating.
    Tracks cross-subject loss variance across human subjects using an EMA tracker.
    Computes gating scores g_r for each of the 5 anatomical brain regions.
    
    A region with consistent loss across subjects gets a high invariance score (g_r > 1.0).
    A region with erratic, noisy loss across subjects gets a low score (g_r < 1.0),
    which leads to heavy Spectral Decoupling penalty in Stage C.
    """
    def __init__(self, n_regions=5, decay=0.9, tau=1.0, eps=1e-6):
        super(InvarianceGate, self).__init__()
        self.n_regions = n_regions
        self.decay = decay
        self.tau = tau
        self.eps = eps
        
        # Buffer to store running variance of loss per region across subjects
        self.register_buffer("variance_ema", torch.zeros(n_regions))
        self.register_buffer("is_initialized", torch.tensor(False))
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    @torch.no_grad()
    def update(self, group_logits, targets, subj_ids):
        """
        group_logits: list of [B, n_classes] of length n_regions
        targets: [B]
        subj_ids: [B]
        """
        unique_subjs = torch.unique(subj_ids)
        if len(unique_subjs) < 2:
            return # Need at least 2 subjects in batch to compute cross-subject variance
            
        region_variances = []
        for r in range(self.n_regions):
            loss_r = self.ce_loss(group_logits[r], targets) # [B]
            
            # Compute mean loss per subject
            subj_means = []
            for s in unique_subjs:
                mask = (subj_ids == s)
                if mask.any():
                    subj_means.append(loss_r[mask].mean())
                    
            if len(subj_means) >= 2:
                subj_means_tensor = torch.stack(subj_means)
                var_r = torch.var(subj_means_tensor, unbiased=False)
            else:
                var_r = torch.tensor(0.0, device=loss_r.device)
                
            region_variances.append(var_r)
            
        batch_var = torch.stack(region_variances).detach() # [n_regions]
        
        if not self.is_initialized:
            self.variance_ema.copy_(batch_var)
            self.is_initialized.fill_(True)
        else:
            self.variance_ema.mul_(self.decay).add_(batch_var * (1.0 - self.decay))
            
    def get_gates(self):
        """
        Computes dynamic invariance gates:
        g_r = R * Softmax(-variance_ema / tau)
        Sum of gates is normalized to R (e.g. 5.0).
        """
        scores = -self.variance_ema / self.tau
        gates = self.n_regions * F.softmax(scores, dim=0)
        return gates

