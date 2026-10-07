import torch
import torch.nn as nn
import torch.nn.functional as F

class InvarianceGate(nn.Module):
    def __init__(self, n_regions=5, decay=0.9, tau=1.0):
        super(InvarianceGate, self).__init__()
        self.n_regions = n_regions
        self.decay = decay
        self.tau = tau
        
        # We will track the EMA of the loss per region per subject.
        # Assuming maximum subject ID is 32 (1-indexed, so size 33).
        self.register_buffer('ema_losses', torch.zeros(33, n_regions))
        self.register_buffer('active_subjects', torch.zeros(33, dtype=torch.bool))

    def update(self, group_logits, y, subject_ids):
        """
        Updates the EMA with the current batch's region losses.
        group_logits: list of 5 tensors of shape [B, 2]
        y: target labels [B]
        subject_ids: [B]
        """
        if not self.training:
            return
            
        ce = nn.CrossEntropyLoss(reduction='none')
        unique_subjects = torch.unique(subject_ids)
        
        with torch.no_grad():
            for s in unique_subjects:
                s_idx = s.item()
                self.active_subjects[s_idx] = True
                mask = (subject_ids == s)
                
                if mask.sum() == 0:
                    continue
                    
                for k, z_k in enumerate(group_logits):
                    # Compute mean CE loss for this subject and region in the batch
                    s_loss = ce(z_k[mask], y[mask]).mean()
                    
                    if self.ema_losses[s_idx, k] == 0:
                        self.ema_losses[s_idx, k] = s_loss
                    else:
                        self.ema_losses[s_idx, k] = self.decay * self.ema_losses[s_idx, k] + (1 - self.decay) * s_loss

    def get_gates(self):
        """
        Computes the invariance gate g_r for each region based on cross-subject variance.
        Returns: [n_regions] tensor of gate values.
        """
        # Only compute variance across active subjects (training subjects)
        active_idx = torch.where(self.active_subjects)[0]
        
        if len(active_idx) <= 1:
            # Not enough subjects to compute variance, return uniform gate
            return torch.ones(self.n_regions, device=self.ema_losses.device)
            
        active_losses = self.ema_losses[active_idx] # [N_active, n_regions]
        
        # v_r: variance across training subjects
        v_r = torch.var(active_losses, dim=0) # [n_regions]
        
        # Normalize: s_r = v_r / mean(v)
        mean_v = torch.mean(v_r) + 1e-8
        s_r = v_r / mean_v
        
        # Gate: g_r = R * softmax(-s_r / tau)
        R = float(self.n_regions)
        g_r = R * F.softmax(-s_r / self.tau, dim=0)
        
        # Detach just to be absolutely sure no gradients flow
        return g_r.detach()
