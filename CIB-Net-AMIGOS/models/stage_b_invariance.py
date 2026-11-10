import torch
import torch.nn as nn
import torch.nn.functional as F


class InvarianceGate(nn.Module):
    """
    Stage B Cross-Subject Invariance Gate.
    Tracks per-subject per-region EMA cross-entropy loss across training subjects
    and assigns stronger Spectral Decoupling pressure to high-variance regions.
    """

    def __init__(self, n_regions: int = 4, decay: float = 0.9, tau: float = 1.0, max_subjects: int = 64):
        super().__init__()
        self.n_regions = int(n_regions)
        self.decay = float(decay)
        self.tau = float(tau)
        self.max_subjects = int(max_subjects)

        # Sized safely above AMIGOS P40 (1-indexed subject IDs up to 40+)
        self.register_buffer("ema_losses", torch.zeros(self.max_subjects, self.n_regions))
        self.register_buffer("active_subjects", torch.zeros(self.max_subjects, dtype=torch.bool))

    def _ensure_capacity(self, needed_idx: int) -> None:
        if needed_idx < self.ema_losses.size(0):
            return
        new_cap = max(needed_idx + 1, self.ema_losses.size(0) * 2)
        device = self.ema_losses.device
        new_ema = torch.zeros(new_cap, self.n_regions, device=device, dtype=self.ema_losses.dtype)
        new_act = torch.zeros(new_cap, device=device, dtype=torch.bool)
        new_ema[: self.ema_losses.size(0)] = self.ema_losses
        new_act[: self.active_subjects.size(0)] = self.active_subjects
        self.ema_losses = new_ema
        self.active_subjects = new_act

    def update(self, group_logits, y: torch.Tensor, subject_ids: torch.Tensor) -> None:
        if not self.training or subject_ids is None or len(y) == 0:
            return

        ce = nn.CrossEntropyLoss(reduction="none")
        unique_subjects = torch.unique(subject_ids)

        with torch.no_grad():
            for s in unique_subjects:
                s_idx = int(s.item())
                if s_idx < 0:
                    continue
                self._ensure_capacity(s_idx)
                self.active_subjects[s_idx] = True
                mask = subject_ids == s
                if mask.sum() == 0:
                    continue

                for k, z_k in enumerate(group_logits):
                    s_loss = ce(z_k[mask].float(), y[mask]).mean()
                    if not torch.isfinite(s_loss):
                        continue
                    if self.ema_losses[s_idx, k] == 0:
                        self.ema_losses[s_idx, k] = s_loss
                    else:
                        self.ema_losses[s_idx, k] = (
                            self.decay * self.ema_losses[s_idx, k] + (1.0 - self.decay) * s_loss
                        )

    def get_gates(self) -> torch.Tensor:
        """
        Computes region invariance gates `g_r = R * softmax(-s_r / tau)` where
        `s_r = v_r / mean(v_r)`. Higher gate => lower cross-subject loss variance (more invariant).
        """
        active_idx = torch.where(self.active_subjects)[0]
        if len(active_idx) <= 1:
            return torch.ones(self.n_regions, device=self.ema_losses.device)

        active_losses = self.ema_losses[active_idx]  # [N_active, n_regions]
        v_r = torch.var(active_losses, dim=0, unbiased=True)
        mean_v = torch.mean(v_r) + 1e-8
        s_r = v_r / mean_v
        R = float(self.n_regions)
        g_r = R * F.softmax(-s_r / self.tau, dim=0)
        return g_r.detach()

