import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.smote import _select_minority_majority


def get_class_weights(y) -> torch.Tensor:
    """Computes standard inverse frequency class weights."""
    y_np = np.asarray(y)
    unique, counts = np.unique(y_np, return_counts=True)
    total = len(y_np)
    weights = {int(u): total / (len(unique) * float(c)) for u, c in zip(unique, counts)}
    return torch.tensor([weights.get(i, 1.0) for i in range(int(max(unique)) + 1)], dtype=torch.float32)


class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjusted Loss (Menon et al., 2020 / ICLR 2021).
    Adjusts logits based on training class prior probabilities.
    """

    def __init__(self, y_train, tau: float = 1.0):
        super().__init__()
        y_np = np.asarray(y_train)
        counts = np.bincount(y_np, minlength=2).astype(np.float64)
        pi = counts / max(counts.sum(), 1.0)
        self.tau = float(tau)
        self.register_buffer("adjustment", torch.tensor(self.tau * np.log(pi + 1e-8), dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.adjustment.to(logits.device)
        return self.ce(adjusted_logits, targets)


class DeepSMOTELatent(nn.Module):
    """
    Autoencoder-based DeepSMOTE operating jointly over flattened EEG channel tokens
    and peripheral features `(channel_embeddings, x_periph)`.
    Encodes minority batch samples into a compact latent bottleneck, runs k-NN
    SMOTE interpolation in latent space, decodes back to joint feature space,
    and contributes a reconstruction auxiliary loss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def oversample(
        self,
        channel_embeddings: torch.Tensor,
        x_periph: torch.Tensor,
        y: torch.Tensor,
        subj_ids: torch.Tensor,
        k_neighbors: int = 5,
    ):
        B, C, D = channel_embeddings.shape
        device = channel_embeddings.device

        flat_in = torch.cat([channel_embeddings.reshape(B, C * D), x_periph], dim=1)
        latent = self.encoder(flat_in.detach())
        recon = self.decoder(latent)
        recon_loss = F.mse_loss(recon, flat_in.detach())

        info = _select_minority_majority(y)
        if info is None:
            return channel_embeddings, x_periph, y, subj_ids, recon_loss

        minority_class, n_minority, n_to_generate = info
        min_idx = torch.where(y == minority_class)[0]
        min_lat = latent[min_idx]
        min_subjs = subj_ids[min_idx]

        k = min(int(k_neighbors), n_minority - 1)
        if k <= 0:
            gen_pick = torch.randint(0, n_minority, (n_to_generate,), device=device)
            syn_lat = min_lat[gen_pick]
            gen_subjs = min_subjs[gen_pick]
        else:
            dist = torch.cdist(min_lat.float(), min_lat.float())
            _, nn_idx = torch.topk(dist, k=k + 1, largest=False)
            nn_idx = nn_idx[:, 1:]
            base = torch.randint(0, n_minority, (n_to_generate,), device=device)
            nbr = nn_idx[base, torch.randint(0, k, (n_to_generate,), device=device)]
            gap = torch.rand((n_to_generate, 1), device=device, dtype=min_lat.dtype)
            syn_lat = min_lat[base] + gap * (min_lat[nbr] - min_lat[base])
            gen_subjs = min_subjs[base]

        syn_decoded = self.decoder(syn_lat)
        syn_emb = syn_decoded[:, : C * D].reshape(n_to_generate, C, D)
        syn_periph = syn_decoded[:, C * D :]
        syn_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)

        new_emb = torch.cat([channel_embeddings, syn_emb.to(channel_embeddings.dtype)], dim=0)
        new_per = torch.cat([x_periph, syn_periph.to(x_periph.dtype)], dim=0)
        new_y = torch.cat([y, syn_y], dim=0)
        new_subjs = torch.cat([subj_ids, gen_subjs], dim=0)

        perm = torch.randperm(new_emb.size(0), device=device)
        return new_emb[perm], new_per[perm], new_y[perm], new_subjs[perm], recon_loss


class GAMOLatent(nn.Module):
    """
    GAMO-inspired conditional generator + discriminator oversampler in joint
    `(channel_embeddings, x_periph)` space with bounded residual perturbation
    and adversarial feature-matching regularization.
    """

    def __init__(self, input_dim: int, noise_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = nn.Sequential(
            nn.Linear(input_dim + noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def oversample(
        self,
        channel_embeddings: torch.Tensor,
        x_periph: torch.Tensor,
        y: torch.Tensor,
        subj_ids: torch.Tensor,
    ):
        B, C, D = channel_embeddings.shape
        device = channel_embeddings.device

        info = _select_minority_majority(y)
        if info is None:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return channel_embeddings, x_periph, y, subj_ids, zero

        minority_class, n_minority, n_to_generate = info
        min_idx = torch.where(y == minority_class)[0]
        flat_min = torch.cat(
            [channel_embeddings[min_idx].reshape(n_minority, C * D), x_periph[min_idx]], dim=1
        ).detach()
        min_subjs = subj_ids[min_idx]

        pick = torch.randint(0, n_minority, (n_to_generate,), device=device)
        base = flat_min[pick]
        noise = torch.randn((n_to_generate, self.noise_dim), device=device, dtype=base.dtype)
        # Residual convex-like perturbation keeps generated embeddings near real manifold
        delta = 0.25 * torch.tanh(self.generator(torch.cat([base, noise], dim=1))) * (base.std(dim=0, keepdim=True) + 0.1)
        syn_flat = base + delta

        # Real vs Fake discrimination without leaking g_loss gradients into discriminator
        real_logits = self.discriminator(flat_min.float())
        fake_logits_d = self.discriminator(syn_flat.detach().float())
        d_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits_d, torch.zeros_like(fake_logits_d))
        )

        # Temporarily freeze discriminator params while computing generator adversarial loss
        prev_reqs = [p.requires_grad for p in self.discriminator.parameters()]
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        fake_logits_g = self.discriminator(syn_flat.float())
        g_loss = F.binary_cross_entropy_with_logits(fake_logits_g, torch.ones_like(fake_logits_g))
        for p, req in zip(self.discriminator.parameters(), prev_reqs):
            p.requires_grad_(req)

        aux_loss = 0.05 * (d_loss + g_loss)

        syn_emb = syn_flat[:, : C * D].reshape(n_to_generate, C, D)
        syn_per = syn_flat[:, C * D :]
        gen_subjs = min_subjs[pick]
        syn_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)

        new_emb = torch.cat([channel_embeddings, syn_emb.to(channel_embeddings.dtype)], dim=0)
        new_per = torch.cat([x_periph, syn_per.to(x_periph.dtype)], dim=0)
        new_y = torch.cat([y, syn_y], dim=0)
        new_subjs = torch.cat([subj_ids, gen_subjs], dim=0)

        perm = torch.randperm(new_emb.size(0), device=device)
        return new_emb[perm], new_per[perm], new_y[perm], new_subjs[perm], aux_loss
