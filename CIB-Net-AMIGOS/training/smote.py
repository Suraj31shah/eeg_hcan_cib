import numpy as np
import torch


def _select_minority_majority(y: torch.Tensor):
    y_np = y.detach().cpu().numpy()
    classes, counts = np.unique(y_np, return_counts=True)
    if len(classes) < 2:
        return None
    min_pos = int(np.argmin(counts))
    max_pos = int(np.argmax(counts))
    majority_class = int(classes[max_pos])
    minority_class = int(classes[min_pos])
    n_majority = int(counts[max_pos])
    n_minority = int(counts[min_pos])
    n_to_generate = n_majority - n_minority
    if n_to_generate <= 0 or n_minority == 0:
        return None
    return minority_class, n_minority, n_to_generate


def apply_latent_smote(
    channel_embeddings: torch.Tensor,
    x_periph: torch.Tensor,
    y: torch.Tensor,
    subj_ids: torch.Tensor,
    k_neighbors: int = 5,
    alpha: float = 1.0,
    variance_preserve: bool = True,
):
    """
    Latent SMOTE / Variance-Preserving V-SMOTE on `(channel_embeddings, x_periph)` pairs.

    Args:
        channel_embeddings: `[B, n_channels, d_embed]`
        x_periph: `[B, n_periph_features]`
        y: `[B]`
        subj_ids: `[B]`
        k_neighbors: number of nearest neighbors among minority samples in the batch
        alpha: maximum interpolation step multiplier
        variance_preserve: if True, scales step factor by per-dimension normalized minority std
                           (V-SMOTE); if False, uses uniform scalar interpolation in `[0, 1]`.
    """
    device = channel_embeddings.device
    info = _select_minority_majority(y)
    if info is None:
        return channel_embeddings, x_periph, y, subj_ids

    minority_class, n_minority, n_to_generate = info
    min_indices = torch.where(y == minority_class)[0]
    min_embeddings = channel_embeddings[min_indices]
    min_periph = x_periph[min_indices]
    min_subjs = subj_ids[min_indices]

    k = min(int(k_neighbors), n_minority - 1)
    if k <= 0:
        gen_idx = torch.randint(0, n_minority, (n_to_generate,), device=device)
        new_embeddings = torch.cat([channel_embeddings, min_embeddings[gen_idx]], dim=0)
        new_periph = torch.cat([x_periph, min_periph[gen_idx]], dim=0)
        new_y = torch.cat(
            [y, torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)], dim=0
        )
        new_subjs = torch.cat([subj_ids, min_subjs[gen_idx]], dim=0)
        perm = torch.randperm(new_embeddings.size(0), device=device)
        return new_embeddings[perm], new_periph[perm], new_y[perm], new_subjs[perm]

    flat_emb = min_embeddings.reshape(n_minority, -1)
    dist_matrix = torch.cdist(flat_emb.float(), flat_emb.float())
    _, nn_indices = torch.topk(dist_matrix, k=k + 1, largest=False)
    nn_indices = nn_indices[:, 1:]  # [N_min, k]

    base_idx = torch.randint(0, n_minority, (n_to_generate,), device=device)
    nbr_slot = torch.randint(0, k, (n_to_generate,), device=device)
    nbr_idx = nn_indices[base_idx, nbr_slot]

    if variance_preserve and n_minority > 1:
        emb_var = torch.var(min_embeddings, dim=0, keepdim=False) + 1e-6
        emb_scale = emb_var / (torch.max(emb_var) + 1e-6)
        per_var = torch.var(min_periph, dim=0, keepdim=False) + 1e-6
        per_scale = per_var / (torch.max(per_var) + 1e-6)

        w_e = torch.rand((n_to_generate, 1, 1), device=device, dtype=min_embeddings.dtype) * alpha * emb_scale.unsqueeze(0)
        w_p = torch.rand((n_to_generate, 1), device=device, dtype=min_periph.dtype) * alpha * per_scale.unsqueeze(0)
    else:
        w_scalar = torch.rand((n_to_generate,), device=device, dtype=min_embeddings.dtype) * alpha
        w_e = w_scalar.view(-1, 1, 1)
        w_p = w_scalar.view(-1, 1).to(dtype=min_periph.dtype)

    gen_emb = min_embeddings[base_idx] + w_e * (min_embeddings[nbr_idx] - min_embeddings[base_idx])
    gen_periph = min_periph[base_idx] + w_p * (min_periph[nbr_idx] - min_periph[base_idx])
    gen_subjs = min_subjs[base_idx]
    gen_y = torch.full((n_to_generate,), minority_class, dtype=y.dtype, device=device)

    new_embeddings = torch.cat([channel_embeddings, gen_emb], dim=0)
    new_periph = torch.cat([x_periph, gen_periph], dim=0)
    new_y = torch.cat([y, gen_y], dim=0)
    new_subjs = torch.cat([subj_ids, gen_subjs], dim=0)

    perm = torch.randperm(new_embeddings.size(0), device=device)
    return new_embeddings[perm], new_periph[perm], new_y[perm], new_subjs[perm]


def apply_v_smote(channel_embeddings, x_periph, y, subj_ids, k_neighbors=5, alpha=1.0):
    return apply_latent_smote(
        channel_embeddings,
        x_periph,
        y,
        subj_ids,
        k_neighbors=k_neighbors,
        alpha=alpha,
        variance_preserve=True,
    )

