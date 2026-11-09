import torch
import torch.nn.functional as F


def compute_graph_consistency_loss(learned_importance: torch.Tensor, plv_centrality: torch.Tensor) -> torch.Tensor:
    """
    Stage D Auxiliary Loss: compares learned channel importance against PLV eigenvector centrality.
    """
    cos_sim = F.cosine_similarity(learned_importance.unsqueeze(0), plv_centrality.unsqueeze(0))
    return (1.0 - cos_sim).mean()

