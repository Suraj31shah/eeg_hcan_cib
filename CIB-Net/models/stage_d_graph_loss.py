import torch
import torch.nn.functional as F

def compute_graph_consistency_loss(learned_importance, plv_centrality):
    """
    Stage D Auxiliary Loss: Compares model's learned importance against neuroscience prior.
    
    learned_importance: tensor [N_channels] 
                        (e.g., L2 norm of the channel embeddings or attention weights)
    plv_centrality: tensor [N_channels] 
                        (Eigenvector centrality from PLV graph)
    """
    # Maximize cosine similarity -> minimize (1 - cos_sim)
    cos_sim = F.cosine_similarity(learned_importance.unsqueeze(0), plv_centrality.unsqueeze(0))
    loss = 1.0 - cos_sim
    return loss.mean()
