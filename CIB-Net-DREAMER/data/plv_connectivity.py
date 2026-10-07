import numpy as np
import networkx as nx
from scipy.signal import hilbert

def compute_plv_centrality(eeg_data):
    """
    Computes Phase Locking Value (PLV) connectivity matrix and extracts
    eigenvector centrality for each channel.
    
    eeg_data: [n_channels, time_samples] for a single trial
    Returns: eigenvector centrality array [n_channels]
    """
    channels = eeg_data.shape[0]
    
    # Instantaneous phase via Hilbert Transform
    analytic_signal = hilbert(eeg_data, axis=1)
    phase = np.angle(analytic_signal)
    
    plv_matrix = np.zeros((channels, channels))
    for i in range(channels):
        for j in range(i+1, channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
            
    # Symmetrize and add self-loops
    plv_matrix += plv_matrix.T
    np.fill_diagonal(plv_matrix, 1.0)
    
    # Compute graph eigenvector centrality
    G = nx.from_numpy_array(plv_matrix)
    try:
        centrality_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
        centrality = np.array([centrality_dict[i] for i in range(channels)])
    except:
        # Fallback if convergence fails
        centrality = np.ones(channels) / channels
        
    return centrality
