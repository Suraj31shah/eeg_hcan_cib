import numpy as np
import networkx as nx
from scipy.signal import hilbert


def compute_plv_centrality(eeg_data: np.ndarray) -> np.ndarray:
    """
    Computes Phase Locking Value (PLV) connectivity matrix and extracts
    eigenvector centrality for each EEG channel.

    Args:
        eeg_data: `[n_channels, time_samples]` for a single window/trial.
    Returns:
        eigenvector centrality array `[n_channels]`
    """
    channels = eeg_data.shape[0]
    analytic_signal = hilbert(eeg_data, axis=1)
    phase = np.angle(analytic_signal)

    plv_matrix = np.zeros((channels, channels), dtype=np.float64)
    for i in range(channels):
        for j in range(i + 1, channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    np.fill_diagonal(plv_matrix, 1.0)
    G = nx.from_numpy_array(plv_matrix)
    try:
        centrality_dict = nx.eigenvector_centrality_numpy(G, weight="weight")
        centrality = np.array([centrality_dict[i] for i in range(channels)], dtype=np.float32)
    except Exception:
        centrality = np.ones(channels, dtype=np.float32) / channels
    return centrality

