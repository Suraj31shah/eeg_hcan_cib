import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y

def preprocess_eeg(eeg_data, fs=128, baseline_sec=3, lowcut=4.0, highcut=45.0):
    """
    Preprocesses DEAP EEG data.
    eeg_data: [n_trials, n_channels, n_samples]
    """
    # 1. Bandpass filter 4-45 Hz
    filtered_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs)
    
    # 2. Baseline removal
    baseline_samples = baseline_sec * fs
    baseline = np.mean(filtered_data[:, :, :baseline_samples], axis=-1, keepdims=True)
    corrected_data = filtered_data[:, :, baseline_samples:] - baseline
    
    # 3. Z-score normalization per channel per trial
    mean = np.mean(corrected_data, axis=-1, keepdims=True)
    std = np.std(corrected_data, axis=-1, keepdims=True)
    normalized_data = (corrected_data - mean) / (std + 1e-8)
    
    return normalized_data
