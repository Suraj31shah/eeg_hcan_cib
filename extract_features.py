import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt

# Append path to import WindowedDEAPDataset
sys.path.append(os.path.join(os.path.dirname(__file__), 'CIB-Net'))
try:
    from data.windowed_loader import WindowedDEAPDataset
except ImportError:
    print("Could not import WindowedDEAPDataset. Make sure CIB-Net is in the same directory.")
    sys.exit(1)

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y

def compute_de(x):
    variance = np.var(x, axis=-1, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * variance + 1e-8)

def extract_de_features(windows, fs=128):
    # windows shape: (N, 32, 512)
    bands = {
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }
    
    N, C, T = windows.shape
    de_features = np.zeros((N, C * len(bands)))
    
    for i, (band_name, (low, high)) in enumerate(bands.items()):
        print(f"Extracting {band_name} band...")
        filtered = butter_bandpass_filter(windows, low, high, fs)
        de = compute_de(filtered) # Shape: (N, 32)
        de_features[:, i*C:(i+1)*C] = de
        
    return de_features

def main():
    print("Loading DEAP Dataset via WindowedDEAPDataset...")
    import yaml
    
    config_path = "CIB-Net/config/hcan_cib_config.yaml"
    if not os.path.exists(config_path):
        config_path = "config/hcan_cib_config.yaml"
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_path = config.get('data_path', '')
    except Exception as e:
        print(f"Could not load config: {e}")
        data_path = ""

    if not os.path.exists(data_path):
        print(f"Data path '{data_path}' not found.")
        data_path = input("Enter the absolute path to your DEAP/DREAMER dataset directory (containing s01.dat etc): ").strip()

    dataset = WindowedDEAPDataset(data_path, binarize='median', window_sec=4.0, overlap_sec=2.0)
    
    print(f"Dataset loaded. Total windows: {len(dataset.windows)}")
    
    windows = np.array(dataset.windows) # (N, 32, 512)
    periph = np.array(dataset.peripheral) # (N, 24)
    y = np.array(dataset.labels) # (N,)
    subject = np.array(dataset.subject_ids) # (N,)
    trial = np.array(dataset.trial_ids) # (N,)
    
    print("Computing DE features (this might take a minute)...")
    eeg_de = extract_de_features(windows, fs=128)
    print(f"DE features shape: {eeg_de.shape}")
    
    out_file = "feats.npz"
    np.savez(out_file, eeg=eeg_de, periph=periph, y=y, subject=subject, trial=trial)
    print(f"Saved all features to {out_file}")

if __name__ == '__main__':
    main()
