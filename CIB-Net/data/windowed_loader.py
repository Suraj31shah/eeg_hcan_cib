import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

class WindowedDEAPDataset(Dataset):
    def __init__(self, data_path, target='valence', binarize='median', window_sec=4.0, overlap_sec=2.0, fs=128, baseline_sec=3.0):
        """
        Loads DEAP dataset with overlapping windows and peripheral features.
        binarize: 'median' (Task P, balanced) or 'fixed' (Task F, fixed threshold 5, imbalanced)
        """
        self.data_path = data_path
        self.target = target
        self.binarize = binarize
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.fs = fs
        self.baseline_sec = baseline_sec
        
        self.window_samples = int(self.window_sec * self.fs)
        self.step_samples = int((self.window_sec - self.overlap_sec) * self.fs)
        self.baseline_samples = int(self.baseline_sec * self.fs)
        
        # Data containers
        self.windows = []          # EEG windows (N, 32, window_samples)
        self.peripheral = []       # Peripheral trial-level features (N, 24)
        self.labels = []           # Binary labels (N,)
        self.subject_ids = []      # Subject ID (N,)
        self.trial_ids = []        # Unique Trial ID (N,)
        
        self._load_and_process_data()
        
    def _extract_peripheral_features(self, periph_data):
        """
        Extracts mean, std, range for the 8 peripheral channels (channels 32-39).
        periph_data shape: [8, n_samples]
        Returns: [24]
        """
        means = np.mean(periph_data, axis=-1)
        stds = np.std(periph_data, axis=-1)
        ranges = np.ptp(periph_data, axis=-1)
        return np.concatenate([means, stds, ranges])
        
    def _load_and_process_data(self):
        global_trial_id = 0
        
        all_subjects_labels = []
        raw_content = {}
        
        # First pass: load everything and compute labels based on median if needed
        for subject_id in range(1, 33):
            file_name = f's{subject_id:02d}.dat'
            file_path = os.path.join(self.data_path, file_name)
            
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                
            raw_content[subject_id] = content
            
            target_idx = 0 if self.target == 'valence' else 1
            raw_labels = content['labels'][:, target_idx]
            all_subjects_labels.append((subject_id, raw_labels))
            
        if not all_subjects_labels:
            raise ValueError(f"CRITICAL ERROR: No DEAP dataset files (s01.dat to s32.dat) were found at path: '{self.data_path}'. Please check your config/hcan_cib_config.yaml data_path!")
            
        # Process each subject
        for subject_id, raw_labels in all_subjects_labels:
            content = raw_content[subject_id]
            data = content['data'] # [40, 40, 8064]
            
            # Determine threshold
            if self.binarize == 'median':
                threshold = np.median(raw_labels)
            else:
                threshold = 5.0
                
            binary_labels = (raw_labels > threshold).astype(np.int64)
            
            for trial_idx in range(40):
                trial_data = data[trial_idx] # [40, 8064]
                
                # Split EEG and Peripheral
                eeg = trial_data[:32, :]
                periph = trial_data[32:, :]
                
                # Preprocess EEG
                eeg_filtered = butter_bandpass_filter(eeg, 4.0, 45.0, self.fs)
                
                # Baseline z-scoring
                baseline_data = eeg_filtered[:, :self.baseline_samples]
                baseline_mean = np.mean(baseline_data, axis=-1, keepdims=True)
                baseline_std = np.std(baseline_data, axis=-1, keepdims=True)
                
                # Normalize the 60s trial
                trial_eeg = eeg_filtered[:, self.baseline_samples:]
                trial_eeg = (trial_eeg - baseline_mean) / (baseline_std + 1e-8)
                
                # Preprocess Peripheral (remove baseline samples, then extract features)
                trial_periph = periph[:, self.baseline_samples:]
                periph_features = self._extract_peripheral_features(trial_periph)
                
                # Windowing
                n_samples = trial_eeg.shape[-1]
                for start in range(0, n_samples - self.window_samples + 1, self.step_samples):
                    end = start + self.window_samples
                    window = trial_eeg[:, start:end]
                    
                    self.windows.append(window)
                    self.peripheral.append(periph_features)
                    self.labels.append(binary_labels[trial_idx])
                    self.subject_ids.append(subject_id)
                    self.trial_ids.append(global_trial_id)
                    
                global_trial_id += 1
                
        self.windows = np.array(self.windows, dtype=np.float32)
        self.peripheral = np.array(self.peripheral, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids, dtype=np.int64)
        self.trial_ids = np.array(self.trial_ids, dtype=np.int64)
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), \
               torch.tensor(self.peripheral[idx]), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.subject_ids[idx]), \
               torch.tensor(self.trial_ids[idx])

def get_windowed_dataloaders(dataset, train_idx, test_idx, batch_size=64):
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader
