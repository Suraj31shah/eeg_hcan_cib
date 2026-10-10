import os
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

class WindowedDREAMERDataset(Dataset):
    def __init__(self, data_path, target='valence', binarize=True, 
                 window_sec=4, overlap_sec=2, eeg_fs=128, ecg_fs=256):
        """
        Loads the DREAMER dataset from the .mat file.
        Slices the EEG (14 channels) and ECG (2 channels) into windows.
        
        target: 'valence', 'arousal', or 'dominance'
        binarize: If True, threshold at 3 (<=3 is 0, >3 is 1)
        """
        super().__init__()
        self.target = target
        self.binarize = binarize
        self.window_samples_eeg = int(window_sec * eeg_fs)
        self.step_samples_eeg = int((window_sec - overlap_sec) * eeg_fs)
        
        self.window_samples_ecg = int(window_sec * ecg_fs)
        self.step_samples_ecg = int((window_sec - overlap_sec) * ecg_fs)
        
        self.windows = []
        self.peripheral = []
        self.labels = []
        self.subject_ids = []
        self.trial_ids = []
        
        self._load_and_window(data_path)
        
    def _extract_ecg_features(self, ecg_window):
        """
        Extracts 6 statistical features from each of the 2 ECG channels.
        Input: [1024, 2]
        Output: [12]
        """
        features = []
        for c in range(ecg_window.shape[1]):
            chan_data = ecg_window[:, c]
            features.extend([
                np.mean(chan_data),
                np.std(chan_data),
                np.min(chan_data),
                np.max(chan_data),
                np.ptp(chan_data), # range
                np.median(chan_data)
            ])
        return np.array(features, dtype=np.float32)

    def _load_and_window(self, data_path):
        if not os.path.exists(data_path):
            raise ValueError(f"DREAMER dataset not found at {data_path}")
            
        print(f"Loading DREAMER from {data_path} (This may take a minute...)")
        mat = sio.loadmat(data_path)
        data = mat['DREAMER'][0, 0]['Data'][0]
        
        n_subjects = len(data)
        global_trial_id = 0
        
        for subj_idx in range(n_subjects):
            subj = data[subj_idx]
            
            # Scores: [1, 1][18, 1]
            if self.target == 'valence':
                scores = subj['ScoreValence'][0, 0]
            elif self.target == 'arousal':
                scores = subj['ScoreArousal'][0, 0]
            else:
                scores = subj['ScoreDominance'][0, 0]
                
            eeg_stimuli = subj['EEG'][0, 0]['stimuli'][0, 0] # [18, 1]
            ecg_stimuli = subj['ECG'][0, 0]['stimuli'][0, 0] # [18, 1]
            
            n_trials = eeg_stimuli.shape[0]
            
            eeg_all = []
            ecg_all = []
            labels_all = []
            
            # 1. Collect all raw trials for the subject
            for trial_idx in range(n_trials):
                # EEG: [samples, 14] -> Transpose to [14, samples]
                trial_eeg = eeg_stimuli[trial_idx, 0].T 
                # ECG: [samples, 2]
                trial_ecg = ecg_stimuli[trial_idx, 0]
                
                label = scores[trial_idx, 0]
                if self.binarize:
                    label = 1 if label > 3 else 0
                    
                eeg_all.append(trial_eeg)
                ecg_all.append(trial_ecg)
                labels_all.append(label)
                
            # 2. Compute TRANSDUCTIVE PER-SUBJECT Z-SCORE NORMALIZATION
            eeg_concat = np.concatenate(eeg_all, axis=1) # [14, total_samples]
            ecg_concat = np.concatenate(ecg_all, axis=0) # [total_samples, 2]
            
            eeg_subj_mean = np.mean(eeg_concat, axis=1, keepdims=True)
            eeg_subj_std = np.std(eeg_concat, axis=1, keepdims=True)
            
            ecg_subj_mean = np.mean(ecg_concat, axis=0, keepdims=True)
            ecg_subj_std = np.std(ecg_concat, axis=0, keepdims=True)
            
            # 3. Apply normalization and extract windows
            for trial_idx in range(n_trials):
                trial_eeg = eeg_all[trial_idx]
                trial_ecg = ecg_all[trial_idx]
                label = labels_all[trial_idx]
                
                # Apply normalization
                trial_eeg = (trial_eeg - eeg_subj_mean) / (eeg_subj_std + 1e-8)
                trial_ecg = (trial_ecg - ecg_subj_mean) / (ecg_subj_std + 1e-8)
                    
                n_samples_eeg = trial_eeg.shape[1]
                n_samples_ecg = trial_ecg.shape[0]
                
                # We iterate based on EEG, and compute corresponding ECG bounds
                for start_eeg in range(0, int(n_samples_eeg - self.window_samples_eeg + 1), int(self.step_samples_eeg)):
                    end_eeg = start_eeg + self.window_samples_eeg
                    
                    # Convert EEG indices to ECG indices (fs ratio is 2)
                    start_ecg = start_eeg * 2
                    end_ecg = end_eeg * 2
                    
                    if end_ecg > n_samples_ecg:
                        break
                        
                    window_eeg = trial_eeg[:, start_eeg:end_eeg]
                    window_ecg = trial_ecg[start_ecg:end_ecg, :]
                    
                    ecg_features = self._extract_ecg_features(window_ecg)
                    
                    self.windows.append(window_eeg)
                    self.peripheral.append(ecg_features)
                    self.labels.append(label)
                    self.subject_ids.append(subj_idx + 1)
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
