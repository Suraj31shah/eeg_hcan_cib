import os
import torch
import numpy as np
import mne
from torch.utils.data import Dataset, DataLoader

class EEGMMIDBDataset(Dataset):
    """
    PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB v1.0.0).
    64 channels, 160 Hz sampling rate.
    
    Supported tasks:
      - 'imagery_left_right': 2-class (Left Fist vs Right Fist) - Runs 4, 8, 12
      - 'imagery_3class': 3-class (Rest vs Left Fist vs Right Fist) - Runs 4, 8, 12
      - 'imagery_4class': 4-class (Left vs Right vs Both Fists vs Both Feet) - Runs 4, 8, 12 & 6, 10, 14
    """
    def __init__(self, data_root, subject_list=None, task='imagery_left_right',
                 tmin=0.0, tmax=4.0, l_freq=4.0, h_freq=40.0,
                 cache_path=None, force_reload=False):
        super().__init__()
        self.data_root = data_root
        self.task = task
        self.tmin = tmin
        self.tmax = tmax
        self.l_freq = l_freq
        self.h_freq = h_freq
        
        # Load cache if available
        if cache_path and os.path.exists(cache_path) and not force_reload:
            print(f"[EEGMMIDB] Loading preprocessed data from cache: {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            self.x = cached['x']
            self.y = cached['y']
            self.subject_ids = cached['subject_ids']
            self.trial_ids = cached['trial_ids']
            self.channel_names = cached.get('channel_names', [])
            return

        all_subjects = sorted([d for d in os.listdir(data_root) if d.startswith('S') and os.path.isdir(os.path.join(data_root, d))])
        if subject_list is not None:
            self.subjects = [s for s in all_subjects if s in subject_list]
        else:
            self.subjects = all_subjects
            
        print(f"[EEGMMIDB] Processing {len(self.subjects)} subjects for task '{self.task}'...")
        
        # Determine runs per task
        if self.task in ['imagery_left_right', 'imagery_3class']:
            self.run_nums = ['04', '08', '12']
        elif self.task == 'imagery_4class':
            self.run_nums = ['04', '08', '12', '06', '10', '14']
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        all_x, all_y, all_subj, all_trial = [], [], [], []
        global_trial_id = 0
        self.channel_names = None
        
        for subj_name in self.subjects:
            subj_num = int(subj_name[1:]) # e.g. S001 -> 1
            subj_dir = os.path.join(data_root, subj_name)
            subj_x_list = []
            subj_y_list = []
            
            for r in self.run_nums:
                edf_path = os.path.join(subj_dir, f"{subj_name}R{r}.edf")
                if not os.path.exists(edf_path):
                    continue
                    
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    if self.channel_names is None:
                        self.channel_names = [ch.strip('.').upper() for ch in raw.ch_names]
                        
                    # Handle sampling rate alignment (S088, S092, S100 were recorded at 128 Hz)
                    if raw.info['sfreq'] != 160.0:
                        raw.resample(160.0, verbose=False)
                        
                    # Bandpass filter for motor imagery mu/beta rhythms
                    if self.l_freq or self.h_freq:
                        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
                        
                    events, event_dict = mne.events_from_annotations(raw, verbose=False)
                    
                    if self.task == 'imagery_left_right':
                        # Keep T1 (left) and T2 (right)
                        keep_events = {k: v for k, v in event_dict.items() if k in ['T1', 'T2']}
                        if len(keep_events) < 2:
                            continue
                        epochs = mne.Epochs(raw, events, event_id=keep_events, tmin=self.tmin, tmax=self.tmax,
                                            baseline=None, preload=True, verbose=False)
                        epochs_data = epochs.get_data(copy=True) # [n_epochs, 64, n_times]
                        # Map: T1 -> 0, T2 -> 1
                        labels = np.array([0 if ev == keep_events['T1'] else 1 for ev in epochs.events[:, -1]], dtype=np.int64)
                        
                    elif self.task == 'imagery_3class':
                        # Keep T0 (rest=0), T1 (left=1), T2 (right=2)
                        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax,
                                            baseline=None, preload=True, verbose=False)
                        epochs_data = epochs.get_data(copy=True)
                        labels = np.array([0 if ev == event_dict.get('T0') else (1 if ev == event_dict.get('T1') else 2) 
                                           for ev in epochs.events[:, -1]], dtype=np.int64)
                        
                    elif self.task == 'imagery_4class':
                        # Runs 4, 8, 12: T1 (left=0), T2 (right=1)
                        # Runs 6, 10, 14: T1 (both fists=2), T2 (both feet=3)
                        is_hands_feet = r in ['06', '10', '14']
                        keep_events = {k: v for k, v in event_dict.items() if k in ['T1', 'T2']}
                        if len(keep_events) < 2:
                            continue
                        epochs = mne.Epochs(raw, events, event_id=keep_events, tmin=self.tmin, tmax=self.tmax,
                                            baseline=None, preload=True, verbose=False)
                        epochs_data = epochs.get_data(copy=True)
                        if is_hands_feet:
                            labels = np.array([2 if ev == keep_events['T1'] else 3 for ev in epochs.events[:, -1]], dtype=np.int64)
                        else:
                            labels = np.array([0 if ev == keep_events['T1'] else 1 for ev in epochs.events[:, -1]], dtype=np.int64)
                            
                    # Truncate/pad to exact 640 samples (4.0s @ 160Hz)
                    target_len = int((self.tmax - self.tmin) * 160)
                    if epochs_data.shape[-1] > target_len:
                        epochs_data = epochs_data[..., :target_len]
                    elif epochs_data.shape[-1] < target_len:
                        pad_width = target_len - epochs_data.shape[-1]
                        epochs_data = np.pad(epochs_data, ((0, 0), (0, 0), (0, pad_width)), mode='edge')
                        
                    subj_x_list.append(epochs_data)
                    subj_y_list.append(labels)
                except Exception as e:
                    print(f"Warning: Failed reading {edf_path}: {e}")
                    continue
                    
            if len(subj_x_list) == 0:
                continue
                
            subj_x = np.concatenate(subj_x_list, axis=0) # [N_trials, 64, 640]
            subj_y = np.concatenate(subj_y_list, axis=0)
            
            # Transductive per-subject Z-score normalization across all channels
            mean = np.mean(subj_x, axis=(0, 2), keepdims=True)
            std = np.std(subj_x, axis=(0, 2), keepdims=True) + 1e-8
            subj_x = (subj_x - mean) / std
            
            n_subj_trials = len(subj_y)
            all_x.append(subj_x)
            all_y.append(subj_y)
            all_subj.append(np.full(n_subj_trials, subj_num, dtype=np.int64))
            all_trial.append(np.arange(global_trial_id, global_trial_id + n_subj_trials, dtype=np.int64))
            global_trial_id += n_subj_trials
            
        self.x = torch.tensor(np.concatenate(all_x, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(all_y, axis=0), dtype=torch.int64)
        self.subject_ids = torch.tensor(np.concatenate(all_subj, axis=0), dtype=torch.int64)
        self.trial_ids = torch.tensor(np.concatenate(all_trial, axis=0), dtype=torch.int64)
        
        print(f"[EEGMMIDB] Loaded Total: {len(self.y)} trials across {len(self.subjects)} subjects. Shape: {self.x.shape}")
        unique_classes, counts = torch.unique(self.y, return_counts=True)
        print(f"[EEGMMIDB] Class distribution: {dict(zip(unique_classes.tolist(), counts.tolist()))}")
        
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"[EEGMMIDB] Saving preprocessed tensor cache to {cache_path}...")
            torch.save({
                'x': self.x,
                'y': self.y,
                'subject_ids': self.subject_ids,
                'trial_ids': self.trial_ids,
                'channel_names': self.channel_names
            }, cache_path)
            
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.subject_ids[idx], self.trial_ids[idx]

def get_dataloaders(dataset, train_idx, test_idx, batch_size=64):
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, test_loader

