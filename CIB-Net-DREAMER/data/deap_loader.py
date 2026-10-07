import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class DEAPDataset(Dataset):
    def __init__(self, data_path, target='valence', preprocess_fn=None):
        """
        Loads DEAP dataset. 
        data_path: Directory containing s01.dat to s32.dat
        """
        self.data_path = data_path
        self.target = target
        self.preprocess_fn = preprocess_fn
        
        self.x = []
        self.y = []
        self.subject_ids = []
        
        self._load_data()
        
    def _load_data(self):
        for subject_id in range(1, 33):
            file_name = f's{subject_id:02d}.dat'
            file_path = os.path.join(self.data_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            with open(file_path, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                
            data = content['data']    # shape (40, 40, 8064)
            labels = content['labels'] # shape (40, 4)
            
            # Extract only the 32 EEG channels
            eeg_data = data[:, :32, :] 
            
            if self.preprocess_fn:
                eeg_data = self.preprocess_fn(eeg_data)
            
            # Extract target label (0: valence, 1: arousal, 2: dominance, 3: liking)
            target_idx = 0 if self.target == 'valence' else 1
            # Binarize labels: > 5 is High (1), <= 5 is Low (0)
            binary_labels = (labels[:, target_idx] > 5).astype(np.int64)
            
            self.x.append(eeg_data)
            self.y.append(binary_labels)
            self.subject_ids.extend([subject_id] * 40)
            
        if len(self.x) > 0:
            self.x = np.concatenate(self.x, axis=0)
            self.y = np.concatenate(self.y, axis=0)
            self.subject_ids = np.array(self.subject_ids)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.long), \
               torch.tensor(self.subject_ids[idx], dtype=torch.long)

def get_dataloaders(dataset, train_idx, test_idx, batch_size=64):
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader
