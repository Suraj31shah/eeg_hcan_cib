import os
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def butter_bandpass_filter(data, lowcut=4.0, highcut=45.0, fs=128, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def preprocess_eeg(eeg_data, fs=128, baseline_sec=3):
    filtered = butter_bandpass_filter(eeg_data, fs=fs)
    baseline_samples = baseline_sec * fs
    baseline = np.mean(filtered[:, :, :baseline_samples], axis=-1, keepdims=True)
    corrected = filtered[:, :, baseline_samples:] - baseline
    
    mean = np.mean(corrected, axis=-1, keepdims=True)
    std = np.std(corrected, axis=-1, keepdims=True)
    return (corrected - mean) / (std + 1e-8)

class DEAPDataset(Dataset):
    def __init__(self, data_path, target='valence'):
        self.x, self.y, self.subject_ids = [], [], []
        
        for subject_id in range(1, 33):
            file_path = os.path.join(data_path, f's{subject_id:02d}.dat')
            if not os.path.exists(file_path): continue
                
            with open(file_path, 'rb') as f:
                content = pickle.load(f, encoding='latin1')
                
            data, labels = content['data'], content['labels']
            eeg_data = data[:, :32, :] 
            eeg_data = preprocess_eeg(eeg_data)
            
            target_idx = 0 if target == 'valence' else 1
            binary_labels = (labels[:, target_idx] > 5).astype(np.int64)
            
            self.x.append(eeg_data)
            self.y.append(binary_labels)
            self.subject_ids.extend([subject_id] * 40)
            
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.subject_ids = np.array(self.subject_ids)
        
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.long), \
               torch.tensor(self.subject_ids[idx], dtype=torch.long)

def get_dataloaders(dataset, train_idx, test_idx, batch_size=16):
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler), \
           DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# ==========================================
# BASELINE MODELS
# ==========================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_classes=2, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.depthwise1 = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pooling2 = nn.AvgPool2d(1, 4)
        self.separable1 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.pooling3 = nn.AvgPool2d(1, 8)
        self.dropout = nn.Dropout(dropout_rate)
        # Use LazyLinear to dynamically infer the flattened shape on the first pass!
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = nn.ELU()(x)
        x = self.pooling2(x)
        x = self.dropout(x)
        x = self.separable1(x)
        x = self.batchnorm3(x)
        x = nn.ELU()(x)
        x = self.pooling3(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class PlainCNN(nn.Module):
    def __init__(self, n_channels=32, n_classes=2):
        super(PlainCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=10, stride=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=10, stride=2), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ACRNN(nn.Module):
    def __init__(self, n_channels=32, n_classes=2):
        super(ACRNN, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2), nn.ELU(),
            nn.Linear(n_channels // 2, n_channels), nn.Sigmoid()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=16, stride=4, padding=8), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4), nn.BatchNorm1d(128), nn.ELU(), nn.MaxPool1d(2)
        )
        self.rnn = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(128, n_classes))

    def forward(self, x):
        avg_pool = torch.mean(x, dim=2) 
        attn_weights = self.channel_attention(avg_pool).unsqueeze(-1)
        x = x * attn_weights
        feat = self.cnn(x).permute(0, 2, 1) 
        rnn_out, _ = self.rnn(feat)
        return self.classifier(torch.mean(rnn_out, dim=1))

# ==========================================
# EVALUATION & TRAINING HARNESS
# ==========================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    
    # Formula for Delta Over Trivial:
    # 1. Trivial Accuracy = (Count of Majority Class) / (Total Samples)
    # 2. Delta = Model Accuracy - Trivial Accuracy
    majority_acc = np.max(np.unique(y_true, return_counts=True)[1]) / len(y_true)
    delta_trivial = acc - majority_acc
    
    return acc, delta_trivial, f1_score(y_true, y_pred, average='macro', zero_division=0)

if __name__ == '__main__':
    LOCAL_DATA_PATH = "D:/Deep Learning Lab/Research/deap-dataset/data_preprocessed_python"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Local environment initialized. Device: {device}")
    
    if not os.path.exists(LOCAL_DATA_PATH) or not os.path.exists(os.path.join(LOCAL_DATA_PATH, 's01.dat')):
        print(f"ERROR: Dataset not found at {LOCAL_DATA_PATH}.")
        exit(1)
        
    print("Loading DEAP Dataset (this will take a minute or two)...")
    dataset = DEAPDataset(data_path=LOCAL_DATA_PATH, target='valence')
    
    logo = LeaveOneGroupOut()
    subject_ids = dataset.subject_ids
    
    # You can change this to PlainCNN() or ACRNN() to test other baselines
    print("Starting LOSO Cross-Validation on Baseline: EEGNet")
    
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(dataset.x, dataset.y, groups=subject_ids)):
        test_subject = subject_ids[test_idx[0]]
        print(f"\n--- Fold {fold + 1}/32 (Test Subject: {test_subject}) ---")
        
        train_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx, batch_size=16)
        
        model = EEGNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(50):
            model.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, y, _ in test_loader:
                    x = x.to(device)
                    logits = model(x)
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_targets.extend(y.numpy())
                    
            acc, delta, macro = compute_metrics(all_targets, all_preds)
            if acc > best_acc: 
                best_acc = acc
                best_delta = delta
            print(f"  Epoch {epoch+1}/50 | Test Acc: {acc:.4f} | Macro F1: {macro:.4f} | Delta: {delta:+.4f}")
            
        fold_metrics.append({'acc': best_acc, 'delta': best_delta})
        print(f"Best Fold Acc: {best_acc:.4f} | Delta Over Trivial: {best_delta:+.4f}")
        
    print("\n==============================")
    final_acc = np.mean([m['acc'] for m in fold_metrics])
    final_std = np.std([m['acc'] for m in fold_metrics])
    final_delta = np.mean([m['delta'] for m in fold_metrics])
    print(f"FINAL BASELINE LOSO ACCURACY: {final_acc:.4f} ± {final_std:.4f}")
    print(f"FINAL DELTA OVER TRIVIAL: {final_delta:+.4f}")
    print("==============================")
