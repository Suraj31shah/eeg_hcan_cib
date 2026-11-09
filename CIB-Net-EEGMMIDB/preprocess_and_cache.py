import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.eegmmidb_loader import EEGMMIDBDataset

cache_file = "E:/kagglehub_cache/eegmmidb_cache_all109.pt"
raw_data_path = "E:/kagglehub_cache/datasets/utsavsinghal2604/eeg-motorimagery-eegmmidb-v1-0-0/versions/1/physionet.org/files/eegmmidb/1.0.0"

print(f"[CACHE] Generating full preprocessed dataset cache for EEGMMIDB...")
print(f"[CACHE] Raw data: {raw_data_path}")
print(f"[CACHE] Output destination: {cache_file}")

dataset = EEGMMIDBDataset(
    data_root=raw_data_path,
    subject_list=None, # All subjects S001-S109
    task="imagery_left_right",
    tmin=0.0,
    tmax=4.0,
    l_freq=4.0,
    h_freq=40.0,
    cache_path=cache_file,
    force_reload=True
)

print(f"[CACHE] Complete! Saved {len(dataset)} trials from {len(dataset.subjects)} subjects.")
print(f"[CACHE] Final Tensor Shape: {dataset.x.shape}")
print(f"[CACHE] Cache Size on Disk: {os.path.getsize(cache_file) / (1024*1024):.2f} MB")

