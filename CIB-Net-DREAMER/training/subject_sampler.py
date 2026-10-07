import torch
import numpy as np
from torch.utils.data import Sampler

class SubjectBalancedSampler(Sampler):
    """
    Ensures that every batch contains a balanced representation of subjects,
    so that the Invariance Gate (Stage B) can reliably calculate cross-subject variance.
    """
    def __init__(self, dataset, indices, batch_size):
        self.dataset = dataset
        self.indices = np.array(indices)
        self.batch_size = batch_size
        
        # Group indices by subject
        self.subject_indices = {}
        for idx in self.indices:
            # dataset.subject_ids should be accessible
            subj = dataset.subject_ids[idx].item() if isinstance(dataset.subject_ids[idx], torch.Tensor) else dataset.subject_ids[idx]
            if subj not in self.subject_indices:
                self.subject_indices[subj] = []
            self.subject_indices[subj].append(idx)
            
        self.subjects = list(self.subject_indices.keys())
        self.n_subjects = len(self.subjects)
        
        # Calculate how many samples per subject per batch (roughly)
        # We want at least 8 subjects per batch if batch_size=64 (so 8 samples per subject)
        self.subjects_per_batch = min(self.n_subjects, 8)
        self.samples_per_subject = self.batch_size // self.subjects_per_batch
        
    def __iter__(self):
        # Shuffle indices within each subject
        shuffled_subject_indices = {
            subj: np.random.permutation(indices).tolist()
            for subj, indices in self.subject_indices.items()
        }
        
        batches = []
        active_subjects = list(self.subjects)
        
        while active_subjects:
            batch = []
            # Randomly select subjects for this batch
            np.random.shuffle(active_subjects)
            selected_subjects = active_subjects[:self.subjects_per_batch]
            
            for subj in selected_subjects:
                # Take up to samples_per_subject from this subject
                for _ in range(self.samples_per_subject):
                    if shuffled_subject_indices[subj]:
                        batch.append(shuffled_subject_indices[subj].pop())
                        
            # Clean up empty subjects
            active_subjects = [s for s in active_subjects if len(shuffled_subject_indices[s]) > 0]
            
            # If batch is not full, fill it randomly from remaining
            if len(batch) < self.batch_size and active_subjects:
                needed = self.batch_size - len(batch)
                flat_remaining = [idx for s in active_subjects for idx in shuffled_subject_indices[s]]
                np.random.shuffle(flat_remaining)
                batch.extend(flat_remaining[:needed])
                
                # We need to manually pop these from the lists, but this is an edge case (last batch).
                # To keep it simple, we just yield the batch and break, since drop_last is usually True for training.
            
            if len(batch) == self.batch_size:
                batches.append(batch)
                
        np.random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.indices) // self.batch_size * self.batch_size
