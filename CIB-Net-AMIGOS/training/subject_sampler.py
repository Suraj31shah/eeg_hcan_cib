import numpy as np
import torch
from torch.utils.data import Sampler


class SubjectBalancedSampler(Sampler):
    """
    Ensures batches mix samples across multiple subjects so the Stage B
    Invariance Gate reliably computes cross-subject variance.
    """

    def __init__(self, dataset, indices, batch_size: int):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)

        self.subject_indices = {}
        for idx in self.indices:
            subj = dataset.subject_ids[idx]
            if isinstance(subj, torch.Tensor):
                subj = subj.item()
            subj = int(subj)
            self.subject_indices.setdefault(subj, []).append(int(idx))

        self.subjects = list(self.subject_indices.keys())
        self.n_subjects = len(self.subjects)
        self.subjects_per_batch = min(max(self.n_subjects, 1), 8)
        self.samples_per_subject = max(1, self.batch_size // self.subjects_per_batch)

    def __iter__(self):
        shuffled = {
            subj: np.random.permutation(idxs).tolist()
            for subj, idxs in self.subject_indices.items()
        }
        batches = []
        active_subjects = list(self.subjects)

        while active_subjects:
            batch = []
            np.random.shuffle(active_subjects)
            selected = active_subjects[: self.subjects_per_batch]
            for subj in selected:
                for _ in range(self.samples_per_subject):
                    if shuffled[subj]:
                        batch.append(shuffled[subj].pop())

            active_subjects = [s for s in active_subjects if len(shuffled[s]) > 0]

            if len(batch) < self.batch_size and active_subjects:
                needed = self.batch_size - len(batch)
                # Pull extra items while popping them cleanly from shuffled lists
                while needed > 0 and active_subjects:
                    s = active_subjects[np.random.randint(0, len(active_subjects))]
                    batch.append(shuffled[s].pop())
                    needed -= 1
                    if not shuffled[s]:
                        active_subjects.remove(s)

            if len(batch) == self.batch_size:
                batches.append(batch)

        np.random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return (len(self.indices) // self.batch_size) * self.batch_size

