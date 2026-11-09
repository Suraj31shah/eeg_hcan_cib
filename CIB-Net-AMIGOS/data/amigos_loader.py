"""
AMIGOS Dataset Loader & Windowed Preprocessor for CIB-Net / HCAN-CIB.

Verified AMIGOS Preprocessed (.mat) Schema (Data_Preprocessed_P01..P40.mat):
  - joined_data: cell array of trials (16 short clips [trials 0..15] + up to 4 long clips [16..19])
      Each non-empty cell has shape (T, 17) sampled at 128 Hz:
        cols  0..13 : 14 EEG channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
        cols 14..15 : 2 ECG channels (ECG Right, ECG Left)
        col  16     : 1 GSR channel
  - labels_selfassessment: cell array or matrix (n_trials, 12) in official AMIGOS order:
        0: arousal (1-9), 1: valence (1-9), 2: dominance (1-9),
        3: liking (1-9), 4: familiarity (1-9), 5..11: discrete emotions

Note on Kaggle Datasets:
  - `nsut123/amigos`          contains ALL 40 participants (Data_Preprocessed_P01..P40.mat)
  - `nsut123/amigos-dataset`  contains 15 participants (P01, P07, P08, P12..P15, P22..P25, P28, P36, P37, P40)
  `find_amigos_files()` recursively discovers whichever is mounted.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None


# Official AMIGOS 12-field labels_selfassessment order (0-indexed!)
AMIGOS_LABEL_COLS = {
    "arousal": 0,
    "valence": 1,
    "dominance": 2,
    "liking": 3,
    "familiarity": 4,
}

# Official Emotiv EPOC 14-channel order in AMIGOS
AMIGOS_EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.999)
    return butter(order, [low, high], btype="band")


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter along last axis `[C, T]`."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    padlen = 3 * max(len(a), len(b))
    if data.shape[-1] <= padlen:
        return data.copy()
    return filtfilt(b, a, data, axis=-1)


def interpolate_nonfinite(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Linearly interpolate NaN/Inf values per channel along the last axis `[C, T]`.
    Returns repaired array and count of non-finite entries replaced.
    """
    arr = np.asarray(arr, dtype=np.float64).copy()
    bad_mask = ~np.isfinite(arr)
    n_bad = int(bad_mask.sum())
    if n_bad == 0:
        return arr, 0

    t_idx = np.arange(arr.shape[-1])
    for c in range(arr.shape[0]):
        good = np.isfinite(arr[c])
        n_good = int(good.sum())
        if n_good == 0:
            arr[c] = 0.0
        elif n_good == 1:
            arr[c] = arr[c, good][0]
        elif not good.all():
            arr[c, ~good] = np.interp(t_idx[~good], t_idx[good], arr[c, good])
    return arr, n_bad


def _subject_id_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"P0*(\d+)", path.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", path.stem)
    return int(m2.group(1)) if m2 else None


def find_amigos_files(data_path: Optional[str] = None) -> List[Tuple[int, str]]:
    """
    Recursively discover AMIGOS `Data_Preprocessed_P*.mat` files.
    Returns sorted list of `(subject_id, file_path)`.
    """
    roots: List[Path] = []
    if data_path and str(data_path).strip():
        p = Path(str(data_path).strip())
        if p.exists():
            roots.append(p)
        else:
            warnings.warn(f"Configured data_path does not exist: '{data_path}'. Searching standard paths...")

    if not roots:
        for candidate in [
            "/kaggle/input",
            os.getcwd(),
            os.path.dirname(os.getcwd()),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ]:
            cp = Path(candidate)
            if cp.exists():
                roots.append(cp)

    found: Dict[int, Tuple[int, str]] = {}
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".mat":
            mat_paths = [root]
        else:
            mat_paths = sorted(root.rglob("*.mat"))

        for mp in mat_paths:
            name_lower = mp.name.lower()
            # Match Data_Preprocessed_PXX.mat (or any PXX.mat inside amigos path)
            if "preprocessed" not in name_lower and "amigos" not in str(mp).lower():
                if not re.match(r"^p?\d+\.mat$", name_lower):
                    continue
            sid = _subject_id_from_filename(mp)
            if sid is None or sid <= 0:
                continue
            size = mp.stat().st_size
            # Prefer larger file if duplicate subject IDs exist across mounts
            if sid not in found or size > found[sid][0]:
                found[sid] = (size, str(mp))
        if found:
            break

    return [(sid, info[1]) for sid, info in sorted(found.items())]


def _h5_decode(node, root):
    if isinstance(node, h5py.Group):
        return {k: _h5_decode(node[k], root) for k in node.keys() if not k.startswith("#")}
    if node.attrs.get("MATLAB_empty", 0):
        return np.empty((0,), dtype=np.float64)
    arr = node[()]
    if getattr(arr.dtype, "kind", None) == "O":
        out = np.empty(arr.shape, dtype=object)
        for idx, ref in np.ndenumerate(arr):
            out[idx] = _h5_decode(root[ref], root) if ref else None
        return out
    return np.asarray(arr)


def _read_mat_file(path: str) -> Dict[str, object]:
    try:
        raw = sio.loadmat(path, simplify_cells=False, verify_compressed_data_integrity=False)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except (NotImplementedError, ValueError, OSError):
        if h5py is None:
            raise
        with h5py.File(path, "r") as f:
            return {k: _h5_decode(f[k], f) for k in f.keys() if not k.startswith("#")}


def _unwrap_cells(var) -> List[Optional[np.ndarray]]:
    """
    Convert MATLAB cell array or stacked array into an index-aligned list.
    Preserves empty cells (`None`) so trial index positions stay aligned.
    """
    if var is None:
        return []
    if isinstance(var, list):
        return [None if v is None else np.asarray(v) for v in var]
    arr = np.asarray(var)
    if arr.dtype == object:
        out = []
        for item in arr.ravel(order="C"):
            if item is None:
                out.append(None)
            else:
                a = np.asarray(item)
                out.append(None if a.size == 0 else a)
        return out
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        axis = 0 if arr.shape[0] <= 32 else -1
        return [np.take(arr, i, axis=axis) for i in range(arr.shape[axis])]
    return [arr]


def _orient_samples_by_channels(arr: Optional[np.ndarray], expected_cols: int = 17) -> Optional[np.ndarray]:
    """Ensure trial array has shape `[n_samples, n_channels]` with `n_samples > n_channels`."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float64).squeeze()
    except Exception:
        return None
    if a.size == 0 or a.ndim != 2:
        return None
    # Standard AMIGOS is (samples, 17); transpose if stored as (17, samples)
    if a.shape[0] <= 32 and a.shape[1] > 32:
        a = a.T
    if a.shape[1] < 14 or a.shape[0] < 64:
        return None
    return a


def _extract_label_matrix(label_var, n_trials_hint: int) -> List[Optional[np.ndarray]]:
    """Extract per-trial 12-D self-assessment vectors, preserving index alignment."""
    if label_var is None:
        return [None] * n_trials_hint
    arr = np.asarray(label_var)
    if arr.dtype != object and arr.ndim == 2:
        if arr.shape[1] == 12 and arr.shape[0] != 12:
            return [np.asarray(arr[i], dtype=np.float64).ravel() for i in range(arr.shape[0])]
        if arr.shape[0] == 12 and arr.shape[1] != 12:
            return [np.asarray(arr[:, i], dtype=np.float64).ravel() for i in range(arr.shape[1])]
        if arr.shape[1] >= 12 and arr.shape[0] < 12:
            return [np.asarray(arr[i], dtype=np.float64).ravel() for i in range(arr.shape[0])]
        if abs(arr.shape[0] - n_trials_hint) <= abs(arr.shape[1] - n_trials_hint):
            return [np.asarray(arr[i], dtype=np.float64).ravel() for i in range(arr.shape[0])]
        return [np.asarray(arr[:, i], dtype=np.float64).ravel() for i in range(arr.shape[1])]
    cells = _unwrap_cells(label_var)
    out: List[Optional[np.ndarray]] = []
    for c in cells:
        if c is None or c.size == 0:
            out.append(None)
        else:
            out.append(np.asarray(c, dtype=np.float64).ravel())
    return out


class WindowedAMIGOSDataset(Dataset):
    """
    Windowed AMIGOS Dataset aligned cleanly with CIB-Net's DEAP and DREAMER loaders.

    Each item returns `(eeg_window, periph_features, label, subject_id, trial_id)`:
      - `eeg_window`:      FloatTensor `[14, window_samples]` (default `[14, 512]`)
      - `periph_features`: FloatTensor `[18]` (6 stats x [ECG_R, ECG_L, GSR])
      - `label`:           LongTensor scalar (`0` or `1`)
      - `subject_id`:      LongTensor scalar (AMIGOS subject number `1..40`)
      - `trial_id`:        LongTensor scalar (unique global trial index across dataset)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        target: str = "valence",
        binarize: str = "fixed",
        threshold: float = 5.0,
        video_subset: str = "short",
        max_short_duration_sec: float = 240.0,
        window_sec: float = 4.0,
        overlap_sec: float = 2.0,
        fs: int = 128,
        baseline_sec: float = 5.0,
        bandpass_low: float = 4.0,
        bandpass_high: float = 45.0,
        normalize: str = "none",
        subjects: Optional[List[int]] = None,
    ):
        super().__init__()
        if target not in AMIGOS_LABEL_COLS:
            raise ValueError(f"Unsupported AMIGOS target '{target}'. Choose from {list(AMIGOS_LABEL_COLS)}.")
        if binarize not in ("fixed", "median"):
            raise ValueError(f"Unsupported binarize mode '{binarize}'. Use 'fixed' or 'median'.")
        if normalize not in ("none", "zscore"):
            raise ValueError(f"Unsupported normalize mode '{normalize}'. Use 'none' or 'zscore'.")

        self.data_path = data_path
        self.target = target
        self.target_col = AMIGOS_LABEL_COLS[target]
        self.binarize = binarize
        self.threshold = float(threshold)
        self.video_subset = video_subset
        self.max_short_duration_sec = float(max_short_duration_sec)
        self.window_sec = float(window_sec)
        self.overlap_sec = float(overlap_sec)
        self.fs = int(fs)
        self.baseline_sec = float(baseline_sec)
        self.bandpass_low = float(bandpass_low)
        self.bandpass_high = float(bandpass_high)
        self.normalize = normalize
        self.subjects_filter = set(int(s) for s in subjects) if subjects is not None else None

        self.window_samples = int(round(self.window_sec * self.fs))
        self.step_samples = max(1, int(round((self.window_sec - self.overlap_sec) * self.fs)))
        self.baseline_samples = int(round(self.baseline_sec * self.fs))

        # Containers exposed to downstream scripts
        self.windows: List[np.ndarray] = []
        self.peripheral: List[np.ndarray] = []
        self.labels: List[int] = []
        self.subject_ids: List[int] = []
        self.trial_ids: List[int] = []
        self.video_ids: List[int] = []

        self.stats = {
            "subjects_loaded": 0,
            "trials_loaded": 0,
            "skipped_empty": 0,
            "skipped_long": 0,
            "skipped_short": 0,
            "skipped_invalid_label": 0,
            "nan_values_repaired": 0,
        }

        self._load_and_process()

    @staticmethod
    def extract_peripheral_features(periph_window: np.ndarray) -> np.ndarray:
        """
        Extract 6 summary statistics from each of the 3 peripheral channels
        (`ECG_R`, `ECG_L`, `GSR`).
        Input shape: `[3, T]` -> Output shape: `[18]`.
        """
        feats = []
        for c in range(periph_window.shape[0]):
            ch = periph_window[c]
            feats.extend([
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.min(ch)),
                float(np.max(ch)),
                float(np.ptp(ch)),
                float(np.median(ch)),
            ])
        return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _load_and_process(self) -> None:
        files = find_amigos_files(self.data_path)
        if self.subjects_filter is not None:
            files = [(sid, p) for sid, p in files if sid in self.subjects_filter]

        if not files:
            raise ValueError(
                f"No AMIGOS Data_Preprocessed_P*.mat files found (searched '{self.data_path}' and /kaggle/input). "
                "Please verify your AMIGOS dataset path."
            )

        transductive_tag = (
            "TRANSDUCTIVE per-subject zscore" if self.normalize == "zscore" else "NONE (raw / zero-leakage)"
        )
        print(
            f"[AMIGOS] Found {len(files)} participant .mat files | target={self.target} "
            f"(col={self.target_col}) | binarize={self.binarize} | video_subset={self.video_subset} "
            f"| normalize={transductive_tag}"
        )

        global_trial_id = 0

        for subject_id, fpath in files:
            mat = _read_mat_file(fpath)
            joined_key = next((k for k in mat if k.lower() == "joined_data"), None)
            label_key = next((k for k in mat if "label" in k.lower() and "self" in k.lower()), None)
            if joined_key is None or label_key is None:
                warnings.warn(f"Skipping subject P{subject_id:02d} ({fpath}): missing joined_data or labels_selfassessment.")
                continue

            trial_cells = _unwrap_cells(mat[joined_key])
            label_rows = _extract_label_matrix(mat[label_key], len(trial_cells))

            vid_key = next((k for k in mat if k.lower() in ("videoids", "video_ids", "video_id")), None)
            vid_cells = _unwrap_cells(mat[vid_key]) if vid_key is not None else []

            subj_eegs: List[np.ndarray] = []
            subj_periphs: List[np.ndarray] = []
            subj_scores: List[float] = []
            subj_vids: List[int] = []

            n_cells = min(len(trial_cells), len(label_rows))
            for ti in range(n_cells):
                arr = _orient_samples_by_channels(trial_cells[ti])
                if arr is None:
                    self.stats["skipped_empty"] += 1
                    continue

                duration_sec = arr.shape[0] / float(self.fs)
                if self.video_subset == "short" and duration_sec > self.max_short_duration_sec:
                    self.stats["skipped_long"] += 1
                    continue

                lbl_vec = label_rows[ti]
                if lbl_vec is None or len(lbl_vec) <= self.target_col:
                    self.stats["skipped_invalid_label"] += 1
                    continue
                raw_score = float(lbl_vec[self.target_col])
                # In AMIGOS self-assessments, 1-9 are valid ratings; 0 or NaN mark unrated/missing trials
                if not np.isfinite(raw_score) or not (1.0 <= raw_score <= 9.0):
                    self.stats["skipped_invalid_label"] += 1
                    continue

                if arr.shape[0] < self.baseline_samples + self.window_samples:
                    self.stats["skipped_short"] += 1
                    continue

                eeg_raw = arr[:, :14].T  # [14, T]
                if arr.shape[1] >= 17:
                    per_raw = arr[:, 14:17].T  # [3, T]
                else:
                    per_raw = np.zeros((3, arr.shape[0]), dtype=np.float64)
                    avail_p = min(3, arr.shape[1] - 14)
                    if avail_p > 0:
                        per_raw[:avail_p] = arr[:, 14:14 + avail_p].T

                eeg_clean, n_bad_e = interpolate_nonfinite(eeg_raw)
                per_clean, n_bad_p = interpolate_nonfinite(per_raw)
                self.stats["nan_values_repaired"] += (n_bad_e + n_bad_p)

                # Broadband bandpass filter 4-45 Hz (zero-phase order-4 Butterworth)
                eeg_filt = butter_bandpass_filter(
                    eeg_clean, self.bandpass_low, self.bandpass_high, self.fs, order=4
                )

                # Baseline removal (crop first 5s fixation period; subtract baseline mean if present)
                if self.baseline_samples > 0:
                    eeg_base = np.mean(eeg_filt[:, :self.baseline_samples], axis=-1, keepdims=True)
                    eeg_post = eeg_filt[:, self.baseline_samples:] - eeg_base
                    per_post = per_clean[:, self.baseline_samples:]
                else:
                    eeg_post = eeg_filt
                    per_post = per_clean

                if eeg_post.shape[1] < self.window_samples:
                    self.stats["skipped_short"] += 1
                    continue

                # Extract VideoID integer if available, else fallback 1-based trial index
                vid_id = ti + 1
                if ti < len(vid_cells) and vid_cells[ti] is not None and vid_cells[ti].size > 0:
                    try:
                        vid_id = int(np.asarray(vid_cells[ti]).ravel()[0])
                    except Exception:
                        m_vid = re.search(r"(\d+)", str(np.asarray(vid_cells[ti]).ravel()[0]))
                        if m_vid:
                            vid_id = int(m_vid.group(1))

                subj_eegs.append(eeg_post)
                subj_periphs.append(per_post)
                subj_scores.append(raw_score)
                subj_vids.append(vid_id)

            if not subj_eegs:
                warnings.warn(f"Subject P{subject_id:02d} yielded 0 valid trials; skipping subject.")
                continue

            self.stats["subjects_loaded"] += 1
            self.stats["trials_loaded"] += len(subj_eegs)

            scores_arr = np.asarray(subj_scores, dtype=np.float64)
            if self.binarize == "median":
                subj_thresh = float(np.median(scores_arr))
            else:
                subj_thresh = self.threshold
            bin_labels = (scores_arr > subj_thresh).astype(np.int64)

            if self.normalize == "zscore":
                # Variable trial lengths across AMIGOS -> compute stats over concatenated time axis
                eeg_cat = np.concatenate(subj_eegs, axis=1)
                per_cat = np.concatenate(subj_periphs, axis=1)
                eeg_mu = np.mean(eeg_cat, axis=1, keepdims=True)
                eeg_sd = np.std(eeg_cat, axis=1, keepdims=True)
                per_mu = np.mean(per_cat, axis=1, keepdims=True)
                per_sd = np.std(per_cat, axis=1, keepdims=True)
                subj_eegs = [(e - eeg_mu) / (eeg_sd + 1e-8) for e in subj_eegs]
                subj_periphs = [(p - per_mu) / (per_sd + 1e-8) for p in subj_periphs]

            for eeg_tr, per_tr, lbl, vid_id in zip(subj_eegs, subj_periphs, bin_labels, subj_vids):
                n_samples = eeg_tr.shape[1]
                for start in range(0, n_samples - self.window_samples + 1, self.step_samples):
                    end = start + self.window_samples
                    win_eeg = eeg_tr[:, start:end].astype(np.float32)
                    win_per = per_tr[:, start:end]
                    per_feat = self.extract_peripheral_features(win_per)

                    self.windows.append(win_eeg)
                    self.peripheral.append(per_feat)
                    self.labels.append(int(lbl))
                    self.subject_ids.append(int(subject_id))
                    self.trial_ids.append(int(global_trial_id))
                    self.video_ids.append(int(vid_id))

                global_trial_id += 1

        if not self.windows:
            raise RuntimeError("No valid AMIGOS windows produced. Inspect dataset files and config parameters.")

        self.windows = np.stack(self.windows, axis=0).astype(np.float32)
        self.peripheral = np.stack(self.peripheral, axis=0).astype(np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.subject_ids = np.asarray(self.subject_ids, dtype=np.int64)
        self.trial_ids = np.asarray(self.trial_ids, dtype=np.int64)
        self.video_ids = np.asarray(self.video_ids, dtype=np.int64)

        pos_rate = float(self.labels.mean())
        print(
            f"[AMIGOS Ready] subjects={self.stats['subjects_loaded']} | trials={self.stats['trials_loaded']} "
            f"| windows={len(self.windows)} | pos_rate={pos_rate:.3f} "
            f"| skipped(empty={self.stats['skipped_empty']}, long={self.stats['skipped_long']}, "
            f"short={self.stats['skipped_short']}, bad_label={self.stats['skipped_invalid_label']}) "
            f"| nan_repaired={self.stats['nan_values_repaired']}"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.peripheral[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.subject_ids[idx], dtype=torch.long),
            torch.tensor(self.trial_ids[idx], dtype=torch.long),
        )
