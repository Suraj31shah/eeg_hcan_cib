"""
Differential Entropy (DE) + Peripheral Feature Extraction for AMIGOS.

Extracts raw (zero-leakage, non-normalized) features from `WindowedAMIGOSDataset(..., normalize='none')`
so downstream `domain_shift_diagnostics.py` and `shallow_baselines.py` can compare
`norm='none'`, `norm='center'`, and `norm='zscore'` fairly:
  - EEG: 4 frequency bands (theta 4-8 Hz, alpha 8-12 Hz, beta 12-30 Hz, gamma 30-45 Hz)
         x 14 Emotiv EPOC channels = 56 features per window (averaged per trial).
  - Peripheral: 6 summary stats x 3 channels (ECG_R, ECG_L, GSR) = 18 features per window.
"""

import argparse
import os
import sys
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.amigos_loader import WindowedAMIGOSDataset, butter_bandpass_filter


BANDS = [
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta", 12.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def compute_de(x: np.ndarray) -> np.ndarray:
    variance = np.var(x, axis=-1, ddof=1)
    return 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(variance, 0.0) + 1e-8)


def extract_de_and_save(
    config_path: str = "config/amigos_config.yaml",
    data_path: str = None,
    target: str = None,
    binarize: str = None,
    video_subset: str = None,
    out_dir: str = "results/features",
    subjects=None,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    target = target or cfg.get("target", "valence")
    binarize = binarize or cfg.get("binarize", "fixed")
    video_subset = video_subset or cfg.get("video_subset", "short")
    fs = int(cfg.get("sample_rate", 128))
    dpath = data_path if data_path is not None else cfg.get("data_path", "")

    # Always extract unnormalized ('none') features so diagnostics/shallow compare raw vs zscore cleanly
    ds = WindowedAMIGOSDataset(
        data_path=dpath,
        target=target,
        binarize=binarize,
        threshold=float(cfg.get("threshold", 5.0)),
        video_subset=video_subset,
        max_short_duration_sec=float(cfg.get("max_short_duration_sec", 240.0)),
        window_sec=float(cfg.get("window_sec", 4.0)),
        overlap_sec=float(cfg.get("overlap_sec", 2.0)),
        fs=fs,
        baseline_sec=float(cfg.get("baseline_sec", 5.0)),
        bandpass_low=float(cfg.get("bandpass_low", 4.0)),
        bandpass_high=float(cfg.get("bandpass_high", 45.0)),
        normalize="none",
        subjects=subjects,
    )

    n_win, n_ch, _ = ds.windows.shape
    n_bands = len(BANDS)
    de_win = np.zeros((n_win, n_ch, n_bands), dtype=np.float32)

    print(f"[DE Extract] Filtering {n_win} windows across {n_bands} frequency bands...")
    for bi, (bname, lo, hi) in enumerate(BANDS):
        # Compute along window axis in chunks to bound peak RAM
        for start in range(0, n_win, 2000):
            end = min(n_win, start + 2000)
            band_sig = butter_bandpass_filter(ds.windows[start:end].astype(np.float64), lo, hi, fs, order=4)
            de_win[start:end, :, bi] = compute_de(band_sig).astype(np.float32)

    eeg_flat = np.nan_to_num(de_win.reshape(n_win, n_ch * n_bands), nan=0.0, posinf=0.0, neginf=0.0)
    periph_flat = np.nan_to_num(ds.peripheral.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    os.makedirs(out_dir, exist_ok=True)
    win_path = os.path.join(out_dir, f"feats_amigos_{target}_{binarize}_win.npz")
    np.savez_compressed(
        win_path,
        eeg=eeg_flat,
        periph=periph_flat,
        y=ds.labels.astype(np.int64),
        subject=ds.subject_ids.astype(np.int64),
        trial=ds.trial_ids.astype(np.int64),
        video=ds.video_ids.astype(np.int64),
    )

    # Aggregate window -> trial level cleanly
    keys = ds.subject_ids.astype(np.int64) * 1_000_000 + ds.trial_ids.astype(np.int64)
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)

    eeg_tr = np.zeros((len(uniq), eeg_flat.shape[1]), dtype=np.float32)
    for j in range(eeg_flat.shape[1]):
        eeg_tr[:, j] = (np.bincount(inv, weights=eeg_flat[:, j]) / cnt).astype(np.float32)

    per_tr = np.zeros((len(uniq), periph_flat.shape[1]), dtype=np.float32)
    for j in range(periph_flat.shape[1]):
        per_tr[:, j] = (np.bincount(inv, weights=periph_flat[:, j]) / cnt).astype(np.float32)

    trial_path = os.path.join(out_dir, f"feats_amigos_{target}_{binarize}_trial.npz")
    np.savez_compressed(
        trial_path,
        eeg=eeg_tr,
        periph=per_tr,
        y=ds.labels[first].astype(np.int64),
        subject=ds.subject_ids[first].astype(np.int64),
        trial=ds.trial_ids[first].astype(np.int64),
        video=ds.video_ids[first].astype(np.int64),
    )

    print(f"[DE Extract] Saved window feats: {win_path} (shape={eeg_flat.shape})")
    print(f"[DE Extract] Saved trial feats : {trial_path} (shape={eeg_tr.shape}, n_subj={len(np.unique(ds.subject_ids))})")
    return trial_path, win_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/amigos_config.yaml")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--target", default=None, choices=["valence", "arousal", "dominance"])
    ap.add_argument("--binarize", default=None, choices=["fixed", "median"])
    ap.add_argument("--video_subset", default=None, choices=["short", "all"])
    ap.add_argument("--out_dir", default="results/features")
    ap.add_argument("--subjects", nargs="*", type=int, default=None)
    args = ap.parse_args()

    extract_de_and_save(
        config_path=args.config,
        data_path=args.data_path,
        target=args.target,
        binarize=args.binarize,
        video_subset=args.video_subset,
        out_dir=args.out_dir,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()

