"""
extract_de_v2.py  — DE feature extraction for the v2 pipeline.

Definitions copied VERBATIM from existing code (sources listed below).
NO labels are read. NO normalization beyond baseline subtraction is applied.

SOURCE DEFINITIONS COPIED:
  windowed_loader.py line 8-18   : butter_bandpass (order=4, filtfilt, btype='band')
  windowed_loader.py line 21     : window_sec=4.0, overlap_sec=2.0, fs=128, baseline_sec=3.0
  windowed_loader.py line 34-36  : window_samples=512, step_samples=256, baseline_samples=384
  windowed_loader.py line 103    : EEG = channels 0..31 of data[trial]
  windowed_loader.py line 104    : periph = channels 32..39 of data[trial]
  windowed_loader.py line 107    : broadband filter 4..45 Hz BEFORE baseline removal
  windowed_loader.py line 110-111: baseline removal = drop first 384 samples
  windowed_loader.py line 47-56  : peripheral = [means(8), stds(8), ptp(8)] = 24-D
  windowed_loader.py line 116-126: PER-SUBJECT Z-SCORE across all 40 trials and time
  extract_features.py line 32-37 : bands theta(4-8) alpha(8-12) beta(12-30) gamma(30-45)
  extract_features.py line 26-28 : DE = 0.5*log(2*pi*e*var(x,ddof=1) + 1e-8)

NOTE: windowed_loader.py filter order=4; extract_features.py uses order=3.
      This script uses order=4 (windowed_loader.py, the training pipeline).
NOTE: windowed_loader.py applies per-subject z-score normalization BEFORE
      windowing (lines 116-126).  This script applies the SAME step.
      This normalization is marked TRANSDUCTIVE because it uses all 40
      trials of a subject including the held-out test subject's own data.
"""

import os, sys, gc, time, pickle
import numpy as np
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------- constants
# --- from windowed_loader.py lines 21, 34-36 ---
FS           = 128
WINDOW_SEC   = 4.0
OVERLAP_SEC  = 2.0
BASELINE_SEC = 3.0
WINDOW_SAMP  = int(WINDOW_SEC * FS)          # 512
STEP_SAMP    = int((WINDOW_SEC - OVERLAP_SEC) * FS)   # 256
BASE_SAMP    = int(BASELINE_SEC * FS)        # 384

# --- from extract_features.py lines 32-37 ---
BANDS = [
    ('theta', 4,  8),
    ('alpha', 8,  12),
    ('beta',  12, 30),
    ('gamma', 30, 45),
]
N_BANDS = len(BANDS)   # 4

DP = r"D:\EEG_5Stages\deap-dataset\data_preprocessed_python"
OUT_WIN   = r"E:\audit_scratch\feats_v2\feats_window.npz"
OUT_TRIAL = r"E:\audit_scratch\feats_v2\feats_trial.npz"

# ---------------------------------------------------------------- functions
# --- from windowed_loader.py lines 8-18 ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y

# --- from extract_features.py lines 26-28 ---
def compute_de(x):
    variance = np.var(x, axis=-1, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * variance + 1e-8)

# --- from windowed_loader.py lines 47-56 ---
def extract_peripheral_features(periph_data):
    """periph_data: (8, n_samples) -> (24,)"""
    means  = np.mean(periph_data, axis=-1)
    stds   = np.std( periph_data, axis=-1)
    ranges = np.ptp( periph_data, axis=-1)
    return np.concatenate([means, stds, ranges])


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    try:
        import psutil
        _psutil = True
    except ImportError:
        _psutil = False

    all_eeg_win, all_periph_win, all_subj_win, all_vid_win, all_widx_win = [], [], [], [], []
    all_eeg_tr,  all_periph_tr,  all_subj_tr,  all_vid_tr                = [], [], [], []

    for subj in range(1, 33):
        fp = os.path.join(DP, f"s{subj:02d}.dat")
        with open(fp, "rb") as f:
            content = pickle.load(f, encoding="latin1")
        data = content["data"]  # (40, 40, 8064) — do NOT use content["labels"]
        del content
        gc.collect()

        # --- broadband filter + baseline removal (windowed_loader.py 107-111) ---
        eeg_all   = np.zeros((40, 32, 8064 - BASE_SAMP), dtype=np.float64)
        periph_all = np.zeros((40, 8,  8064 - BASE_SAMP), dtype=np.float64)
        for t in range(40):
            eeg_raw = data[t, :32, :].astype(np.float64)
            per_raw = data[t, 32:, :].astype(np.float64)
            eeg_filt = butter_bandpass_filter(eeg_raw, 4.0, 45.0, FS)   # line 107
            eeg_all[t]    = eeg_filt[:, BASE_SAMP:]                      # line 110
            periph_all[t] = per_raw[:, BASE_SAMP:]                       # line 111
        del data; gc.collect()

        # --- per-subject z-score (windowed_loader.py 116-126, TRANSDUCTIVE) ---
        eeg_mean = np.mean(eeg_all,    axis=(0, 2), keepdims=True)
        eeg_std  = np.std( eeg_all,    axis=(0, 2), keepdims=True)
        per_mean = np.mean(periph_all, axis=(0, 2), keepdims=True)
        per_std  = np.std( periph_all, axis=(0, 2), keepdims=True)
        eeg_norm  = (eeg_all    - eeg_mean)  / (eeg_std  + 1e-8)
        per_norm  = (periph_all - per_mean)  / (per_std  + 1e-8)
        del eeg_all, periph_all; gc.collect()

        # --- per-trial windowing ---
        for trial_idx in range(40):
            vid = trial_idx + 1
            trial_eeg  = eeg_norm[trial_idx]   # (32, n_samp)
            trial_per  = per_norm[trial_idx]    # (8,  n_samp)
            periph_feat = extract_peripheral_features(trial_per).astype(np.float32)  # (24,)
            n_samp = trial_eeg.shape[-1]

            win_eeg_list = []
            w_idx = 0
            for start in range(0, n_samp - WINDOW_SAMP + 1, STEP_SAMP):
                win = trial_eeg[:, start:start+WINDOW_SAMP]  # (32, 512)
                # --- band-DE: (32, 4) ---
                de_bands = np.zeros((32, N_BANDS), dtype=np.float32)
                for bi, (bname, lo, hi) in enumerate(BANDS):
                    filt = butter_bandpass_filter(win, lo, hi, FS)  # (32,512)
                    de_bands[:, bi] = compute_de(filt).astype(np.float32)
                win_eeg_list.append(de_bands)
                all_eeg_win.append(de_bands)
                all_periph_win.append(periph_feat)
                all_subj_win.append(subj)
                all_vid_win.append(vid)
                all_widx_win.append(w_idx)
                w_idx += 1

            # trial-level = mean over windows
            if win_eeg_list:
                tr_eeg  = np.mean(np.stack(win_eeg_list, axis=0), axis=0)  # (32,4)
            else:
                tr_eeg = np.zeros((32, N_BANDS), dtype=np.float32)
            all_eeg_tr.append(tr_eeg.astype(np.float32))
            all_periph_tr.append(periph_feat)
            all_subj_tr.append(subj)
            all_vid_tr.append(vid)

        del eeg_norm, per_norm; gc.collect()
        print(f"  Subject {subj:02d} done", flush=True)

    # --- save window-level ---
    eeg_win_arr    = np.stack(all_eeg_win,    axis=0).astype(np.float32)  # (N,32,4)
    periph_win_arr = np.stack(all_periph_win, axis=0).astype(np.float32)  # (N,24)
    subj_win_arr   = np.array(all_subj_win,   dtype=np.int32)
    vid_win_arr    = np.array(all_vid_win,    dtype=np.int32)
    widx_arr       = np.array(all_widx_win,   dtype=np.int32)
    np.savez_compressed(OUT_WIN,
        eeg_de=eeg_win_arr, periph=periph_win_arr,
        subject=subj_win_arr, video=vid_win_arr, window_idx=widx_arr)
    print(f"Saved window feats: {OUT_WIN}")
    print(f"  eeg_de shape: {eeg_win_arr.shape}  periph shape: {periph_win_arr.shape}")

    # --- save trial-level ---
    eeg_tr_arr    = np.stack(all_eeg_tr,    axis=0).astype(np.float32)  # (1280,32,4)
    periph_tr_arr = np.stack(all_periph_tr, axis=0).astype(np.float32)  # (1280,24)
    subj_tr_arr   = np.array(all_subj_tr,   dtype=np.int32)
    vid_tr_arr    = np.array(all_vid_tr,    dtype=np.int32)
    np.savez_compressed(OUT_TRIAL,
        eeg_de_trial=eeg_tr_arr, periph_trial=periph_tr_arr,
        subject=subj_tr_arr, video=vid_tr_arr)
    print(f"Saved trial feats:  {OUT_TRIAL}")
    print(f"  eeg_de_trial shape: {eeg_tr_arr.shape}  periph_trial shape: {periph_tr_arr.shape}")

    # --- diagnostics ---
    print(f"\n=== SHAPES AND DTYPES ===")
    print(f"Window eeg_de:  {eeg_win_arr.shape}  dtype={eeg_win_arr.dtype}")
    print(f"Window periph:  {periph_win_arr.shape}  dtype={periph_win_arr.dtype}")
    print(f"Trial eeg_de:   {eeg_tr_arr.shape}  dtype={eeg_tr_arr.dtype}")
    print(f"Trial periph:   {periph_tr_arr.shape}  dtype={periph_tr_arr.dtype}")

    print(f"\n=== NaN/Inf counts ===")
    print(f"Window eeg_de  NaN={np.isnan(eeg_win_arr).sum()} Inf={np.isinf(eeg_win_arr).sum()}")
    print(f"Trial  eeg_de  NaN={np.isnan(eeg_tr_arr).sum()}  Inf={np.isinf(eeg_tr_arr).sum()}")
    print(f"Trial  periph  NaN={np.isnan(periph_tr_arr).sum()} Inf={np.isinf(periph_tr_arr).sum()}")

    print(f"\n=== per-band stats (trial-level eeg_de_trial) ===")
    for bi, (bname,_,_) in enumerate(BANDS):
        b_data = eeg_tr_arr[:,:,bi]
        print(f"  {bname:6s}: min={b_data.min():.4f}  max={b_data.max():.4f}  mean={b_data.mean():.4f}")

    fwin = os.path.getsize(OUT_WIN);  ftr = os.path.getsize(OUT_TRIAL)
    print(f"\n=== FILE SIZES ===")
    print(f"feats_window.npz: {fwin} bytes ({fwin/1e6:.2f} MB)")
    print(f"feats_trial.npz:  {ftr}  bytes ({ftr/1e6:.2f} MB)")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f} s")

    if _psutil:
        import psutil
        proc = psutil.Process(os.getpid())
        peak = proc.memory_info().rss / 1e9
        print(f"Current RSS: {peak:.2f} GB")

if __name__ == "__main__":
    main()
