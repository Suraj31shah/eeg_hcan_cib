# Step 3 Clean Labels

**Python used:** `D:\EEG_5Stages\venv\Scripts\python.exe`

---

## 1. Headline Numbers

- **D: free space at audit start:** 1489.5 MB (1.455 GB). Above 300 MB threshold; execution continued.
- **Task 1 SHA256 match:** NPY SHA256 = `3301F3AA5503A0891A3E57AB4A6459C7BAB8BD007E58A3DB29FFF56ED6B54F2D` — source and both copies identical. CSV SHA256 = `85A890D850CDB11F640B21635E21E3B26B3FD9CB52FC92CEC54A03434BF93741` — source and both copies identical.
- **Task 2 mismatch counts:** 439 valence mismatches, 439 arousal mismatches, 439 both, 0 either-only. Max |dat+xls−9| = 0.000000 (below 0.011). All 439 mismatches fall exclusively in HVHA quadrant (xls_val ≥ 5 AND xls_aro ≥ 5).
- **Task 3 tests:** 6 PASS, 0 FAIL.
- **Task 4 clean valence fixed5 AUROC:** mean=0.8521, SD=0.1020, SE=0.0180, n=32, 95% CI [0.8166, 0.8839]. Clean valence Spearman (continuous prior): mean=0.7024.
- **Task 4 altered valence fixed5 AUROC:** mean=0.7320, SD=0.1283, SE=0.0227, n=32.
- **Null SD formula sqrt((n1+n2+1)/(12·n1·n2)) at n1=n2=20:** 0.0924.

---

## 2. Task 1 — Install Clean Labels in v2

### 1a — Free space and directory creation

**Command:** `Get-PSDrive D | Select-Object Name, @{N='FreeGB'; ...}`
```
D: Free space: 1.455 GB (1489.6 MB)
OK: above 300 MB
```

**Command:** `New-Item -ItemType Directory -Force -Path "D:\EEG_5Stages\eeg_hcan_cib\v2\data"` etc.
```
Directories created:
  D:\EEG_5Stages\eeg_hcan_cib\v2\data    → True
  D:\EEG_5Stages\eeg_hcan_cib\v2\scripts → True
  E:\audit_scratch\backup                 → True
```

### 1b — Copy and SHA256 verification

**Command:** `Copy-Item` then `Get-FileHash ... -Algorithm SHA256` for source and each copy.

```
NPY E:\audit_scratch\deap_labels_from_xls.npy              SHA256=3301F3AA5503A0891A3E57AB4A6459C7BAB8BD007E58A3DB29FFF56ED6B54F2D
NPY D:\EEG_5Stages\eeg_hcan_cib\v2\data\deap_labels_from_xls.npy  SHA256=3301F3AA5503A0891A3E57AB4A6459C7BAB8BD007E58A3DB29FFF56ED6B54F2D
NPY E:\audit_scratch\backup\deap_labels_from_xls.npy        SHA256=3301F3AA5503A0891A3E57AB4A6459C7BAB8BD007E58A3DB29FFF56ED6B54F2D

CSV E:\audit_scratch\deap_labels_from_xls.csv               SHA256=85A890D850CDB11F640B21635E21E3B26B3FD9CB52FC92CEC54A03434BF93741
CSV D:\EEG_5Stages\eeg_hcan_cib\v2\data\deap_labels_from_xls.csv  SHA256=85A890D850CDB11F640B21635E21E3B26B3FD9CB52FC92CEC54A03434BF93741
CSV E:\audit_scratch\backup\deap_labels_from_xls.csv         SHA256=85A890D850CDB11F640B21635E21E3B26B3FD9CB52FC92CEC54A03434BF93741
```

All three copies of each file have identical SHA256. File sizes: npy = 41088 bytes, csv = 33198 bytes.

### 1c — Build labels_from_dat.npy

**Script:** `E:\audit_scratch\t1c_dat.py` — loads only `d["labels"]` from each .dat file; `d["data"]` is not accessed.

**Command:** `& "D:\EEG_5Stages\venv\Scripts\python.exe" "E:\audit_scratch\t1c_dat.py"`
```
Shape: (32, 40, 4)
Dtype: float64
Saved: E:\audit_scratch\labels_from_dat.npy
File size bytes: 41088
```

---

## 3. Task 2 — Mismatch Characterization

**Script:** `E:\audit_scratch\t2.py`
**Command:** `& "D:\EEG_5Stages\venv\Scripts\python.exe" "E:\audit_scratch\t2.py"`

### 2a — Counts
```
Valence mismatches: 439
Arousal mismatches: 439
Both:               439
Either:             439
```
Every mismatched row is mismatched in both valence and arousal simultaneously; zero rows mismatch in only one dimension.

### 2b — max |dat + xls − 9.0| over mismatched rows
```
Valence: max |dat_val + xls_val - 9.0| = 0.000000   → below 0.011: True
Arousal: max |dat_aro + xls_aro - 9.0| = 0.000000   → below 0.011: True
```
For every mismatched row, dat + xls = exactly 9.0 for both valence and arousal.

### 2c — Quadrant table (threshold = 5)
```
HVHA (xls_val≥5 AND xls_aro≥5): n=458  val_mm=439  aro_mm=439
HVLA (xls_val≥5 AND xls_aro<5): n=266  val_mm=0    aro_mm=0
LVHA (xls_val<5 AND xls_aro≥5): n=296  val_mm=0    aro_mm=0
LVLA (xls_val<5 AND xls_aro<5): n=260  val_mm=0    aro_mm=0
```
All 439 mismatches are in HVHA. Zero mismatches in any other quadrant.

### 2d — HVHA fraction mismatched
```
HVHA total: 458   mismatched: 439   fraction: 0.9585
Mismatched rows:    mean xls_val=7.1079   mean xls_aro=6.5798
Not-mismatched rows: mean xls_val=6.1432  mean xls_aro=5.2968
```

### 2e — Mismatches by video group
```
video_id 01-10: val_mm=204  aro_mm=204
video_id 11-20: val_mm=143  aro_mm=143
video_id 21-30: val_mm=34   aro_mm=34
video_id 31-40: val_mm=58   aro_mm=58
```

---

## 4. Task 3 — Clean Label Module and Tests

### Module: `D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\labels_clean.py`

Public API (full source in the file):
- `load_ratings() -> np.ndarray`  shape (32, 40, 4) float64. Reads from `v2/data/deap_labels_from_xls.npy` via path relative to `__file__`.
- `get_binary_labels(target, mode) -> (labels, mask)` both shape (32, 40). `target` ∈ {'valence','arousal'}. `mode` ∈ {'fixed5', 'median_drop_ties'}.
- `get_ratings(target) -> np.ndarray` shape (32, 40).
- Docstring states: "The .dat files' valence and arousal columns (columns 0 and 1) must NOT be used: 439 of 1280 (subject, video) pairs have altered values in those columns."

### Tests: `D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\test_labels_clean.py`

**Command:** `& "D:\EEG_5Stages\venv\Scripts\python.exe" "D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\test_labels_clean.py"`

```
PASS: shape of load_ratings is (32,40,4)  [got (32, 40, 4)]
PASS: no NaN in load_ratings  [NaN count=0]
PASS: fixed5 positive fraction equals CSV fraction  [api=0.553125  csv=0.553125]
PASS: median_drop_ties class balance within ties+1 per subject  [all OK]
PASS: dominance in load_ratings equals .dat column 2 exactly
PASS: liking in load_ratings equals .dat column 3 exactly

Summary: 6 PASS  0 FAIL  (total 6)
```

---

## 5. Task 4 — No-EEG Baselines on Clean vs Altered Labels

**Script:** `D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\video_prior_baseline.py`
**Command:** `& "D:\EEG_5Stages\venv\Scripts\python.exe" "...video_prior_baseline.py" 2>&1 | Out-File "E:\audit_scratch\t4_baseline_out.txt"`
**Results saved:** `E:\audit_scratch\video_prior_results.json` (11691 bytes)

### AUROC and Accuracy Table (8 combos)

```
Source   Target   Mode                 AUROC_mean  AUROC_sd  AUROC_se  AUROC_n  CI95_lo  CI95_hi  NaN_s  Acc_mean  Acc_sd   Acc_se   Trivial  Pooled_AUC  Null_SD    n1    n2
clean    valence  fixed5                   0.8521    0.1020    0.0180       32   0.8166   0.8839      0    0.7750   0.0946   0.0167   0.5813      0.8334   0.0929    22    18
clean    valence  median_drop_ties         0.8606    0.1051    0.0186       32   0.8237   0.8943      0    0.7919   0.0976   0.0173   0.5018      0.8538   0.0924    20    20
clean    arousal  fixed5                   0.7041    0.1477    0.0261       32   0.6538   0.7539      0    0.6555   0.1094   0.0193   0.6430      0.6635   0.0935    23    17
clean    arousal  median_drop_ties         0.6925    0.1647    0.0291       32   0.6362   0.7476      0    0.6734   0.1326   0.0234   0.5018      0.6814   0.0924    20    20
altered  valence  fixed5                   0.7320    0.1283    0.0227       32   0.6879   0.7747      0    0.6242   0.0844   0.0149   0.7914      0.7061   0.1155     8    32
altered  valence  median_drop_ties         0.6515    0.1002    0.0177       32   0.6169   0.6852      0    0.5980   0.0876   0.0155   0.5004      0.6426   0.0924    20    20
altered  arousal  fixed5                   0.8361    0.1363    0.0245       31   0.7871   0.8801      1    0.6672   0.1158   0.0205   0.7672      0.8045   0.1107     9    31
altered  arousal  median_drop_ties         0.6604    0.1466    0.0259       32   0.6081   0.7096      0    0.6197   0.1264   0.0223   0.5004      0.6494   0.0924    20    20
```

NaN subjects: `altered arousal fixed5` has 1 NaN subject (single class among masked-in trials).

### Spearman Correlation (continuous prior vs raw ratings)

```
Source   Target     Mean      SD       SE     N
clean    valence   0.7024   0.1718   0.0304   32
clean    arousal   0.3740   0.2810   0.0497   32
altered  valence   0.2530   0.2097   0.0371   32
altered  arousal   0.3622   0.2832   0.0501   32
```

### Null SD Comparison

Formula: `sqrt((n1+n2+1) / (12 * n1 * n2))`

At n1=n2=20: null SD = **0.0924**.

```
Source   Target   Mode                  null_SD (n1,n2)      observed_SD
clean    valence  fixed5                0.0929 (22,18)       0.1020
clean    valence  median_drop_ties      0.0924 (20,20)       0.1051
clean    arousal  fixed5                0.0935 (23,17)       0.1477
clean    arousal  median_drop_ties      0.0924 (20,20)       0.1647
altered  valence  fixed5                0.1155  (8,32)       0.1283
altered  valence  median_drop_ties      0.0924 (20,20)       0.1002
altered  arousal  fixed5                0.1107  (9,31)       0.1363
altered  arousal  median_drop_ties      0.0924 (20,20)       0.1466
```

### Per-video Mean Ratings Table (clean labels, 32 subjects)

```
video_id  mean_valence  mean_arousal
       1        6.7691        5.7075
       2        6.4078        6.3231
       3        7.5325        6.1853
       4        6.7831        5.9000
       5        6.1653        6.1897
       6        6.6653        5.5856
       7        6.0572        5.7447
       8        7.0569        5.5337
       9        7.2263        5.8612
      10        4.8494        5.6322
      11        7.4066        5.3422
      12        6.3116        4.0409
      13        6.6778        4.4688
      14        7.3309        5.3194
      15        6.1509        3.9253
      16        5.0375        3.7703
      17        5.8450        4.1613
      18        7.3144        4.9128
      19        7.1513        5.6803
      20        6.5963        5.3050
      21        3.8147        4.4244
      22        4.7244        3.7050
      23        3.1116        4.2947
      24        4.0497        4.9581
      25        4.4984        4.5012
      26        4.9941        3.4466
      27        5.4772        4.4559
      28        4.4113        3.7300
      29        3.7538        4.3822
      30        3.2259        5.3400
      31        3.8169        5.5487
      32        3.8978        6.6294
      33        3.9694        5.2725
      34        4.0075        5.8659
      35        3.1006        5.8163
      36        4.0091        5.8750
      37        2.9056        5.7622
      38        2.6469        5.8641
      39        3.5500        5.6066
      40        4.8728        5.2003
```

---

## 6. Task 5 — Source Code Verbatim (read-only)

**Command:** `Select-String -Path <file> -Pattern "add_argument"` and `"^def |^    def "` for argparse and function signatures; `Select-String -Pattern "labels|y\[|target|binarize|median|norm|zscore|subject_normalize|video|trial"` for keyword lines.

### shallow_baselines.py

**File:** `D:\EEG_5Stages\eeg_hcan_cib\shallow_baselines.py`

**add_argument lines:**
```
141:     ap.add_argument("--data")
142:     ap.add_argument("--out", default="results/shallow")
143:     ap.add_argument("--norm", default="none", choices=["none", "center", "zscore"])
144:     ap.add_argument("--seed", type=int, default=0)
145:     ap.add_argument("--n_val", type=int, default=4)
146:     ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
147:     ap.add_argument("--variants", nargs="*",
150:     ap.add_argument("--demo", action="store_true")
```

**def lines:**
```
37:  def _sigmoid(z):
41:  def fit_select(Xfit, yfit, Xval, yval, Xte, kind, class_weight, seed):
66:  def _blocks_for(variant, blocks):
76:  def run_loso(blocks, y, subj, variant, norm="none", seed=0, n_val=4, class_weight="balanced",
123: def summarize(name, folds):
131:     def mean(key, thr=False):
139: def main():
```

**Keyword lines:**
```
15:   python shallow_baselines.py --data feats.npz --norm zscore --out results/shallow_zscore
29: from eeg_common import (load_npz, make_demo, subject_normalize, make_fold, compute_metrics,
76: def run_loso(blocks, y, subj, variant, norm="none", seed=0, n_val=4, class_weight="balanced",
80:     nblocks = {k: subject_normalize(v, subj, norm) for k, v in blocks.items()}
86:         yfit, yval, yte = y[fit], y[val], y[te]
118:         print(f"  {variant:11s} norm={norm:6s} AUROC {np.nanmean(a):.4f} "
143:     ap.add_argument("--norm", default="none", choices=["none", "center", "zscore"])
164:     print(f"[data] {len(y)} trials, {len(np.unique(subj))} subjects, "
170:         folds, preds = run_loso(blocks, y, subj, v, args.norm, args.seed, args.n_val,
172:         tag = f"{v}_{args.norm}"
175:         np.savez(f"{args.out}/trial_predictions_{tag}.npz",
183:     open(f"{args.out}/summary_{args.norm}.md", "w").write(table + "\n")
```

### extract_features.py

**File:** `D:\EEG_5Stages\eeg_hcan_cib\extract_features.py`

**def lines:**
```
14: def butter_bandpass(lowcut, highcut, fs, order=3):
21: def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
26: def compute_de(x):
30: def extract_de_features(windows, fs=128):
50: def main():
```

**Keyword lines:**
```
72:     dataset = WindowedDEAPDataset(data_path, binarize='median', window_sec=4.0, overlap_sec=2.0)
78:     y = np.array(dataset.labels) # (N,)
80:     trial = np.array(dataset.trial_ids) # (N,)
87:     np.savez(out_file, eeg=eeg_de, periph=periph, y=y, subject=subject, trial=trial)
```

No `add_argument` lines found in extract_features.py.

### eeg_common.py

**File:** `D:\EEG_5Stages\eeg_hcan_cib\eeg_common.py`

**def lines:**
```
22:  def load_npz(path, aggregate=True):
54:  def make_demo(path, n_subj=20, n_trials=40, n_feat=30, signal=0.0, subj_shift=1.5, seed=0):
70:  def subject_normalize(X, subj, mode="none"):
89:  def make_fold(subj, test_subject, fold, seed=0, n_val=4):
101: def safe_auroc(y, p):
105: def tune_threshold(y, p):
117: def _ece(y, p, bins=10):
128: def compute_metrics(y, prob, pred):
137:     def prf(c):
167: def boot_ci(v, n=5000, seed=0):
174: def holm(pvals):
```

**Keyword lines:**
```
4:  DATA FORMAT (one .npz file, trial-level or window-level):
7:      y        (N,)         int     binary label 0/1 (already binarized: Task P or Task F)
9:      trial    (N,)         int     OPTIONAL trial id within subject. If rows are windows,
10:                                    rows sharing (subject, trial) are averaged into one trial.
32:     trial = np.asarray(z["trial"]).astype(int) if "trial" in z.files else None
34:     if aggregate and trial is not None:
35:         key = subj.astype(np.int64) * 1_000_000 + trial
37:         if len(uniq) < len(key):  # rows are windows -> average to trial level
44:             if not np.all(y[first][inv] == y):
45:                 raise ValueError("label differs inside a trial")
46:             y, subj, trial = y[first], subj[first], trial[first]
47:             print(f"[data] aggregated windows -> {len(uniq)} trials")
51:     return blocks, y, subj, trial
54: def make_demo(path, n_subj=20, n_trials=40, n_feat=30, signal=0.0, subj_shift=1.5, seed=0):
62:         lab = (rng.rand(n_trials) < 0.5).astype(int)
63:         x = rng.randn(n_trials, n_feat) * sc + off + signal * np.outer(lab - 0.5, w) * sc
64:         X.append(x); y.append(lab); s.append(np.full(n_trials, i + 1)); t.append(np.arange(n_trials))
66:              trial=np.concatenate(t))
69: # ------------------------------------------------------------------- normalization
70: def subject_normalize(X, subj, mode="none"):
71:     """Label-free per-subject normalization. Uses the held-out subject's own UNLABELED
81:         elif mode == "zscore":
124:             e += m.mean() * abs(y[m].mean() - p[m].mean())
```

No `add_argument` lines found in eeg_common.py.

**First 60 lines of eeg_common.py verbatim:**

**Command:** `Get-Content "D:\EEG_5Stages\eeg_hcan_cib\eeg_common.py" -TotalCount 60`

```
1:  """
2:  Shared utilities for the shallow baselines and domain-shift diagnostics.
3:  
4:  DATA FORMAT (one .npz file, trial-level or window-level):
5:      eeg      (N, F_eeg)   float   e.g. DE features, bands x channels flattened
6:      periph   (N, F_p)     float   OPTIONAL (peripheral / ECG features)
7:      y        (N,)         int     binary label 0/1 (already binarized: Task P or Task F)
8:      subject  (N,)         int     subject id
9:      trial    (N,)         int     OPTIONAL trial id within subject. If rows are windows,
10:                                    rows sharing (subject, trial) are averaged into one trial.
11: 
12: The fold split reproduces loso_evaluator_fixed.py exactly (same RandomState seed,
13: same np.unique ordering), so each held-out subject gets the SAME validation subjects
14: as the deep model. That makes paired comparisons valid.
15: """
16: import numpy as np
17: from sklearn.metrics import (roc_auc_score, average_precision_score,
18:                              matthews_corrcoef, brier_score_loss)
19: 
20: 
21: # ----------------------------------------------------------------------------- data
22: def load_npz(path, aggregate=True):
23:     z = np.load(path, allow_pickle=False)
24:     for k in ("y", "subject"):
25:         if k not in z.files:
26:             raise KeyError(f"'{k}' missing from {path}; found {z.files}")
27:     blocks = {k: np.asarray(z[k], dtype=np.float64) for k in ("eeg", "periph") if k in z.files}
28:     if not blocks:
29:         raise KeyError("need at least one of 'eeg' / 'periph'")
30:     y = np.asarray(z["y"]).astype(int)
31:     subj = np.asarray(z["subject"]).astype(int)
32:     trial = np.asarray(z["trial"]).astype(int) if "trial" in z.files else None
33: 
34:     if aggregate and trial is not None:
35:         key = subj.astype(np.int64) * 1_000_000 + trial
36:         uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
37:         if len(uniq) < len(key):  # rows are windows -> average to trial level
38:             cnt = np.bincount(inv).astype(float)
39:             for name, X in blocks.items():
40:                 out = np.zeros((len(uniq), X.shape[1]))
41:                 for j in range(X.shape[1]):
42:                     out[:, j] = np.bincount(inv, weights=X[:, j]) / cnt
43:                 blocks[name] = out
44:             if not np.all(y[first][inv] == y):
45:                 raise ValueError("label differs inside a trial")
46:             y, subj, trial = y[first], subj[first], trial[first]
47:             print(f"[data] aggregated windows -> {len(uniq)} trials")
48:     for name, X in blocks.items():
49:         if not np.isfinite(X).all():
50:             raise ValueError(f"non-finite values in '{name}'")
51:     return blocks, y, subj, trial
52: 
53: 
54: def make_demo(path, n_subj=20, n_trials=40, n_feat=30, signal=0.0, subj_shift=1.5, seed=0):
55:     """Synthetic data: subject-specific offsets/scales plus an optional weak label signal."""
56:     rng = np.random.RandomState(seed)
57:     X, y, s, t = [], [], [], []
58:     w = rng.randn(n_feat)
59:     for i in range(n_subj):
60:         off = rng.randn(n_feat) * subj_shift
```

---

## 7. UNVERIFIED Items

None. All tasks completed and verified by command output.

---

## 8. Appendix: Every Command Run, In Order, and Files Created

### Commands

```
 1. Get-PSDrive D | Select-Object Name, FreeGB, FreeMB
 2. New-Item -ItemType Directory -Force -Path "D:\EEG_5Stages\eeg_hcan_cib\v2\data"
 3. New-Item -ItemType Directory -Force -Path "D:\EEG_5Stages\eeg_hcan_cib\v2\scripts"
 4. New-Item -ItemType Directory -Force -Path "E:\audit_scratch\backup"
 5. Copy-Item E:\audit_scratch\deap_labels_from_xls.npy → v2\data\ and backup\
 6. Copy-Item E:\audit_scratch\deap_labels_from_xls.csv → v2\data\ and backup\
 7. Get-FileHash (source and copies) -Algorithm SHA256 for npy and csv
 8. [Write E:\audit_scratch\t1c_dat.py]
    & "D:\EEG_5Stages\venv\Scripts\python.exe" "E:\audit_scratch\t1c_dat.py"
 9. [Write E:\audit_scratch\t2.py]
    & "D:\EEG_5Stages\venv\Scripts\python.exe" "E:\audit_scratch\t2.py"
10. [Write D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\labels_clean.py]
11. [Write D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\test_labels_clean.py]
    & "D:\EEG_5Stages\venv\Scripts\python.exe" "...test_labels_clean.py"
12. [Write D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\video_prior_baseline.py]
    & "D:\EEG_5Stages\venv\Scripts\python.exe" "...video_prior_baseline.py" 2>&1 | Out-File "E:\audit_scratch\t4_baseline_out.txt"
    Get-Content "E:\audit_scratch\t4_baseline_out.txt"
13. Select-String -Pattern "add_argument" on shallow_baselines.py, extract_features.py, eeg_common.py
14. Select-String -Pattern "^def |^    def " on the same three files
15. Select-String -Pattern "labels|y\[|target|binarize|median|norm|zscore|subject_normalize|video|trial" on the same three files
16. Get-Content "D:\EEG_5Stages\eeg_hcan_cib\eeg_common.py" -TotalCount 60
17. Get-ChildItem "D:\EEG_5Stages\eeg_hcan_cib\v2" -Recurse -File | ForEach-Object { "$($_.Length) $($_.FullName)" }
18. [Verify all expected files exist]
    Get-Item <each file> | Select-Object Name, Length
19. [Write D:\EEG_5Stages\eeg_hcan_cib\v2\STEP3_REPORT.md]
```

### Files Created

| Path | Size (bytes) |
|------|-------------|
| `D:\EEG_5Stages\eeg_hcan_cib\v2\data\deap_labels_from_xls.npy` | 41088 |
| `D:\EEG_5Stages\eeg_hcan_cib\v2\data\deap_labels_from_xls.csv` | 33198 |
| `D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\labels_clean.py` | 2852 |
| `D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\test_labels_clean.py` | 3455 |
| `D:\EEG_5Stages\eeg_hcan_cib\v2\scripts\video_prior_baseline.py` | 11370 |
| `D:\EEG_5Stages\eeg_hcan_cib\v2\STEP3_REPORT.md` | (this file) |
| `E:\audit_scratch\labels_from_dat.npy` | 41088 |
| `E:\audit_scratch\backup\deap_labels_from_xls.npy` | 41088 |
| `E:\audit_scratch\backup\deap_labels_from_xls.csv` | 33198 |
| `E:\audit_scratch\video_prior_results.json` | 11691 |
| `E:\audit_scratch\t1c_dat.py` (scratch script) | — |
| `E:\audit_scratch\t2.py` (scratch script) | — |
| `E:\audit_scratch\t4_baseline_out.txt` (output capture) | — |
