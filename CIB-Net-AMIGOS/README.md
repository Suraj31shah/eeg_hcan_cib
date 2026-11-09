# CIB-Net / HCAN-CIB on AMIGOS (Kaggle-Ready Replication Suite)

Standalone benchmark suite replicating the full **DEAP & DREAMER** experimental protocol on the **AMIGOS** multimodal emotion recognition dataset (`Data_Preprocessed_P01.mat` ... `Data_Preprocessed_P40.mat`).

---

## 1. AMIGOS Verified Data Schema & Adaptations

Every loader, region mapping, and diagnostic module has been adapted specifically to AMIGOS's verified preprocessing schema:

| Component | DEAP / DREAMER | AMIGOS (`CIB-Net-AMIGOS`) |
|---|---|---|
| **EEG Montage & Channels** | 32 ch (DEAP) / 14 ch Emotiv EPOC (DREAMER) | **14 ch Emotiv EPOC**: `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4` (cols `0..13`) |
| **Cortical Regions (`Stage C`)** | 5 regions (DEAP) / 4 regions (DREAMER) | **4 regions**: Frontal `[0,1,2,3,10,11,12,13]`, Temporal `[4,9]`, Parietal `[5,8]`, Occipital `[6,7]` |
| **Peripheral Modalities** | 8 ch (DEAP) / 2 ECG ch (DREAMER) | **3 channels** (`ECG_R, ECG_L, GSR`, cols `14..16`) -> **18 stats** (6 stats x 3 ch) |
| **Label Column Order** | `[valence, arousal]` (`0, 1`) | **`labels_selfassessment`**: **`0: arousal`, `1: valence`, `2: dominance`** *(arousal comes first in AMIGOS!)* |
| **Trial Types & Baseline** | Fixed-length trials | 16 short clips (`51-150s`, default `video_subset: short`) + 4 long clips; **5s** fixation baseline removed (`baseline_sec: 5`), NaN interpolation prior to `4-45 Hz` bandpass |
| **Subject ID Capacity** | `1..32` (DEAP) / `1..23` (DREAMER) | Supports non-contiguous IDs up to **`P40`** (`max_subjects: 64` + dynamic `InvarianceGate` growth) |
| **Evaluation Protocol** | Validation-subject LOSO (`make_fold`, zero leakage) | Identical validation-subject LOSO (`n_val=4` when >=12 subjects, `resolve_n_val` fallback for smaller subsets) |

> **Zero-Leakage vs. Transductive Disclosure:** All `normalize: "none"` models (shallow, deep baselines, HCAN-CIB base, `v_smote`, `gamo`, `deepsmote`, `class_weight`, `logit_adj`) run strictly without access to held-out test statistics. Any `normalize: "zscore"` variant is label-free per-subject normalization over a subject's own recording and is explicitly marked **transductive**.

---

## 2. Replicated Experiment Suite

Running the full pipeline evaluates every model family under identical LOSO splits and threshold tuning (`balanced_acc` on pooled validation subjects):

1. **Feature Extraction (`extract_features_amigos.py`):**
   - 4-band EEG Differential Entropy (`theta 4-8 Hz`, `alpha 8-12 Hz`, `beta 12-30 Hz`, `gamma 30-45 Hz` x 14 ch = **56-D**) + **18-D** Peripheral stats, stored raw (`normalize='none'`) at window and trial level.
2. **Domain-Shift Diagnostics (`domain_shift_diagnostics.py`):**
   - Runs `D0` (label prior skew), `D1` (blocked & shuffled within-subject CV vs LOSO), `D2` (subject-ID decodability), `D3` (variance decomposition `eta^2_subject` vs `eta^2_label|subject`), and `D4` (paired normalization shift) across `eeg`, `periph`, and `concat` feature blocks + `diagnostics.png`.
3. **Shallow Baselines (`shallow_baselines.py`):**
   - `T0` (prior baseline), `LR_eeg`, `SVC_eeg`, `LR_periph`, `SVC_periph`, `LR_concat`, `SVC_concat`, `LR_late` under both `norm='none'` and `norm='zscore'`.
4. **Deep EEG-Only Baselines (`experiments/run_baselines.py`):**
   - `PlainCNN` (`none`), `EEGNet` (`none`), `ACRNN` (`none`), plus `EEGNet` (`zscore`) [and optionally `PlainCNN`/`ACRNN` (`zscore`) via `--also_zscore_all`].
5. **Multimodal `HCAN-CIB` Suite (`experiments/run_hcan_cib.py`):**
   - `HCAN-CIB` base (`none/none`), `zscore` (`none/zscore`, transductive), `V-SMOTE` (`v_smote/none`), `GAMO` (`gamo/none`), `DeepSMOTE` (`deepsmote/none`), plus optional ablations (`--include_ablations`: `class_weight`, `logit_adj`, `no_gating`).
6. **Paired Statistical Testing & Plots (`compare_models.py`, `evaluation/plots.py`, `experiments/generate_summary_report.py`):**
   - Generates `AMIGOS_Comprehensive_Results_{target}.md` with bootstrapped 95% CIs, paired Wilcoxon + Holm-Bonferroni comparisons against `hcan_none_none_{target}`, pooled ROC/PR curves, t-SNE class/subject embeddings, and 4-region Stage B Invariance Gate heatmaps.

---

## 3. Running on Kaggle (Recommended Workflow)

### Step A — Attach Data & Code in a Kaggle Notebook
1. **AMIGOS Dataset (Input Data):**
   - Recommended (full 40 subjects `P01..P40`): **`nsut123/amigos`**
   - Lightweight alternative (15 subjects): `nsut123/amigos-dataset`
   - No path edits needed: `find_amigos_files()` recursively discovers `Data_Preprocessed_P*.mat` under `/kaggle/input`.
2. **Code Upload:**
   - Zip this `CIB-Net-AMIGOS/` folder and upload it to Kaggle as a private Dataset (e.g., `cib-net-amigos-code`), OR upload directly into `/kaggle/working/CIB-Net-AMIGOS/`.
3. **Notebook & Accelerator:**
   - Open [`notebooks/AMIGOS_Full_Benchmarks_Kaggle.ipynb`](notebooks/AMIGOS_Full_Benchmarks_Kaggle.ipynb) in Kaggle, select **GPU T4 x2** or **GPU P100** accelerator, and click **Run All**.

---

## 4. CLI Usage & Resumable Phased Execution

On 40 AMIGOS subjects (~300 windows/subject), each 40-subject LOSO deep run takes ~25-45 minutes on a Kaggle P100/T4 (12-hour session limit is plenty for one target's full suite; use `--resume` across sessions if running both `valence` and `arousal` with ablations).

```bash
# 1. Inspect / verify loader & extracted features
python run_all_amigos_kaggle.py --phase stage1 --target valence --resume

# 2. Run deep EEG-only baselines (PlainCNN, EEGNet, ACRNN + EEGNet zscore)
python run_all_amigos_kaggle.py --phase baselines --target valence --resume

# 3. Run HCAN-CIB variants (base, zscore, v_smote, gamo, deepsmote) + optional ablations
python run_all_amigos_kaggle.py --phase hcan --target valence --include_ablations --resume

# 4. Regenerate statistical comparison + summary markdown
python run_all_amigos_kaggle.py --phase summary --target valence

# Or run everything for arousal in one command:
python run_all_amigos_kaggle.py --phase all --target arousal --include_ablations --resume
```

### Directory Structure

```text
CIB-Net-AMIGOS/
├── config/amigos_config.yaml           # Verified AMIGOS hyperparameters & paths
├── data/
│   ├── amigos_loader.py                # WindowedAMIGOSDataset (14 EEG + 3 Periph, short/all, v7.3 HDF5 support)
│   └── plv_connectivity.py             # Stage D PLV spectral graph centrality
├── models/
│   ├── stage_a_encoder.py              # Per-channel temporal/spectral CNN
│   ├── stage_b_invariance.py           # Dynamic-capacity EMA InvarianceGate (Frontal/Temporal/Parietal/Occipital)
│   ├── stage_c_fusion.py               # AMIGOS 14-ch 4-region fusion + region dropout
│   ├── stage_d_graph_loss.py           # Graph topology consistency regularization
│   ├── peripheral_encoder.py           # 18-D ECG/GSR MLP encoder
│   ├── hcan_fusion.py                  # Bidirectional EEG <-> Peripheral cross-attention
│   ├── hcan_cib_net.py                 # Full HCAN-CIB network + latent oversampling hooks
│   └── baselines/                      # PlainCNN, EEGNet (dynamic 128Hz*4s), ACRNN
├── training/
│   ├── imbalance.py                    # class_weight, LogitAdjustedLoss, DeepSMOTE, GAMO
│   ├── smote.py                        # Latent SMOTE & V-SMOTE
│   ├── losses.py                       # Spectral Decoupling + hierarchical CIB losses
│   ├── subject_sampler.py              # SubjectBalancedSampler
│   └── loso_evaluator.py               # Zero-leakage validation-subject LOSO loop
├── evaluation/
│   ├── metrics.py, diagnostics.py, plots.py
├── experiments/
│   ├── run_baselines.py, run_hcan_cib.py, generate_summary_report.py
├── notebooks/
│   └── AMIGOS_Full_Benchmarks_Kaggle.ipynb
├── eeg_common.py, extract_features_amigos.py, domain_shift_diagnostics.py
├── shallow_baselines.py, compare_models.py, run_all_amigos_kaggle.py
└── requirements.txt
```

