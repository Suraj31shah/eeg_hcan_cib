# PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB v1.0.0)
## Comprehensive Dataset Analysis, Neurophysiology, and CIB-Net Adaptation Roadmap

---

## 1. Executive Summary

This document presents the complete architectural adaptation of the **Channel-Invariance Balancing Network (CIB-Net)** to the **PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB v1.0.0)** (Kaggle: `utsavsinghal2604/eeg-motorimagery-eegmmidb-v1-0-0`).

While previously evaluated on affective computing datasets (**DEAP** with 32 channels and **DREAMER** with 14 channels + ECG), this adaptation scales CIB-Net to a **high-density 64-channel Brain-Computer Interface (BCI)** benchmark across **109 human subjects**, focusing on **Motor Imagery (MI)** decoding (Left Fist vs. Right Fist, and 4-Class limbs).

---

## 2. Dataset Characteristics & Quantitative Breakdown

```
========================================================================================
                          EEGMMIDB v1.0.0 AT A GLANCE
========================================================================================
  Source Institution:     Wadsworth Center (BCI2000 System, Schalk et al.)
  Total Subjects:         109 Human Subjects (S001 – S109)
  Total Electrodes:       64 EEG Channels (International 10-10 System)
  Sampling Rate:          160 Hz (Continuous EDF+ Format)
  Trials per Subject:     ~45 Motor Imagery Trials (Left vs Right Fist across Runs 4, 8, 12)
  Total Dataset Volume:   ~4,900 High-Density Motor Imagery Trials across 109 Subjects
  Target Tasks:           2-Class (Left vs Right Fist), 3-Class (Rest/Left/Right), 4-Class
  Storage Footprint:      1.87 GB (Compressed) / ~3.4 GB (Extracted EDF files)
  Cached Location:        E:\kagglehub_cache\datasets\utsavsinghal2604\...\1.0.0\
========================================================================================
```

### 2.1 Experimental Protocol & Runs Breakdown
Each subject performed **14 experimental runs**:
1. **Run 01:** Baseline, eyes open (1 minute continuous recording).
2. **Run 02:** Baseline, eyes closed (1 minute continuous recording).
3. **Runs 03, 07, 11 (Task 1 - Motor Execution):** Target appears on left or right of screen; subject opens and closes corresponding fist until target disappears.
4. **Runs 04, 08, 12 (Task 2 - Motor Imagery):** Target appears on left or right of screen; subject **imagines** opening and closing corresponding fist until target disappears.
5. **Runs 05, 09, 13 (Task 3 - Motor Execution):** Target appears on top or bottom of screen; subject opens and closes both fists (top) or both feet (bottom).
6. **Runs 06, 10, 14 (Task 4 - Motor Imagery):** Target appears on top or bottom of screen; subject **imagines** opening and closing both fists (top) or both feet (bottom).

### 2.2 Event Codes & Timings
* `T0`: Rest period between task cues (~4.2 seconds).
* `T1`: Left fist (Runs 3, 4, 7, 8, 11, 12) or Both fists (Runs 5, 6, 9, 10, 13, 14) (~4.1 seconds).
* `T2`: Right fist (Runs 3, 4, 7, 8, 11, 12) or Both feet (Runs 5, 6, 9, 10, 13, 14) (~4.1 seconds).

---

## 3. Neurophysiological Foundations: Sensorimotor Rhythms (ERD / ERS)

In human motor execution and motor imagery, the **primary motor cortex (M1)** and **somatosensory cortex** exhibit distinct frequency desynchronization patterns:

$$\text{Contralateral Movement} \implies \mu \text{-band (8–12 Hz)} \ \& \ \beta \text{-band (13–30 Hz) Event-Related Desynchronization (ERD)}$$

* **Imagining Left Hand Movement:** Triggers $\mu$ and $\beta$ power suppression in the **Right Hemispheric Sensorimotor Cortex** (electrodes **C4, FC4, CP4**).
* **Imagining Right Hand Movement:** Triggers $\mu$ and $\beta$ power suppression in the **Left Hemispheric Sensorimotor Cortex** (electrodes **C3, FC3, CP3**).
* **Rest:** Synchronization of the $\mu$ rhythm across Central electrodes (**Cz**).

```
                      ANATOMICAL CHANNEL MAPPING (64 ELECTRODES)
                                      FPz
                              FP1             FP2
                          AF7     AF3     AF4     AF8
                                      AFz
                    F7    F5    F3    F1  Fz  F2    F4    F6    F8
               FT7     FC5   FC3   FC1   FCz   FC2   FC4   FC6     FT8
               T7      C5    C3    C1    Cz    C2    C4    C6      T8
               TP7     CP5   CP3   CP1   CPz   CP2   CP4   CP6     TP8
                    P7    P5    P3    P1  Pz  P2    P4    P6    P8
                          PO7     PO3     PO4     PO8
                                      POz
                              O1      Oz      O2
                                      Iz
```

---

## 4. Adapting the CIB-Net Grand Architecture to 64-Channel Motor Imagery

```
                    64-CHANNEL EEG INPUT [B, 64, 640] (4s @ 160 Hz)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │                      STAGE A                        │
             │           Per-Channel 1D-CNN Encoder                │
             │         Shared Temporal Weights Across 64 Ch        │
             │       Kernels: 25 (Slow Mu), 15 (Beta), 7 (Gamma)   │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                [B, 64, 64] Tokens
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │                      STAGE C                        │
             │             5-Lobe Cortical Tokenizer               │
             │    Frontal (17) │ Central (21) │ Temporal (8)       │
             │    Parietal (14) │ Occipital (4)                    │
             │      Intra-Region Attention + Structured Drop       │
             └──────────┬───────────────────────────────┬──────────┘
                        │                               │
             Region Tokens: [B, 5, 64]                  │
                        │                               ▼
                        ├─────────────────┐   ┌───────────────────┐
                        │                 │   │      STAGE D      │
                        ▼                 ▼   │ Graph Consistency │
             ┌─────────────────────┐          │ PLV Synchrony     │
             │       STAGE B       │          └─────────┬─────────┘
             │  Subject Invariance │                    │
             │  EMA Cross-Loss Var │                    ▼
             │   g_r = 5*Softmax   │ ──► GATED SPECTRAL DECOUPLING
             └──────────┬──────────┘     L_SD = sum (lambda_0/g_r)*||z_r||^2
                        │
                        ▼
             ┌─────────────────────────────────────────────────────┐
             │           INTER-REGION TRANSFORMER ENCODER          │
             │      Multi-Head Self-Attention Across 5 Lobes       │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                                 CLASSIFIER HEAD
                        Left Fist vs. Right Fist Decision
```

### 4.1 Stage A: 64-Channel Independent 1D-CNN
* **Gradient Starvation Defense:** Conventional architectures (e.g., standard EEGNet) squash all 64 channels in a single depthwise convolution ($64 \times 1$). Pezeshki et al. (NeurIPS 2021) proved this starves slower-learning channels and causes single-electrode shortcut learning.
* **CIB-Net Solution:** Treats each of the 64 electrodes independently with shared 1D temporal kernels, learning pure temporal dynamics (Mu/Beta filters) before spatial interaction.

### 4.2 Stage B: Invariance Gating
* **Invariance Formulation:** Computes cross-subject loss variance $s_r$ across human subjects.
* **Physiological Goal in Motor Imagery:** The **Sensorimotor Cortex (C3/Cz/C4)** contains the true task-related neurophysiological signal. Non-motor regions (e.g. Occipital eye blinks or Temporal muscle tension) vary erratically across subjects and receive low invariance scores ($g_r < 1.0$), triggering heavy Spectral Decoupling penalties.

### 4.3 Stage C: 5-Lobe Cortical Tokenization
1. **Frontal (17 Channels):** Executive planning & attention (`FP1, FPZ, FP2, AF7, AF3, AFZ, AF4, AF8, F7, F5, F3, F1, FZ, F2, F4, F6, F8`).
2. **Central / Sensorimotor (21 Channels - Core Motor Area):** Mu/Beta desynchronization (`FC5, FC3, FC1, FCZ, FC2, FC4, FC6, C5, C3, C1, CZ, C2, C4, C6, CP5, CP3, CP1, CPZ, CP2, CP4, CP6`).
3. **Temporal (8 Channels):** Auditory/language (`FT7, FT8, T7, T8, T9, T10, TP7, TP8`).
4. **Parietal (14 Channels):** Spatial orientation (`P7, P5, P3, P1, PZ, P2, P4, P6, P8, PO7, PO3, POZ, PO4, PO8`).
5. **Occipital (4 Channels):** Visual feedback (`O1, OZ, O2, IZ`).

---

## 5. Repository File Structure & Deliverables

```
d:\EEG_5Stages\eeg_hcan_cib\
├── EEGMMIDB_Dataset_Exploration_and_Roadmap.md   <-- (This Document)
└── CIB-Net-EEGMMIDB/
    ├── config/
    │   └── config.yaml                           <-- Full parameters, regions & paths
    ├── data/
    │   └── eegmmidb_loader.py                    <-- MNE-powered EDF parser & PyTorch Dataset
    ├── models/
    │   ├── stage_a_encoder.py                    <-- 64-Channel 1D-CNN
    │   ├── stage_b_invariance.py                 <-- Cross-Subject Loss EMA Tracker
    │   ├── stage_c_fusion.py                     <-- 5-Lobe Cortical Tokenizer & Transformer
    │   └── cib_net_motor.py                      <-- Unified End-to-End Model
    ├── training/
    │   ├── losses.py                             <-- Gated Spectral Decoupling Loss
    │   └── loso_evaluator.py                     <-- Trainer, checkpointing & evaluation loop
    ├── evaluation/
    │   ├── metrics.py                            <-- Accuracy, BalAcc, AUROC, F1, MCC
    │   └── plots.py                              <-- Curves, Confusion Matrix, Gates, t-SNE
    ├── notebooks/
    │   └── cib_net_eegmmidb_pipeline.ipynb       <-- Interactive End-to-End Jupyter Pipeline
    └── results/
        ├── checkpoints/                          <-- best_checkpoint.pth
        └── plots/                                <-- training_curves, confusion_matrix, etc.
```

---

## 6. Execution Instructions

### A. Launch Interactive Jupyter Notebook
Open and run [`cib_net_eegmmidb_pipeline.ipynb`](file:///d:/EEG_5Stages/eeg_hcan_cib/CIB-Net-EEGMMIDB/notebooks/cib_net_eegmmidb_pipeline.ipynb):
1. **Cell 1–3:** Environment and YAML configuration load.
2. **Cell 4–6:** Data loading (MNE bandpass filter 4–40 Hz, Z-score standardization) and multi-channel plot of C3/Cz/C4.
3. **Cell 7:** CIB-Net model forward pass verification.
4. **Cell 8:** Training execution with real-time loss tracking and checkpoint saving.
5. **Cell 9–10:** Display of all 4 generated diagnostic plots.

### B. Command-Line Training Execution
To train and evaluate across subjects from PowerShell:
```powershell
& "D:\EEG_5Stages\venv\Scripts\python.exe" -c "
import sys; sys.path.append('CIB-Net-EEGMMIDB')
import yaml
from data.eegmmidb_loader import EEGMMIDBDataset, get_dataloaders
from training.loso_evaluator import Trainer

with open('CIB-Net-EEGMMIDB/config/config.yaml') as f:
    config = yaml.safe_load(f)

dataset = EEGMMIDBDataset(
    data_root=config['dataset']['raw_data_path'],
    subject_list=[f'S{i:03d}' for i in range(1, 11)],
    task='imagery_left_right',
    cache_path='E:/kagglehub_cache/eegmmidb_10subjs.pt'
)

train_idx = [i for i in range(len(dataset)) if dataset.subject_ids[i].item() <= 8]
val_idx = [i for i in range(len(dataset)) if dataset.subject_ids[i].item() > 8]

train_loader, val_loader = get_dataloaders(dataset, train_idx, val_idx, batch_size=config['training']['batch_size'])
trainer = Trainer(config)
trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'])
"
```

---

## 7. Next Milestones & Scientific Roadmap

1. **Full 109-Subject Benchmark:** Generate a complete preprocessed `.pt` cache for all 109 subjects on `E:\kagglehub_cache\` for fast iterative training.
2. **Physiological Invariance Validation:** Confirm that Stage B's $g_r$ gate systematically converges to $>1.0$ for the **Central / Sensorimotor lobe** while damping **Occipital / Temporal lobes**.
3. **4-Class Motor Imagery Scaling:** Extend target extraction to 4 limbs (Left Fist, Right Fist, Both Fists, Both Feet) by combining Runs 4, 8, 12 and Runs 6, 10, 14.
4. **Subject-Adversarial Domain Adaptation (Stage E):** Attach a Gradient Reversal Layer (GRL) subject discriminator to align representations across all 109 subjects.

