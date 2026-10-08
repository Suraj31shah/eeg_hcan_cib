# CIB-Net Implementation Status
**Current Phase:** Stage A, B, and C fully implemented, optimized, and evaluated.

This document summarizes the exact components of the novel Channel-Invariance Balancing Network (CIB-Net) that have been successfully built, integrated, and mathematically verified on both the DEAP and DREAMER datasets.

---

## 1. Data Pipeline & Preprocessing
* **Multi-Dataset Support:** Custom PyTorch `Dataset` loaders created for both DEAP (`deap_loader.py`) and DREAMER (`dreamer_loader.py`).
* **Signal Windowing:** Raw EEG signals are dynamically sliced into overlapping time windows (e.g., 4-second windows with 2-second overlap).
* **Target Binarization:** Support for both Task F (fixed threshold) and Task P (median per-subject threshold) for Valence and Arousal.
* **Dataloader Optimization:** Eliminated CPU thrashing/freezing on shared HPC nodes by strictly enforcing `num_workers=0`.

## 2. Stage A: Per-Channel Temporal Encoder
* **File:** `models/stage_a_encoder.py`
* **Architecture:** A deep 1D Convolutional Neural Network (CNN).
* **Function:** Instead of mixing all channels immediately, it treats each of the 32 channels (DEAP) or 14 channels (DREAMER) as an independent spatial entity. It learns temporal features (frequencies, amplitudes) for each channel independently, outputting a rich 64-dimensional embedding for every single channel.

## 3. Stage B: Invariance Gating
* **File:** `models/stage_b_invariance.py`
* **Architecture:** Subject-Aware Exponential Moving Average (EMA) Tracker.
* **Function:** Tracks the cross-entropy loss contribution of different brain regions across different human subjects.
* **Math:** If a region's contribution is highly erratic across subjects (high variance), it assigns a **low invariance score**. If a region is consistently useful across all subjects, it gets a **high invariance score**.
* **Fixes Applied:** Implemented Automatic Mixed Precision (AMP) gradient clipping to absolutely guarantee that the mathematical variance division never explodes into `NaN`.

## 4. Stage C: Region-Grouped Fusion + Spectral Decoupling
* **File:** `models/stage_c_fusion.py` & `training/losses.py`
* **Architecture:** Physiological Attention + Gated L2 Penalty.
* **Function:** 
  1. Groups the raw channel embeddings into 5 biological brain regions (Frontal, Temporal, Central, Parietal, Occipital).
  2. Averages features within each region.
  3. Fuses the 5 regional embeddings into a final prediction using Cross-Attention.
* **The Novel Contribution (Spectral Decoupling):** Applies an L2 weight penalty (`||z||^2`) to the regions. Crucially, the penalty is **gated by Stage B**. Highly variant/spurious regions receive massive penalties (forcing the model to ignore them), while robust regions receive almost zero penalty.

## 5. Evaluation Harness & Plotting Suite
* **File:** `training/loso_evaluator.py` & `evaluation/plots.py`
* **Protocol:** A strict, zero-leakage **Leave-One-Subject-Out (LOSO)** evaluation loop (32 folds for DEAP, 23 folds for DREAMER).
* **Metrics Engine:** Dynamically calculates AUROC, PR-AUC, Macro/Weighted F1, MCC, G-Mean, Brier Score, and Delta-Trivial.
* **Weight Saving:** Automatically saves the raw `.pth` weights of the best performing epoch for every single fold to `results/weights/`.
* **Automated Plotting:** Upon completion of the final fold, the script instantly reads the logs and generates:
  1. **Region Gate Heatmap** (to visualize Stage B's dynamic scoring).
  2. **t-SNE Embeddings** (to visualize the 64-D separation in Stage C).

---

## What is Left to Build?
1. **Phase 4 (Class Imbalance):** Applying SMOTE, Logit-Adjusted Loss, or DeepSMOTE to rectify the F1-Minority gap.
2. **Stage D (Graph-Consistency Loss):** Extracting PLV matrices from the EEG frequencies to force the model's attention maps to align with actual neuroscience/functional connectivity.
3. **The Final Ablation Runs:** Generating the final comparison tables for publication.
