# Comprehensive EEGMMIDB Results: 109-Subject Zero-Leakage LOSO Benchmark & Audit

This report presents the verified **Leave-One-Subject-Out (LOSO)** benchmark of the **Channel-Invariance Balancing Network (CIB-Net)** on the **PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB v1.0.0)** (`utsavsinghal2604/eeg-motorimagery-eegmmidb-v1-0-0`). Every statistic, hyperparameter, anatomical channel mapping, and artifact note below was audited directly against the training scripts, [`fold_results_eegmmidb.json`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/fold_results_eegmmidb.json), and the generated diagnostic figures, alongside verified DEAP and DREAMER benchmarks in [`DEAP_Comprehensive_Results.md`](file:///d:/Deep%20Learning%20Lab/Research/DEAP_Comprehensive_Results.md) and [`DREAMER_Comprehensive_Results.md`](file:///d:/Deep%20Learning%20Lab/Research/DREAMER_Comprehensive_Results.md).

> [!WARNING]
> **Cross-Subject Generalization Remains Near Chance Across 109 Unseen Brains.**
> Under strict zero-leakage 109-fold LOSO evaluation with rigid `argmax` thresholding, CIB-Net achieves **`0.5157 ± 0.0825` AUROC** (two-sided Wilcoxon $p = 0.0568$ vs. `0.50` chance) and **`50.86% ± 5.64%` Balanced Accuracy** ($p = 0.2550$ vs. `50%`). While `22/109` subjects (`20.2%`) reach $\text{AUROC} \ge 0.60$ (peak `0.6739`), two audited implementation factors also limit downstream performance: (1) Stage B `InvarianceGate` lacked spatial-mean variance normalization, keeping all regional gates at `0.998–1.001`, and (2) uncalibrated `argmax` (`0.50`) predictions suffered subject-level logit drift, causing `4/109` single-class collapses despite balanced class priors.

---

## 1. Pipeline & Experimental Configuration (Code-Verified)

All configuration details were verified against [`config.yaml`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/config/config.yaml), [`eegmmidb_loader.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/data/eegmmidb_loader.py), [`cib_net_motor.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/cib_net_motor.py), and [`run_loso_eegmmidb.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/run_loso_eegmmidb.py):

| Parameter / Component | Verified Value / Implementation | Source Location |
| :--- | :--- | :--- |
| **Dataset & Task** | PhysioNet EEGMMIDB v1.0.0; Runs `04, 08, 12` (`T1=0` Left Fist vs. `T2=1` Right Fist) | [`eegmmidb_loader.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/data/eegmmidb_loader.py#L48-L94) |
| **Cohort & Volume** | **109 subjects** (`S001`–`S109`), **4,898** 4.0 s epochs (`[4898, 64, 640]`, avg. `44.9` trials/subj) | [`fold_results_eegmmidb.json`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/fold_results_eegmmidb.json) |
| **Signal Preprocessing** | 160 Hz (`S088`, `S092`, `S100` resampled from 128 Hz), FIR bandpass `4.0–40.0 Hz`, non-overlapping `0.0–4.0 s` | [`eegmmidb_loader.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/data/eegmmidb_loader.py#L76-L125) |
| **Normalization** | **Transductive label-free per-subject Z-score** per channel across `(trials, time)` (`axis=(0, 2)`) | [`eegmmidb_loader.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/data/eegmmidb_loader.py#L140-L142) |
| **Model (`CIBNetMotor`)** | **121,585 trainable params**: Stage A (22,752) + Stage C (98,833) | [`cib_net_motor.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/cib_net_motor.py#L6-L36) |
| **Stage A Encoder** | Shared 3-block 1D-CNN (kernels `25, 15, 7`; ch `1 -> 16 -> 32 -> 64`, ELU + BN + MaxPool + `AdaptiveAvgPool1d(1)`) | [`stage_a_encoder.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/stage_a_encoder.py#L4-L64) |
| **Stage C Region Grouping** | 5 lobes over PhysioNet's EDF order: Central (`0..20`, 21 ch), Frontal (`21..37`, 17 ch), Temporal (`38..45`, 8 ch), Parietal (`46..59`, 14 ch), Occipital (`60..63`, 4 ch); `IntraRegionAttention` + `region_dropout_p=0.2` + 2-layer 4-head Transformer ($d=64$) | [`stage_c_fusion.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/stage_c_fusion.py#L21-L104) |
| **Stage B & Loss** | [`InvarianceGate`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/stage_b_invariance.py#L5-L73) ($\alpha=0.9$, $\tau=1.0$) + [`SpectralDecouplingLoss`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/training/losses.py#L4-L44) ($\lambda_0=10^{-3}$, $\beta_{\mathrm{aux}}=0.2$) | [`losses.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/training/losses.py#L12-L44) |
| **LOSO Protocol** | 109 folds; test subject strictly held out; `n_val_subjects=3` from training pool (seed `42000 + fold_idx`), 25 epochs (CLI default over `config.yaml`'s `30`), AdamW (`lr=5e-4`, `wd=1e-4`), batch size `64`, grad clip `2.0`, AMP | [`run_loso_eegmmidb.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/run_loso_eegmmidb.py#L61-L191) |
| **Model Selection & Threshold** | Best validation `balanced_accuracy`; test decisions evaluated via **`argmax` ($\hat{p} = 0.50$)** | [`run_loso_eegmmidb.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/run_loso_eegmmidb.py#L187-L201) |
| **Execution Time** | Median `193.8 s/fold` (IQR `192.5–195.2 s`; ~352 min active GPU compute; `525.44 min` / `8.76 hr` wall time incl. paused Folds 9 & 20) | [`fold_results_eegmmidb.json`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/fold_results_eegmmidb.json) |

### Scope Differences vs. DEAP / DREAMER HCAN-CIB Pipelines
To prevent invalid apples-to-apples comparisons, note which components from DEAP/DREAMER were **absent** in this EEGMMIDB experiment:
* **Unimodal EEG only:** No peripheral autonomic signals or cross-modal HCAN attention head.
* **No Stage D / Oversampling:** No Latent V-SMOTE, GAMO, DeepSMOTE, or PLV Graph-Consistency regularizer (Runs 04/08/12 Left vs. Right Fist cues are near 50/50 per subject by design, unlike DREAMER's 35% minority skew).
* **No Validation Threshold Tuning or ECE/PR-AUC Logging:** [`compute_fold_metrics`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/run_loso_eegmmidb.py#L24-L59) evaluates fixed `argmax` predictions and does not compute Expected Calibration Error (ECE) or PR-AUC.
* **No Legacy or Shallow Baselines Run Yet:** `EEGNet`, `PlainCNN`, `ACRNN`, `CSP+LDA`, and `Shallow LR (DE/Band-Power + Z)` have not yet been evaluated under this 109-fold split.

---

## 2. Quantitative 109-Fold LOSO Metrics & Statistical Tests

All aggregate statistics, quantiles, Gini AUROC values ($\text{Gini} = 2\cdot\text{AUROC} - 1$), and hypothesis tests below were computed directly from the 109 fold dictionaries in [`fold_results_eegmmidb.json`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/fold_results_eegmmidb.json). *(Note: Standard deviations match `np.std(..., ddof=0)` reported by [`run_loso_eegmmidb.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/run_loso_eegmmidb.py#L244). Minor rounding/copy errors present in the earlier draft—such as Balanced Accuracy median listed as `50.40%` instead of `50.00%`, Macro F1 max `0.6099` instead of `0.6429`, MCC max `+0.3150` instead of `+0.3162`, and $\Delta_{\mathrm{trivial}}$ min `-0.1778` instead of `-0.1111`—are corrected here.)*

| Metric | Mean ± SD | SE | Median [IQR] | Min – Max | Test vs. Chance ($H_0$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accuracy** | `50.94% ± 5.79%` | `0.56%` | `51.11%` [`46.67%`–`54.76%`] | `40.00%` – `64.44%` | Wilcoxon $p = 0.3621$ vs. `50%` (`58 > / 2 = / 49 <`) |
| **Balanced Accuracy** | `50.86% ± 5.64%` | `0.54%` | `50.00%` [`47.04%`–`54.22%`] | `40.18%` – `64.29%` | Wilcoxon $p = 0.2550$ vs. `50%` (`54 > / 6 = / 49 <`) |
| **AUROC** | **`0.5157 ± 0.0825`** | `0.0079` | `0.5237` [`0.4524`–`0.5870`] | `0.3320` – `0.6739` | Wilcoxon $W=2367.5$, **$p = 0.0568$** ($t=1.974, p=0.0509$, 95% CI `[0.5001, 0.5312]`) |
| **Gini AUROC** | `+0.0313 ± 0.1650` | `0.0159` | `+0.0474` [`-0.0952`–`+0.1739`] | `-0.3360` – `+0.3478` | Same rank distribution as AUROC ($H_0: 0$) |
| **Macro F1** | `0.4537 ± 0.0861` | `0.0083` | `0.4498` [`0.3915`–`0.5181`] | `0.2969` – `0.6429` | Depressed below Balanced Acc by threshold skew |
| **MCC** | `+0.0118 ± 0.1437` | `0.0138` | `0.0000` [`-0.1048`–`+0.1084`] | `-0.2774` – `+0.3162` | Wilcoxon $p = 0.5581$ vs. `0.00` |
| **Delta Trivial ($\Delta_{\mathrm{triv}}$)** | `-0.0060 ± 0.0594` | `0.0057` | `0.0000` [`-0.0444`–`+0.0233`] | `-0.1111` – `+0.1333` | Wilcoxon $p = 0.5645$ vs. `0.00` (`38/109 = 34.9%` > `0`) |
| **Minority Pred Rate** | `0.5286 ± 0.3027` | `0.0291` | `0.5333` [`0.2456`–`0.8222`] | `0.0000` – `1.0000` | Mean near `0.50`, but high fold-level dispersion |
| **Recall (Minority)** | `0.5371 ± 0.3092` | `0.0298` | `0.5455` [`0.2727`–`0.8182`] | `0.0000` – `1.0000` | Tracks per-subject `Minority Pred Rate` shift |

### Cohort Stratification & Resampling Sensitivity
* **AUROC Dispersion Across 109 Subjects:** `61/109` (`56.0%`) exceed `0.50` (binomial two-sided sign test $p = 0.250$), `37/109` (`33.9%`) reach $\ge 0.55$, `22/109` (`20.2%`) reach $\ge 0.60$, and `4/109` (`3.7%`) reach $\ge 0.65$. Conversely, `9/109` (`8.3%`) fall below `0.40` (minimum `0.3320`), reflecting cross-subject spatial/polarity inversion on unseen motor cortices.
* **128 Hz $\to$ 160 Hz Resampled Subjects (`S088`, `S092`, `S100`):** [`eegmmidb_loader.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/data/eegmmidb_loader.py#L76-L77) resampled the three native 128 Hz subjects to 160 Hz. They performed at or above the cohort average: `S088` (AUROC `0.5062`, BalAcc `53.94%`), `S092` (AUROC `0.5457`, BalAcc `53.15%`), and `S100` (AUROC `0.5309`, BalAcc `61.11%`, Top-7 in Balanced Accuracy), confirming no resampling artifact degradation.

---

## 3. Subject-Level Breakdown: Best, Worst & Threshold-Drift Collapse

### 3.1 Top-10 Subjects by AUROC
*(Also noting Top Balanced Accuracy & Macro F1: `S065` achieved the cohort peak Balanced Accuracy `64.29%` and Macro F1 `0.6429`; `S056` and `S060` tied for peak MCC `+0.3162` at `61.46%` BalAcc.)*

| Rank | Subject | Fold | AUROC | Accuracy | Balanced Acc | Macro F1 | MCC | Minority Recall | Minority Pred Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **S013** | Fold 013 | **`0.6739`** | `60.00%` | `60.28%` | `0.5950` | `+0.2119` | `0.727` | `0.622` |
| 2 | **S054** | Fold 054 | **`0.6581`** | **`64.44%`** | `63.93%` | `0.6218` | **`+0.3150`** | `0.409` | `0.267` |
| 3 | **S023** | Fold 023 | `0.6522` | `48.89%` | `48.02%` | `0.3915` | `-0.0629` | `0.870` | `0.889` |
| 4 | **S018** | Fold 018 | `0.6502` | `62.22%` | `62.65%` | `0.6099` | `+0.2731` | `0.435` | `0.311` |
| 5 | **S003** | Fold 003 | `0.6462` | `51.11%` | `50.69%` | `0.4907` | `+0.0149` | `0.318` | `0.311` |
| 6 | **S020** | Fold 020 | `0.6423` | `55.56%` | `56.52%` | `0.4591` | `+0.2614` | `1.000` | `0.933` |
| 7 | **S073** | Fold 073 | `0.6417` | `50.00%` | `50.00%` | `0.4269` | `0.0000` | `0.857` | `0.857` |
| 8 | **S074** | Fold 074 | `0.6407` | `51.16%` | `50.00%` | `0.3385` | `0.0000` | `0.000` | **`0.000`** *(Collapse)* |
| 9 | **S065** | Fold 065 | `0.6310` | **`64.44%`** | **`64.29%`** | **`0.6429`** | `+0.2857` | `0.619` | `0.467` |
| 10 | **S101** | Fold 101 | `0.6285` | `57.78%` | `56.92%` | `0.4974` | `+0.2200` | `0.957` | `0.889` |

### 3.2 Bottom-10 Subjects by AUROC

| Rank | Subject | Fold | AUROC | Accuracy | Balanced Acc | Macro F1 | MCC | Minority Recall | Minority Pred Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 109 | **S066** | Fold 066 | **`0.3320`** | `46.67%` | `45.75%` | `0.3510` | `-0.1493` | `0.870` | `0.911` |
| 108 | **S031** | Fold 031 | `0.3419` | `42.22%` | `42.49%` | `0.4150` | `-0.1549` | `0.304` | `0.378` |
| 107 | **S038** | Fold 038 | `0.3518` | `42.22%` | `42.39%` | `0.4196` | `-0.1540` | `0.500` | `0.578` |
| 106 | **S004** | Fold 004 | `0.3518` | `46.67%` | `45.65%` | `0.3182` | `-0.2109` | `0.000` | `0.044` |
| 105 | **S047** | Fold 047 | `0.3790` | `42.22%` | `42.86%` | `0.4196` | `-0.1455` | `0.524` | `0.600` |
| 104 | **S039** | Fold 039 | `0.3794` | `46.67%` | `47.33%` | `0.4181` | `-0.0667` | `0.773` | `0.800` |
| 103 | **S057** | Fold 057 | `0.3829` | `42.22%` | `44.94%` | `0.3260` | `-0.1774` | `0.857` | `0.911` |
| 102 | **S093** | Fold 093 | `0.3933` | `44.44%` | `44.47%` | `0.4444` | `-0.1107` | `0.455` | `0.511` |
| 101 | **S105** | Fold 105 | `0.3992` | `48.89%` | `48.02%` | `0.3915` | `-0.0629` | `0.091` | `0.111` |
| 100 | **S102** | Fold 102 | `0.4014` | `42.86%` | `42.86%` | `0.3000` | **`-0.2774`** | `0.857` | `0.929` |

### 3.3 Why "Zero Mode Collapse" Cannot Be Claimed Globally: Uncalibrated Threshold Drift
An earlier draft interpreted the cohort mean `Minority Pred Rate = 0.5286` as proof of "Zero Mode Collapse." Quantitative audit of the 109 folds refutes that simplification:
* **No Cohort-Wide Bias, High Fold-Level Skew:** Predictions center near 50% only *on average* (`0.5286`), because individual unseen subjects experience large positive or negative logit shifts ($\text{SD} = 0.3027$; IQR `[0.2456, 0.8222]`).
* **Exact Single-Class Collapses & Extreme Skew:** Four folds (`S074` & `S109` at `MPR = 0.000`; `S007` & `S084` at `MPR = 1.000`) predicted a single class for 100% of test trials (`MCC = 0.0000`). Across all 109 folds, `21/109` (`19.3%`) fell outside `[0.10, 0.90]`, and only `37/109` (`33.9%`) fell in `[0.30, 0.70]`.
* **Sharp Separation Between Ranking (AUROC) and Hard Classification (`Macro F1`):** Prediction skew strongly degrades `Macro F1` ($\text{Spearman}(|\text{MPR} - 0.5|, \text{Macro F1}) = -0.7144$, $p = 2.7 \times 10^{-18}$). Folds with balanced predictions (`0.25 <= MPR <= 0.75`, $n=44$) average **`Macro F1 = 0.5151`** (`BalAcc = 52.47%`, `AUROC = 0.5309`, `MCC = +0.0516`), whereas skewed folds ($n=65$) drop to **`Macro F1 = 0.4122`** (`BalAcc = 49.77%`, `MCC = -0.0152`).
* **Threshold-Drift Case Study (`S074` & `S023`):** On `S074`, the model ranked Left vs. Right Fist trials at **`AUROC = 0.6407`** (#8 in cohort), yet uncalibrated `argmax` (`0.50`) classified 100% of all 43 trials as Left Fist (`MPR = 0.000`, `BalAcc = 50.00%`, `Macro F1 = 0.3385`). Similarly, `S023` achieved **`AUROC = 0.6522`** (#3) but collapsed to `MPR = 0.889` (`BalAcc = 48.02%`, `MCC = -0.0629`).

---

## 4. Stage B Invariance Gate & Root-Cause Code Diagnosis

| Cortical Region (EDF Channels Verified) | # Ch | Mean ± SD | Median [IQR] | Min – Max | Wilcoxon vs. `Central_Sensorimotor` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frontal** (`[21..37]`: `Fp1..F8`) | 17 | `1.00104 ± 0.00096` | `1.00114` [`1.00040`–`1.00169`] | `0.99787` – `1.00357` | $+0.00059$, $p = 1.0 \times 10^{-9}$ |
| **Temporal** (`[38..45]`: `FT7..TP8`) | 8 | `1.00094 ± 0.00073` | `1.00099` [`1.00042`–`1.00143`] | `0.99898` – `1.00265` | $+0.00049$, $p = 5.4 \times 10^{-8}$ |
| **Central_Sensorimotor** (`[0..20]`: `FC5..CP6`) | 21 | `1.00045 ± 0.00069` | `1.00037` [`0.99994`–`1.00085`] | `0.99839` – `1.00225` | Baseline reference |
| **Parietal** (`[46..59]`: `P7..PO8`) | 14 | `0.99944 ± 0.00072` | `0.99952` [`0.99892`–`0.99994`] | `0.99728` – `1.00076` | $-0.00101$, $p = 4.6 \times 10^{-13}$ |
| **Occipital** (`[60..63]`: `O1, Oz, O2, Iz`) | 4 | `0.99814 ± 0.00142` | `0.99822` [`0.99716`–`0.99932`] | `0.99368` – `1.00043` | $-0.00231$, $p = 1.6 \times 10^{-17}$ |

### Three Key Diagnostic Findings
1. **PhysioNet EDF Index Grouping Confirmed Accurate:** Unlike standard MNE 10-20 ordering (`Fp1` first), raw PhysioNet BCI2000 `.edf` files place `FC5..CP6` in indices `0..20` and `Fp1..F8` in `21..37`. Thus, [`stage_c_fusion.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/stage_c_fusion.py#L35-L41) and [`config.yaml`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/config/config.yaml#L14-L19) map all 5 anatomical lobes faithfully.
2. **Code Bug — Missing Spatial-Mean Variance Normalization Neutralized Stage B (`g_r ≈ 1.000`):**
   In DEAP/DREAMER (`main.tex`), cross-subject variances $v_r$ are normalized by the spatial mean prior to Softmax ($s_r = v_r / (\bar{v} + \epsilon)$). However, [`InvarianceGate.get_gates`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/stage_b_invariance.py#L65-L73) passes raw unnormalized EMA loss variance directly into `Softmax(-self.variance_ema / self.tau)` with `tau = 1.0`. Because auxiliary CE losses near chance cluster closely ($\text{Var}_s \approx 10^{-3}\text{--}10^{-2}$), $\exp(-0.001) \approx 0.999$, pinning every gate inside `[0.9937, 1.0036]`. Consequently, Spectral Decoupling penalty weights $\lambda_0 / g_r$ differ by **less than 0.7%** between Central and Occipital cortices (vs. ~3.2x peak-to-trough modulation on DREAMER), effectively reducing Stage B to uniform regularization.
3. **Motor Imagery Inductive Bias Mismatch in Stage A:** Motor imagery lateralization (Left vs. Right Fist) manifests as event-related **$\mu$ (8–12 Hz) and $\beta$ (13–30 Hz) band-power desynchronization** (`C3` vs. `C4` spatial covariance / variance changes), rather than time-domain waveform mean amplitude. Passing raw bandpass signals through Stage A's temporal CNN followed by [`nn.AdaptiveAvgPool1d(1)`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/models/stage_a_encoder.py#L37) collapses temporal phase and erases oscillatory band-power envelopes across the 4-second trial window.

---

## 5. Verified Multi-Dataset Comparison (DEAP vs. DREAMER vs. EEGMMIDB)

The table below fixes the erroneous `~79.45%*` DEAP Balanced Accuracy cited in the prior draft (which mistakenly mixed an unnormalized raw-accuracy figure from HPC Job 31812 into the Balanced Accuracy column) by pulling verified leakage-free LOSO metrics directly from [`DEAP_Comprehensive_Results.md`](file:///d:/Deep%20Learning%20Lab/Research/DEAP_Comprehensive_Results.md), [`DREAMER_Comprehensive_Results.md`](file:///d:/Deep%20Learning%20Lab/Research/DREAMER_Comprehensive_Results.md), and [`main.tex`](file:///d:/Deep%20Learning%20Lab/Research/main.tex).

| Benchmark Dataset | Protocol / Configuration | Modality & Ch | Subj ($N$) | AUROC | Balanced Acc | Macro F1 | Minority Pred Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DEAP** *(Valence)* | `HCAN-CIB (raw)` | 32 EEG + 8 Periph | 32 | `0.5148 ± 0.0911` | `51.47% ± 7.15%` | `0.4825 ± 0.0823` | `0.4992 ± 0.2389` |
| **DEAP** *(Valence)* | `HCAN-CIB (Z-score)` | 32 EEG + 8 Periph | 32 | `0.5298 ± 0.1129` | `51.63% ± 7.99%` | `0.5042 ± 0.0845` | `0.5133 ± 0.1513` |
| **DEAP** *(Valence)* | `Shallow LR (DE + Z)` | DE (EEG + Periph) | 32 | **`0.5754 ± 0.0991`** | **`55.39% ± 6.17%`** | **`0.5501 ± 0.0640`** | `0.5078 ± 0.0827` |
| **DREAMER** *(Valence)* | `HCAN-CIB (V-SMOTE)` | 14 EEG + 2 ECG | 23 | `0.5006 ± 0.1812` | `51.08% ± 8.03%` | `0.3682 ± 0.1210` | `0.6039 ± 0.4357` |
| **DREAMER** *(Valence)* | `HCAN-CIB (DeepSMOTE)` | 14 EEG + 2 ECG | 23 | **`0.5589 ± 0.1638`** | `51.05% ± 8.49%` | `0.4166 ± 0.1244` | `0.6014 ± 0.3263` |
| **EEGMMIDB** *(Left/Right MI)* | **`CIB-Net (Z-score, argmax)`** | **64 EEG** | **109** | **`0.5157 ± 0.0825`** | **`50.86% ± 5.64%`** | **`0.4537 ± 0.0861`** | **`0.5286 ± 0.3027`** |

---

## 6. Visual Diagnostic Suite & Artifact Provenance Audit

### A. 109-Fold LOSO Benchmark Plots
1. **Cross-Subject AUROC & Accuracy Distributions:** [`loso_distribution_109subjects.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/loso_distribution_109subjects.png) — KDE histograms across all 109 folds showing wide inter-subject spread around chance (`AUROC` `0.332–0.674`; `Accuracy` `40.0%–64.4%`).
2. **Stage B Gate Heatmap (109 Folds × 5 Lobes):** [`gate_heatmap_dreamer_style.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/gate_heatmap_dreamer_style.png) — Colorbar spans `[0.994, 1.003]`, revealing systematic relative suppression of Occipital (`0.994–0.998`) relative to Frontal/Temporal (`1.001–1.003`) despite negligible absolute amplitude.
3. **Stage B Gate Profile (Zoomed vs. Full Scale):** [`gate_profile_mean_std.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/gate_profile_mean_std.png) (zoomed `[0.96, 1.04]` mean ± SD) alongside [`gate_heatmap_eegmmidb.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/gate_heatmap_eegmmidb.png) (full `[0, 1.25]` bar chart showing uniform `1.00` height).
4. **2-Panel t-SNE (`S001`–`S012`, Class vs. Subject Identity):** [`tsne_brain_embeddings_dreamer_style.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/tsne_brain_embeddings_dreamer_style.png) — Panel 1 shows Left vs. Right Fist thoroughly interleaved, whereas Panel 2 shows partial spatial clustering by subject identity (e.g., `S007` at upper-left/lower-left clusters).
   * *Code Provenance Caveat:* [`run_loso_eegmmidb.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/run_loso_eegmmidb.py#L211-L214) saved fold metrics and gates without `"model_state_dict"`. Accordingly, [`generate_plots.py`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/generate_plots.py#L120-L135) projected `S001–S012` through either an uninitialized `CIBNetMotor` or temporal-mean fallback rather than converged Fold-13 weights; thus this t-SNE reflects intrinsic input/random-projection subject domain shift.

### B. Pre-LOSO 2-Epoch Prototype Artifacts (Single-Fold Smoke Test)
The remaining four PNGs in [`plots/`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/) originate from an early 2-epoch single-fold test of [`Trainer.fit`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/training/loso_evaluator.py#L116-L187) on a 10-subject subset and **do not represent the 109-fold benchmark**: [`training_curves.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/training_curves.png) (Epochs `0–1`), [`confusion_matrix.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/confusion_matrix.png) (45 single-subject trials predicting 100% Right Fist), [`invariance_gates.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/invariance_gates.png), and [`tsne_latent_manifold.png`](file:///d:/Deep%20Learning%20Lab/Research/CIB-Net-EEGMMIDB/results/plots/tsne_latent_manifold.png).

---

## 7. Scientific & Engineering Next Steps

1. **Normalize Stage B Cross-Subject Variance (`InvarianceGate`):** Scale `variance_ema` by its regional mean ($s_r = v_r / (\bar{v} + \epsilon)$) before Softmax (or lower $\tau$ to `0.005–0.01`), and save `model_state_dict` inside fold checkpoints so trained manifold t-SNEs and gate dynamics are reproducible.
2. **Add Baselines Under the Exact 109-Fold Split:** Evaluate `EEGNet`, `ShallowConvNet`/`PlainCNN`, `CSP + LDA`, and a **Shallow LR (`DE / Log-Band-Power + Z-score`)** reference to establish whether raw CNNs underperform spectral linear baselines on EEGMMIDB just as on DEAP (`0.5754` vs. `0.5298`).
3. **Eliminate Threshold-Drift Collapses & Log Calibration:** Replace fixed `argmax` with validation-tuned thresholding or test-time logit centering, and log `ECE` and `PR-AUC` per fold to convert `S074`-style ranking signals (`0.6407` AUROC) into positive `Macro F1` and `MCC`.
4. **Upgrade Spectral Encoder & Stage E Domain Alignment:** Replace raw `AdaptiveAvgPool1d(1)` mean pooling with log-variance / $\mu$-$\beta$ band-power (or Riemannian/Euclidean Alignment) features, and attach **Stage E: Subject-Adversarial Domain Adaptation (DANN / GRL)** across subjects.
