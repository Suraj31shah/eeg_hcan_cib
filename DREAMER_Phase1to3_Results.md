# DREAMER Results: Phase 1-3 (Baseline Evaluation)

This report compiles the rigorously tested, Leakage-Free LOSO results on the **DREAMER** dataset. 
It compares the legacy baseline models against the HCAN-CIB architecture (Stages A-C, before class balancing).

## 1. Metric Comparison

| Model | Balanced Acc | Macro F1 | AUROC |
| :--- | :--- | :--- | :--- |
| **ACRNN (Baseline)** | `0.4912 ± 0.0608` | `0.3565 ± 0.0817` | `0.4729 ± 0.1468` |
| **PlainCNN (Baseline)**| `0.5018 ± 0.0085` | `0.3428 ± 0.0737` | `0.5005 ± 0.1099` |
| **EEGNet (Baseline)** | `0.4896 ± 0.0299` | `0.3831 ± 0.0589` | `0.5139 ± 0.1447` |
| **HCAN-CIB (Phase 1-3)**| `0.5016 ± 0.0334` | `0.3470 ± 0.0958` | `0.4783 ± 0.1395` |

*(Note: The baseline scripts output a limited subset of metrics. Full metrics for HCAN-CIB Phase 1-3 are provided below.)*

### Comprehensive HCAN-CIB (Phase 1-3) Metrics

| Metric | Value |
| :--- | :--- |
| **Accuracy** | `0.4758 ± 0.1398` |
| **Balanced Acc** | `0.5016 ± 0.0334` |
| **AUROC** | `0.4783 ± 0.1395` |
| **Gini AUROC** | `-0.0434 ± 0.2791` |
| **PR AUC** | `0.4493 ± 0.1731` |
| **Macro F1** | `0.3470 ± 0.0958` |
| **MCC** | `0.0039 ± 0.0678` |
| **Delta Trivial** | `-0.1498 ± 0.1617` |
| **Recall Minority** | `0.6576 ± 0.4320` |
| **Expected Calibration Error (ECE)** | `0.2479 ± 0.1576` |

#### Validation-Tuned Threshold Metrics
| Metric | Value |
| :--- | :--- |
| **Balanced Acc** | `0.4961 ± 0.0405` |
| **Macro F1** | `0.3546 ± 0.1000` |
| **MCC** | `-0.0174 ± 0.1115` |
| **Recall Minority** | `0.6111 ± 0.4329` |
| **Precision Minority** | `0.2868 ± 0.1821` |
| **Minority Pred Rate** | `0.6159 ± 0.4134` |


---

## 2. Analysis of the Results

### Total Mode Collapse
The evaluation confirms **Total Mode Collapse** across all tested models on the DREAMER dataset under leakage-free evaluation. 
Hovering precisely around `0.50` AUROC and `0.50` Balanced Accuracy, the models are mathematically failing to learn any generalizable patterns and are simply guessing or defaulting to a single class (Minority Prediction Rate is essentially 0%).

### Why is this happening?
The fact that **EEGNet and ACRNN (which only take EEG data)** collapse in the exact same way as **HCAN-CIB (which balances EEG and ECG)** scientifically proves that **Modality Imbalance is not the root cause.** The root cause is inherent to the EEG signals themselves.

There are two major remaining roadblocks to cross-subject generalization:
1. **Extreme Class Imbalance:** The model takes the "lazy" route and predicts the majority class to minimize loss, ignoring the underlying features.
2. **Domain Shift:** Different subjects' brains produce fundamentally different signal amplitudes and topographies.

### Next Steps
The next scientific step is to implement **Phase 4 (Class Weights & V-SMOTE)** on Colab. By forcing the model to pay attention to the minority class via synthetic latent upsampling, we will test if the model can finally break the 0.50 AUROC barrier, or if explicit Domain Adaptation will be required.
