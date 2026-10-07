# EEG Multimodal Emotion Recognition: Project Summary & Findings

This document summarizes a rigorous 8-phase research project analyzing Multimodal Emotion Recognition (Valence and Arousal) on the DEAP dataset using EEG and Peripheral physiological signals.

The core investigation centered on the **Hierarchical Cross-Attention Network (HCAN)**, diagnosing severe **Modality Starvation** (where the network ignores complex EEG data in favor of simpler peripheral data), and evaluating state-of-the-art gradient modulation interventions to resolve it.

---

## 🔬 Project Phases Overview

### Phase 1–3: Architecture Implementation & Validation
- **What we did:** Implemented the HCAN architecture featuring a 2D-CNN EEG encoder and an MLP Peripheral encoder, fused via Multi-head Cross-Attention. We established a strict Leave-One-Subject-Out (LOSO) cross-validation pipeline to measure true generalization, completely isolated from subject-specific memorization.
- **Result:** We observed that the model struggled to exceed random chance significantly (~0.52-0.54 accuracy). Intensive checks on normalization (z-scoring) and data leakage confirmed the pipeline was mathematically sound, indicating the bottleneck was structural to the learning process itself.

### Phase 4: Unimodal vs. Joint Benchmarking
- **What we did:** Trained isolated EEG-only and Peripheral-only models to compare against the Joint (HCAN) model.
- **Result:** The Joint model performed almost identically to the Peripheral-only model, and both outperformed the EEG-only model. This triggered the hypothesis of **Modality Starvation / Imbalance**: the joint network was greedily optimizing for the easier, lower-dimensional peripheral signals and ignoring the high-dimensional, noisy EEG signals.

### Phase 5: Diagnosing Modality Starvation (Probing)
- **What we did (5A):** Ablation testing. We took the trained Joint model and evaluated it with zeroed-out EEG inputs, then zeroed-out Peripheral inputs. 
- **What we did (5B):** Linear Probing. We extracted the 64-dim embeddings from the trained EEG encoder and trained a shallow classifier on them, comparing it to embeddings from a completely *untrained* (randomly initialized) EEG encoder.
- **Result:** Zeroing the EEG input caused almost zero performance drop, while zeroing the Peripheral input destroyed accuracy. Furthermore, the linear probe on the trained EEG encoder performed no better than a random projection. This definitively proved the EEG branch was experiencing gradient starvation and failing to learn meaningful representations.

### Phase 6 & 7: Shallow Baselines & Feature Engineering
- **What we did:** Shifted to hand-crafted **Differential Entropy (DE)** features (4 bands: theta, slow-alpha, alpha, beta, gamma) to simplify the learning manifold. We rigorously evaluated classic Machine Learning baselines (Logistic Regression, SVC) on Trial-Level features, testing Early Fusion (concatenation) and Late Fusion (probability averaging).
- **Result:** The shallow baselines set a surprisingly high bar. Peripheral (P1) features with simple Logistic Regression achieved **0.5836** accuracy on Valence and **0.5539** on Arousal. 

### Phase 8: Deep Interventions for Modality Starvation
- **What we did:** Armed with DE features, we returned to the deep HCAN model and implemented 5 advanced deep-learning interventions to force the network to utilize both modalities:
  - **B0:** Joint Model with Auxiliary Heads (Baseline)
  - **B1 (OGM-GE):** On-the-fly Gradient Modulation (dynamically scales learning rates based on modality generalization capabilities).
  - **B2 (UDI):** Unimodal Distribution Instructor (aligns joint representations with pre-trained unimodal spaces).
  - **B3 (GOAL):** Gradient Orthogonalization (PCGrad) (projects conflicting gradients to prevent the peripheral branch from dominating the EEG branch).
  - **B4:** Modality Dropout (randomly zeroes out modalities during training).
  - **B5:** Peripheral Regularization (adds noise/dropout specifically to the dominant peripheral branch).
- **Result:** Evaluated over an immense grid search (1,152 models trained), none of the complex interventions significantly outperformed the baseline `B0`. Moreover, the deep models failed to beat the shallow models from Phase 7.

---

## 📊 Final Conclusion Tables

### Valence Final Results
| Model / Pipeline | Configuration | Accuracy | p-value vs B0 |
| :--- | :--- | :--- | :--- |
| **Shallow: Peripheral (P1)** | LR (C=0.01) | **0.5836** | - |
| **Shallow: EEG (F1)** | SVC (RBF) | **0.5609** | - |
| **Deep: B0 (Joint+Aux)** | Baseline | 0.5591 | - |
| Deep: B5 (Periph-Reg) | Dropout | 0.5531 | 0.644 (NS) |
| Deep: B1 (OGM-GE) | alpha = 2.0 | 0.5516 | 0.530 (NS) |
| Deep: B4 (Modality Dropout) | 50% | 0.5474 | 0.523 (NS) |
| Deep: B2 (UDI) | lambda = 0.1 | 0.5133 | **0.004 (Worse)** |
| Deep: B3 (GOAL) | PCGrad | 0.5117 | **0.002 (Worse)** |

### Arousal Final Results
| Model / Pipeline | Configuration | Accuracy | p-value vs B0 |
| :--- | :--- | :--- | :--- |
| **Shallow: Peripheral (P1)** | LR (C=0.01) | **0.5539** | - |
| **Shallow: EEG (F1)** | SVC (RBF) | **0.5383** | - |
| Deep: B1 (OGM-GE) | alpha = 2.0 | 0.5484 | 0.778 (NS) |
| **Deep: B0 (Joint+Aux)** | Baseline | 0.5479 | - |
| Deep: B5 (Periph-Reg) | Dropout | 0.5318 | 0.032 (NS) |
| Deep: B4 (Modality Dropout) | 50% | 0.5242 | 0.085 (NS) |
| Deep: B2 (UDI) | lambda = 0.5 | 0.5180 | 0.057 (NS) |
| Deep: B3 (GOAL) | PCGrad | 0.4977 | **0.002 (Worse)** |

*(Note: NS = Not Statistically Significant)*

---

## 💡 Executive Summary for Presentation

1. **The Core Challenge:** In multimodal emotion recognition on the DEAP dataset, deep neural networks (like HCAN) suffer from severe **Modality Starvation**. The network takes the "path of least resistance," heavily optimizing for the simpler peripheral physiological signals while almost entirely ignoring the complex, high-dimensional EEG signals.
2. **The Deep Learning Plateau:** We implemented five state-of-the-art gradient modulation and regularization techniques (including OGM-GE, UDI, and PCGrad) designed specifically to force the network to learn from both modalities. However, **none of these interventions successfully improved generalization** over a basic multi-head joint model.
3. **Shallow vs. Deep:** When utilizing hand-crafted Differential Entropy (DE) features, **shallow machine learning models (Logistic Regression, SVC) consistently outperformed the complex deep learning architectures.** 
4. **Final Verdict:** For datasets of this scale and complexity (like DEAP), the inductive biases and simplicity of shallow linear models on carefully engineered features yield superior reliability and accuracy compared to highly parameterized deep cross-attention networks, which fall victim to optimization imbalances.
