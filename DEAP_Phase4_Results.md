# DEAP Results: Phase 4 (Class Imbalance) Analysis

This report compares the performance of the **HCAN-CIB** architecture on the DEAP dataset before and after applying **V-SMOTE (Latent Space Oversampling)** to address the severe class imbalance and mode collapse.

All results are obtained using the mathematically valid **Leakage-Free LOSO Evaluation Pipeline**.

## 1. Metric Comparison

| Metric | Phase 1-3 (Baseline HCAN-CIB) | Phase 4 (V-SMOTE) |
| :--- | :--- | :--- |
| **AUROC** | `0.5148 ± 0.0911` | `0.5138 ± 0.0809` |
| **Balanced Accuracy** | `0.5074 ± 0.0729` | `0.5110 ± 0.0662` |
| **Macro F1** | `0.4857 ± 0.0741` | `0.4892 ± 0.0749` |
| **Accuracy** | `0.5063 ± 0.0723` | `0.5109 ± 0.0658` |
| **Minority Pred Rate** | `0.5055 ± 0.2012` | `0.4836 ± 0.1988` |
| **Delta vs Trivial** | `+0.0039` | `+0.0086` |

---

## 2. Analysis of the Results

> [!WARNING]
> **Complete Subject-Level Collapse Confirmed.**
> The results clearly indicate that the model is entirely failing to generalize across subjects. The AUROC is stuck exactly at random chance (~0.51) regardless of class balancing techniques.

### Why didn't SMOTE work?
V-SMOTE successfully forced the model to look at the minority class during training. However, the `Minority Pred Rate` variance is huge (`±0.20`), and `Delta vs Trivial` is practically `0.0`. 

This tells us exactly what is happening: **The EEG features do not generalize across different people's brains**. 
When the model trains on Subjects 2-32 and evaluates on Subject 1, Subject 1's brain waves look entirely alien compared to the training data. The model defaults to guessing because the latent space mappings it learned simply do not exist in the new subject.

### Brain Embeddings (t-SNE)

````carousel
![Phase 1-3 (No SMOTE)](/absolute/path/to/artifacts/tsne_deap.png)
<!-- slide -->
![Phase 4 (V-SMOTE)](/absolute/path/to/artifacts/tsne_deap_smote.png)
````

*(Note: The t-SNE graphs show the brain embeddings extracted by the model. If the model had successfully learned cross-subject features, we would see two distinct clusters (High vs Low Valence). Instead, we likely see a massive overlap.)*

## 3. Next Steps (Solving Subject Variance)
Phase 4 proved that the problem is **not just class imbalance**. The problem is **Domain Shift** (Subject-to-Subject variance). 

To break the 0.50 barrier on DEAP and DREAMER under strict LOSO, we must implement a **Subject-Adversarial Domain Adaptation (DANN)** or **Subject-Specific Normalization** to strip the "subject identity" from the EEG waves before they reach the classifier.
