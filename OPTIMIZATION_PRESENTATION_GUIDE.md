# Linear and Nonlinear Optimization: The 5-Act Presentation Guide
## Presenting HCAN-CIB / REGAL-Net as an Optimization Study

> **Course Context:** Linear and Nonlinear Optimization (Graduate / Seminar)  
> **Core Theme:** Multi-objective non-convex optimization pathologies, gradient projection geometry, stationary point regularization, and the convex vs. non-convex generalization gap.

---

## Executive Narrative Overview

When presenting this paper for an optimization evaluation, structure your talk into **five chronological acts**:

```
 ACT 1: The Root Optimization Problem
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Joint Multimodal ERM: Asymmetric gradient descent & loss saturation    │
 │ (Peripheral descends fast, EEG gradient vanishes: 19x–30x disparity)  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     ▼
 ACT 2: First Attempt — Off-the-Shelf Modality Optimizers
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Applied OGM-GE, UDI, PCGrad across 1,152 trained models                │
 │ Result: Failed! 99.88% gradient orthogonality & monolithic EEG penalty │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     ▼
 ACT 3: The Paradigm Shift — Channel-Level Optimization
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Discovery: Subject variance is 349.6x label variance                   │
 │ Modality balancing fails because channels are heterogeneous            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     ▼
 ACT 4: The Solution — HCAN-CIB Multi-Stage Optimization
 ┌────────────────────────────────────────────────────────────────────────┐
 │ • Stage A: Channel-isolated 1D-CNN (prevents early spatial coupling)   │
 │ • Stage B: Simplex-constrained EMA invariance gate γ_r (preconditioner)│
 │ • Stage C: Region-coupled Spectral Decoupling (Bound: |f*| < 1/2λ)    │
 │ • Stage D: Latent manifold minority oversampling (fixes class collapse)│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     ▼
 ACT 5: Results & The Convex vs. Non-Convex Revelation
 ┌────────────────────────────────────────────────────────────────────────┐
 │ • AUROC 0.478 ➔ 0.559, Gini flips positive, ECE drops to 0.225         │
 │ • Why Convex LR beats 1,152 Non-Convex models: Hessian ill-conditioning│
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Act 1: The Root Optimization Problem — Asymmetric Descent & Modality Starvation

### 1.1 The Multi-Branch Optimization Formulation
Consider a multimodal classification system trained via Empirical Risk Minimization (ERM) on a joint cross-entropy objective:
$$\min_{\theta_{\text{eeg}}, \theta_{\text{per}}, \phi} \frac{1}{N}\sum_{i=1}^N \mathcal{L}_{\text{CE}}\Big(h_\phi\big(f(\mathbf{x}_i^{\text{eeg}};\theta_{\text{eeg}}),\, g(\mathbf{x}_i^{\text{per}};\theta_{\text{per}})\big),\, y_i\Big)$$
where:
* $\mathbf{x}^{\text{eeg}} \in \mathbb{R}^{C \times T}$ ($32 \times 512 = 16{,}384$ continuous values) is the noisy, high-dimensional neural stream.
* $\mathbf{x}^{\text{per}} \in \mathbb{R}^{d_p}$ ($d_p = 24$ statistical features) is the compact autonomic stream.
* $h_\phi$ represents the fusion neck and classifier head producing scalar logit $f$.

### 1.2 The Optimization Pathology
1. **Asymmetric Descent Trajectories:** The low-dimensional peripheral branch has a smoother, steeper initial gradient descent trajectory. It descends rapidly during early epochs.
2. **Loss Saturation:** Rapid reduction in cross-entropy drives predictions to extreme values ($\hat{p} \to 1$ or $\hat{p} \to 0$), causing the derivative with respect to logits to vanish:
   $$\nabla_f \mathcal{L}_{\text{CE}} = \sigma(f) - y \to 0$$
3. **Gradient Starvation:** By the chain rule:
   $$\nabla_{\theta_{\text{eeg}}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial f} \cdot \frac{\partial f}{\partial \mathbf{h}_{\text{eeg}}} \cdot \frac{\partial \mathbf{h}_{\text{eeg}}}{\partial \theta_{\text{eeg}}}$$
   Because $\frac{\partial \mathcal{L}}{\partial f} \to 0$, the gradient update reaching the slow-learning EEG branch collapses to zero before the encoder learns discriminative cortical representations.

### 1.3 Empirical Diagnostic Proof
* **Gradient Ratio Disparity:** Tracking encoder gradient norms during training revealed an extreme **$19\times$ to $30\times$ gradient norm imbalance** ($\|\nabla_{\theta_{\text{per}}}\| \gg \|\nabla_{\theta_{\text{eeg}}}\|$).
* **Linear Probe Degeneracy:** A linear probe trained on the frozen representations of the joint EEG encoder achieved an accuracy of $0.5370$, which was statistically indistinguishable from a probe trained on a completely random, untrained projection ($0.5340$). The joint network failed to optimize cortical weights.

---

## Act 2: Testing SOTA Modality Optimizers & Why They Failed

### 2.1 Applied Modality Optimizers (1,152 Trained Models Benchmark)
To test if existing gradient-level optimizers could rescue the EEG branch, five optimization interventions were systematically evaluated across 1,152 trained models on DEAP:

1. **OGM-GE (On-the-fly Gradient Modulation by Generalization Discrepancy):**  
   Monitors training vs. validation loss descent discrepancies between branches and dynamically attenuates the dominant peripheral branch gradient:
   $$\mathbf{g}_{\text{per}} \leftarrow \mathbf{g}_{\text{per}} \cdot k_t, \qquad k_t = \min\left(1, \, \frac{\Delta L_{\text{eeg}}}{\Delta L_{\text{per}}}\right)$$
2. **UDI (Unimodal Distribution Instructor):**  
   Constrains the joint optimization manifold by adding a geometric penalty aligning joint representations with pre-trained unimodal manifolds.
3. **PCGrad (Projected Conflicting Gradients):**  
   Treats branches as competing Pareto objectives. When branch gradients conflict ($\langle \mathbf{g}_{\text{eeg}}, \mathbf{g}_{\text{per}} \rangle < 0$), it projects $\mathbf{g}_{\text{eeg}}$ onto the normal hyperplane of $\mathbf{g}_{\text{per}}$:
   $$\mathbf{g}_1 \leftarrow \mathbf{g}_1 - \frac{\langle \mathbf{g}_1, \mathbf{g}_2 \rangle}{\|\mathbf{g}_2\|^2}\mathbf{g}_2$$

### 2.2 The Empirical Failure (Table 8 in Paper)
* None of the five gradient modulation frameworks improved cross-subject generalization over the auxiliary baseline (B0: $0.5591$ valence).
* Stricter geometric constraints actively damaged performance: **UDI ($0.5133, p=0.004$)** and **PCGrad ($0.5117, p=0.002$)** significantly degraded accuracy below baseline.

### 2.3 Optimization Root Cause Analysis
1. **The $99.88\%$ Orthogonality Discovery:** Tracking branch gradient vectors across thousands of iterations revealed that conflicting gradients ($\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0$) occurred in fewer than **$0.12\%$** of iterations. In the other **$99.88\%$**, branch gradients were already mutually orthogonal ($\langle \mathbf{g}_1, \mathbf{g}_2 \rangle \approx 0$). PCGrad was projecting into unconstrained null voids, destroying optimizer momentum.
2. **The Monolithic Modality Fallacy:** All existing modality optimizers treat EEG as a **single monolithic entity**. Scaling up EEG gradients as a single block blindly amplified noisy, subject-specific artifacts (e.g., ocular blinks and electromyographic noise).

---

## Act 3: The Paradigm Shift — From Modality to Channel Invariance

### 3.1 Diagnosing the Optimization Landscape
To understand why monolithic scaling failed, we conducted ANOVA variance decomposition on differential entropy (DE) features across 32 subjects:

| Quantity | Raw Features | Per-Subject $Z$-Scored |
| :--- | :---: | :---: |
| **Variance explained by Subject Identity ($\eta^2_{\text{subj}}$)** | **$47.75\%$** | $0.0\%$ (by construction) |
| **Variance explained by Emotion Label ($\eta^2_{\text{label}}$)** | **$0.137\%$** | -- |
| **Ratio $\eta^2_{\text{subj}} / \eta^2_{\text{label}}$** | **$349.6 : 1$** | -- |
| Subject Decoding Accuracy (Chance $3.125\%$) | $97.81\%$ | $3.125\%$ (Chance) |

### 3.2 The Core Optimization Insight
Scalp electrodes are biologically, physiologically, and statistically heterogeneous:
* Frontal electrodes are contaminated by high-amplitude ocular artifacts.
* Temporal electrodes are subject to idiosyncratic electromyographic (EMG) jaw/muscle noise.
* Parietal and occipital electrodes exhibit stable cognitive alpha rhythms.

> **Key Takeaway:** You cannot solve modality starvation by scaling the entire EEG modality together. Optimization must move from **modality-level balancing** to **channel-level invariance filtering**, penalizing unstable channels while allowing invariant channels to receive gradient updates.

---

## Act 4: The HCAN-CIB Pipeline — Multi-Stage Constrained Optimization

HCAN-CIB resolves channel imbalance, modality starvation, and class collapse through four distinct optimization stages:

```
   Raw EEG (32 x 512) 
          │
   [Stage A: Channel-Isolated 1D-CNNs (Shared Weights)]
          │  --> Prevents early spatial gradient mixing
          ▼
   [Stage D: Latent-Space Manifold Oversampling]
          │  --> Solves minority class collapse (V-SMOTE / GAMO / DeepSMOTE)
          ▼
   [Stage C: Region Grouping + Intra-Region Attention]
          │  --> Aggregates channels into R functional regions
          ▼
   [Stage B: Simplex-Constrained EMA Invariance Gate γ_r]
          │  --> Down-weights high-variance channels, promotes stable regions
          ▼
   [Cross-Modal Attention + Region-Coupled Spectral Decoupling (L_total)]
             --> Strictly bounds logit norm: |f*| < 1 / (2λ)
```

### 4.1 Stage A: Decoupling Channel Gradients
* Standard depthwise spatial convolutions mix all 32 electrodes immediately, coupling the gradient of every electrode to the loudest (noisiest) channel.
* Stage A applies **channel-independent 1D temporal convolutional encoders with shared weights**, allowing generalizable temporal filters to learn while decoupling spatial gradient interference.

### 4.2 Stage B: Simplex-Constrained Adaptive Metric Scaling (Invariance Gate)
* Tracks inter-subject loss variance $v_r$ for each cortical region via Exponential Moving Average (EMA, decay $\rho = 0.9$):
  $$m_{r,s} \leftarrow \rho \, m_{r,s} + (1-\rho) \, \ell_{r,s}, \qquad v_r = \Var_{s \in \mathcal{S}_{\text{tr}}}(m_{r,s})$$
* Normalizes variance relative to spatial mean: $s_r = v_r / (\bar{v} + \epsilon)$.
* Maps normalized variances onto a **scaled probability simplex** via temperature-scaled softmax ($\tau = 1.0$):
  $$\gamma_r = R \, \frac{\exp(-s_r / \tau)}{\sum_{r'=1}^R \exp(-s_{r'} / \tau)}, \qquad \text{subject to } \frac{1}{R}\sum_{r=1}^R \gamma_r = 1$$
* **Optimization Role:** Acts as an adaptive preconditioning factor that attenuates unstable regions ($\gamma_r < 1$) and promotes cross-subject invariant channels ($\gamma_r > 1$).

### 4.3 Stage C: Region-Coupled Spectral Decoupling (Theoretical Core)
* Formulates a regularized composite objective function:
  $$L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{SD}} R_{\text{SD}} + \alpha L_{\text{mod}} + \beta L_{\text{reg}}$$
  $$R_{\text{SD}} = \sum_{r=1}^R \frac{1}{\gamma_r} \|\hat{\mathbf{y}}_r\|_2^2 + \|\hat{\mathbf{y}}_{\text{eeg}}\|_2^2 + \|\hat{\mathbf{y}}_{\text{per}}\|_2^2$$
  where the penalty on region logit magnitude is inversely coupled ($1/\gamma_r$) to Stage B invariance.

#### Proposition 1 (Logit Saturation Bound) & Mathematical Proof
> **Proposition:** Let a binary classifier producing logit $f \in \mathbb{R}$ be trained with $L = \text{CE}(\sigma(f), y) + \lambda f^2$, where $\lambda > 0$. Any stationary point $f^*$ satisfies:
> $$|f^*| < \frac{1}{2\lambda}$$

**Proof:**
1. Compute the gradient of $L$ with respect to logit $f$:
   $$\nabla_f L = \frac{\partial}{\partial f}\big[ -y \ln \sigma(f) - (1-y)\ln(1-\sigma(f)) \big] + \frac{\partial}{\partial f}\big[\lambda f^2\big] = 0$$
2. Using $\sigma'(f) = \sigma(f)(1-\sigma(f))$:
   $$\nabla_f L = (\sigma(f) - y) + 2\lambda f = 0$$
3. Solving for the stationary point $f^*$:
   $$f^* = -\frac{\sigma(f^*) - y}{2\lambda}$$
4. For all finite $f^*$, $\sigma(f^*) \in (0, 1)$ and $y \in \{0, 1\}$, implying $|\sigma(f^*) - y| < 1$. Taking absolute values:
   $$|f^*| = \frac{|\sigma(f^*) - y|}{2\lambda} < \frac{1}{2\lambda} \quad \blacksquare$$

* **Optimization Significance:** By strictly bounding $|f^*|$, the cross-entropy gradient $|\nabla_f L_{\text{CE}}| = 2\lambda |f^*| > 0$ **never vanishes**, guaranteeing continuous gradient flow to slower branches throughout training.

### 4.4 Stage D: Latent Manifold Minority Class Recovery
* In held-out subjects, severe class imbalance causes majority-class collapse (standard models predicted the minority class for only $29\%$ of windows).
* Synthesizing raw EEG waveforms directly distorts phase dynamics.
* **Stage D performs interpolation on learned latent tokens** (V-SMOTE, GAMO, DeepSMOTE):
  * **GAMO:** Uses a conditional generator with Gaussian noise dispersion $\to$ achieves peak Balanced Accuracy ($0.5167$), PR-AUC ($0.5237$), and MCC ($0.0398$).
  * **DeepSMOTE:** Employs a bottleneck autoencoder ($d \to 48 \to 32 \to 48 \to d$) $\to$ filters high-frequency noise, achieves peak AUROC ($0.5589$) and best calibration (ECE: $0.2249$).

---

## Act 5: Empirical Results & The Ultimate Convex vs. Non-Convex Lesson

### 5.1 Empirical Verification of Optimization Success
* **AUROC Progression:** $0.4783$ (Base HCAN-CIB) $\to 0.5381$ ($+Z$-score) $\to \mathbf{0.5589}$ ($+$DeepSMOTE).
* **Gini Coefficient ($2 \times \text{AUROC} - 1$):** Flipped from negative ($-0.0434$) to decisively positive ($\mathbf{+0.1178}$).
* **Calibration Error (ECE):** Dropped from $0.306$ to $\mathbf{0.2249}$.
* **Minority Recall:** Rose from $0.2910$ (EEGNet) to $>0.540$ (DEAP) and $\mathbf{0.6385}$ (DREAMER).
* **Gate Interpretability:** Frontal ($\gamma \approx 1.14$) and Parietal ($\gamma \approx 1.10$) promoted; noisy Temporal ($\gamma \approx 0.83$) suppressed.

### 5.2 The Grand Optimization Paradox (The Negative Result)
A simple **$L_2$-regularized Logistic Regression** on $Z$-scored band-limited Differential Entropy (DE) features achieved **$0.5754 \pm 0.0991$ AUROC**, outperforming every deep learning architecture evaluated.

```
       DEAP Benchmark: Convex Floor vs. Deep Non-Convex Architectures
  AUROC
  0.58 ┌─────────────────────────────────────────────────────────────┐
       │                                            [Shallow LR]     │  0.5754 (Convex)
  0.56 │                                              ▲              │
       │                                              │              │
  0.54 │                           [PlainCNN]         │              │  0.5407
       │                             0.5407           │              │
  0.52 │            [EEGNet]      [HCAN-CIB]          │              │  0.5298 (Z-score)
       │             0.5220         0.5298            │              │
  0.50 ┼──────────────────────────────────────────────┼──────────────┤  Chance Level (0.5000)
       └──────────────────────────────────────────────┴──────────────┘
```

### 5.3 Optimization Breakdown: Why Convex Beat Non-Convex

| Axis | Shallow Logistic Regression | Deep Waveform Architectures (EEGNet, ACRNN, Deep CIB) |
| :--- | :--- | :--- |
| **Problem Type** | **Strictly Convex Optimization** | **High-Dimensional Non-Convex Optimization** |
| **Objective Function** | $\min_{\mathbf{w}} \sum \ln(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}_i}) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$ | Non-linear composite composition $\mathcal{L}(f_\theta(\mathbf{x}), y)$ |
| **Parameter Space** | $d = 128$ (DE band features) | $d \approx 10^5 - 10^6$ parameters; $16{,}384$ input samples |
| **Global Optimum** | Unique global minimum guaranteed ($\nabla^2 L \succ 0$) | Spurious local minima, non-isolated saddle points |
| **Hessian Condition Number** | Well-conditioned after $Z$-score normalization | Extremely ill-conditioned ($349.6:1$ subject-to-label variance ratio) |
| **Empirical Result** | **Wins the benchmark ($0.5754$ AUROC)** | Memorizes subject signatures; fails cross-subject transfer |

#### Theoretical Explanation for the Examiner
1. **Hessian Curvature Dominance:** In raw waveform space, subject identity carries $349.6\times$ more variance than emotion labels. The leading eigenvalues of the loss Hessian correspond to subject separation directions.
2. **Greedy First-Order Descent:** Gradient descent takes steps along the directions of maximum curvature (steepest descent). In overparameterized non-convex networks, the optimizer effortlessly minimizes training loss by fitting idiosyncratic subject signatures rather than generalizable emotion boundaries.
3. **Domain Physics as Pre-conditioning:** Band-limited DE ($h(X) = \frac{1}{2}\ln(2\pi e \sigma^2)$) analytically projects $16{,}384$ phase-dependent samples into $128$ phase-invariant power features. Convex optimization on this preconditioned manifold avoids saddle-point entrapment entirely.

---

## Slide-by-Slide Presentation Outline (10–12 Minutes)

| Slide # | Slide Title | Key Mathematical Point | Key Visual / Table |
| :---: | :--- | :--- | :--- |
| **1** | Title & Motivation | Multimodal Optimization under Non-IID Domain Shift | Title, Authors, Abstract Overview |
| **2** | Act 1: The Optimization Pathology | Multi-branch ERM; Asymmetric descent rates; $\nabla_f \mathcal{L} \to 0$ | $19\times - 30\times$ gradient norm bar chart |
| **3** | Act 2: Testing SOTA Modality Optimizers | Formulations of OGM-GE, UDI, PCGrad | Benchmark Table 8 (1,152 models) |
| **4** | Act 2 Analysis: The Orthogonality Discovery | $\langle \mathbf{g}_1, \mathbf{g}_2 \rangle \approx 0$ in $99.88\%$ of steps | Gradient angle pie chart / momentum plot |
| **5** | Act 3: The Paradigm Shift | ANOVA variance decomposition: $349.6 : 1$ ratio | Table 1 (Domain Shift Diagnostics) |
| **6** | Act 4: HCAN-CIB Multi-Stage Optimization | Architecture pipeline overview (Stages A, B, C, D) | System Architecture Diagram (Figure 1) |
| **7** | Act 4: Invariance Gating & Spectral Decoupling | Proposition 1 Statement & Proof ($|f^*| < \frac{1}{2\lambda}$) | Mathematical proof derivation box |
| **8** | Act 4: Latent Minority Class Recovery | Non-linear manifold interpolation (GAMO & DeepSMOTE) | Minority recall jump ($0.29 \to 0.64$) |
| **9** | Act 5: Empirical Benchmark Validation | AUROC $0.478 \to 0.559$; Gini $-0.043 \to +0.118$ | Stepwise Ablation Table 6 & 7 |
| **10** | Act 5: Invariance Gate Convergence | Regional gate hierarchy (Frontal promoted, Temporal suppressed) | Heatmaps (Figures 2 & 3) |
| **11** | Act 5: The Convex vs. Non-Convex Lesson | Strictly Convex LR ($0.5754$) vs. Ill-Conditioned Non-Convex CNNs | Convex vs Non-Convex Comparison Table |
| **12** | Conclusion & Future Work | Min-Max formulation: $\min_\theta \max_\phi \mathcal{L}_{\text{emotion}} - \lambda \mathcal{L}_{\text{subject}}$ | Summary takeaways & GitHub link |

---

## Anticipated Examiner / Professor Questions & Answers

### Q1: "Why could standard AdamW not resolve gradient starvation?"
> **Answer:** AdamW maintains separate running moments per parameter ($\hat{m}_t / \sqrt{\hat{v}_t}$), but it cannot counteract cross-entropy saturation. When $\sigma(f) - y \to 0$, the common scalar coefficient multiplying the entire backpropagated gradient vector becomes zero. Rescaling coordinates cannot revive a gradient vector whose magnitude is uniformly zero.

### Q2: "Can the EMA invariance gate be interpreted as a constrained optimization problem?"
> **Answer:** Yes. Stage B corresponds to an online mirror descent update on a probability simplex $\Delta_R$ under an entropy regularizer:
> $$\max_{\bm{\gamma} \in \Delta_R} \sum_{r=1}^R \gamma_r \mu_r - \tau \sum_{r=1}^R \gamma_r \ln \gamma_r - \beta \sum_{r=1}^R \gamma_r s_r$$
> The softmax update $\gamma_r \propto \exp(-s_r / \tau)$ is the exact closed-form solution to this entropy-regularized objective.

### Q3: "What makes the linear model's objective strictly convex?"
> **Answer:** With $L_2$ regularization, the logistic regression objective is $L(\mathbf{w}) = \sum_i \ln(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}_i}) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$. Its Hessian is:
> $$\nabla^2 L(\mathbf{w}) = \sum_{i=1}^N \sigma(\mathbf{w}^\top \mathbf{x}_i)(1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)) \mathbf{x}_i \mathbf{x}_i^\top + \lambda \mathbf{I} \succeq \lambda \mathbf{I} \succ \mathbf{0}$$
> Because the Hessian has a strictly positive lower eigenvalue bound $\lambda_{\min} \ge \lambda > 0$, the function is strongly convex, guaranteeing a unique global minimizer and linear convergence under gradient descent.
