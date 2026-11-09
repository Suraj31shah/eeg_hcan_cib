# HCAN-CIB / REGAL-Net: Comprehensive Architecture Stages Guide

> **Core Objective:** Modular architecture designed to resolve three interconnected bottlenecks in cross-subject physiological computing:
> 1. **Subject Domain Shift & Amplitude Bias** ($349.6 : 1$ subject-to-label variance ratio).
> 2. **Modality & Channel Gradient Starvation** ($19\times - 30\times$ gradient norm disparity).
> 3. **Subject-Level Majority Class Collapse** (models predicting minority only $\approx 29\%$ of the time).

---

## End-to-End Pipeline Overview

```
                      RAW INPUTS
       ┌───────────────────────────────┐
       │ EEG Window: B x C x 512       │ Peripheral Descriptors: B x d_p
       └──────────────┬────────────────┘                │
                      ▼                                 │
     [ Transductive Per-Subject Z-Scoring ]             │
                      │                                 │
                      ▼                                 ▼
       ┌───────────────────────────────┐        ┌───────────────┐
       │ STAGE A: Channel 1D-CNNs      │        │ Peripheral    │
       │ (Weight-shared across C ch)   │        │ Feature MLP   │
       └──────────────┬────────────────┘        └───────┬───────┘
                      │ {h_c} : B x C x 64              │
                      ▼                                 │
       ┌───────────────────────────────┐                │
       │ STAGE D: Latent Oversampling  │◄───────────────┘ (Train only)
       │ (V-SMOTE / GAMO / DeepSMOTE)  │
       └──────────────┬────────────────┘
                      │ Synthetic balance
                      ▼
       ┌───────────────────────────────┐
       │ STAGE C: Region Grouping      │
       │ (Attention over electrodes)   │
       └──────────────┬────────────────┘
                      │ u_r : B x R x 64
                      ▼
       ┌───────────────────────────────┐
       │ STAGE B: Invariance Gating    │
       │ (EMA cross-subject variance)  │
       └──────────────┬────────────────┘
                      │ u_r_tilde = γ_r * u_r
                      ▼
       ┌───────────────────────────────┐
       │ Cross-Modal Attention Neck    │◄── Peripheral Query Token q
       │ + Region Spectral Decoupling  │
       └──────────────┬────────────────┘
                      ▼
                 Final Logit
```

---

## 0. Preliminary Stage: Transductive Per-Subject $Z$-Scoring

* **What it does:**  
  Before feeding windows into the network, each recording session is normalized independently across time for each channel $c$:
  $$\tilde{x}_{s,c,t} = \frac{x_{s,c,t} - \mu_{s,c}}{\sigma_{s,c} + 10^{-8}}$$
* **The Problem it Solves (Subject Amplitude Bias):**  
  Skull thickness, electrode impedances, and resting alpha power vary dramatically across subjects. In raw differential entropy (DE) features, **subject identity explains $349.6\times$ more variance than emotional valence** ($47.75\%$ vs. $0.137\%$).
* **Optimization Impact:**  
  Centering and scaling each subject eliminates the massive DC broadband power offsets that create elongated, ill-conditioned canyons in the loss surface. It drops subject-decoding accuracy from $97.81\%$ down to chance ($3.125\%$).

---

## Stage A: Channel-Independent Temporal Encoders

* **What it does:**  
  Instead of mixing all electrodes immediately, each electrode signal $\mathbf{x}_c \in \mathbb{R}^{512}$ is passed through an isolated 1D temporal convolutional encoder $\phi$:
  $$\mathbf{h}_c = \text{GAP}\Big(\sigma\big(\text{BN}(\mathbf{w}_3 * \text{AAP}_{64}(\sigma(\text{BN}(\mathbf{w}_2 * \sigma(\text{BN}(\mathbf{w}_1 * \mathbf{x}_c))))))\big)\Big) \in \mathbb{R}^{64}$$
  * Convolutions use kernel sizes $k \in \{25, 15, 7\}$ (capturing slow $\delta/\theta$ rhythms down to fast $\beta/\gamma$ oscillations).
  * Feature channel dimensions across layers: $\{32, 64, 64\}$.
  * $\text{AAP}_{64}$ denotes Adaptive Average Pooling downsampling the sequence length to 64; $\text{GAP}$ denotes Global Average Pooling across time.
  * **Crucial detail:** The convolutional weights $\phi$ are **strictly shared** across all $C$ channels ($C=32$ for DEAP, $C=14$ for DREAMER).
* **The Problem it Solves (Early Spatial Gradient Coupling):**  
  Standard deep models (like EEGNet) apply a depthwise spatial filter at Layer 1 that immediately mixes all channels into a handful of virtual nodes. This couples the parameter gradient of every electrode to the loudest, highest-amplitude channel (often an eye-blink or muscle artifact).
* **Optimization Impact:**  
  Decouples spatial parameter gradients. It forces the 1D temporal filters to learn universal oscillatory dynamics that hold true across *any* scalp location before spatial fusion begins.

---

## Stage D: Latent-Space Minority Oversampling *(Training Only)*

* **What it does:**  
  Takes the extracted intermediate channel tokens $\mathbf{H} = [\mathbf{h}_1; \dots; \mathbf{h}_C] \in \mathbb{R}^{C \times 64}$ and peripheral vector $\mathbf{p}$ and synthesizes new minority-class training representations inside the latent space:
  1. **V-SMOTE (Vector SMOTE):** Non-parametric interpolation along linear chords between 5-nearest minority neighbors in Euclidean latent space:
     $$\tilde{\mathbf{h}}_{\text{syn}} = \tilde{\mathbf{h}}_i + \lambda (\tilde{\mathbf{h}}_{nn} - \tilde{\mathbf{h}}_i), \qquad \lambda \sim \mathcal{U}(0, 1)$$
  2. **GAMO (Generative Adversarial Minority Oversampling):** A conditional generator MLP perturbed with Gaussian noise $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ that generates non-linear stochastic dispersions into local density voids around minority exemplars.
  3. **DeepSMOTE:** Compresses latent tokens through a non-linear autoencoder bottleneck ($d \to 48 \to 32 \to 48 \to d$) and performs $k$-NN interpolation inside the compressed 32-dimensional noise-filtered subspace.
* **The Problem it Solves (Waveform Distortion & Majority Collapse):**  
  Applying standard SMOTE on raw EEG waveforms is mathematically and biologically invalid: linearly blending two out-of-phase oscillatory waveforms causes destructive phase cancellation, creating artificial high-frequency boundary transients that real brains never produce. Conversely, doing nothing leads to majority-class collapse (EEGNet predicted the minority class only $29\%$ of the time).
* **Optimization Impact:**  
  Operates on the **compact feature manifold** after noise filtering. It balances mini-batch class priors to exact 1:1 parity, recovering minority recall from $0.29$ up to $>0.54$ (DEAP) and $0.6385$ (DREAMER) without distorting raw physiological dynamics.

---

## Stage C: Region Grouping & Intra-Region Attention

* **What it does:**  
  Partitions the electrodes into anatomical cortical regions based on the standard 10–20 international system:
  * **DEAP ($R=5$):** Frontal (7 channels), Temporal (4), Central (5), Parietal (4), Occipital (4).
  * **DREAMER ($R=4$):** Frontal (8 channels), Temporal (2), Parietal (2), Occipital (2).
  
  Inside each region $G_r$, an attention pooling mechanism learns which specific electrodes matter most for that region:
  $$a_c = \frac{\exp\big(\mathbf{v}_r^T \tanh(\mathbf{W}_r \mathbf{h}_c)\big)}{\sum_{c' \in G_r} \exp\big(\mathbf{v}_r^T \tanh(\mathbf{W}_r \mathbf{h}_{c'})\big)}, \qquad \mathbf{u}_r = \sum_{c \in G_r} a_c \mathbf{h}_c \in \mathbb{R}^{64}$$
  *(During training, entire regions are randomly dropped with probability $p=0.2$ to prevent co-adaptation).*
* **Coupled with Spectral Decoupling ($R_{\text{SD}}$):**  
  Stage C introduces intermediate auxiliary classification heads $\hat{\mathbf{y}}_r$ for each region and penalizes their squared $L_2$ norm:
  $$L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{SD}} R_{\text{SD}} + \alpha L_{\text{mod}} + \beta L_{\text{reg}}$$
  $$R_{\text{SD}} = \sum_{r=1}^R \frac{1}{\gamma_r} \|\hat{\mathbf{y}}_r\|_2^2 + \|\hat{\mathbf{y}}_{\text{eeg}}\|_2^2 + \|\hat{\mathbf{y}}_{\text{per}}\|_2^2$$
* **Optimization Proof (Proposition 1):**  
  Setting the gradient of $L = \text{CE}(\sigma(f), y) + \lambda f^2$ to zero yields:
  $$\nabla_f L = (\sigma(f) - y) + 2\lambda f = 0 \implies |f^*| < \frac{1}{2\lambda}$$
  **Why this matters:** Unbounded logits normally saturate cross-entropy ($\sigma(f) - y \to 0$). Bounding the logit prevents gradient saturation, guaranteeing that the auxiliary heads continuously pull gradients into intermediate cortical layers.

---

## Stage B: Dynamic Regional Invariance Gating

* **What it does:**  
  Evaluates how reliably each cortical region generalizes across training subjects on the fly:
  1. Computes the auxiliary loss $\ell_{r,s}$ of region $r$ on the mini-batch data of subject $s$.
  2. Maintains a running Exponential Moving Average (EMA, decay $\rho=0.9$) of subject-specific loss:
     $$m_{r,s} \leftarrow \rho \, m_{r,s} + (1-\rho) \, \ell_{r,s}$$
  3. Computes the cross-subject loss variance $v_r = \Var_{s \in \mathcal{S}_{\text{tr}}}(m_{r,s})$ and normalizes it by the spatial mean: $s_r = v_r / (\bar{v} + \epsilon)$.
  4. Maps normalized variances onto a **scaled probability simplex** using temperature softmax ($\tau=1.0$):
     $$\gamma_r = R \, \frac{\exp(-s_r / \tau)}{\sum_{r'=1}^R \exp(-s_{r'} / \tau)}, \qquad \tilde{\mathbf{u}}_r = \gamma_r \mathbf{u}_r$$
* **The Problem it Solves (Subject Artifact Sensitivity):**  
  Some cortical areas are severely corrupted by subject-specific noise (e.g., temporal electrodes pick up jaw clenching; frontal electrodes pick up blinks). Standard models overfit to these subject-specific artifacts.
* **Optimization Role:**  
  Acts as an **adaptive metric preconditioner**:
  * Unstable regions across subjects have high loss variance $\implies s_r > 1 \implies \gamma_r < 1$ (attenuated).
  * Consistent regions across subjects have low variance $\implies s_r < 1 \implies \gamma_r > 1$ (amplified).
  * Because $\gamma_r$ enters Stage C's regularizer as $1/\gamma_r$, unstable regions simultaneously receive a harsher penalty on their representation norms!
* **Neurophysiological Findings:**  
  * On **DREAMER** (audiovisual film clips): Frontal ($\gamma \approx 1.14$) and Parietal ($\gamma \approx 1.10$) were promoted, while Temporal ($\gamma \approx 0.83$, masseter muscle noise) was heavily suppressed.
  * On **DEAP** (music videos): Temporal ($\gamma \approx 1.13$, auditory processing) was promoted, while Frontal ($\gamma \approx 0.81$, blink noise) was suppressed.

---

## Cross-Modal Attention Neck & Output Fusion

* **What it does:**  
  Fuses the gated EEG cortical tokens with the peripheral descriptors:
  1. The peripheral vector $\mathbf{p}$ is mapped to a query token: $\mathbf{q} = \text{MLP}(\mathbf{p}) \in \mathbb{R}^{64}$.
  2. The query attends over the set of gated region tokens $\tilde{\mathbf{U}} = [\tilde{\mathbf{u}}_1; \dots; \tilde{\mathbf{u}}_R]$:
     $$\mathbf{m} = \softmax\left(\frac{(\mathbf{W}_Q \mathbf{q})(\mathbf{W}_K \tilde{\mathbf{U}})^T}{\sqrt{d}}\right) \mathbf{W}_V \tilde{\mathbf{U}}$$
  3. The attended context $\mathbf{m}$ is concatenated with the query $\mathbf{q}$ to form fused embedding $\mathbf{z} = [\mathbf{m}; \mathbf{q}] \in \mathbb{R}^{128}$.
  4. **Residual Shallow Prior:** A linear skip connection adds a direct linear projection of the peripheral features $\mathbf{w}_s^T \mathbf{p}$ directly to the final classification logit.
* **Optimization Role:**  
  The residual peripheral prior establishes a performance floor: the non-linear multimodal network is guaranteed to perform *at least as well* as an optimal linear peripheral model, preventing the deep fusion neck from degrading below simple physiological baselines.

---

## Summary Matrix of All Stages

| Stage | Name | Input $\to$ Output | Core Problem Solved | Optimization Mechanism |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Transductive Z-Score** | Raw Signals $\to$ Normalized | Amplitude bias across people ($349.6:1$ variance ratio) | Eliminates DC shifts; well-conditions input space |
| **A** | **Channel 1D-CNNs** | $B \times C \times 512 \to B \times C \times 64$ | Early spatial gradient coupling | Weight sharing across channels; decouples parameter gradients |
| **D** | **Latent Oversampling** | Minority Tokens $\to$ 1:1 Parity | Subject-level majority collapse; raw waveform distortion | Noise-filtered manifold interpolation (GAMO / DeepSMOTE) |
| **C** | **Region Attention + SD** | $C$ channels $\to R$ region tokens | Spatial aggregation; gradient starvation in joint loss | Auxiliary heads + Proposition 1 bound ($|f^*| < \frac{1}{2\lambda}$) |
| **B** | **Invariance Gate ($\gamma_r$)** | Regional Tokens $\to \gamma_r \mathbf{u}_r$ | Non-generalizing, noisy electrodes | EMA loss variance tracking; simplex-constrained preconditioning |
| **Neck**| **Cross-Modal Attention**| Gated EEG + Periph $\to$ Logit | Modality asymmetry & fusion bottlenecks | Peripheral query attention + residual linear prior floor |

