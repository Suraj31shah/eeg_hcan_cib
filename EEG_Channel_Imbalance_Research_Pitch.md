rge# Channel-Invariance Balancing for EEG Emotion Recognition — Full Research Pitch

## 0. One-line pitch

Multichannel EEG emotion models silently let a few high-SNR channels dominate training via a proven optimization pathology (gradient starvation), and no existing method distinguishes *genuine* neural dominance from *spurious* dominance before "fixing" it, or wires the neuroscience validation into training as a testable signal instead of a decorative figure. This project builds and evaluates a framework — **CIB-Net (Channel-Invariance Balancing Network)** — that does both, and evaluates it under a leakage-free protocol that most of the field currently fails to use.

---

## 1. What the survey actually shows (216-paper systematic review + primary sources)

### 1.1 Timeline of the field
- **2013–2018**: Handcrafted features (differential entropy, band power) + SVM/classical ML. Channel selection = dimensionality reduction, not bias correction.
- **2018–2021**: Deep learning takes over. Channel-wise + self-attention (ACRNN, 2020) lets the network "discover" important channels instead of hand-picking them.
- **2019–2023**: Multimodal fusion (EEG + peripheral signals) matures; "modality imbalance" becomes a named, studied problem — but almost entirely in general multimodal ML (audio-visual, video-text), not EEG. Core techniques: OGM-GE, Greedy, PMR, G-Blending, Shapley-based modality valuation. Most only handle 2 modalities cleanly.
- **2023–2024**: EEG-specific fusion papers (GJFusion) bring channel-level correlation graphs and post-hoc brain-region visualization into the picture, but don't import gradient-level imbalance correction.
- **2024–2026**: EEG foundation models (LaBraM, EEGPT, CBraMod, **BrainMoE**, EEGMamba) treat channel-wise specialization (via Mixture-of-Experts) as a core design axis — capacity, not bias-diagnosis.
- **2025**: A separate, critical thread emerges — **systematic evaluation audits** (Kukhilava et al., 2025, 216-paper review; Lei et al., 2025 on trial/test leakage; a 2026 follow-up on cross-subject protocols) show the field's reported accuracies are largely **not trustworthy** (see §1.2). This is the single most important thing you need to internalize before designing your experiments.

### 1.2 The reproducibility/leakage crisis — read this before you trust any accuracy number
A systematic review of 216 EEG-ER papers (2018–2023) found the field lacks a unified evaluation protocol, with major inconsistencies in ground-truth definition, metric choice, data-splitting strategy (subject-dependent vs. independent), and dataset choice. When the same authors reran standard models (EEGNet, DeepConvNet, ShallowConvNet, TSception) under **rigorous, leakage-free, subject-independent** splits with trivial baselines: improvements over trivial baselines on DEAP and AMIGOS for valence/arousal were **small or non-existent**, and for DREAMER, **no model beat the trivial baseline on arousal at all**. Meanwhile, papers using naive random/subject-dependent splits routinely report 94–99% accuracy on DEAP.

**Practical implication for you:** the "97–99% accuracy" papers you'll find everywhere are very likely leakage-inflated (segment-level random splits leak trial/subject information across train/test). If your paper reports a new SOTA number under the same leaky protocol, reviewers who know this literature (and increasingly, they will) may not take it seriously. If instead you report honest, leakage-free, subject-independent numbers, even a modest absolute accuracy can be a legitimate, defensible contribution **if it clearly beats a trivial baseline and beats prior methods under the same protocol.** This is not a minor footnote — it should shape your entire experimental design (§5).

### 1.3 The three seed papers, and exactly where they stop
- **ACRNN**: channel-wise attention re-weights channels at inference; no gradient-level correction, no genuine/spurious distinction, no neuroscience validation as a training signal.
- **GJFusion**: channel-level correlation graphs + post-hoc brain-region visualization, but for cross-modality (EEG + peripheral) fusion, and validation is qualitative/visual, not a differentiable loss.
- **BrainMoE**: channel-wise Mixture-of-Experts gives channels dedicated capacity, addressing functional heterogeneity — but does not diagnose or correct dominance, and (unstudied so far) MoE routers can silently recreate the same dominance pathology through router collapse.

---

## 2. Why this problem, mathematically — the actual mechanism, not a heuristic

### 2.1 Gradient starvation (root cause of "channel dominance")
Pezeshki et al. (NeurIPS 2021) formalize: under cross-entropy loss, if feature *i* is more strongly correlated with the label than feature *j*, gradient descent's own dynamics — provably, via Neural Tangent Kernel margin analysis — starve the gradient signal to feature *j*, even when *j* is independently informative. Their fix, **Spectral Decoupling (SD)**, adds an L2-style penalty directly on a feature branch's logit output:

```
L_SD = L_CE(y, ŷ) + Σ_k  λ_k · ||z_k||²
```

where `z_k` is the model's output contribution attributable to channel-group *k*, and `λ_k` is a per-group decoupling strength. This decouples the *learning speed* of different channel groups without needing to monitor anything during training — it's a structural fix, not a heuristic monitor-then-modulate loop like OGM-GE. This is your Stage-C mechanism.

### 2.2 Invariant Risk Minimization — separating genuine from spurious dominance
Arjovsky et al. (2019), practical surrogate IRMv1:

```
min_Φ  Σ_e∈E_tr [ R(1·Φ, e) + λ·||∇_{w|w=1} R(w·Φ, e)||²₂ ]
```

Treat each **subject** (or session) as an environment `e`. A channel embedding `Φ` whose gradient-norm penalty is near zero across all subjects is (approximately) an **invariant predictor** — it's predictive of emotion in a way that doesn't depend on which subject you're looking at, which is the operational definition of "genuinely neurally grounded" here. A channel that dominates in only a few subjects but has high penalty (non-invariant) is a strong candidate for **spurious/artifact-driven dominance** (line noise, movement artifact idiosyncratic to a subject's session, electrode-gel drift, etc.). This is your Stage-B diagnostic, and it's what makes your "fix" conditional and defensible rather than a blind rebalancing.

**Important honest caveat** (cite this, don't skip it): IRM has documented theoretical weaknesses — the guarantees in the original paper require many "non-degenerate" environments (effectively as many as your feature dimensionality in the worst case) and don't hold cleanly for non-linear models, and a well-known critique paper ("The Risks of Invariant Risk Minimization") shows IRM can fail even in fairly simple non-linear settings. With ~15–40 subjects as your "environments," you are far below IRM's theoretical comfort zone. Budget for this not working cleanly, and have a fallback: a simpler **cross-subject variance-of-contribution** proxy (just compute the variance of each channel's Shapley/probe-based contribution across subjects, no gradient-norm penalty) as Plan B if full IRMv1 training is unstable — this is much easier to implement and still gives you a genuine/spurious signal, just without IRM's causal framing.

### 2.3 Functional connectivity as an independent structural prior
Phase Locking Value (PLV) between channels *i, j* at frequency band and time window:

```
PLV_ij = |⟨ e^(i·(θ_i(t) − θ_j(t))) ⟩_t|
```

where θ is the instantaneous phase (via Hilbert transform). Build a channel×channel connectivity matrix per trial, derive per-channel **eigenvector centrality** — this gives you a graph-theoretic, non-attention-based, independently computed importance score per channel. This is what Stage D compares your learned importance against, as a **training-time auxiliary loss**, not a post-hoc plot:

```
L_graph = 1 − cos_sim( importance_learned , centrality_PLV )
```

applied selectively (e.g., only where valence-relevant frontal asymmetry is theoretically expected), with the mismatch cases explicitly reported as findings rather than hidden.

---

## 3. The architecture (CIB-Net), tied to the math above

| Stage | What it does | Math from §2 | Solves |
|---|---|---|---|
| A | Per-channel temporal CNN encoder → embeddings | — (infrastructure) | — |
| B | Per-channel invariance scoring across subjects-as-environments | IRMv1 penalty (or variance-of-contribution fallback) | Genuine vs. spurious dominance (P2) |
| C | Region-grouped fusion with Spectral Decoupling, strength gated by Stage-B score | SD penalty, λ_k ∝ 1/invariance_k | Gradient starvation (P1), cross-subject generalization (P3) |
| D | Auxiliary graph-consistency loss vs. PLV-centrality | Cosine loss vs. PLV eigenvector centrality | Training-time neuroscience validation (P4) |
| E (stretch) | Router entropy regularization if backbone is MoE | Standard load-balancing loss (Switch Transformer-style) | MoE routing collapse (P5) |

The dependency that makes this a real hybrid (not kitchen-sink stacking): **Stage C's correction strength is gated by Stage B's output** — you only push gradient toward channels certified as invariant/genuine, not toward every underperforming channel. Stage D is an independent check that doesn't touch the loss used for classification decisions unless you choose to make it a soft auxiliary term.

---

## 4. Step-by-step implementation plan

1. **Reproduce baselines** (ACRNN, plain CNN, EEGNet) on DEAP + SEED, under both (a) the leaky subject-dependent random-split protocol most papers use, and (b) a leakage-free Leave-One-Subject-Out (LOSO) protocol with trivial baselines reported alongside. This dual reporting is itself a small contribution and immediately signals methodological seriousness to reviewers.
2. **Build Stage A + a simple per-channel contribution diagnostic** (Shapley-style or single-channel-probe accuracy ratio). Reproduce the "dominance exists" finding as your Figure 1, under the leakage-free protocol only.
3. **Implement Stage B** — start with the simpler variance-of-contribution proxy (Plan B from §2.2), get it working end-to-end, *then* attempt full IRMv1 gradient-penalty version and compare the two; report whichever is more stable, and report the comparison itself as a finding (this is a legitimate contribution on its own: "does full causal IRM beat a cheap variance proxy for this problem, on N≈15–40 environments?").
4. **Implement Stage C** — Spectral Decoupling per channel-group (frontal/temporal/parietal/occipital), gated by Stage B's score. Ablate: no correction / SD-uncontrolled / SD-gated-by-B.
5. **Implement Stage D** — compute PLV connectivity offline per trial, derive centrality, add as auxiliary loss or purely as evaluation metric (do the pure-eval version first — it's much lower-risk and still gives you the "training-time validation" framing if you report the correlation trend across training epochs, even without back propagating through it).
6. **Full ablation matrix**: baseline vs. +B vs. +C vs. +B+C vs. +B+C+D, under LOSO, on both DEAP and SEED, reporting accuracy/F1/weighted-F1 **and** trivial-baseline deltas per the EEGain-style protocol.
7. **Neuroscience alignment analysis**: for the best model, report per-channel-group importance before/after correction, overlay on a scalp topography plot, and explicitly discuss agreement/disagreement with frontal-asymmetry literature (Davidson-style valence lateralization) — including where it *doesn't* match, as a genuine finding.
8. **Write-up**: position explicitly against ACRNN / GJFusion / BrainMoE with the gap table in §1.3, and against the leakage-crisis papers in §1.2 as your evaluation-methodology justification.

---

## 5. Evaluation protocol (non-negotiable, this is where most papers in this space fail)

- **Primary split**: Leave-One-Subject-Out (LOSO) or subject-independent k-fold. Report subject-dependent numbers only as a secondary comparison against prior literature, explicitly labeled as such.
- **Always report trivial baselines** (majority-class predictor, class-distribution-weighted random predictor) alongside your model, per dataset per task.
- **Datasets**: SEED (62 ch, best for spatial resolution / genuine channel competition) as primary, DEAP (32 ch) as secondary for comparability with the bulk of prior literature. Consider DREAMER as a stress test given no prior model beats trivial baseline on its arousal task under rigorous evaluation — genuinely improving on that would be a strong, citable result.
- **Metrics**: Accuracy, macro-F1, weighted-F1, and **report delta over trivial baseline**, not raw accuracy alone.
- Consider building on / comparing against the open-source **EEGain** framework (Kukhilava et al., 2025) directly, since it's designed exactly for this kind of standardized comparison.

---

## 6. Honest calibration — read this section as carefully as the rest

- "Top 0.01% novel" is not something anyone, including me, can certify in advance — novelty is judged by reviewers against a moving target, and the honest goal is *defensible, well-evidenced novelty*, not a guaranteed ranking.
- What you have here is a genuinely underexplored intersection (gradient-starvation-aware, causally-gated, training-time-neuro-validated channel correction) that no located paper does in combination — but combination-novelty is judged partly on execution quality, not just the idea.
- Full scope (Stages A–E) is realistically a two-paper research program, not a single-semester project. If time-boxed, cut to A+B+C first (a complete, publishable contribution on its own), treat D as evaluation-only initially, and drop E entirely unless you move to an MoE backbone.
- IRM is the highest-risk component technically (see §2.2 caveat) — do not let your whole timeline depend on it working; the variance-proxy fallback is legitimate and should be implemented in parallel from day one, not as a last-resort patch.
- The single highest-leverage, lowest-risk thing you can do for credibility, regardless of which stages you finish, is running everything under the leakage-free LOSO protocol with trivial baselines (§5) — this alone differentiates you from a large fraction of the existing literature.

---

## 7. Key references to read in full (not just abstracts)

- Pezeshki et al., "Gradient Starvation: A Learning Proclivity in Neural Networks," NeurIPS 2021
- Arjovsky et al., "Invariant Risk Minimization," 2019 (+ "The Risks of Invariant Risk Minimization" critique)
- Peng et al. (OGM-GE), "Balanced Multimodal Learning via On-the-fly Gradient Modulation," CVPR 2022
- Kukhilava et al., "Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework," 2025 (arXiv:2505.18175) — and the EEGain framework/codebase
- Lei et al., "Impact of trial-wise and test data leakage on EEG-based emotion classification," 2025
- Tao et al. (ACRNN), Huang et al. (GJFusion), BrainMoE (2025/26) — your three seed papers
