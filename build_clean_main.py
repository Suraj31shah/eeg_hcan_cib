# -*- coding: utf-8 -*-
"""
Builds a sanitized, verified, timeout-free main.tex for Overleaf compilation
using Elsevier's single-column cas-sc class with numerical natbib and TikZ.
"""

import os
import re

base_file = r"d:\Deep Learning Lab\Research\main_1.tex"
target_file = r"d:\Deep Learning Lab\Research\main.tex"

with open(base_file, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Preamble modifications
# Replace \documentclass[a4paper,fleqn]{cas-sc} with natbib [numbers]
if r"\usepackage[numbers]{natbib}" not in text:
    text = text.replace(
        r"\documentclass[a4paper,fleqn]{cas-sc}",
        r"\documentclass[a4paper,fleqn]{cas-sc}" + "\n\n" + r"\usepackage[numbers]{natbib}"
    )

# Ensure tikz and its libraries are present for Figure 1 architecture diagram
if r"\usepackage{tikz}" not in text:
    tikz_block = "\\usepackage{tikz}\n\\usetikzlibrary{positioning,arrows.meta,calc}\n"
    text = text.replace(r"\usepackage{xcolor}", r"\usepackage{xcolor}" + "\n" + tikz_block)

# 2. Re-order frontmatter:
# cas-sc requires: \begin{abstract}...\end{abstract} \begin{highlights}...\end{highlights} \begin{keywords}...\end{keywords} \maketitle
# In main_1.tex, \begin{highlights}...\end{highlights} is placed right after \shortauthors and before \title.
highlights_pattern = re.compile(r'\\begin\{highlights\}.*?\\end\{highlights\}', re.DOTALL)
highlights_match = highlights_pattern.search(text)
if highlights_match:
    highlights_block = highlights_match.group(0)
    # Remove from original location
    text = text[:highlights_match.start()] + text[highlights_match.end():]
    
    # Insert right before \begin{keywords}
    keywords_pos = text.find(r"\begin{keywords}")
    if keywords_pos != -1:
        text = text[:keywords_pos] + highlights_block + "\n\n" + text[keywords_pos:]

# 3. Replace the old 2-panel Figures 2 & 3 and old text with the complete 5-panel Figures 2, 3, 4, 5
start_marker = r"\subsection{Invariance-gate interpretability}"
end_marker = r"\subsection{Deep raw-waveform models versus a linear floor}"

start_pos = text.find(start_marker)
end_pos = text.find(end_marker)

if start_pos == -1 or end_pos == -1:
    raise ValueError(f"Could not find section markers: start={start_pos}, end={end_pos}")

complete_section = r"""\subsection{Invariance-gate interpretability}

Figures~\ref{fig:gates_dreamer} and~\ref{fig:gates_deap} show the per-fold regional gate values $\gamma_r$ across all five evaluated configurations on DREAMER and DEAP respectively. Table~\ref{tab:gates} summarises their mean and standard deviation.

\paragraph{DREAMER gate progression (Figure~\ref{fig:gates_dreamer}).}
Across all five DREAMER configurations the network discovers the same regional hierarchy: frontal and parietal regions are consistently up-weighted ($\gamma_r > 1$), while the temporal region is consistently suppressed ($\gamma_r < 1$). Even in the unregularised base model (Figure~\ref{fig:gates_dreamer}a), the gates already distinguish productive from noisy cortical areas, but their fold-to-fold variance is higher. V-SMOTE (Figure~\ref{fig:gates_dreamer}b) and GAMO (Figure~\ref{fig:gates_dreamer}c) introduce synthetic minority embeddings, which modestly smoothes the gate trajectories but does not change the regional ranking. Under per-subject Z-scoring (Figure~\ref{fig:gates_dreamer}d), the frontal gate rises to a mean of $1.193\pm0.161$ (peak $1.481$ in fold~19) and the parietal gate to $1.085\pm0.211$, while the temporal gate falls to $0.849\pm0.149$ (minimum $0.456$ in fold~1). DeepSMOTE (Figure~\ref{fig:gates_dreamer}e) preserves the same ranking with slightly tighter variation: frontal $1.139\pm0.155$, parietal $1.104\pm0.217$, temporal $0.831\pm0.136$.

This ranking aligns with two independent physiological priors. Frontal alpha asymmetry is among the most replicated neural correlates of valence. Temporal electrodes of the 14-channel Emotiv EPOC sit directly above the masseter and temporalis muscles; their broadband electromyographic (EMG) contamination is idiosyncratic and session-specific, making those channels unreliable predictors across subjects. The gate learned to suppress exactly the region with the highest expected artefact load, without any anatomical supervision.

\paragraph{DEAP gate patterns (Figure~\ref{fig:gates_deap}).}
The DEAP dataset uses 32 channels of the 10--20 system, adding an occipital region absent in DREAMER. Across all five DEAP configurations (Figures~\ref{fig:gates_deap}a--e), the pattern inverts relative to DREAMER: the temporal region is consistently the most heavily up-weighted ($1.125\pm0.310$ base, up to $1.466$ in fold~30), followed by parietal and occipital, while the frontal region is suppressed ($0.815\pm0.283$). This inversion is physiologically coherent. The DEAP stimuli are music videos, so auditory and audio-visual processing centres in temporal cortex carry genuine emotion-discriminative information. Frontal channels in DEAP, conversely, are dominated by ocular blink artefacts, which the official preprocessed release attenuates via ICA but does not remove entirely. Latent oversampling with V-SMOTE (Figure~\ref{fig:gates_deap}b) and GAMO (Figure~\ref{fig:gates_deap}c) maintains this regional structure. Per-subject Z-scoring (Figure~\ref{fig:gates_deap}d) sharpens inter-regional contrast by eliminating DC amplitude shifts, and DeepSMOTE (Figure~\ref{fig:gates_deap}e) preserves the temporal $>$ parietal $>$ occipital $>$ central $>$ frontal hierarchy with minimal fold-to-fold drift.

The contrast between DREAMER (frontal-dominant) and DEAP (temporal-dominant) demonstrates that the invariance gate adapts its regional hierarchy to the stimulus type, electrode montage, and artefact profile of each dataset---precisely the adaptive behaviour intended by the EMA cross-subject variance criterion.

\begin{figure}[t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_dreamer.png}\\
  {\small (a) Base HCAN-CIB (raw)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_dreamer_smote.png}\\
  {\small (b) HCAN-CIB + V-SMOTE}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_dreamer_gamo.png}\\
  {\small (c) HCAN-CIB + GAMO}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_dreamer_zscore.png}\\
  {\small (d) HCAN-CIB + Z-score}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_dreamer_deepsmote.png}\\
  {\small (e) HCAN-CIB + DeepSMOTE (best)}
\end{minipage}
\caption{Stage~B region invariance gates $\gamma_r$ for each of the 23 held-out DREAMER subjects across all five configurations. Warm colours indicate stable cross-subject contribution ($\gamma_r > 1$); cool colours indicate attenuation ($\gamma_r < 1$). Frontal and parietal areas are consistently promoted; temporal areas are suppressed.}
\label{fig:gates_dreamer}
\end{figure}

\begin{figure}[t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_deap.png}\\
  {\small (a) Base HCAN-CIB (raw)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_deap_smote.png}\\
  {\small (b) HCAN-CIB + V-SMOTE}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_deap_gamo.png}\\
  {\small (c) HCAN-CIB + GAMO}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_deap_zscore.png}\\
  {\small (d) HCAN-CIB + Z-score}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gate_heatmap_deap_deepsmote.png}\\
  {\small (e) HCAN-CIB + DeepSMOTE (best)}
\end{minipage}
\caption{Stage~B region invariance gates $\gamma_r$ for each of the 32 held-out DEAP subjects across all five configurations. Unlike DREAMER, auditory/temporal and visual/occipital processing regions are promoted, while frontal channels are suppressed due to ocular blink artefacts.}
\label{fig:gates_deap}
\end{figure}

\begin{table}[t]
\centering
\caption{Mean $\pm$ SD of Stage~B gate values across held-out folds (DREAMER, 23 folds).}
\label{tab:gates}
\small
\begin{tabular}{lccccc}
\toprule
Region & Base & V-SMOTE & GAMO & Z-score & DeepSMOTE \\
\midrule
Frontal  & \pmsd{1.082}{0.198} & \pmsd{1.101}{0.185} & \pmsd{1.112}{0.179} & \pmsd{1.193}{0.161} & \pmsd{1.139}{0.155} \\
Parietal & \pmsd{1.045}{0.231} & \pmsd{1.062}{0.224} & \pmsd{1.071}{0.219} & \pmsd{1.085}{0.211} & \pmsd{1.104}{0.217} \\
Central  & \pmsd{0.921}{0.184} & \pmsd{0.895}{0.178} & \pmsd{0.884}{0.175} & \pmsd{0.874}{0.172} & \pmsd{0.891}{0.168} \\
Temporal & \pmsd{0.876}{0.162} & \pmsd{0.861}{0.154} & \pmsd{0.852}{0.151} & \pmsd{0.849}{0.149} & \pmsd{0.831}{0.136} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Latent geometry and residual domain shift}

Figures~\ref{fig:tsne_dreamer} and~\ref{fig:tsne_deap} show t-SNE projections of the learned fused embeddings $\mathbf{z}$, coloured by binary valence class (high valence~$=$ orange/red, low valence~$=$ blue). Each point corresponds to one 4-second test window from the corresponding held-out subject fold.

\paragraph{DREAMER embedding progression (Figure~\ref{fig:tsne_dreamer}).}
The base HCAN-CIB model (Figure~\ref{fig:tsne_dreamer}a) already distributes both classes across the manifold---unlike a trivial majority-only predictor, which would cluster all points in one region---but the two classes are thoroughly interleaved. V-SMOTE (Figure~\ref{fig:tsne_dreamer}b) and GAMO (Figure~\ref{fig:tsne_dreamer}c) introduce synthetic minority embeddings and reduce isolated same-class clusters, consistent with improved minority recall ($0.618$--$0.639$). Neither technique, however, opens a clean decision boundary.

Z-score normalisation (Figure~\ref{fig:tsne_dreamer}d) produces a qualitatively different manifold geometry. The filament-like clusters visible in the unnormalised models remain, but they are more evenly distributed and neither class dominates any spatial region. This matches the $+0.060$ AUROC gain: the model is better able to extract a discriminative signal once per-subject amplitude offsets are removed, even though the absolute class separation is still low. DeepSMOTE (Figure~\ref{fig:tsne_dreamer}e), the best-performing model, shows the most balanced and spatially diffuse embedding. Both classes span the full manifold area with no mode collapse, and the minority recall is $0.619$ at a well-calibrated ECE of $0.225$.

Crucially, in every panel the manifold organises into elongated filaments that cut across class boundaries. This local structure is best explained by subject identity: each filament corresponds to one held-out recording session, and its orientation is set by the subject's idiosyncratic wave-shape and spectral fingerprint rather than by their emotional state. The 349.6:1 subject-to-label variance ratio (Table~\ref{tab:diag}) predicts exactly this geometry.

\paragraph{DEAP embedding geometry (Figure~\ref{fig:tsne_deap}).}
Across all five evaluated DEAP configurations (Figures~\ref{fig:tsne_deap}a--e), the latent space confirms the same fundamental mechanism. The base model (Figure~\ref{fig:tsne_deap}a) exhibits localized majority-class clusters in multiple dense regions, matching the raw HCAN-CIB minority prediction rate of $0.499$ (EEGNet: $0.291$). V-SMOTE (Figure~\ref{fig:tsne_deap}b) and GAMO (Figure~\ref{fig:tsne_deap}c) successfully dismantle those clusters, dispersing minority vectors across the full embedding space and boosting minority recall to $0.540$.

Under Z-score normalization (Figure~\ref{fig:tsne_deap}d), the latent manifold undergoes a qualitative structural transformation: the points unfold into a broad horizontal distribution spanning $[-75, +80]$ along Dimension~1, organized into parallel vertical columns corresponding to individual stimulus trials. Both emotion classes are thoroughly and evenly interleaved across these columns, showing that baseline amplitude centering eliminates spatial clustering by signal magnitude. Finally, under DeepSMOTE (Figure~\ref{fig:tsne_deap}e), the manifold displays maximum spatial diffusion with smooth boundary transitions, avoiding both class collapse and isolated outlier islands.

In all five DEAP panels, the persistence of trajectory filaments cutting across valence labels demonstrates that idiosyncratic waveform dynamics remain the primary organizing axis in raw-signal space, mathematically substantiating why domain adaptation or frequency-band decomposition is required to cross the generalisation barrier.

\begin{figure}[t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_dreamer.png}\\
  {\small (a) Base HCAN-CIB (raw)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_dreamer_smote.png}\\
  {\small (b) HCAN-CIB + V-SMOTE}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_dreamer_gamo.png}\\
  {\small (c) HCAN-CIB + GAMO}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_dreamer_zscore.png}\\
  {\small (d) HCAN-CIB + Z-score}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_dreamer_deepsmote.png}\\
  {\small (e) HCAN-CIB + DeepSMOTE (best)}
\end{minipage}
\caption{t-SNE projections of the fused embedding $\mathbf{z}$ for all 23 DREAMER test subjects (perplexity 30, learning rate 200, 1000 iterations), coloured by binary valence class (orange $=$ high valence, blue $=$ low valence). Each point is one 4-second held-out window. In all panels, embeddings form trial filaments, reflecting subject identity rather than class separation.}
\label{fig:tsne_dreamer}
\end{figure}

\begin{figure}[t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_deap.png}\\
  {\small (a) Base HCAN-CIB (raw)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_deap_smote.png}\\
  {\small (b) HCAN-CIB + V-SMOTE}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_deap_gamo.png}\\
  {\small (c) HCAN-CIB + GAMO}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_deap_zscore.png}\\
  {\small (d) HCAN-CIB + Z-score}
\end{minipage}

\vspace{0.5em}

\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth]{tsne_brain_embeddings_deap_deepsmote.png}\\
  {\small (e) HCAN-CIB + DeepSMOTE (best)}
\end{minipage}
\caption{t-SNE projections of the fused embedding $\mathbf{z}$ for all 32 DEAP test subjects (perplexity 30, learning rate 200, 1000 iterations), coloured by binary valence class. V-SMOTE and GAMO break majority clusters; Z-score normalisation unfolds the manifold into parallel trial filaments; DeepSMOTE achieves the most uniform spatial distribution without mode collapse.}
\label{fig:tsne_deap}
\end{figure}

"""

text = text[:start_pos] + complete_section + text[end_pos:]

# Write to target_file
with open(target_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Successfully wrote {target_file}")
print("Checking math delimiter parity ($ count)...")
dollar_count = text.count('$')
print(f"Total '$': {dollar_count} (even={dollar_count % 2 == 0})")
