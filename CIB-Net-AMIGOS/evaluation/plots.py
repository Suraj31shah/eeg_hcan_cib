import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


DEFAULT_REGION_NAMES = {
    4: ["Frontal", "Temporal", "Parietal", "Occipital"],
    5: ["Frontal", "Temporal", "Central", "Parietal", "Occipital"],
}


def load_results(json_path="results/fold_results.json"):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_roc_pr_curves(y_true_list, y_prob_list, save_dir="results/plots", filename="roc_pr_curves.png"):
    os.makedirs(save_dir, exist_ok=True)
    all_y = np.concatenate(y_true_list)
    all_prob = np.concatenate(y_prob_list)
    if len(np.unique(all_y)) < 2:
        return

    fpr, tpr, _ = roc_curve(all_y, all_prob)
    auc = roc_auc_score(all_y, all_prob)
    prec, rec, _ = precision_recall_curve(all_y, all_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"Pooled ROC (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Aggregated ROC Curve")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color="green", lw=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Aggregated Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close(fig)


def plot_tsne(embeddings, labels, subjects=None, save_path="results/plots/tsne.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n_samples = len(embeddings)
    if n_samples < 5:
        return

    # Cap embeddings for t-SNE speed if very large (>4000 windows)
    if n_samples > 4000:
        rng = np.random.RandomState(42)
        pick = rng.choice(n_samples, 4000, replace=False)
        embeddings = embeddings[pick]
        labels = labels[pick]
        if subjects is not None:
            subjects = subjects[pick]

    perplexity = min(30, max(5, (len(embeddings) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeds_2d = tsne.fit_transform(embeddings)

    if subjects is not None and len(np.unique(subjects)) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sc1 = ax1.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.65, s=12)
        ax1.set_title("t-SNE Colored by Emotion Class")
        plt.colorbar(sc1, ax=ax1)

        sc2 = ax2.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=subjects, cmap="tab20", alpha=0.65, s=12)
        ax2.set_title("t-SNE Colored by Subject ID")
        plt.colorbar(sc2, ax=ax2)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.65, s=12)
        ax.set_title("t-SNE Colored by Emotion Class")
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_region_gate_heatmap(fold_results, save_path="results/plots/gate_heatmap.png", region_names=None):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    gates = [fold["gate_values"] for fold in fold_results if "gate_values" in fold]
    if not gates:
        return

    gates_array = np.asarray(gates, dtype=np.float64)
    n_regions = gates_array.shape[1]
    if region_names is None:
        region_names = DEFAULT_REGION_NAMES.get(n_regions, [f"Region {i+1}" for i in range(n_regions)])

    subj_labels = [
        f"P{fold['test_subject']:02d}" if "test_subject" in fold else f"Fold {i+1}"
        for i, fold in enumerate(fold_results)
        if "gate_values" in fold
    ]

    height = max(6, min(14, 0.28 * len(gates_array) + 2))
    fig, ax = plt.subplots(figsize=(9, height))
    sns.heatmap(
        gates_array,
        cmap="viridis",
        annot=len(gates_array) <= 45,
        fmt=".2f",
        xticklabels=region_names,
        yticklabels=subj_labels,
        ax=ax,
    )
    ax.set_title("Stage B Region Invariance Gates Across LOSO Folds")
    ax.set_xlabel("Brain Region")
    ax.set_ylabel("Held-Out Test Subject")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

