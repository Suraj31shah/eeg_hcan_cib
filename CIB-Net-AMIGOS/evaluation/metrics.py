import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes classification + calibration metrics including trivial baseline delta,
    per-role (majority / minority) precision/recall/F1, AUROC, PR-AUC, Brier, and ECE.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else float("nan")
    balanced_acc = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) >= 1 and len(y_true) > 0
        else float("nan")
    )
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prev1 = float(y_true.mean()) if len(y_true) > 0 else 0.5
    minority_class = 1 if prev1 <= 0.5 else 0
    majority_class = 1 - minority_class

    def _prf(c: int):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return float(p), float(r), float(f)

    p_min, r_min, f_min = _prf(minority_class)
    p_maj, r_maj, f_maj = _prf(majority_class)
    g_mean = float(np.sqrt(r_min * r_maj))

    trivial_baseline_acc = max(prev1, 1.0 - prev1)
    delta_trivial = acc - trivial_baseline_acc
    minority_pred_rate = float((y_pred == minority_class).mean()) if len(y_pred) > 0 else 0.0

    results = {
        "accuracy": acc,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "g_mean": g_mean,
        "precision_majority": p_maj,
        "recall_majority": r_maj,
        "f1_majority": f_maj,
        "precision_minority": p_min,
        "recall_minority": r_min,
        "f1_minority": f_min,
        "minority_class": int(minority_class),
        "minority_pred_rate": minority_pred_rate,
        "trivial_baseline_acc": float(trivial_baseline_acc),
        "delta_trivial": float(delta_trivial),
    }

    if y_prob is not None:
        p_clip = np.clip(np.nan_to_num(y_prob, nan=0.5), 0.0, 1.0)
        if len(np.unique(y_true)) == 2:
            auroc = float(roc_auc_score(y_true, p_clip))
            pr_auc = float(average_precision_score(y_true, p_clip))
            gini_auroc = 2.0 * auroc - 1.0
        else:
            auroc = float("nan")
            pr_auc = float("nan")
            gini_auroc = float("nan")

        brier = float(brier_score_loss(y_true, p_clip)) if len(y_true) > 0 else float("nan")
        edges = np.linspace(0.0, 1.0, 11)
        idx = np.clip(np.digitize(p_clip, edges[1:-1]), 0, 9)
        ece = 0.0
        for b in range(10):
            m = idx == b
            if m.any():
                ece += float(m.mean() * abs(y_true[m].mean() - p_clip[m].mean()))

        results["auroc"] = auroc
        results["gini_auroc"] = gini_auroc
        results["pr_auc"] = pr_auc
        results["brier_score"] = brier
        results["ece"] = float(ece)

    return results


def print_metrics(metrics_dict, fold_name: str = ""):
    prefix = f"[{fold_name}] " if fold_name else ""
    print(
        f"{prefix}Acc: {metrics_dict['accuracy']:.4f} | Trivial: {metrics_dict['trivial_baseline_acc']:.4f} "
        f"| Delta: {metrics_dict['delta_trivial']:+.4f}"
    )
    if "auroc" in metrics_dict:
        print(
            f"{prefix}Macro F1: {metrics_dict['macro_f1']:.4f} | AUROC: {metrics_dict['auroc']:.4f} "
            f"| Bal Acc: {metrics_dict.get('balanced_acc', 0.0):.4f}"
        )
    else:
        print(
            f"{prefix}Macro F1: {metrics_dict['macro_f1']:.4f} "
            f"| Weighted F1: {metrics_dict['weighted_f1']:.4f}"
        )

