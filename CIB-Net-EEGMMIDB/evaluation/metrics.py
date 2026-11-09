import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix
)

def compute_all_metrics(y_true, y_pred, y_probs=None):
    """
    Computes all standard metrics matching DREAMER & DEAP audits:
    Accuracy, Balanced Acc, AUROC, Gini AUROC, PR AUC, Macro F1, MCC,
    Delta Trivial, Recall Minority, ECE, Minority Pred Rate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Base classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # 2. Minority & Trivial baselines
    unique, counts = np.unique(y_true, return_counts=True)
    if len(counts) > 1:
        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        majority_count = np.max(counts)
    elif len(counts) == 1:
        majority_class = unique[0]
        minority_class = 1 - unique[0]
        majority_count = counts[0]
    else:
        majority_class, minority_class, majority_count = 0, 1, 0
        
    trivial_base = float(majority_count / max(len(y_true), 1))
    delta_trivial = float(acc - trivial_base)
    
    # Minority prediction rate and recall
    minority_pred_rate = float(np.mean(y_pred == minority_class)) if len(y_pred) > 0 else 0.0
    if (y_true == minority_class).any():
        recall_minority = float(np.mean(y_pred[y_true == minority_class] == minority_class))
    else:
        recall_minority = 0.0
        
    # 3. Probabilistic Metrics
    auroc = 0.50
    gini_auroc = 0.00
    pr_auc = 0.50
    ece = 0.00
    
    if y_probs is not None and len(np.unique(y_true)) == 2:
        try:
            p1 = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
            auroc = float(roc_auc_score(y_true, p1))
            gini_auroc = float(2.0 * auroc - 1.0)
            pr_auc = float(average_precision_score(y_true, p1))
            
            # Expected Calibration Error (ECE) with 10 bins
            bins = np.linspace(0.0, 1.0, 11)
            binids = np.digitize(p1, bins) - 1
            ece_val = 0.0
            for i in range(10):
                mask = (binids == i)
                if np.any(mask):
                    bin_prob = p1[mask]
                    bin_true = y_true[mask]
                    ece_val += (len(bin_prob) / len(y_true)) * np.abs(np.mean(bin_prob) - np.mean(bin_true))
            ece = float(ece_val)
        except Exception:
            pass
            
    return {
        "accuracy": acc,
        "balanced_acc": bal_acc,
        "auroc": auroc,
        "gini_auroc": gini_auroc,
        "pr_auc": pr_auc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "delta_trivial": delta_trivial,
        "recall_minority": recall_minority,
        "ece": ece,
        "minority_pred_rate": minority_pred_rate
    }
