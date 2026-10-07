import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, 
    roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss
)

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes rigorous metrics including the trivial baseline delta, AUROC, PR-AUC, ECE, etc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
        
    # 1. Base Metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 2. Per-class & Aggregated F1/Precision/Recall
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Ensure there are two classes (minority=1, majority=0 for convention, or compute dynamically)
    # Actually, we can just save them as lists and figure out majority/minority later
    # Let's assume class 1 is minority if it is imbalanced, else just class 1 and 0
    if len(precision_per_class) == 2:
        precision_0, precision_1 = precision_per_class
        recall_0, recall_1 = recall_per_class
        f1_0, f1_1 = f1_per_class
    else:
        precision_0 = precision_per_class[0] if len(precision_per_class) > 0 else 0
        precision_1 = precision_0
        recall_0 = recall_per_class[0] if len(recall_per_class) > 0 else 0
        recall_1 = recall_0
        f1_0 = f1_per_class[0] if len(f1_per_class) > 0 else 0
        f1_1 = f1_0

    # G-mean: sqrt(Recall_maj * Recall_min)
    g_mean = np.sqrt(recall_0 * recall_1)
    
    # 3. Trivial Baseline & Prediction Rates
    unique, counts = np.unique(y_true, return_counts=True)
    if len(counts) > 0:
        majority_class_count = np.max(counts)
        minority_class_count = np.min(counts)
        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
    else:
        majority_class_count = 0
        trivial_baseline_acc = 0.5
        minority_class = 1
        majority_class = 0

    trivial_baseline_acc = majority_class_count / len(y_true) if len(y_true) > 0 else 0.5
    delta_trivial = acc - trivial_baseline_acc
    
    # Minority prediction rate (how often did we predict the actual minority class?)
    # E.g. if minority class is 1, how many 1s did we predict?
    # Note: if balanced, it doesn't matter much. Let's just track fraction of 1s predicted.
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    pred_dict = dict(zip(pred_unique, pred_counts))
    minority_pred_rate = pred_dict.get(minority_class, 0) / len(y_pred) if len(y_pred) > 0 else 0
    
    results = {
        "accuracy": acc,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "g_mean": g_mean,
        
        "precision_majority": precision_0 if majority_class == 0 else precision_1,
        "recall_majority": recall_0 if majority_class == 0 else recall_1,
        "f1_majority": f1_0 if majority_class == 0 else f1_1,
        
        "precision_minority": precision_1 if minority_class == 1 else precision_0,
        "recall_minority": recall_1 if minority_class == 1 else recall_0,
        "f1_minority": f1_1 if minority_class == 1 else f1_0,
        
        "minority_pred_rate": minority_pred_rate,
        
        "trivial_baseline_acc": trivial_baseline_acc,
        "delta_trivial": delta_trivial
    }
    
    # 4. Probabilistic Metrics
    if y_prob is not None:
        try:
            auroc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            
            # Expected Calibration Error (ECE) - 10 bins
            bins = np.linspace(0., 1., 11)
            binids = np.digitize(y_prob, bins) - 1
            ece = 0.0
            for i in range(10):
                mask = binids == i
                if np.any(mask):
                    bin_prob = y_prob[mask]
                    bin_true = y_true[mask]
                    avg_prob = np.mean(bin_prob)
                    avg_true = np.mean(bin_true)
                    ece += (len(bin_prob) / len(y_true)) * np.abs(avg_prob - avg_true)
                    
        except ValueError:
            # Handle case where y_true only has one class
            auroc = 0.5
            pr_auc = 0.5
            brier = 0.0
            ece = 0.0
            
        results["auroc"] = auroc
        results["gini_auroc"] = 2 * auroc - 1
        results["pr_auc"] = pr_auc
        results["brier_score"] = brier
        results["ece"] = ece
            
    return results

def print_metrics(metrics_dict, fold_name=""):
    prefix = f"[{fold_name}] " if fold_name else ""
    print(f"{prefix}Accuracy: {metrics_dict['accuracy']:.4f} | Trivial Baseline: {metrics_dict['trivial_baseline_acc']:.4f} | Delta: {metrics_dict['delta_trivial']:+.4f}")
    if "auroc" in metrics_dict:
        print(f"{prefix}Macro F1: {metrics_dict['macro_f1']:.4f} | AUROC: {metrics_dict['auroc']:.4f} | Balanced Acc: {metrics_dict.get('balanced_acc', 0):.4f}")
    else:
        print(f"{prefix}Macro F1: {metrics_dict['macro_f1']:.4f} | Weighted F1: {metrics_dict['weighted_f1']:.4f}")
