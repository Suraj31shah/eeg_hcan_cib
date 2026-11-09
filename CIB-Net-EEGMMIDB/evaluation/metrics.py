import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Computes all standard BCI and classification metrics.
    y_true: [N] 1D array of ground truth labels
    y_pred: [N] 1D array of predicted classes
    y_probs: [N, n_classes] array of predicted class probabilities
    """
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    if y_probs is not None:
        try:
            if y_probs.shape[1] == 2:
                metrics['auroc'] = float(roc_auc_score(y_true, y_probs[:, 1]))
            else:
                metrics['auroc'] = float(roc_auc_score(y_true, y_probs, multi_class='ovr'))
        except Exception:
            metrics['auroc'] = 0.50
    else:
        metrics['auroc'] = 0.50
        
    return metrics

def format_metrics_table(metrics):
    rows = [
        f"Accuracy:          {metrics['accuracy']*100:.2f}%",
        f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%",
        f"Macro F1 Score:    {metrics['macro_f1']:.4f}",
        f"AUROC:             {metrics['auroc']:.4f}",
        f"MCC:               {metrics['mcc']:.4f}"
    ]
    return "\n".join(rows)

