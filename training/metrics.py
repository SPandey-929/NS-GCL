"""Classification metrics for held-out subject evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def classification_metrics(y_true, y_pred, labels=None):
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not y_true.size:
        raise ValueError("y_true and y_pred must be non-empty 1-D arrays of equal shape.")
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    labels = np.asarray(labels, dtype=np.int64)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "class_labels": labels.tolist()}
