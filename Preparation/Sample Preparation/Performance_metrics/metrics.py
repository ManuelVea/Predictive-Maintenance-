"""Performance metrics helpers: confusion matrix, accuracy, classification report, and feature importance plotting."""
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"confusion_matrix": cm, "accuracy": acc, "report": report}


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return ax


def plot_feature_importance(importances: np.ndarray, feature_names: List[str], top_n: int = 20, ax=None):
    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in idx]
    top_vals = importances[idx]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_vals)), top_vals[::-1])
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_feats[::-1])
    ax.set_xlabel('Importance')
    return ax
