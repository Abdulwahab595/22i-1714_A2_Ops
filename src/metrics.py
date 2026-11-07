"""Threshold selection uses validation labels only; final test scoring is separate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def choose_threshold(labels, probabilities, real_false_positive_cost=2.0):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities).reshape(-1)
    candidates = np.unique(np.r_[0.0, 0.5, probabilities,
                                  np.nextafter(probabilities, np.inf), 1.0])
    candidates = candidates[(candidates >= 0) & (candidates <= 1)]
    best = None
    for threshold in candidates:
        predictions = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        cost = (real_false_positive_cost * fp + fn) / len(labels)
        score = f1_score(labels, predictions, zero_division=0)
        key = (cost, -score, -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), {"weighted_error": float(cost),
                    "real_called_ai": int(fp), "ai_called_real": int(fn),
                    "true_positive": int(tp), "true_negative": int(tn)})
    return best[1], best[2]


def classification_metrics(labels, probabilities, threshold):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities).reshape(-1)
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "samples": int(len(labels)),
            "weighted_error": float((2 * fp + fn) / len(labels))}


def plot_curves(history, output):
    output = Path(output)
    best = int(np.argmin(history["val_loss"])) + 1
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[metric], label="training")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        ax.axvline(best, linestyle="--", color="#444", label=f"best val loss: epoch {best}")
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend(fontsize=8)
    fig.suptitle(f"22i-1714_A2_Ops | {output.name}")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    return best


def save_evaluation(labels, probabilities, threshold, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(labels, probabilities, threshold)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=18)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "AI-generated"],
           yticklabels=["Real", "AI-generated"], xlabel="Predicted", ylabel="Actual",
           title="22i-1714_A2_Ops | Confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate (real called AI)", ylabel="True positive rate",
           title="22i-1714_A2_Ops | ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=150)
    plt.close(fig)
    return metrics
