from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def compute_topk_accuracy(
    y_true: list[int],
    y_prob: np.ndarray,
    topk: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    results = {}

    if y_prob is None or y_prob.size == 0:
        return results

    n_classes = y_prob.shape[1]
    labels = np.arange(n_classes)

    for k in topk:
        if k <= n_classes:
            results[f"top_{k}_accuracy"] = float(
                top_k_accuracy_score(y_true, y_prob, k=k, labels=labels)
            )

    return results


def build_metrics_summary(
    y_true: list[int],
    y_pred: list[int],
    y_prob: np.ndarray | None = None,
    topk: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

    if y_prob is not None:
        summary.update(compute_topk_accuracy(y_true, y_prob, topk=topk))

    return summary


def build_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> tuple[str, dict]:
    labels = list(range(len(class_names)))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return report_text, report_dict


def build_confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
