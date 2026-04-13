"""
Evaluation metrics for binary classification.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def threshold_predictions(proba, threshold=0.5):
    """
    Convert probabilities to class predictions.
    
    Args:
        proba: predicted probabilities, shape (1, n_samples)
        threshold: decision threshold
    
    Returns:
        predictions: binary class labels
    """
    proba_flat = proba.flatten()
    return (proba_flat >= threshold).astype(np.int32)


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute classification metrics.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_proba: predicted probabilities (optional)
    
    Returns:
        dict with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def evaluate_model(y_true, y_proba, threshold=0.5):
    """
    Evaluate model on full metrics.
    
    Args:
        y_true: true labels
        y_proba: predicted probabilities
        threshold: decision threshold
    
    Returns:
        dict with all metrics
    """
    y_pred = threshold_predictions(y_proba, threshold)
    return compute_metrics(y_true, y_pred, y_proba)
