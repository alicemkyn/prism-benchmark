"""
PRISM reliability metrics: AUROC, ECE, Brier, CRI.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.optimize import minimize_scalar


def compute_auroc(labels, probs, multi_class=False):
    """Compute AUROC. Supports binary and multiclass (OvR macro)."""
    if multi_class:
        return roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return roc_auc_score(labels, probs)


def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        probs: predicted probabilities for positive class, shape (N,)
        labels: binary ground truth labels, shape (N,)
        n_bins: number of calibration bins

    Returns:
        float: ECE value
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compute_brier(labels, probs):
    """Compute Brier score."""
    return float(brier_score_loss(labels, probs))


def temperature_scale(logits, labels):
    """
    Find optimal temperature T via NLL minimization.

    Args:
        logits: raw decision function output, shape (N,)
        labels: binary ground truth labels, shape (N,)

    Returns:
        float: optimal temperature
    """
    def nll(T):
        p = np.clip(1 / (1 + np.exp(-logits / T)), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def compute_cri(auroc, ece_scaled, ood_stability):
    """
    Compute Clinical Readiness Index (CRI).

    CRI = AUROC * (1 - ECE_scaled) * OOD_Stability

    Args:
        auroc: float in [0, 1]
        ece_scaled: ECE after temperature scaling, float in [0, 1]
        ood_stability: 1 - CV(AUROC across datasets), float in [0, 1]

    Returns:
        float: CRI score
    """
    ece_clamped = min(max(float(ece_scaled), 0.0), 1.0)
    return float(auroc * (1 - ece_clamped) * ood_stability)
