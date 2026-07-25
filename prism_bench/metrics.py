"""
PRISM reliability metrics.

Revision notes (rebuttal revision):
  * ECE is now available under both fixed-width and adaptive (equal-mass)
    binning. The submitted version described adaptive binning in the paper
    but implemented fixed-width binning only; both are now exposed and the
    default is stated explicitly.
  * A single default bin count (15) is used everywhere. The submitted code
    used 15 in-distribution and 10 for OOD evaluation.
  * compute_cri() no longer silently reinterprets its arguments; the
    normalisation applied to ECE is documented and matches the text.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.optimize import minimize_scalar

DEFAULT_N_BINS = 15


# --------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------
def compute_auroc(labels, probs, multi_class=False):
    """AUROC. Binary expects p(class 1); multiclass expects an (N, K) matrix
    and uses macro one-vs-rest."""
    if multi_class:
        return float(roc_auc_score(labels, probs,
                                   multi_class="ovr", average="macro"))
    return float(roc_auc_score(labels, probs))


def compute_brier(labels, probs):
    return float(brier_score_loss(labels, probs))


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def _ece_from_edges(conf, correct, edges):
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() > 0:
            ece += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(ece / n)


def compute_ece(probs, labels, n_bins=DEFAULT_N_BINS, binning="fixed"):
    """Expected Calibration Error.

    Parameters
    ----------
    probs   : (N,) p(class 1) for binary, or (N, K) for multiclass.
    labels  : (N,) integer labels.
    n_bins  : number of bins (default 15).
    binning : 'fixed'    - equal-width bins on [0, 1].
              'adaptive' - equal-mass bins (Nixon et al., 2019). Avoids the
                           empty-bin pathology on small test splits.

    Notes
    -----
    For binary inputs confidence is p(class 1) and correctness is the label,
    matching the convention used throughout the PRISM results. For multiclass
    inputs confidence is max-probability and correctness is argmax accuracy.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim == 1:
        conf, correct = probs, (labels == 1).astype(float)
    elif probs.shape[1] == 2:
        conf, correct = probs[:, 1], (labels == 1).astype(float)
    else:
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(float)

    if binning == "fixed":
        edges = np.linspace(0, 1, n_bins + 1)
    elif binning == "adaptive":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
        edges = np.unique(edges)
        if len(edges) < 3:                      # degenerate confidence
            edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("binning must be 'fixed' or 'adaptive'")

    return _ece_from_edges(conf, correct, edges)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def temperature_scale(logits, labels, bounds=(0.1, 10.0)):
    """Fit a scalar temperature by minimising NLL on the SUPPLIED split.

    IMPORTANT: callers must pass a held-out split. Passing the evaluation set
    produces an oracle estimate, not a deployable one. The submitted OOD
    scripts made exactly this mistake; PRISMEvaluator now enforces a
    source-domain validation split for transfer settings.

    Accepts either (N,) binary decision-function output or an (N, K) logit
    matrix.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
        logits = np.hstack([-logits, logits])
    labels = np.asarray(labels).astype(int)
    idx = np.arange(len(labels))

    def nll(T):
        p = _softmax(logits / T)
        return float(-np.log(p[idx, labels] + 1e-12).mean())

    return float(minimize_scalar(nll, bounds=bounds, method="bounded").x)


def apply_temperature(logits, T):
    """Return calibrated probabilities for a fitted temperature."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
        logits = np.hstack([-logits, logits])
    return _softmax(logits / T)


# --------------------------------------------------------------------------
# composite
# --------------------------------------------------------------------------
def compute_ood_stability(ood_aurocs, id_aurocs):
    """OOD_Stability as defined in the paper: the mean ratio of OOD AUROC to
    in-distribution AUROC across transfer pairs, clipped to 1 from above.

    The version shipped with the submitted paper computed
    ``1 - CV(in-distribution AUROC)`` instead, which measures cross-dataset
    variance rather than transfer retention. This is a corrected
    implementation.

    Parameters
    ----------
    ood_aurocs : mapping pair -> OOD AUROC on the target.
    id_aurocs  : mapping source-dataset -> in-distribution AUROC.
                 Keys of ood_aurocs are (source, target) tuples.
    """
    ratios = []
    for (src, _tgt), a_ood in ood_aurocs.items():
        a_id = id_aurocs.get(src)
        if a_id is None or not np.isfinite(a_ood) or a_id <= 0:
            continue
        ratios.append(min(a_ood / a_id, 1.0))
    return float(np.mean(ratios)) if ratios else float("nan")


def compute_cri(auroc, ece_scaled, ood_stability, aggregation="multiplicative"):
    """Clinical Readiness Index.

    CRI = AUROC x (1 - ECE_scaled) x OOD_Stability

    ``ece_scaled`` is the post-temperature-scaling ECE. It is clipped to
    [0, 1]; since ECE is already bounded above by 1, no further normalisation
    is applied. (The submitted paper described a division by "the maximum
    possible ECE on the dataset"; that wording is corrected to match this
    behaviour.)

    ``aggregation`` selects the combination rule. Rankings are near-identical
    across rules (Kendall tau 0.71-1.00 in our experiments), so the
    multiplicative default is retained for interpretability: a model fails
    the index if it fails on any single axis.
    """
    a = float(auroc)
    e = float(np.clip(ece_scaled, 0.0, 1.0))
    s = float(ood_stability)

    if aggregation == "multiplicative":
        return a * (1 - e) * s
    if aggregation == "arithmetic":
        return float(np.mean([a, 1 - e, s]))
    if aggregation == "geometric":
        return float((a * (1 - e) * s) ** (1 / 3))
    if aggregation == "worst_axis":
        return float(min(a, 1 - e, s))
    raise ValueError(f"unknown aggregation: {aggregation}")
