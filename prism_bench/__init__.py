"""PRISM: Pathology Reliability In Scarce-label Medicine."""

from .evaluator import PRISMEvaluator, stratified_sample
from .metrics import (
    compute_auroc, compute_brier, compute_ece,
    temperature_scale, apply_temperature,
    compute_ood_stability, compute_cri,
    DEFAULT_N_BINS,
)

__version__ = "0.2.0"
__all__ = [
    "PRISMEvaluator", "stratified_sample",
    "compute_auroc", "compute_brier", "compute_ece",
    "temperature_scale", "apply_temperature",
    "compute_ood_stability", "compute_cri", "DEFAULT_N_BINS",
]
