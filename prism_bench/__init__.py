"""PRISM: Pathology Reliability In Scarce-label Medicine.

A tile-level benchmark of pathology foundation models: discrimination,
calibration, label scarcity and transfer, summarised by the Composite
Reliability Index (CRI).
"""

__version__ = "0.3.0"

from .evaluator import (
    PRISMEvaluator, stratified_sample, bundled_reference_dir, dataset_key,
    DATASETS, DATASET_DISPLAY, REFERENCE_MODELS, LABEL_FRACTIONS, SEEDS,
)
from .metrics import (
    compute_auroc, compute_brier, compute_ece,
    temperature_scale, apply_temperature,
    compute_ood_stability, compute_cri,
    DEFAULT_N_BINS,
)

__all__ = [
    "PRISMEvaluator", "stratified_sample", "bundled_reference_dir",
    "dataset_key", "DATASETS", "DATASET_DISPLAY", "REFERENCE_MODELS",
    "LABEL_FRACTIONS", "SEEDS",
    "compute_auroc", "compute_brier", "compute_ece",
    "temperature_scale", "apply_temperature",
    "compute_ood_stability", "compute_cri", "DEFAULT_N_BINS",
    "__version__",
]
