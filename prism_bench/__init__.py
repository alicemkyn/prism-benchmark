"""
PRISM: Pathology Reliability In Scarce-label Medicine
A benchmark for evaluating pathology foundation model reliability
under label scarcity and domain shift.
"""

from .evaluator import PRISMEvaluator
from .metrics import compute_auroc, compute_ece, compute_brier, compute_cri

__version__ = "0.1.0"
__all__ = ["PRISMEvaluator", "compute_auroc", "compute_ece", "compute_brier", "compute_cri"]
