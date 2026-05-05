# PRISM: Pathology Reliability In Scarce-label Medicine

A benchmark for evaluating pathology foundation model **reliability** under label scarcity and domain shift.

> THUNDER tells you which FM performs best. PRISM tells you which FM you can **trust** most when labels are scarce.

## Installation

```bash
pip install git+https://github.com/alicemkyn/prism-benchmark.git
```

## Quick Start

```python
import numpy as np
from prism_bench import PRISMEvaluator

evaluator = PRISMEvaluator()
results = evaluator.evaluate(
    train_features=X_train,
    train_labels=y_train,
    test_features=X_test,
    test_labels=y_test,
    dataset="pcam",
    model_name="MyModel",
)
print(results.groupby("fraction")[["auroc", "ece", "ece_scaled"]].mean())
```

## CLI

```bash
prism evaluate \
  --train-features X_train.npy --train-labels y_train.npy \
  --test-features X_test.npy --test-labels y_test.npy \
  --dataset pcam --model-name MyModel
```

## Models
CLIP, PLIP, CONCH, VIRCHOW2, UNI, GigaPath, H-Optimus-0, MIDNIGHT

## Datasets
PCam, MHIST, CRC, BRACS, LungHist700, SPIDER-Breast

## Key Findings
1. **Performance-Reliability Decoupling**: High AUROC does not mean well-calibrated
2. **Calibration Inversion**: On LungHist700, ECE worsens as label fraction increases for large models
3. **Scaling Law of Reliability**: T>2.7 for 1B+ models vs T≈2.1 for 86M models
4. **OOD Pair-Dependence**: AUROC drops 0.15-0.57 across transfer pairs

## Pre-computed Embeddings
Will be released on HuggingFace Datasets upon acceptance.

## Citation
```bibtex
@inproceedings{prism2026,
  title={{PRISM}: Pathology Reliability In Scarce-label Medicine},
  author={Anonymous},
  booktitle={NeurIPS Evaluations and Datasets Track},
  year={2026}
}
```

## License
MIT License
