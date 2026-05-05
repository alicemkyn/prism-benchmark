# PRISM: Pathology Reliability In Scarce-label Medicine

<p align="center">
  <img src="assets/logo.png" alt="PRISM Logo" width="400"/>
</p>

<p align="center">
  <a href="https://neurips.cc/Conferences/2026"><img src="https://img.shields.io/badge/NeurIPS-2026-blue.svg" alt="NeurIPS 2026"/></a>
  <a href="https://github.com/alicemkyn/prism-benchmark"><img src="https://img.shields.io/badge/pip-prism--bench-green.svg" alt="pip"/></a>
  <a href="https://github.com/alicemkyn/prism-benchmark/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License MIT"/></a>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/Models-8-purple.svg" alt="Models 8"/>
  <img src="https://img.shields.io/badge/Datasets-6-orange.svg" alt="Datasets 6"/>
</p>

A benchmark for evaluating pathology foundation model **reliability** under label scarcity and domain shift.

> [THUNDER](https://github.com/MICS-Lab/thunder) tells you which FM performs best. PRISM tells you which FM you can **trust** most when labels are scarce.

## Installation

```bash
pip install git+https://github.com/alicemkyn/prism-benchmark.git
```

## How It Works

PRISM evaluates model reliability using a linear probe protocol:

1. You extract embeddings from your model for any of the 6 PRISM datasets
2. PRISM trains a logistic regression at 6 label fractions (1%, 5%, 10%, 25%, 50%, 100%) with 3 random seeds
3. At each fraction, PRISM computes AUROC, ECE, Brier score, temperature scaling, and CRI
4. Results are compared against 8 reference models (CLIP, PLIP, CONCH, VIRCHOW2, UNI, GigaPath, H-Optimus-0, MIDNIGHT)

**Key insight:** Embeddings are extracted once. PRISM samples different label fractions automatically - no model retraining needed.

## API Reference

### PRISMEvaluator

```python
from prism_bench import PRISMEvaluator
evaluator = PRISMEvaluator(results_dir=None)
```

`results_dir`: Path to PRISM reference CSVs. Required for `compare()`, optional for `evaluate()`.

### evaluate()

```python
results = evaluator.evaluate(
    train_features,   # np.ndarray (N, D) - training embeddings
    train_labels,     # np.ndarray (N,)   - integer class labels
    test_features,    # np.ndarray (M, D) - test embeddings
    test_labels,      # np.ndarray (M,)   - integer class labels
    dataset,          # str - e.g. "pcam"
    model_name,       # str - display name (default: "CustomModel")
    label_fractions,  # list of floats (default: [0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    seeds,            # list of ints (default: [42, 123, 456])
)
```

Returns `pd.DataFrame` with columns: `model, dataset, fraction, seed, auroc, ece, brier, temperature, ece_scaled`

Works on **any dataset**.

### compare()

```python
comparison = evaluator.compare(
    custom_results,   # pd.DataFrame from evaluate()
    dataset,          # str - must be one of the 6 PRISM datasets
    fraction,         # float or None
)
```

Returns `pd.DataFrame` sorted by CRI, comparing your model against 8 PRISM reference models.

**Requires:** `results_dir` set + embeddings from one of the 6 PRISM benchmark datasets.

**Does NOT work** with custom datasets outside the 6 PRISM datasets.

### Standalone metric functions

```python
from prism_bench import compute_auroc, compute_ece, compute_brier, compute_cri

auroc = compute_auroc(labels, probs)
ece   = compute_ece(probs, labels, n_bins=10)
brier = compute_brier(labels, probs)
cri   = compute_cri(auroc, ece_scaled, ood_stability)
```

### Metrics

| Metric | Description |
|---|---|
| AUROC | Area under ROC curve (macro OvR for multiclass) |
| ECE | Expected Calibration Error (10 bins) |
| Brier | Brier score |
| ece_scaled | ECE after temperature scaling |
| temperature | Optimal temperature T (higher = more overconfident) |
| CRI | Clinical Readiness Index = AUROC x (1 - ECE_scaled) x OOD_Stability |

## CLI Reference

```bash
# Evaluate
prism evaluate \
  --train-features X_train.npy \
  --train-labels   y_train.npy \
  --test-features  X_test.npy  \
  --test-labels    y_test.npy  \
  --dataset        pcam        \
  --model-name     MyModel

# Compare against reference models
prism compare \
  --results      results.csv             \
  --results-dir  /path/to/prism/results  \
  --dataset      pcam                    \
  --fraction     0.1
```

## Supported Datasets

| Dataset | Task | Classes |
|---|---|---|
| pcam | Binary | 2 |
| mhist | Binary | 2 |
| crc | Multiclass | 9 |
| bracs | Multiclass | 7 |
| lunghist700 | Multiclass | 7 |
| spider_breast | Multiclass | 18 |

## Repository Structure

```
prism-benchmark/
├── notebooks/
│   ├── PRISM_analysis_v2.ipynb         # Main analysis, 7 figures
│   ├── PRISM_OOD_analysis.ipynb        # OOD analysis, 3 figures
│   ├── 00_Setup_Datasets.ipynb         # Dataset setup
│   ├── Pcam/                           # 8 model feature extraction notebooks
│   ├── bracs/
│   ├── crc/
│   ├── mhist/
│   ├── lunghist700/
│   ├── spider_breast/
│   └── ood_embeddings_notebooks/       # GPU-free OOD evaluation notebooks
├── prism_bench/
│   ├── __init__.py
│   ├── metrics.py                      # AUROC, ECE, Brier, CRI, temperature scaling
│   ├── evaluator.py                    # PRISMEvaluator class
│   └── cli.py                          # prism CLI
└── setup.py
```

## Key Findings

1. **Performance-Reliability Decoupling**: High AUROC does not mean well-calibrated. On CRC, all models exceed AUROC=0.977 at 1% labels but ECE ranges from 0.18 to 0.36
2. **Calibration Inversion**: On LungHist700, ECE worsens for UNI, GigaPath, H-Optimus-0 as label fraction increases from 1% to 100%
3. **Scaling Law of Reliability**: T>2.7 for 1B+ parameter models vs T~2.1 for 86M models
4. **OOD Pair-Dependence**: AUROC drops of 0.15-0.57 across transfer pairs, largely model-independent

## Limitations

- `compare()` requires embeddings from one of the 6 PRISM benchmark datasets
- `evaluate()` works on any dataset and any embeddings
- Pre-computed embeddings will be released on HuggingFace Datasets upon paper acceptance

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
