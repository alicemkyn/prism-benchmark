# PRISM: Pathology Reliability In Scarce-label Medicine

<p align="center">
  <img src="assets/logo_v4.png" alt="PRISM" width="420"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-blue.svg" alt="NeurIPS 2026"/>
  <img src="https://img.shields.io/badge/version-0.2.0-green.svg" alt="version"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/Models-8-purple.svg" alt="8 models"/>
  <img src="https://img.shields.io/badge/Datasets-6-orange.svg" alt="6 datasets"/>
</p>

A benchmark for evaluating pathology foundation model **reliability** under
label scarcity and domain shift.

> [THUNDER](https://github.com/MICS-Lab/thunder) tells you which foundation
> model performs best. PRISM tells you which one you can **trust** when
> labels are scarce and the target population differs from training.

---

## Installation

```bash
# from a local checkout
pip install -e .
```

If you are reading this through an anonymised mirror, the proxy serves a
browsable copy but exposes no git endpoint, so `pip install git+<mirror-url>`
will fail. Download the repository archive first and install from the
extracted directory as above. The package will be published to PyPI on
acceptance.

Requirements: `numpy`, `pandas`, `scikit-learn`, `scipy`. No GPU.

---

## How it works

PRISM evaluates reliability with a frozen-feature linear probe:

1. You extract embeddings once, with your own model and preprocessing.
2. PRISM draws **class-stratified** subsets at six label fractions
   (1%, 5%, 10%, 25%, 50%, 100%) with three seeds (42, 123, 456).
3. At each cell it fits a logistic-regression probe and reports AUROC,
   macro-F1, Brier score, ECE under two binning schemes, the optimal
   temperature fitted on a held-out validation split, and post-scaling ECE.
4. Results can be compared against eight reference foundation models:
   CLIP, PLIP, CONCH, VIRCHOW2, UNI, GigaPath, H-Optimus-0, MIDNIGHT.

Embeddings are extracted once; label fractions are sampled from them. No
retraining and no GPU after extraction.

---

## API

### `PRISMEvaluator`

```python
from prism_bench import PRISMEvaluator

ev = PRISMEvaluator(results_dir=None, n_bins=15)
```

| argument | meaning |
|---|---|
| `results_dir` | path to the reference result CSVs; required for `compare()`, optional for `evaluate()` |
| `n_bins` | ECE bin count, used for both binning schemes |

### `evaluate()`

```python
res = ev.evaluate(
    train_features,          # (N, D) float array
    train_labels,            # (N,)  integer labels
    test_features,           # (M, D)
    test_labels,             # (M,)
    dataset         = "pcam",
    model_name      = "MyModel",
    val_features    = X_val,   # optional, but needed for calibration
    val_labels      = y_val,
    label_fractions = None,    # default [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    seeds           = None,    # default [42, 123, 456]
    C               = 1.0,
    max_iter        = 1000,
)
```

Works on **any** dataset, not only the six benchmark ones.

Returns one row per (fraction, seed):

| column | meaning |
|---|---|
| `auroc` | AUROC; macro one-vs-rest for multiclass |
| `f1_macro` | macro-F1 |
| `brier` | Brier score, binary tasks only |
| `ece_fixed` | ECE, equal-width bins |
| `ece_adaptive` | ECE, equal-mass bins, following Nixon et al. 2019 |
| `temperature` | optimal T, fitted on the validation split |
| `ece_scaled_fixed` | ECE after temperature scaling, equal-width bins |
| `ece_scaled_adaptive` | ECE after temperature scaling, equal-mass bins |
| `degeneracy_share` | share of test samples assigned to the most-predicted class |
| `degenerate` | true when `degeneracy_share > 0.99` |
| `forced_classes` | classes whose stratified quota fell below one sample |
| `n_train`, `n_classes`, `seed`, `fraction` | bookkeeping |

**On the validation split.** Temperature scaling needs held-out data. If you
omit `val_features`, the temperature and both scaled-ECE columns are `NaN`
and a warning is raised; CRI cannot be computed. PRISM never falls back to
fitting the temperature on the evaluation set, since that yields an oracle
estimate rather than a deployable one.

**On degeneracy.** At low label fractions a linear probe often assigns every
test sample to a single class. AUROC can remain informative in that regime
while F1 and ECE describe a degenerate classifier. `degeneracy_share` makes
this visible rather than leaving it implicit.

### `compare()`

```python
cmp = ev.compare(
    custom_results,          # DataFrame returned by evaluate()
    dataset     = "pcam",    # must be one of the six benchmark datasets
    fraction    = 0.10,      # None shows every fraction
    ece_column  = "ece_scaled_fixed",
    aggregation = "multiplicative",
)
```

Returns a table sorted by CRI with your model placed among the eight
reference models. Requires `results_dir`, and requires `dataset` to be one of
the six benchmark datasets, since reference results exist only for those.

A model with no measured transfer performance is scored using the reference
mean OOD stability and flagged in the `ood_stability_imputed` column.

### Standalone functions

```python
from prism_bench import (
    compute_auroc, compute_brier, compute_ece,
    temperature_scale, apply_temperature,
    compute_ood_stability, compute_cri,
)

ece_f = compute_ece(probs, labels, n_bins=15, binning="fixed")
ece_a = compute_ece(probs, labels, n_bins=15, binning="adaptive")

T      = temperature_scale(val_logits, val_labels)   # held-out split only
scaled = apply_temperature(test_logits, T)

stab = compute_ood_stability(
    ood_aurocs = {("pcam", "mhist"): 0.62},   # (source, target) -> OOD AUROC
    id_aurocs  = {"pcam": 0.98},              # source -> in-distribution AUROC
)

cri = compute_cri(auroc, ece_scaled, stab, aggregation="multiplicative")
```

### Metrics

| metric | definition |
|---|---|
| AUROC | area under the ROC curve, macro OvR for multiclass |
| ECE | expected calibration error, 15 bins, fixed-width or equal-mass |
| Brier | Brier score |
| temperature | scalar T minimising NLL on a held-out split |
| OOD_Stability | mean over transfer pairs of OOD AUROC divided by in-distribution AUROC, clipped to 1 |
| CRI | Clinical Readiness Index, AUROC x (1 - ECE_scaled) x OOD_Stability |

`compute_cri` also accepts `arithmetic`, `geometric` and `worst_axis`
aggregations. Rankings agree across all four at Kendall tau between 0.71 and
1.00, so the choice of rule does not drive the conclusions.

---

## Command line

```bash
prism evaluate \
  --train-features X_train.npy --train-labels y_train.npy \
  --test-features  X_test.npy  --test-labels  y_test.npy \
  --val-features   X_val.npy   --val-labels   y_val.npy \
  --dataset pcam --model-name MyModel --output my_results.csv

prism compare \
  --results my_results.csv --results-dir results/ \
  --dataset pcam --fraction 0.10
```

---

## Datasets

| dataset | task | classes |
|---|---|---|
| `pcam` | binary | 2 |
| `mhist` | binary | 2 |
| `crc` | multiclass | 9 |
| `bracs` | multiclass | 7 |
| `lunghist700` | multiclass | 7 |
| `spider_breast` | multiclass | 18 |

Transfer pairs: PCam to MHIST, MHIST to PCam, CRC to BRACS, BRACS to CRC.
The multiclass datasets are binarised for transfer, with CRC using TUM
against the rest and BRACS using IC and DCIS against the rest. These pairs
involve a change of label definition alongside covariate shift and are not
clean covariate-shift benchmarks.

---

## Key findings

1. **Performance and calibration decouple as labels become scarce.** Under
   full supervision, AUROC rank predicts calibration rank consistently across
   all six datasets: pooled Spearman rho 0.57, 95% CI [0.28, 0.76], with
   between-dataset heterogeneity I-squared of 0%. At 1% labels the
   relationship is no longer detectable and becomes dataset-specific: rho
   -0.30, CI [-0.73, 0.31], I-squared 67%. The difference between the two
   regimes is significant, delta rho 0.75, CI [0.19, 1.27].

2. **Calibration inversion.** On LungHist700, ECE worsens for UNI, GigaPath
   and H-Optimus-0 as the label fraction rises from 1% to 100%.

3. **Reverse OOD scaling.** On MHIST to PCam, more source labels produce
   worse target calibration for the four pathology SSL models, with ECE
   rising from roughly 0.23 to between 0.44 and 0.49.

4. **Post-hoc calibration can be harmful under shift.** Fitting the
   temperature on a held-out source validation split increases target ECE for
   all eight models on MHIST to PCam and on CRC to BRACS, and helps only on
   PCam to MHIST.

5. **Calibration ranking is fragile to the probe hyperparameter.** Varying
   the regularisation constant leaves the AUROC ranking essentially intact,
   mean Kendall tau 0.82 with no sign inversions across 36 comparisons, while
   scrambling the calibration ranking, tau 0.02 with 16 inversions.

---

## Repository layout

```
prism-benchmark/
├── prism_bench/
│   ├── metrics.py         AUROC, ECE under both binnings, temperature
│   │                      scaling, OOD stability, CRI
│   ├── evaluator.py       PRISMEvaluator
│   └── cli.py             prism command line
├── notebooks/
│   ├── PRISM_analysis_v2.ipynb        main analysis, seven figures
│   ├── PRISM_OOD_analysis.ipynb       transfer analysis, three figures
│   ├── PRISM_bootstrap_ci.ipynb       bootstrap intervals
│   ├── PRISM_rebuttal_phase1.ipynb    corrected re-computation
│   ├── 00_Setup_Datasets.ipynb        dataset preparation
│   ├── Pcam/ bracs/ crc/ mhist/ lunghist700/ spider_breast/
│   │                                  per-model feature extraction
│   ├── ood_embeddings_notebooks/      transfer evaluation; produced every
│   │                                  reported OOD result
│   └── ood_notebooks/                 superseded GPU implementation, kept
│                                      for completeness; produced none of
│                                      the reported results
├── results/                as submitted
├── results_v2/             corrected re-run, see CHANGELOG_REBUTTAL.md
└── setup.py
```

Both `results/` and `results_v2/` are kept so the two can be compared
directly. Nothing in `results/` has been edited.

---

## Limitations

- `compare()` requires embeddings from one of the six benchmark datasets;
  `evaluate()` works on any dataset.
- Linear probes on frozen features. PEFT and full fine-tuning are outside the
  main protocol.
- Four transfer pairs, each involving a change of label definition alongside
  covariate shift.
- CRI is a screening heuristic, not a validated clinical instrument. Its
  OOD_Stability term is a ratio and therefore partially rewards models with
  weaker in-distribution performance.
- Pre-computed embeddings will be released on Hugging Face Datasets on
  acceptance.

---

## Citation

```bibtex
@inproceedings{prism2026,
  title     = {{PRISM}: Pathology Reliability In Scarce-label Medicine},
  author    = {Anonymous},
  booktitle = {NeurIPS Evaluations and Datasets Track},
  year      = {2026}
}
```

## License

MIT
