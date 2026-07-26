# Changelog — rebuttal revision

This file documents every change made to the codebase after the reviews were
received. Reviewers inspected the submitted version; this record exists so
that any difference they observe is accounted for. The submitted code remains
in the git history and the original result files are untouched in `results/`.
Corrected outputs are written to `results_v2/` alongside them, so both can be
compared directly.

---

## 1. OOD temperature scaling was fitted on target test labels

**Reported by Reviewer tp5b.** Correct.

Submitted (`notebooks/ood_embeddings_notebooks/*.ipynb`):

```python
T = temperature_scale_binary(logits, tgt_int)   # tgt_int = target TEST labels
```

The temperature was optimised against the labels of the evaluation set,
which is target-test leakage. Every scaled-ECE value in the OOD experiments
is therefore an oracle estimate rather than a deployable one.

**Fix.** The temperature is now fitted on a held-out *source-domain*
validation split and applied to the target without access to target labels.
`prism_bench.metrics.temperature_scale` documents this requirement, and
`PRISMEvaluator.evaluate` returns `NaN` rather than silently falling back to
the evaluation set when no validation split is supplied.

**Effect on reported results.** None of the tables in the paper report scaled
OOD ECE; Table 12 reports raw ECE, which does not involve the temperature.
The raw values are unchanged. The corrected run additionally shows that
source-fitted temperature scaling *increases* OOD ECE on 16 of 32
(model, pair) combinations, which strengthens rather than weakens the
paper's conclusion. Both the corrected and the original oracle values are
recorded side by side in `results_v2/ood_all_v2.csv` as
`ece_scaled_fixed` and `ece_scaled_oracle`.

## 2. ECE binning did not match its description

**Reported by Reviewer tp5b.** Correct.

The paper describes adaptive bin widths following Nixon et al. The submitted
code used fixed-width bins via `np.linspace`, with 15 bins in-distribution
and 10 bins for OOD evaluation.

**Fix.** `compute_ece` now accepts `binning='fixed'` or `binning='adaptive'`
(equal-mass) and a single default bin count of 15 is used throughout. All
corrected results are reported under both schemes so that conclusions can be
checked for sensitivity to the choice.

## 3. Subset sampling was class-agnostic in-distribution

**Reported by Reviewer tp5b.** Correct.

In-distribution experiments drew label-fraction subsets with
`np.random.choice` over all training indices, without stratification, while
the OOD experiments used stratified sampling. On LungHist700 (7 classes,
483 training tiles) the 1% subset is roughly five samples.

**Fix.** `stratified_sample` is now used in every setting. It also returns
the set of classes for which the exact quota fell below one sample and a
single example had to be forced, which is recorded as `forced_classes`.

**Effect.** Mean absolute change in AUROC is 0.011 at 1% labels, 0.005 at
10%, and exactly zero at 100% (where the subset is the full training set
regardless of sampling scheme). Every table reported at full supervision is
unchanged.

## 4. Degenerate probes were not reported

**Not raised explicitly, but implied by Reviewer tp5b's hypothesis about
sampling artefacts.**

At low label fractions the linear probe frequently assigns every test sample
to a single class. On MHIST at 1% labels this occurs for all eight foundation
models; on the MHIST→PCam transfer it occurs for seven of eight at every
label fraction. The submitted paper did not report this, and the "breaking
point" analysis is measuring the point at which the probe begins predicting
the minority class.

**Fix.** Every result row now carries `degeneracy_share`, the proportion of
test samples assigned to the most-predicted class, and a `degenerate` flag at
a threshold of 0.99. `PRISMEvaluator.evaluate` emits a warning when collapse
is detected. A full audit is in `results_v2/degeneracy_audit.csv`.

## 5. `OOD_Stability` in the package did not implement Equation 1

**Not raised by reviewers.** Found during our own audit.

Submitted `evaluator.py`:

```python
cv = np.std(aurocs) / (np.mean(aurocs) + 1e-10)
stab = max(0, 1 - cv)
```

This is one minus the coefficient of variation of the *in-distribution* AUROC
across datasets. The paper defines OOD_Stability as the ratio of OOD AUROC to
in-distribution AUROC, averaged over transfer pairs and clipped to 1. These
are different quantities.

**Fix.** `metrics.compute_ood_stability` implements the definition in the
paper, and `PRISMEvaluator` reads the shipped OOD result files to compute it.

**Effect.** The CRI ranking changes. Under the corrected definition VIRCHOW2
still leads, but CONCH moves from first to sixth at 1% labels, and CLIP rises
to second because a ratio metric rewards models whose in-distribution
performance was already low. We report this and discuss it as a limitation
rather than adjusting the definition after the fact.

## 6. F1 was binary in-distribution and macro for OOD

**Not raised by reviewers.** Found during our own audit.

The in-distribution notebooks called `f1_score(y_true, y_pred)`, which
defaults to binary averaging, while the OOD scripts passed
`average='macro'`. The paper describes macro-F1 in both places.

**Fix.** Macro-F1 everywhere. Binary F1 is retained as a separate column for
the two binary datasets so the submitted numbers remain traceable.

## 7. `ECE_scaled` normalisation described incorrectly

**Not raised by reviewers.** Found during our own audit.

The paper states that the post-temperature ECE is "rescaled to [0,1] via
division by the maximum possible ECE on the dataset". The code clips to
[0,1] without dividing. Since ECE is already bounded above by 1, the two are
equivalent in effect.

**Resolution.** The code is correct and unchanged; the *paper text* is being
corrected to describe clipping.

## 8. Documented API did not exist

**Reported indirectly by Reviewer fWEj**, who could not install the package.

Appendix E of the submitted paper shows three calls, none of which are valid
signatures:

```python
PRISMEvaluator(datasets=[...], label_fractions=[...])
evaluator.evaluate(my_model, seeds=[...])
evaluator.compare(results, reference=[...])
```

**Resolution.** The appendix is being rewritten to document the real API,
rather than the package rewritten to match the appendix. The real API takes
embeddings rather than a model, which is what allows the benchmark to run
without a GPU; changing it to accept models would remove that property.

`evaluate` gains `val_features` and `val_labels` parameters, which the
submitted signature lacked and which are required for temperature scaling.

## 9. Installation instructions did not work

**Reported by Reviewer fWEj**, who received:

```
ERROR: Failed to build 'git+https://anonymous.4open.science/r/prism-benchmark-0108.git'
```

The anonymisation proxy serves a browsable mirror but exposes no git
endpoint, so `pip install git+...` cannot clone from it. This is a property
of the proxy, not of the package.

**Fix.** README now instructs reviewers to download the repository archive
and run `pip install -e .`, which we verified in a clean environment. The
package will be published to PyPI at camera-ready; publishing it during
review would compromise anonymity.

## 10. Preprocessing: which notebooks produced the results

**Reported by Reviewer tp5b.**

The repository contains two OOD implementations.
`notebooks/ood_notebooks/` is an earlier GPU-based version that applies a
uniform ImageNet transform. It was superseded before any results were
produced and is retained only for completeness.
`notebooks/ood_embeddings_notebooks/` is what actually generated every OOD
result file; those notebooks import no imaging libraries at all and operate
purely on cached embeddings.

The preprocessing discrepancy the reviewer identified is therefore in a
script that produced none of the reported numbers. We keep both folders in
place so this can be verified directly.

Separately, and not raised by any reviewer: the paper's claim that each model
was used through its own native preprocessing pipeline is inaccurate for the
*in-distribution* extraction as well. A single shared transform (resize 224,
centre crop, ImageNet normalisation) was applied to all models, and the
resulting features were L2-normalised, which the paper does not mention. Both
points are being corrected in the text rather than by re-extracting features,
since all models received identical treatment and the comparison remains
controlled.

## 11. Notebook metadata normalised

All notebooks were passed through a metadata check while preparing this
revision. Colab writes per-cell execution metadata (`executionInfo`,
`outputId`) into saved notebooks; those blocks were normalised away in a
single commit touching 59 files. Cell contents, cell order and every stored
output are unchanged. The commit is mechanical and contains no substantive
edit.

---

## Files added

| path | contents |
|---|---|
| `results_v2/indomain_all_v2.csv` | 864 corrected in-distribution runs |
| `results_v2/ood_all_v2.csv` | corrected OOD runs, with the original oracle values retained for comparison |
| `results_v2/degeneracy_audit.csv` | cells where the probe collapses to one class |
| `results_v2/rank_correlations.csv` | Spearman and Kendall with bootstrap intervals |
| `results_v2/c_ablation.csv` | regularisation sensitivity |
| `results_v2/cri_variants_1.csv`, `cri_variants_100.csv` | four aggregation rules |
| `results_v2/old_vs_new_indomain.csv` | submitted versus corrected, per cell |
| `results_v2/rank_diff_by_fraction.csv` | mean rank displacement by label fraction |
| `results_v2/peft_all.csv` | 36 LoRA fine-tuning runs |
| `results_v2/peft_vs_linear.csv` | LoRA against the frozen linear probe, matched cells |
| `notebooks/PRISM_rebuttal_phase1.ipynb` | reproduces every corrected number |
| `notebooks/PRISM_rebuttal_phase3_peft.ipynb` | LoRA experiment requested by Reviewer fWEj |

The LoRA experiment covers two models (UNI, H-Optimus-0) on two datasets
(MHIST, LungHist700) at three label fractions with three seeds. Two caveats
apply to any comparison of absolute performance between the two probes, and
we state them rather than leave them implicit. First, the frozen-probe results
use cached embeddings extracted with a shared preprocessing pipeline, whereas
the LoRA runs use each model's own `timm` data configuration. Second, the
training budget is a fixed ten epochs rather than a fixed number of gradient
steps, so cells at 1% labels receive far fewer updates than cells at 100%.
Neither caveat affects the finding the experiment was run to test, which is a
within-probe comparison across label fractions.

## Files unchanged

`results/` retains the original submitted result files in full. Nothing in
that directory has been edited or deleted.
