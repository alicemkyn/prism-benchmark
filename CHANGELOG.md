# Changelog

Changes after the NeurIPS 2026 rebuttal round. That round is documented
separately in `CHANGELOG_REBUTTAL.md`, which is kept as it was written, since
it is a record of a specific exchange with reviewers rather than a running log.

---

## v0.3.0 — camera-ready release (2026-10-04)

First packaged release at version 0.3.0. The covariate-shift extension
documented in the next section was written under the 0.3.0 heading while the
package metadata still read 0.2.0; both are now released together, and the
version is 0.3.0 in `pyproject.toml`, `setup.py` (shim), `prism_bench.__version__`
and the README badge.

### Naming and metadata
- Title: *PRISM: Pathology Reliability In Scarce-label Medicine. A Tile-level
  Benchmark of Pathology Foundation Models*. Author: Alicem Koyun, Westcliff
  University; NeurIPS 2026 Workshop on GenAI for Health.
- CRI is renamed **Composite Reliability Index** (previously "Clinical
  Readiness Index"); the acronym and all function names (`compute_cri`,
  `cri` columns) are unchanged.

### Fixes
1. **`prism evaluate` crashed** with `KeyError: "Columns not found: 'ece',
   'ece_scaled'"`: the summary requested columns that `evaluate()` does not
   return. It now summarises `auroc, f1_macro, brier, ece_fixed, temperature,
   ece_scaled_fixed, degeneracy_share`.
2. **`--val-features` / `--val-labels` were documented but not defined** in the
   CLI, so temperature scaling and CRI could not be computed from the command
   line. Both are added (and must be given together), together with
   `--label-fractions`, `--seeds`, `--C`; `prism compare` gains
   `--aggregation` and `--ood-stability`; `--results-dir` is now optional;
   new `prism reference` and `prism --version`; `python -m prism_bench` works.
3. **`compare()` read the as-submitted v1 files** (`results/*_results.csv`,
   `*_temperature_scaling.csv`), so users saw reference values that differ
   from the paper. It now reads the corrected `results_v2` layout
   (`indomain_all_v2.csv`, `ood_all_v2.csv`), mapping dataset keys onto the
   names used there (`pcam -> PCam`, `spider_breast -> SPIDER-Breast`, ...).
   OOD_Stability is computed from `ood_all_v2.csv` in the same way. The legacy
   layout is still readable, with a warning, for traceability.
4. **Bundled reference.** The `results_v2` files `compare()` needs are shipped
   as package data (`prism_bench/reference/`, byte-identical copies), so
   `PRISMEvaluator()` works after `pip install prism-bench`. A
   `results_dir="results_v2"` that does not exist falls back to the bundled
   copy with a warning.
5. **`reference_cri(fraction)` and `compare(dataset="all")`** pool the six
   datasets and reproduce `results_v2/cri_variants_100.csv` and
   `cri_variants_1.csv` (max abs. difference < 1e-15).
6. **Binary temperature convention.** The binary logit `d` was expanded to
   `[-d, d]`, whose softmax is sigmoid(2d) rather than the probe's sigmoid(d),
   so a fitted binary temperature was twice the temperature of the probe
   itself and T = 1 did not reproduce the raw probabilities. Logits are now
   `[-d/2, d/2]`. Scaled probabilities, `ece_scaled_*` and CRI are unchanged;
   the binary `temperature` column in `results_v2` (produced by
   `PRISM_rebuttal_phase1.ipynb` with the old convention) should be divided by
   2 to be read on the probe's scale. Under that reading binary probes are
   close to calibrated (over-confident in 33 of 96 cells), while multiclass
   probes are under-confident (T < 1 in 189 of 192 cells).
7. Input validation in `evaluate()` (length mismatches; `val_features`
   without `val_labels`).
8. Packaging: PEP 621 `pyproject.toml`, `MANIFEST.in` (sdist excludes
   notebooks and result directories), `LICENSE` (MIT, as stated in the
   README), `scripts/build_pypi.sh` (build + `twine check`; the upload
   command is left commented out).
9. Tests: `tests/test_end_to_end.py` (pytest, synthetic embeddings) covers
   `evaluate` with and without a validation split, the Appendix E snippet
   verbatim, reference rows against `results_v2`, the CRI regression targets,
   the bundled and legacy fallbacks, and the CLI via subprocess.

### Documentation
- README: new title and author; *Composite Reliability Index* wording;
  installation via `pip install prism-bench` and from GitHub; Hugging Face
  embeddings section; non-anonymous citation. Placeholders still to be filled
  are marked `TODO(camera-ready)` (`GITHUB_URL`, `HF_DATASET_URL`).
- Key Findings corrected against `results_v2`: no "50x" claim (temperature
  scaling removes up to 0.45 of ECE yet fails to reduce it in 49 of 288
  cells); direction of miscalibration stated by task type; decoupling stated
  as a property of the fixed-probe evaluation pipeline; the 0.85 transfer-AUROC
  value described as observed in these settings and correlational; the
  source-temperature result scoped to the source fractions where it holds for
  all eight models (MHIST→PCam ≥ 25%, CRC→BRACS ≥ 5%); CLIP's Camelyon17 AUROC
  labelled as a mean over label fractions; CRC example updated to v2 values
  (AUROC > 0.977, raw ECE 0.18–0.37).
- Dataset count aligned with the paper: six datasets plus Camelyon17 (badge
  previously read "Datasets 7").
- `CHANGELOG_REBUTTAL.md` item 10: dated correction note added (the
  in-distribution notebooks do not use a single shared transform). The
  original text is left in place.

### Corrections to earlier entries in this file
- *v0.2.1, "Breaking points across thresholds"* states that the ordering of
  models "is stable at every threshold tested". `results_v2/breaking_points_by_threshold.csv`
  does not support this on LungHist700: at threshold 0.4 PLIP and CONCH cross
  at 1% labels and UNI only at 25%, so the ordering changes at that threshold.
  (Correction added 2026-10-04; the original sentence is left unchanged
  below.)

### Unchanged
- `results/` (v1, as submitted) and `results_v2/` are untouched.
- Fixed C = 1 remains the main protocol; tuned-C and slide-disjoint
  Camelyon17 replications are deferred to a journal version.

---

## v0.3.0 — covariate-shift extension

### What changed in the claim

The submitted version reported that additional source-domain labels worsen
out-of-distribution calibration, based on four transfer pairs. All four change
the label definition as well as the acquisition conditions, so the effect could
not be attributed to distribution shift alone. Reviewer tp5b raised this and
was right to.

Repeating the protocol on Camelyon17, where five hospitals share one binary
task and one label definition, **inverts the direction**: target ECE rises in
10 of 160 (model, pair) combinations rather than 28 of 32, and the mean change
is −0.149 rather than +0.082.

A mechanism analysis over all 192 combinations from both designs resolves this.
Transfer AUROC predicts the slope of target ECE at Spearman −0.78 overall, and
at −0.65 within Camelyon17 alone, where dataset, task and kind of shift are all
held constant. Below a transfer AUROC of roughly 0.85 the effect is present in
86–88% of combinations; above 0.95, in 1%.

The claim is therefore restated: **reverse calibration scaling is a signature
of failing transfer, not of distribution shift as such.** Label-definition
shift is the most reliable way to push a probe below the threshold, but not the
only one. CLIP under pure covariate shift transfers at AUROC 0.849 and shows
the effect in 8 of its 20 hospital pairs, while the seven models transferring
at 0.97–0.99 show it in 2 of 140.

This supersedes items 2 and 4 of the previous Key Findings list. Item 2 there
(calibration inversion on LungHist700) is retained but scoped: it holds under a
linear probe and reverses under an MLP probe, as recorded in
`CHANGELOG_REBUTTAL.md`.

### Camelyon17 study

- Five hospitals from the WILDS release of Camelyon17, verified at 455,954
  patches against the published figures before use
- 20,000 label-stratified patches per hospital, split 14,000 / 3,000 / 3,000
  at exactly 50/50 in every partition
- All 20 directed hospital pairs, plus the five in-distribution grids: 720
  in-distribution and 2,880 transfer runs on the standard protocol
- Each model embedded under **its own** documented preprocessing transform,
  resolved programmatically and recorded in
  `data/camelyon17/preprocessing_table.csv`. This dataset therefore carries no
  preprocessing heterogeneity, unlike the six main datasets

**Split protocol.** The first extraction used a slide-disjoint split. Because
each hospital contributes about ten slides and tumour concentrates in two or
three of them, that left tumour prevalence between 0.1% and 83% across
partitions, ranging within a single hospital from 5.8% in train to 81% in test.
ECE is prevalence-sensitive, so cross-hospital calibration comparisons under
such a split would have measured prevalence rather than calibration.

The split is now at patch level, stratified, giving exactly 50% prevalence
everywhere. `data/camelyon17/resplit.json` is an index map applied at load
time; the extracted `.npy` files were not modified. The cost is within-hospital
slide overlap between train and test, which affects only the in-distribution
numbers; in the transfer grid, source and target are different hospitals with
disjoint patients and slides, so no leakage is possible there. In-distribution
results on this dataset are reported as an upper bound.

**Label budget.** Both temperature protocols are computed from the same probe
fit and reported side by side: the full validation split, matching the six main
datasets, and a proportional split subsampled by the same label fraction. Raw
ECE and AUROC do not touch the validation split and are identical under both.
Scaled ECE differs by at most 0.001 at any fraction. The proportional
temperature reaches a search bound in 47% of cells at 1% labels, where only 30
validation patches remain, so the full-validation value is the one reported.

### Files added

| path | contents |
|---|---|
| `data/camelyon17/splits.json` | sampled patch indices per hospital and split |
| `data/camelyon17/resplit.json` | balanced re-split index map applied at load time |
| `data/camelyon17/split_report.csv` | sizes and prevalence under the original slide-disjoint split |
| `data/camelyon17/resplit_report.csv` | the same under the balanced split |
| `data/camelyon17/preprocessing_table.csv` | transform resolved from each of the eight models |
| `results_v2/camelyon17_indomain.csv` | 720 in-distribution runs |
| `results_v2/camelyon17_transfer.csv` | 2,880 transfer runs over 20 directed pairs |
| `results_v2/camelyon17_ece_trend.csv` | per (model, pair) ECE trend summary |
| `results_v2/camelyon17_cri_components.csv` | CRI axes on same-task transfer |
| `results_v2/camelyon17_label_budget.csv` | realised label budget per cell |
| `results_v2/mechanism_summary.csv` | all 192 combinations with slope, transfer AUROC and degeneracy |
| `results_v2/mechanism_bands.csv` | rate of rising ECE by transfer AUROC band |
| `figures/fig_contrast.pdf` | the two transfer designs side by side |
| `figures/fig_mechanism.pdf` | ECE slope against transfer AUROC, 192 combinations |
| `figures/fig_hospital_matrix.pdf` | AUROC and ECE by source and target hospital |
| `notebooks/PRISM_camelyon17_extract.ipynb` | embedding extraction, GPU |
| `notebooks/PRISM_camelyon17_transfer.ipynb` | both grids, CPU |
| `notebooks/PRISM_mechanism.ipynb` | transfer-quality analysis across both designs |
| `notebooks/PRISM_camelyon17_figures.ipynb` | the three figures above |

---

## v0.2.1 — analyses requested in the second review round

Reviewer tp5b raised six further points after the first rebuttal. Five were
answered with measurements; the sixth required new data and is answered by the
Camelyon17 study above.

**Breaking points across thresholds.** The first rebuttal referred to a sweep
of macro-F1 thresholds that was not in the repository. It is now in
`results_v2/breaking_points_by_threshold.csv`, and it supports the reviewer's
caution: sweeping the threshold from 0.4 to 0.8 leaves the breaking point
unchanged on PCam and CRC but moves it by 0.64 on MHIST and 0.72 on
LungHist700. On those two datasets a breaking point is not a stable model
property, and the paper now reports the ordering, which is stable at every
threshold tested, rather than a single fraction per model.

**Subset-selection variability.** The regularisation sweep measured
hyperparameter sensitivity, which is a different question from the variability
introduced by drawing few samples. The right quantity is now in
`results_v2/seed_variability_auroc.csv`: at 1% labels the between-seed standard
deviation in AUROC is 0.055 on MHIST, 0.034 on LungHist700 and 0.026 on BRACS,
against 0.002 on CRC and PCam, roughly twice the spread induced by sweeping C
over two orders of magnitude. It does not account for the decoupling, which
appears within each of the three seeds taken separately (−0.17, −0.25, −0.10 at
1% labels).

**Bootstrap at seed level.** The earlier interval resampled datasets after
collapsing seeds to a mean. `results_v2/rank_correlations_seedlevel.csv`
resamples datasets and one seed per cell, so seed noise propagates into the
interval. The contrast between label regimes narrows from Δρ = 0.75, CI
[0.19, 1.27] to Δρ = 0.62, CI [0.08, 1.13], and still excludes zero. The
conservative value is the one now reported.

**CRI components.** Reported separately in `results_v2/cri_components.csv` and
`cri_components_per_pair.csv`, at the reviewer's suggestion. On the four
original pairs the composite tracks the OOD axis (Kendall tau 0.79) far more
closely than the in-distribution axis (0.29), which is the behaviour the
reviewer predicted from the ratio form of `OOD_Stability`.

**Proportional label budget.** `results_v2/indomain_propval.csv` and
`label_budget_accounting.csv`. At 1% labels the validation split supplies
between 90% and 96% of all labelled examples a cell consumes. Re-running with
validation subsampled by the same fraction changes scaled ECE at 1% labels by
+0.009 and calibration recoverability by −0.009, and leaves the model rankings
intact (mean Kendall tau 0.75, no sign inversions in 36 comparisons). One
caveat applies to both protocols: on the small datasets the proportional budget
leaves three validation samples on MHIST and seven on LungHist700 at 1% labels,
too few to fit a scalar temperature stably, while the full-validation protocol
borrows labels the condition should not have. Calibration at 1% labels is
reported but not relied upon.

**Transfer-setting regularisation sweep.** `results_v2/ood_c_ablation.csv` and
`ood_c_ranking.csv`, 1,440 fits over all four original pairs. Both rankings
survive: mean Kendall tau 0.71 for OOD AUROC with no sign inversions in 48
comparisons, and 0.59 for OOD calibration with two. The ranking instability
reported in distribution does not carry into transfer.

### Files added

`results_v2/breaking_points_by_threshold.csv`,
`seed_variability_auroc.csv`, `seed_variability_ece.csv`,
`rank_correlations_seedlevel.csv`, `cri_components.csv`,
`cri_components_per_pair.csv`, `indomain_propval.csv`,
`label_budget_accounting.csv`, `ood_c_ablation.csv`, `ood_c_ranking.csv`,
`mlp_probe.csv`, `mlp_vs_linear.csv`, `mlp_vs_linear_ranking.csv`,
`ood_mlp.csv`, `ood_mlp_vs_linear.csv`

`notebooks/PRISM_rebuttal_round2.ipynb`,
`PRISM_rebuttal_propval.ipynb`, `PRISM_rebuttal_ood_c_ablation.ipynb`,
`PRISM_rebuttal_mlp_probe.ipynb`, `PRISM_rebuttal_ood_mlp.ipynb`

---

## v0.2.0 — rebuttal revision

See `CHANGELOG_REBUTTAL.md`. Eleven items, covering the target-test leakage in
temperature scaling, ECE binning, subset stratification, degenerate probes, the
`OOD_Stability` implementation, F1 averaging, `ECE_scaled` normalisation, the
documented API, installation, preprocessing provenance and notebook metadata.
