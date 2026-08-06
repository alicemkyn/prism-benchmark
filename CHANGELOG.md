# Changelog

Changes after the NeurIPS 2026 rebuttal round. That round is documented
separately in `CHANGELOG_REBUTTAL.md`, which is kept as it was written, since
it is a record of a specific exchange with reviewers rather than a running log.

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
