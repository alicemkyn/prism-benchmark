# Extended benchmark comparison

Referenced from Appendix C. This is the long form of Table 1 in the main
paper, with the columns that did not fit on the page.

Reviewer EiR7 asked for a citation supporting the claim that no prior study
covers the four-way interaction of foundation-model choice, label fraction,
out-of-distribution transfer and calibration. Reviewer tp5b asked what
separates PRISM from THUNDER. This table is the answer to both: the claim is
not a citation but a gap, and a gap is shown by enumeration.

---

## Full comparison

| Benchmark | Year | FMs | Datasets | Level | Label fractions | Calibration metrics | Post-hoc recovery | OOD treatment | Probe |
|---|---|---|---|---|---|---|---|---|---|
| Wölflein et al. | 2023 | 8 extractors | 4 cohorts | slide | no | no | no | no | MIL |
| Kang et al. | 2023 | 4 SSL | 4 | tile + slide | no | no | no | no | linear, MIL |
| eva (Gatopoulos et al.) | 2024 | open harness | 6 | tile + slide | no | no | no | no | linear, MIL |
| Neidlinger et al. | 2025 | 19 | 13 cohorts | slide | no | no | no | no | MIL |
| PathBench (Ma et al.) | 2025 | 19 | 21 | tile + slide | no | no | no | no | linear, MIL |
| Patho-Bench (Zhang et al.) | 2025 | multiple | 42 tasks | slide | no | no | no | no | MIL |
| HistoVL (Al Majzoub et al.) | 2025 | 4 VLMs | 11 | tile | no | ECE | no | no | zero-shot |
| Lee et al. | 2025 | 4 | 14 | tile | few-shot only | no | no | near / mid / out domain | linear, FT |
| Nunes et al. | 2025 | position paper | — | — | no | argues for ECE | no | no | — |
| Roschewitz et al.† | 2025 | fine-tuned | 8 (natural images) | — | no | ECE, Brier | no | natural shift | FT |
| Thiringer et al. | 2026 | 14 | 4 | tile | no | ECE, Brier | no | scanner shift | linear |
| THUNDER (Marza et al.) | 2026 | 23 | 16 | tile | few-shot only (1, 2, 4, 8, 16) | ECE, MCE, ACE, TACE | no | adversarial perturbation | linear, kNN, zero-shot |
| **PRISM (ours)** | **2026** | **8** | **6** | **tile** | **6 points, 1–100%** | **ECE (2 binnings), Brier** | **yes** | **4 directed cross-dataset pairs** | **linear (+ MLP ablation)** |

† Image classification rather than pathology; included as the closest
methodological precedent.

---

## Where PRISM is alone

**Continuous label-fraction sweep.** THUNDER and Lee et al. vary supervision
but only through discrete few-shot counts, which probe a different regime:
1–16 examples per class is a demonstration setting, whereas 1–100% of an
institutional cohort is a procurement decision. No other benchmark varies
supervision at all.

**Post-hoc recoverability as a reported axis.** Several benchmarks report raw
ECE. None report the gap between raw and temperature-scaled ECE, which is the
quantity a deployment team can act on: it distinguishes a model whose logits
are ordered but mis-scaled from one whose logits are uninformative.

**Directed cross-dataset transfer.** Thiringer et al. cover scanner shift and
THUNDER covers adversarial perturbation. Neither is a change of source
population. PRISM's four pairs involve a change of tissue, task definition and
acquisition site simultaneously, which is closer to what a deployment
encounters and, as we report, harder.

**The joint interaction.** Each of the four axes appears somewhere in the
table. No row other than the last has all four.

---

## Where PRISM is narrower than prior work

The axes above are not the only ones that matter.

**Fewer models.** THUNDER covers 23 FMs, PathBench and Neidlinger et al. 19.
PRISM covers 8. Our axis count is higher and our model count is lower.

**Fewer datasets.** THUNDER covers 16, PathBench 21, Patho-Bench 42 tasks.
PRISM covers 6.

**Tile level only.** Wölflein et al., eva, Patho-Bench and Neidlinger et al.
evaluate slide-level tasks through a MIL aggregator. PRISM does not, because
slide-level results are confounded with aggregator choice.

**Four transfer pairs, each with a label-definition change.** Our pairs are
built from publicly licensed data, so covariate shift and label-definition
shift arrive together and cannot be separated. Thiringer et al.'s scanner
shift is cleaner as a covariate-shift measurement.

**One probe family in the main results.** All headline numbers use L2
regularised logistic regression. THUNDER additionally reports kNN and
zero-shot. Our MLP ablation is reported separately and is not the basis for
any main-text claim.

---

## Sources

Full bibliographic entries are in `references.bib` in the paper source. Keys:
`wolflein2023benchmarking`, `kang2023benchmarking`, `gatopoulos2024eva`,
`neidlinger2025`, `ma2025pathbench`, `zhang2025pathobench`,
`majzoub2025histovl`, `lee2025adaptation`, `nunes2025beyond`,
`roschewitz2025calibration`, `thiringer2026`, `marza2026thunder`.
