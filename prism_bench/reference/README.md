# Bundled reference results (results_v2)

Byte-identical copies of the corrected result files in the repository's
`results_v2/` directory, shipped so that `PRISMEvaluator().compare(...)` works
after `pip install prism-bench` without a repository checkout.

| file | used by |
|---|---|
| `indomain_all_v2.csv` | `compare()`, `reference_cri()`; in-distribution AUROC / ECE / ECE_scaled for 8 FMs x 6 datasets x 6 fractions x 3 seeds (864 rows) |
| `ood_all_v2.csv` | OOD_Stability (four transfer pairs, 576 rows) |
| `cri_variants_100.csv`, `cri_variants_1.csv` | regression targets for `reference_cri(1.0)` / `reference_cri(0.01)` |
| `cri_components.csv`, `cri_components_per_pair.csv` | CRI components, for reporting alongside the composite |

The as-submitted results (`results/`, v1) are not bundled.
