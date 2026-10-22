---
license: cc-by-nc-4.0
pretty_name: "PRISM: frozen pathology foundation-model embeddings"
language:
- en
tags:
- pathology
- histopathology
- computational-pathology
- foundation-models
- embeddings
- calibration
- benchmark
- medical
task_categories:
- image-classification
- feature-extraction
size_categories:
- 1M<n<10M
extra_gated_heading: "Access to PRISM embeddings"
extra_gated_prompt: >-
  These embeddings were computed from public pathology datasets, some of which
  (BRACS, SPIDER-Breast) are licensed CC BY-NC 4.0. They are provided for
  non-commercial research only. They are not a medical device and must not be
  used for diagnosis or patient care. By requesting access you agree to use them
  for non-commercial research, to cite the original datasets and models, and to
  comply with the upstream terms listed in RELEASE_POLICY.md.
extra_gated_fields:
  Name: text
  Affiliation: text
  I will use these embeddings for non-commercial research only and not for clinical use: checkbox
configs:
- config_name: pcam
  data_files:
  - split: train
    path: data/pcam/*/train.parquet
  - split: validation
    path: data/pcam/*/validation.parquet
  - split: test
    path: data/pcam/*/test.parquet
- config_name: crc
  data_files:
  - split: train
    path: data/crc/*/train.parquet
  - split: validation
    path: data/crc/*/validation.parquet
  - split: test
    path: data/crc/*/test.parquet
- config_name: bracs
  data_files:
  - split: train
    path: data/bracs/*/train.parquet
  - split: validation
    path: data/bracs/*/validation.parquet
  - split: test
    path: data/bracs/*/test.parquet
- config_name: lunghist700
  data_files:
  - split: train
    path: data/lunghist700/*/train.parquet
  - split: validation
    path: data/lunghist700/*/validation.parquet
  - split: test
    path: data/lunghist700/*/test.parquet
- config_name: spider_breast
  data_files:
  - split: train
    path: data/spider_breast/*/train.parquet
  - split: validation
    path: data/spider_breast/*/validation.parquet
  - split: test
    path: data/spider_breast/*/test.parquet
- config_name: camelyon17
  data_files:
  - split: train
    path: data/camelyon17/*/train.parquet
  - split: validation
    path: data/camelyon17/*/validation.parquet
  - split: test
    path: data/camelyon17/*/test.parquet
---

# PRISM embeddings

Frozen tile-level embeddings from pathology foundation models (FMs), released with
**PRISM: Pathology Reliability In Scarce-label Medicine. A Tile-level Benchmark of Pathology
Foundation Models** (Alicem Koyun, Westcliff University; NeurIPS 2026 GenAI4Health workshop).
They let anyone rerun PRISM's linear-probe protocol (label fractions 1–100%, three seeds,
AUROC, raw and temperature-scaled ECE, Brier score, Composite Reliability Index (CRI), and
the cross-dataset and Camelyon17 hospital-transfer grids) without a GPU.

Code: the `prism_bench` package and extraction notebooks are in the PRISM code repository
(link in the paper). Upstream licences decide which (model, dataset) pairs are included here.

## What is released

Embeddings are released for **5 of the 6 main datasets and for Camelyon17**, for the **5 of 8
FMs** whose licences allow redistribution of model outputs:

| | PCam | CRC | MHIST | BRACS | LungHist700 | SPIDER-Breast | Camelyon17 |
|---|---|---|---|---|---|---|---|
| CLIP | ✓ | ✓ | checksums | ✓ | ✓ | ✓ | ✓ |
| PLIP | ✓ | ✓ | checksums | ✓ | ✓ | ✓ | ✓ |
| GigaPath | ✓ | ✓ | checksums | ✓ | ✓ | ✓ | ✓ |
| H-Optimus-0 | ✓ | ✓ | checksums | ✓ | ✓ | ✓ | ✓ |
| MIDNIGHT | ✓ | ✓ | checksums | ✓ | ✓ | ✓ | ✓ |
| UNI, CONCH, VIRCHOW2 | checksums | checksums | checksums | checksums | checksums | checksums | checksums |

Withheld cells:

- **MHIST** (all 8 models): the Dataset Research Use Agreement forbids distributing any portion of MHIST and creating derivative works from it. Only file-level SHA256 values and summary fingerprints are published. No embeddings, labels or image names are published.
- **UNI, CONCH, VIRCHOW2**: the CC BY-NC-ND 4.0 terms of use define "datasets created from the model" as derivatives.

For these cells, `checksums.csv` gives the SHA256 and numerical fingerprints of the files used
in the paper. `code/` gives the extraction recipe, and `splits/` gives the split indices.
With these, users who hold the upstream licences can regenerate the files and check them.
`RELEASE_POLICY.md` has the full licence review and sources.

<!-- INVENTORY_TABLE_START -->
*(Filled in by `02_hf_upload.ipynb` from `embedding_inventory.csv`: number of rows and embedding dimension per released (model, dataset, split).)*
<!-- INVENTORY_TABLE_END -->

## Repository layout

```
README.md                  this card
RELEASE_POLICY.md          licence review: classification and sources
checksums.csv              every inventory file (released and withheld): sha256, shape, dtype, fingerprints
embeddings/{model}/{dataset}/{train,val,test}_{features,labels}.npy             canonical files (main datasets)
embeddings_camelyon17/{model}/hospital_{0..4}/{train,val,test}_{features,labels}.npy
data/{dataset}/{model}/{train,validation,test}.parquet                          the same values, for `datasets.load_dataset`
splits/{dataset}.json                                  split indices for the main datasets (see code/make_split_indices.py)
splits/camelyon17/{splits.json,resplit.json,...}       Camelyon17 sampling and balanced re-split
code/notebook_map.csv, code/extract_embeddings.py, code/make_split_indices.py
```

Model keys: `clip`, `plip`, `gigapath`, `h_optimus_0`, `midnight` (withheld: `conch`, `uni`, `virchow2`).
Dataset keys: `pcam`, `crc`, `bracs`, `lunghist700`, `spider_breast` (withheld: `mhist`), and `camelyon17`.

## Schema

**`.npy` (canonical; the SHA256 values in `checksums.csv` refer to these files).**
`{split}_features.npy` is a float32 array of shape `(n, d)`; row *i* is the embedding of tile *i*.
`{split}_labels.npy` is an integer array of shape `(n,)`. Split names are `train`, `val` and `test`.

**Parquet configs** (one config per dataset; every released model sits in the same table):

| column | type | meaning |
|---|---|---|
| `model` | string | model key |
| `dataset` | string | dataset key |
| `row` | int64 | row index into the canonical `{split}_features.npy` (main datasets) or into concat(train, val, test) of the stored hospital files (Camelyon17) |
| `label` | int64 | class id (conventions below) |
| `embedding` | list<float32> | the feature vector; its length depends on the model |
| `hospital` | int64 | Camelyon17 only, 0–4 (WILDS numbering) |
| `orig_split` | string | Camelyon17 only: which stored file the row comes from |

```python
from datasets import load_dataset
# every released model for one dataset (large for PCam):
ds = load_dataset("<HF_REPO_ID>", "bracs")
# one model only:
ds = load_dataset("<HF_REPO_ID>", data_files={
    "train": "data/bracs/h_optimus_0/train.parquet"})          # path pattern: data/{dataset}/{model}/{split}.parquet
# or the canonical numpy files:
from huggingface_hub import hf_hub_download
import numpy as np
X = np.load(hf_hub_download("<HF_REPO_ID>", "embeddings/gigapath/bracs/train_features.npy", repo_type="dataset"))
```

**Normalisation.** The stored features are the raw model outputs. CONCH (main datasets only)
is the exception: its features were saved with `normalize=True`, so they have unit norm.
PRISM L2-normalises at load time (`X / ‖X‖₂`). Do the same to reproduce the paper's numbers;
`checksums.csv` reports `mean_l2_norm` for each file.

## Splits and label conventions

Test sets are fixed per dataset and identical across models. Every extraction used
`DataLoader(shuffle=False)`, so row order is the dataset-construction order below.
`splits/{dataset}.json` records the indices and, where the licence allows, the file names.

| dataset | split rule | labels |
|---|---|---|
| PCam | official train / valid / test H5 files (torchvision `PCAM`) | 0 = normal, 1 = tumour |
| CRC | `ImageFolder(NCT-CRC-HE-100K)`; `random_split` 70/15/15, `torch.Generator().manual_seed(42)` | 0–8 in alphabetical folder order: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM |
| BRACS | official RoI train / val / test folders (`latest_version`) | 0–6 in alphabetical folder order (N, PB, UDH, FEA, ADH, DCIS, IC); the exact folder names are in `splits/bracs.json` |
| LungHist700 | `ImageFolder(images/)` with 7 class folders; `random_split` 70/15/15, seed 42 | 0–6 in alphabetical folder order; names in `splits/lunghist700.json` |
| SPIDER-Breast | 224×224 patches at 20× (`metadata.json`); official train / test; val = first 15% of train after `random.seed(42); random.shuffle` | 0–17 = the 18 classes sorted alphabetically; names in `splits/spider_breast.json` |
| MHIST (withheld) | `annotations.csv` Partition; val = train rows `.sample(frac=0.15, random_state=42)` | 0 = HP, 1 = SSA |
| Camelyon17 | 20,000 patches per hospital, label-stratified 50/50 (`splits/camelyon17/splits.json`, indices into the concatenated `wltjr1007/Camelyon17-WILDS` parquet, verified at 455,954 patches against WILDS v1.0). The stored files follow an initial slide-disjoint partition. **The paper's protocol** applies the balanced patch-level 70/15/15 re-split in `resplit.json`: concatenate the stored train, val and test files of a hospital, then index | 0 = non-tumour, 1 = tumour |

The parquet `camelyon17` config is already in the paper's re-split. The `.npy` files are as stored.
Because the re-split is at patch level, train and test within a hospital share slides, so
in-hospital numbers are an upper bound. Transfer pairs (different hospitals) have no leakage.

## Preprocessing

Each FM was embedded once under one pipeline, held fixed across all label fractions and transfer pairs.
Preprocessing differs between models, so between-model comparisons at a fixed fraction are confounded by it.

**Main datasets** (extraction notebooks `notebooks/<dataset>/<model>_<dataset>.ipynb`, `notebooks/Pcam/0X_*.ipynb`):

| model | weights | resize | crop | normalisation (mean / std) | feature |
|---|---|---|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | 224 | 224 | ImageNet | vision `pooler_output` |
| PLIP | `vinid/plip` | 224 | 224 | ImageNet | vision `pooler_output` |
| CONCH (withheld) | `MahmoodLab/conch` | loader | loader | loader transform | `encode_image(proj_contrast=False, normalize=True)` |
| UNI (withheld) | `MahmoodLab/UNI` | 224 | 224 | ImageNet | CLS |
| VIRCHOW2 (withheld) | `paige-ai/Virchow2` | 224 | 224 | ImageNet | concat(CLS, mean of patch tokens 5:) |
| GigaPath | `prov-gigapath/prov-gigapath` | 256 | 224 | ImageNet | CLS |
| H-Optimus-0 | `bioptimus/H-optimus-0` | 224 | 224 | (0.707223, 0.578729, 0.703617) / (0.211883, 0.230117, 0.177517) | CLS |
| MIDNIGHT | `kaiko-ai/midnight` | 224 | 224 | ImageNet (the model card recommends 0.5 / 0.5) | `pooler_output` |

ImageNet = (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225). PCam tiles are 96×96 and are resized to 224×224.
The CLIP and PLIP PCam notebooks use `Resize((224, 224))` without a crop, which gives the same result for square tiles.

**Camelyon17** (`notebooks/PRISM_camelyon17_extract.ipynb`; transforms resolved from each model; `splits/camelyon17/preprocessing_table.csv`):

| model | resize | crop | normalisation | feature | dim |
|---|---|---|---|---|---|
| CLIP, PLIP | 224 | 224 | CLIP statistics (0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758) | vision `pooler_output` | 768 |
| CONCH (withheld) | loader | loader | loader | `encode_image(proj_contrast=False, normalize=False)` | 512 |
| UNI (withheld) | 224 | 224 | ImageNet | CLS | 1024 |
| VIRCHOW2 (withheld) | 224 | 224 | ImageNet | concat(CLS, mean patch) | 2560 |
| GigaPath | 224 | 224 | ImageNet | CLS | 1536 |
| H-Optimus-0 | 256 | 224 | model-specific | CLS | 1536 |
| MIDNIGHT | 224 | 224 | (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5) | `pooler_output` | 1536 |

The CLIP, PLIP and MIDNIGHT Camelyon17 embeddings were re-extracted (notebook cell 9). The repository's first-pass table
listed 512 / 512 / 3072 dimensions for these, which no longer applies. The authoritative dimensions are the shapes in `checksums.csv`.

## Verifying and regenerating withheld embeddings

```bash
python code/extract_embeddings.py map --model uni --dataset bracs   # which notebook, weights, preprocessing
# run that notebook with your own access to the model and dataset, then
python code/extract_embeddings.py verify --checksums checksums.csv --root /path/to/PRISM --model uni --dataset bracs
```

Expect **EXACT** for downloaded copies. For re-extractions expect **CONSISTENT**: GPU kernels,
fp16 autocast and library versions change the low-order bits, so the SHA256 will differ. Labels and
shapes must still match exactly, and the fingerprints must agree within tolerance.

## Intended use and limitations

- Research on the reliability and calibration of frozen pathology representations, label-efficiency studies, and probe design.
- Not for clinical use. None of these models or embeddings has regulatory approval.
- Tile-level only. The datasets come from a limited number of institutions and scanners.
- Commercial use is excluded by CC BY-NC 4.0 and by the BRACS and SPIDER-Breast terms.

## Citation

```bibtex
@inproceedings{koyun2026prism,
  title     = {{PRISM}: Pathology Reliability In Scarce-label Medicine. A Tile-level Benchmark of Pathology Foundation Models},
  author    = {Koyun, Alicem},
  booktitle = {NeurIPS 2026 Workshop on Generative AI for Health (GenAI4Health)},
  year      = {2026}
}
```

Please also cite the source datasets and models. Datasets: PCam (Veeling et al., MICCAI 2018);
NCT-CRC-HE-100K (Kather et al., Zenodo 10.5281/zenodo.1214456; PLoS Med 2019); BRACS (Brancati et al.,
Database 2022, 10.1093/database/baac093); LungHist700 (Diosdado et al., Sci. Data 2024,
10.1038/s41597-024-03944-3); SPIDER (Nechaev et al., arXiv:2503.02876); Camelyon17 (Bándi et al.,
IEEE TMI 2019) and WILDS (Koh et al., ICML 2021); MHIST (Wei et al., AIME 2021) if you regenerate MHIST.
Models: CLIP (Radford et al., 2021); PLIP (Huang et al., Nat. Med. 2023); GigaPath (Xu et al., Nature 2024);
H-optimus-0 (Bioptimus, 2024); Midnight (Karasikov et al., 2025); UNI (Chen et al., Nat. Med. 2024),
CONCH (Lu et al., Nat. Med. 2024) and Virchow2 (Zimmermann et al., 2024) if you regenerate those cells.
