# PRISM embeddings: release policy

Status: licence review completed 2026-10-04 for the PRISM camera-ready (NeurIPS 2026 GenAI4Health).
This is a conservative reading of the public licence texts listed below. It is not legal advice.
Where two published statements conflict, the more restrictive one is applied.

A (model, dataset) embedding file is released only if **both** the source dataset and the
model allow redistribution of derived outputs. The release carries the most restrictive condition of
any released component (CC BY-NC 4.0, non-commercial research use, no clinical use).
For withheld cells we publish the extraction code, the split indices and the expected SHA256
checksums (plus numerical fingerprints), so users who hold their own licence can regenerate and verify the files.

## 1. Datasets

| Dataset | Licence / terms found | What the terms say about derived data | Class | Sources |
|---|---|---|---|---|
| PCam | CC0 1.0 (follows Camelyon16) | Public-domain dedication; no restriction | **release** | https://github.com/basveeling/pcam (README, "License" section) |
| CRC (NCT-CRC-HE-100K; CRC-VAL-HE-7K) | CC BY 4.0 (Zenodo record 10.5281/zenodo.1214456) | Redistribution and adaptation allowed with attribution | **release** (attribution) | https://zenodo.org/records/1214456 ; HF mirror metadata `cc-by-4.0`: https://huggingface.co/datasets/1aurent/NCT-CRC-HE |
| MHIST | Dataset Research Use Agreement (Hassanpour lab, Dartmouth) | §1 non-commercial research only; §4 "may not distribute, publish, or reproduce a copy of any portion or all of the MHIST to others without specific prior written permission"; §6 "must not … create derivative works from the MHIST"; closing clause forbids retransmitting or distributing "any material" | **do-not-release** (embeddings, labels and file names withheld; integer split indices, code and checksums only) | https://bmirds.github.io/MHIST/ ; https://bmirds.github.io/MHIST/Dataset%20Research%20Use%20Agreement.pdf |
| BRACS | Conflicting statements: the dataset paper says registration plus "terms of use (Creative Commons CC0 license)"; the BRACS website is marked CC BY-NC 4.0. The site's *Rules* page asks only for non-anonymous registration and citation of Brancati et al. 2022; it has no redistribution clause | Applied the stricter reading, CC BY-NC 4.0: non-commercial use, attribution, cite the paper | **release-with-conditions** (CC BY-NC 4.0, citation) | https://www.bracs.icar.cnr.it/rules/ ; https://www.bracs.icar.cnr.it/background/ (CC BY-NC 4.0 notice) ; https://doi.org/10.1093/database/baac093 (Data organisation section) |
| LungHist700 | CC BY 4.0 (figshare 10.6084/m9.figshare.25459174) | Redistribution and adaptation allowed with attribution | **release** (attribution) | https://figshare.com/articles/dataset/LungHist700_A_Dataset_of_Histological_Images_for_Deep_Learning_in_Pulmonary_Pathology/25459174 ; https://www.nature.com/articles/s41597-024-03944-3 |
| SPIDER-Breast | **CC BY-NC 4.0**, "research use only"; access to the HF dataset is gated (auto-approval) | Non-commercial; adaptation allowed with attribution. The submitted paper's Table 2 says "CC-BY 4.0", which is **wrong** | **release-with-conditions** (CC BY-NC 4.0, research use) | https://huggingface.co/datasets/histai/SPIDER-breast (card, "License"; API tag `license:cc-by-nc-4.0`, `gated: auto`) ; https://github.com/histai/spider |
| Camelyon17 (WILDS v1.0) | CC0 1.0 | Public domain. PRISM read the patches from the parquet mirror `wltjr1007/Camelyon17-WILDS` (tagged `license: other`, with no added terms) and checked the patch count against WILDS v1.0 (455,954) | **release** | https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py ("distributed under CC0") ; https://academic.oup.com/gigascience/article/7/6/giy065/5026175 ; https://registry.opendata.aws/camelyon/ |

## 2. Models (licence on the *outputs*)

| Model | Licence / access terms | Clause relevant to embeddings | Class | Sources |
|---|---|---|---|---|
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | MIT (openai/CLIP repo) | None; the card's research-use statement is guidance, not a licence term | **release** | https://github.com/openai/CLIP/blob/main/LICENSE ; https://huggingface.co/openai/clip-vit-base-patch32 |
| PLIP (`vinid/plip`) | **No licence declared** on the HF repo or the GitHub repo (PathologyFoundation/plip). The card reuses CLIP's "research output … any deployed use case is out of scope" text | No term restricts outputs, but the absence of any licence is a risk | **release-with-conditions** (non-commercial research only, inherited; flagged for author confirmation) | https://huggingface.co/vinid/plip ; https://github.com/PathologyFoundation/plip |
| CONCH (`MahmoodLab/CONCH`) | CC BY-NC-ND 4.0 plus gated terms of use | Gated prompt: "derivatives, which include models trained on outputs from the CONCH model or datasets created from the CONCH model"; ND forbids sharing adapted material; users "agree not to distribute, publish or reproduce a copy of the model" | **do-not-release** (unless the Mahmood Lab grants written permission) | https://huggingface.co/MahmoodLab/CONCH (License and Terms of Use) |
| UNI (`MahmoodLab/UNI`) | CC BY-NC-ND 4.0 plus gated terms of use (institutional e-mail) | Same clause as CONCH: datasets created from the model are derivatives | **do-not-release** (unless permission is granted) | https://huggingface.co/MahmoodLab/UNI (License and Terms of Use) |
| Virchow2 (`paige-ai/Virchow2`) | CC BY-NC-ND 4.0 plus gated terms of use (the repo README's "Apache 2.0" is **wrong**; that is Virchow v1) | Derivatives include "datasets created from the Virchow2 Model"; academic research use only; no clinical, RUO or IUO use | **do-not-release** (unless Paige grants permission) | https://huggingface.co/paige-ai/Virchow2 (gated prompt, Terms of use) |
| Prov-GigaPath (`prov-gigapath/prov-gigapath`) | Apache 2.0; access gated (contact sharing) | Apache 2.0 allows redistribution. The README states the checkpoints are "intended to be used solely for (I) future research … and (II) reproducibility", with no clinical or deployed use. The authors publish their own PCam embeddings | **release-with-conditions** (research use, non-clinical notice) | https://huggingface.co/prov-gigapath/prov-gigapath ; https://github.com/prov-gigapath/prov-gigapath (LICENSE; "Usage and License Notices") |
| H-optimus-0 (`bioptimus/H-optimus-0`) | Apache 2.0; access gated (prompt is a disclaimer only) | No restriction on outputs; carry the as-is and not-regulatory-approved disclaimer | **release** (disclaimer carried) | https://huggingface.co/bioptimus/H-optimus-0 |
| Midnight-12k (`kaiko-ai/midnight`) | MIT | None | **release** | https://huggingface.co/kaiko-ai/midnight (card: "publicly available … under the permissive MIT license") |

## 3. Resulting release matrix

Released = dataset class ∈ {release, release-with-conditions} AND model class ∈ {release, release-with-conditions}.

| | PCam | CRC | MHIST | BRACS | LungHist700 | SPIDER-Breast | Camelyon17 (5 hospitals) |
|---|---|---|---|---|---|---|---|
| CLIP | R | R | W-d | R | R | R | R |
| PLIP | R | R | W-d | R | R | R | R |
| CONCH | W-m | W-m | W-d,m | W-m | W-m | W-m | W-m |
| UNI | W-m | W-m | W-d,m | W-m | W-m | W-m | W-m |
| VIRCHOW2 | W-m | W-m | W-d,m | W-m | W-m | W-m | W-m |
| GigaPath | R | R | W-d | R | R | R | R |
| H-Optimus-0 | R | R | W-d | R | R | R | R |
| MIDNIGHT | R | R | W-d | R | R | R | R |

R = released; W-d = withheld because of the dataset licence; W-m = withheld because of the model licence.
Released: 25 of 48 (model, dataset) cells on the six main datasets and 25 of 40 (model, hospital) cells on Camelyon17
(150 + 150 of the expected 288 + 240 `.npy` files, provisional until counted against `embedding_inventory.csv`).
The upload notebook applies this table mechanically (`RELEASE_MODELS`, `RELEASE_DATASETS`).

## 4. What is published for withheld cells

1. `checksums.csv` lists every file in the inventory, released or not: path, shape, dtype, size, SHA256, and for feature files `mean_l2_norm`, `global_mean`, `global_std` and an 8-number random-projection sketch of the column means (`colmean_proj8`). No row-level values are published for withheld cells.
2. Extraction code: `code/notebook_map.csv` maps each (model, dataset) to the notebook in the PRISM repository that produced it, and `code/extract_embeddings.py map|verify` prints the recipe and checks regenerated files.
3. Split indices: `splits/<dataset>.json`, produced by `code/make_split_indices.py`. For MHIST these are integer row positions into the user's own `annotations.csv`; no file names or labels are included. Camelyon17 indices are in `splits/camelyon17/`.

**Bit-exact regeneration is not expected.** GPU kernels, fp16 autocast (Camelyon17), and the versions of timm, transformers and torch change the low-order bits. A SHA256 match therefore confirms a faithful *copy*. For a *regenerated* file, `extract_embeddings.py verify` requires an exact match for labels and shapes and a tolerance match (default rtol 1e-2) for the feature fingerprints.

## 5. Paper text

Replace "All six licenses permit research use, which makes the release of pre-computed embeddings possible." with:

> Pre-computed embeddings are released for 5 of the 6 datasets and for Camelyon17, for the five FMs whose licences allow redistribution of model outputs (CLIP, PLIP, GigaPath, H-Optimus-0, MIDNIGHT), under CC BY-NC 4.0. MHIST's research-use agreement forbids redistribution and derivative works, and the CC-BY-NC-ND terms of UNI, CONCH and VIRCHOW2 classify datasets created from the model as derivatives. For these we release extraction scripts, split indices and SHA256 checksums so that licensed users can regenerate and verify the embeddings.

Short form (abstract or checklist): "embeddings released for 5 of 6 datasets (and Camelyon17) and 5 of 8 models; extraction scripts, split indices and checksums for the rest."

Table 2 licence column corrections: SPIDER-Breast "CC-BY 4.0" → "CC BY-NC 4.0"; BRACS "Custom" → "CC BY-NC 4.0"; MHIST "Custom (RUA)" → "Research Use Agreement (no redistribution)".

Variants:
- If the BRACS and SPIDER-Breast embeddings are also withheld (strict non-commercial reading): "embeddings released for 3 of 6 datasets (PCam, CRC, LungHist700) and Camelyon17 …".
- If written permission is obtained from the Mahmood Lab (UNI, CONCH) and Paige (VIRCHOW2), add those models as a separate gated, non-commercial configuration: "… for all eight FMs on 5 of 6 datasets and Camelyon17; MHIST is covered by extraction scripts and checksums only."
