#!/usr/bin/env python
"""Rebuild the PRISM train/val/test split indices for the six main datasets.

The code repeats the dataset construction in the extraction notebooks
(notebooks/<dataset>/<model>_<dataset>.ipynb, notebooks/Pcam/0X_*.ipynb)
without loading any images. Each output row i is the i-th row of the stored
{split}_features.npy / {split}_labels.npy, because every notebook extracted with
DataLoader(shuffle=False).

Run on Colab with Drive mounted (the raw datasets live in MyDrive/PRISM/datasets):

    python make_split_indices.py --datasets-root /content/drive/MyDrive/PRISM/datasets \
        --emb-root /content/drive/MyDrive/PRISM/embeddings --out splits/

Each rebuilt label vector is compared against the stored *_labels.npy of a
reference model (--ref-model, default clip). A mismatch stops the script.

Output: splits/<dataset>.json with
  index_space   what the integers index into
  indices       {train, val, test}: list[int] in stored row order
  items         {train, val, test}: list[str] relative paths or image names, for
                datasets whose licence allows it (omitted for MHIST)
  classes       label id -> class name
  n             split sizes
  label_check   result of the comparison with the stored labels
MHIST: only integer row positions into annotations.csv as distributed. No file
names and no labels are written (MHIST Research Use Agreement §4 and §6).
"""
import argparse
import json
import os
import random

import numpy as np


def stored_labels(emb_root, model, dataset):
    d = os.path.join(emb_root, model, dataset)
    if model == "h_optimus_0" and not os.path.isdir(d):
        d = os.path.join(emb_root, "hoptimus", dataset)
    return {s: np.load(os.path.join(d, f"{s}_labels.npy")).astype(int)
            for s in ("train", "val", "test")}


def check(rebuilt, stored):
    out = {}
    for s in ("train", "val", "test"):
        a, b = np.asarray(rebuilt[s], dtype=int), stored[s]
        out[s] = bool(a.shape == b.shape and np.array_equal(a, b))
    return out


def random_split_indices(n, fracs=(0.70, 0.15), seed=42):
    """Same as torch.utils.data.random_split(ds, [n_tr, n_va, n_te], Generator().manual_seed(seed))."""
    import torch
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}


def build_pcam(root):
    import h5py
    base = os.path.join(root, "pcam", "pcam") if os.path.isdir(os.path.join(root, "pcam", "pcam")) \
        else os.path.join(root, "pcam")
    names = {"train": "train", "val": "valid", "test": "test"}
    idx, lab = {}, {}
    for s, f in names.items():
        with h5py.File(os.path.join(base, f"camelyonpatch_level_2_split_{f}_y.h5"), "r") as h:
            y = h["y"][:, 0, 0, 0].astype(int)
        idx[s], lab[s] = list(range(len(y))), y
    return dict(index_space="row position in camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5 "
                            "(torchvision.datasets.PCAM order)",
                indices=idx, classes=["normal", "tumor"]), lab


def build_imagefolder_random(root, sub):
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(os.path.join(root, sub))
    parts = random_split_indices(len(ds.samples))
    rel = [os.path.relpath(p, os.path.join(root, sub)) for p, _ in ds.samples]
    lab = {s: [ds.samples[i][1] for i in v] for s, v in parts.items()}
    return dict(index_space=f"position in torchvision ImageFolder('{sub}').samples (sorted class folders, sorted files); "
                            "torch random_split 70/15/15 with Generator().manual_seed(42)",
                indices=parts, items={s: [rel[i] for i in v] for s, v in parts.items()},
                classes=ds.classes), lab


def build_bracs(root):
    from torchvision.datasets import ImageFolder
    base = os.path.join(root, "bracs", "latest_version")
    idx, items, lab, classes = {}, {}, {}, None
    for s in ("train", "val", "test"):
        ds = ImageFolder(os.path.join(base, s))
        classes = classes or ds.classes
        assert ds.classes == classes, f"BRACS class folders differ in {s}: {ds.classes}"
        idx[s] = list(range(len(ds.samples)))
        items[s] = [os.path.relpath(p, base) for p, _ in ds.samples]
        lab[s] = [t for _, t in ds.samples]
    return dict(index_space="position in ImageFolder(latest_version/<split>).samples (official BRACS RoI split)",
                indices=idx, items=items, classes=classes), lab


def build_mhist(root):
    import pandas as pd
    ann = pd.read_csv(os.path.join(root, "mhist", "annotations.csv"))
    ann["_row"] = np.arange(len(ann))
    train_df = ann[ann["Partition"] == "train"].reset_index(drop=True)
    test_df = ann[ann["Partition"] == "test"].reset_index(drop=True)
    val_df = train_df.sample(frac=0.15, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    parts = {"train": train_df, "val": val_df, "test": test_df}
    idx = {s: df["_row"].tolist() for s, df in parts.items()}
    lab = {s: (df["Majority Vote Label"] == "SSA").astype(int).tolist() for s, df in parts.items()}
    return dict(index_space="0-based row position in MHIST annotations.csv as distributed (header excluded); "
                            "val = train rows .sample(frac=0.15, random_state=42)",
                indices=idx, classes=["HP", "SSA"]), lab


def build_spider(root):
    base = os.path.join(root, "spider_breast", "extracted", "SPIDER-breast")
    meta = json.load(open(os.path.join(base, "metadata.json")))
    for i, r in enumerate(meta):
        r["_pos"] = i
    train = [r for r in meta if r["split"] == "train"]
    test = [r for r in meta if r["split"] == "test"]
    n_val = int(0.15 * len(train))
    random.seed(42)
    random.shuffle(train)
    parts = {"train": train[n_val:], "val": train[:n_val], "test": test}
    # The notebooks built class_to_idx separately for each split. The ids agree only
    # if every split contains all classes; this is checked here.
    per_split = {s: sorted(set(r["class"] for r in v)) for s, v in parts.items()}
    classes = per_split["train"]
    for s, c in per_split.items():
        assert c == classes, f"SPIDER {s} split has a different class set ({len(c)} vs {len(classes)})"
    c2i = {c: i for i, c in enumerate(classes)}
    return dict(index_space="position in SPIDER-breast metadata.json; val = first 15% of train records "
                            "after random.seed(42); random.shuffle",
                indices={s: [r["_pos"] for r in v] for s, v in parts.items()},
                items={s: [r["image_name"] for r in v] for s, v in parts.items()},
                classes=classes), {s: [c2i[r["class"]] for r in v] for s, v in parts.items()}


BUILDERS = {
    "pcam": build_pcam,
    "crc": lambda root: build_imagefolder_random(os.path.join(root, "crc"), "NCT-CRC-HE-100K"),
    "lunghist700": lambda root: build_imagefolder_random(os.path.join(root, "lunghist700"), "images"),
    "bracs": build_bracs,
    "mhist": build_mhist,
    "spider_breast": build_spider,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="/content/drive/MyDrive/PRISM/datasets")
    ap.add_argument("--emb-root", default="/content/drive/MyDrive/PRISM/embeddings")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--ref-model", default="clip")
    ap.add_argument("--datasets", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in a.datasets:
        rec, lab = BUILDERS[d](a.datasets_root)
        res = check(lab, stored_labels(a.emb_root, a.ref_model, d))
        rec.update(dataset=d, n={s: len(v) for s, v in rec["indices"].items()},
                   label_check={"reference_model": a.ref_model, "equal": res})
        if not all(res.values()):
            raise SystemExit(f"{d}: rebuilt labels differ from stored labels {res}; not writing")
        json.dump(rec, open(os.path.join(a.out, f"{d}.json"), "w"))
        print(f"{d:14s} n={rec['n']}  labels match: {res}")


if __name__ == "__main__":
    main()
