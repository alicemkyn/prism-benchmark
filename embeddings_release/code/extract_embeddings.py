#!/usr/bin/env python
"""PRISM embeddings: find the extraction recipe for a (model, dataset) cell and
verify regenerated or downloaded embeddings against the published checksums.

Embeddings are not extracted by this file. The exact extraction code is in the
PRISM repository notebooks listed in notebook_map.csv: run the listed notebook,
which writes {split}_{features,labels}.npy, then verify the output here.

    python extract_embeddings.py map --model uni --dataset bracs
    python extract_embeddings.py verify --checksums checksums.csv \
        --root /path/to/PRISM --model uni --dataset bracs

--root is the directory that contains embeddings/ and embeddings_camelyon17/
(the same layout as the Hugging Face repo and as MyDrive/PRISM).

Verdicts
  EXACT       SHA256 identical: a faithful copy of the file used in the paper.
  CONSISTENT  SHA256 differs but shape and dtype match, labels are identical, and
              the feature fingerprints (mean L2 norm, global mean/std, 8-d
              projection of the column means) agree within --rtol. This is the
              expected outcome for a re-extraction on different hardware or
              library versions.
  MISMATCH    shape, labels or fingerprints disagree; the regeneration differs
              from the paper's pipeline (wrong split order, preprocessing, ...).
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALIASES = {"hoptimus": "h_optimus_0", "h-optimus-0": "h_optimus_0",
           "spider-breast": "spider_breast", "spider": "spider_breast"}


def canon(x):
    x = x.strip().lower()
    return ALIASES.get(x, x)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(path, chunk=50_000):
    """Return the same statistics that 02_hf_upload.ipynb writes to checksums.csv."""
    X = np.load(path, mmap_mode="r")
    n, d = X.shape
    rng = np.random.default_rng(0)
    P = rng.standard_normal((d, 8)).astype(np.float64) / np.sqrt(d)
    s = np.zeros(d)
    ss = 0.0
    norm_sum = 0.0
    for i in range(0, n, chunk):
        B = np.asarray(X[i:i + chunk], dtype=np.float64)
        s += B.sum(0)
        ss += (B ** 2).sum()
        norm_sum += np.linalg.norm(B, axis=1).sum()
    colmean = s / n
    gmean = colmean.mean()
    gstd = np.sqrt(max(ss / (n * d) - gmean ** 2, 0.0))
    return dict(mean_l2_norm=norm_sum / n, global_mean=gmean, global_std=gstd,
                colmean_proj8=colmean @ P)


def cmd_map(a):
    rows = list(csv.DictReader(open(os.path.join(HERE, "notebook_map.csv"))))
    hits = [r for r in rows
            if (a.model is None or r["model"] == canon(a.model))
            and (a.dataset is None or r["dataset"] == canon(a.dataset))]
    if not hits:
        sys.exit("no match; models: clip plip conch uni virchow2 gigapath h_optimus_0 "
                 "midnight; datasets: pcam crc mhist bracs lunghist700 spider_breast camelyon17")
    for r in hits:
        print("\n".join(f"{k:>14}: {v}" for k, v in r.items()))
        print("-" * 60)


def cmd_verify(a):
    rows = list(csv.DictReader(open(a.checksums)))
    sel = [r for r in rows
           if (a.model is None or r["model"] == canon(a.model))
           and (a.dataset is None or r["dataset"] == canon(a.dataset))]
    if not sel:
        sys.exit("no checksum rows match the filter")
    n_bad = 0
    for r in sel:
        p = os.path.join(a.root, r["hf_path"])
        if not os.path.exists(p):
            print(f"MISSING     {r['hf_path']}")
            n_bad += 1
            continue
        if sha256(p) == r["sha256"]:
            print(f"EXACT       {r['hf_path']}")
            continue
        arr = np.load(p, mmap_mode="r")
        problems = []
        if str(tuple(arr.shape)) != r["shape"]:
            problems.append(f"shape {tuple(arr.shape)} != {r['shape']}")
        if str(arr.dtype) != r["dtype"]:
            problems.append(f"dtype {arr.dtype} != {r['dtype']}")
        if r["kind"] == "labels":
            problems.append("labels differ (labels must be bit-identical: check split order)")
        elif not problems and r.get("mean_l2_norm"):
            fp = fingerprint(p)
            ref = dict(mean_l2_norm=float(r["mean_l2_norm"]),
                       global_mean=float(r["global_mean"]),
                       global_std=float(r["global_std"]),
                       colmean_proj8=np.array([float(v) for v in r["colmean_proj8"].split(";")]))
            for k in ("mean_l2_norm", "global_std"):
                if not np.isclose(fp[k], ref[k], rtol=a.rtol, atol=a.atol):
                    problems.append(f"{k} {fp[k]:.5g} vs {ref[k]:.5g}")
            if not np.allclose(fp["colmean_proj8"], ref["colmean_proj8"],
                               rtol=a.rtol, atol=a.atol * max(1.0, ref["mean_l2_norm"])):
                problems.append("colmean_proj8 differs")
        print(("CONSISTENT  " if not problems else "MISMATCH    ") + r["hf_path"] +
              ("" if not problems else "  <- " + "; ".join(problems)))
        n_bad += bool(problems)
    sys.exit(1 if n_bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="print the notebook and preprocessing for a cell")
    m.add_argument("--model")
    m.add_argument("--dataset")
    v = sub.add_parser("verify", help="check files against checksums.csv")
    v.add_argument("--checksums", required=True)
    v.add_argument("--root", required=True)
    v.add_argument("--model")
    v.add_argument("--dataset")
    v.add_argument("--rtol", type=float, default=1e-2)
    v.add_argument("--atol", type=float, default=1e-4)
    a = ap.parse_args()
    {"map": cmd_map, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
