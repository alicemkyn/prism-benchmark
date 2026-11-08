#!/usr/bin/env python3
"""CPU quick example for the PRISMEvaluator evaluate and compare APIs.

The arrays in this script are synthetic. They are drawn from a fixed NumPy
Generator and L2-normalised. They are not tiles from PCam or any other
pathology dataset, and the metrics printed here are not results from the
paper.

``dataset="pcam"`` is passed only so ``compare()`` can line the synthetic
probe up with the bundled PCam reference rows from ``results_v2``. The
reference rows are the packaged reference numbers. The ``SyntheticProbe``
row is not.

Run from a checkout, after ``pip install -e .`` (or ``pip install prism-bench``)::

    python examples/quickstart_cpu.py

No network access and no Hugging Face login are required.

Optional real embeddings
------------------------
Precomputed embeddings that the upstream licences allow are published as the
gated dataset ``alicemkyn/PRISM-embeddings``:

    https://huggingface.co/datasets/alicemkyn/PRISM-embeddings

Open that page and accept the click-through agreement with the account that
will download the files. This script does not request access, authenticate,
or download anything. Access is granted by Hugging Face after that
agreement, not by this repository.

After access is in place, one released cell (GigaPath on BRACS) can be
loaded from the canonical ``.npy`` names in ``embeddings_release/README.md``::

    from huggingface_hub import hf_hub_download
    import numpy as np

    def load_split(split):
        features = hf_hub_download(
            "alicemkyn/PRISM-embeddings",
            f"embeddings/gigapath/bracs/{split}_features.npy",
            repo_type="dataset",
        )
        labels = hf_hub_download(
            "alicemkyn/PRISM-embeddings",
            f"embeddings/gigapath/bracs/{split}_labels.npy",
            repo_type="dataset",
        )
        X = np.load(features)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X, np.load(labels)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

Pass those arrays to ``PRISMEvaluator.evaluate`` with ``dataset="bracs"``.
The main datasets use unit-norm features; Camelyon17 ``.npy`` files are raw
model outputs and need the same L2 normalisation. MHIST, and the UNI, CONCH
and VIRCHOW2 cells, are not redistributed. ``embeddings_release/RELEASE_POLICY.md``
states which cells are checksums only. ``compare()`` accepts the six main
dataset keys, or ``"all"``.
"""

from __future__ import annotations

import numpy as np

from prism_bench import PRISMEvaluator

# Reduced grid so the check finishes quickly on CPU. The paper protocol is
# the package defaults: six fractions and seeds 42, 123 and 456.
FRACTIONS = [0.1, 1.0]
SEEDS = [42]
MODEL_NAME = "SyntheticProbe"


def make_synthetic(
    n_train: int = 240,
    n_val: int = 80,
    n_test: int = 80,
    n_classes: int = 2,
    dim: int = 16,
    seed: int = 0,
) -> dict:
    """Deterministic synthetic embeddings. Not a pathology dataset."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    y = rng.integers(0, n_classes, size=n)
    centers = rng.normal(size=(n_classes, dim))
    # A modest offset keeps the toy from collapsing to a perfect probe.
    X = rng.normal(size=(n, dim)) + 0.35 * centers[y]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return {
        "source": "synthetic",
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[n_train:n_train + n_val],
        "y_val": y[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:],
        "y_test": y[n_train + n_val:],
    }


def run():
    """Evaluate a synthetic probe and compare it with the bundled references."""
    data = make_synthetic()
    if data["source"] != "synthetic":
        raise RuntimeError("quickstart data must stay labeled synthetic")
    evaluator = PRISMEvaluator()
    results = evaluator.evaluate(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        dataset="pcam",
        model_name=MODEL_NAME,
        val_features=data["X_val"],
        val_labels=data["y_val"],
        label_fractions=FRACTIONS,
        seeds=SEEDS,
    )
    comparison = evaluator.compare(results, dataset="pcam", fraction=0.1)
    return data, results, comparison


def main() -> None:
    data, results, comparison = run()
    print("PRISM CPU quick example")
    print("data_source=synthetic")
    print(
        "Software check of evaluate() and compare(). "
        "Reference rows are the bundled results_v2 numbers, rounded by compare(). "
        "The SyntheticProbe row is not the paper's results."
    )
    print(
        "SyntheticProbe CRI uses an imputed OOD_Stability "
        "(the reference mean). It is not a model ranking."
    )
    print("paper=https://arxiv.org/abs/2610.36517")
    print("code=https://github.com/alicemkyn/prism-benchmark")
    print("package=https://pypi.org/project/prism-bench/")
    print(
        "embeddings=https://huggingface.co/datasets/alicemkyn/PRISM-embeddings"
        " (gated; accept the click-through agreement before download)"
    )
    print(
        "dataset_key_for_compare=pcam "
        "(selects bundled reference rows; features are synthetic)"
    )
    print(
        f"n_train={len(data['y_train'])} n_val={len(data['y_val'])} "
        f"n_test={len(data['y_test'])} dim={data['X_train'].shape[1]}"
    )
    mine = comparison.loc[comparison["model"] == MODEL_NAME].iloc[0]
    if not np.isfinite(mine["cri"]):
        raise SystemExit("synthetic probe CRI was not finite")
    print(f"synthetic_row_is_reference={bool(mine['is_reference'])}")
    print(f"synthetic_ood_stability_imputed={bool(mine['ood_stability_imputed'])}")
    print(f"n_compare_rows={len(comparison)}")
    print("evaluate() rows (synthetic probe):")
    cols = ["model", "dataset", "fraction", "seed", "auroc",
            "ece_fixed", "ece_scaled_fixed", "temperature"]
    print(results[cols].to_string(index=False))
    print("compare() at fraction 0.1 (reference rows plus SyntheticProbe):")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
