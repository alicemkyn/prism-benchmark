"""
PRISMEvaluator: evaluate custom model embeddings against PRISM reference models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .metrics import compute_auroc, compute_ece, compute_brier, compute_cri, temperature_scale


REFERENCE_MODELS = ["clip", "plip", "conch", "virchow2", "uni", "gigapath", "h_optimus_0", "midnight"]

REFERENCE_MODEL_DISPLAY = {
    "clip": "CLIP",
    "plip": "PLIP",
    "conch": "CONCH",
    "virchow2": "VIRCHOW2",
    "uni": "UNI",
    "gigapath": "GigaPath",
    "h_optimus_0": "H-Optimus-0",
    "midnight": "MIDNIGHT",
}

DATASETS = ["pcam", "mhist", "crc", "bracs", "lunghist700", "spider_breast"]

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [42, 123, 456]

# Naming exceptions for result CSVs
NAMING_EXCEPTIONS = {("h_optimus_0", "pcam"): "hoptimus"}


class PRISMEvaluator:
    """
    Evaluate model embeddings using the PRISM reliability protocol.

    Usage:
        evaluator = PRISMEvaluator(results_dir="path/to/prism/results")

        # Evaluate custom model embeddings
        report = evaluator.evaluate(
            train_features=X_train,  # np.ndarray (N, D)
            train_labels=y_train,    # np.ndarray (N,)
            test_features=X_test,    # np.ndarray (M, D)
            test_labels=y_test,      # np.ndarray (M,)
            dataset="pcam",
            model_name="MyModel",
        )

        # Compare against reference models
        comparison = evaluator.compare(report, dataset="pcam")
        print(comparison)
    """

    def __init__(self, results_dir=None):
        """
        Args:
            results_dir: path to PRISM results directory containing
                         reference model CSVs. If None, only custom
                         model evaluation is available (no comparison).
        """
        self.results_dir = results_dir
        self._reference_cache = {}

    def evaluate(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        dataset,
        model_name="CustomModel",
        label_fractions=None,
        seeds=None,
    ):
        """
        Evaluate model embeddings using PRISM protocol.

        Args:
            train_features: np.ndarray (N, D) - training embeddings
            train_labels:   np.ndarray (N,)   - training labels (int)
            test_features:  np.ndarray (M, D) - test embeddings
            test_labels:    np.ndarray (M,)   - test labels (int)
            dataset:        str - dataset name (e.g., "pcam")
            model_name:     str - display name for the model
            label_fractions: list of floats (default: PRISM standard fractions)
            seeds:          list of ints (default: [42, 123, 456])

        Returns:
            pd.DataFrame with columns:
                fraction, seed, auroc, ece, brier, ece_scaled, temperature, cri
        """
        if label_fractions is None:
            label_fractions = LABEL_FRACTIONS
        if seeds is None:
            seeds = SEEDS

        rows = []
        n_train = len(train_labels)
        idx = np.arange(n_train)

        for frac in label_fractions:
            for seed in seeds:
                # Stratified sample
                sampled = self._stratified_sample(idx, train_labels, frac, seed)
                X_tr = train_features[sampled]
                y_tr = train_labels[sampled].astype(int)

                # Train linear probe
                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                clf.fit(X_tr, y_tr)

                # Predict on test
                logits = clf.decision_function(test_features)
                probs = 1 / (1 + np.exp(-logits))
                y_test = test_labels.astype(int)

                try:
                    auroc = compute_auroc(y_test, probs)
                except Exception:
                    auroc = float("nan")

                ece_raw = compute_ece(probs, y_test)
                brier = compute_brier(y_test, probs)

                try:
                    T = temperature_scale(logits, y_test)
                    sp = 1 / (1 + np.exp(-logits / T))
                    ece_sc = compute_ece(sp, y_test)
                except Exception:
                    T = float("nan")
                    ece_sc = float("nan")

                rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "fraction": frac,
                    "seed": seed,
                    "auroc": auroc,
                    "ece": ece_raw,
                    "brier": brier,
                    "temperature": T,
                    "ece_scaled": ece_sc,
                })

        return pd.DataFrame(rows)

    def compare(self, custom_results, dataset, fraction=None):
        """
        Compare custom model results against PRISM reference models.

        Args:
            custom_results: pd.DataFrame from evaluate()
            dataset:        str - dataset name
            fraction:       float or None (if None, shows all fractions)

        Returns:
            pd.DataFrame with comparison table sorted by CRI
        """
        if self.results_dir is None:
            raise ValueError(
                "results_dir must be set to compare against reference models. "
                "Pass results_dir= when creating PRISMEvaluator."
            )

        rows = []

        # Load reference model results
        for mkey in REFERENCE_MODELS:
            df = self._load_reference(mkey, dataset)
            if df is None:
                continue
            df_ts = self._load_reference_ts(mkey, dataset)

            summary = df.groupby("fraction")[["auroc", "ece", "brier"]].mean().reset_index()
            if df_ts is not None:
                ts_summary = df_ts.groupby("fraction")[["ece_scaled"]].mean().reset_index()
                summary = summary.merge(ts_summary, on="fraction", how="left")
            else:
                summary["ece_scaled"] = float("nan")

            for _, row in summary.iterrows():
                rows.append({
                    "model": REFERENCE_MODEL_DISPLAY.get(mkey, mkey),
                    "fraction": row["fraction"],
                    "auroc": row["auroc"],
                    "ece": row["ece"],
                    "brier": row["brier"],
                    "ece_scaled": row.get("ece_scaled", float("nan")),
                })

        # Add custom model
        custom_summary = custom_results.groupby("fraction")[
            ["auroc", "ece", "brier", "ece_scaled"]
        ].mean().reset_index()

        for _, row in custom_summary.iterrows():
            rows.append({
                "model": custom_results["model"].iloc[0],
                "fraction": row["fraction"],
                "auroc": row["auroc"],
                "ece": row["ece"],
                "brier": row["brier"],
                "ece_scaled": row["ece_scaled"],
            })

        df_all = pd.DataFrame(rows)

        # Compute OOD stability per model
        ood_stab = self._compute_ood_stability()

        # Compute CRI
        df_all["ood_stability"] = df_all["model"].map(ood_stab).fillna(1.0)
        df_all["cri"] = df_all.apply(
            lambda r: compute_cri(r["auroc"], r["ece_scaled"], r["ood_stability"]),
            axis=1,
        )

        if fraction is not None:
            df_all = df_all[df_all["fraction"] == fraction]

        return (
            df_all[["model", "fraction", "auroc", "ece", "ece_scaled", "brier", "cri"]]
            .round(4)
            .sort_values(["fraction", "cri"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def _stratified_sample(self, indices, labels, fraction, seed):
        np.random.seed(seed)
        classes = np.unique(labels)
        sampled = []
        for c in classes:
            c_idx = indices[labels == c]
            n = max(1, int(len(c_idx) * fraction))
            sampled.extend(np.random.choice(c_idx, size=n, replace=False))
        return np.array(sampled)

    def _load_reference(self, mkey, dataset):
        actual_key = NAMING_EXCEPTIONS.get((mkey, dataset), mkey)
        path = os.path.join(self.results_dir, f"{actual_key}_{dataset}_results.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df["model"] = REFERENCE_MODEL_DISPLAY.get(mkey, mkey)
        return df

    def _load_reference_ts(self, mkey, dataset):
        actual_key = NAMING_EXCEPTIONS.get((mkey, dataset), mkey)
        path = os.path.join(self.results_dir, f"{actual_key}_{dataset}_temperature_scaling.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    def _compute_ood_stability(self):
        """Compute OOD stability for reference models from results CSVs."""
        if self.results_dir is None:
            return {}

        stab = {}
        for mkey in REFERENCE_MODELS:
            aurocs = []
            for ds in DATASETS:
                df = self._load_reference(mkey, ds)
                if df is not None:
                    a = df[df["fraction"] == 1.0]["auroc"].mean()
                    aurocs.append(a)
            if len(aurocs) >= 2:
                cv = np.std(aurocs) / (np.mean(aurocs) + 1e-10)
                stab[REFERENCE_MODEL_DISPLAY.get(mkey, mkey)] = max(0, 1 - cv)
            else:
                stab[REFERENCE_MODEL_DISPLAY.get(mkey, mkey)] = 1.0

        return stab
