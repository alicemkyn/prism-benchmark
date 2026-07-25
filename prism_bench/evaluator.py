"""
PRISMEvaluator - evaluate frozen embeddings under the PRISM reliability
protocol and compare against the eight reference foundation models.

Revision notes (rebuttal revision)
----------------------------------
* ``_compute_ood_stability`` previously returned ``1 - CV`` of the
  *in-distribution* AUROC across datasets. That measures cross-dataset
  variance, not transfer retention, and does not match Equation 1 of the
  paper. It now computes the mean ratio of OOD AUROC to in-distribution
  AUROC across transfer pairs, clipped to 1 from above, reading the shipped
  OOD result files.
* ``evaluate`` now accepts an explicit validation split. Temperature scaling
  requires held-out data; without it the returned temperature and scaled ECE
  are ``NaN`` rather than silently fitted on the evaluation set.
* Every row now carries ``degeneracy_share``: the proportion of test samples
  assigned to the single most-predicted class. Values near 1 mean the probe
  has collapsed and ranking metrics should be read with care.
* Subsets are drawn with class-stratified sampling in all settings. The
  submitted version used class-agnostic sampling in-distribution and
  stratified sampling only for transfer experiments.
* ECE is reported under both fixed-width and adaptive binning, with one
  consistent bin count (15) everywhere.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

from .metrics import (
    compute_ece,
    temperature_scale,
    apply_temperature,
    compute_cri,
    compute_ood_stability,
    DEFAULT_N_BINS,
)

# --------------------------------------------------------------------------
# benchmark constants
# --------------------------------------------------------------------------
REFERENCE_MODELS = ["clip", "plip", "conch", "virchow2",
                    "uni", "gigapath", "h_optimus_0", "midnight"]

DISPLAY = {
    "clip": "CLIP", "plip": "PLIP", "conch": "CONCH", "virchow2": "VIRCHOW2",
    "uni": "UNI", "gigapath": "GigaPath", "h_optimus_0": "H-Optimus-0",
    "midnight": "MIDNIGHT",
}

DATASETS = ["pcam", "mhist", "crc", "bracs", "lunghist700", "spider_breast"]

OOD_PAIRS = [("pcam", "mhist"), ("mhist", "pcam"),
             ("crc", "bracs"), ("bracs", "crc")]

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [42, 123, 456]

# One result file was written under a legacy name.
NAMING_EXCEPTIONS = {("h_optimus_0", "pcam"): "hoptimus"}

DEGENERACY_THRESHOLD = 0.99


def _resolve(mkey: str, dkey: str) -> str:
    return NAMING_EXCEPTIONS.get((mkey, dkey), mkey)


def stratified_sample(labels: np.ndarray, fraction: float, seed: int):
    """Class-stratified subset.

    Returns the selected indices and the list of classes for which the
    exact quota fell below one sample and a single example had to be forced.
    A non-empty second return value is a warning sign: at that fraction the
    subset composition is dictated by the guarantee rather than by the data.
    """
    np.random.seed(seed)
    idx_all = np.arange(len(labels))
    picked, forced = [], []
    for c in np.unique(labels):
        c_idx = idx_all[labels == c]
        exact = len(c_idx) * fraction
        n = max(1, int(exact))
        if exact < 1:
            forced.append(int(c))
        picked.extend(np.random.choice(c_idx, size=n, replace=False))
    return np.array(sorted(picked)), forced


def _degeneracy(pred: np.ndarray, n_classes: int):
    counts = np.bincount(pred, minlength=n_classes)
    return float(counts.max() / counts.sum())


def _auroc(y_true, proba):
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba,
                                   multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _logits(clf, X):
    d = clf.decision_function(X)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
        d = np.hstack([-d, d])
    return d


# --------------------------------------------------------------------------
class PRISMEvaluator:
    """Evaluate model embeddings under the PRISM protocol.

    Parameters
    ----------
    results_dir : path to the shipped reference results. Required for
        :meth:`compare`; optional for :meth:`evaluate`.
    n_bins : ECE bin count (default 15, used for both binning schemes).

    Notes
    -----
    PRISM operates on embeddings, not models. Extract features once with
    your own model and preprocessing; everything here runs on CPU.
    """

    def __init__(self, results_dir: Optional[str] = None,
                 n_bins: int = DEFAULT_N_BINS):
        self.results_dir = results_dir
        self.n_bins = n_bins
        self._stability_cache: Optional[dict] = None

    # ---------------------------------------------------------------- eval
    def evaluate(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        dataset: str,
        model_name: str = "CustomModel",
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        label_fractions: Optional[Sequence[float]] = None,
        seeds: Optional[Sequence[int]] = None,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> pd.DataFrame:
        """Train linear probes across label fractions and seeds.

        ``val_features`` / ``val_labels`` are used to fit the temperature.
        If omitted, ``temperature`` and the scaled ECE columns are ``NaN``;
        CRI cannot be computed without them. Temperature is never fitted on
        ``test_features``.

        Works on any dataset. :meth:`compare` is the method restricted to the
        six PRISM benchmark datasets.
        """
        label_fractions = list(label_fractions or LABEL_FRACTIONS)
        seeds = list(seeds or SEEDS)

        train_labels = np.asarray(train_labels).astype(int)
        test_labels = np.asarray(test_labels).astype(int)
        n_classes = len(np.unique(train_labels))

        if val_features is None:
            warnings.warn(
                "No validation split supplied: temperature scaling is "
                "skipped and CRI cannot be computed. Pass val_features and "
                "val_labels to enable calibration. Fitting the temperature "
                "on the test split would produce an oracle estimate, not a "
                "deployable one.",
                UserWarning,
            )
        else:
            val_labels = np.asarray(val_labels).astype(int)

        rows = []
        for frac in label_fractions:
            for seed in seeds:
                idx, forced = stratified_sample(train_labels, frac, seed)
                clf = LogisticRegression(max_iter=max_iter, C=C,
                                         random_state=seed)
                clf.fit(train_features[idx], train_labels[idx])

                proba = clf.predict_proba(test_features)
                pred = proba.argmax(axis=1)

                if val_features is not None:
                    T = temperature_scale(_logits(clf, val_features), val_labels)
                    scaled = apply_temperature(_logits(clf, test_features), T)
                    ece_s_fix = compute_ece(scaled, test_labels,
                                            self.n_bins, "fixed")
                    ece_s_ada = compute_ece(scaled, test_labels,
                                            self.n_bins, "adaptive")
                else:
                    T = ece_s_fix = ece_s_ada = float("nan")

                rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "fraction": frac,
                    "seed": seed,
                    "n_train": len(idx),
                    "n_classes": n_classes,
                    "auroc": _auroc(test_labels, proba),
                    "f1_macro": float(f1_score(test_labels, pred,
                                               average="macro",
                                               zero_division=0)),
                    "brier": (float(brier_score_loss(test_labels, proba[:, 1]))
                              if n_classes == 2 else float("nan")),
                    "ece_fixed": compute_ece(proba, test_labels,
                                             self.n_bins, "fixed"),
                    "ece_adaptive": compute_ece(proba, test_labels,
                                                self.n_bins, "adaptive"),
                    "temperature": T,
                    "ece_scaled_fixed": ece_s_fix,
                    "ece_scaled_adaptive": ece_s_ada,
                    "degeneracy_share": _degeneracy(pred, n_classes),
                    "forced_classes": len(forced),
                })

        df = pd.DataFrame(rows)
        df["degenerate"] = df["degeneracy_share"] > DEGENERACY_THRESHOLD
        if df["degenerate"].any():
            bad = sorted(df.loc[df["degenerate"], "fraction"].unique())
            warnings.warn(
                f"Probe collapsed to a single predicted class at label "
                f"fractions {bad}. AUROC may still be informative but F1 and "
                f"ECE at those fractions describe a degenerate classifier.",
                UserWarning,
            )
        return df

    # ------------------------------------------------------------- compare
    def compare(
        self,
        custom_results: pd.DataFrame,
        dataset: str,
        fraction: Optional[float] = None,
        ece_column: str = "ece_scaled_fixed",
        aggregation: str = "multiplicative",
    ) -> pd.DataFrame:
        """Rank a custom model against the eight reference FMs by CRI.

        Requires ``results_dir`` and one of the six PRISM datasets, since
        reference results exist only for those.
        """
        if self.results_dir is None:
            raise ValueError(
                "results_dir must be set to compare against the reference "
                "models: PRISMEvaluator(results_dir='path/to/results')."
            )
        dkey = dataset.lower().replace("-", "_")
        if dkey not in DATASETS:
            raise ValueError(
                f"Reference results exist only for {DATASETS}. "
                f"'{dataset}' is not a PRISM benchmark dataset; use "
                f"evaluate() alone for custom datasets."
            )

        stability = self._ood_stability()
        rows = []

        for mkey in REFERENCE_MODELS:
            ref = self._load_reference(mkey, dkey)
            if ref is None:
                continue
            g = ref.groupby("fraction")
            for frac, sub in g:
                rows.append({
                    "model": DISPLAY[mkey],
                    "fraction": float(frac),
                    "auroc": sub["auroc"].mean(),
                    "ece": sub["ece"].mean() if "ece" in sub else np.nan,
                    "ece_scaled": self._ref_scaled_ece(mkey, dkey, frac),
                    "ood_stability": stability.get(DISPLAY[mkey], np.nan),
                    "is_reference": True,
                })

        cs = custom_results.groupby("fraction").mean(numeric_only=True)
        cname = str(custom_results["model"].iloc[0])
        for frac, sub in cs.iterrows():
            rows.append({
                "model": cname,
                "fraction": float(frac),
                "auroc": sub.get("auroc", np.nan),
                "ece": sub.get("ece_fixed", np.nan),
                "ece_scaled": sub.get(ece_column, np.nan),
                "ood_stability": np.nan,   # unknown for an unseen model
                "is_reference": False,
            })

        out = pd.DataFrame(rows)

        # A model with no measured transfer performance is scored on the
        # reference mean, and flagged, rather than silently dropped.
        ref_mean = np.nanmean(list(stability.values())) if stability else np.nan
        out["ood_stability_imputed"] = out["ood_stability"].isna()
        out["ood_stability"] = out["ood_stability"].fillna(ref_mean)

        out["cri"] = [
            compute_cri(r.auroc, r.ece_scaled, r.ood_stability, aggregation)
            if np.isfinite([r.auroc, r.ece_scaled, r.ood_stability]).all()
            else np.nan
            for r in out.itertuples()
        ]

        if fraction is not None:
            out = out[np.isclose(out["fraction"], fraction)]

        return (out[["model", "fraction", "auroc", "ece", "ece_scaled",
                     "ood_stability", "ood_stability_imputed", "cri"]]
                .round(4)
                .sort_values(["fraction", "cri"], ascending=[True, False])
                .reset_index(drop=True))

    # -------------------------------------------------------------- private
    def _load_reference(self, mkey: str, dkey: str) -> Optional[pd.DataFrame]:
        p = os.path.join(self.results_dir,
                         f"{_resolve(mkey, dkey)}_{dkey}_results.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    def _load_reference_ts(self, mkey: str, dkey: str) -> Optional[pd.DataFrame]:
        p = os.path.join(self.results_dir,
                         f"{_resolve(mkey, dkey)}_{dkey}_temperature_scaling.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    def _load_ood(self, mkey: str, src: str, tgt: str) -> Optional[pd.DataFrame]:
        p = os.path.join(self.results_dir,
                         f"{mkey}_ood_{src}_to_{tgt}_results.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    def _ref_scaled_ece(self, mkey: str, dkey: str, frac: float) -> float:
        ts = self._load_reference_ts(mkey, dkey)
        if ts is None:
            return float("nan")
        col = ("ece_scaled" if "ece_scaled" in ts.columns
               else ("ece_scaled_fixed" if "ece_scaled_fixed" in ts.columns
                     else None))
        if col is None:
            return float("nan")
        sub = ts[np.isclose(ts["fraction"], frac)]
        return float(sub[col].mean()) if len(sub) else float("nan")

    def _ood_stability(self) -> dict:
        """OOD_Stability per Equation 1: mean over transfer pairs of
        (OOD AUROC / in-distribution AUROC), clipped to 1 from above,
        evaluated at full supervision.

        The submitted package computed 1 - CV of the in-distribution AUROC
        instead. This is the corrected implementation.
        """
        if self._stability_cache is not None:
            return self._stability_cache

        out = {}
        for mkey in REFERENCE_MODELS:
            id_auroc, ood_auroc = {}, {}

            for dkey in DATASETS:
                ref = self._load_reference(mkey, dkey)
                if ref is None:
                    continue
                full = ref[np.isclose(ref["fraction"], 1.0)]
                if len(full):
                    id_auroc[dkey] = float(full["auroc"].mean())

            for src, tgt in OOD_PAIRS:
                od = self._load_ood(mkey, src, tgt)
                if od is None:
                    continue
                full = od[np.isclose(od["fraction"], 1.0)]
                if len(full):
                    ood_auroc[(src, tgt)] = float(full["auroc"].mean())

            out[DISPLAY[mkey]] = compute_ood_stability(ood_auroc, id_auroc)

        if all(not np.isfinite(v) for v in out.values()):
            warnings.warn(
                "No OOD result files found in results_dir; OOD_Stability "
                "cannot be computed and CRI will be NaN. Expected files named "
                "'<model>_ood_<src>_to_<tgt>_results.csv'.",
                UserWarning,
            )

        self._stability_cache = out
        return out
