"""
PRISMEvaluator - evaluate frozen embeddings under the PRISM reliability
protocol and compare against the eight reference foundation models.

Revision notes
--------------
v0.3.0 (camera-ready)
* ``compare`` and the OOD_Stability computation read the corrected
  ``results_v2`` layout (``indomain_all_v2.csv``, ``ood_all_v2.csv``), which
  is the source of every number in the camera-ready paper. A copy of those
  files ships inside the package (``prism_bench/reference``), so
  ``PRISMEvaluator()`` works without a repository checkout. The legacy
  per-file ``results/`` layout (as submitted, v1) is still readable, with a
  warning, for traceability only.
* Dataset keys are mapped onto the display names used in ``results_v2``
  (``'pcam' -> 'PCam'``, ``'spider_breast' -> 'SPIDER-Breast'``, ...).
* ``compare(dataset="all")`` and :meth:`PRISMEvaluator.reference_cri` pool
  the six datasets and reproduce ``results_v2/cri_variants_{1,100}.csv``.
* Binary logits: the probe's decision function ``d`` is now expanded to
  ``[-d/2, d/2]`` so that softmax at T = 1 reproduces ``predict_proba``
  exactly. The v0.2 code and the notebook that produced ``results_v2`` used
  ``[-d, d]`` (= sigmoid(2d)), so the binary ``temperature`` column in
  ``results_v2`` is on a doubled-logit scale: divide it by 2 to read it
  against the raw probe. Scaled probabilities, and hence ``ece_scaled_*``
  and CRI, are unaffected (the optimum is a reparametrisation of the same
  one-parameter family).

v0.2.0 (rebuttal)
* ``_compute_ood_stability`` previously returned ``1 - CV`` of the
  in-distribution AUROC across datasets. It now computes the mean ratio of
  OOD AUROC to in-distribution AUROC across transfer pairs, clipped to 1.
* ``evaluate`` accepts an explicit validation split; without it the
  temperature and scaled ECE are ``NaN`` rather than fitted on the test set.
* Every row carries ``degeneracy_share``.
* Class-stratified subsets in all settings.
* ECE under fixed-width and adaptive binning, 15 bins everywhere.
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Dict, Optional, Sequence

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

# dataset key -> name used in results_v2/*.csv
DATASET_DISPLAY = {
    "pcam": "PCam", "mhist": "MHIST", "crc": "CRC", "bracs": "BRACS",
    "lunghist700": "LungHist700", "spider_breast": "SPIDER-Breast",
}
_DATASET_ALIASES = {"spider": "spider_breast", "spiderbreast": "spider_breast",
                    "lunghist": "lunghist700", "crc100k": "crc",
                    "nct_crc_he_100k": "crc", "patchcamelyon": "pcam"}

OOD_PAIRS = [("pcam", "mhist"), ("mhist", "pcam"),
             ("crc", "bracs"), ("bracs", "crc")]

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [42, 123, 456]

# One legacy (v1) result file was written under a different model key.
NAMING_EXCEPTIONS = {("h_optimus_0", "pcam"): "hoptimus"}

DEGENERACY_THRESHOLD = 0.99

V2_INDOMAIN = "indomain_all_v2.csv"
V2_OOD = "ood_all_v2.csv"

_REF_COLS = ["model", "dataset", "fraction", "seed",
             "auroc", "ece_fixed", "ece_scaled_fixed"]
_OOD_COLS = ["model", "src", "tgt", "fraction", "seed", "auroc"]


def bundled_reference_dir() -> str:
    """Directory of the results_v2 reference CSVs shipped with the package."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")


def dataset_key(name: str) -> str:
    """Normalise a dataset name ('PCam', 'SPIDER-Breast', 'spider_breast',
    ...) to the package key ('pcam', 'spider_breast', ...)."""
    k = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _DATASET_ALIASES.get(k, k)


def _model_key(name: str) -> Optional[str]:
    k = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    k = {"hoptimus": "h_optimus_0", "h_optimus": "h_optimus_0"}.get(k, k)
    return k if k in DISPLAY else None


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
    """Logit matrix whose softmax equals ``clf.predict_proba`` at T = 1.

    Binary: ``decision_function`` returns d with p(1) = sigmoid(d); the
    two-column logits [-d/2, d/2] give softmax = sigmoid(d).
    """
    d = clf.decision_function(X)
    if d.ndim == 1:
        d = d.reshape(-1, 1) / 2.0
        d = np.hstack([-d, d])
    return d


# --------------------------------------------------------------------------
class PRISMEvaluator:
    """Evaluate model embeddings under the PRISM protocol.

    Parameters
    ----------
    results_dir : where the reference results live. Accepted values:

        * ``None`` (default) - the ``results_v2`` reference CSVs bundled with
          the package (``prism_bench/reference``).
        * a ``results_v2``-style directory containing ``indomain_all_v2.csv``
          and ``ood_all_v2.csv`` (e.g. ``"results_v2"`` from a repository
          checkout), or a repository root that contains ``results_v2/``.
        * a legacy ``results/`` directory (v1, as submitted;
          ``<model>_<dataset>_results.csv`` files). Supported for
          traceability only: these numbers are *not* the paper's.

        If a path whose basename is ``results_v2`` does not exist (e.g. the
        package is pip-installed and the snippet is run outside a checkout),
        the bundled copy is used and a warning is emitted.
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
        self._ref_cache: Optional[pd.DataFrame] = None
        self._ood_cache: Optional[pd.DataFrame] = None
        self._layout: Optional[str] = None
        self._ref_path: Optional[str] = None

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
        the Composite Reliability Index cannot be computed without them.
        Temperature is never fitted on ``test_features``.

        Works on any dataset. :meth:`compare` is the method restricted to the
        six PRISM benchmark datasets.
        """
        label_fractions = list(label_fractions or LABEL_FRACTIONS)
        seeds = list(seeds or SEEDS)

        train_features = np.asarray(train_features)
        test_features = np.asarray(test_features)
        train_labels = np.asarray(train_labels).astype(int)
        test_labels = np.asarray(test_labels).astype(int)
        if len(train_features) != len(train_labels):
            raise ValueError("train_features and train_labels differ in length")
        if len(test_features) != len(test_labels):
            raise ValueError("test_features and test_labels differ in length")
        n_classes = len(np.unique(train_labels))

        if (val_features is None) != (val_labels is None):
            raise ValueError("pass both val_features and val_labels, or neither")
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
            val_features = np.asarray(val_features)
            val_labels = np.asarray(val_labels).astype(int)
            if len(val_features) != len(val_labels):
                raise ValueError("val_features and val_labels differ in length")

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
        custom_ood_stability: Optional[float] = None,
    ) -> pd.DataFrame:
        """Rank a custom model against the eight reference FMs by the
        Composite Reliability Index (CRI).

        Parameters
        ----------
        custom_results : DataFrame returned by :meth:`evaluate`.
        dataset : one of the six PRISM datasets (any spelling accepted by
            :func:`dataset_key`, e.g. ``'pcam'``, ``'PCam'``,
            ``'SPIDER-Breast'``), or ``'all'`` to pool the six datasets as
            in ``results_v2/cri_variants_*.csv``.
        fraction : keep only this label fraction (``None`` = all).
        ece_column : column of ``custom_results`` used as ECE_scaled.
        aggregation : CRI combination rule (see :func:`compute_cri`).
        custom_ood_stability : OOD_Stability of the custom model if you have
            measured it on the PRISM transfer pairs. Otherwise the mean of
            the eight reference models is used and the row is flagged
            ``ood_stability_imputed=True``.

        Reference rows use ECE after temperature scaling with a held-out
        validation split (``ece_scaled_fixed`` in ``results_v2``).
        """
        pooled = str(dataset).strip().lower() == "all"
        dkey = None
        if not pooled:
            dkey = dataset_key(dataset)
            if dkey not in DATASETS:
                raise ValueError(
                    f"Reference results exist only for {DATASETS} (or 'all'). "
                    f"'{dataset}' is not a PRISM benchmark dataset; use "
                    f"evaluate() alone for custom datasets."
                )

        ref = self._reference_table()
        stability = self._ood_stability()

        sub = ref if pooled else ref[ref["dataset"] == dkey]
        if sub.empty:
            raise FileNotFoundError(
                f"No reference rows for dataset '{dataset}' in "
                f"{self._ref_path!r}.")
        agg = (sub.groupby(["model", "fraction"])
               [["auroc", "ece_fixed", "ece_scaled_fixed"]].mean()
               .reset_index())
        rows = []
        for r in agg.itertuples():
            rows.append({
                "model": r.model,
                "fraction": float(r.fraction),
                "auroc": r.auroc,
                "ece": r.ece_fixed,
                "ece_scaled": r.ece_scaled_fixed,
                "ood_stability": stability.get(r.model, np.nan),
                "is_reference": True,
            })

        if "model" not in custom_results.columns:
            raise ValueError("custom_results needs a 'model' column "
                             "(as returned by evaluate()).")
        if pooled and "dataset" in custom_results.columns:
            n_ds = custom_results["dataset"].map(dataset_key).nunique()
            if n_ds < len(DATASETS):
                warnings.warn(
                    f"compare(dataset='all') pools references over "
                    f"{len(DATASETS)} datasets but custom_results covers "
                    f"{n_ds}; the custom row is not directly comparable.",
                    UserWarning)
        if ece_column not in custom_results.columns:
            raise KeyError(f"'{ece_column}' not in custom_results")
        cs = custom_results.groupby("fraction").mean(numeric_only=True)
        cname = str(custom_results["model"].iloc[0])
        for frac, r in cs.iterrows():
            rows.append({
                "model": cname,
                "fraction": float(frac),
                "auroc": r.get("auroc", np.nan),
                "ece": r.get("ece_fixed", np.nan),
                "ece_scaled": r.get(ece_column, np.nan),
                "ood_stability": (np.nan if custom_ood_stability is None
                                  else float(custom_ood_stability)),
                "is_reference": False,
            })

        out = pd.DataFrame(rows)

        # A model with no measured transfer performance is scored on the
        # reference mean, and flagged, rather than silently dropped.
        ref_vals = [v for v in stability.values() if np.isfinite(v)]
        ref_mean = float(np.mean(ref_vals)) if ref_vals else np.nan
        out["ood_stability_imputed"] = out["ood_stability"].isna()
        out["ood_stability"] = out["ood_stability"].fillna(ref_mean)

        out["cri"] = [
            compute_cri(r.auroc, r.ece_scaled, r.ood_stability, aggregation)
            if np.isfinite([r.auroc, r.ece_scaled, r.ood_stability]).all()
            else np.nan
            for r in out.itertuples()
        ]

        if fraction is not None:
            out = out[np.isclose(out["fraction"], float(fraction))]
            if out.empty:
                raise ValueError(f"no rows at fraction={fraction}")

        return (out[["model", "fraction", "auroc", "ece", "ece_scaled",
                     "ood_stability", "ood_stability_imputed", "cri",
                     "is_reference"]]
                .round(4)
                .sort_values(["fraction", "cri"], ascending=[True, False])
                .reset_index(drop=True))

    def reference_cri(self, fraction: float = 1.0) -> pd.DataFrame:
        """Model-level CRI of the eight reference FMs, pooled over the six
        datasets at one label fraction, under all four aggregation rules.

        Reproduces ``results_v2/cri_variants_100.csv`` (``fraction=1.0``) and
        ``cri_variants_1.csv`` (``fraction=0.01``): AUROC and ECE_scaled are
        averaged over datasets and seeds, OOD_Stability is evaluated at full
        supervision, and the index is computed from those means.
        """
        ref = self._reference_table()
        sub = ref[np.isclose(ref["fraction"], float(fraction))]
        if sub.empty:
            raise ValueError(f"no reference rows at fraction={fraction}")
        stab = self._ood_stability()
        g = sub.groupby("model")[["auroc", "ece_scaled_fixed"]].mean()
        rows = []
        for mkey in REFERENCE_MODELS:
            m = DISPLAY[mkey]
            if m not in g.index:
                continue
            a, e, s = g.loc[m, "auroc"], g.loc[m, "ece_scaled_fixed"], stab.get(m, np.nan)
            row = {"model": m, "auroc": a, "ece_scaled": e, "ood_stability": s}
            for agg in ("multiplicative", "arithmetic", "geometric", "worst_axis"):
                row[f"cri_{agg}"] = (compute_cri(a, e, s, agg)
                                     if np.isfinite([a, e, s]).all() else np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def reference_layout(self) -> str:
        """'v2' (results_v2 / bundled) or 'legacy_v1' (as-submitted results/)."""
        self._locate()
        return self._layout

    @property
    def reference_path(self) -> str:
        self._locate()
        return self._ref_path

    # -------------------------------------------------------------- private
    def _locate(self) -> None:
        if self._layout is not None:
            return
        rd = self.results_dir
        if rd is None:
            path = bundled_reference_dir()
        else:
            path = os.path.abspath(os.path.expanduser(str(rd)))
            if not os.path.isdir(path):
                if os.path.basename(path.rstrip(os.sep)) == "results_v2":
                    warnings.warn(
                        f"results_dir {rd!r} not found; using the results_v2 "
                        f"reference CSVs bundled with prism-bench "
                        f"({bundled_reference_dir()}).", UserWarning)
                    path = bundled_reference_dir()
                else:
                    raise FileNotFoundError(
                        f"results_dir {rd!r} does not exist. Use "
                        f"PRISMEvaluator() for the bundled results_v2 "
                        f"reference, or point results_dir at a results_v2 "
                        f"directory.")
            elif (not os.path.exists(os.path.join(path, V2_INDOMAIN))
                  and os.path.exists(os.path.join(path, "results_v2", V2_INDOMAIN))):
                path = os.path.join(path, "results_v2")   # repository root

        if os.path.exists(os.path.join(path, V2_INDOMAIN)):
            self._layout = "v2"
        elif glob.glob(os.path.join(path, "*_results.csv")):
            self._layout = "legacy_v1"
            warnings.warn(
                f"{path!r} uses the legacy per-file layout of the submitted "
                f"version (results/, v1). These are NOT the corrected numbers "
                f"reported in the paper; use results_v2 instead.", UserWarning)
        else:
            raise FileNotFoundError(
                f"No reference results found in {path!r}: expected "
                f"{V2_INDOMAIN} (results_v2 layout) or "
                f"'<model>_<dataset>_results.csv' files (legacy layout).")
        self._ref_path = path

    def _reference_table(self) -> pd.DataFrame:
        """Long table [model, dataset(key), fraction, seed, auroc,
        ece_fixed, ece_scaled_fixed] for all reference models/datasets."""
        if self._ref_cache is not None:
            return self._ref_cache
        self._locate()
        if self._layout == "v2":
            df = pd.read_csv(os.path.join(self._ref_path, V2_INDOMAIN))
            df = df.assign(dataset=df["dataset"].map(dataset_key))
            df = df[df["model"].isin(DISPLAY.values())
                    & df["dataset"].isin(DATASETS)]
            out = df[_REF_COLS].copy()
        else:
            parts = []
            for mkey in REFERENCE_MODELS:
                for dkey in DATASETS:
                    stem = f"{_resolve(mkey, dkey)}_{dkey}"
                    p = os.path.join(self._ref_path, f"{stem}_results.csv")
                    if not os.path.exists(p):
                        continue
                    r = pd.read_csv(p)
                    r = r.rename(columns={"ece": "ece_fixed"})
                    if "ece_fixed" not in r:
                        r["ece_fixed"] = np.nan
                    ts_p = os.path.join(self._ref_path,
                                        f"{stem}_temperature_scaling.csv")
                    r["ece_scaled_fixed"] = np.nan
                    if os.path.exists(ts_p):
                        ts = pd.read_csv(ts_p)
                        col = next((c for c in ("ece_scaled", "ece_scaled_fixed")
                                    if c in ts.columns), None)
                        if col is not None:
                            keys = [k for k in ("fraction", "seed")
                                    if k in ts.columns and k in r.columns]
                            m = ts[keys + [col]].rename(columns={col: "_s"})
                            r = r.drop(columns="ece_scaled_fixed").merge(
                                m, on=keys, how="left").rename(
                                columns={"_s": "ece_scaled_fixed"})
                    r["model"] = DISPLAY[mkey]
                    r["dataset"] = dkey
                    if "seed" not in r:
                        r["seed"] = -1
                    parts.append(r[_REF_COLS])
            out = (pd.concat(parts, ignore_index=True) if parts
                   else pd.DataFrame(columns=_REF_COLS))
        out["fraction"] = out["fraction"].astype(float).round(4)
        self._ref_cache = out
        return out

    def _ood_table(self) -> pd.DataFrame:
        """Long table [model, src(key), tgt(key), fraction, seed, auroc]."""
        if self._ood_cache is not None:
            return self._ood_cache
        self._locate()
        if self._layout == "v2":
            p = os.path.join(self._ref_path, V2_OOD)
            if os.path.exists(p):
                df = pd.read_csv(p)
                df = df.assign(src=df["src"].map(dataset_key),
                               tgt=df["tgt"].map(dataset_key))
                out = df[_OOD_COLS].copy()
            else:
                out = pd.DataFrame(columns=_OOD_COLS)
        else:
            parts = []
            for mkey in REFERENCE_MODELS:
                for src, tgt in OOD_PAIRS:
                    p = os.path.join(self._ref_path,
                                     f"{mkey}_ood_{src}_to_{tgt}_results.csv")
                    if not os.path.exists(p):
                        continue
                    r = pd.read_csv(p)
                    r["model"], r["src"], r["tgt"] = DISPLAY[mkey], src, tgt
                    if "seed" not in r:
                        r["seed"] = -1
                    parts.append(r[_OOD_COLS])
            out = (pd.concat(parts, ignore_index=True) if parts
                   else pd.DataFrame(columns=_OOD_COLS))
        out["fraction"] = out["fraction"].astype(float).round(4)
        self._ood_cache = out
        return out

    def _ood_stability(self) -> Dict[str, float]:
        """OOD_Stability per Equation 1: mean over the four transfer pairs of
        (OOD AUROC / in-distribution AUROC of the source dataset), clipped to
        1 from above, both at full supervision (fraction 1.0, mean over
        seeds).

        On the CRC->BRACS and BRACS->CRC pairs the numerator is the binary
        OOD AUROC of the transferred task while the denominator is the
        source's 9-/7-class macro AUROC.
        """
        if self._stability_cache is not None:
            return self._stability_cache

        ref = self._reference_table()
        ood = self._ood_table()
        full_id = (ref[np.isclose(ref["fraction"], 1.0)]
                   .groupby(["model", "dataset"])["auroc"].mean())
        full_ood = (ood[np.isclose(ood["fraction"], 1.0)]
                    .groupby(["model", "src", "tgt"])["auroc"].mean())

        out = {}
        for mkey in REFERENCE_MODELS:
            m = DISPLAY[mkey]
            id_auroc = {d: float(v) for (mm, d), v in full_id.items() if mm == m}
            ood_auroc = {(s, t): float(v)
                         for (mm, s, t), v in full_ood.items() if mm == m}
            out[m] = compute_ood_stability(ood_auroc, id_auroc)

        if all(not np.isfinite(v) for v in out.values()):
            warnings.warn(
                "No OOD results found; OOD_Stability cannot be computed and "
                "CRI will be NaN. Expected ood_all_v2.csv (results_v2) or "
                "'<model>_ood_<src>_to_<tgt>_results.csv' (legacy).",
                UserWarning,
            )

        self._stability_cache = out
        return out
