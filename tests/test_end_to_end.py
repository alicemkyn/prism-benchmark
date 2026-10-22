"""End-to-end tests for prism-bench on synthetic embeddings.

Run from the repository root:  python -m pytest -q
The results_v2 tests use the repository's results_v2/ directory when it is
present and otherwise the copy bundled with the package.
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prism_bench
from prism_bench import PRISMEvaluator, bundled_reference_dir, dataset_key
from prism_bench.evaluator import DATASET_DISPLAY, _logits
from prism_bench.metrics import temperature_scale, apply_temperature

REPO = Path(__file__).resolve().parents[1]
V2_DIR = REPO / "results_v2"
REF_DIR = V2_DIR if (V2_DIR / "indomain_all_v2.csv").exists() else Path(bundled_reference_dir())
LEGACY_DIR = REPO / "results"

FRACTIONS = [0.1, 1.0]
SEEDS = [42, 123]


def _synthetic(n, n_classes=2, dim=32, shift=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    centers = rng.normal(size=(n_classes, dim))
    X = rng.normal(size=(n, dim)) + shift * centers[y]
    X /= np.linalg.norm(X, axis=1, keepdims=True)    # PRISM uses L2-normalised features
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def binary_data():
    X, y = _synthetic(1800, 2, seed=1)
    return dict(X_train=X[:1000], y_train=y[:1000],
                X_val=X[1000:1300], y_val=y[1000:1300],
                X_test=X[1300:], y_test=y[1300:])


@pytest.fixture(scope="module")
def results_with_val(binary_data):
    d = binary_data
    ev = PRISMEvaluator(results_dir=str(REF_DIR))
    return ev.evaluate(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                       dataset="pcam", model_name="MyFM",
                       val_features=d["X_val"], val_labels=d["y_val"],
                       label_fractions=FRACTIONS, seeds=SEEDS)


# ------------------------------------------------------------------ version
def test_version_is_consistent():
    assert prism_bench.__version__ == "0.3.0"
    pyproject = (REPO / "pyproject.toml").read_text()
    assert 'version = "0.3.0"' in pyproject
    assert "version-0.3.0" in (REPO / "README.md").read_text()


# ----------------------------------------------------------------- evaluate
EXPECTED_COLUMNS = {"model", "dataset", "fraction", "seed", "n_train",
                    "n_classes", "auroc", "f1_macro", "brier", "ece_fixed",
                    "ece_adaptive", "temperature", "ece_scaled_fixed",
                    "ece_scaled_adaptive", "degeneracy_share", "degenerate",
                    "forced_classes"}


def test_evaluate_with_val(results_with_val):
    r = results_with_val
    assert EXPECTED_COLUMNS <= set(r.columns)
    assert len(r) == len(FRACTIONS) * len(SEEDS)
    assert r["temperature"].notna().all()
    assert r["ece_scaled_fixed"].between(0, 1).all()
    assert r["auroc"].between(0.5, 1.0).all()


def test_evaluate_without_val(binary_data):
    d = binary_data
    with pytest.warns(UserWarning, match="No validation split"):
        r = PRISMEvaluator().evaluate(
            d["X_train"], d["y_train"], d["X_test"], d["y_test"],
            dataset="pcam", model_name="NoVal",
            label_fractions=FRACTIONS, seeds=SEEDS)
    assert r["temperature"].isna().all()
    assert r["ece_scaled_fixed"].isna().all()
    assert r["ece_fixed"].notna().all()


def test_evaluate_multiclass_with_val():
    X, y = _synthetic(1500, 5, seed=2)
    r = PRISMEvaluator().evaluate(X[:900], y[:900], X[1200:], y[1200:],
                                  dataset="crc", model_name="MC",
                                  val_features=X[900:1200], val_labels=y[900:1200],
                                  label_fractions=[1.0], seeds=[42])
    assert r["n_classes"].iloc[0] == 5
    assert np.isnan(r["brier"].iloc[0])
    assert np.isfinite(r["ece_scaled_fixed"].iloc[0])


def test_val_features_without_labels_rejected(binary_data):
    d = binary_data
    with pytest.raises(ValueError):
        PRISMEvaluator().evaluate(d["X_train"], d["y_train"], d["X_test"],
                                  d["y_test"], dataset="pcam",
                                  val_features=d["X_val"])


def test_binary_logits_reproduce_predict_proba(binary_data):
    from sklearn.linear_model import LogisticRegression
    d = {k: (v.astype(np.float64) if k.startswith("X") else v)
         for k, v in binary_data.items()}
    clf = LogisticRegression(max_iter=1000).fit(d["X_train"], d["y_train"])
    p_T1 = apply_temperature(_logits(clf, d["X_test"]), 1.0)
    np.testing.assert_allclose(p_T1, clf.predict_proba(d["X_test"]), atol=1e-10)
    # 1-D decision-function input follows the same convention
    p_1d = apply_temperature(clf.decision_function(d["X_test"]), 1.0)
    np.testing.assert_allclose(p_1d, clf.predict_proba(d["X_test"]), atol=1e-10)
    assert 0.1 <= temperature_scale(clf.decision_function(d["X_val"]), d["y_val"]) <= 10


# ------------------------------------------------------------------ compare
def test_dataset_key_mapping():
    for k, disp in DATASET_DISPLAY.items():
        assert dataset_key(disp) == k
        assert dataset_key(k) == k
    assert dataset_key("SPIDER-Breast") == "spider_breast"
    assert dataset_key("PCam") == "pcam"


def test_appendix_e_snippet(binary_data, monkeypatch):
    """The exact Appendix E snippet, run from the repository root."""
    monkeypatch.chdir(REPO)
    d = binary_data
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # degeneracy / bundled fallback
        ev = PRISMEvaluator(results_dir="results_v2")
        res = ev.evaluate(X_train, y_train, X_test, y_test, dataset="pcam",
                          model_name="MyFM", val_features=X_val, val_labels=y_val)
        cmp = ev.compare(res, dataset="pcam", fraction=0.1)
    assert ev.reference_layout == "v2"
    assert len(cmp) == 9
    assert set(cmp["model"]) == {"CLIP", "PLIP", "CONCH", "VIRCHOW2", "UNI",
                                 "GigaPath", "H-Optimus-0", "MIDNIGHT", "MyFM"}
    assert cmp["cri"].notna().all()
    assert cmp["cri"].is_monotonic_decreasing
    mine = cmp[cmp["model"] == "MyFM"].iloc[0]
    assert bool(mine["ood_stability_imputed"]) and not bool(mine["is_reference"])


def test_compare_reference_rows_match_results_v2(results_with_val):
    ev = PRISMEvaluator(results_dir=str(REF_DIR))
    ind = pd.read_csv(REF_DIR / "indomain_all_v2.csv")
    comp = pd.read_csv(REF_DIR / "cri_components.csv").set_index("model")
    for dkey in ["pcam", "spider_breast"]:
        cmp = ev.compare(results_with_val, dataset=dkey, fraction=1.0)
        sub = ind[(ind["dataset"] == DATASET_DISPLAY[dkey]) & (ind["fraction"] == 1.0)]
        exp = sub.groupby("model")[["auroc", "ece_scaled_fixed"]].mean()
        ref = cmp[cmp["is_reference"]].set_index("model")
        assert len(ref) == 8
        for m, row in ref.iterrows():
            assert abs(row["auroc"] - exp.loc[m, "auroc"]) < 1e-4
            assert abs(row["ece_scaled"] - exp.loc[m, "ece_scaled_fixed"]) < 1e-4
            assert abs(row["ood_stability"] - comp.loc[m, "ood_stability"]) < 1e-4
            cri = exp.loc[m, "auroc"] * (1 - exp.loc[m, "ece_scaled_fixed"]) * comp.loc[m, "ood_stability"]
            assert abs(row["cri"] - cri) < 1e-4


@pytest.mark.parametrize("fraction,fname", [(1.0, "cri_variants_100.csv"),
                                            (0.01, "cri_variants_1.csv")])
def test_reference_cri_matches_cri_variants(fraction, fname):
    ev = PRISMEvaluator(results_dir=str(REF_DIR))
    got = ev.reference_cri(fraction).set_index("model")
    exp = pd.read_csv(REF_DIR / fname).set_index("model").loc[got.index]
    assert len(got) == 8
    pd.testing.assert_frame_equal(got[exp.columns], exp, atol=1e-9, rtol=0,
                                  check_names=False)


def test_compare_all_pooled_matches_cri_variants(results_with_val):
    ev = PRISMEvaluator(results_dir=str(REF_DIR))
    with pytest.warns(UserWarning, match="pools references"):
        cmp = ev.compare(results_with_val, dataset="all", fraction=1.0)
    ref = cmp[cmp["is_reference"]].set_index("model")["cri"]
    exp = pd.read_csv(REF_DIR / "cri_variants_100.csv").set_index("model")["cri_multiplicative"]
    assert (ref - exp.loc[ref.index]).abs().max() < 5e-5     # compare() rounds to 4 dp


def test_bundled_reference_is_default_and_identical(results_with_val):
    ev = PRISMEvaluator()
    assert ev.reference_layout == "v2"
    assert Path(ev.reference_path) == Path(bundled_reference_dir())
    a = ev.compare(results_with_val, dataset="PCam", fraction=0.1)
    b = PRISMEvaluator(results_dir=str(REF_DIR)).compare(results_with_val, dataset="pcam", fraction=0.1)
    pd.testing.assert_frame_equal(a, b)


def test_missing_results_v2_falls_back_to_bundled(tmp_path, monkeypatch, results_with_val):
    monkeypatch.chdir(tmp_path)
    ev = PRISMEvaluator(results_dir="results_v2")
    with pytest.warns(UserWarning, match="bundled"):
        cmp = ev.compare(results_with_val, dataset="pcam", fraction=0.1)
    assert len(cmp) == 9


@pytest.mark.skipif(not (LEGACY_DIR / "clip_pcam_results.csv").exists(),
                    reason="legacy results/ not present")
def test_legacy_layout_fallback(results_with_val):
    ev = PRISMEvaluator(results_dir=str(LEGACY_DIR))
    with pytest.warns(UserWarning, match="legacy"):
        cmp = ev.compare(results_with_val, dataset="pcam", fraction=1.0)
    assert ev.reference_layout == "legacy_v1"
    assert (cmp["is_reference"]).sum() == 8
    assert cmp["cri"].notna().all()


def test_compare_rejects_unknown_dataset(results_with_val):
    with pytest.raises(ValueError, match="not a PRISM benchmark dataset"):
        PRISMEvaluator().compare(results_with_val, dataset="my_private_set")


# ---------------------------------------------------------------------- CLI
def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "prism_bench", *args],
                          cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=600)


@pytest.fixture(scope="module")
def npy_files(tmp_path_factory, binary_data):
    d = tmp_path_factory.mktemp("emb")
    for k, v in binary_data.items():
        np.save(d / f"{k}.npy", v)
    return d


def test_cli_evaluate_with_val_and_compare(npy_files):
    d = npy_files
    out = d / "myfm_pcam_results.csv"
    r = _run_cli(["evaluate",
                  "--train-features", "X_train.npy", "--train-labels", "y_train.npy",
                  "--test-features", "X_test.npy", "--test-labels", "y_test.npy",
                  "--val-features", "X_val.npy", "--val-labels", "y_val.npy",
                  "--dataset", "pcam", "--model-name", "MyFM",
                  "--label-fractions", "0.1,1.0", "--seeds", "42,123",
                  "--output", str(out)], cwd=d)
    assert r.returncode == 0, r.stderr
    assert "ece_scaled_fixed" in r.stdout and "KeyError" not in r.stderr
    res = pd.read_csv(out)
    assert res["temperature"].notna().all() and len(res) == 4

    cmp_out = d / "cmp.csv"
    r = _run_cli(["compare", "--results", str(out), "--results-dir", str(REF_DIR),
                  "--dataset", "pcam", "--fraction", "0.1",
                  "--output", str(cmp_out)], cwd=d)
    assert r.returncode == 0, r.stderr
    cmp = pd.read_csv(cmp_out)
    assert len(cmp) == 9 and cmp["cri"].notna().all()
    assert "Composite Reliability Index" in r.stdout


def test_cli_evaluate_without_val(npy_files):
    d = npy_files
    out = d / "noval.csv"
    r = _run_cli(["evaluate",
                  "--train-features", "X_train.npy", "--train-labels", "y_train.npy",
                  "--test-features", "X_test.npy", "--test-labels", "y_test.npy",
                  "--dataset", "pcam", "--model-name", "NoVal",
                  "--label-fractions", "1.0", "--seeds", "42",
                  "--output", str(out)], cwd=d)
    assert r.returncode == 0, r.stderr
    assert pd.read_csv(out)["temperature"].isna().all()


def test_cli_val_flags_must_be_paired(npy_files):
    r = _run_cli(["evaluate",
                  "--train-features", "X_train.npy", "--train-labels", "y_train.npy",
                  "--test-features", "X_test.npy", "--test-labels", "y_test.npy",
                  "--val-features", "X_val.npy", "--dataset", "pcam"], cwd=npy_files)
    assert r.returncode != 0


def test_cli_compare_default_bundled_and_reference(npy_files, tmp_path):
    d = npy_files
    res = d / "myfm_pcam_results.csv"
    if not res.exists():
        pytest.skip("depends on test_cli_evaluate_with_val_and_compare")
    r = _run_cli(["compare", "--results", str(res), "--dataset", "SPIDER-Breast",
                  "--fraction", "1.0"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    ref_out = tmp_path / "ref.csv"
    r = _run_cli(["reference", "--fraction", "1.0", "--output", str(ref_out)], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    got = pd.read_csv(ref_out).set_index("model")
    exp = pd.read_csv(REF_DIR / "cri_variants_100.csv").set_index("model").loc[got.index]
    assert (got["cri_multiplicative"] - exp["cri_multiplicative"]).abs().max() < 1e-9


def test_cli_version():
    r = _run_cli(["--version"], cwd=REPO)
    assert r.returncode == 0 and "0.3.0" in r.stdout
