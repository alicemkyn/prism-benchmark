"""The CPU quick example runs, stays on synthetic data, and calls both APIs."""

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "quickstart_cpu.py"
CITATION = REPO / "CITATION.cff"


def test_quickstart_script_runs_and_is_deterministic():
    first = _run_script()
    second = _run_script()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    out = first.stdout
    assert "data_source=synthetic" in out
    assert "not the paper's results" in out
    assert "https://arxiv.org/abs/2610.36517" in out
    assert "SyntheticProbe" in out
    assert "synthetic_row_is_reference=False" in out
    assert "synthetic_ood_stability_imputed=True" in out
    assert "n_compare_rows=9" in out
    for model in ("CLIP", "PLIP", "CONCH", "VIRCHOW2", "UNI",
                  "GigaPath", "H-Optimus-0", "MIDNIGHT"):
        assert model in out


def test_quickstart_does_not_call_the_hub():
    tree = ast.parse(SCRIPT.read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert "hf_hub_download" not in called
    assert "snapshot_download" not in called
    text = SCRIPT.read_text()
    assert "click-through" in text
    assert "alicemkyn/PRISM-embeddings" in text
    assert "synthetic" in text


def test_quickstart_evaluate_and_compare_frames():
    import importlib.util

    spec = importlib.util.spec_from_file_location("quickstart_cpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, results, comparison = module.run()
    assert data["source"] == "synthetic"
    assert set(results["model"]) == {"SyntheticProbe"}
    assert results["temperature"].notna().all()
    assert results["ece_scaled_fixed"].between(0, 1).all()
    assert list(comparison["model"]).count("SyntheticProbe") == 1
    refs = comparison.loc[comparison["is_reference"], "model"]
    assert len(refs) == 8
    mine = comparison.loc[comparison["model"] == "SyntheticProbe"].iloc[0]
    assert bool(mine["ood_stability_imputed"])
    assert pd.notna(mine["cri"])


def test_citation_cff_and_readme_agree_on_the_paper():
    citation = CITATION.read_text()
    readme = (REPO / "README.md").read_text()
    for text in (citation, readme):
        assert "Alicem" in text and "Koyun" in text
        assert "2610.36517" in text
        assert "GenAI4Health" in text
        assert "Poster" in text or "poster" in text
        assert "10.48550/arXiv.2610.36517" in text
    assert "cff-version: 1.2.0" in citation
    assert "license: MIT" in citation
    assert 'version: 0.3.0' in citation
    assert "Third Workshop on GenAI for Health (GenAI4Health) at NeurIPS 2026" in citation
    assert "not a NeurIPS main-track paper" in citation
    assert "@inproceedings{koyun2026prism" in readme
    # The venue string must stay a workshop poster, not the main conference.
    assert "booktitle = {NeurIPS}" not in readme
    assert "booktitle = {Third Workshop on GenAI for Health (GenAI4Health) at NeurIPS 2026}" in readme


def _run_script():
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
