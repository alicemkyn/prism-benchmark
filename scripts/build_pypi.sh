#!/usr/bin/env bash
# Build and check the prism-bench distribution. Does NOT upload.
#
#   bash scripts/build_pypi.sh
#
# Requires: pip install build twine pytest
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python -c "import re;print(re.search(r'^version = \"([^\"]+)\"', open('pyproject.toml').read(), re.M).group(1))")
echo "prism-bench ${VERSION}"

rm -rf dist build ./*.egg-info
python -m pytest -q
python -m build                      # sdist + wheel into dist/
python -m twine check --strict dist/*

# Sanity: the wheel must contain the bundled results_v2 reference CSVs.
python - <<'PY'
import glob, zipfile
whl = glob.glob("dist/*.whl")[0]
names = zipfile.ZipFile(whl).namelist()
for f in ("prism_bench/reference/indomain_all_v2.csv",
          "prism_bench/reference/ood_all_v2.csv"):
    assert f in names, f"{f} missing from {whl}"
print("wheel OK:", whl)
PY

# --- Upload (run manually, after the GitHub/HF placeholders are filled) ------
# TODO(camera-ready): replace GITHUB_URL / HF_DATASET_URL in README.md and
# pyproject.toml before uploading; PyPI releases cannot be overwritten.
#
# Test upload first:
#   python -m twine upload --repository testpypi dist/*
#   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple prism-bench==${VERSION}
# Then:
#   python -m twine upload dist/*
