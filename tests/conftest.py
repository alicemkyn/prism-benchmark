import sys
from pathlib import Path

# Make the checkout importable without installation (`python -m pytest`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
