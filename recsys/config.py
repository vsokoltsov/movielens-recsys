import os
from pathlib import Path

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA = os.path.join(ROOT, "data")
RAW_DATA = os.path.join(DATA, "raw")
MOVIELENS_PATH = os.path.join(RAW_DATA, "ml-1m")
MODELS = os.path.join(DATA, "models")
