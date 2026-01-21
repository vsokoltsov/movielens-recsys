import os
from pathlib import Path
from dotenv import load_dotenv

from recsys.db.session import DATABASE_URL
load_dotenv()

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA = os.path.join(ROOT, "data")
RAW_DATA = os.path.join(DATA, "raw")
MOVIELENS_PATH = os.path.join(RAW_DATA, "ml-1m")
MODELS = os.path.join(DATA, "models")
RAW_BUCKET = os.environ.get("RAW_BUCKET")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://user:pass@postgres:5432/recsys")