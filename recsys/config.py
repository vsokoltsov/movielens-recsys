import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA = os.path.join(ROOT, "data")
RAW_DATA = os.path.join(DATA, "raw")
MOVIELENS_PATH = os.path.join(RAW_DATA, "ml-1m")
MODELS = os.path.join(DATA, "models")
RAW_BUCKET = os.environ.get("RAW_BUCKET")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DATABASE_URL = os.environ.get("DATABASE_URL", f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")