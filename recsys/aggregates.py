from pydantic import BaseModel
from enum import Enum

class ModelType(Enum):
    ALS = "als"
    ITEM_KNN = "item_knn"

class Source(Enum):
    CSV = "csv"
    DB = "db"

class Movie(BaseModel):
    id: int
    title: str
    genre: str