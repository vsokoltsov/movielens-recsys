import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class Dataset:
    dataset_path: str

    @property
    def users(self):
        return pd.read_csv(
            os.path.join(self.dataset_path, "users.dat"),
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"],
            encoding="latin-1"
        )

    @property
    def ratings(self):
        return pd.read_csv(
            os.path.join(self.dataset_path, "ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

    @property
    def movies(self):
        return pd.read_csv(
            os.path.join(self.dataset_path, "movies.dat"),
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1"
        )