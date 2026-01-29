import os
from dataclasses import dataclass
from recsys.utils import read_from_csv


@dataclass
class Dataset:
    dataset_path: str

    @property
    def users(self):
        return read_from_csv(
            path=os.path.join(self.dataset_path, "users.dat"),
            columns=["user_id", "gender", "age", "occupation", "zip"],
        )

    @property
    def ratings(self):
        return read_from_csv(
            path=os.path.join(self.dataset_path, "ratings.dat"),
            columns=["user_id", "movie_id", "rating", "timestamp"],
        )

    @property
    def movies(self):
        return read_from_csv(
            path=os.path.join(self.dataset_path, "movies.dat"),
            columns=["movie_id", "title", "genres"],
        )
