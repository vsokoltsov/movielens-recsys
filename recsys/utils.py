from typing import List
import pandas as pd

def read_from_csv(path: str, columns: List[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )