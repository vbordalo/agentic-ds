from pathlib import Path

import pandas as pd
import numpy as np


DATA_PATH = Path("data/raw/communities.csv")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?")


def inspect_dataframe(df: pd.DataFrame) -> dict:
    dtype_counts = (
        df.dtypes.astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtype_counts": dtype_counts,
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / 1024**2,
            3,
        ),
    }

def inspect_missing_values(df: pd.DataFrame) -> dict:
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    return {
        "columns_with_missing": missing_counts.index.tolist(),
        "missing_counts": {
            column: int(count)
            for column, count in missing_counts.items()
        },
        "total_missing_values": int(missing_counts.sum()),
    }