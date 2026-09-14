from pathlib import Path

import pandas as pd


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


def describe_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict:
    numeric_df = df.select_dtypes(include="number")

    if columns is None:
        selected = numeric_df
    else:
        invalid_columns = [
            column
            for column in columns
            if column not in numeric_df.columns
        ]

        if invalid_columns:
            return {
                "error": "Some requested columns are not numeric or do not exist.",
                "invalid_columns": invalid_columns,
            }

        selected = numeric_df[columns]

    description = selected.describe().T

    result = {}

    for column, row in description.iterrows():
        result[column] = {
            "count": int(row["count"]),
            "mean": round(float(row["mean"]), 4),
            "std": round(float(row["std"]), 4),
            "min": round(float(row["min"]), 4),
            "25%": round(float(row["25%"]), 4),
            "median": round(float(row["50%"]), 4),
            "75%": round(float(row["75%"]), 4),
            "max": round(float(row["max"]), 4),
        }

    return {
        "numeric_columns": len(numeric_df.columns),
        "described_columns": selected.columns.tolist(),
        "statistics": result,
    }