import os
import random

import numpy as np
import pandas as pd
from IPython.display import Markdown, display_markdown


def set_global_seed(seed: int) -> None:
    """Set global random state across common libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def display_dataset_info(df: pd.DataFrame) -> pd.DataFrame:
    len_df = len(df)

    df_values = (
        df.count()
        .to_frame("Non-null values")
        .rename_axis("Column")
        .reset_index()
    )

    df_uniques = (
        df.nunique()
        .to_frame("Unique values")
        .rename_axis("Column")
        .reset_index()
    )

    df_missing_values = (
        df.isnull()
        .sum()
        .to_frame("Missing values")
        .rename_axis("Column")
        .reset_index()
    )

    df_missing_values["Missing values"] = df_missing_values.apply(
        lambda row: (
            f"{row['Missing values']} "
            f"({row['Missing values'] / len_df * 100:.2f}%)"
        ),
        axis=1,
    )

    df_types = df.dtypes.to_frame("Type").rename_axis("Column").reset_index()

    numeric_summary = (
        df.select_dtypes(include=np.number)
        .agg(["min", "max", "mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "Column"})
    )

    numeric_summary["Top values"] = numeric_summary.apply(
        lambda row: (
            f"min={row['min']:.2f}, "
            f"max={row['max']:.2f}, "
            f"mean={row['mean']:.2f}, "
            f"std={row['std']:.2f}"
        ),
        axis=1,
    )

    numeric_top_values = numeric_summary[["Column", "Top values"]]

    categorical_top_values = []
    for column in df.select_dtypes(include="object").columns:
        value_counts = df[column].value_counts(dropna=False)
        if value_counts.empty:
            continue
        top_value = value_counts.index[0]
        top_count = value_counts.iloc[0]
        top_percentage = top_count / len_df * 100
        categorical_top_values.append(
            {
                "Column": column,
                "Top values": f"{top_value} "
                f"({top_count} rows, {top_percentage:.1f}%)",
            }
        )

    categorical_top_values = pd.DataFrame(categorical_top_values)

    top_values = pd.concat(
        [numeric_top_values, categorical_top_values], ignore_index=True
    )

    df_info = (
        df_types.merge(df_values, on="Column", how="outer")
        .merge(df_uniques, on="Column", how="outer")
        .merge(df_missing_values, on="Column", how="outer")
        .merge(top_values, on="Column", how="outer")
    )

    duplicate_rows = df.duplicated().sum()

    display_markdown(
        Markdown(
            f"## Dataset info:\n"
            f"**Duplicate rows:** {duplicate_rows} | "
            f"{duplicate_rows / len_df * 100:.2f}%\n"
            f"## Dataset column info:\n"
            f"{df_info.to_markdown(index=False)}"
        )
    )

    return
