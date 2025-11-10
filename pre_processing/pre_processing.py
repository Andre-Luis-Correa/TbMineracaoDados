import logging
from typing import Dict, Any

import numpy as np
import pandas as pd


def convert_to_binary(df: pd.DataFrame, column_name: str, mapping: Dict[Any, int]) -> pd.DataFrame:
    df_copy = df.copy()

    df_copy[column_name] = df_copy[column_name].map(mapping)

    try:
        df_copy[column_name] = df_copy[column_name].astype(int)
    except ValueError:
        logging.info(
            f"Aviso: Não foi possível converter a coluna '{column_name}' para 'int' diretamente. Verifique se há valores não mapeados (NaN) ou use outro método.")

    return df_copy


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_check = df.columns.difference(['ID'])

    return df.drop_duplicates(subset=columns_to_check, keep='first')


def show_outliers_iqr(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_dict = {}

    print("--- Outlier Detection (IQR Method 1.5x) ---\n")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if not outliers.empty:
            outlier_dict[col] = outliers[col].to_dict()
            print(f"{col}: {len(outliers)} outliers found")
            # Optional: Print first 3 outliers to show examples
            # print(f"  Examples: {outliers[col].head(3).values}\n")
        else:
            print(f"{col}: No outliers found")

    return outlier_dict
