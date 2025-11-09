import logging
from typing import Dict, Any

import pandas as pd


def convert_to_binary(df: pd.DataFrame, column_name: str, mapping: Dict[Any, int]) -> pd.DataFrame:
    df_copia = df.copy()

    df_copia[column_name] = df_copia[column_name].map(mapping)

    try:
        df_copia[column_name] = df_copia[column_name].astype(int)
    except ValueError:
        logging.info(
            f"Aviso: Não foi possível converter a coluna '{column_name}' para 'int' diretamente. Verifique se há valores não mapeados (NaN) ou use outro método.")

    return df_copia
