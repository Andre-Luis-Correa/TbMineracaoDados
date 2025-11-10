import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_REPOSITORY_ID = 17

def get_loaded_data():
    return fetch_ucirepo(id=DATA_REPOSITORY_ID)

def get_dataframes():
    x = get_loaded_data().data.features
    y = get_loaded_data().data.targets

    return x, y

def get_full_dataframe():
    x, y = get_dataframes()
    return pd.concat([x, y], axis=1)

def print_variables():
    print(get_loaded_data().variables)

