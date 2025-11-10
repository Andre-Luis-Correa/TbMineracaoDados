from os import remove

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from load_data.load_data import get_full_dataframe, get_dataframes
from pre_processing.pre_processing import convert_to_binary, remove_duplicates, show_outliers_iqr

full_df = get_full_dataframe()

map_binary_diagnosis = {'M': 1, 'B': 0}
full_df = convert_to_binary(full_df, 'Diagnosis', map_binary_diagnosis)
full_df_without_duplicates = remove_duplicates(full_df)

x, y = get_dataframes()
outliers_details = show_outliers_iqr(x)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Usa um modelo simples (Regressão Logística) para avaliar as features
model = LogisticRegression(solver='liblinear')
rfe = RFE(estimator=model, n_features_to_select=10)
x_rfe = rfe.fit_transform(x, y.values.ravel())

print(f"Número de features após Wrapper (RFE): {x_rfe.shape[1]}")