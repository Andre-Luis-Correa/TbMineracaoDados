from load_data.load_data import get_dataframes
from pre_processing.data_conversion import convert_to_binary

data, targets = get_dataframes()

map_binary_diagnosis = {'M': 1, 'B': 0}
df = convert_to_binary(targets, 'Diagnosis', map_binary_diagnosis)

print("\nDataFrame Codificado:")
print(df)


