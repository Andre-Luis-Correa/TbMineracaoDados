from ucimlrepo import fetch_ucirepo

DATA_REPOSITORY_ID = 17

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=DATA_REPOSITORY_ID)

def get_dataframes():
    x = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    return x, y

def print_variables():
    print(breast_cancer_wisconsin_diagnostic.variables)

print_variables()

