import pandas as pd

MODEL_PATH = "../models/modelo_pipeline.joblib"

TRAIN_PATH = "../data/mpg_train.csv"
TEST_PATH = "../data/mpg_test.csv"

def get_dataframe(path=TRAIN_PATH):
    df_result = pd.read_csv(path)

    new_names = ["year", "brand", "model", "vehicle_class", "engine_size", "cylinders", "transmission", "fuel_type", "fuel_city_Lkm", "fuel_hwy_Lkm", "fuel_comb_Lkm", "fuel_comb_mpg", "co2"]
    df_result.columns = new_names

    return df_result

def get_dataframe_train():
    return get_dataframe()

def get_dataframe_test():
    return get_dataframe(TEST_PATH)

def get_target():
    return "fuel_comb_mpg"

def get_features_num(dataframe):
    features_num = list(dataframe.columns[dataframe.dtypes != "object"])
    features_num.remove(get_target())
    features_num.remove('cylinders')
    features_num.remove('fuel_comb_Lkm')
    return features_num


def get_features_cat(dataframe):
    features_cat = (list(dataframe.select_dtypes(include = ['object']).columns))
    features_cat.append('cylinders') 
    return features_cat


def data_report(df):
    '''Esta funcion describe los campos de un dataframe de pandas de forma bastante clara, crack'''
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)

    return concatenado.T

def clean_categoricals(df):
    return df.apply(lambda col: col.astype(str).str.lower())
