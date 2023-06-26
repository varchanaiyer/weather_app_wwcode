import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)
    df = pd.get_dummies(df)
    return df
