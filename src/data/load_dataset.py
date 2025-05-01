# src/data/load_dataset.py

import kagglehub
import pandas as pd

def load_heart_data() -> pd.DataFrame:
    """
    Downloads the heart dataset using kagglehub and returns a pandas DataFrame.
    """
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    df = pd.read_csv(path + "/heart.csv")
    return df
