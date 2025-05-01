# Wczyta dane train.csv i test.csv
# ToDo Wykona:
# - One-hot encoding dla kolumn kategorycznych
# - Skalowanie (np. StandardScaler) dla numerycznych
# Zapisze train_preprocessed.csv i test_preprocessed.csv

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def preprocess_and_save(train_path, test_path, output_dir="data/processed"):
    # Wczytujemy dane
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separacja cech i etykiet
    X_train = train_df.drop(columns=["HeartDisease"])
    y_train = train_df["HeartDisease"]

    X_test = test_df.drop(columns=["HeartDisease"])
    y_test = test_df["HeartDisease"]

    # Kolumny kategoryczne i numeryczne
    categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numerical = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical)
    ])

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Tworzymy ramki danych z numpy arrays
    processed_train = pd.DataFrame(X_train_processed)
    processed_train["HeartDisease"] = y_train.values

    processed_test = pd.DataFrame(X_test_processed)
    processed_test["HeartDisease"] = y_test.values

    # Zapis
    os.makedirs(output_dir, exist_ok=True)
    processed_train.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)
    processed_test.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)

    print("✅ Dane przetworzone i zapisane.")
