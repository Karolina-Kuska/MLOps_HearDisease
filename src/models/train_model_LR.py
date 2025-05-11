import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import os

# Funkcja trenująca model i zapisująca go
def train_and_save_model_LR(train_path="data/processed/train_preprocessed.csv", test_path="data/processed/test_preprocessed.csv", model_output_dir="models"):
    # Wczytanie danych
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Oddzielenie cech od etykiet
    X_train = train_df.drop(columns=["HeartDisease"])
    y_train = train_df["HeartDisease"]
    X_test = test_df.drop(columns=["HeartDisease"])
    y_test = test_df["HeartDisease"]
    
    # Inicjalizacja modelu
    model = LogisticRegression(max_iter=1000)

    # Trening modelu
    model.fit(X_train, y_train)

    # Predykcja
    y_pred = model.predict(X_test)

    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Zapis modelu jako .pkl
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "heart_disease_model_LogisticRegression.pkl")
    joblib.dump(model, model_path)

    print(f"Model zapisany jako {model_path}")

    # Logowanie w MLflow
    mlflow.set_experiment("heart_disease_experiment_LR")
    mlflow.start_run()
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_test.iloc[:1],
        signature=mlflow.models.infer_signature(X_test, y_pred)
        )
    mlflow.end_run()

    print("✅ LogisticRegression i metryki zapisane do MLflow.")
