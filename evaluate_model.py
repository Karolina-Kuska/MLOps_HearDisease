import joblib
import pandas as pd
import mlflow
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def evaluate_model(model_name):
    model_path = f"models/heart_disease_model_{model_name}.pkl"
    df_test = pd.read_csv("data/processed/test_preprocessed.csv")
    X_test = df_test.drop(columns=["HeartDisease"])
    y_test = df_test["HeartDisease"]

    # Załaduj model
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Wyświetlenie wyników
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Ustawienie eksperymentu w MLflow
    mlflow.set_experiment(f"heart_disease_experiment_{model_name}")

    # Zmień lokalizację zapisywania artefaktów
    # Przykład lokalizacji w obrębie projektu
    artifact_path = "mlflow_artifacts"
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)

    # Rozpocznij uruchomienie MLflow
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
            }
        )

        # Zapisz model w MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_test.iloc[:1],
            signature=mlflow.models.infer_signature(X_test, y_pred),
        )


if __name__ == "__main__":
    evaluate_model("LogisticRegression")
    evaluate_model("RandomForestClassifier")
