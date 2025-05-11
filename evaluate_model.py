import sys
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

def evaluate_model(model_name):
    model_path = f"models/heart_disease_model_{model_name}.pkl"
    df_test = pd.read_csv("data/processed/test_preprocessed.csv")
    X_test = df_test.drop(columns=["HeartDisease"])
    y_test = df_test["HeartDisease"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    mlflow.set_experiment(f"heart_disease_experiment_{model_name}")
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        mlflow.sklearn.log_model(
            model, "model",
            input_example=X_test.iloc[:1],
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )

if __name__ == "__main__":
    #model_name = sys.argv[1] if len(sys.argv) > 1 else "LogisticRegression"
    evaluate_model("LogisticRegression")
    evaluate_model("RandomForestClassifier")
