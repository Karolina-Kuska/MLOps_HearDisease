import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri(
    "file:///tmp/mlruns"
)  # 👈 zmień lokalizację artefaktów na poprawną w Linuxie


def train_and_save_model_RFC(
    train_path="data/processed/train_preprocessed.csv",
    test_path="data/processed/test_preprocessed.csv",
    model_output_dir="models",
):
    # Wczytanie danych
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["HeartDisease"])
    y_train = train_df["HeartDisease"]
    X_test = test_df.drop(columns=["HeartDisease"])
    y_test = test_df["HeartDisease"]

    # Trening RFC
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Zapis modelu
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(
        model_output_dir, "heart_disease_model_RandomForestClassifier.pkl"
    )
    joblib.dump(model, model_path)
    print(f"Model zapisany jako {model_path}")

    # Logowanie do MLflow
    mlflow.set_experiment("heart_disease_experiment_RFC")
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Confusion matrix
        os.makedirs("figures", exist_ok=True)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, y_pred),
            display_labels=["No", "Yes"],
        )
        disp.plot()
        fig_path = "figures/confusion_matrix_rfc.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)

        # Logowanie modelu
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model, "model", input_example=X_test.iloc[:1], signature=signature
        )

    print("✅ RandomForestClassifier i metryki zapisane do MLflow.")
