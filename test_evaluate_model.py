import mlflow
import pickle
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Wczytanie danych testowych
#df_test = pd.read_csv("data/processed/test.csv")
df_test = pd.read_csv("data/processed/test_preprocessed.csv")

# Wczytanie modelu
model = joblib.load("models/heart_disease_model.pkl")

# Przygotowanie danych
X_test = df_test.drop(columns=["HeartDisease"])
y_test = df_test["HeartDisease"]

# Przewidywanie
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Pokazanie wyników
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Logowanie nowych metryk do MLflow
mlflow.set_experiment("heart_disease_experiment_RFC")
mlflow.start_run()
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)
mlflow.log_param("model", "RandomForestClassifier")
mlflow.sklearn.log_model(
    model,
    "model",
    input_example=X_test.iloc[:1],
    signature=mlflow.models.infer_signature(X_test, y_pred)
    )
mlflow.end_run()