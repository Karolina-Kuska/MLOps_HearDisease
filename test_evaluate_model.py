import mlflow
import mlflow.sklearn
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Wczytanie danych testowych
df_test = pd.read_csv("data/processed/test.csv")

# Wczytanie modelu
model = pickle.load(open("models/heart_disease_model.pkl", "rb"))

# Przewidywanie wyników
X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Logowanie do MLflow
mlflow.log_param("model_type", "RandomForest")
mlflow.log_metric("accuracy", accuracy)
mlflow.log_artifact("models/heart_disease_model.pkl")

# Pokazywanie wyników
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{class_report}")

# Zakończenie sesji MLflow
mlflow.end_run()
