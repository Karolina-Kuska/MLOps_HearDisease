from src.models.train_model_LR import train_and_save_model_LR
from src.models.train_model_RFC import train_and_save_model_RFC

if __name__ == "__main__":
    train_and_save_model_LR()
    print("Model - LogisticRegression wytrenowany i zapisany.")


if __name__ == "__main__":
    train_and_save_model_RFC()
    print("Model - RandomForestClassifier wytrenowany i zapisany.")