from src.features.build_features import preprocess_and_save

if __name__ == "__main__":
    preprocess_and_save("data/processed/train.csv", "data/processed/test.csv")
    print("✅ Dane przetworzone i zapisane.")
