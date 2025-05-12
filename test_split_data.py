from src.data.load_dataset import load_heart_data
from src.data.split_data import split_and_save_data

if __name__ == "__main__":
    print("▶ Start: ładowanie danych...")
    df = load_heart_data()

    print("✅ Dane załadowane. Podział danych...")
    split_and_save_data(df)

    print("🏁 Koniec. Dane podzielone i zapisane.")
