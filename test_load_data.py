from src.data.load_dataset import load_heart_data

if __name__ == "__main__":
    print("Ładuję dane...")  # Żeby było wiadomo, że działa
    df = load_heart_data()
    print("Pierwsze 5 wierszy:")
    print(df.head())
