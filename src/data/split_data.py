import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    output_dir: str = "data/processed"
):
    # Tworzymy folder, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Podział
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )

    # Zapis
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Dane zapisane w: {output_dir}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
