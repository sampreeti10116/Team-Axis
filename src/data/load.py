import pandas as pd
from pathlib import Path


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw crop yield dataset.

    Parameters:
        file_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(path)

    return df


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save processed dataset to CSV.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Processed data saved to: {path}")