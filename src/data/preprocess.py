import pandas as pd


TARGET_COLUMN = "yield_tonnes_per_hectare"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.
    """

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove identifier column
    if "record_id" in df.columns:
        df = df.drop(columns=["record_id"])

    # Convert sowing_date to datetime
    if "sowing_date" in df.columns:
        df["sowing_date"] = pd.to_datetime(
            df["sowing_date"],
            errors="coerce"
        )

    # Remove rows where target is missing
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in numerical and categorical columns.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    # Fill numerical missing values with median
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill categorical missing values with mode
    for column in categorical_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    df = clean_data(df)
    df = handle_missing_values(df)

    return df