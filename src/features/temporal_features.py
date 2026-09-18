import pandas as pd


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date and time-related features.
    """

    df = df.copy()

    if "sowing_date" in df.columns:

        df["sowing_date"] = pd.to_datetime(
            df["sowing_date"],
            errors="coerce"
        )

        # Extract date components
        df["sowing_year"] = df["sowing_date"].dt.year
        df["sowing_month"] = df["sowing_date"].dt.month
        df["sowing_day"] = df["sowing_date"].dt.day
        df["sowing_day_of_year"] = (
            df["sowing_date"].dt.dayofyear
        )

        # Agricultural season indicator
        df["sowing_quarter"] = (
            df["sowing_date"].dt.quarter
        )

        # Remove original date after extraction
        df = df.drop(columns=["sowing_date"])

    return df