import pandas as pd


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-related features.
    """

    df = df.copy()

    # Temperature range
    if (
        "temperature_max_c" in df.columns
        and "temperature_min_c" in df.columns
    ):
        df["temperature_range_c"] = (
            df["temperature_max_c"] -
            df["temperature_min_c"]
        )

    # Rainfall per growing season day
    if (
        "rainfall_mm" in df.columns
        and "growing_season_days" in df.columns
    ):
        df["rainfall_per_day"] = (
            df["rainfall_mm"] /
            df["growing_season_days"].replace(0, 1)
        )

    # Temperature deviation from average
    if (
        "temperature_avg_c" in df.columns
        and "temperature_min_c" in df.columns
        and "temperature_max_c" in df.columns
    ):
        calculated_avg = (
            df["temperature_min_c"] +
            df["temperature_max_c"]
        ) / 2

        df["temperature_deviation"] = (
            calculated_avg -
            df["temperature_avg_c"]
        )

    return df