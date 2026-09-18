import pandas as pd


def create_soil_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create soil-related features.
    """

    df = df.copy()

    # Total NPK content
    nutrient_columns = [
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha"
    ]

    if all(column in df.columns for column in nutrient_columns):
        df["total_npk_kg_ha"] = (
            df["nitrogen_kg_ha"] +
            df["phosphorus_kg_ha"] +
            df["potassium_kg_ha"]
        )

    # Nitrogen to Phosphorus ratio
    if (
        "nitrogen_kg_ha" in df.columns
        and "phosphorus_kg_ha" in df.columns
    ):
        df["np_ratio"] = (
            df["nitrogen_kg_ha"] /
            df["phosphorus_kg_ha"].replace(0, 1)
        )

    # Potassium to Nitrogen ratio
    if (
        "potassium_kg_ha" in df.columns
        and "nitrogen_kg_ha" in df.columns
    ):
        df["kn_ratio"] = (
            df["potassium_kg_ha"] /
            df["nitrogen_kg_ha"].replace(0, 1)
        )

    # Soil pH category
    if "soil_ph" in df.columns:
        df["soil_ph_category"] = pd.cut(
            df["soil_ph"],
            bins=[0, 5.5, 6.5, 7.5, 8.5, 14],
            labels=[
                "Strongly Acidic",
                "Slightly Acidic",
                "Neutral",
                "Alkaline",
                "Strongly Alkaline"
            ]
        )

    # Moisture stress indicator
    if "soil_moisture_pct" in df.columns:
        df["moisture_stress"] = (
            df["soil_moisture_pct"] < 30
        ).astype(int)

    return df