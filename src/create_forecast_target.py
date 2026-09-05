from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "demand_features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecast_model_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DYNAMIC PRICING ENGINE")
print("STEP 8B - FUTURE DEMAND TARGET")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"])

df["StockCode"] = (
    df["StockCode"]
    .astype(str)
    .str.strip()
)

df = df.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)


print(f"\nInput rows: {len(df):,}")
print(f"Products: {df['StockCode'].nunique():,}")


# ============================================================
# CREATE NEXT-DAY DEMAND
# ============================================================

print("\nCreating next-day demand target...")

df["Demand_NextDay"] = (
    df.groupby("StockCode")["Demand"]
    .shift(-1)
)


# ============================================================
# CHECK DATE CONTINUITY
# ============================================================

df["NextDate"] = (
    df.groupby("StockCode")["Date"]
    .shift(-1)
)

df["DaysToNextObservation"] = (
    df["NextDate"] - df["Date"]
).dt.days


# ============================================================
# ONLY KEEP TRUE NEXT-DAY TARGETS
# ============================================================

df["Valid_NextDay_Target"] = (
    df["DaysToNextObservation"] == 1
)


valid_count = (
    df["Valid_NextDay_Target"].sum()
)

invalid_count = (
    (~df["Valid_NextDay_Target"]).sum()
)


print(
    f"\nValid next-day targets: "
    f"{valid_count:,}"
)

print(
    f"Invalid/non-consecutive targets: "
    f"{invalid_count:,}"
)


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df[
    df["Valid_NextDay_Target"]
].copy()


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "NextDate",
        "DaysToNextObservation",
        "Valid_NextDay_Target"
    ]
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

print(
    f"\nRows available for forecasting: "
    f"{len(df):,}"
)

print(
    f"Products: "
    f"{df['StockCode'].nunique():,}"
)

print(
    f"Target missing values: "
    f"{df['Demand_NextDay'].isna().sum():,}"
)

print(
    f"Target mean: "
    f"{df['Demand_NextDay'].mean():.2f}"
)

print(
    f"Target median: "
    f"{df['Demand_NextDay'].median():.2f}"
)

print(
    f"Target maximum: "
    f"{df['Demand_NextDay'].max():,.0f}"
)


# ============================================================
# SHOW EXAMPLE
# ============================================================

print("\nExample:")

example = (
    df[
        [
            "Date",
            "StockCode",
            "Demand",
            "Demand_NextDay"
        ]
    ]
    .head(10)
)

print(
    example.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FUTURE DEMAND TARGET CREATED")
print("=" * 70)

print(
    f"\nSaved to:"
    f"\n{OUTPUT_FILE}"
)

print("\nTarget definition:")
print(
    "Features at date T → Demand at date T+1"
)

print("\nNext:")
print(
    "→ Train advanced demand forecasting model"
)