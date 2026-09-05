from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecasting_panel.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "demand_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DYNAMIC PRICING ENGINE")
print("STEP 6 - TIME-SERIES FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df["StockCode"] = (
    df["StockCode"]
    .astype(str)
    .str.strip()
)

df = df.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)

print(f"\nInput records: {len(df):,}")
print(f"Products: {df['StockCode'].nunique():,}")


# ============================================================
# GROUP BY PRODUCT
# ============================================================

grouped = df.groupby(
    "StockCode",
    group_keys=False
)


# ============================================================
# DEMAND LAG FEATURES
# ============================================================

print("\nCreating demand lag features...")

for lag in [1, 7, 14, 28]:

    df[f"Demand_Lag_{lag}"] = (
        grouped["Demand"]
        .shift(lag)
    )


# ============================================================
# ROLLING DEMAND FEATURES
# ============================================================

print("Creating rolling demand features...")

# IMPORTANT:
# Shift by one day before rolling.
# This prevents today's demand from being used
# to predict today's demand.

past_demand = (
    grouped["Demand"]
    .shift(1)
)

df["Demand_RollingMean_7"] = (
    past_demand
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            7,
            min_periods=3
        ).mean()
    )
)

df["Demand_RollingMean_14"] = (
    past_demand
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            14,
            min_periods=5
        ).mean()
    )
)

df["Demand_RollingMean_28"] = (
    past_demand
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            28,
            min_periods=7
        ).mean()
    )
)

df["Demand_RollingStd_7"] = (
    past_demand
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            7,
            min_periods=3
        ).std()
    )
)

df["Demand_RollingStd_28"] = (
    past_demand
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            28,
            min_periods=7
        ).std()
    )
)


# ============================================================
# PRICE FEATURES
# ============================================================

print("Creating price features...")

df["Price_Lag_1"] = (
    grouped["AveragePrice_INR"]
    .shift(1)
)

df["Price_Lag_7"] = (
    grouped["AveragePrice_INR"]
    .shift(7)
)


df["Price_Change_Pct"] = np.where(
    df["Price_Lag_1"] > 0,

    (
        (
            df["AveragePrice_INR"]
            - df["Price_Lag_1"]
        )
        / df["Price_Lag_1"]
    ) * 100,

    0
)


past_price = (
    grouped["AveragePrice_INR"]
    .shift(1)
)

df["Price_RollingMean_7"] = (
    past_price
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            7,
            min_periods=3
        ).mean()
    )
)


# ============================================================
# REVENUE FEATURES
# ============================================================

print("Creating revenue features...")

df["Revenue_Lag_1"] = (
    grouped["Revenue_INR"]
    .shift(1)
)

df["Revenue_Lag_7"] = (
    grouped["Revenue_INR"]
    .shift(7)
)


# ============================================================
# TRANSACTION FEATURES
# ============================================================

df["Transactions_Lag_1"] = (
    grouped["TransactionCount"]
    .shift(1)
)

df["Transactions_RollingMean_7"] = (
    grouped["TransactionCount"]
    .shift(1)
    .groupby(df["StockCode"])
    .transform(
        lambda x: x.rolling(
            7,
            min_periods=3
        ).mean()
    )
)


# ============================================================
# DEMAND TREND
# ============================================================

print("Creating demand trend features...")

df["Demand_Trend_7_28"] = np.where(
    df["Demand_RollingMean_28"] > 0,

    (
        (
            df["Demand_RollingMean_7"]
            - df["Demand_RollingMean_28"]
        )
        / df["Demand_RollingMean_28"]
    ) * 100,

    0
)


# ============================================================
# PRICE POSITION
# ============================================================

df["Price_Position_7"] = np.where(
    df["Price_RollingMean_7"] > 0,

    (
        df["AveragePrice_INR"]
        / df["Price_RollingMean_7"]
    ),

    1
)


# ============================================================
# CALENDAR FEATURES
# ============================================================

print("Creating calendar features...")

df["DayOfWeek_Sin"] = np.sin(
    2 * np.pi * df["DayOfWeek"] / 7
)

df["DayOfWeek_Cos"] = np.cos(
    2 * np.pi * df["DayOfWeek"] / 7
)

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

df["WeekOfYear_Sin"] = np.sin(
    2 * np.pi * df["WeekOfYear"] / 52
)

df["WeekOfYear_Cos"] = np.cos(
    2 * np.pi * df["WeekOfYear"] / 52
)


# ============================================================
# PRODUCT AGE / TIME INDEX
# ============================================================

product_start = (
    df.groupby("StockCode")["Date"]
    .transform("min")
)

df["Product_Age_Days"] = (
    df["Date"] - product_start
).dt.days


# ============================================================
# CLEAN NUMERICAL VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_columns] = (
    df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

feature_columns = [
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Lag_28",
    "Demand_RollingMean_7",
    "Demand_RollingMean_14",
    "Demand_RollingMean_28",
    "Demand_RollingStd_7",
    "Demand_RollingStd_28",
    "Price_Lag_1",
    "Price_Lag_7",
    "Price_Change_Pct",
    "Price_RollingMean_7",
    "Demand_Trend_7_28",
    "Price_Position_7",
]


print(
    f"\nTotal columns: {len(df.columns)}"
)

print(
    f"Feature columns: "
    f"{len(feature_columns)}"
)


print("\nFeature missing values:")

missing_features = (
    df[feature_columns]
    .isna()
    .sum()
)

print(
    missing_features[
        missing_features > 0
    ].to_string()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nRows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)