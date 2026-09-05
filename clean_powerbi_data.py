import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecast_model_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "powerbi_clean_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("POWER BI DATA CLEANING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ============================================================
# BASIC CLEANING
# ============================================================

print("\nCleaning column names...")

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# DATE
# ============================================================

if "Date" in df.columns:

    print("Cleaning Date...")

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )


# ============================================================
# STOCK CODE
# ============================================================

if "StockCode" in df.columns:

    print("Cleaning StockCode...")

    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
        .str.strip()
    )

    # Remove accidental .0
    df["StockCode"] = (
        df["StockCode"]
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


# ============================================================
# DESCRIPTION
# ============================================================

if "Description" in df.columns:

    print("Cleaning Description...")

    df["Description"] = (
        df["Description"]
        .fillna("Unknown Product")
        .astype(str)
        .str.strip()
    )

    df.loc[
        df["Description"].isin(
            ["", "nan", "None"]
        ),
        "Description"
    ] = "Unknown Product"


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Demand",
    "AveragePrice_INR",
    "Revenue_INR",
    "TransactionCount",
    "Observed",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",
    "Quarter",
    "IsWeekend"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# HANDLE INVALID CORE VALUES
# ============================================================

# Demand cannot be negative
if "Demand" in df.columns:

    negative_demand = (
        df["Demand"] < 0
    ).sum()

    print(
        f"Negative Demand values: "
        f"{negative_demand:,}"
    )

    df.loc[
        df["Demand"] < 0,
        "Demand"
    ] = 0


# Price cannot be negative
if "AveragePrice_INR" in df.columns:

    negative_price = (
        df["AveragePrice_INR"] < 0
    ).sum()

    print(
        f"Negative Price values: "
        f"{negative_price:,}"
    )

    df.loc[
        df["AveragePrice_INR"] < 0,
        "AveragePrice_INR"
    ] = np.nan


# Revenue cannot be negative
if "Revenue_INR" in df.columns:

    negative_revenue = (
        df["Revenue_INR"] < 0
    ).sum()

    print(
        f"Negative Revenue values: "
        f"{negative_revenue:,}"
    )

    df.loc[
        df["Revenue_INR"] < 0,
        "Revenue_INR"
    ] = 0


# Transaction count cannot be negative
if "TransactionCount" in df.columns:

    negative_transactions = (
        df["TransactionCount"] < 0
    ).sum()

    print(
        f"Negative Transaction values: "
        f"{negative_transactions:,}"
    )

    df.loc[
        df["TransactionCount"] < 0,
        "TransactionCount"
    ] = 0


# ============================================================
# REPAIR PRICE
# ============================================================

if "AveragePrice_INR" in df.columns:

    df["AveragePrice_INR"] = (
        df["AveragePrice_INR"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Use product median price
    df["AveragePrice_INR"] = (
        df.groupby("StockCode")[
            "AveragePrice_INR"
        ]
        .transform(
            lambda x: x.fillna(x.median())
        )
    )

    # If product median is also unavailable,
    # use overall median
    df["AveragePrice_INR"] = (
        df["AveragePrice_INR"]
        .fillna(
            df["AveragePrice_INR"].median()
        )
    )


# ============================================================
# REPAIR REVENUE
# ============================================================

if (
    "Revenue_INR" in df.columns
    and
    "Demand" in df.columns
    and
    "AveragePrice_INR" in df.columns
):

    missing_revenue = (
        df["Revenue_INR"].isna()
    )

    df.loc[
        missing_revenue,
        "Revenue_INR"
    ] = (
        df.loc[
            missing_revenue,
            "Demand"
        ]
        *
        df.loc[
            missing_revenue,
            "AveragePrice_INR"
        ]
    )


# ============================================================
# TRANSACTION COUNT
# ============================================================

if "TransactionCount" in df.columns:

    df["TransactionCount"] = (
        df["TransactionCount"]
        .fillna(0)
    )


# ============================================================
# OBSERVED
# ============================================================

if "Observed" in df.columns:

    df["Observed"] = (
        df["Observed"]
        .fillna(0)
        .astype(int)
    )

else:

    if "Demand" in df.columns:

        df["Observed"] = (
            df["Demand"] > 0
        ).astype(int)


# ============================================================
# CALENDAR VALUES
# ============================================================

if "Date" in df.columns:

    valid_dates = df["Date"].notna()

    if "Year" in df.columns:
        df.loc[valid_dates, "Year"] = (
            df.loc[valid_dates, "Date"].dt.year
        )

    if "Month" in df.columns:
        df.loc[valid_dates, "Month"] = (
            df.loc[valid_dates, "Date"].dt.month
        )

    if "Day" in df.columns:
        df.loc[valid_dates, "Day"] = (
            df.loc[valid_dates, "Date"].dt.day
        )

    if "DayOfWeek" in df.columns:
        df.loc[valid_dates, "DayOfWeek"] = (
            df.loc[valid_dates, "Date"].dt.dayofweek
        )

    if "Quarter" in df.columns:
        df.loc[valid_dates, "Quarter"] = (
            df.loc[valid_dates, "Date"].dt.quarter
        )

    if "WeekOfYear" in df.columns:

        df.loc[valid_dates, "WeekOfYear"] = (
            df.loc[valid_dates, "Date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

    if "IsWeekend" in df.columns:

        df.loc[valid_dates, "IsWeekend"] = (
            df.loc[valid_dates, "Date"]
            .dt.dayofweek
            .ge(5)
            .astype(int)
        )


# ============================================================
# REMOVE INVALID DATE ROWS
# ============================================================

before_date = len(df)

df = df.dropna(
    subset=["Date"]
)

print(
    f"Rows removed due to invalid dates: "
    f"{before_date - len(df):,}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

print(
    f"Duplicate rows removed: "
    f"{before_duplicates - len(df):,}"
)


# ============================================================
# SORT
# ============================================================

sort_columns = []

if "StockCode" in df.columns:
    sort_columns.append("StockCode")

if "Date" in df.columns:
    sort_columns.append("Date")

if sort_columns:

    df = (
        df
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


# ============================================================
# IMPORTANT:
# DO NOT CHANGE VALID NEGATIVE TIME-SERIES FEATURES
# ============================================================

# Examples that may legitimately be negative:
#
# Demand_Trend_7_28
# Price_Change_Pct
# some lag/derived features
#
# These are signals for the ML model and should NOT
# automatically be converted to zero.


# ============================================================
# FINAL MISSING-VALUE REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATA QUALITY REPORT")
print("=" * 60)

print(
    f"Final rows    : {len(df):,}"
)

print(
    f"Final columns : {len(df.columns)}"
)

print("\nMissing values:")

missing = (
    df.isna()
    .sum()
)

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

if missing.empty:

    print("No missing values.")

else:

    print(missing.to_string())


# ============================================================
# SAVE
# ============================================================

print("\nSaving cleaned Power BI dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved successfully:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)