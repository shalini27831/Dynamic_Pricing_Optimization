from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DYNAMIC PRICING ENGINE")
print("STEP 8 - DEMAND FORECASTING")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

validation = pd.read_csv(
    VALIDATION_FILE,
    low_memory=False
)

test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)


# ============================================================
# DATE CONVERSION
# ============================================================

for df in [train, validation, test]:

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
        .str.strip()
    )


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"\nTrain rows      : {len(train):,}"
)

print(
    f"Validation rows : {len(validation):,}"
)

print(
    f"Test rows       : {len(test):,}"
)

print(
    f"Products        : {train['StockCode'].nunique():,}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DEMAND SUMMARY")
print("=" * 70)

print("\nTrain demand:")

print(
    train["Demand"]
    .describe()
    .to_string()
)

print("\nValidation demand:")

print(
    validation["Demand"]
    .describe()
    .to_string()
)

print("\nTest demand:")

print(
    test["Demand"]
    .describe()
    .to_string()
)


# ============================================================
# BASELINE MODEL
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL")
print("=" * 70)

print(
    "\nBaseline assumption:"
)

print(
    "Predicted demand = previous 7-day average"
)


# ------------------------------------------------------------
# Create baseline predictions
# ------------------------------------------------------------

history = pd.concat(
    [
        train,
        validation
    ],
    ignore_index=True
)

history = history.sort_values(
    ["StockCode", "Date"]
)


# Use the latest 7 observations available for each product
recent_demand = (
    history
    .groupby("StockCode")
    .tail(7)
    .groupby("StockCode")["Demand"]
    .mean()
)


test_baseline = test.copy()

test_baseline["BaselinePrediction"] = (
    test_baseline["StockCode"]
    .map(recent_demand)
)


# Products without a baseline value
missing_baseline = (
    test_baseline["BaselinePrediction"]
    .isna()
    .sum()
)

print(
    f"\nTest rows without baseline: "
    f"{missing_baseline:,}"
)


# Remove rows where baseline cannot be calculated
test_baseline = test_baseline.dropna(
    subset=["BaselinePrediction"]
)


# ============================================================
# METRICS
# ============================================================

actual = test_baseline["Demand"].values

predicted = (
    test_baseline["BaselinePrediction"]
    .values
)


mae = np.mean(
    np.abs(
        actual - predicted
    )
)


rmse = np.sqrt(
    np.mean(
        (actual - predicted) ** 2
    )
)


# WAPE
total_error = np.sum(
    np.abs(
        actual - predicted
    )
)

total_actual = np.sum(
    np.abs(actual)
)

if total_actual > 0:

    wape = (
        total_error
        / total_actual
    ) * 100

else:

    wape = np.nan


# R2
ss_res = np.sum(
    (actual - predicted) ** 2
)

ss_tot = np.sum(
    (actual - np.mean(actual)) ** 2
)

if ss_tot > 0:

    r2 = 1 - (
        ss_res / ss_tot
    )

else:

    r2 = np.nan


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE PERFORMANCE")
print("=" * 70)

print(
    f"\nMAE  : {mae:,.2f}"
)

print(
    f"RMSE : {rmse:,.2f}"
)

print(
    f"WAPE : {wape:.2f}%"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# SAVE BASELINE RESULTS
# ============================================================

baseline_results = pd.DataFrame(
    {
        "Model": ["7-Day Average Baseline"],
        "MAE": [mae],
        "RMSE": [rmse],
        "WAPE": [wape],
        "R2": [r2]
    }
)


output_file = (
    REPORT_DIR
    / "forecast_baseline_results.csv"
)


baseline_results.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
    f"\n{output_file}"
)

print(
    "\nNext:"
)

print(
    "→ Advanced ML demand forecasting"
)