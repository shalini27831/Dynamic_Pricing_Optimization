from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

INPUT_FILE = DATA_DIR / "forecast_model_data.csv"
MODEL_FILE = MODEL_DIR / "lightgbm_demand_model.pkl"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DYNAMIC PRICING ENGINE")
print("STEP 10 - FORECAST ACCURACY ANALYSIS")
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


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Demand",
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Lag_28",
    "Demand_RollingMean_7",
    "Demand_RollingMean_14",
    "Demand_RollingMean_28",
    "Demand_RollingStd_7",
    "Demand_RollingStd_28",
    "AveragePrice_INR",
    "Price_Lag_1",
    "Price_Lag_7",
    "Price_Change_Pct",
    "Price_RollingMean_7",
    "Price_Position_7",
    "Revenue_INR",
    "Revenue_Lag_1",
    "Revenue_Lag_7",
    "TransactionCount",
    "Transactions_Lag_1",
    "Transactions_RollingMean_7",
    "Demand_Trend_7_28",
    "DayOfWeek",
    "WeekOfYear",
    "Month",
    "Quarter",
    "IsWeekend",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "Month_Sin",
    "Month_Cos",
    "WeekOfYear_Sin",
    "WeekOfYear_Cos",
    "Product_Age_Days",
    "Observed"
]

TARGET = "Demand_NextDay"


# ============================================================
# PREPARE DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

X = df[FEATURES].fillna(0)

y = df[TARGET]


# ============================================================
# TEST PERIOD
# ============================================================

test_start = pd.Timestamp("2011-08-21")

test = df[
    df["Date"] >= test_start
].copy()

X_test = X.loc[test.index]

y_test = y.loc[test.index]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# PREDICTIONS
# ============================================================

print("Generating test predictions...")

test["ActualDemand"] = y_test.values

test["PredictedDemand"] = model.predict(
    X_test
)

test["PredictedDemand"] = np.maximum(
    test["PredictedDemand"],
    0
)

test["Error"] = (
    test["PredictedDemand"]
    - test["ActualDemand"]
)

test["AbsoluteError"] = (
    np.abs(test["Error"])
)


# ============================================================
# OVERALL METRICS
# ============================================================

actual = test["ActualDemand"].values
predicted = test["PredictedDemand"].values

mae = np.mean(
    np.abs(actual - predicted)
)

rmse = np.sqrt(
    np.mean(
        (actual - predicted) ** 2
    )
)

bias = np.mean(
    predicted - actual
)

total_actual = np.sum(
    np.abs(actual)
)

wape = (
    np.sum(np.abs(actual - predicted))
    / total_actual
) * 100


print("\n" + "=" * 70)
print("OVERALL TEST ANALYSIS")
print("=" * 70)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")
print(f"Bias : {bias:.2f}")


# ============================================================
# DEMAND LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE BY DEMAND LEVEL")
print("=" * 70)


def demand_category(x):

    if x == 0:
        return "Zero"

    elif x <= 5:
        return "Low (1-5)"

    elif x <= 20:
        return "Medium (6-20)"

    elif x <= 100:
        return "High (21-100)"

    else:
        return "Very High (100+)"


test["DemandLevel"] = (
    test["ActualDemand"]
    .apply(demand_category)
)


category_results = (
    test
    .groupby("DemandLevel")
    .agg(
        Records=("ActualDemand", "size"),
        ActualDemand=("ActualDemand", "sum"),
        AverageActual=("ActualDemand", "mean"),
        AveragePrediction=("PredictedDemand", "mean"),
        MAE=("AbsoluteError", "mean"),
        Bias=("Error", "mean")
    )
    .reset_index()
)


category_results["WAPE"] = (
    category_results["MAE"]
    * category_results["Records"]
    /
    category_results["ActualDemand"]
    * 100
)


print(
    category_results.to_string(
        index=False
    )
)


# ============================================================
# PRODUCT-LEVEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT-LEVEL PERFORMANCE")
print("=" * 70)


product_results = (
    test
    .groupby(
        ["StockCode", "Description"]
    )
    .agg(
        TestDays=("ActualDemand", "size"),
        ActualDemand=("ActualDemand", "sum"),
        PredictedDemand=("PredictedDemand", "sum"),
        MAE=("AbsoluteError", "mean"),
        Bias=("Error", "mean")
    )
    .reset_index()
)


product_results["AbsoluteDifference"] = (
    np.abs(
        product_results["PredictedDemand"]
        -
        product_results["ActualDemand"]
    )
)


product_results["WAPE"] = np.where(
    product_results["ActualDemand"] > 0,

    (
        product_results["AbsoluteDifference"]
        /
        product_results["ActualDemand"]
    ) * 100,

    np.nan
)


# ============================================================
# BEST PRODUCTS
# ============================================================

print("\nTOP 10 PRODUCTS BY ACTUAL DEMAND:")

top_products = (
    product_results
    .sort_values(
        "ActualDemand",
        ascending=False
    )
    .head(10)
)

print(
    top_products[
        [
            "StockCode",
            "Description",
            "TestDays",
            "ActualDemand",
            "PredictedDemand",
            "MAE",
            "WAPE",
            "Bias"
        ]
    ].to_string(index=False)
)


# ============================================================
# WORST PRODUCTS
# ============================================================

print("\nTOP 10 PRODUCTS WITH HIGHEST WAPE:")

worst_products = (
    product_results[
        product_results["ActualDemand"] > 0
    ]
    .sort_values(
        "WAPE",
        ascending=False
    )
    .head(10)
)

print(
    worst_products[
        [
            "StockCode",
            "Description",
            "ActualDemand",
            "PredictedDemand",
            "MAE",
            "WAPE",
            "Bias"
        ]
    ].to_string(index=False)
)


# ============================================================
# HIGH-DEMAND ERROR
# ============================================================

high_demand = test[
    test["ActualDemand"] > 100
]

if len(high_demand) > 0:

    high_mae = (
        high_demand["AbsoluteError"]
        .mean()
    )

    high_bias = (
        high_demand["Error"]
        .mean()
    )

    print("\n" + "=" * 70)
    print("HIGH-DEMAND ANALYSIS")
    print("=" * 70)

    print(
        f"\nRecords with demand > 100: "
        f"{len(high_demand):,}"
    )

    print(
        f"High-demand MAE: "
        f"{high_mae:.2f}"
    )

    print(
        f"High-demand Bias: "
        f"{high_bias:.2f}"
    )


# ============================================================
# SAVE REPORTS
# ============================================================

test_output = (
    REPORT_DIR
    / "forecast_test_predictions.csv"
)

category_output = (
    REPORT_DIR
    / "forecast_demand_level_analysis.csv"
)

product_output = (
    REPORT_DIR
    / "forecast_product_performance.csv"
)


test.to_csv(
    test_output,
    index=False
)

category_results.to_csv(
    category_output,
    index=False
)

product_results.to_csv(
    product_output,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)

print("\nReports saved:")

print(test_output)
print(category_output)
print(product_output)

print("\nNext:")
print("→ Diagnose forecast errors")
print("→ Improve forecasting if required")
print("→ Build price-demand relationship")