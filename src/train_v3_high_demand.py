import os
import pickle
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/forecast_model_data.csv"

MODEL_PATH = "models/lightgbm_v3_high_demand.pkl"

REPORT_PATH = "reports/v3_validation_results.csv"

PREDICTION_PATH = "reports/v3_validation_predictions.csv"

TRAIN_END = "2011-05-01"
VALIDATION_START = "2011-05-02"
VALIDATION_END = "2011-08-20"

TEST_START = "2011-08-21"
TEST_END = "2011-12-09"

TARGET = "Demand_NextDay"


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURES = [
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

    "Demand_Trend_7_28",

    "AveragePrice_INR",
    "Price_Lag_1",
    "Price_Change_Pct",
    "Price_RollingMean_7",

    "Revenue_INR",
    "Revenue_Lag_1",

    "TransactionCount",
    "Transactions_RollingMean_7",

    "DayOfWeek",
    "WeekOfYear",
    "Month",
    "Quarter",
    "IsWeekend",

    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "WeekOfYear_Sin",
    "WeekOfYear_Cos",
    "Month_Sin",
    "Month_Cos",

    "Product_Age_Days",
    "Observed",
]


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_wape(actual, predicted):

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(actual - predicted))
        / denominator
        * 100
    )


def calculate_bias(actual, predicted):

    return np.mean(
        predicted - actual
    )


def evaluate(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    wape = calculate_wape(
        actual,
        predicted
    )

    bias = calculate_bias(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, wape, bias, r2


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DYNAMIC PRICING ENGINE")
print("STEP 12 - V3 HIGH-DEMAND-AWARE FORECASTING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading forecast data...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Products: {df['StockCode'].nunique():,}"
)

print(
    f"Date range: "
    f"{df['Date'].min().date()} "
    f"to "
    f"{df['Date'].max().date()}"
)


# ============================================================
# FEATURE CHECK
# ============================================================

missing = [
    col
    for col in BASE_FEATURES
    if col not in df.columns
]

if missing:

    print("\nERROR: Missing features:")

    for col in missing:
        print(f" - {col}")

    raise ValueError(
        "Required features are missing."
    )

print(
    "\nAll base features are available."
)


# ============================================================
# SORT DATA
# ============================================================

print(
    "\nSorting by product and date..."
)

df = df.sort_values(
    ["StockCode", "Date"]
).reset_index(
    drop=True
)


# ============================================================
# CAUSAL PRODUCT HISTORY FEATURES
# ============================================================

print(
    "\nCreating causal product-level history features..."
)

# These features use information known at date T.
# The target is demand at T+1.
#
# Therefore they do NOT use future demand.

group = df.groupby(
    "StockCode",
    sort=False
)["Demand"]


# Historical mean including current-day demand.
#
# This is valid because current-day demand is already known
# when making a next-day prediction.

df["Product_HistoricalMean"] = (
    group.expanding()
    .mean()
    .reset_index(
        level=0,
        drop=True
    )
)


# Historical maximum demand observed up to today.

df["Product_HistoricalMax"] = (
    group.cummax()
)


# Historical non-zero demand rate.

df["Product_HistoricalNonZeroRate"] = (
    df["Demand"]
    .gt(0)
    .groupby(
        df["StockCode"]
    )
    .expanding()
    .mean()
    .reset_index(
        level=0,
        drop=True
    )
)


# Number of observed product days so far.

df["Product_HistoricalDays"] = (
    df.groupby(
        "StockCode"
    ).cumcount() + 1
)


# Ratio of today's demand to historical mean.
# Protected against division by zero.

df["Demand_vs_ProductHistory"] = (
    df["Demand"]
    /
    df["Product_HistoricalMean"]
    .replace(0, np.nan)
)

df["Demand_vs_ProductHistory"] = (
    df["Demand_vs_ProductHistory"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# FINAL FEATURES
# ============================================================

HISTORY_FEATURES = [
    "Product_HistoricalMean",
    "Product_HistoricalMax",
    "Product_HistoricalNonZeroRate",
    "Product_HistoricalDays",
    "Demand_vs_ProductHistory",
]

FEATURES = (
    BASE_FEATURES
    + HISTORY_FEATURES
)


print(
    f"\nTotal model features: "
    f"{len(FEATURES)}"
)


# ============================================================
# MODEL DATA
# ============================================================

model_df = df[
    FEATURES
    + [TARGET, "Date", "StockCode"]
].copy()

model_df = model_df.dropna(
    subset=FEATURES + [TARGET]
)

print(
    f"Rows available for modeling: "
    f"{len(model_df):,}"
)


# ============================================================
# TIME SPLIT
# ============================================================

print(
    "\nCreating chronological train/validation/test split..."
)

train_df = model_df[
    model_df["Date"]
    <= pd.Timestamp(TRAIN_END)
].copy()

validation_df = model_df[
    (
        model_df["Date"]
        >= pd.Timestamp(VALIDATION_START)
    )
    &
    (
        model_df["Date"]
        <= pd.Timestamp(VALIDATION_END)
    )
].copy()

test_df = model_df[
    (
        model_df["Date"]
        >= pd.Timestamp(TEST_START)
    )
    &
    (
        model_df["Date"]
        <= pd.Timestamp(TEST_END)
    )
].copy()


print(
    f"Train      : {len(train_df):,}"
)

print(
    f"Validation : {len(validation_df):,}"
)

print(
    f"Test       : {len(test_df):,}"
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X_train = train_df[FEATURES]

X_validation = validation_df[FEATURES]

X_test = test_df[FEATURES]

y_train = train_df[TARGET]

y_validation = validation_df[TARGET]

y_test = test_df[TARGET]


# ============================================================
# HIGH-DEMAND TRAINING WEIGHTS
# ============================================================

print(
    "\nCreating demand-aware training weights..."
)

# Base weight is 1.
#
# Larger demand receives more influence, but the square-root
# transformation prevents extreme products from dominating.

sample_weights = (
    1
    + np.sqrt(
        np.maximum(
            y_train,
            0
        )
    )
)


# Cap extreme weights.

sample_weights = np.minimum(
    sample_weights,
    15
)

print(
    f"Minimum weight: "
    f"{sample_weights.min():.2f}"
)

print(
    f"Maximum weight: "
    f"{sample_weights.max():.2f}"
)

print(
    f"Average weight: "
    f"{sample_weights.mean():.2f}"
)


# ============================================================
# V3 LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING V3 LIGHTGBM MODEL")
print("=" * 70)

model = lgb.LGBMRegressor(

    objective="regression",

    n_estimators=900,

    learning_rate=0.025,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=40,

    subsample=0.85,

    colsample_bytree=0.90,

    reg_alpha=0.15,

    reg_lambda=0.75,

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


print(
    "\nTraining model..."
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print(
    "V3 model training complete."
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print(
    "\nGenerating validation predictions..."
)

validation_prediction = model.predict(
    X_validation
)

validation_prediction = np.maximum(
    validation_prediction,
    0
)


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("V3 VALIDATION PERFORMANCE")
print("=" * 70)

mae, rmse, wape, bias, r2 = evaluate(
    y_validation,
    validation_prediction
)

print(
    f"\nMAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"WAPE : {wape:.2f}%"
)

print(
    f"Bias : {bias:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# DEMAND LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("V3 VALIDATION BY DEMAND LEVEL")
print("=" * 70)

analysis = pd.DataFrame({
    "Actual": y_validation.values,
    "Prediction": validation_prediction
})

analysis["Error"] = (
    analysis["Prediction"]
    - analysis["Actual"]
)

analysis["AbsError"] = (
    np.abs(
        analysis["Error"]
    )
)


def demand_level(value):

    if value == 0:
        return "Zero"

    if value <= 5:
        return "Low (1-5)"

    if value <= 20:
        return "Medium (6-20)"

    if value <= 100:
        return "High (21-100)"

    return "Very High (100+)"


analysis["DemandLevel"] = (
    analysis["Actual"]
    .apply(demand_level)
)


summary = (
    analysis
    .groupby("DemandLevel")
    .agg(
        Records=("Actual", "size"),
        ActualDemand=("Actual", "sum"),
        AverageActual=("Actual", "mean"),
        AveragePrediction=("Prediction", "mean"),
        MAE=("AbsError", "mean"),
        Bias=("Error", "mean")
    )
    .reset_index()
)


wape_values = []

for level in summary["DemandLevel"]:

    subset = analysis[
        analysis["DemandLevel"] == level
    ]

    wape_level = calculate_wape(
        subset["Actual"].values,
        subset["Prediction"].values
    )

    wape_values.append(
        wape_level
    )


summary["WAPE"] = wape_values


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# HIGH-DEMAND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("V3 HIGH-DEMAND ANALYSIS")
print("=" * 70)

high = analysis[
    analysis["Actual"] > 100
]

print(
    f"\nRecords with demand > 100: "
    f"{len(high):,}"
)

if len(high) > 0:

    print(
        f"Actual average demand: "
        f"{high['Actual'].mean():.2f}"
    )

    print(
        f"Predicted average demand: "
        f"{high['Prediction'].mean():.2f}"
    )

    print(
        f"High-demand MAE: "
        f"{high['AbsError'].mean():.2f}"
    )

    print(
        f"High-demand Bias: "
        f"{high['Error'].mean():.2f}"
    )

    print(
        f"High-demand WAPE: "
        f"{calculate_wape(high['Actual'], high['Prediction']):.2f}%"
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "reports",
    exist_ok=True
)

prediction_output = validation_df[
    ["Date", "StockCode"]
].copy()

prediction_output["ActualDemand"] = (
    y_validation.values
)

prediction_output["PredictedDemand"] = (
    validation_prediction
)

prediction_output["Error"] = (
    prediction_output["PredictedDemand"]
    - prediction_output["ActualDemand"]
)

prediction_output["AbsoluteError"] = (
    np.abs(
        prediction_output["Error"]
    )
)

prediction_output.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model": model,

    "features": FEATURES,

    "model_version": "V3",

    "model_type":
        "High-Demand-Aware LightGBM",

    "training_weight":
        "1 + sqrt(Demand), capped at 15",

    "target":
        TARGET
}


with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model_package,
        f
    )


# ============================================================
# SAVE REPORT
# ============================================================

report = pd.DataFrame([
    {
        "Model": "V3 High-Demand-Aware LightGBM",

        "MAE": mae,

        "RMSE": rmse,

        "WAPE": wape,

        "Bias": bias,

        "R2": r2,

        "High_Demand_MAE":
            high["AbsError"].mean()
            if len(high) > 0
            else np.nan,

        "High_Demand_Bias":
            high["Error"].mean()
            if len(high) > 0
            else np.nan,

        "High_Demand_WAPE":
            calculate_wape(
                high["Actual"],
                high["Prediction"]
            )
            if len(high) > 0
            else np.nan
    }
])

report.to_csv(
    REPORT_PATH,
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    "reports/v3_feature_importance.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)

print("\nSaved model:")
print(
    f"→ {MODEL_PATH}"
)

print("\nSaved reports:")
print(
    f"→ {REPORT_PATH}"
)

print(
    "→ reports/v3_validation_predictions.csv"
)

print(
    "→ reports/v3_feature_importance.csv"
)

print("\nNext:")
print(
    "→ Compare V1 vs V2 vs V3"
)

print(
    "→ Select final forecasting model"
)

print(
    "→ Evaluate final model on untouched test data"
)