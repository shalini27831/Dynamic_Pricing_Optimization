import os
import numpy as np
import pandas as pd
import joblib

# ============================================================
# STEP 18 - BALANCED RL PRICING OPTIMIZATION ENGINE
# ============================================================

DATA_FILE = "data/processed/forecast_model_data.csv"
ELASTICITY_FILE = "reports/reliable_price_elasticity.csv"
MODEL_FILE = "models/lightgbm_v3_high_demand.pkl"

OUTPUT_FILE = "reports/rl_price_recommendations_v2.csv"

os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("STEP 18 - BALANCED RL PRICING OPTIMIZATION ENGINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading pricing dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Rows loaded: {len(df):,}")
print(f"Products: {df['StockCode'].nunique():,}")


# ============================================================
# 2. LOAD ELASTICITY
# ============================================================

print("\nLoading reliable elasticity estimates...")

elasticity_df = pd.read_csv(
    ELASTICITY_FILE,
    low_memory=False
)

print(
    f"Elasticity records: "
    f"{len(elasticity_df):,}"
)


# ============================================================
# 3. LOAD V3 MODEL
# ============================================================

print("\nLoading V3 demand model...")

model_package = joblib.load(MODEL_FILE)

if isinstance(model_package, dict):

    print("V3 model file contains a saved model package.")

    model = model_package["model"]

    features = model_package["features"]

    print(
        f"V3 model features configured: "
        f"{len(features)}"
    )

else:

    model = model_package

    features = list(
        model.feature_name_
    )

print(
    f"V3 model features detected: "
    f"{len(features)}"
)


# ============================================================
# 4. RECREATE HISTORICAL PRODUCT FEATURES
# ============================================================

print("\nRecreating V3 historical product features...")

df = df.sort_values(
    ["StockCode", "Date"]
).copy()

global_demand_mean = df["Demand"].mean()
global_demand_max = df["Demand"].max()
global_nonzero_rate = (
    df["Demand"] > 0
).mean()

previous_demand = (
    df.groupby("StockCode")["Demand"]
    .shift(1)
)

df["Product_HistoricalMean"] = (
    df.groupby("StockCode")["Demand"]
    .transform(
        lambda x:
        x.shift(1).expanding().mean()
    )
)

df["Product_HistoricalMax"] = (
    df.groupby("StockCode")["Demand"]
    .transform(
        lambda x:
        x.shift(1).expanding().max()
    )
)

df["Product_HistoricalNonZeroRate"] = (
    df.groupby("StockCode")["Demand"]
    .transform(
        lambda x:
        x.shift(1)
        .gt(0)
        .expanding()
        .mean()
    )
)

df["Product_HistoricalDays"] = (
    df.groupby("StockCode")
    .cumcount()
)

df["Demand_vs_ProductHistory"] = (
    df["Demand"] /
    df["Product_HistoricalMean"]
)

df["Product_HistoricalMean"] = (
    df["Product_HistoricalMean"]
    .fillna(global_demand_mean)
)

df["Product_HistoricalMax"] = (
    df["Product_HistoricalMax"]
    .fillna(global_demand_max)
)

df["Product_HistoricalNonZeroRate"] = (
    df["Product_HistoricalNonZeroRate"]
    .fillna(global_nonzero_rate)
)

df["Demand_vs_ProductHistory"] = (
    df["Demand_vs_ProductHistory"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(1.0)
)

print(
    "V3 historical features recreated successfully."
)


# ============================================================
# 5. GET LATEST PRODUCT STATE
# ============================================================

latest = (
    df.sort_values("Date")
    .groupby("StockCode")
    .tail(1)
    .copy()
)

print(
    f"\nLatest product states: "
    f"{len(latest):,}"
)


# ============================================================
# 6. PREPARE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in latest.columns
]

if missing_features:

    raise ValueError(
        f"Missing V3 features: "
        f"{missing_features}"
    )

X_latest = latest[features].copy()

X_latest = X_latest.replace(
    [np.inf, -np.inf],
    np.nan
)

X_latest = X_latest.fillna(0)


# ============================================================
# 7. FORECAST DEMAND
# ============================================================

latest["V3ForecastDemand"] = (
    model.predict(X_latest)
)

latest["V3ForecastDemand"] = (
    latest["V3ForecastDemand"]
    .clip(lower=0)
)


# ============================================================
# 8. MERGE ELASTICITY
# ============================================================

elasticity_columns = [
    "StockCode",
    "Elasticity",
    "Confidence"
]

available_columns = [
    col
    for col in elasticity_columns
    if col in elasticity_df.columns
]

elasticity_lookup = elasticity_df[
    available_columns
].copy()

latest = latest.merge(
    elasticity_lookup,
    on="StockCode",
    how="left"
)

latest["Elasticity"] = (
    latest["Elasticity"]
    .fillna(
        elasticity_df["Elasticity"].median()
    )
)

if "Confidence" not in latest.columns:
    latest["Confidence"] = "Fallback"

latest["Confidence"] = (
    latest["Confidence"]
    .fillna("Fallback")
)


# ============================================================
# 9. CURRENT PRICE
# ============================================================

latest["CurrentPrice_INR"] = (
    latest["AveragePrice_INR"]
)

latest["CurrentPrice_INR"] = (
    latest["CurrentPrice_INR"]
    .clip(lower=0.01)
)


# ============================================================
# 10. BALANCED PRICE ACTIONS
# ============================================================

price_actions = np.array([
    -0.10,
    -0.075,
    -0.05,
    -0.025,
     0.00,
     0.025,
     0.05,
     0.075,
     0.10
])


# ============================================================
# 11. PRICING SIMULATION
# ============================================================

results = []

print("\nRunning balanced pricing simulation...")

for _, row in latest.iterrows():

    stock_code = row["StockCode"]

    current_price = float(
        row["CurrentPrice_INR"]
    )

    base_demand = float(
        row["V3ForecastDemand"]
    )

    elasticity = float(
        row["Elasticity"]
    )

    confidence = row["Confidence"]

    best_reward = -np.inf

    best_action = None
    best_price = None
    best_demand = None
    best_revenue = None

    current_revenue = (
        current_price *
        base_demand
    )

    # --------------------------------------------------------
    # Test every possible price action
    # --------------------------------------------------------

    for price_change in price_actions:

        candidate_price = (
            current_price *
            (1 + price_change)
        )

        demand_ratio = (
            candidate_price /
            current_price
        )

        expected_demand = (
            base_demand *
            (
                demand_ratio
                ** elasticity
            )
        )

        expected_demand = max(
            expected_demand,
            0
        )

        expected_revenue = (
            candidate_price *
            expected_demand
        )

        # ----------------------------------------------------
        # Balanced reward
        # ----------------------------------------------------

        price_change_penalty = (
            abs(price_change)
            * current_revenue
            * 0.10
        )

        low_demand_penalty = (
            max(
                0,
                base_demand -
                expected_demand
            )
            *
            candidate_price
            *
            0.05
        )

        reward = (
            expected_revenue
            - price_change_penalty
            - low_demand_penalty
        )

        # ----------------------------------------------------
        # Prefer smaller changes when rewards are nearly equal
        # ----------------------------------------------------

        if reward > best_reward:

            best_reward = reward

            best_action = price_change

            best_price = candidate_price

            best_demand = expected_demand

            best_revenue = expected_revenue

    # ========================================================
    # ACTION LABEL
    # ========================================================

    if best_action > 0:
        action = "Increase Price"

    elif best_action < 0:
        action = "Decrease Price"

    else:
        action = "Keep Price"

    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({

        "StockCode": stock_code,

        "Date": row["Date"],

        "CurrentPrice_INR":
            current_price,

        "RecommendedPrice_INR":
            best_price,

        "PriceChangePct":
            best_action * 100,

        "V3ForecastDemand":
            base_demand,

        "ExpectedDemand":
            best_demand,

        "Elasticity":
            elasticity,

        "ElasticityConfidence":
            confidence,

        "CurrentExpectedRevenue":
            current_revenue,

        "OptimizedExpectedRevenue":
            best_revenue,

        "RevenueImprovement_INR":
            best_revenue -
            current_revenue,

        "RevenueImprovementPct":
    (
        (
            best_revenue -
            current_revenue
        )
        /
        current_revenue
        * 100
        if current_revenue > 0
        else 0
    ),

        "RLReward":
            best_reward,

        "Action":
            action
    })


# ============================================================
# 12. CREATE OUTPUT
# ============================================================

recommendations = pd.DataFrame(
    results
)

recommendations = recommendations.sort_values(
    "RLReward",
    ascending=False
)


# ============================================================
# 13. SAVE
# ============================================================

recommendations.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BALANCED RL PRICING OPTIMIZATION COMPLETED")
print("=" * 70)

print(
    f"\nProducts evaluated: "
    f"{len(recommendations):,}"
)

print("\nRecommended actions:")

print(
    recommendations["Action"]
    .value_counts()
)

print(
    f"\nAverage current price: "
    f"₹{recommendations['CurrentPrice_INR'].mean():,.2f}"
)

print(
    f"Average recommended price: "
    f"₹{recommendations['RecommendedPrice_INR'].mean():,.2f}"
)

print(
    f"\nAverage price change: "
    f"{recommendations['PriceChangePct'].mean():.2f}%"
)

print(
    f"Average expected revenue improvement: "
    f"{recommendations['RevenueImprovementPct'].mean():.2f}%"
)

print(
    f"\nTotal expected revenue improvement: "
    f"₹{recommendations['RevenueImprovement_INR'].sum():,.2f}"
)

print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 15. TOP RECOMMENDATIONS
# ============================================================

print("\nTop 10 pricing recommendations:")

display_columns = [
    "StockCode",
    "CurrentPrice_INR",
    "RecommendedPrice_INR",
    "PriceChangePct",
    "V3ForecastDemand",
    "ExpectedDemand",
    "Elasticity",
    "ElasticityConfidence",
    "RevenueImprovementPct",
    "RLReward",
    "Action"
]

print(
    recommendations[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)