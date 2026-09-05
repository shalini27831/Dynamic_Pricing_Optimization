import os
import pandas as pd
import numpy as np

INPUT_FILE = "reports/rl_price_recommendations_v2.csv"

OUTPUT_SUMMARY = "reports/pricing_v2_validation_summary.csv"
OUTPUT_ACTIONS = "reports/pricing_v2_action_analysis.csv"

os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("STEP 19 - FINAL PRICING VALIDATION")
print("=" * 70)

# ============================================================
# LOAD
# ============================================================

print("\nLoading Step 18 pricing recommendations...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "StockCode",
    "CurrentPrice_INR",
    "RecommendedPrice_INR",
    "PriceChangePct",
    "V3ForecastDemand",
    "ExpectedDemand",
    "Elasticity",
    "Action"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ============================================================
# REVENUE CALCULATION
# ============================================================

df["CurrentExpectedRevenue"] = (
    df["CurrentPrice_INR"] *
    df["V3ForecastDemand"]
)

df["OptimizedExpectedRevenue"] = (
    df["RecommendedPrice_INR"] *
    df["ExpectedDemand"]
)

df["RevenueImprovement_INR"] = (
    df["OptimizedExpectedRevenue"]
    -
    df["CurrentExpectedRevenue"]
)

df["RevenueImprovementPct"] = np.where(
    df["CurrentExpectedRevenue"] > 0,
    (
        df["RevenueImprovement_INR"]
        /
        df["CurrentExpectedRevenue"]
    ) * 100,
    0
)

# ============================================================
# DEMAND CHANGE
# ============================================================

df["DemandChangePct"] = np.where(
    df["V3ForecastDemand"] > 0,
    (
        (
            df["ExpectedDemand"]
            -
            df["V3ForecastDemand"]
        )
        /
        df["V3ForecastDemand"]
    ) * 100,
    0
)

# ============================================================
# PRICE-DEMAND CONSISTENCY
# ============================================================

df["PriceDemandConsistent"] = (

    (
        (df["PriceChangePct"] > 0)
        &
        (df["DemandChangePct"] <= 0)
    )

    |

    (
        (df["PriceChangePct"] < 0)
        &
        (df["DemandChangePct"] >= 0)
    )

    |

    (df["PriceChangePct"] == 0)
)

# ============================================================
# REVENUE IMPROVEMENT
# ============================================================

df["RevenueImproved"] = (
    df["RevenueImprovement_INR"] > 0
)

# ============================================================
# ACTION SUMMARY
# ============================================================

action_summary = (
    df.groupby("Action")
    .agg(
        Products=("StockCode", "count"),

        AvgCurrentPrice_INR=(
            "CurrentPrice_INR",
            "mean"
        ),

        AvgRecommendedPrice_INR=(
            "RecommendedPrice_INR",
            "mean"
        ),

        AvgPriceChangePct=(
            "PriceChangePct",
            "mean"
        ),

        AvgForecastDemand=(
            "V3ForecastDemand",
            "mean"
        ),

        AvgExpectedDemand=(
            "ExpectedDemand",
            "mean"
        ),

        AvgDemandChangePct=(
            "DemandChangePct",
            "mean"
        ),

        AvgRevenueImprovementPct=(
            "RevenueImprovementPct",
            "mean"
        ),

        RevenueImprovementRate=(
            "RevenueImproved",
            "mean"
        ),

        PriceDemandConsistency=(
            "PriceDemandConsistent",
            "mean"
        )
    )
    .reset_index()
)

action_summary["RevenueImprovementRate"] *= 100
action_summary["PriceDemandConsistency"] *= 100

# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Products Evaluated",
        "Average Current Price INR",
        "Average Recommended Price INR",
        "Average Price Change %",
        "Average Forecast Demand",
        "Average Expected Demand",
        "Average Demand Change %",
        "Average Revenue Improvement %",
        "Total Current Expected Revenue INR",
        "Total Optimized Expected Revenue INR",
        "Total Revenue Improvement INR",
        "Products With Revenue Improvement",
        "Revenue Improvement Rate %",
        "Price Demand Consistency %",
        "Increase Price Count",
        "Decrease Price Count",
        "Keep Price Count"
    ],

    "Value": [

        len(df),

        df["CurrentPrice_INR"].mean(),

        df["RecommendedPrice_INR"].mean(),

        df["PriceChangePct"].mean(),

        df["V3ForecastDemand"].mean(),

        df["ExpectedDemand"].mean(),

        df["DemandChangePct"].mean(),

        df["RevenueImprovementPct"].mean(),

        df["CurrentExpectedRevenue"].sum(),

        df["OptimizedExpectedRevenue"].sum(),

        df["RevenueImprovement_INR"].sum(),

        df["RevenueImproved"].sum(),

        df["RevenueImproved"].mean() * 100,

        df["PriceDemandConsistent"].mean() * 100,

        (df["Action"] == "Increase Price").sum(),

        (df["Action"] == "Decrease Price").sum(),

        (df["Action"] == "Keep Price").sum()
    ]
})

# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

action_summary.to_csv(
    OUTPUT_ACTIONS,
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PRICING VALIDATION COMPLETED")
print("=" * 70)

print(
    f"\nProducts evaluated: "
    f"{len(df):,}"
)

print(
    f"Average current price: "
    f"₹{df['CurrentPrice_INR'].mean():,.2f}"
)

print(
    f"Average recommended price: "
    f"₹{df['RecommendedPrice_INR'].mean():,.2f}"
)

print(
    f"\nAverage price change: "
    f"{df['PriceChangePct'].mean():.2f}%"
)

print(
    f"Average forecast demand: "
    f"{df['V3ForecastDemand'].mean():.2f}"
)

print(
    f"Average expected demand: "
    f"{df['ExpectedDemand'].mean():.2f}"
)

print(
    f"Average demand change: "
    f"{df['DemandChangePct'].mean():.2f}%"
)

print(
    f"\nAverage revenue improvement: "
    f"{df['RevenueImprovementPct'].mean():.2f}%"
)

print(
    f"Total revenue improvement: "
    f"₹{df['RevenueImprovement_INR'].sum():,.2f}"
)

print(
    f"Products improving revenue: "
    f"{df['RevenueImproved'].sum():,}"
    f" ({df['RevenueImproved'].mean() * 100:.2f}%)"
)

print(
    f"\nPrice-demand consistency: "
    f"{df['PriceDemandConsistent'].mean() * 100:.2f}%"
)

print("\nAction distribution:")
print(
    df["Action"].value_counts()
)

print("\n" + "-" * 70)
print("ACTION-LEVEL ANALYSIS")
print("-" * 70)

print(
    action_summary.to_string(index=False)
)

print("\nReports saved:")

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_ACTIONS
)

print("\n" + "=" * 70)