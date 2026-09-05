import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/forecast_model_data.csv"
OUTPUT_FILE = "reports/price_elasticity.csv"

MIN_OBSERVATIONS = 30
MIN_PRICE_VARIATION = 0.01

os.makedirs("reports", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 14 - PRICE ELASTICITY ESTIMATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")
print(f"Products: {df['StockCode'].nunique():,}")


# ============================================================
# CLEAN REQUIRED COLUMNS
# ============================================================

required_columns = [
    "StockCode",
    "AveragePrice_INR",
    "Demand"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[
    required_columns
].copy()

df["StockCode"] = df["StockCode"].astype(str)

df["AveragePrice_INR"] = pd.to_numeric(
    df["AveragePrice_INR"],
    errors="coerce"
)

df["Demand"] = pd.to_numeric(
    df["Demand"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "StockCode",
        "AveragePrice_INR",
        "Demand"
    ]
)

# Price and demand must be non-negative
df = df[
    (df["AveragePrice_INR"] > 0) &
    (df["Demand"] >= 0)
].copy()

print(f"Rows after cleaning: {len(df):,}")


# ============================================================
# LOG TRANSFORMATION
# ============================================================

# Elasticity model:
#
# log(Demand + 1) = alpha + beta * log(Price)
#
# beta approximates price elasticity.
#
# beta < 0  -> demand decreases as price increases
# beta > 0  -> unusual positive relationship
#
# We use log1p(Demand) because demand contains zeros.

df["LogPrice"] = np.log(
    df["AveragePrice_INR"]
)

df["LogDemand"] = np.log1p(
    df["Demand"]
)


# ============================================================
# ESTIMATE ELASTICITY PRODUCT BY PRODUCT
# ============================================================

results = []

products = df["StockCode"].unique()

print(
    f"\nEstimating elasticity for "
    f"{len(products):,} products..."
)

for i, stock_code in enumerate(products, start=1):

    product_df = df[
        df["StockCode"] == stock_code
    ].copy()

    # --------------------------------------------------------
    # Minimum observations
    # --------------------------------------------------------

    if len(product_df) < MIN_OBSERVATIONS:
        continue

    # --------------------------------------------------------
    # Check price variation
    # --------------------------------------------------------

    price_min = product_df["AveragePrice_INR"].min()
    price_max = product_df["AveragePrice_INR"].max()

    if price_min <= 0:
        continue

    price_variation = (
        (price_max - price_min) / price_min
    )

    if price_variation < MIN_PRICE_VARIATION:
        continue

    # --------------------------------------------------------
    # Prepare model
    # --------------------------------------------------------

    X = product_df[
        ["LogPrice"]
    ]

    y = product_df[
        "LogDemand"
    ]

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------

    model = Ridge(
        alpha=1.0
    )

    model.fit(X, y)

    elasticity = float(
        model.coef_[0]
    )

    intercept = float(
        model.intercept_
    )

    r2 = float(
        model.score(X, y)
    )

    # --------------------------------------------------------
    # Additional statistics
    # --------------------------------------------------------

    avg_price = product_df[
        "AveragePrice_INR"
    ].mean()

    avg_demand = product_df[
        "Demand"
    ].mean()

    median_demand = product_df[
        "Demand"
    ].median()

    observations = len(product_df)

    # --------------------------------------------------------
    # Elasticity classification
    # --------------------------------------------------------

    if elasticity < -1:
        elasticity_type = "Elastic"

    elif elasticity < 0:
        elasticity_type = "Inelastic"

    elif elasticity > 0:
        elasticity_type = "Positive"

    else:
        elasticity_type = "Neutral"

    # --------------------------------------------------------
    # Confidence classification
    # --------------------------------------------------------

    if observations >= 100 and r2 >= 0.20:
        confidence = "High"

    elif observations >= 50 and r2 >= 0.10:
        confidence = "Medium"

    else:
        confidence = "Low"

    results.append({
        "StockCode": stock_code,
        "Observations": observations,
        "AveragePrice_INR": avg_price,
        "AverageDemand": avg_demand,
        "MedianDemand": median_demand,
        "MinPrice_INR": price_min,
        "MaxPrice_INR": price_max,
        "PriceVariationPct": price_variation * 100,
        "Elasticity": elasticity,
        "Elasticity_R2": r2,
        "Intercept": intercept,
        "ElasticityType": elasticity_type,
        "Confidence": confidence
    })

    if i % 500 == 0:
        print(
            f"Processed {i:,} / "
            f"{len(products):,} products..."
        )


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

elasticity_df = pd.DataFrame(
    results
)

if elasticity_df.empty:
    raise ValueError(
        "No products passed the elasticity criteria."
    )


# ============================================================
# SORT
# ============================================================

elasticity_df = elasticity_df.sort_values(
    by="Elasticity"
)


# ============================================================
# SAVE
# ============================================================

elasticity_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ELASTICITY ESTIMATION COMPLETED")
print("=" * 70)

print(
    f"\nProducts with elasticity estimates: "
    f"{len(elasticity_df):,}"
)

print(
    f"Elastic products: "
    f"{(elasticity_df['ElasticityType'] == 'Elastic').sum():,}"
)

print(
    f"Inelastic products: "
    f"{(elasticity_df['ElasticityType'] == 'Inelastic').sum():,}"
)

print(
    f"Positive elasticity products: "
    f"{(elasticity_df['ElasticityType'] == 'Positive').sum():,}"
)

print(
    "\nElasticity statistics:"
)

print(
    elasticity_df["Elasticity"].describe()
)

print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\nTop 10 most price-sensitive products:")

print(
    elasticity_df[
        [
            "StockCode",
            "Elasticity",
            "Elasticity_R2",
            "Observations",
            "Confidence"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)