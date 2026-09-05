import os
import pandas as pd
import numpy as np

INPUT_FILE = "reports/price_elasticity.csv"
OUTPUT_FILE = "reports/reliable_price_elasticity.csv"

MIN_R2 = 0.03
MIN_OBSERVATIONS = 50
MIN_ELASTICITY = -3.0
MAX_ELASTICITY = -0.10

os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("STEP 16 - ELASTICITY RELIABILITY FILTERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"\nInput elasticity records: {len(df):,}")

# ------------------------------------------------------------
# Calculate fallback median
# ------------------------------------------------------------

valid_values = df.loc[
    (df["Elasticity"] < 0) &
    (df["Elasticity"] >= MIN_ELASTICITY) &
    (df["Elasticity"] <= MAX_ELASTICITY),
    "Elasticity"
]

fallback_elasticity = valid_values.median()

print(
    f"Fallback median elasticity: "
    f"{fallback_elasticity:.4f}"
)


# ------------------------------------------------------------
# Reliability rule
# ------------------------------------------------------------

df["Reliable"] = (
    (df["Observations"] >= MIN_OBSERVATIONS) &
    (df["Elasticity_R2"] >= MIN_R2) &
    (df["Elasticity"] >= MIN_ELASTICITY) &
    (df["Elasticity"] <= MAX_ELASTICITY)
)


# ------------------------------------------------------------
# Preserve original value
# ------------------------------------------------------------

df["OriginalElasticity"] = df["Elasticity"]


# ------------------------------------------------------------
# Keep reliable elasticity
# Replace unreliable values with fallback
# ------------------------------------------------------------

df["Elasticity"] = np.where(
    df["Reliable"],
    df["Elasticity"],
    fallback_elasticity
)


# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

def confidence(row):

    if (
        row["Reliable"]
        and row["Observations"] >= 100
        and row["Elasticity_R2"] >= 0.20
    ):
        return "High"

    elif row["Reliable"]:
        return "Medium"

    return "Fallback"


df["Confidence"] = df.apply(
    confidence,
    axis=1
)


# ------------------------------------------------------------
# Elasticity type
# ------------------------------------------------------------

df["ElasticityType"] = np.where(
    df["Elasticity"] < -1,
    "Elastic",
    "Inelastic"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ELASTICITY FILTERING COMPLETED")
print("=" * 70)

print(
    f"\nReliable estimates: "
    f"{df['Reliable'].sum():,}"
)

print(
    f"Fallback estimates: "
    f"{(~df['Reliable']).sum():,}"
)

print("\nConfidence:")
print(
    df["Confidence"].value_counts()
)

print("\nFinal elasticity statistics:")
print(
    df["Elasticity"].describe()
)

print("\nElasticity range:")
print(
    f"Minimum: {df['Elasticity'].min():.4f}"
)

print(
    f"Maximum: {df['Elasticity'].max():.4f}"
)

print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 70)