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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DYNAMIC PRICING ENGINE")
print("STEP 7 - TIME-BASED TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Date", "StockCode"]
).reset_index(drop=True)

print(f"\nTotal records: {len(df):,}")
print(f"Products: {df['StockCode'].nunique():,}")
print(f"Start date: {df['Date'].min().date()}")
print(f"End date: {df['Date'].max().date()}")


# ============================================================
# DETERMINE TIME BOUNDARIES
# ============================================================

unique_dates = sorted(
    df["Date"].unique()
)

n_dates = len(unique_dates)

train_end_index = int(
    n_dates * 0.70
)

validation_end_index = int(
    n_dates * 0.85
)

train_end_date = (
    unique_dates[train_end_index - 1]
)

validation_end_date = (
    unique_dates[validation_end_index - 1]
)


# ============================================================
# CREATE SPLITS
# ============================================================

train = df[
    df["Date"] <= train_end_date
].copy()

validation = df[
    (df["Date"] > train_end_date)
    &
    (df["Date"] <= validation_end_date)
].copy()

test = df[
    df["Date"] > validation_end_date
].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TIME SPLIT")
print("=" * 70)

print("\nTRAIN")
print(f"Start: {train['Date'].min().date()}")
print(f"End:   {train['Date'].max().date()}")
print(f"Rows:  {len(train):,}")

print("\nVALIDATION")
print(f"Start: {validation['Date'].min().date()}")
print(f"End:   {validation['Date'].max().date()}")
print(f"Rows:  {len(validation):,}")

print("\nTEST")
print(f"Start: {test['Date'].min().date()}")
print(f"End:   {test['Date'].max().date()}")
print(f"Rows:  {len(test):,}")


# ============================================================
# CHECK FOR DATE OVERLAP
# ============================================================

train_dates = set(train["Date"].unique())
validation_dates = set(validation["Date"].unique())
test_dates = set(test["Date"].unique())

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    f"\nTrain ∩ Validation: "
    f"{len(train_dates & validation_dates)} dates"
)

print(
    f"Train ∩ Test: "
    f"{len(train_dates & test_dates)} dates"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_dates & test_dates)} dates"
)


if (
    len(train_dates & validation_dates) == 0
    and
    len(train_dates & test_dates) == 0
    and
    len(validation_dates & test_dates) == 0
):

    print("\n✓ No date overlap detected.")
    print("✓ Time-based split is valid.")

else:

    print("\nWARNING: Date overlap detected.")


# ============================================================
# SAVE
# ============================================================

train_file = (
    OUTPUT_DIR
    / "train.csv"
)

validation_file = (
    OUTPUT_DIR
    / "validation.csv"
)

test_file = (
    OUTPUT_DIR
    / "test.csv"
)


train.to_csv(
    train_file,
    index=False
)

validation.to_csv(
    validation_file,
    index=False
)

test.to_csv(
    test_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(train_file)
print(validation_file)
print(test_file)