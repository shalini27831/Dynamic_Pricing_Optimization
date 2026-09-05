from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "online_retail_II.xlsx"

# ---------------------------------------------------------
# LOAD WORKBOOK
# ---------------------------------------------------------

print("=" * 70)
print("DYNAMIC PRICING ENGINE - DATA AUDIT")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "Place online_retail_II.xlsx inside the data folder."
    )

excel = pd.ExcelFile(DATA_PATH)

print("\nWorkbook:")
print(DATA_PATH)

print("\nSheets:")
for sheet in excel.sheet_names:
    print(f"  - {sheet}")

# ---------------------------------------------------------
# AUDIT EACH SHEET
# ---------------------------------------------------------

for sheet in excel.sheet_names:

    print("\n" + "=" * 70)
    print(f"SHEET: {sheet}")
    print("=" * 70)

    df = pd.read_excel(DATA_PATH, sheet_name=sheet)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isna().sum()

    for column, count in missing.items():
        if count > 0:
            percentage = count / len(df) * 100
            print(
                f"  {column}: {count:,} "
                f"({percentage:.2f}%)"
            )

    # -----------------------------------------------------
    # DATE AUDIT
    # -----------------------------------------------------

    if "InvoiceDate" in df.columns:

        dates = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        print("\nDate range:")
        print(f"  Start: {dates.min()}")
        print(f"  End:   {dates.max()}")

        print(
            f"  Invalid dates: "
            f"{dates.isna().sum():,}"
        )

    # -----------------------------------------------------
    # PRODUCT AUDIT
    # -----------------------------------------------------

    if "StockCode" in df.columns:

        print("\nProduct information:")
        print(
            f"  Unique StockCodes: "
            f"{df['StockCode'].nunique():,}"
        )

    if "Description" in df.columns:

        print(
            f"  Unique descriptions: "
            f"{df['Description'].nunique():,}"
        )

    # -----------------------------------------------------
    # QUANTITY AUDIT
    # -----------------------------------------------------

    if "Quantity" in df.columns:

        print("\nQuantity audit:")

        print(
            f"  Negative quantity: "
            f"{(df['Quantity'] < 0).sum():,}"
        )

        print(
            f"  Zero quantity: "
            f"{(df['Quantity'] == 0).sum():,}"
        )

        print(
            f"  Positive quantity: "
            f"{(df['Quantity'] > 0).sum():,}"
        )

        print(
            f"  Maximum quantity: "
            f"{df['Quantity'].max():,.2f}"
        )

    # -----------------------------------------------------
    # PRICE AUDIT
    # -----------------------------------------------------

    price_column = None

    if "Price" in df.columns:
        price_column = "Price"

    elif "UnitPrice" in df.columns:
        price_column = "UnitPrice"

    if price_column:

        print("\nPrice audit:")

        print(
            f"  Negative price: "
            f"{(df[price_column] < 0).sum():,}"
        )

        print(
            f"  Zero price: "
            f"{(df[price_column] == 0).sum():,}"
        )

        print(
            f"  Positive price: "
            f"{(df[price_column] > 0).sum():,}"
        )

        print(
            f"  Minimum price: "
            f"{df[price_column].min():,.4f}"
        )

        print(
            f"  Maximum price: "
            f"{df[price_column].max():,.4f}"
        )

    # -----------------------------------------------------
    # DUPLICATE AUDIT
    # -----------------------------------------------------

    print("\nDuplicate rows:")
    print(f"  {df.duplicated().sum():,}")

    # -----------------------------------------------------
    # SAMPLE
    # -----------------------------------------------------

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

print("\n" + "=" * 70)
print("DATA AUDIT COMPLETE")
print("=" * 70)