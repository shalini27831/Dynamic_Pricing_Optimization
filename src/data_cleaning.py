from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "online_retail_II.xlsx"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset is originally in GBP.
# This is a project normalization rate so that the
# business-facing pricing system operates in INR.
GBP_TO_INR = 100.0


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():

    print("\nLoading Online Retail II...")

    sheets = pd.read_excel(
        RAW_FILE,
        sheet_name=None
    )

    frames = []

    for sheet_name, data in sheets.items():

        print(
            f"  {sheet_name}: "
            f"{len(data):,} rows"
        )

        data["SourceSheet"] = sheet_name

        frames.append(data)

    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"\nTotal raw transactions: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):

    df = df.copy()

    column_mapping = {}

    for column in df.columns:

        clean_name = (
            str(column)
            .strip()
            .replace(" ", "")
            .replace("_", "")
            .lower()
        )

        if clean_name in ["invoiceno", "invoice"]:
           column_mapping[column] = "InvoiceNo"

        elif clean_name == "stockcode":
            column_mapping[column] = "StockCode"

        elif clean_name == "description":
            column_mapping[column] = "Description"

        elif clean_name == "quantity":
            column_mapping[column] = "Quantity"

        elif clean_name == "invoicedate":
            column_mapping[column] = "InvoiceDate"

        elif clean_name in ["price", "unitprice"]:
            column_mapping[column] = "UnitPrice"

        elif clean_name in ["customerid", "customer id"]:
            column_mapping[column] = "CustomerID"

        elif clean_name == "country":
            column_mapping[column] = "Country"

    df = df.rename(columns=column_mapping)

    return df


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

def convert_data_types(df):

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
        .str.strip()
    )

    df["Description"] = (
        df["Description"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

def clean_transactions(df):

    df = df.copy()

    original_rows = len(df)

    # --------------------------------------------------------
    # Missing essential information
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "StockCode",
            "InvoiceDate",
            "Quantity",
            "UnitPrice"
        ]
    )

    # --------------------------------------------------------
    # Remove cancelled invoices
    # --------------------------------------------------------

    cancellation_mask = (
        df["InvoiceNo"]
        .astype(str)
        .str.upper()
        .str.startswith("C")
    )

    cancelled_count = cancellation_mask.sum()

    df = df.loc[
        ~cancellation_mask
    ].copy()

    # --------------------------------------------------------
    # Demand must be positive
    # --------------------------------------------------------

    negative_quantity = (
        df["Quantity"] < 0
    ).sum()

    zero_quantity = (
        df["Quantity"] == 0
    ).sum()

    df = df[
        df["Quantity"] > 0
    ].copy()

    # --------------------------------------------------------
    # Price must be positive
    # --------------------------------------------------------

    invalid_price = (
        df["UnitPrice"] <= 0
    ).sum()

    df = df[
        df["UnitPrice"] > 0
    ].copy()

    # --------------------------------------------------------
    # Remove duplicate transactions
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Remove invalid descriptions
    # --------------------------------------------------------

    df = df[
        df["Description"].notna()
    ].copy()

    df = df[
        df["Description"].str.strip() != ""
    ].copy()

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    df["Revenue_GBP"] = (
        df["Quantity"] *
        df["UnitPrice"]
    )

    # --------------------------------------------------------
    # GBP → INR
    # --------------------------------------------------------

    df["UnitPrice_INR"] = (
        df["UnitPrice"] *
        GBP_TO_INR
    )

    df["Revenue_INR"] = (
        df["Quantity"] *
        df["UnitPrice_INR"]
    )

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    df["Date"] = (
        df["InvoiceDate"]
        .dt.normalize()
    )

    df["Year"] = (
        df["InvoiceDate"].dt.year
    )

    df["Month"] = (
        df["InvoiceDate"].dt.month
    )

    df["DayOfWeek"] = (
        df["InvoiceDate"].dt.dayofweek
    )

    # --------------------------------------------------------
    # Cleaning report
    # --------------------------------------------------------

    final_rows = len(df)

    report = {
        "raw_rows": original_rows,
        "cancelled_transactions": int(cancelled_count),
        "negative_quantity": int(negative_quantity),
        "zero_quantity": int(zero_quantity),
        "invalid_price": int(invalid_price),
        "duplicates": int(duplicate_count),
        "final_rows": final_rows,
        "rows_removed": original_rows - final_rows,
        "retention_rate": (
            final_rows / original_rows * 100
        )
    }

    return df, report


# ============================================================
# PRODUCT DAILY AGGREGATION
# ============================================================

def create_daily_dataset(df):

    print("\nCreating product-level daily dataset...")

    daily = (
        df.groupby(
            [
                "Date",
                "StockCode",
                "Description"
            ],
            as_index=False
        )
        .agg(
            Demand=(
                "Quantity",
                "sum"
            ),

            AveragePrice_INR=(
                "UnitPrice_INR",
                "mean"
            ),

            Revenue_INR=(
                "Revenue_INR",
                "sum"
            ),

            TransactionCount=(
                "InvoiceNo",
                "nunique"
            )
        )
    )

    daily = daily.sort_values(
        [
            "StockCode",
            "Date"
        ]
    ).reset_index(drop=True)

    print(
        f"Daily product records: "
        f"{len(daily):,}"
    )

    print(
        f"Unique products: "
        f"{daily['StockCode'].nunique():,}"
    )

    return daily


# ============================================================
# SAVE DATA
# ============================================================

def save_outputs(df, daily, report):

    transaction_file = (
        PROCESSED_DIR /
        "clean_transactions.csv"
    )

    daily_file = (
        PROCESSED_DIR /
        "daily_product_demand.csv"
    )

    report_file = (
        REPORT_DIR /
        "data_quality_report.csv"
    )

    df.to_csv(
        transaction_file,
        index=False
    )

    daily.to_csv(
        daily_file,
        index=False
    )

    report_df = pd.DataFrame(
        [report]
    )

    report_df.to_csv(
        report_file,
        index=False
    )

    print("\nFiles created:")
    print(f"  {transaction_file}")
    print(f"  {daily_file}")
    print(f"  {report_file}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("DYNAMIC PRICING ENGINE")
    print("STEP 2 - DATA CLEANING PIPELINE")
    print("=" * 70)

    df = load_raw_data()

    df = standardize_columns(df)

    df = convert_data_types(df)

    clean_df, report = clean_transactions(df)

    daily_df = create_daily_dataset(
        clean_df
    )

    save_outputs(
        clean_df,
        daily_df,
        report
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"\nRaw rows: "
        f"{report['raw_rows']:,}"
    )

    print(
        f"Final rows: "
        f"{report['final_rows']:,}"
    )

    print(
        f"Rows removed: "
        f"{report['rows_removed']:,}"
    )

    print(
        f"Retention: "
        f"{report['retention_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()