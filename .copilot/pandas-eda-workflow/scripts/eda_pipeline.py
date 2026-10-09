"""Four-step pandas EDA pipeline for weekly dataset auditing."""

import pandas as pd


## Step 1: Load and type the data
def load_and_type(path: str, date_cols: list = None, cat_cols: list = None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame and convert specified columns to appropriate types."""
    df = pd.read_csv(path)

    if date_cols:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if cat_cols:
        for col in cat_cols:
            df[col] = df[col].astype("category")

    return df


## Step 2: Missing value audit
def missing_audit(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Audit missing values in the DataFrame and flag columns exceeding the threshold."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total).round(4)

    audit = pd.DataFrame({"missing_count": missing, "missing_pct": pct, "flag": pct > threshold})

    return audit[audit["missing_count"] > 0].sort_values("missing_pct", ascending=False)


## Step 3: Distribution summary
def distribution_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of the distribution of numeric and categorical columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    summary = {
        "numeric": df[numeric_cols].describe().round(2),
        "categorical": {col: df[col].value_counts().head(10) for col in categorical_cols},
    }
    return summary


## Step 4: Outlier scan using IQR
def outlier_scan(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Scan for outliers in numeric columns using the IQR method and flag columns exceeding the threshold."""
    numeric_cols = df.select_dtypes(include="number").columns
    results = []

    for col in numeric_cols:
        series = df[col].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        pct = len(outliers) / len(series)

        results.append({"column": col, "outlier_count": len(outliers), "outlier_pct": round(pct, 4), "flag": pct > threshold})

    return pd.DataFrame(results).sort_values("outlier_pct", ascending=False)


## Run the full pipeline
def run_eda(path: str, date_cols: list = None, cat_cols: list = None) -> None:
    """Run the full EDA pipeline on the dataset at the specified path."""
    df = load_and_type(path, date_cols, cat_cols)

    print("\n== DATASET OVERVIEW ==")
    print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(df.dtypes.to_string())

    print("\n== MISSING VALUE AUDIT ==")
    print(missing_audit(df).to_string())

    print("\n== DISTRIBUTION SUMMARY ==")
    summary = distribution_summary(df)
    print(summary["numeric"].to_string())
    for col, counts in summary["categorical"].items():
        print(f"\n{col}:\n{counts.to_string()}")

    print("\n== OUTLIER SCAN ==")
    print(outlier_scan(df).to_string())
