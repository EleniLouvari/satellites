<!-- examples/example_prompt_1.md -->

## Example: Weekly Sales EDA

**User prompt:**
Run an EDA on weekly_sales.csv. The order_date column is a date.
The channel and product_category columns are categorical.

**Expected response structure:**

1. Load data with load_and_type(), passing date_cols=["order_date"]
   and cat_cols=["channel", "product_category"]
2. Print dataset overview: row count, column count, dtypes
3. Run missing_audit() and flag columns above 10% missing
4. Run distribution_summary() for numeric and categorical columns
5. Run outlier_scan() and flag columns above 5% outlier rate
6. Summarize findings in 2-3 sentences before presenting output
