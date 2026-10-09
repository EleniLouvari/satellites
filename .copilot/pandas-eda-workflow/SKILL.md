---
name: pandas-eda-workflow
description: >
  Use this skill when the user asks to explore, profile, summarize, or
  audit a dataset using Python and pandas. Triggers include: EDA,
  exploratory data analysis, data profile, missing values, column types,
  distribution summary, cardinality check, outlier scan, and data audit.
  Also triggers when the user uploads a CSV or DataFrame and asks
  what to do first.
---

# Pandas EDA Workflow Skill

## Purpose

This skill packages a four-step exploratory data analysis pipeline using
pandas. It encodes column typing conventions, missing value thresholds,
distribution summary methods, and outlier detection logic so the analyst
does not need to re-explain the workflow for each new dataset.

## EDA Steps

1. **Load and type**: Read the CSV, infer types, coerce dates and categoricals
2. **Missing value audit**: Count and percentage by column, flag columns above threshold
3. **Distribution summary**: Descriptive stats for numeric columns, value counts for categoricals
4. **Outlier scan**: IQR-based flagging for numeric columns

## Output Conventions

- Print section headers using `print("\\n== SECTION ==")` for readability
- Flag columns where missing values exceed 10% of rows
- Flag numeric columns where outlier count exceeds 5% of non-null rows
- Round all float output to 2 decimal places
