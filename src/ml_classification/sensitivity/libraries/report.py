"""HTML summary of sensitivity experiments."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def write_html_report(output_dir, results_df: pd.DataFrame, plot_path: Path | None) -> Path:
    """Generate an HTML summary report for sensitivity analysis.

    Args:
        results_df: Aggregated sensitivity results dataframe.
        plot_path: Optional plot image path.

    Returns:
        Path to the generated HTML report.
    """
    # Prepare aggregates for successful runs only.
    success_df = results_df.loc[results_df["success"] & results_df["test_accuracy"].notna()].copy()
    run_count = len(results_df)
    success_count = len(success_df)
    failure_count = int(run_count - success_count)
    mean_acc = float(success_df["test_accuracy"].mean()) if not success_df.empty else float("nan")
    std_acc = float(success_df["test_accuracy"].std(ddof=0)) if not success_df.empty else float("nan")

    # Build table rows with escaped content.
    table_rows = []
    for _, row in results_df.sort_values("seed").iterrows():
        accuracy_text = "" if pd.isna(row["test_accuracy"]) else f"{float(row['test_accuracy']):.4f}"
        table_rows.append(
            "<tr>"
            f"<td>{int(row['seed'])}</td>"
            f"<td>{'✅' if bool(row['success']) else '❌'}</td>"
            f"<td>{html.escape(str(row['selected_model'])) if pd.notna(row['selected_model']) else ''}</td>"
            f"<td>{accuracy_text}</td>"
            f"<td>{html.escape(str(row['error'])) if pd.notna(row['error']) else ''}</td>"
            "</tr>"
        )

    img_html = ""
    if plot_path is not None and plot_path.exists():
        # Use a relative image reference so HTML remains portable with the folder.
        img_html = (
            "<h2>Accuracy Plot</h2>"
            f"<img src='{html.escape(plot_path.name)}' alt='Accuracy by seed' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;'>"
        )

    html_text = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>ML Classification Sensitivity Report</title>
  <style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2937; }}
h1, h2 {{ color: #111827; }}
.meta {{ padding: 12px; background: #f3f4f6; border-radius: 8px; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
th {{ background: #f9fafb; }}
.muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>ML Classification Sensitivity Report</h1>
  <p class='muted'>Generated on {datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <div class='meta'>
<b>Total runs:</b> {run_count} &nbsp; | &nbsp;
<b>Successful:</b> {success_count} &nbsp; | &nbsp;
<b>Failed:</b> {failure_count} &nbsp; | &nbsp;
<b>Accuracy mean:</b> {"" if np.isnan(mean_acc) else f"{mean_acc:.4f}"} &nbsp; | &nbsp;
<b>Accuracy std:</b> {"" if np.isnan(std_acc) else f"{std_acc:.4f}"}
  </div>

  {img_html}

  <h2>Per-seed Results</h2>
  <table>
<thead>
  <tr>
    <th>Seed</th><th>Success</th><th>Selected Model</th><th>Test Accuracy</th><th>Error</th>
  </tr>
</thead>
<tbody>
  {"".join(table_rows)}
</tbody>
  </table>
</body>
</html>"""

    report_path = output_dir / "sensitivity_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path
