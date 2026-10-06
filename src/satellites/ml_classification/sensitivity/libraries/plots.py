"""Compare accuracy across sensitivity seeds and export summary statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from satellites.shared.io import write_data


def create_accuracy_plot(output_dir, results_df: pd.DataFrame) -> Path | None:
    """Create and save an accuracy-versus-seed plot.

    Args:
        results_df: Aggregated sensitivity results dataframe.

    Returns:
        Path to the saved plot, or ``None`` when no successful rows exist.
    """
    # Plot only successful runs with non-null accuracy values.
    plot_df = results_df.loc[results_df["success"] & results_df["test_accuracy"].notna(), ["seed", "test_accuracy"]].copy()
    if plot_df.empty:
        # No successful metrics available, so a plot cannot be generated.
        return None

    plot_df.sort_values("seed", inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["seed"], plot_df["test_accuracy"], marker="o", linewidth=2, color="#1f77b4")
    ax.set_title("Sensitivity Analysis: Test Accuracy by Random Seed")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)

    mean_acc = float(plot_df["test_accuracy"].mean())
    std_acc = float(plot_df["test_accuracy"].std(ddof=0))

    # Draw mean, ±1σ, and ±2σ reference lines for quick visual spread checks.
    ax.axhline(mean_acc, linestyle="--", linewidth=1.6, color="#d62728", label=f"mean={mean_acc:.4f}")
    if std_acc > 0:
        ax.axhline(mean_acc + std_acc, linestyle=":", linewidth=1.2, color="#ff7f0e", label="+1σ")
        ax.axhline(mean_acc - std_acc, linestyle=":", linewidth=1.2, color="#ff7f0e", label="-1σ")
        ax.axhline(mean_acc + (2 * std_acc), linestyle="-.", linewidth=1.1, color="#2ca02c", label="+2σ")
        ax.axhline(mean_acc - (2 * std_acc), linestyle="-.", linewidth=1.1, color="#2ca02c", label="-2σ")

    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_dir / "accuracy_by_seed.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    # Also persist basic aggregate statistics.
    stats_df = pd.DataFrame(
        [
            {"metric": "accuracy_mean", "value": mean_acc},
            {"metric": "accuracy_std", "value": std_acc},
            {"metric": "accuracy_plus_1std", "value": mean_acc + std_acc},
            {"metric": "accuracy_minus_1std", "value": mean_acc - std_acc},
            {"metric": "accuracy_plus_2std", "value": mean_acc + (2 * std_acc)},
            {"metric": "accuracy_minus_2std", "value": mean_acc - (2 * std_acc)},
        ]
    )
    write_data(stats_df, str(output_dir / "accuracy_summary_stats.csv"), plain_csv=True)
    return plot_path
