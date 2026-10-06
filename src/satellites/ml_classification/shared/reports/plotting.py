"""Common figure styling and deterministic Matplotlib file export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from satellites.ml_classification.shared.persistence import ensure_dir

sns.set_theme(style="whitegrid")


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Save and close a matplotlib figure to the requested output path."""
    # Standardize layout and export settings across plotting helpers.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
