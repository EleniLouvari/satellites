"""
Multiclass Ranking-Based Ensemble Methodology Library

Adapts the binary classification ranking methodology for multiclass problems.
Implements One-vs-Rest (OvR) approach where each class gets its own ranking based on
class-specific probabilities and ranking metrics.

"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_probabilities_for_multiclass(df_features, df_predictions, models, n_classes):
    """
    Calculate multiclass probabilities from all models.

    Parameters:
    -----------
    df_features : pd.DataFrame
        Feature dataframe used for predictions
    df_predictions : pd.DataFrame
        Predictions dataframe to append probability columns to
    models : dict
        Dictionary of trained models {model_name: model_instance}
    n_classes : int
        Number of classes

    Returns:
    --------
    pd.DataFrame
        Updated predictions dataframe with probability columns for each class from each model
    """
    df_predictions = df_predictions.copy()

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            print(f"Calculating probabilities for {model_name}...")
            probs = model.predict_proba(df_features)  # Shape: (N, n_classes)
            model_classes = list(getattr(model, "classes_", range(probs.shape[1])))
            class_to_index = {cls: idx for idx, cls in enumerate(model_classes)}
            for class_idx in range(n_classes):
                col_name = f"{model_name}_prob_class_{class_idx}"
                if class_idx in class_to_index and class_to_index[class_idx] < probs.shape[1]:
                    df_predictions[col_name] = probs[:, class_to_index[class_idx]]
                else:
                    print(
                        f"Warning: {model_name} does not provide probability for class {class_idx}; "
                        "filling with NaN"
                    )
                    df_predictions[col_name] = np.nan
        else:
            print(f"Warning: {model_name} does not have predict_proba method")

    return df_predictions


def create_multiclass_rankings(df_predictions, models, n_classes,
                               rank_col_prefix="rank"):
    """
    Create per-class rankings from multiclass probabilities.

    Parameters:
    -----------
    df_predictions : pd.DataFrame
        Dataframe with probability columns
    models : dict
        Dictionary of models (for reference)
    n_classes : int
        Number of classes
    rank_col_prefix : str
        Prefix for ranking column names

    Returns:
    --------
    pd.DataFrame
        Dataframe with average and median rank columns for each class
    """
    df_ranked = df_predictions.copy()

    for target_class in range(n_classes):
        # Get probability columns for this class from all models
        prob_cols = [col for col in df_ranked.columns if f"_prob_class_{target_class}" in col]
        rank_cols = []

        if len(prob_cols) == 0:
            print(f"Warning: No probability columns found for class {target_class}")
            continue

        if df_ranked[prob_cols].notna().sum().sum() == 0:
            print(f"Warning: Probability columns for class {target_class} contain only missing values")
            continue

        print(f"Creating rankings for class {target_class} ({len(prob_cols)} models)...")

        # Method 1: Average rank across models (remove high and low)
        if len(prob_cols) >= 3:
            ranks = df_ranked[prob_cols].rank(axis=0, ascending=False)
            rank_cols = [f"{col}_rank" for col in prob_cols]
            df_ranked[rank_cols] = ranks

            # Remove highest and lowest rank
            min_rank = ranks.min(axis=1)
            max_rank = ranks.max(axis=1)
            sum_rank = ranks.sum(axis=1)
            df_ranked[f'{rank_col_prefix}_avg_class_{target_class}'] = \
                (sum_rank - min_rank - max_rank) / (len(prob_cols) - 2)
        else:
            ranks = df_ranked[prob_cols].rank(axis=0, ascending=False)
            df_ranked[f'{rank_col_prefix}_avg_class_{target_class}'] = ranks.mean(axis=1)

        # Method 2: Median rank across models
        df_ranked[f'{rank_col_prefix}_median_class_{target_class}'] = ranks.median(axis=1)

        # Method 3: Max probability (confidence-based ranking)
        df_ranked[f'{rank_col_prefix}_maxprob_class_{target_class}'] = \
            df_ranked[prob_cols].max(axis=1)

        # Clean up temporary rank columns
        if rank_cols:
            df_ranked.drop(columns=rank_cols, inplace=True)

    return df_ranked


def calculate_top_accuracy_multiclass(df, target_class, col_target, top_perc_dict,
                                      metrics=None):
    """
    Calculate top-k accuracy for a specific class.

    Parameters:
    -----------
    df : pd.DataFrame
        Ranked dataframe sorted by ranking position
    target_class : int
        The class to evaluate
    col_target : str
        Name of target column
    top_perc_dict : dict
        Dictionary with percentages and weights: {name: (percentage, weight)}
    metrics : dict, optional
        Dictionary to append metrics to

    Returns:
    --------
    dict
        Updated metrics dictionary
    """
    if metrics is None:
        metrics = {}

    total_records = len(df)
    # Count records of target class
    total_target = len(df[df[col_target] == target_class])

    if total_target == 0:
        print(f"Warning: No records found for class {target_class}")
        return metrics

    total_score = 0
    for col_name, (percentage, weight) in top_perc_dict.items():
        num_records = int(percentage * total_records / 100)
        df_top = df.head(num_records).copy()

        top_target_records = len(df_top[df_top[col_target] == target_class])
        top_records = len(df_top)

        metrics[f"{col_name}_count"] = top_target_records
        metrics[f"{col_name}_perc"] = 100 * top_target_records / total_target if total_target > 0 else 0
        metrics[f"{col_name}_acc"] = 100 * top_target_records / top_records if top_records > 0 else 0

        total_score += metrics[f"{col_name}_perc"] * weight

    metrics["top_weighted_score"] = total_score
    return metrics


def calculate_median_accuracy_multiclass(df, target_class, col_target, col_rank):
    """
    Calculate median rank accuracy for a specific class (OvR).

    Parameters:
    -----------
    df : pd.DataFrame
        Ranked dataframe sorted by ranking position
    target_class : int
        The class to evaluate
    col_target : str
        Name of target column
    col_rank : str
        Name of rank column

    Returns:
    --------
    float
        Normalized median accuracy score (0-100)
    """
    # Filter to records of target class and others (binary)
    target_df = df[(df[col_target] == target_class)].copy()

    if len(target_df) == 0:
        return 0.0

    # Create ideal dataframe (all target class records ranked at top)
    ideal_df = target_df.sort_values(by=col_target, ascending=False).copy()
    ideal_df[f"{col_rank}_ideal"] = list(range(1, len(target_df) + 1))

    scores = []
    # Actual scores
    if col_rank in target_df.columns:
        median_pos = target_df[col_rank].median()
        median_score = 1 - (median_pos / len(df))
        scores.append(median_score)

    # Ideal scores
    median_pos_ideal = ideal_df[f"{col_rank}_ideal"].median()
    median_score_ideal = 1 - (median_pos_ideal / len(df))
    scores.append(median_score_ideal)

    # Normalized median score
    norm_median_score = 100.0 * scores[0] / scores[1] if scores[1] > 0 else 0.0
    return norm_median_score


def calculate_power_accuracy_multiclass(df, target_class, col_target, col_rank):
    """
    Calculate power accuracy (exponential penalty) for a specific class.

    Parameters:
    -----------
    df : pd.DataFrame
        Ranked dataframe sorted by ranking position
    target_class : int
        The class to evaluate
    col_target : str
        Name of target column
    col_rank : str
        Name of rank column

    Returns:
    --------
    float
        Normalized power accuracy score (0-100)
    """
    df = df.copy()
    number_of_records = len(df)

    # Actual power score
    df['pos_weight'] = (100 - 100.0 * (df[col_rank] - 1) / number_of_records) ** 3 / 10000
    power_score = df.loc[df[col_target] == target_class, 'pos_weight'].mean()

    # Ideal power score
    ideal_df = df[df[col_target] == target_class].copy()
    if len(ideal_df) == 0:
        return 0.0

    ideal_df['pos_weight_ideal'] = (100 - 100.0 * (np.arange(len(ideal_df))) / number_of_records) ** 3 / 10000
    power_score_ideal = ideal_df['pos_weight_ideal'].mean()

    # Normalized power score
    norm_power_score = 100.0 * power_score / power_score_ideal if power_score_ideal > 0 else 0.0
    return norm_power_score


def calc_dcg(relevance_scores):
    """
    Calculate Discounted Cumulative Gain (DCG).

    Parameters:
    -----------
    relevance_scores : array-like
        Relevance scores (binary: 0 or 1 for class membership)

    Returns:
    --------
    float
        DCG value
    """
    if len(relevance_scores) == 0:
        return 0.0

    positions = list(range(2, len(relevance_scores) + 1))
    scores = relevance_scores[1:]

    dcg = relevance_scores[0]
    for pos, reli in zip(positions, scores):
        dcg += reli / math.log2(pos)
    return dcg


def calculate_ndcg_accuracy_multiclass(df, target_class, col_target, col_rank,
                                       top_rank_perc=0.2):
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG) for a specific class.

    Parameters:
    -----------
    df : pd.DataFrame
        Ranked dataframe sorted by ranking position
    target_class : int
        The class to evaluate
    col_target : str
        Name of target column
    col_rank : str
        Name of rank column
    top_rank_perc : float
        Percentage of top ranks to consider (default 0.2 = 20%)

    Returns:
    --------
    float
        Normalized nDCG score (0-100)
    """
    df = df.copy()
    df = df.sort_values(by=col_rank)
    number_of_records = len(df)
    k = int(top_rank_perc * number_of_records)
    k = max(k, 1)  # At least 1

    # Binary relevance: 1 if target class, 0 otherwise
    df['is_target'] = (df[col_target] == target_class).astype(int)

    # Get actual scores
    actual_scores = df['is_target'].values[:k]

    # Get ideal scores (all 1s for target class records)
    ideal_scores = np.ones(k)

    scores = []
    scores.append(calc_dcg(actual_scores))
    scores.append(calc_dcg(ideal_scores))

    # Calculate nDCG
    norm_dcg = 100.0 * scores[0] / scores[1] if scores[1] > 0 else 0.0
    return norm_dcg


def calculate_multiclass_ranking_metrics(df_ranked, col_target, col_rank, n_classes,
                                        top_perc_dict=None, top_rank_perc=0.2):
    """
    Calculate all ranking metrics for each class.

    Parameters:
    -----------
    df_ranked : pd.DataFrame
        Ranked dataframe
    col_target : str
        Name of target column
    col_rank : str
        Name of rank column
    n_classes : int
        Number of classes
    top_perc_dict : dict, optional
        Top-k percentages and weights
    top_rank_perc : float
        Percentage for nDCG (default 0.2)

    Returns:
    --------
    pd.DataFrame
        Metrics for each class
    """
    if top_perc_dict is None:
        top_perc_dict = {'top_1': (1, 1.0), 'top_5': (5, 0.8), 'top_10': (10, 0.6)}

    results = []

    for target_class in range(n_classes):
        df_class = df_ranked.copy()

        # Apply class-specific rank column if it exists
        class_rank_col = f"{col_rank}_class_{target_class}"
        if class_rank_col not in df_class.columns:
            class_rank_col = col_rank

        if class_rank_col not in df_class.columns:
            print(f"Warning: No rank column available for class {target_class}")
            continue

        if df_class[class_rank_col].notna().sum() == 0:
            print(f"Warning: Rank column for class {target_class} contains only missing values")
            continue

        df_class = df_class.sort_values(by=class_rank_col)

        metrics = {'class': target_class}

        # Top accuracy
        metrics = calculate_top_accuracy_multiclass(df_class, target_class, col_target, top_perc_dict, metrics)

        # Median accuracy
        metrics['median'] = calculate_median_accuracy_multiclass(df_class, target_class, col_target, class_rank_col)

        # Power accuracy
        metrics['power'] = calculate_power_accuracy_multiclass(df_class, target_class, col_target, class_rank_col)

        # nDCG accuracy
        metrics['ndcg'] = calculate_ndcg_accuracy_multiclass(df_class, target_class, col_target, class_rank_col, top_rank_perc)

        results.append(metrics)

    return pd.DataFrame(results)


def plot_multiclass_ranking_metrics(metrics_df, figsize=(14, 6)):
    """
    Plot multiclass ranking metrics for comparison.

    Parameters:
    -----------
    metrics_df : pd.DataFrame
        Metrics dataframe from calculate_multiclass_ranking_metrics
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Some callers may provide metrics without top_weighted_score.
    # Use a safe fallback so plotting never fails with KeyError.
    if 'top_weighted_score' not in metrics_df.columns:
        top_weighted_values = metrics_df[[c for c in metrics_df.columns if c.startswith('top_') and \
            c.endswith('_perc')]].sum(axis=1) if any(c.startswith('top_') and c.endswith('_perc') \
                for c in metrics_df.columns) else pd.Series([0] * len(metrics_df), index=metrics_df.index)
    else:
        top_weighted_values = metrics_df['top_weighted_score']

    # Plot 1: Median accuracy per class
    ax = axes[0, 0]
    ax.bar(metrics_df['class'], metrics_df['median'], color='skyblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Median Accuracy (%)')
    ax.set_title('Median Rank Accuracy per Class')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Power accuracy per class
    ax = axes[0, 1]
    ax.bar(metrics_df['class'], metrics_df['power'], color='lightcoral')
    ax.set_xlabel('Class')
    ax.set_ylabel('Power Accuracy (%)')
    ax.set_title('Power Accuracy per Class')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: nDCG accuracy per class
    ax = axes[1, 0]
    ax.bar(metrics_df['class'], metrics_df['ndcg'], color='lightgreen')
    ax.set_xlabel('Class')
    ax.set_ylabel('nDCG Accuracy (%)')
    ax.set_title('nDCG Accuracy per Class')
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Top weighted score per class
    ax = axes[1, 1]
    ax.bar(metrics_df['class'], top_weighted_values, color='gold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Top Weighted Score (%)')
    ax.set_title('Top-K Weighted Score per Class')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def create_multiclass_ranking_report(df_predictions, models, df_test, col_target,
                                    col_id, n_classes, top_perc_dict=None,
                                    top_rank_perc=0.2, plot_graphs=True):
    """
    Create a comprehensive multiclass ranking report.

    Parameters:
    -----------
    df_predictions : pd.DataFrame
        Predictions dataframe with probabilities
    models : dict
        Dictionary of trained models
    df_test : pd.DataFrame
        Test dataset with target labels
    col_target : str
        Name of target column
    col_id : str
        Name of ID column
    n_classes : int
        Number of classes
    top_perc_dict : dict, optional
        Top-k percentages and weights
    top_rank_perc : float
        Percentage for nDCG
    plot_graphs : bool
        Whether to generate plots

    Returns:
    --------
    dict
        Report containing metrics, rankings, and visualizations
    """
    if top_perc_dict is None:
        top_perc_dict = {'top_1': (1, 1.0), 'top_5': (5, 0.8), 'top_10': (10, 0.6)}

    report = {}

    # 1. Calculate probabilities for all models
    print("Step 1: Calculating probabilities...")
    df_probs = calculate_probabilities_for_multiclass(
        df_predictions.select_dtypes(include=[np.number]),
        df_predictions[[col_id, col_target]].copy(),
        models,
        n_classes
    )

    # 2. Create rankings for each class
    print("Step 2: Creating per-class rankings...")
    df_ranked = create_multiclass_rankings(df_probs, models, n_classes)

    # 3. Calculate metrics for each class
    print("Step 3: Calculating ranking metrics...")
    rank_col_prefix = "rank"
    rank_cols = [col for col in df_ranked.columns if rank_col_prefix in col]

    if len(rank_cols) > 0:
        rank_col_example = rank_cols[0]
        metrics_df = calculate_multiclass_ranking_metrics(
            df_ranked, col_target, rank_col_example, n_classes,
            top_perc_dict, top_rank_perc
        )
        report['metrics'] = metrics_df
    else:
        print("Warning: No rank columns found")
        report['metrics'] = None

    # 4. Generate plots
    if plot_graphs and report['metrics'] is not None:
        print("Step 4: Generating plots...")
        fig = plot_multiclass_ranking_metrics(report['metrics'])
        report['figure'] = fig

    # 5. Summary statistics
    report['n_classes'] = n_classes
    report['n_samples'] = len(df_ranked)
    report['class_distribution'] = df_ranked[col_target].value_counts().to_dict()
    report['models_used'] = list(models.keys())

    return report


def print_multiclass_ranking_report(report):
    """
    Print a formatted multiclass ranking report.

    Parameters:
    -----------
    report : dict
        Report dictionary from create_multiclass_ranking_report
    """
    print("\n" + "="*80)
    print("MULTICLASS RANKING-BASED ENSEMBLE REPORT")
    print("="*80)

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {report['n_samples']}")
    print(f"  - Number of classes: {report['n_classes']}")
    print(f"  - Models used: {', '.join(report['models_used'])}")

    print(f"\nClass Distribution:")
    for class_id, count in sorted(report['class_distribution'].items()):
        print(f"  - Class {class_id}: {count} samples")

    if report['metrics'] is not None:
        print(f"\nRanking Metrics Summary:")
        print("-" * 80)
        display_columns = ["class", "median", "power", "ndcg"]
        if "top_weighted_score" in report["metrics"].columns:
            display_columns.append('top_weighted_score')
        metrics_display = report['metrics'][display_columns].copy()
        rename_map = {
            'class': 'Class',
            'median': 'Median Acc (%)',
            'power': 'Power Acc (%)',
            'ndcg': 'nDCG (%)',
            'top_weighted_score': 'Top-K Weighted (%)',
        }
        metrics_display = metrics_display.rename(columns=rename_map)
        print(metrics_display.to_string(index=False))

    print("\n" + "="*80)
