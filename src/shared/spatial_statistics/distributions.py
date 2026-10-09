"""Univariate distribution comparisons, normality diagnostics, and transformations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.spatial.distance import minkowski
from scipy.stats import (
    anderson,
    boxcox,
    chisquare,
    jarque_bera,
    ks_2samp,
    kurtosis,
    mannwhitneyu,
    normaltest,
    shapiro,
    skew,
    wilcoxon,
)


def qq_plot_with_distribution(values, attribute_column, exclude_zeros=True, plot=True, figsize=(20, 8)):
    """Create a Q-Q plot with distribution analysis for normality assessment.

    Parameters
    ----------
    values : pandas.DataFrame or list
        The input DataFrame or list of values.
    attribute_column : str
        The name of the attribute column to analyze.
    exclude_zeros : bool, optional
        Whether to exclude zero values. Defaults to True.
    plot : bool, optional
        Whether to display the plot. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (20, 8).

    Returns
    -------
    bool
        True if data appears normally distributed, False otherwise.

    """
    # Check if the column is numerical
    if isinstance(values, pd.DataFrame):
        if not np.issubdtype(values[attribute_column].dtype, np.number):
            raise ValueError(f"Error: Column {attribute_column} must be numerical.")
        df = values.copy()
    else:
        if not all(isinstance(x, (int, float)) for x in values):
            raise ValueError("Error: All values must be numerical.")
        df = pd.DataFrame({attribute_column: values})

    df = df.copy()
    # Extract the non-null values
    df = df[df[attribute_column].notna()].copy()
    if exclude_zeros:
        df = df[df[attribute_column] != 0].copy()

     # Shapiro-Wilk test
    _stat, p_value = shapiro(df[attribute_column])
    print(f"The p value is: {p_value}")
    # Legacy behavior reverses the usual Shapiro decision: large p-values ordinarily fail to reject normality.
    # The existing messages and return value are preserved here rather than silently changing callers' results.
    if p_value > 0.05:
        print(f"The {attribute_column} does not follow a normal distribution.")
        is_normal = False
    elif p_value <= 0.05 and p_value > 0:
        print(f"The {attribute_column} follows a normal distribution.")
        is_normal = True
    elif p_value == 0:
        print("Cannot define normal distribution.")
        is_normal = False

    if plot:
        # Calculate the minimum & maximum value of the attribute column
        min_value = df[attribute_column].min()
        max_value = df[attribute_column].max()

        # Create subplots
        _fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=False, sharey=False)

        # Create QQ plot comparing attribute values to a theoretical normal distribution
        # Fit the reference location and scale so the Q-Q diagnostic compares shape rather than raw units alone.
        sm.qqplot(df[attribute_column], line="45", fit=True, ax=ax[0])
        ax[0].plot([min_value, max_value], [min_value, max_value], color="gray", linestyle="--")
        ax[0].set_xlabel("Theoretical Quantiles")
        ax[0].set_ylabel("Sample Quantiles")
        ax[0].set_title(f"QQ Plot of '{attribute_column}'")

        # Overlay a distribution plot (histogram or KDE) on the second subplot
        sns.histplot(df[attribute_column], kde=True, ax=ax[1], color="skyblue", alpha=0.5)
        ax[1].set_xlabel(attribute_column)
        ax[1].set_ylabel("Density")
        ax[1].set_title(f"Distribution of '{attribute_column}'")
        plt.show()

    return is_normal


def plot_two_distributions(values1, values2, compare_col, color1="blue", color2="red", bins=20, figsize=(15, 6)):
    """Plot two distributions on the same histogram for comparison.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    compare_col : str
        The label for the values.
    color1 : str, optional
        The color for the first distribution. Defaults to 'blue'.
    color2 : str, optional
        The color for the second distribution. Defaults to 'red'.
    bins : int, optional
        The number of bins for the histogram. Defaults to 20.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 6).

    Returns
    -------
    None
        Displays the histogram plot.

    """
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=figsize)
    # With an integer bins argument, each histogram chooses its own edges; shared y limits do not align the bins.
    axs[0].hist(values1, bins=bins, alpha=0.5, label=compare_col, color=color1)
    axs[1].hist(values2, bins=bins, alpha=0.5, label=compare_col, color=color2)

    for ax in axs:
        ax.set_xlabel(compare_col)
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.suptitle(f"Compare distributions of {compare_col} between two datasets")
    plt.show()


def compare_distributions_ks(values1, values2, p_value_threshold=0.05):
    """Compare two distributions using the Kolmogorov-Smirnov test.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    sorted_values1 = np.sort(values1)
    sorted_values2 = np.sort(values2)

    # KS compares empirical cumulative distributions, allowing unequal sample lengths.
    ks_statistic, p_value = ks_2samp(sorted_values1, sorted_values2)
    print(f"\nks_statistic: {ks_statistic}")

    # Failure to reject a difference is not evidence that the two populations are equivalent.
    if p_value < p_value_threshold:
        print(f"The distributions are statistically different (p_value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The distributions are statistically similar (p_value: {p_value}>={p_value_threshold}).")


def compare_distributions_minkowski(values1, values2, p_value_threshold=0.05):
    """Compare two distributions using Minkowski distance and Mann-Whitney U test.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the distance and test results.

    """
    sorted_values1 = np.sort(values1)
    sorted_values2 = np.sort(values2)

    # Euclidean distance pairs sorted order statistics, requiring equal lengths and depending on measurement scale.
    distance = minkowski(sorted_values1, sorted_values2, p=2)  # with p=2 for Euclidean distance
    # perform Mann-Whitney U test
    # The U-test supplies a separate rank-based comparison; it is not a significance test for the distance above.
    stat, p_value = mannwhitneyu(sorted_values1, sorted_values2)

    print(f"\nThe Minkowski distance between the two lists is {distance:.2f}.")
    print(f"Mann-Whitney U test: Statistics={stat:.3f}, p={p_value:.3f}")

    if p_value < p_value_threshold:
        print(f"The distributions are statistically different (p_value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The distributions are statistically similar (p_value: {p_value}>={p_value_threshold}).")


def compare_distributions_ttest(values1, values2, p_value_threshold=0.05):
    """Compare two distributions using the two-sample t-test.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    sorted_values1 = np.sort(values1)
    sorted_values2 = np.sort(values2)

    # The default independent-sample t-test assumes equal population variances and compares means, not full shapes.
    t_statistic, p_value = stats.ttest_ind(sorted_values1, sorted_values2)
    print(f"\nT-statistic: {t_statistic}, P-value: {p_value}")

    if p_value < p_value_threshold:
        print(f"The distributions are statistically different (p_value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The distributions are statistically similar (p_value: {p_value}>={p_value_threshold}).")


def compare_distributions_expected(values1, values2, p_value_threshold=0.05):
    """Compare observed distribution against expected using Chi-Square Goodness of Fit test.

    Parameters
    ----------
    values1 : array-like
        The expected values.
    values2 : array-like
        The observed values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    # Independent sorting pairs counts by rank and discards original category alignment.
    # For a category-specific goodness-of-fit interpretation, that alignment is normally essential.
    expected = np.sort(values1)
    observed = np.sort(values2)

    # Perform Chi-Square Goodness of Fit test
    # Inputs must be frequency counts with matching totals and compatible lengths, not arbitrary continuous samples.
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    print("\nResults of the Chi-Square Goodness of Fit test with the expected:")
    print(f"Chi-square statistic: {chi2_stat}, P-value: {p_value}")

    # Decision based on p-value
    if p_value < p_value_threshold:
        print("Reject the null hypothesis: The observed distribution does not fit the expected distribution.")
    else:
        print("Fail to reject the null hypothesis: The observed distribution fits the expected distribution.")


def compare_distributions_after_event(values1, values2, p_value_threshold=0.05):
    """Compare distributions before and after an event using Wilcoxon signed-rank test.

    Parameters
    ----------
    values1 : array-like
        The values before the event.
    values2 : array-like
        The values after the event.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    # Sorting each sample independently replaces observation pairs with rank-matched values.
    # The resulting test does not preserve subject-level before/after pairing.
    before = np.sort(values1)
    after = np.sort(values2)

    # Perform Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(before, after)

    # Print results
    print("\nResults of the Wilcoxon Signed-Rank Test:")
    print(f"Statistic: {statistic}, P-value: {p_value}")

    # Decision based on p-value
    if p_value < p_value_threshold:
        print("Reject the null hypothesis: There is a significant difference between the before and after lists.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference between the before and after lists.")


def compare_two_distribution(values1, values2, compare_col, p_value_threshold=0.05, plot=True, figsize=(20, 8)):
    """Compare two distributions using multiple statistical methods.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    compare_col : str
        The label for the values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.
    plot : bool, optional
        Whether to plot the distributions. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (20, 8).

    Returns
    -------
    None
        Prints statistical comparison results.

    """
    # These tests have different sampling assumptions; the dispatcher neither validates them nor adjusts for multiple tests.
    compare_distributions_ks(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_minkowski(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_ttest(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_expected(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_after_event(values1, values2, p_value_threshold=p_value_threshold)

    # The descriptive NumPy standard deviations and variances use population normalization (ddof=0).
    print(f"\nFirst  list: mean: {round(np.mean(values1),3)}, median: {round(np.median(values1),3)}, sdev: {round(np.std(values1),3)}, var: {round(np.var(values1),3)}")
    print(f"Second list: mean: {round(np.mean(values2),3)}, median: {round(np.median(values2),3)}, sdev: {round(np.std(values2),3)}, var: {round(np.var(values2),3)}")

    if plot:
        plot_two_distributions(values1, values2, compare_col, figsize=figsize)


def check_normality_distribution(values, label, p_value_threshold=0.05, exclude_zeros=False, plot=True, bins=50, color="skyblue", figsize=(20, 8)):
    """Check normality of distribution using multiple statistical tests.

    Parameters
    ----------
    values : array-like
        The set of values to test.
    label : str
        The label for the values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.
    exclude_zeros : bool, optional
        Whether to exclude zero values. Defaults to False.
    plot : bool, optional
        Whether to create diagnostic plots. Defaults to True.
    bins : int, optional
        The number of bins for histograms. Defaults to 50.
    color : str, optional
        The color for histogram. Defaults to 'skyblue'.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (20, 8).

    Returns
    -------
    None
        Prints normality test results.

    """
    # Only NaNs are removed here; sample-size requirements and infinite values remain the caller's responsibility.
    values = [x for x in values if not np.isnan(x)]
    if exclude_zeros:
        values = [x for x in values if x != 0]

    # Perform Shapiro-Wilk test
    stat, p_value = shapiro(values)
    print(f"\nShapiro-Wilk test: Statistics={stat:.3f}, p={p_value:.3f}")
    # Decision based on p-value
    if p_value < p_value_threshold:
        print(f"The {label} does not follow a normal distribution (p-value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The {label} follows a normal distribution (p-value: {p_value}>={p_value_threshold}).")

    # Perform D'Agostino's K^2 test
    # D'Agostino combines skewness and kurtosis evidence and needs at least eight observations.
    stat, p_value = normaltest(values)
    print(f"\nD'Agostino's K^2 test: Statistics={stat:.3f}, p={p_value:.3f}")
    if p_value > p_value_threshold:
        print("The data is likely normally distributed (fail to reject H0)")
    else:
        print("The data is likely not normally distributed (reject H0)")

    stat, p_value = jarque_bera(values)
    print(f"\nJarque-Bera test statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value > p_value_threshold:
        print("The data is likely normally distributed (fail to reject H0)")
    else:
        print("The data is likely not normally distributed (reject H0)")

    # Perform Anderson-Darling test
    result = anderson(values)
    print(f"\nAnderson-Darling test statistic: {result.statistic:.4f}")
    print("Critical values:", result.critical_values)
    print("Significance levels:", result.significance_level)
    # Interpret the result
    # Anderson reports critical values at percentage significance levels rather than one p-value threshold.
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f"At {sl}% significance level, the data is normally distributed (fail to reject H0)")
        else:
            print(f"At {sl}% significance level, the data is not normally distributed (reject H0)")

    # SciPy's default excess kurtosis is zero for a normal distribution; the 0.5 bounds are descriptive heuristics.
    skewness, kurt = skew(values), kurtosis(values)
    print(f"\nSkewness: {skewness:.4f}, Kurtosis: {kurt:.4f}")
    if abs(skewness) < 0.5 and abs(kurt) < 0.5:
        print("The data is approximately normally distributed based on skewness and kurtosis.")
    else:
        print("The data may not be normally distributed based on skewness and kurtosis.")

    _ = qq_plot_with_distribution(values, label, exclude_zeros=exclude_zeros, plot=plot, figsize=figsize)


def check_distribution_of_transformed_values(values, figsize=(15, 5)):
    """Analyze and compare transformed distributions (log and Box-Cox).

    Parameters
    ----------
    values : array-like
        The set of values to transform and analyze.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 5).

    Returns
    -------
    None
        Displays transformation results and prints normality test results.

    """
    # Shift the data to make all values positive
    min_value = np.min(values)
    if min_value <= 0:
        # Move the minimum to one so logarithms and Box-Cox both receive strictly positive values.
        shift = abs(min_value) + 1
        print(f"Negative values are detected: the data will be shifted (shift value: {shift}).")
    else:
        shift = 0

    shifted_values = values + shift
    # Apply log transformation
    log_values = np.log(shifted_values)

    # Apply Box-Cox transformation
    # Box-Cox estimates its power from these same observations; the transformation is not fitted on a separate sample.
    boxcox_values, _lambda_param = boxcox(shifted_values)

    # Compare distributions
    _fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1.hist(values, bins=30, density=True, alpha=0.7)
    ax1.set_title("Original Data")

    ax2.hist(log_values, bins=30, density=True, alpha=0.7)
    ax2.set_title("Log Transformed")

    ax3.hist(boxcox_values, bins=30, density=True, alpha=0.7)
    ax3.set_title("Box-Cox Transformed")

    plt.tight_layout()
    plt.show()

    # Test normality of transformed data
    # These in-sample diagnostics describe the transformed data; they do not validate generalization to new observations.
    _, p_log = shapiro(log_values)
    _, p_boxcox = shapiro(boxcox_values)

    print(f"Log transform Shapiro-Wilk p-value: {p_log:.4f}")
    print(f"Box-Cox transform Shapiro-Wilk p-value: {p_boxcox:.4f}")
