"""CUSUM and penalized change-point diagnostics for ordered observations."""

import pandas as pd


def time_sereis_cusum(data, target, threshold, k):
    """Detect mean deviation in time series using CUSUM algorithm.

    Parameters
    ----------
    data : list
        Time series data.
    target : float
        Target mean value.
    threshold : float
        The threshold for change detection.
    k : float
        The sensitivity parameter.

    Returns
    -------
    list
        Cumulative sum values indicating deviation.

    """
    # Index zero anchors the statistic at zero; accumulation starts at the second observation.
    cumulative_sum = [0]
    for i in range(1, len(data)):
        d_t = data[i] - target
        s_t = cumulative_sum[i-1] + d_t
        # Shrink the signed accumulator toward zero by k, creating a dead band for small deviations.
        if s_t > 0:
            s_t = max(0, s_t - k)
        elif s_t < 0:
            s_t = min(0, s_t + k)
        cumulative_sum.append(s_t)
        # Threshold crossings only print an alert; the accumulator is not reset, so consecutive alerts can occur.
        if abs(s_t) > threshold:
            print("Change detected at time step", i)
    return cumulative_sum


def ruptures_change_point_detection(df, kpi, penalty=7):
    """Detect change points in time series using PELT algorithm.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time series data.
    kpi : str
        The column name to analyze for change points.
    penalty : float, optional
        The penalty parameter for the PELT algorithm. Defaults to 7.

    Returns
    -------
    list
        Indices of detected change points.

    """
    # Load the optional change-point dependency only when this diagnostic is requested.
    import ruptures as rpt
    # Assuming df has Date column
    # This mutates the caller's frame; observation order still comes from existing rows and is not sorted by date.
    df['Date'] = pd.to_datetime(df.index).date

    # Fit the ruptures algorithm
    # The RBF cost detects distributional changes; the penalty trades additional segments against improved fit.
    algo = rpt.Pelt(model="rbf").fit(df[kpi].values)
    result = algo.predict(pen=penalty)

    # Return segment-end positions, including the final sample count, rather than datetime labels.
    return result
