"""
Slight modification of the qlrtest python package (https://github.com/JacobSKN/pyqlrtest). Extension with
a normal Chow test for 1D and VAR-like regressions.

@author: Wouter Vermeulen
@date: 2026-01-06
"""

from qlrtest._exceptions import InvalidTrimmingError, QLRTestError
from qlrtest.critical_values import get_qlr_pvalue
from qlrtest.utils import _calculate_rss

import numpy as np
import pandas as pd
import statsmodels.api as sm


def qlr_test(df,  target_col, cols, lags: int = 1, trim=None, start_index = None, stop_index = None, sig_level=0.05, return_f_stats_series=False):
    """
    Performs the Quandt-Likelihood Ratio (QLR) test for structural breaks.

    Args:
        y (array-like): Dependent variable, time series.
        X (array-like): Independent variable(s), including an intercept if desired.
        trim (float, optional): Trimming percentage. Defaults to 0.15.
        sig_level (float, optional): Significance level for 'significant' flag. Defaults to 0.05.
        return_f_stats_series (bool, optional): If True, returns F-stats series. Defaults to False.

    Returns:
        dict: A dictionary containing the QLR test results:
            - 'max_f_stat' (float): The maximum F-statistic found.
            - 'breakpoint' (int): The index of the estimated breakpoint.
            - 'p_value' (float): The asymptotic p-value for the max_f_stat (from Hansen).
            - 'significant' (bool): True if p_value < sig_level, False otherwise.
            - 'n_observations' (int): Total number of observations.
            - 'n_parameters' (int): Number of parameters in the model (k).
            - 'trim_used' (float): The trimming percentage used.
            - 'tested_indices' (np.ndarray, optional): If return_f_stats_series is True.
            - 'f_stats' (np.ndarray, optional): If return_f_stats_series is True.
    """
    # 1. Build lagged regressors
    df_lagged = df.copy()
    for col in cols:
        for lag in range(1, lags + 1):
            df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 2. Drop missing rows
    df_lagged = df_lagged.dropna()

    # 3. Extract y and X
    y_arr = df_lagged[target_col].to_numpy()
    X_arr = df_lagged[[f"{col}_lag{lag}" for col in cols for lag in range(1, lags + 1)]].to_numpy()


    if y_arr.ndim > 1 and y_arr.shape[1] != 1:
        y_arr = y_arr.squeeze()
    if y_arr.ndim > 1:
        raise ValueError("y must be a 1-dimensional array or a 2D array with one column.")

    if X_arr.ndim == 1:
        X_arr = X_arr[:, np.newaxis]

    n_obs = len(y_arr)
    if X_arr.shape[0] != n_obs:
        raise ValueError("y and X must have the same number of observations.")

    if trim is not None:
        if (not 0.01 <= trim < 0.5) and (start_index is None or stop_index is None):
            raise InvalidTrimmingError("Trimming 'trim' must be between 0.01 and 0.49.")
    else:
        trim = (stop_index - start_index)// (n_obs*2) if (start_index is not None and stop_index is not None) else 0.15

    k_params = X_arr.shape[1]

    start_index = int(np.floor(n_obs * trim)) if start_index is None else start_index
    end_index = int(np.ceil(n_obs * (1 - trim))) if stop_index is None else stop_index

    min_obs_per_segment = k_params + 1
    if start_index < min_obs_per_segment:
        start_index = min_obs_per_segment
    if end_index > n_obs - min_obs_per_segment:
        end_index = n_obs - min_obs_per_segment

    if start_index >= end_index:
        raise QLRTestError(
            f"Not enough data points to test for breaks with k={k_params} "
            f"parameters and trim={trim}. "
            f"Effective range [{start_index}, {end_index - 1}] is empty or too small."
        )

    f_stats_list = []
    tested_indices_list = []

    try:
        full_model = sm.OLS(y_arr, X_arr).fit()
        rss_full = full_model.ssr
    except Exception as e:
        raise QLRTestError(f"Failed to fit full model: {e}")

    i = 0
    n = end_index - start_index
    for bp in range(start_index, end_index):
        if i % 100 == 0 and i > 0:
            print(f"QLR Test Progress: {i}/{n} breakpoints tested...")
        i += 1
        y1, X1 = y_arr[:bp], X_arr[:bp, :]
        y2, X2 = y_arr[bp:], X_arr[bp:, :]

        if len(y1) < k_params or len(y2) < k_params:
            continue

        try:
            rss1 = _calculate_rss(y1, X1)
            rss2 = _calculate_rss(y2, X2)
        except np.linalg.LinAlgError:
            continue

        rss_unrestricted = rss1 + rss2

        numerator = (rss_full - rss_unrestricted) / k_params
        denominator = rss_unrestricted / (n_obs - 2 * k_params)

        if denominator <= 1e-9:  # Avoid division by zero or very small positive
            f_stat = np.inf if numerator > 0 else 0.0
        else:
            f_stat = numerator / denominator

        if f_stat < 0 and abs(f_stat) < 1e-9:  # Handle numerical precision for very small negative F-stats
            f_stat = 0.0
        elif f_stat < 0:  # Should not happen if RSS_full >= RSS_unrestricted
            # This might indicate an issue, but we'll cap at 0 for robustness.
            # print(f"Warning: Negative F-stat {f_stat} at bp {bp}. RSS_full={rss_full}, RSS_UR={rss_unrestricted}")
            f_stat = 0.0

        f_stats_list.append(f_stat)
        tested_indices_list.append(bp)

    if not f_stats_list:
        raise QLRTestError("Could not compute F-statistics for any breakpoint. "
                           "Check data and trimming parameter.")

    f_stats_np = np.array(f_stats_list)
    max_f_stat = np.max(f_stats_np)

    breakpoint_index_in_tested = np.argmax(f_stats_np)
    estimated_breakpoint = tested_indices_list[breakpoint_index_in_tested]

    # Calculate p-value using Hansen's approximation
    # The effective trim used for p-value calculation in your original script was `start_index / n_obs`
    # This matches how Hansen's tables are often indexed by the actual start of the restricted period.
    # Using the input `trim` parameter for get_qlr_pvalue is also common if tables are for symmetric trimming.
    # Let's ensure `get_qlr_pvalue` expects the nominal `trim` proportion.
    # Your `_pv_sup` in the original script used `pi = start_index / n` for `l`.
    # Let's stick to user-provided `trim` for `get_qlr_pvalue` for simplicity,
    # assuming `get_qlr_pvalue` and `_get_hansen_coeffs` correctly map this `trim` to the table rows.
    # If your Hansen tables are indexed by asymmetric start (like pi_0), then `start_index / n_obs` might be more appropriate for p-value lookup.
    # For now, using the input `trim`.
    p_value = get_qlr_pvalue(max_f_stat, k_params, trim)

    significant = p_value < sig_level if not np.isnan(p_value) else False  # Default to False if p_value is NaN

    results = {
        'max_f_stat': max_f_stat,
        'breakpoint': estimated_breakpoint,
        'p_value': p_value,
        'significant': significant,
        'n_observations': n_obs,
        'n_parameters': k_params,
        'trim_used': trim,
    }

    if return_f_stats_series:
        results['tested_indices'] = np.array(tested_indices_list)
        results['f_stats'] = f_stats_np

    return results

#####################
# Chow test functions (not part of qlrtest package)
#####################

def make_lagged(series, nlags):
    """
    Create a lagged dataframe for a univariate time series.

    :param series: a 1D array-like
    :param nlags: number of lags
    :return: a dataframe with the original series and its lags
    """
    df = pd.DataFrame({"y": series})
    for L in range(1, nlags + 1):
        df[f"y_lag{L}"] = df["y"].shift(L)
    return df.dropna()


def chow_test_1D(series, break_index, nlags=6):
    """
    Standard Chow test for a univariate time series with lags.

    :param series: a 1D array-like
    :param break_index: index of the break (integer position)
    :param nlags: number of lags to include in the AR (OLS) model
    :return: chow statistic (F-value), model before break, model after break
    """
    df_lag = make_lagged(series, nlags)
    y = df_lag["y"].values
    X = df_lag[[c for c in df_lag.columns if c != "y"]].values
    X = sm.add_constant(X)

    # adjust break index because of dropped lags
    b = break_index - nlags
    X1, X2 = X[:b], X[b:]
    y1, y2 = y[:b], y[b:]

    m_full = sm.OLS(y, X).fit()
    m1 = sm.OLS(y1, X1).fit()
    m2 = sm.OLS(y2, X2).fit()

    RSS_full = np.sum(m_full.resid**2)
    RSS_1 = np.sum(m1.resid**2)
    RSS_2 = np.sum(m2.resid**2)

    k = X.shape[1]
    n1, n2 = len(y1), len(y2)

    chow = ((RSS_full - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n1 + n2 - 2*k))
    return chow, m1, m2


def make_lagged_df(df, y_col, x_cols, nlags):
    """
    Create a lagged dataframe for multiple time series.

    :param df: original dataframe
    :param y_col: dependent variable (string)
    :param x_cols: list of regressors (strings)
    :param nlags: number of lags
    :return: dataframe with y_col and lagged versions of x_cols
    """
    df_lag = pd.DataFrame()
    df_lag[y_col] = df[y_col]

    # Add lags of y
    for L in range(1, nlags + 1):
        df_lag[f"{y_col}_lag{L}"] = df[y_col].shift(L)

    # Add contemporaneous and lagged x's
    for x in x_cols:
        df_lag[x] = df[x]  # contemporaneous
        for L in range(1, nlags + 1):
            df_lag[f"{x}_lag{L}"] = df[x].shift(L)

    return df_lag.dropna()


def chow_test_VAR(df, y_col, x_cols, break_index, nlags=1):
    """
    Chow test for a VAR-like regression with lags.

    :param df: original dataframe
    :param y_col: dependent variable (string)
    :param x_cols: list of regressors (strings)
    :param break_index: index of the break (integer position)
    :param nlags: number of lags to include in the model
    :return: chow statistic (F-value), model before break, model after break
    """

    # Build lagged dataset (same as fit_model)
    df_lag = make_lagged_df(df, y_col, x_cols, nlags)

    # y as Series
    y = df_lag[y_col]

    # X as DataFrame (keep names!)
    X = df_lag.drop(columns=[y_col])
    X = sm.add_constant(X)

    # Adjust break index for lag loss
    b = break_index - nlags
    if b <= 0 or b >= len(df_lag):
        raise ValueError("Break index invalid after lag adjustment")

    # Split samples
    X1, X2 = X.iloc[:b], X.iloc[b:]
    y1, y2 = y.iloc[:b], y.iloc[b:]

    # Fit models
    m_full = sm.OLS(y, X).fit()
    m1 = sm.OLS(y1, X1).fit()
    m2 = sm.OLS(y2, X2).fit()

    # Residual sums of squares
    RSS_full = np.sum(m_full.resid**2)
    RSS_1 = np.sum(m1.resid**2)
    RSS_2 = np.sum(m2.resid**2)

    # Degrees of freedom
    k = X.shape[1]          # number of parameters
    n1, n2 = len(y1), len(y2)

    # Chow F-statistic
    chow = ((RSS_full - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n1 + n2 - 2*k))

    from scipy.stats import f

    alpha = 0.05
    df1 = k
    df2 = n1 + n2 - 2 * k

    F_crit = f.ppf(1 - alpha, df1, df2)

    return chow, m1, m2, F_crit
