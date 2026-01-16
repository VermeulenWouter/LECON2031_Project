"""
Analysis of structural breaks in offshore wind farm capacity factors considering wind direction.

@author Wouter Vermeulen
@date 2025-01-14
"""

import os.path as path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from Helper import load_timeseries_data, qlrtest
from Helper.forecasting import forecast_model
from Helper.qlrtest import chow_test_VAR, chow_test_1D, make_lagged_df
from ModelNoWindDirection import subset_farms_from_date


def plot_capacity_factor_by_direction(df, bin_size=10, filepath=None):
    """
    Compute and plot mean capacity factor by wind-direction bins for two independent time windows,
    with Welch CI for differences.

    :param df: DataFrame with columns 'AverageWindDirection', 'AverageWindSpeed_sadjusted', 'Belwind_sadjusted', 'Thorntonbank_NE_sadjusted'
    :param bin_size: Size of wind direction bins in degrees
    :param filepath: If provided, save the plot to this path; otherwise, show the plot.
    :return: Dictionary with mean capacity factors and differences with confidence intervals.
    """

    before = df.loc[: "2020-04-27 23:00:00"]
    after = df.loc["2020-12-01 00:00:00":]

    bins = np.arange(0, 360 + bin_size, bin_size)
    labels = bins[:-1] + bin_size / 2

    def compute_stats(subset):
        tmp = subset.copy()
        tmp["dir_bin"] = pd.cut(tmp["AverageWindDirection"], bins=bins, labels=labels, right=False)

        grouped = tmp.groupby("dir_bin", observed=False)

        mean = grouped[["Belwind_sadjusted", "Thorntonbank_NE_sadjusted"]].mean()
        std = grouped[["Belwind_sadjusted", "Thorntonbank_NE_sadjusted"]].std()
        n = grouped[["Belwind_sadjusted", "Thorntonbank_NE_sadjusted"]].count()

        sem = std / np.sqrt(n)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem

        return mean, ci_low, ci_high, sem, n

    before = before.copy()
    after = after.copy()
    before["dir_bin"] = pd.cut(before["AverageWindDirection"], bins=bins, labels=labels, right=False)
    after["dir_bin"] = pd.cut(after["AverageWindDirection"], bins=bins, labels=labels, right=False)

    before_mean, before_low, before_high, before_sem, before_n = compute_stats(before)
    after_mean, after_low, after_high, after_sem, after_n = compute_stats(after)
    before_ws = before.groupby("dir_bin", observed=False)["AverageWindSpeed_sadjusted"].mean()
    after_ws = after.groupby("dir_bin", observed=False)["AverageWindSpeed_sadjusted"].mean()
    diff_ws = after_ws - before_ws

    # Confidence intervals
    diff_mean = after_mean - before_mean
    diff_sem = np.sqrt(after_sem ** 2 + before_sem ** 2)
    diff_low = diff_mean - 1.96 * diff_sem
    diff_high = diff_mean + 1.96 * diff_sem

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # 1. Belwind
    ax = axes[0]
    ax.plot(labels, before_mean["Belwind_sadjusted"], label="Before", marker="o")
    ax.plot(labels, after_mean["Belwind_sadjusted"], label="After", marker="o")
    ax2 = ax.twinx()
    ax2.plot(labels, before_ws, color="gray", linestyle="--", label="Wind speed before")
    ax2.plot(labels, after_ws, color="black", linestyle="--", label="Wind speed after")
    ax2.set_ylabel("Wind speed (m/s)")

    ax.set_title(f"Belwind Capacity Factor (bin size {bin_size}°)")
    ax.set_ylabel("Capacity Factor")
    ax.legend()
    ax2.legend(loc="lower right")

    # 2. Thorntonbank
    ax = axes[1]
    ax.plot(labels, before_mean["Thorntonbank_NE_sadjusted"], label="Before", marker="o")
    ax.plot(labels, after_mean["Thorntonbank_NE_sadjusted"], label="After", marker="o")
    ax2 = ax.twinx()
    ax2.plot(labels, before_ws, color="gray", linestyle="--", label="Wind speed before")
    ax2.plot(labels, after_ws, color="black", linestyle="--", label="Wind speed after")
    ax2.set_ylabel("Wind speed (m/s)")

    ax.set_title("Thorntonbank Capacity Factor")
    ax.set_ylabel("Capacity Factor")
    ax.legend()
    ax2.legend(loc="lower right")

    # 3. Difference with CI
    ax = axes[2]
    for col, label in [("Belwind_sadjusted", "Belwind"), ("Thorntonbank_NE_sadjusted", "Thorntonbank")]:
        ax.plot(labels, diff_mean[col], label=f"{label} diff", marker="o")
        ax.fill_between(labels, diff_low[col], diff_high[col], alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(labels, diff_ws, color="black", linestyle="--", label="Wind speed diff")
    ax2.set_ylabel("Δ Wind speed (m/s)")

    ax.set_xlabel("Wind Direction Bin (°)")
    ax.set_ylabel("Δ Capacity Factor")
    ax.legend()
    ax2.legend(loc="lower right")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()
    plt.close(fig)

    return {"before_mean": before_mean, "after_mean": after_mean, "diff_mean": diff_mean, "diff_ci_low": diff_low, "diff_ci_high": diff_high}


def drop_short_segments(df, lag, freq="1h", n=3):
    """
    Remove contiguous segments of hourly data shorter than n*lag.

    :param df: DataFrame with DateTimeIndex
    :param lag: Lag order used in the model
    :param freq: Expected frequency of the data (e.g., "1h" for hourly)
    :param n: Minimum number of lags required in a segment (i.e. minimum length is n*lag)
    :return: DataFrame with short segments removed
    """
    # Compute time differences
    diffs = df.index.to_series().diff()

    # Keep only sufficiently long segments
    segment_id = (diffs > pd.Timedelta(freq)).cumsum()
    groups = df.groupby(segment_id)
    min_len = n * lag
    kept_segments = [g for _, g in groups if len(g) >= min_len]

    if kept_segments: # Concatenate back into a single dataframe
        return pd.concat(kept_segments).sort_index()
    else:
        # If everything is too short, return empty df
        return df.iloc[0:0]


def fit_model(df, y_col, x_cols, nlags=6):
    """
    Fit ARX-like model with nlags for y and each x in x_cols.
    Returns fitted model and list of regressor names.

    :param df: DataFrame with data
    :param y_col: Name of the dependent variable column
    :param x_cols: List of names of exogenous variable columns
    :param nlags: Number of lags to include for each variable
    :return: Fitted OLS model and list of regressor names
    """
    df_lag = make_lagged_df(df, y_col, x_cols, nlags)

    y = df_lag[y_col]
    X = df_lag.drop(columns=[y_col])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model, X.columns.tolist()


def wald_test_coefficients(model_before, model_after, coef_names):
    """
    Joint Wald test for equality of selected coefficients between two models.

    :param model_before: Fitted statsmodels OLS model before the break
    :param model_after: Fitted statsmodels OLS model after the break
    :param coef_names: List of coefficient names to test
    :return: Wald statistic and p-value
    """
    # Extract coefficients and covariance matrices
    b1 = model_before.params[coef_names].values
    b2 = model_after.params[coef_names].values
    V1 = model_before.cov_params().loc[coef_names, coef_names].values
    V2 = model_after.cov_params().loc[coef_names, coef_names].values

    # Wald statistic
    diff = b1 - b2
    V = V1 + V2  # covariance of difference
    W = diff.T @ np.linalg.inv(V) @ diff

    # p-value
    dof = len(coef_names)
    pval = chi2.sf(W, dof)

    return W, pval


def coef_table(m_before, m_after, coef_names):
    """
    Create a DataFrame comparing coefficients from two models.

    :param m_before: model before the break
    :param m_after: model after the break
    :param coef_names: names of coefficients to include - need to be present in both models
    :return: a DataFrame with coefficients and stats
    """
    rows = []
    for name in coef_names:
        rows.append({
            "coef": name,
            "before_coef": m_before.params[name],
            "before_se":   m_before.bse[name],
            "before_t":    m_before.tvalues[name],
            "before_p":    m_before.pvalues[name],
            "after_coef":  m_after.params[name],
            "after_se":    m_after.bse[name],
            "after_t":     m_after.tvalues[name],
            "after_p":     m_after.pvalues[name],
        })
    return pd.DataFrame(rows)


#%%

if __name__ == "__main__":
    #%% Load data
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    PRODUCTION_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_OffshoreWind_CapacityFactors_SeasonAdjusted.csv")
    PRODUCTIONUNITS_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_ProductionUnits.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "Model")
    WINDDIRECTION_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindDirectionsAvg.csv")
    WINDSPEEDS_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindSpeedsAvg_SeasonalityRemoved.csv")

    df_winddirection = load_timeseries_data(WINDDIRECTION_DATA_PATH)
    df_windspeeds = load_timeseries_data(WINDSPEEDS_DATA_PATH)
    df_capacityfactors = load_timeseries_data(PRODUCTION_DATA_PATH)
    df_winddirection.set_index("DateTime (UTC)", inplace=True)
    df_windspeeds.set_index("DateTime (UTC)", inplace=True)
    df_capacityfactors.set_index("DateTime (UTC)", inplace=True)

    df_combined = pd.merge(df_capacityfactors, df_winddirection, how="inner", left_index=True, right_index=True)
    df_combined = pd.merge(df_combined, df_windspeeds, how="inner", left_index=True, right_index=True)
    sadjusted_cols = [col for col in df_capacityfactors.columns if col.endswith("_sadjusted")]
    sadjusted_cols_and_wind = sadjusted_cols + ["AverageWindDirection", "AverageWindSpeed_sadjusted"]
    first_dates = {col: df_combined[col].first_valid_index() for col in sadjusted_cols_and_wind}

    lag_1a = 6  # From previous analysis in ModelNoWindDirection.py

    #%% Subset data from 2015-01-01 and get seasonal components
    df_1a, df_1a_seasonal, farms = subset_farms_from_date(df_combined, first_dates, "2015-01-01 00:00:00", give_farms=True)
    df_1a_seasonal = df_1a_seasonal.assign(AverageWindDirection= df_1a["AverageWindDirection"])

    #%% A first analysis simply checking the mean capacity factor by wind direction bins
    plot_capacity_factor_by_direction(df_1a, 45, filepath=path.join(VISUALISATION_PATH, "CapacityFactorByWindDirection_Bins45.png"))

    #%% Check wind speed for structural breaks (not epxected)
    expected_break_index_fullset = df_1a.index.get_loc(pd.Timestamp("2020-12-02 00:00:00"))  # First index after break

    F_value, m1, m2 = chow_test_1D(df_1a["AverageWindSpeed_sadjusted"], break_index=expected_break_index_fullset, nlags=lag_1a)
    print("Structural change test for wind speed (unconditional):")
    print(f"\tF-statistic: {F_value:.4f}")
    print("\tparams before:", m1.params)
    print("\tparams after:", m2.params)
    print("\tabs diff:", np.abs(m1.params - m2.params))
    print("\trelative diff:", np.abs(m1.params - m2.params) / np.abs(m1.params))

    F_values = []
    window = 20  #TODO adjust back to 24*20 or similar
    for b in range(expected_break_index_fullset-window, expected_break_index_fullset+window):
        F_value, _, _ = qlrtest.chow_test_1D(df_1a["AverageWindSpeed_sadjusted"], break_index=b, nlags=lag_1a)
        F_values.append(F_value)

    plt.figure(figsize=(10, 6))
    plt.plot(range(expected_break_index_fullset-window, expected_break_index_fullset+window), F_values, marker='.', linestyle='-')
    plt.axvline(expected_break_index_fullset, color='r', linestyle='--', label=f"Actual Breakpoint: {expected_break_index_fullset}")
    plt.title('Chow Test F-statistics for Wind Speed (Unconditional)')
    plt.xlabel('Breakpoint Index')
    plt.ylabel('F-statistic')
    plt.legend()
    plt.grid(True)
    plt.savefig(path.join(VISUALISATION_PATH, "ChowTest_WindSpeed_Unconditional.png"), dpi=300)

    #%%

    # Split wind direction into categories, then dataframe groupby wind direction
    bins = [0, 90, 180, 270, 360]
    labels = ['NE', 'SE', 'SW', 'NW']
    df_1a['WindDirection_Category'] = pd.cut(df_1a['AverageWindDirection'], bins=bins, labels=labels, right=False, include_lowest=True)
    df_1a_grouped = df_1a.drop(df_1a.loc["2020-04-28 00:00:00":"2020-12-01 23:00:00"].index).groupby("WindDirection_Category", observed=False)
    df_1a_seasonal["WindDirection_Category"] = pd.cut(df_1a_seasonal['AverageWindDirection'], bins=bins, labels=labels, right=False, include_lowest=True)
    df_1a_seasonal_grouped = df_1a_seasonal.drop(df_1a_seasonal.loc["2020-04-28 00:00:00":"2020-12-01 23:00:00"].index).groupby("WindDirection_Category", observed=False)

    # for direction, group in df_grouped:
    #     print(f"Wind Direction: {direction}")
    #     print(group[farms].describe())
    #     print("\n")

    #%% Analyze each wind direction bin separately

    for direction in labels:
        print(f"\n--- Analyzing wind direction bin: {direction} ---\n")
        df_1a_grouped_subset = df_1a_grouped.get_group(direction)
        df_1a_seasonal_grouped_subset = df_1a_seasonal_grouped.get_group(direction)

        # Compute time gaps in the subset and visualize "missing" data
        time_diffs = df_1a_grouped_subset.index.to_series().diff()

        discontinuities = (time_diffs != pd.Timedelta(hours=1)).sum()
        total_n = len(df_1a_grouped_subset)
        print(f"Total number of observations in {direction} subset:", total_n)
        print("Number of discontinuities:", discontinuities)

        gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]

        plt.figure(figsize=(12, 2))
        plt.plot(df_1a_grouped_subset.index, [1] * len(df_1a_grouped_subset), '|', color='lightgray', markersize=12)
        plt.plot(gaps.index, [1] * len(gaps), '|', color='red', markersize=18)
        plt.yticks([])
        plt.title(f"Missing Data Visualization for {direction} Wind Direction")
        plt.xlabel("Timestamp")
        plt.tight_layout()
        plt.savefig(path.join(VISUALISATION_PATH, f"MissingData_{direction}_WindDirection.png"), dpi=300)

        # CONCLUSION: There are many short gaps, need to drop short segments before analysis

        #%% Drop short segments
        df_1a_grouped_subset = drop_short_segments(df_1a_grouped_subset, lag_1a, freq="1h", n=10)
        print(f"After dropping short segments, number of observations in {direction} subset:", len(df_1a_grouped_subset))

        idx = df_1a_grouped_subset.index
        start_ts = pd.Timestamp("2020-04-27 00:00:00")
        stop_ts = pd.Timestamp("2020-12-02 23:00:00")
        window = 10  # Hours #TODO adjust back to like 300

        existing_before = idx.asof(start_ts)  # First existing timestamp BEFORE or equal to start_ts
        existing_after = idx[idx > stop_ts].min()  # First existing timestamp AFTER stop_ts

        print("Existing timestamps around expected change/break:")
        print(existing_before, existing_after)

        #%% Chow test for structural change/break
        F_value, m1, m2, F_crit = chow_test_VAR(df_1a_grouped_subset, y_col="Belwind_sadjusted", x_cols=[
            "Belwind_sadjusted", "Northwind_sadjusted",
            "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted"], break_index=df_1a_grouped_subset.index.get_loc(existing_after), nlags=lag_1a)

        print(f"Structural change test for Belwind in {direction} direction bin:")
        print(f"\tF-statistic: {F_value:.4f} (F-crit 5%: {F_crit})")
        print("\tparams before:", m1.params)
        print("\tparams after:", m2.params)
        print("\tabs diff:", np.abs(m1.params - m2.params))
        print("\trelative diff:", np.abs(m1.params - m2.params) / np.abs(m1.params))
        # TODO uncomment
        """
        F_values = []
        for b in range(df_1a_grouped_subset.index.get_loc(existing_before)-window, df_1a_grouped_subset.index.get_loc(existing_after)+window):
            F_value, _, _ = chow_test_VAR(df_1a_grouped_subset, y_col="Belwind_sadjusted", x_cols=[
                "Belwind_sadjusted", "Northwind_sadjusted",
                "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted"], break_index=b, nlags=lag_1a)
            F_values.append(F_value)
        plt.figure(figsize=(10, 6))
        plt.plot(range(df_1a_grouped_subset.index.get_loc(existing_before)-window, df_1a_grouped_subset.index.get_loc(existing_after)+window), F_values, marker='.', linestyle='-')
        plt.axvline(df_1a_grouped_subset.index.get_loc(existing_after), color="r", linestyle="--", label=f"Expected potential breakpoint: {df_1a_grouped_subset.index.get_loc(existing_after)}")
        plt.title(f"Chow Test F-statistics for Belwind in {direction} Direction Bin")
        plt.xlabel("Breakpoint Index")
        plt.ylabel("F-statistic")
        plt.legend()
        plt.grid()
        plt.savefig(path.join(VISUALISATION_PATH, f"ChowTest_Belwind_{direction}_Direction.png"), dpi=300)
        """

        #%% Wald test for equality of coefficients before/after change

        y_col = "Belwind_sadjusted"
        x_cols = ["Northwind_sadjusted", "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted"]

        # Split before/after
        df_before = df_1a_grouped_subset.loc[:existing_after]
        df_after = df_1a_grouped_subset.loc[existing_after:]
        df_before_seasonal = df_1a_seasonal_grouped_subset.loc[:existing_after]
        df_after_seasonal = df_1a_seasonal_grouped_subset.loc[existing_after:]

        # Fit models
        m_before, cols_before = fit_model(df_before, y_col, x_cols, nlags=6)
        m_after,  cols_after  = fit_model(df_after,  y_col, x_cols, nlags=6)

        # Coefficients to test (contemporaneous only)
        coef_names = x_cols

        # Wald test
        W, pval = wald_test_coefficients(m_before, m_after, coef_names)

        print("Joint Wald statistic:", W)
        print("p-value:", pval)

        # Coefficient table
        coef_comparison = coef_table(m_before, m_after, coef_names)
        print("Coefficient comparison table:")
        print(coef_comparison)

        #%% Check forecasting performance with lag 6 is actually ok -- otherwise Wald test is meaningless

        cutoffs_before = ["2017-12-01", "2018-06-01", "2019-12-01"]
        cutoffs_after = ["2022-06-01", "2023-01-01", "2023-06-01"]

        rows_before, rows_after = [], []

        # BEFORE
        for c in cutoffs_before:
            _, m = forecast_model(df_before, "ARX", y_col, x_cols, lag_1a, c, [1, 2, 3, 6],
                                  df_before_seasonal["Belwind_season"])
            rows_before += [{"cutoff": c, "h": h, "RMSE": m[h]["RMSE"], "MAE": m[h]["MAE"]} for h in [1, 2, 3, 6]]

        # AFTER
        for c in cutoffs_after:
            _, m = forecast_model(df_after, "ARX", y_col, x_cols, lag_1a, c, [1, 2, 3, 6],
                                  df_after_seasonal["Belwind_season"])
            rows_after += [{"cutoff": c, "h": h, "RMSE": m[h]["RMSE"], "MAE": m[h]["MAE"]} for h in [1, 2, 3, 6]]

        df_before_metrics = pd.DataFrame(rows_before)
        df_after_metrics = pd.DataFrame(rows_after)


        #TODO make a visualisation of some forecasts vs actual?

    #%%
    #TODO check for the same results if using Thorntonbank_SW as target variable