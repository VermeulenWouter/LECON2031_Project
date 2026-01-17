"""
Initial exploration of VAR and AR models for offshore wind farm production data. Does not yet
include the wind direction.

@author: Wouter Vermeulen
@date: 2026-01-10
"""

import os.path as path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels import tsa
from statsmodels.tsa.api import VAR

from Helper import load_timeseries_data, correlation_matrix
from Helper import qlrtest
from Helper.forecasting import forecast_model
from InvestigateProductionData import plot_acf


def inspect_nans(df, col, filepath: str | None = None):
    """Plot the locations of NaN values in a time series column."""
    s = df[col]

    nan_positions = s[s.isna()].index

    fig = plt.figure(figsize=(12, 4))
    plt.plot(s.index, s, label=col, alpha=0.7)
    plt.scatter(nan_positions, [s.mean()] * len(nan_positions),
                color='red', label='NaN', s=20)
    plt.title(f"NaN locations in {col}")
    plt.legend()
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved NaN inspection plot to {filepath}")
    else:
        plt.show()
    plt.close(fig)


def adf_for_sadjusted(df, sadjusted_cols: list[str], augmented: bool = True) -> pd.DataFrame:
    """Perform ADF test on seasonally adjusted columns and return results as DataFrame.

    :param df: DataFrame containing the time series data
    :param sadjusted_cols: List of column names that are seasonally adjusted
    :param augmented: Whether to use augmented ADF test with automatic lag selection

    :return: DataFrame with ADF or DF test results for each column
    """
    cols = sadjusted_cols

    results = []
    for col in cols:
        series = df[col].dropna()
        if augmented:
            adf_stat, pvalue, used_lags, nobs, crit_vals, *_ = tsa.stattools.adfuller(series, regression='c', autolag='AIC')
        else:
            adf_stat, pvalue, used_lags, nobs, crit_vals, *_ = tsa.stattools.adfuller(series, regression='c', maxlag=0)

        if pvalue < 0.05:
            if adf_stat > crit_vals["5%"]:
                raise ValueError("Inconsistent ADF test results.")
            stationary = True
        else:
            stationary = False
            if adf_stat < crit_vals["5%"]:
                raise ValueError("Inconsistent ADF test results.")

        results.append({
            "column": col,
            "adf_statistic": adf_stat,
            "p_value": pvalue,
            "used_lags": used_lags,
            "nobs": nobs,
            "crit_1%": crit_vals["1%"],
            "crit_5%": crit_vals["5%"],
            "crit_10%": crit_vals["10%"],
            "is_stationary": stationary
        })

    # Return as DataFrame
    return pd.DataFrame(results).set_index("column")


def subset_farms_from_date(df: pd.DataFrame, first_dates: dict[str, pd.Timestamp], start_date: str | pd.Timestamp, give_farms: bool = False) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns a copy of df containing only the farms that have meaningful data starting on or before start_date.
    The returned df is also trimmed to start at the first joint meaningful timestamp (which can be before start_date).

    :param df: DataFrame with time series data
    :param first_dates: Dictionary mapping farm column names to their first meaningful timestamps
    :param start_date: The date from which to consider farms
    :return: Subset DataFrame with farms valid from start_date onward, trimmed to start at the first joint meaningful timestamp
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    valid_farms = [farm for farm, first_date in first_dates.items() if first_date is not None and first_date <= start_date]
    if not valid_farms:
        raise ValueError("No farms have meaningful data starting on or before the specified start_date.")

    joint_start = max(first_dates[farm] for farm in valid_farms)
    out = df[valid_farms].loc[joint_start:].copy()

    out_seasonal = pd.DataFrame(index=out.index, columns=out.columns)
    for farm in valid_farms:
        if farm.endswith("_sadjusted"):
            seasonal_col_1, seasonal_col_2 = farm[:-10] + "_daily_season", farm[:-10] + "_yearly_season"
            out_seasonal[farm[:-10] + "_season"] = df[[seasonal_col_1, seasonal_col_2]].loc[joint_start:].sum(axis=1)
    valid_farms = [farm for farm in valid_farms if farm.endswith("_sadjusted")]

    if give_farms:
        return out, out_seasonal, valid_farms

    return out, out_seasonal


def select_var_lag(df, cols: list[str], maxlags=48):
    """
    Select optimal VAR lag length using AIC and BIC.
    Returns a small DataFrame with the chosen lags.
    """
    model = VAR(df[cols], missing="drop", freq='h')
    sel = model.select_order(maxlags=maxlags)

    results = {
        "AIC_optimal_lag": sel.aic,
        "BIC_optimal_lag": sel.bic,
        "HQIC_optimal_lag": sel.hqic,
        "FPE_optimal_lag": sel.fpe
    }

    raw = pd.DataFrame({"lag": range(0, maxlags + 1), "AIC": sel.ics["aic"], "BIC": sel.ics["bic"], "HQIC": sel.ics["hqic"], "FPE": sel.ics["fpe"]}).set_index("lag")

    return results, raw, model



#%%

if __name__ == "__main__":
    #%%
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    PRODUCTION_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_OffshoreWind_CapacityFactors_SeasonAdjusted.csv")
    PRODUCTIONUNITS_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_ProductionUnits.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "Model")

    df = load_timeseries_data(PRODUCTION_DATA_PATH)
    df = df.set_index("DateTime (UTC)")
    sadjusted_cols = [col for col in df.columns if col.endswith("_sadjusted")]

    #%% Small overview of the data statistics
    print(df[sadjusted_cols].describe())
    print("\n")

    for col in sadjusted_cols:
        number_nan = df[col].loc[df[col].first_valid_index():].isna().sum()
        if number_nan > 0:
            number_total = df[col].loc[df[col].first_valid_index():].shape[0]
            print(f"{col}: {number_nan} NaN values out of {number_total} total values")
            inspect_nans(df, col, filepath=path.join(VISUALISATION_PATH, f"NaN_Inspection_{col}.png"))
    print("\n")
    # CONCLUSION: Rentel has one full missing day, Norther has a lot of missing data and should probably be excluded.

    #%% Patch Rentel missing day with average of surrounding days
    col = "Rentel_sadjusted"

    nan_mask = df[col].isna()
    nan_times = df.index[nan_mask]
    blocks = []
    start = nan_times[0]
    prev = nan_times[0]

    # Identify continuous blocks of missing data
    for ts in nan_times[1:]:
        if ts == prev + pd.Timedelta(hours=1):
            prev = ts
        else:
            blocks.append((start, prev))
            start = ts
            prev = ts
    blocks.append((start, prev))

    first_timestamp = df.index[0]
    internal_blocks = [(s, e) for (s, e) in blocks if s != first_timestamp]

    # Take the first and only internal missing block
    missing_start, missing_end = internal_blocks[0]

    replacement_values = []
    for ts in pd.date_range(missing_start, missing_end, freq="h"):
        prev_day = ts - pd.Timedelta(days=1)
        next_day = ts + pd.Timedelta(days=1)

        val_prev = df[col].loc[prev_day]
        val_next = df[col].loc[next_day]

        replacement_values.append((val_prev + val_next) / 2)

    df.loc[missing_start:missing_end, col] = replacement_values

    if df[col].loc[df[col].first_valid_index():].isna().sum() != 0:
        raise ValueError(f"Failed to patch all missing data in {col}.")
    print(f"Patched missing data for {col} from {missing_start} to {missing_end}.")

    #%%
    # ADF test for all seasonally adjusted columns
    adf_results = adf_for_sadjusted(df, sadjusted_cols)
    print(adf_results)
    # CONCLUSION: All series are stationary, no unit root present.

    #%%
    corr_lag0 = correlation_matrix(df, sadjusted_cols)
    corr_lag0_off_diag = corr_lag0.to_numpy()[~np.eye(corr_lag0.shape[0], dtype=bool)]  # Off-diagonal elements (strictly speaking, double info)
    print("\nCorrelation matrix at lag 0:")
    print(corr_lag0)
    print(f"\tMean off-diagonal Pearson correlation coefficient: {np.nanmean(corr_lag0_off_diag):.4f}")
    print(f"\tMin off-diagonal Pearson correlation coefficient: {np.nanmin(corr_lag0_off_diag):.4f}")
    # CONCLUSION: High correlations between different wind farms => VAR model could be appropriate.

    #%% Part 1.a: try detecting shock due to Borssele using VAR lag selection on farms active on 2015-01-01
    # Create a dict with the first meaningful timestamp for each farm
    first_dates = {col: df[col].first_valid_index() for col in sadjusted_cols}

    df_1a, df_1a_seasonal = subset_farms_from_date(df, first_dates, "2015-01-01 00:00:00")

    #%%

    lagselection_stats_1a, lagselection_raw_1a, model_1a = select_var_lag(df_1a, df_1a.columns, maxlags=48)
    print(f"\nVAR lag selection on data from farms active on 2015-01-01 (farms {df_1a.columns.to_list}):")
    print("\n\nVAR lag selection statistics:")
    print(lagselection_raw_1a)
    print("\nOptimal lags according to different criteria:")
    print(lagselection_stats_1a)
    lag_1a = lagselection_stats_1a["BIC_optimal_lag"]
    print(f"\tChosen lag for Part 1.a according to BIC: {lag_1a}")

    #%%
    results_1a = model_1a.fit(lag_1a)
    stability_1a = results_1a.is_stable()
    print(f"\tModel stability: {'ok' if stability_1a else 'not ok'}")

    print(results_1a.summary())
    coefs_1a = results_1a.params
    stderr_1a = results_1a.stderr
    tstats_1a = results_1a.tvalues
    pvals_1a = results_1a.pvalues

    # Check normal distribution of residuals
    jb_test_1a = results_1a.test_normality()
    print("\nJarque-Bera test for normality of residuals:")
    print(jb_test_1a.summary())

    # Check for autocorrelation of residuals
    lb_test_1a = results_1a.test_whiteness(nlags=lag_1a+1)
    print("\nLjung-Box test for autocorrelation of residuals:")
    print(lb_test_1a.summary())

    # Check for heteroscedasticity of residuals
    from statsmodels.stats.diagnostic import het_arch
    for eq in df_1a.columns:
        het_test_1a = het_arch(results_1a.resid[eq])
        print("\nHeteroscedasticity test for residuals:")
        print(het_test_1a)

    for col in df_1a.columns:
        plot_acf(results_1a.resid, col, lags=48, gridline_interval_lags=12, filepath=path.join(VISUALISATION_PATH, f"VAR_Residuals_ACF_{col}_Part1a.png"))

    # CONCLUSION: Residuals are not normally distributed, show autocorrelation and heteroscedasticity. BUT VAR model can still be used for forecasting.
    # the CI is tiny due to large sample size
    # the ACF shows only the trivial autocorrelation at lag 0, so residuals are basically white noise.

    #%% Same as 1.a, but for farms active on 2019-01-01 (Part 1.b) => including Nobelwind and Rentel
    df_1b, df_1b_seasonal = subset_farms_from_date(df, first_dates, "2019-01-01 00:00:00")

    lagselection_stats_1b, lagselection_raw_1b, model_1b = select_var_lag(df_1b, df_1b.columns, maxlags=48)
    print(f"\nVAR lag selection on data from farms active on 2019-01-01 (farms {df_1b.columns.to_list()}):")
    print("\n\nVAR lag selection statistics:")
    print(lagselection_raw_1b)
    print("\nOptimal lags according to different criteria:")
    print(lagselection_stats_1b)

    lag_1b = lagselection_stats_1b["BIC_optimal_lag"]
    print(f"\tChosen lag for Part 1.b according to BIC: {lag_1b}")

    #%%
    # Check model accuracy for both 1.a and 1.b using VAR models (AR model as benchmark)
    for farm in ["Belwind_sadjusted", "Thorntonbank_NE_sadjusted"]:
        forecasts, metrics = forecast_model(
            df=df_1a,
            model_type="VAR",
            target_col=farm,
            cols=["Belwind_sadjusted", "Northwind_sadjusted",
                  "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted"],
            lag_order=lag_1a,
            cutoff_date="2021-11-01 00:00:00",
            horizons=[1, 3, 6, 12, 24],
            seasonal_component=df_1a_seasonal[farm[:-10] + "_season"]
        )
        print(f"\nForecast metrics for {farm} using VAR model (set 1.a):")
        for h, mets in metrics.items():
            print(f"\tHorizon {h}h: RMSE={mets['RMSE']:.4f}, MAE={mets['MAE']:.4f}")

        # Benchmark: univariate AR model
        forecasts_ar, metrics_ar = forecast_model(
            df=df_1a,
            model_type="AR",
            target_col=farm,
            cols=None,  # ignored for AR
            lag_order=lag_1a,
            cutoff_date="2021-11-01 00:00:00",
            horizons=[1, 3, 6, 12, 24],
            seasonal_component=df_1a_seasonal[farm[:-10] + "_season"]
        )
        print(f"\nForecast metrics for {farm} using AR model (set 1.a):")
        for h, mets in metrics_ar.items():
            print(f"\tHorizon {h}h: RMSE={mets['RMSE']:.4f}, MAE={mets['MAE']:.4f}")

        forecasts, metrics = forecast_model(
            df=df_1b,
            model_type="VAR",
            target_col=farm,
            cols=["Belwind_sadjusted", "Northwind_sadjusted",
                  "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted",
                  "Nobelwind_sadjusted", "Rentel_sadjusted"],
            lag_order=lag_1b,
            cutoff_date="2021-11-01 00:00:00",
            horizons=[1, 3, 6, 12, 24],
            seasonal_component=df_1b_seasonal[farm[:-10] + "_season"]
        )
        print(f"\nForecast metrics for {farm} using VAR model (set 1.b):")
        for h, mets in metrics.items():
            print(f"\tHorizon {h}h: RMSE={mets['RMSE']:.4f}, MAE={mets['MAE']:.4f}")


        #TODO make one plot showing the 1h forecasts from the 3 models + the real one
        #TODO make one plot showing the MAE in function of horizon for the 3 models

    #%% Check for structural break with Chow test
    results = qlrtest.qlr_test(df_1a,
    target_col="Belwind_sadjusted",
    cols=["Belwind_sadjusted", "Northwind_sadjusted",
          "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted"], lags=lag_1a, start_index=df_1a.index.get_loc(pd.Timestamp("2020-11-01 00:00:00")),# "2020-04-28 00:00:00")),
        stop_index=df_1a.index.get_loc(pd.Timestamp("2020-12-01 23:00:00")), sig_level=0.05, return_f_stats_series=True)

    print("QLR Test Results:")
    print(f"  Maximum F-statistic: {results['max_f_stat']:.4f}")
    print(f"  Estimated Breakpoint Index: {results['breakpoint']}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Significant at 5% level: {results['significant']}")

    if 'f_stats' in results and results['f_stats'] is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(results['tested_indices'], results['f_stats'], marker='.', linestyle='-')
        plt.axvline(results['breakpoint'], color='r', linestyle='--',
                    label=f"Estimated Breakpoint: {results['breakpoint']}")
        plt.title('QLR F-statistics across potential breakpoints')
        plt.xlabel('Breakpoint Index')
        plt.ylabel('F-statistic')
        plt.legend()
        plt.grid(True)
        plt.savefig(path.join(VISUALISATION_PATH, "QLR_Test_FStatistics_Part1a.png"), dpi=300)
        plt.show()

    #%%
    # Then deleting between 2020-05-01 and 2020-12-01 and retesting
    df_1a_subset = df_1a.drop(df_1a.loc["2020-04-28 00:00:00":"2020-12-01 23:00:00"].index)

    results_subset = qlrtest.qlr_test(df_1a_subset,
    target_col="Belwind_sadjusted",
    cols=["Belwind_sadjusted", "Northwind_sadjusted",
          "Thorntonbank_NE_sadjusted", "Thorntonbank_SW_sadjusted"],
    lags=lag_1a, sig_level=0.05, return_f_stats_series=True,
    start_index=df_1a_subset.index.get_loc(pd.Timestamp("2020-04-27 00:00:00")),
                                      stop_index=df_1a_subset.index.get_loc(pd.Timestamp("2020-12-02 23:00:00")))


    print("QLR Test Results:")
    print(f"  Maximum F-statistic: {results_subset['max_f_stat']:.4f}")
    print(f"  Estimated Breakpoint Index: {results_subset['breakpoint']}")
    print(f"  P-value: {results_subset['p_value']:.4f}")
    print(f"  Significant at 5% level: {results_subset['significant']}")

    if 'f_stats' in results_subset and results_subset['f_stats'] is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(results_subset['tested_indices'], results_subset['f_stats'], marker='.', linestyle='-')
        plt.axvline(results_subset['breakpoint'], color='r', linestyle='--',
                    label=f"Estimated Breakpoint: {results_subset['breakpoint']}")
        plt.title('QLR F-statistics across potential breakpoints')
        plt.xlabel('Breakpoint Index')
        plt.ylabel('F-statistic')
        plt.legend()
        plt.grid(True)
        plt.savefig(path.join(VISUALISATION_PATH, "QLR_Test_FStatistics_Part1a_Subset.png"), dpi=300)
        plt.show()