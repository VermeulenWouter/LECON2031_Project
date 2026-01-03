"""
Investigate offshore wind production data for Belgium. Loads data, visualises time series, then
analyzes and removes daily and yearly seasonalities using OLS regression with dummy variables.

@author: Wouter Vermeulen
@date: 2025-01-03
"""

import os.path as path

import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from Helper import *
from Helper.Plots import plot_series_with_zooms, plot_series
from InvestigateWindSpeed import plot_daily_profile, plot_yearly_profile

import statsmodels.api as sm
import statsmodels.formula.api as smf

#%%

def plot_acf(df, column, lags=40, gridline_interval_lags=24, filepath: str | None = None):
    """
    Plot the ACF of df[column] with a 95% confidence interval.
    NaNs are handled automatically by statsmodels (missing='drop').

    :param df: dataframe
    :param column: column name existing in the df to plot
    :param lags: number of lags to show and calculate the ACF for
    :param gridline_interval_lags: interval of lags at which to draw minor grid lines
    :param filepath: filepath to save the figure, if None the figure is shown instead
    """
    series = df[column]

    fig = sm.graphics.tsa.plot_acf(series, lags=lags, alpha=0.05, missing='drop')
    ax = fig.axes[0]
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(gridline_interval_lags))
    ax.grid(True, axis='x', which="minor")
    ax.set_title(f"Autocorrelation Function - {column}")
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved ACF plot to {filepath}")
    else:
        plt.show()

    plt.close(fig)


def test_seasonality(df, column, freq: str = "daily"):
    """
    Runs a daily (that is, with a cycle period of one day) or yearly (that is, with a cycle period of one year) seasonality test using OLS with hour dummies, extracts the estimated 24-hour seasonal pattern.

    :param df: dataframe with datetime index
    :param column: column name existing in the df to test
    :param freq: 'daily' or 'yearly' to indicate which seasonality to test
    :return: (estimated_pattern: pd.Series indexed by hour 0-23 (or month 1-11), f_test: statsmodels F-test result, model: statsmodels regression results)
    """
    df2 = df[[column]].dropna().copy()
    df2["hour" if freq == "daily" else "month"] = df2.index.hour if freq == "daily" else df2.index.month

    model = smf.ols(f"{column} ~ C(hour)" if freq == "daily" else f"{column} ~ C(month)", data=df2).fit()

    terms = [name for name in model.params.index if name.startswith("C(hour)[T.") or name.startswith("C(month)[T.")]
    constraint = " = 0, ".join(terms) + " = 0"
    f_test = model.f_test(constraint)

    if freq == "daily":
        idx = np.arange(24)
        pattern = [model.params["Intercept"] if h == 0 else model.params.get(f"C(hour)[T.{h}]", 0.0) + model.params["Intercept"] for h in idx]
    else:
        idx = range(1, 13)
        pattern = [model.params["Intercept"] if m == 1 else model.params.get(f"C(month)[T.{m}]", 0.0) + model.params["Intercept"] for m in idx]

    pattern = pd.Series(pattern, index=idx, name="estimated_pattern")
    return pattern, f_test, model


def remove_given_seasonalities(df, columns, daily_pattern, yearly_pattern):
    """
    Removes given daily and yearly seasonal components from the dataframe for the specified columns.

    :param df: the dataframe with datetime index
    :param columns: list of column names for which to remove seasonalities
    :param daily_pattern: dict {col: daily_pattern_series}
    :param yearly_pattern: dict {col: yearly_pattern_series}
    :return: new dataframe with original values, seasonal components, and deseasonalized values. Columns added:
        - {col}_daily_season
        - {col}_yearly_season
        - {col}_deseason
    """

    out = df.copy()

    # Extract hour and month once
    out["hour"] = out.index.hour
    out["month"] = out.index.month

    for col in columns:
        # Map seasonal components
        out[f"{col}_daily_season"] = out["hour"].map(daily_pattern[col])
        out[f"{col}_yearly_season"] = out["month"].map(yearly_pattern[col])

        # Remove both seasonal components
        out[f"{col}_deseason"] = out[col] - out[f"{col}_daily_season"] - out[f"{col}_yearly_season"]

    return out


#%%
if __name__ == "__main__":
    FIGS = False
    # %%
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    PRODUCTION_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_GenerationPerProductionUnit.csv")
    PRODUCTIONUNITS_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_ProductionUnits.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "ElectricityPreProcessing")

    df_allproductions = load_timeseries_data(PRODUCTION_DATA_PATH)
    df_generators = pd.read_csv(PRODUCTIONUNITS_DATA_PATH, sep=';')

    df_generators_ow = df_generators[df_generators['GenerationUnitType'] == "Wind Offshore"]
    df_productions_ow = df_allproductions[[*df_generators_ow["GenerationUnitCode"], "DateTime (UTC)"]]

    #%%
    fig, ax = plt.subplots(figsize=(12, 8), nrows=6, ncols=2)
    for i, col in enumerate(df_productions_ow.columns):
        if col != "DateTime (UTC)":
            generator_name = unitname_to_commonname[df_generators_ow[df_generators_ow["GenerationUnitCode"] == col]["GenerationUnitName"].values[0]]
            a, b = divmod(i-1, 2)
            ax[a][b].plot(df_productions_ow["DateTime (UTC)"], df_productions_ow[col], label=f"{generator_name} ({col})", color=farms_to_color.get(generator_name, None))
            ax[a][b].grid()
            ax[a][b].legend(loc="lower left")
            ax[a][b].set_xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2026-12-31"))
            i += 1

    plt.tight_layout()
    plt.savefig(path.join(VISUALISATION_PATH, "Stacked_Offshore_Wind_Generation.png"))
    if FIGS:
        plt.show()
    plt.close(fig)

    # CONCLUSION: Norther Offshore WP is present twice but for different time periods => Merge
    #%%

    # Union the two Norther Offshore WP columns
    norther_duplicates = df_generators_ow[df_generators_ow["GenerationUnitName"] == "Norther Offshore WP"]["GenerationUnitCode"].to_list()
    print(f"UnitGenerationCode for Northern: {norther_duplicates}")
    df_productions_ow = df_productions_ow.assign(**{"Norther Offshore WP - Merged": np.where(df_productions_ow[norther_duplicates[0]] == 0, df_productions_ow[norther_duplicates[1]], df_productions_ow[norther_duplicates[0]])})
    df_productions_ow = df_productions_ow.drop(columns=[norther_duplicates[0], norther_duplicates[1]])
    df_productions_ow.rename(columns={"Norther Offshore WP - Merged": norther_duplicates[0]}, inplace=True)

    # Rename columns to common names
    rename_dict = df_generators_ow[["GenerationUnitCode", "GenerationUnitName"]].set_index("GenerationUnitCode")["GenerationUnitName"].to_dict()
    rename_dict = {k: unitname_to_commonname[v] for k, v in rename_dict.items()}
    df_productions_ow.rename(columns=rename_dict, inplace=True)

    #%%
    # Convert generation data in MW to capacity factors
    df_cf_ow = df_productions_ow.copy()
    df_cf_ow = df_cf_ow.set_index("DateTime (UTC)")
    for col in df_cf_ow.columns:
        capacity_given = df_generators_ow[df_generators_ow["GenerationUnitName"] == commonname_to_unitname[col]]["GenerationUnitInstalledCapacity(MW)"].values[0]
        capacity_exp = df_cf_ow[col].max()
        if not capacity_exp <= capacity_given:
            print(f"WARNING: Capacity mismatch for {col}: Given={capacity_given}, Experimental={capacity_exp}")
        df_cf_ow[col] = df_cf_ow[col] / max(capacity_given, capacity_exp)

    fig, ax = plt.subplots(figsize=(12, 8), nrows=5, ncols=2)
    for i, col in enumerate(df_cf_ow.columns):
        b, a = divmod(i - 1, 5)
        ax[a][b].plot(df_cf_ow[col], label=col, color=farms_to_color.get(col, None))
        ax[a][b].grid()
        ax[a][b].legend(loc="lower left")
        ax[a][b].set_xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2026-12-31"))
        i += 1

    plt.tight_layout()
    plt.savefig(path.join(VISUALISATION_PATH, "Stacked_Offshore_Wind_CapacityFactors.png"))
    if FIGS:
        plt.show()
    plt.close(fig)

    for col in df_cf_ow.columns:
        plot_series_with_zooms(df_cf_ow, [col], filepath=path.join(VISUALISATION_PATH, f"TimeSeriesWithZooms_Offshore_Wind_CapacityFactor_{col.replace(' ', '_')}.png"), quantity=f"Capacity Factor of {col}", unit="", ymax=1.0, colors=[farms_to_color.get(col, None)])

    #%%
    # Visually inspect seasonality and/or trends
    mean_values_daily = {}
    mean_values_yearly = {}
    for col in df_cf_ow.columns:
        mean_values_daily[col] = plot_daily_profile(df_cf_ow, col, filepath=path.join(VISUALISATION_PATH, f"DailyProfile_Offshore_{col.replace(' ', '_')}.png"), title=f"Daily Profile Offshore Wind Capacity Factor - {col}", ylabel="Capacity Factor")
        mean_values_yearly[col] = plot_yearly_profile(df_cf_ow, col, filepath=path.join(VISUALISATION_PATH, f"YearlyProfile_Offshore_{col.replace(' ', '_')}.png"), title=f"Yearly Profile Offshore Wind Capacity Factor - {col}", ylabel="Capacity Factor", ylim=(0.0, 1.0))

    plot_series(pd.DataFrame(mean_values_daily), list(mean_values_daily.keys()), filepath=path.join(VISUALISATION_PATH, f"DailyProfile_Offshore_Wind_CapacityFactors_AllFarms.png"), quantity="Daily Profile Offshore Wind Capacity Factor", unit="Capacity Factor")
    plot_series(pd.DataFrame(mean_values_yearly), list(mean_values_yearly.keys()), filepath=path.join(VISUALISATION_PATH, f"YearlyProfile_Offshore_Wind_CapacityFactors_AllFarms.png"), quantity="Yearly Profile Offshore Wind Capacity Factor", unit="Capacity Factor")

    #%%
    for col in df_cf_ow.columns:
        # Yearly
        yearly_lags = 365*24*4 if col != "Norther" else 365*24*1.3  # Norther has less data
        plot_acf(df_cf_ow, col, lags=yearly_lags, gridline_interval_lags=365*24, filepath=path.join(VISUALISATION_PATH, f"ACF_Offshore_Wind_CapacityFactor_Yearly_{col.replace(' ', '_')}.png"))
        # CONCLUSION: a sine-like yearly seasonality is visible

        # Daily
        plot_acf(df_cf_ow, col, lags=24*10, gridline_interval_lags=24, filepath=path.join(VISUALISATION_PATH, f"ACF_Offshore_Wind_CapacityFactor_Daily_{col.replace(' ', '_')}.png"))
        # CONCLUSION: a daily seasonality is visible
    #%%

    patterns_daily = {}
    patterns_yearly = {}
    f_tests_daily = {}
    f_tests_yearly = {}
    models_daily = {}
    models_yearly = {}
    df_tmp = df_cf_ow.copy()
    for col in df_cf_ow.columns:
        patterns_daily[col], f_tests_daily[col], models_daily[col] = test_seasonality(df_cf_ow, col, freq="daily")
        df_tmp[col + "_daily_removed"] = df_cf_ow[col] - df_cf_ow.index.hour.map(patterns_daily[col])
        patterns_yearly[col], f_tests_yearly[col], models_yearly[col] = test_seasonality(df_tmp, col + "_daily_removed", freq="yearly")

    rows = []
    for col in df_cf_ow.columns:
        # Daily
        md = models_daily[col]
        fd = f_tests_daily[col]
        r2_d = md.rsquared
        f_d = float(fd.fvalue)
        p_d = float(fd.pvalue)
        c_d = "seasonality" if p_d < 0.05 else "no seasonality"

        # Yearly
        my = models_yearly[col]
        fy = f_tests_yearly[col]
        r2_y = my.rsquared
        f_y = float(fy.fvalue)
        p_y = float(fy.pvalue)
        c_y = "seasonality" if p_y < 0.05 else "no seasonality"

        rows.append([col, r2_d, f_d, p_d, c_d, r2_y, f_y, p_y, c_y])

    summary = pd.DataFrame(rows, columns=["Wind Farm", "Daily R²", "Daily F", "Daily p", "Daily Conclusion", "Yearly R²", "Yearly F", "Yearly p", "Yearly Conclusion"])
    print(f"\nSeasonality Test Summary for Offshore Wind Capacity Factors:")
    print(summary)

    # CONCLUSION: Both daily and yearly seasonalities are significant for all farms

    #%%
    df_cf_ow_noseason = remove_given_seasonalities(df_cf_ow, df_cf_ow.columns, patterns_daily, patterns_yearly)

    #%%
    # Visually inspect seasonality and/or trends
    mean_values_daily_noseason = {}
    mean_values_yearly_noseason = {}
    for col in df_cf_ow_noseason.columns:
        if col.endswith("_deseason"):
            mean_values_daily_noseason[col] = plot_daily_profile(df_cf_ow_noseason, col, filepath=path.join(VISUALISATION_PATH, f"DailyProfileNoSeason_Offshore_{col.replace(' ', '_')}.png"), title=f"Daily Profile Offshore Wind Capacity Factor - {col}", ylabel="Capacity Factor")
            mean_values_yearly_noseason[col] = plot_yearly_profile(df_cf_ow_noseason, col, filepath=path.join(VISUALISATION_PATH, f"YearlyProfileNoSeason_Offshore_{col.replace(' ', '_')}.png"), title=f"Yearly Profile Offshore Wind Capacity Factor - {col}", ylabel="Capacity Factor")

    plot_series(pd.DataFrame(mean_values_daily_noseason), list(mean_values_daily_noseason.keys()), filepath=path.join(VISUALISATION_PATH, f"DailyProfileNoSeason_Offshore_Wind_CapacityFactors_AllFarms.png"), quantity="Daily Profile Offshore Wind Capacity Factor", unit="Capacity Factor")
    plot_series(pd.DataFrame(mean_values_yearly_noseason), list(mean_values_yearly_noseason.keys()), filepath=path.join(VISUALISATION_PATH, f"YearlyProfileNoSeason_Offshore_Wind_CapacityFactors_AllFarms.png"), quantity="Yearly Profile Offshore Wind Capacity Factor", unit="Capacity Factor")

    for col in df_cf_ow_noseason.columns:
        if col.endswith("_deseason"):
            # Yearly
            yearly_lags = 365*24*4 if "Norther" not in col else 365*24*1.3  # Norther has less data
            plot_acf(df_cf_ow_noseason, col, lags=yearly_lags, gridline_interval_lags=365*24, filepath=path.join(VISUALISATION_PATH, f"NoSeason_ACF_Offshore_Wind_CapacityFactor_Yearly_{col.replace(' ', '_')}.png"))
            # CONCLUSION: the sine-like yearly seasonality is gone

            # Daily
            plot_acf(df_cf_ow_noseason, col, lags=24*10, gridline_interval_lags=24, filepath=path.join(VISUALISATION_PATH, f"NoSeason_ACF_Offshore_Wind_CapacityFactor_Daily_{col.replace(' ', '_')}.png"))
            # CONCLUSION: the daily seasonality is more or less gone


    for col in df_cf_ow_noseason.columns:
        if col.endswith("_deseason"):
            _, f_tests_daily[col], models_daily[col] = test_seasonality(df_cf_ow_noseason, col, freq="daily")
            _, f_tests_yearly[col], models_yearly[col] = test_seasonality(df_cf_ow_noseason, col, freq="yearly")

    rows = []
    for col in df_cf_ow_noseason.columns:
        if col.endswith("_deseason"):
            # Daily
            md = models_daily[col]
            fd = f_tests_daily[col]
            r2_d = md.rsquared
            f_d = float(fd.fvalue)
            p_d = float(fd.pvalue)
            c_d = "seasonality" if p_d < 0.05 else "no seasonality"

            # Yearly
            my = models_yearly[col]
            fy = f_tests_yearly[col]
            r2_y = my.rsquared
            f_y = float(fy.fvalue)
            p_y = float(fy.pvalue)
            c_y = "seasonality" if p_y < 0.05 else "no seasonality"

            rows.append([col, r2_d, f_d, p_d, c_d, r2_y, f_y, p_y, c_y])

    summary = pd.DataFrame(rows, columns=["Wind Farm", "Daily R²", "Daily F", "Daily p", "Daily Conclusion", "Yearly R²", "Yearly F", "Yearly p", "Yearly Conclusion"])
    print(f"\nSeasonality Test Summary for Offshore Wind Capacity Factors after removing Seasonality:")
    print(summary)

    #%%
    # Save cleaned data
    df_cf_ow_noseason.to_csv(path.join(_FILE_PATH, "Data", "Electricity", "BE_OffshoreWind_CapacityFactors_Deseasonalized.csv"), sep=";", index_label="DateTime (UTC)")
