"""
Unionize wind speed data from multiple stations (after checking correlations between them are sufficient
to do so). Then analyze typical daily and yearly wind speed profiles.

@author: Wouter Vermeulen
@date: 2025-01-01
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import pandas as pd
from scipy.stats import t

from Helper import correlation_matrix, load_timeseries_data
from Helper.Plots import plot_series, plot_series_with_zooms
from Helper.TestSeasonality import test_seasonality, remove_given_seasonalities


def station_deviation_stats(df):
    """
    Compute basic statistics (mean, and 95%) of wind speed deviation from average at each station.

    :param df:
    :return:
    """
    stats = pd.DataFrame(index=df.columns, columns=["Mean Deviation (m/s)", "95% Deviation (m/s)"], dtype=float)
    mean_wind = df.mean(axis=1)
    for station in df.columns:
        deviation = (df[station] - mean_wind).abs()
        stats.at[station, "Mean Deviation (m/s)"] = deviation.mean()
        stats.at[station, "95% Deviation (m/s)"] = deviation.quantile(0.95)
        deviation_percent = (deviation / mean_wind).replace([np.inf, -np.inf], np.nan).dropna() * 100
        stats.at[station, "Mean Deviation (%)"] = deviation_percent.mean()
        stats.at[station, "95% Deviation (%)"] = deviation_percent.quantile(0.95)
    return stats

#%%

def plot_windspeed_histogram(df, filepath: str | None = None):
    """
    Plot histogram of wind speeds.
    :param df: DataFrame with wind speed data.
    :param filepath: Filepath to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['AverageWindSpeed'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Average Wind Speeds')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.grid(True)

    if filepath is not None:
        plt.savefig(filepath)
        print(f"Saved wind speed histogram to {filepath}")
    else:
        plt.show()


def _make_speed_bins(speeds, step=0.5, max_bins=5):
    """Create readable speed bins rounded to 0.5 increments."""
    smin = float(np.nanmin(speeds))
    smax = float(np.nanmax(speeds))

    # Round outward to nearest 0.5
    low = np.floor(smin / step) * step
    high = np.ceil(smax / step) * step

    # Limit number of bins
    nbins = int((high - low) / step)
    if nbins > max_bins:
        # Increase step size until bins are reasonable
        factor = np.ceil(nbins / max_bins)
        step = step * factor
        low = np.floor(smin / step) * step
        high = np.ceil(smax / step) * step

    edges = np.arange(low, high + step, step)

    # Safety: ensure at least 2 edges
    if len(edges) < 2:
        edges = np.array([low, low + step])

    return edges


def plot_windrose_with_speed(df, filepath: str | None = None, area_proportional: bool = True, bins: int = 36):
    """
    Plot wind rose with wind speed information.
    :param df: DataFrame with wind speed and direction data.
    :param filepath: Filepath to save the figure.
    :param area_proportional: Whether to use area-proportional representation.
    :param bins: Number of direction bins.
    """

    directions_deg = pd.to_numeric(df["AverageWindDirection"], errors="coerce")
    speeds = pd.to_numeric(df["AverageWindSpeed"], errors="coerce")
    sub = pd.DataFrame({"dir": directions_deg, "spd": speeds}).dropna()
    dirs = np.deg2rad(sub["dir"])
    spd = sub["spd"]

    speed_bins = _make_speed_bins(spd)
    speed_labels = [f"{speed_bins[i]:.1f}–{speed_bins[i+1]:.1f}" for i in range(len(speed_bins)-1)]
    cmap = plt.get_cmap("viridis")
    speed_colors = [cmap(i / (len(speed_bins)-2)) for i in range(len(speed_bins)-1)]

    H, dir_edges, spd_edges = np.histogram2d(dirs, spd, bins=[bins, speed_bins])
    P = H / H.sum() # sums to 1
    P_dir = P.sum(axis=1)
    n_dir = P.shape[0]
    n_spd = P.shape[1] #
    radii_bottom = np.zeros_like(P)
    radii_top = np.zeros_like(P)

    if area_proportional:
        for d in range(n_dir):
            r2 = 0.0
            for i in range(n_spd):
                p = P[d, i]
                radii_bottom[d, i] = np.sqrt(r2)
                r2 += p
                radii_top[d, i] = np.sqrt(r2)
        max_r = radii_top.max()
    else:
        for d in range(n_dir):
            r = 0.0
            for i in range(n_spd):
                p = P[d, i]
                radii_bottom[d, i] = r
                r += p
                radii_top[d, i] = r
        max_r = radii_top.max()

    plt.close("all")
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")

    widths = np.diff(dir_edges)
    gap = widths * 0.05
    widths = widths - gap

    theta = dir_edges[:-1] + widths / 2
    for i in range(n_spd):
        ax.bar( theta, radii_top[:, i] - radii_bottom[:, i], width=widths, bottom=radii_bottom[:, i], align="center", color=speed_colors[i], edgecolor="white", linewidth=0.8, label=speed_labels[i])
    tick_r = np.linspace(0, max_r, 5)
    if area_proportional:
        tick_percent = (tick_r**2) * 100
    else:
        tick_percent = tick_r * 100

    ax.set_rticks(tick_r)
    ax.set_yticklabels([f"{p:.1f}%" for p in tick_percent])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        np.arange(0, 360, 45),
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    )

    title = "Wind Direction Distribution with Speed Coloring"
    if area_proportional:
        title += " (Area-Proportional)"
    else:
        title += " (Radius-Proportional)"
    ax.set_title(title)

    ax.legend()
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved wind rose to {filepath}")
    else:
        plt.show()


def plot_daily_profile(df, column: str, include_ci=True, ci_level=0.95, filepath=None, ylabel: str | None = None, title: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 4))

    df = df.copy()
    df["hour"] = df.index.hour
    grouped = df.groupby("hour")[column]
    mean = grouped.mean()

    # Confidence intervals
    if include_ci:
        n = grouped.count()
        std = grouped.std()
        se = std / np.sqrt(n)
        tcrit = t.ppf((1 + ci_level) / 2, df=n - 1)

        lower_ci = mean - tcrit * se
        upper_ci = mean + tcrit * se

        ax.fill_between(mean.index, lower_ci, upper_ci, color="C1", alpha=0.15, label=f"{int(ci_level*100)}% CI")

    ax.plot(mean.index, mean.values, color="C0", label="Raw hourly mean")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(24))
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved daily profile plot to {filepath}")
    else:
        plt.show()

    return mean.values


def plot_yearly_profile(df, column: str, include_raw=True, include_ci=True, ci_level=0.95, filepath=None, ylabel: str | None = None, title: str | None = None, ylim: tuple[float, float] | None = None):
    fig, ax = plt.subplots(figsize=(14, 4))

    df = df.copy()
    df["date"] = df.index.date
    df["doy"] = df.index.dayofyear
    daily = df.groupby(["date", "doy"])[column].mean().reset_index()
    grouped = daily.groupby("doy")[column]
    mean = grouped.mean()

    # Confidence intervals
    if include_ci:
        n = grouped.count()
        std = grouped.std()
        se = std / np.sqrt(n)
        tcrit = t.ppf((1 + ci_level) / 2, df=n - 1)

        lower_ci = mean - tcrit * se
        upper_ci = mean + tcrit * se

        ax.fill_between(mean.index, lower_ci, upper_ci, color="C1", alpha=0.15, label=f"{int(ci_level*100)}% CI")

    if include_raw:
        ax.plot(mean.index, mean.values, color="C0", label="Raw daily mean", alpha=0.8)

    smooth = mean.rolling(window=30, center=True, min_periods=1).mean()
    ax.plot(smooth.index, smooth.values, color="C2", linewidth=2.0, label="30‑day smoothed")

    ax.set_xlabel("Day of Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 366)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved yearly profile plot to {filepath}")
    else:
        plt.show()

    return mean.values


#%%
if __name__ == "__main__":

    # %%
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    WINDSPEEDS_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindSpeeds.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "WindPreprocessing")

    AVG_WINDDIRECTIONS_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindDirectionsAvg.csv")

    EXCLUDE_BORSSELE_ALPHA = True  # After analysis, this station does not cover the full time range

    # %%
    windspeed_df = load_timeseries_data(WINDSPEEDS_DATA_PATH)
    windspeed_df = windspeed_df.drop(columns=["Europoint"])  # Afer analysis, geographically redundant with Borssele Alpha and further away
    windspeed_df = windspeed_df.set_index("DateTime (UTC)")
    stations = windspeed_df.columns

    print(f"Loaded wind speed data from {WINDSPEEDS_DATA_PATH}")
    print(f"\tStations: {stations}")
    print(f"\tData sample:\n\t{windspeed_df.head()}")

    if EXCLUDE_BORSSELE_ALPHA:
        # After investigation, Borssele Alpha does not give data for the full time range (2015-2024), so to be excluded from the average
        windspeed_df = windspeed_df.drop(columns=["Borssele Alpha"])
        stations = windspeed_df.columns
        print(f"\nExcluding Borssele Alpha from average wind speed calculation. Remaining stations: {stations}")

    #%%
    plot_series(windspeed_df, stations, path.join(VISUALISATION_PATH, "WindSpeeds_OverTime.png"), quantity="Wind Speed", unit="m/s")
    plot_series_with_zooms(windspeed_df, stations, path.join(VISUALISATION_PATH, "WindSpeeds_OverTime_Zoomed.png"), quantity="Wind Speed", unit="m/s", ymax=30)

    # CONCLUSION: Seems to show all stations give very similar wind speeds in time

    #%%
    C = correlation_matrix(windspeed_df, stations)
    C_off_diag = C.to_numpy()[~np.eye(C.shape[0], dtype=bool)]  # Off-diagonal elements (strictly speaking, double info)
    print(f"\nPearson correlation coefficient matrix between wind speed stations:\n{C}")
    print(f"\tMean off-diagonal Pearson correlation coefficient: {np.nanmean(C_off_diag):.4f}")
    print(f"\tMin off-diagonal Pearson correlation coefficient: {np.nanmin(C_off_diag):.4f}")

    deviation_stats = station_deviation_stats(windspeed_df)
    print("\nStation deviation statistics:")
    print(deviation_stats)

    # CONCLUSION: Justifies using an average wind speed over all stations as a representative wind speed time series.

    #%%
    windspeed_avg = windspeed_df.mean(axis=1)
    windspeed_df = windspeed_df.assign(AverageWindSpeed=windspeed_avg)

    print("\nSample of representative wind speed time series (averaged over stations):")
    print(windspeed_df.head())
    print(f"\n\tTotal samples: {len(windspeed_df)}")
    print(f"\tValid samples (not NaN): {windspeed_df['AverageWindSpeed'].notna().sum()}")

    windspeed_df["AverageWindSpeed"].to_csv(path.join(_FILE_PATH, "Data", "Weather", "WindSpeedsAvg.csv"), index=True, sep=";")

    #%%
    plot_windspeed_histogram(windspeed_df, path.join(VISUALISATION_PATH, "WindSpeeds_Histogram.png"))

    #%%
    winddirections_df = load_timeseries_data(AVG_WINDDIRECTIONS_DATA_PATH)
    winddirections_df.set_index("DateTime (UTC)", inplace=True)
    windspeed_df = windspeed_df.assign(AverageWindDirection=winddirections_df["AverageWindDirection"])

    plot_windrose_with_speed(windspeed_df, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_WindRose_AreaProportional.png"))
    plot_windrose_with_speed(windspeed_df, area_proportional=False, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_WindRose_HeightProportional.png"))

    #%%
    plot_daily_profile(windspeed_df, "AverageWindSpeed", include_ci=True, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_DailyProfile.png"), ylabel="Wind Speed [m/s]", title="Typical Daily Wind Speed Profile")
    plot_yearly_profile(windspeed_df, "AverageWindSpeed", include_raw=True, include_ci=True, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_YearlyProfile.png"), ylabel="Wind Speed [m/s]", title="Typical Yearly Wind Speed Profile")

    # %%
    # Check seasonality (same as for production data in InvestigateProductionData.py)
    test_seasonality(windspeed_df, "AverageWindSpeed", freq="daily")
    test_seasonality(windspeed_df, "AverageWindSpeed", freq="yearly")

    df_tmp = windspeed_df.copy()
    pattern_daily, f_test_daily, model_daily = test_seasonality(windspeed_df, "AverageWindSpeed", freq="daily")
    df_tmp["AverageWindSpeed_daily_removed"] = windspeed_df["AverageWindSpeed"] - windspeed_df.index.hour.map(pattern_daily)
    pattern_yearly, f_test_yearly, model_yearly = test_seasonality(df_tmp, "AverageWindSpeed_daily_removed", freq="yearly")

    # Daily
    md = model_daily
    fd = f_test_daily
    r2_d = md.rsquared
    f_d = float(fd.fvalue)
    p_d = float(fd.pvalue)
    c_d = "seasonality" if p_d < 0.05 else "no seasonality"

    # Yearly
    my = model_yearly
    fy = f_test_yearly
    r2_y = my.rsquared
    f_y = float(fy.fvalue)
    p_y = float(fy.pvalue)
    c_y = "seasonality" if p_y < 0.05 else "no seasonality"

    rows = ["AverageWindSpeed", r2_d, f_d, p_d, c_d, r2_y, f_y, p_y, c_y]

    summary = pd.DataFrame([rows], columns=["Wind Farm", "Daily R²", "Daily F", "Daily p", "Daily Conclusion", "Yearly R²", "Yearly F", "Yearly p", "Yearly Conclusion"])
    print(f"\nSeasonality Test Summary for Wind Speeds:")
    print(summary)

    df_windspeed_sadjusted = remove_given_seasonalities(windspeed_df, ["AverageWindSpeed"], {"AverageWindSpeed": pattern_daily}, {"AverageWindSpeed": pattern_yearly})

    plot_daily_profile(df_windspeed_sadjusted, "AverageWindSpeed_sadjusted", include_ci=True, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_DailyProfile_SeasonalityRemoved.png"), ylabel="Wind Speed [m/s]", title="Typical Daily Wind Speed Profile (Seasonality Removed)")
    plot_yearly_profile(df_windspeed_sadjusted, "AverageWindSpeed_sadjusted", include_raw=True, include_ci=True, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_YearlyProfile_SeasonalityRemoved.png"), ylabel="Wind Speed [m/s]", title="Typical Yearly Wind Speed Profile (Seasonality Removed)", ylim=(0, 12))

    #%%
    # Save
    df_windspeed_sadjusted[["AverageWindSpeed", "AverageWindSpeed_sadjusted", "AverageWindSpeed_daily_season", "AverageWindSpeed_yearly_season"]].to_csv(path.join(_FILE_PATH, "Data", "Weather", "WindSpeedsAvg_SeasonalityRemoved.csv"), index=True, sep=";")

