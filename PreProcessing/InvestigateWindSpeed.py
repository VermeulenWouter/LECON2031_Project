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

from InvestigateWindDirection import load_timeseries_data, plot_wind, plot_wind_with_zooms
from Helper import *

def correlation_matrix(df, stations):
    C = pd.DataFrame(index=stations, columns=stations, dtype=float)
    for i, station_i in enumerate(stations):
        for j, station_j in enumerate(stations):
            if i <= j:
                corr_ij = df[station_i].corr(df[station_j])
                C.at[station_i, station_j] = corr_ij
                C.at[station_j, station_i] = corr_ij
    return C


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
    edgecolor = "white"

    directions_deg = pd.to_numeric(df["AverageWindDirection"], errors="coerce")
    valid_dir = directions_deg.dropna()
    np.deg2rad(valid_dir)

    speeds = pd.to_numeric(df["AverageWindSpeed"], errors="coerce")
    sub = pd.DataFrame({"dir": directions_deg, "spd": speeds}).dropna()
    dirs = np.deg2rad(sub["dir"])
    spd = sub["spd"]

    # Adaptive speed bins
    speed_bins = _make_speed_bins(spd)
    speed_labels = [f"{speed_bins[i]:.1f}–{speed_bins[i+1]:.1f}" for i in range(len(speed_bins)-1)]
    cmap = plt.get_cmap("viridis")
    speed_colors = [cmap(i / (len(speed_bins)-2)) for i in range(len(speed_bins)-1)]

    H, dir_edges, spd_edges = np.histogram2d(dirs, spd, bins=[bins, speed_bins])

    if area_proportional:
        R = np.sqrt(H)
    else:
        R = H

    widths = np.diff(dir_edges)
    gap = widths * 0.05
    widths = widths - gap

    plt.close("all")
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")

    # Stacked bars
    bottom = np.zeros(bins)
    for i in range(len(speed_bins) - 1):
        radii = R[:, i]
        ax.bar(dir_edges[:-1]+widths/2, radii, width=widths, bottom=bottom, align="center", color=speed_colors[i], edgecolor=edgecolor, linewidth=0.8, label=speed_labels[i])
        bottom += radii

    # Radial labels (percentages)
    total = H.sum()
    if area_proportional:
        max_r = bottom.max()
        tick_r = np.linspace(0, max_r, 5)
        tick_percent = (tick_r**2) / total * 100
        ax.set_rticks(tick_r)
        ax.set_yticklabels([f"{p:.1f}%" for p in tick_percent])

    # Cardinal directions
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["N","NE","E","SE","S","SW","W","NW"])

    ax.set_title("Wind Rose with Speed Distribution")
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.5))

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved wind rose to {filepath}")
    else:
        plt.show()


def plot_daily_profile(df, include_ci=True, ci_level=0.95, filepath=None):
    fig, ax = plt.subplots(figsize=(12, 4))

    df = df.copy()
    df["hour"] = df["DateTime (UTC)"].dt.hour
    grouped = df.groupby("hour")["AverageWindSpeed"]
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
    ax.set_ylabel("Wind Speed [m/s]")
    ax.set_xticks(range(24))
    ax.grid(True)
    ax.legend()
    ax.set_title("Typical Daily Wind Speed Profile")

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved daily profile plot to {filepath}")
    else:
        plt.show()


def plot_yearly_profile(df, include_raw=True, include_ci=True, ci_level=0.95, filepath=None):
    fig, ax = plt.subplots(figsize=(14, 4))

    df = df.copy()
    df["date"] = df["DateTime (UTC)"].dt.date
    df["doy"] = df["DateTime (UTC)"].dt.dayofyear
    daily = df.groupby(["date", "doy"])["AverageWindSpeed"].mean().reset_index()
    grouped = daily.groupby("doy")["AverageWindSpeed"]
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
    ax.set_ylabel("Wind Speed [m/s]")
    ax.set_xlim(1, 366)
    ax.grid(True)
    ax.legend()
    ax.set_title("Typical Yearly Wind Speed Profile")

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved yearly profile plot to {filepath}")
    else:
        plt.show()


#%%
if __name__ == "__main__":

    # %%
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    WINDSPEEDS_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindSpeeds.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "WindPreprocessing")

    AVG_WINDDIRECTIONS_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindDirectionsAvg.csv")

    # %%
    windspeed_df = load_timeseries_data(WINDSPEEDS_DATA_PATH)
    windspeed_df = windspeed_df.drop(columns=["Europoint"])  # Afer analysis, geographically redundant with Borssele Alpha and further away
    stations = [col for col in windspeed_df.columns if col != 'DateTime (UTC)']

    print(f"Loaded wind speed data from {WINDSPEEDS_DATA_PATH}")
    print(f"\tStations: {stations}")
    print(f"\tData sample:\n\t{windspeed_df.head()}")

    #%%
    plot_wind(windspeed_df, stations, path.join(VISUALISATION_PATH, "WindSpeeds_OverTime.png"), quantity="Wind Speed", unit="m/s")
    plot_wind_with_zooms(windspeed_df, stations, path.join(VISUALISATION_PATH, "WindSpeeds_OverTime_Zoomed.png"), quantity="Wind Speed", unit="m/s", ymax=30)

    # CONCLUSION: Seems to show all stations give very similar wind speeds in time

    #%%
    C = correlation_matrix(windspeed_df, stations)
    C_off_diag = C.to_numpy()[~np.eye(C.shape[0], dtype=bool)]  # Off-diagonal elements (strictly speaking, double info)
    print(f"\nPearson correlation coefficient matrix between wind speed stations:\n{C}")
    print(f"\tMean off-diagonal Pearson correlation coefficient: {np.nanmean(C_off_diag):.4f}")
    print(f"\tMin off-diagonal Pearson correlation coefficient: {np.nanmin(C_off_diag):.4f}")

    windspeed_df_notime = windspeed_df.copy()
    windspeed_df_notime = windspeed_df_notime.drop(columns=['DateTime (UTC)'])
    deviation_stats = station_deviation_stats(windspeed_df_notime)
    print("\nStation deviation statistics:")
    print(deviation_stats)

    # CONCLUSION: Justifies using an average wind speed over all stations as a representative wind speed time series.

    #%%
    windspeed_avg = windspeed_df_notime.mean(axis=1)
    windspeed_df = windspeed_df.assign(AverageWindSpeed=windspeed_avg)

    print("\nSample of representative wind speed time series (averaged over stations):")
    print(windspeed_df.head())
    print(f"\n\tTotal samples: {len(windspeed_df)}")
    print(f"\tValid samples (not NaN): {windspeed_df['AverageWindSpeed'].notna().sum()}")

    windspeed_df[["DateTime (UTC)", "AverageWindSpeed"]].to_csv(path.join(_FILE_PATH, "Data", "Weather", "WindSpeedsAvg.csv"), index=False, sep=";")

    #%%
    plot_windspeed_histogram(windspeed_df, path.join(VISUALISATION_PATH, "WindSpeeds_Histogram.png"))

    #%%
    winddirections_df = load_timeseries_data(AVG_WINDDIRECTIONS_DATA_PATH)
    windspeed_df = windspeed_df.assign(AverageWindDirection=winddirections_df["AverageWindDirection"])

    plot_windrose_with_speed(windspeed_df, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_WindRose_AreaProportional.png"))
    plot_windrose_with_speed(windspeed_df, area_proportional=False, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_WindRose_HeightProportional.png"))

    #%%
    plot_daily_profile(windspeed_df, include_ci=True, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_DailyProfile.png"))
    plot_yearly_profile(windspeed_df, include_raw=True, include_ci=True, filepath=path.join(VISUALISATION_PATH, "WindSpeeds_YearlyProfile.png"))
