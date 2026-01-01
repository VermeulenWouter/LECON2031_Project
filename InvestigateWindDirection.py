"""
Unionize wind direction data from multiple stations (after checking correlations between them are sufficient
to do so). Then analyze typical daily and yearly wind direction profiles.

@author: Wouter Vermeulen
@date: 2025-12-29
"""

from astropy.stats import circcorrcoef
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import os.path as path
import pandas as pd
import warnings


def filter_wind_directions(df):
    """Filter wind direction data: set 0 (no wind) and 990 (variable wind) to NaN."""
    return df.replace({0: np.nan, 990: np.nan})


def circular_nanmean(df: pd.DataFrame, return_radians: bool = False) -> pd.Series:
    """Compute the circular mean of a dataframe of angles in degrees. Ignore NaN values."""
    rad = np.deg2rad(df.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # Some slices might be empty
        sin_mean = np.nanmean(np.sin(rad), axis=1)
        cos_mean = np.nanmean(np.cos(rad), axis=1)
    mean_rad = (np.arctan2(sin_mean, cos_mean)+ 2*np.pi) % (2*np.pi)
    if return_radians:
        return mean_rad
    return np.rad2deg(mean_rad)


def circular_correlation_coefficient(angles1: pd.Series, angles2: pd.Series) -> float:
    """
    Compute the circular correlation coefficient between two series of angles in degrees.

    *Note: uses the formula from Jammalamadaka and Sengupta (2001).*
    """
    return circcorrcoef(np.deg2rad(filter_wind_directions(angles1)),
                        np.deg2rad(filter_wind_directions(angles2)))


def load_timeseries_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=['DateTime (UTC)'])
    df = df[df["DateTime (UTC)"] > pd.Timestamp("2014-12-31 23:59:59")]
    return df


def plot_wind(wind_df: pd.DataFrame, stations: list[str], filepath: str = None, quantity: str = "Wind Direction", unit: str = "°"):
    plt.figure(figsize=(12, 6))
    for station in stations:
        plt.plot(wind_df["DateTime (UTC)"], wind_df[station], label=f'{station}', marker="o", markersize=2, linestyle=None)
    plt.xlabel("DateTime (UTC)")
    plt.ylabel(f"{quantity} [{unit}]")
    plt.title(f"{quantity} Over Time")
    plt.legend()
    plt.grid()
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved wind plot to {filepath}")
    else:
        plt.show()


def plot_wind_with_zooms(df, stations: list[str], filepath=None, quantity: str = "Wind Direction", unit: str = "°", ymax: float = 360):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1])
    ax_main = fig.add_subplot(gs[0, :])

    for station in stations:
        ax_main.plot(df['DateTime (UTC)'], df[station], marker="o", markersize=2, linestyle=None, label=station)

    ax_main.set_title(f"{quantity} Over Time")
    ax_main.set_ylabel(f"{quantity} [{unit}]")
    ax_main.grid(True)
    ax_main.legend(loc="upper right")

    def add_zoom_subplot(row, center_idx, width_days=30):
        ax_zoom = fig.add_subplot(gs[row, :])
        center_time = df["DateTime (UTC)"].iloc[center_idx]
        tmin = center_time - pd.Timedelta(days=width_days/2)
        tmax = center_time + pd.Timedelta(days=width_days/2)

        for station in stations:
            ax_zoom.plot(df["DateTime (UTC)"], df[station], marker="o", markersize=2, linestyle=None)

        ax_zoom.set_xlim(tmin, tmax)
        ax_zoom.set_ylim(0, ymax)
        ax_zoom.grid(True)
        ax_zoom.set_ylabel(unit)

        rect = Rectangle((tmin, 0), tmax - tmin, ymax, linewidth=5, edgecolor="red", facecolor="none")
        ax_main.add_patch(rect)

        ax_main.add_line(Line2D([tmin, tmin], [0, ax_zoom.get_ylim()[1]], transform=ax_main.transData, color="red", linewidth=0.8))
        ax_main.add_line(Line2D([tmax, tmax], [0, ax_zoom.get_ylim()[1]], transform=ax_main.transData, color="red", linewidth=0.8))

    n = len(df)
    add_zoom_subplot(1, int(n * 0.20))
    add_zoom_subplot(2, int(n * 0.50))
    add_zoom_subplot(3, int(n * 0.80))

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved wind plot with zooms to {filepath}")
    else:
        plt.show()

def correlation_matrix_circular(df, stations: list[str], min_samples=30):
    """
    Calculate the pairwise circular correlation coefficient matrix between wind direction time series.

    directions: ndarray of length n_stations, each element is a ndarray of wind directions (in degrees) for that station
    returns: ndarray (n_stations, n_stations) of circular correlation coefficients (Jammalamadaka and Sengupta)
    """
    n = len(stations)
    C = np.full((n, n), np.nan)

    for i, s1 in enumerate(stations):
        for j, s2 in enumerate(stations):
            x = df[s1]
            y = df[s2]
            mask = x.notna() & y.notna()

            if mask.sum() >= min_samples:
                C[i, j] = circular_correlation_coefficient(x[mask], y[mask])

    return pd.DataFrame(C, index=stations, columns=stations)


def angular_diff_deg(a, b):
    diff = (a - b + 180) % 360 - 180
    return np.abs(diff)


def station_deviation_stats(df):
    """

    df: DataFrame, wind directions in degrees.

    Returns:
        per_time: deviation per station per time
        per_station_summary: mean + 95% deviation per station
    """

    rad = np.deg2rad(df.to_numpy())
    mean_rad = circular_nanmean(df, return_radians=True)[:, np.newaxis]

    # Compute deviation of each station from the mean direction
    diff_rad = np.angle(np.exp(1j * (rad - mean_rad)))
    diff_deg = np.abs(np.rad2deg(diff_rad))

    # Summary per station
    per_station_summary = pd.DataFrame({
        "mean_deviation": np.nanmean(diff_deg, axis=0),
        "p95_deviation": np.nanpercentile(diff_deg, 95, axis=0)
    }, index=df.columns)

    return per_station_summary


def plot_windrose(df, bins=36, color="C0", edgecolor="white", area_proportional=True, filepath=None, title="Wind Rose"):
    """Plot a wind rose from wind direction data.

    :param df: DataFrame with 'AverageWindDirection' column in degrees.
    :param bins: Number of direction bins (default 36 for 10° bins).
    :param color: Fill color for bars.
    :param edgecolor: Edge color for bars.
    :param area_proportional: If True, area of bars is proportional to frequency; if False, height is proportional.
    :param filepath: If provided, save the plot to this file path.
    :param title: Title of the plot.
    """
    directions_deg = pd.to_numeric(df['AverageWindDirection'], errors='coerce')
    directions = np.deg2rad(directions_deg.dropna())
    counts, edges = np.histogram(directions, bins=bins)

    widths = np.diff(edges)

    if area_proportional:
        radii = np.sqrt(counts)
    else:
        radii = counts

    gap = widths * 0.05
    widths = widths - gap
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.bar(edges[:-1]+widths/2, radii, width=widths, bottom=0, align='center', color=color, edgecolor=edgecolor, linewidth=0.8)

    # Fix radial labels to show correct percentages
    if area_proportional:
        max_r = radii.max()
        tick_r = np.linspace(0, max_r, 5)
        tick_percent = (tick_r**2) / counts.sum() * 100
        ax.set_rticks(tick_r)
        ax.set_yticklabels([f"{p:.1f}%" for p in tick_percent])

    # Cardinal directions
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["N","NE","E","SE","S","SW","W","NW"])

    ax.set_title(title)

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved windrose plot to {filepath}")
    else:
        plt.show()


def shift_angles(angles_deg, shift):
    return (angles_deg - shift) % 360


def circular_ci_analytical(angles_deg, ci=0.95):
    """
    Compute the confidence interval for circular data using an analytical approximation.

    Loosely based on https://github.com/tobiste/tectonicr/blob/main/R/statistics.R which
    in turn is based on Batschelet, E. (1971). Recent statistical methods for orientation data. "Animal Orientation, Symposium 1970 on Wallops Island". Amer. Inst. Biol. Sciences, Washington.
    """
    if len(angles_deg) < 5:
        return np.nan, np.nan

    rad = np.deg2rad(angles_deg)
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    R = np.sqrt(s**2 + c**2)
    n = len(rad)

    from scipy.stats import norm
    z = norm.ppf((1 + ci) / 2)

    if R == 0:
        return np.nan, np.nan

    arg = (R - z / np.sqrt(n)) / R
    arg = np.clip(arg, -1, 1)

    delta = np.arccos(arg)
    mean_angle = np.arctan2(s, c)

    lower = (mean_angle - delta) % (2*np.pi)
    upper = (mean_angle + delta) % (2*np.pi)

    return np.rad2deg(lower), np.rad2deg(upper)


def circular_profile_shifted(df, shift=50, smooth_window=7, timescale: str = "day"):
    """
    Compute the circular mean profile (daily or yearly) of wind directions, with smoothing and angle shifting.

    :param df: the DataFrame with 'DateTime (UTC)' and 'AverageWindDirection' columns
    :param shift: how much to shift the angles by (to avoid wrap-around issues)
    :param smooth_window: the window size for smoothing
    :param timescale: 'year' for day of year, 'day' for hour of day
    :return:
    """
    df2 = df.copy()
    timescale = "doy" if timescale == "year" else "hour" if timescale == "day" else None
    if timescale is None:
        raise ValueError("timescale must be 'year' or 'day'")

    if timescale == "hour":
        df2[timescale] = df2['DateTime (UTC)'].dt.hour
    else:
        df2[timescale] = df2['DateTime (UTC)'].dt.dayofyear

    sin_mean = df2.groupby(timescale)['AverageWindDirection'].apply(lambda x: np.mean(np.sin(np.deg2rad(x))))
    cos_mean = df2.groupby(timescale)['AverageWindDirection'].apply(lambda x: np.mean(np.cos(np.deg2rad(x))))

    mean_rad = np.arctan2(sin_mean, cos_mean)
    mean_deg = (np.rad2deg(mean_rad) + 360) % 360

    rad = np.deg2rad(mean_deg)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)

    # Pad for circular smoothing
    pad = smooth_window // 2
    sin_ext = np.concatenate([sin_vals[-pad:], sin_vals, sin_vals[:pad]])
    cos_ext = np.concatenate([cos_vals[-pad:], cos_vals, cos_vals[:pad]])
    sin_smooth = pd.Series(sin_ext).rolling(smooth_window, center=True, min_periods=1).mean().values
    cos_smooth = pd.Series(cos_ext).rolling(smooth_window, center=True, min_periods=1).mean().values
    sin_smooth = sin_smooth[pad:-pad]
    cos_smooth = cos_smooth[pad:-pad]

    smooth_rad = np.arctan2(sin_smooth, cos_smooth)
    smooth_deg = (np.rad2deg(smooth_rad) + 360) % 360

    # Apply shift
    mean_shifted = (mean_deg - shift) % 360
    smooth_shifted = (smooth_deg - shift) % 360

    return pd.DataFrame({
        timescale: mean_deg.index,
        'mean_shifted': mean_shifted,
        'smooth_shifted': smooth_shifted
    })


def plot_daily_profile(day_profile, full_df, shift=90, include_ci=True, ci_level=0.95, filepath=None):
    fig,ax=plt.subplots(figsize=(12,4))

    if include_ci:
        df2=full_df.copy()
        df2['hour']=df2['DateTime (UTC)'].dt.hour
        df2['shifted']=shift_angles(df2['AverageWindDirection'],shift)

        lower_ci=[0]
        upper_ci=[355]

        for hour in day_profile['hour']:
            subset=df2[df2['hour']==hour]['shifted']
            lo,hi=circular_ci_analytical(subset.values,ci=ci_level)

            if lo > hi:
                if angular_diff_deg(lo,lower_ci[-1]) > angular_diff_deg(hi,lower_ci[-1]):
                    hi+=360
                else:
                    lo-=360

            mean_val=day_profile.loc[day_profile['hour']==hour,'mean_shifted'].values[0]

            if mean_val > hi:
                lo+=360
                hi+=360
            elif mean_val < lo:
                lo-=360
                hi-=360

            lower_ci.append(lo)
            upper_ci.append(hi)

        lower_ci=np.array(lower_ci[1:])
        upper_ci=np.array(upper_ci[1:])
        hours=day_profile['hour'].values

        ax.fill_between(hours,lower_ci,upper_ci,color='C1',alpha=0.15,label=f'{int(ci_level*100)}% CI')
    ax.plot(day_profile['hour'],day_profile['mean_shifted'], label='Raw hourly mean', color='C0')

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Wind Direction [°] (shifted baseline)")

    ax.set_ylim(shift-10,shift+360+10)
    ax.set_xlim(0,23)

    ax.grid()

    ticks=np.arange(0,361,60)
    ax.set_yticks(ticks)
    ax.set_yticklabels((ticks+shift)%360)

    ax.legend()
    plt.title("Typical Daily Wind Direction Profile")
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved daily profile plot to {filepath}")
    else:
        plt.show()


def plot_yearly_profile(year_profile, full_df, shift=90, include_raw=True, include_ci=True, ci_level=0.95, filepath=None):
    fig, ax = plt.subplots(figsize=(12,4))

    if include_ci:
        df2 = full_df.copy()
        df2['doy'] = df2['DateTime (UTC)'].dt.dayofyear
        df2['shifted'] = shift_angles(df2['AverageWindDirection'], shift)

        lower_ci = [0]
        upper_ci = [0]

        for doy in year_profile['doy']:
            subset = df2[df2['doy'] == doy]['shifted']
            lo, hi = circular_ci_analytical(subset.values, ci=ci_level)

            if lower_ci[-1]:
                pass

            if lo > hi:
                # CI wraps around 0
                if angular_diff_deg(lo, lower_ci[-1]) > angular_diff_deg(hi, lower_ci[-1]):
                    hi += 360
                else:
                    lo -= 360


            # Ensure the mean lies inside the CI by shifting both bounds together
            mean_val = year_profile.loc[year_profile['doy'] == doy, 'mean_shifted'].values[0]

            # If mean is above the CI → shift CI up
            if mean_val > hi:
                lo += 360
                hi += 360

            # If mean is below the CI → shift CI down
            elif mean_val < lo:
                lo -= 360
                hi -= 360

            lower_ci.append(lo)
            upper_ci.append(hi)

        # Convert to arrays
        lower_ci = np.array(lower_ci[1:])
        upper_ci = np.array(upper_ci[1:])
        doy = year_profile['doy'].values

        # Plot CI band
        ax.fill_between(doy, lower_ci, upper_ci, color='C1', alpha=0.15, label=f'{int(ci_level * 100)}% CI')

    if include_raw:
        ax.plot(year_profile['doy'], year_profile['mean_shifted'], alpha=0.7, label='Raw daily mean', color='C0')

    ax.plot(year_profile['doy'], year_profile['smooth_shifted'], label='Smoothed circular mean', color='C1')

    # Axis formatting
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Wind Direction [°] (shifted baseline)")
    ax.set_ylim(shift-10, shift + 360+10)
    ax.set_xlim(1, 366)

    ax.grid()

    ticks = np.arange(0, 361, 60)
    ax.set_yticks(ticks)
    ax.set_yticklabels((ticks + shift) % 360)

    ax.legend()
    plt.title("Typical Yearly Wind Direction Profile")

    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved yearly profile plot to {filepath}")
    else:
        plt.show()


if __name__ == "__main__":

    # %%
    _FILE_PATH = path.dirname(__name__)
    WINDDIRECTIONS_DATA_PATH = path.join(_FILE_PATH, "Data", "Weather", "WindDirections.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "WindPreprocessing")

    #%%
    winddirection_df = load_timeseries_data(WINDDIRECTIONS_DATA_PATH)

    # Afer analysis, geographically redundant with Borssele Alpha and further away
    winddirection_df = winddirection_df.drop(columns=["Europoint"])

    winddirection_df_onlyreal = filter_wind_directions(winddirection_df)  # DataFrame with 0 and 990 replaced by NaN
    stations = [col for col in winddirection_df.columns if col != 'DateTime (UTC)']

    print(f"Loaded wind direction data from {WINDDIRECTIONS_DATA_PATH}")
    print(f"\tStations: {stations}")
    print(f"\tData sample:\n\t{winddirection_df_onlyreal.head()}")

    #%%
    plot_wind(winddirection_df_onlyreal, stations, path.join(VISUALISATION_PATH, "WindDirections_OverTime.png"))
    plot_wind_with_zooms(winddirection_df_onlyreal, stations, path.join(VISUALISATION_PATH, "WindDirections_OverTime_Zoomed.png"))

    # CONCLUSION: Seems to show all stations give very similar wind directions in time

    #%%
    C = correlation_matrix_circular(winddirection_df_onlyreal, stations)
    C_off_diag = C.to_numpy()[~np.eye(C.shape[0], dtype=bool)]  # Off-diagonal elements (strictly speaking, double info)
    print(f"\nCircular correlation coefficient matrix between wind direction stations:\n{C}")
    print(f"\tMean off-diagonal circular correlation coefficient: {np.nanmean(C_off_diag):.4f}")
    print(f"\tMin off-diagonal circular correlation coefficient: {np.nanmin(C_off_diag):.4f}")

    winddirection_df_onlyreal_notime = winddirection_df_onlyreal.copy()
    winddirection_df_onlyreal_notime = winddirection_df_onlyreal_notime.drop(columns=['DateTime (UTC)'])
    deviation_stats = station_deviation_stats(winddirection_df_onlyreal_notime)
    print("\nStation deviation statistics:")
    print(deviation_stats)

    # CONCLUSION: Justifies using an average wind direction over all stations as a representative wind direction time series.

    #%%
    winddirection_avg = circular_nanmean(winddirection_df_onlyreal_notime)
    winddirection_df_onlyreal = winddirection_df_onlyreal.assign(AverageWindDirection=winddirection_avg)

    print("\nSample of representative wind direction time series (averaged over stations):")
    print(winddirection_df_onlyreal.head())
    print(f"\n\tTotal samples: {len(winddirection_df_onlyreal)}")
    print(f"\tValid samples (not NaN): {winddirection_df_onlyreal['AverageWindDirection'].notna().sum()}")

    winddirection_df_onlyreal[["DateTime (UTC)", "AverageWindDirection"]].to_csv(path.join(_FILE_PATH, "Data", "Weather", "WindDirectionsAvg.csv"), index=False, sep=";")

    #%%
    plot_windrose(winddirection_df_onlyreal, filepath=path.join(VISUALISATION_PATH, "WindDirections_WindRose_AreaProportional.png"))
    plot_windrose(winddirection_df_onlyreal, area_proportional=False, filepath=path.join(VISUALISATION_PATH, "WindDirections_WindRose_HeightProportional.png"))

    #%%
    df_daily = circular_profile_shifted(winddirection_df_onlyreal, shift=60, smooth_window=5, timescale="day")
    plot_daily_profile(df_daily, winddirection_df_onlyreal, 60, filepath=path.join(VISUALISATION_PATH, "WindDirections_Daily_Profile.png"))

    #%%
    year_profile = circular_profile_shifted(winddirection_df_onlyreal, shift=60, smooth_window=30, timescale="year")
    plot_yearly_profile(year_profile, winddirection_df_onlyreal, include_raw=True, include_ci=True, shift=60, filepath=path.join(VISUALISATION_PATH, "WindDirections_Yearly_Profile.png"))
