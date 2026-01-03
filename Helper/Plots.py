"""
Helper functions for plotting wind data and time series.

@author: Wouter Vermeulen
@data: 2026-01-03 (note: refactor from earlier code in InvestigateWindDirection.py)
"""

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def plot_series(df: pd.DataFrame, cols: list[str], filepath: str = None, quantity: str | None = None, unit: str | None = None):
    """
    Plot one or multiple timeseries.

    :param df: dataframe with datetime index
    :param cols: column names existing in the df to plot
    :param filepath: filepath to save the figure, if None the figure is shown instead
    :param quantity: the quantity being plotted (used in title and y-label)
    :param unit: the unit of the quantity (used in y-label)
    """
    plt.figure(figsize=(12, 6))
    for col in cols:
        plt.plot(df.index, df[col], label=f'{col}', marker="o", markersize=2, linestyle=None)
    plt.xlabel("DateTime (UTC)")

    if quantity is not None:
        plt.title(f"{quantity} Over Time")
        if unit is not None:
            plt.ylabel(f"{quantity} [{unit}]")
        else:
            plt.ylabel(f"{quantity}")

    plt.legend()
    plt.grid()
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved wind plot to {filepath}")
    else:
        plt.show()


def plot_series_with_zooms(df, cols: list[str], filepath: str | None = None, quantity: str | None = None, unit: str | None = None, ymax: float | None = None):
    """
    Plot one or multiple timeseries and make zooms at 20%, 50% and 80% of 30 days of data.

    :param df: dataframe with datetime index
    :param cols: column names existing in the df to plot
    :param filepath: filepath to save the figure, if None the figure is shown instead
    :param quantity: the quantity being plotted (used in title and y-label)
    :param unit: the unit of the quantity (used in y-label)
    :param ymax: the maximum y-value for the zoomed plots, if None the y-limits are automatic
    """

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1])
    ax_main = fig.add_subplot(gs[0, :])

    for col in cols:
        ax_main.plot(df.index, df[col], marker="o", markersize=2, linestyle=None, label=col)

    if quantity is not None:
        ax_main.set_title(f"{quantity} Over Time")
        if unit is not None:
            ax_main.set_ylabel(f"{quantity} [{unit}]")
        else:
            ax_main.set_ylabel(f"{quantity}")
    ax_main.grid(True)
    ax_main.legend(loc="upper right")

    def _add_zoom_subplot(row, center_idx, width_days=30):
        ax_zoom = fig.add_subplot(gs[row, :])
        center_time = df.index[center_idx]
        tmin = center_time - pd.Timedelta(days=width_days/2)
        tmax = center_time + pd.Timedelta(days=width_days/2)

        for station in cols:
            ax_zoom.plot(df.index, df[station], marker="o", markersize=2, linestyle=None)

        ax_zoom.set_xlim(tmin, tmax)
        if ymax is not None:
            ax_zoom.set_ylim(0, ymax)
            rect = Rectangle((tmin, 0), tmax - tmin, ymax, linewidth=5, edgecolor="red", facecolor="none")
            ax_main.add_patch(rect)

        ax_zoom.xaxis.set_minor_locator(mdates.DayLocator())
        ax_zoom.grid(True, axis='x', which='minor')
        ax_zoom.grid(True)
        if unit is not None:
            ax_zoom.set_ylabel(unit)

        ax_main.add_line(Line2D([tmin, tmin], [0, ax_zoom.get_ylim()[1]], transform=ax_main.transData, color="red", linewidth=0.8))
        ax_main.add_line(Line2D([tmax, tmax], [0, ax_zoom.get_ylim()[1]], transform=ax_main.transData, color="red", linewidth=0.8))

    n = len(df)
    _add_zoom_subplot(1, int(n * 0.20))
    _add_zoom_subplot(2, int(n * 0.50))
    _add_zoom_subplot(3, int(n * 0.80))

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
        print(f"Saved wind plot with zooms to {filepath}")
    else:
        plt.show()