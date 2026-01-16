"""
Plot the geography of wind farms and weather stations using GeoPandas and Matplotlib.
Includes options to customize the visualization.

@author: Wouter Vermeulen
@date: 2025-12-29
"""

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import pandas as pd
from shapely.geometry import box, Point


# %%

_FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
FARMS_SHAPEFILE_PATH = path.join(_FILE_PATH, "Data", "Geography", "OffshoreWindZones.shp")
WEATHER_STATIONS_CSV_PATH = path.join(_FILE_PATH, "Data", "Geography", "WeatherStationCoordinates.csv")
BOUNDARIES_SHAPEFILE_PATH = path.join(_FILE_PATH, "Data", "Geography", "EEZBoundaries.shp")


#%%

# Default colors and styles for zones
colors = {"foreign": {"Production": "#1f77b4", "Construction": "#ff7f0e", "Planned": "#2ca02c", "Approved": "#d62728"},
          "Mermaid": "#98df8a",
          "Northwester 2": "#c5b0d5",
          "Nobelwind": "#ffbb78",
          "Belwind": "#ff9896",
          "Seastar": "#9467bd",
          "Northwind": "#8c564b",
          "Rentel": "#e377c2",
          "Thorntonbank SW": "#7f7f7f",
          "Thorntonbank NE": "#bcbd22",
          "Norther": "#17becf",
          "Princess Elisabeth 1": "#393b79",
          "Princess Elisabeth 2": "#5254a3",
          "Princess Elisabeth 3": "#6b6ecf",
          }

zone_styles = {'Mermaid': {"Display": "Mermaid", "Status": "Production", "color": colors["Mermaid"]},
         'Northwester 2': {"Display": "Northwester 2", "Status": "Production", "color": colors["Northwester 2"]},
         'Nobelwind': {"Display": None, "Status": None, "color": None},
         'Belwind phase 2 (Nobelwind) (Zone 1)': {"Display": "Nobelwind", "Status": "Production", "color": colors["Nobelwind"]},
         'Belwind phase 2 (Nobelwind) (Zone 2)': {"Display": "", "Status": "Production", "color": colors["Nobelwind"]},
         'Belwind phase 1': {"Display": "Belwind", "Status": "Production", "color": colors["Belwind"]},
         'Belwind I & II': {"Display": None, "Status": None, "color": None},
         'Seamade (SeaStar)': {"Display": "Seastar", "Status": "Production", "color": colors["Seastar"]},
         'Northwind': {"Display": "Northwind", "Status": "Production", "color": colors["Northwind"]},
         'Rentel': {"Display": "Rentel", "Status": "Production", "color": colors["Rentel"]},
         'Thorntonbank Part 1': {"Display": "Thorntonbank SW", "Status": "Production", "color": colors["Thorntonbank SW"]},
         'Thorntonbank Part 2': {"Display": "Thorntonbank NE", "Status": "Production", "color": colors["Thorntonbank NE"]},
         'C-Power': {"Display": None, "Status": None, "color": None},
         'Norther': {"Display": "Norther", "Status": "Production", "color": colors["Norther"]},
         'Princess Elisabeth Zone Lot 1': {"Display": "Princess Elisabeth 1", "Status": "Planned", "color": colors["Princess Elisabeth 1"]},
         'Princess Elisabeth Zone Lot 2.1': {"Display": "Princess Elisabeth 2", "Status": "Planned", "color": colors["Princess Elisabeth 2"]},
         'Princess Elisabeth Zone Lot 2.2': {"Display": "", "Status": "Planned", "color": colors["Princess Elisabeth 2"]},
         'Princess Elisabeth Zone Lot 3': {"Display": "Princess Elisabeth 3", "Status": "Planned", "color": colors["Princess Elisabeth 3"]},

         'Borssele Kavel I': {"Display": None, "Status": None, "color": None},
         'Borssele Kavel II': {"Display": "Borssele II", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Borssele Kavel III': {"Display": "Borssele III", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Borssele Kavel IV': {"Display": "Borssele IV", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Borssele Kavel V': {"Display": "Borssele V", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Windenergiegebied Borssele noordzijde': {"Display": None, "Status": None, "color": None},
         'Windenergiegebied Borssele zuidzijde': {"Display": None, "Status": None, "color": None},
         'Borssele':  {"Display": None, "Status": None, "color": None},
         'Borssele I': {"Display": "Borssele I", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Hollandse Kust (Zuid)': {"Display": None, "Status": None, "color": None},
         'HKZ Kavel IV': {"Display": "", "Status": "Production", "color": colors["foreign"]["Production"]},
         'HKZ Kavel III': {"Display": "", "Status": "Production", "color": colors["foreign"]["Production"]},
         'HKZ Kavel II': {"Display": "Hollandse Kust Zuid", "Status": "Production", "color": colors["foreign"]["Production"]},
         'HKZ Kavel I': {"Display": "", "Status": "Production", "color": colors["foreign"]["Production"]},
         'WP Q10 / Eneco Luchterduinen': {"Display": "Eneco Luchterduinen", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Hollandse Kust (West)': {"Display": "Hollandse Kust West", "Status": "Construction", "color": colors["foreign"]["Construction"]},
         'Hollandse Kust west zuidelijk deel (HK-w-z)': {"Display": None, "Status": None, "color": None},
         'Hollandse Kust D': {"Display": None, "Status": None, "color": None},
         'Hollandse Kust F': {"Display": None, "Status": None, "color": None},

         'London Array': {"Display": "London Array", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Thanet': {"Display": "Thanet", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Gunfleet Sands Demo': {"Display": "Gunfleet Sans III", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Gunfleet Sands I': {"Display": "Gunfleet Sans I", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Gunfleet Sands II': {"Display": "Gunfleet Sans II", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Kentish Flats': {"Display": "Kentish Flats", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Kentish Flats Extension': {"Display": "Kentish Flats Extension", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Five Estuaries': {"Display": "Five Estuaries", "Status": "Planned", "color": colors["foreign"]["Planned"]},
         'North Falls': {"Display": "North Falls", "Status": "Planned", "color": colors["foreign"]["Planned"]},
         'Greater Gabbard': {"Display": "Greater Gabbard", "Status": "Production", "color": colors["foreign"]["Production"]},
         'Galloper': {"Display": "Galloper", "Status": "Production", "color": colors["foreign"]["Production"]},
         'East Anglia One North': {"Display": "East Anglia One", "Status": "Planned", "color": colors["foreign"]["Planned"]},
         'East Anglia One': {"Display": "East Anglia One", "Status": "Production", "color": colors["foreign"]["Production"]},
         'East Anglia Two': {"Display": "East Anglia Two", "Status": "Planned", "color": colors["foreign"]["Planned"]},
         'East Anglia Three': {"Display": "East Anglia Three", "Status": "Construction", "color": colors["foreign"]["Construction"]},
         'East Anglia Southern Met Mast (1B)': {"Display": None, "Status": None, "color": None},
         'Dunkerque': {"Display": "Dunkerque", "Status": "Approved", "color": colors["foreign"]["Approved"]},
}


def is_center(name: str) -> bool:
    """The farms under research here are 'center' farms."""
    return zone_styles[name]["color"] in colors.values()


def plot_geography(
    include_center_wind_farm_zones=True,
    show_center_wind_farm_zone_labels=True,
    color_center_wind_farm_zones=None,
    include_other_wind_farm_zones=True,
    show_other_wind_farm_zone_labels=False,
    include_other_wind_farm_zones_under_construction=False,
    include_weather_stations=True,
    show_weather_station_names=False,
    include_basemap=True,
    include_seaborders=True,
    include_grid=True,
    zoom_extent=100,
    zoom_center=(51.6, 2.9),
    filepath: str | None = None
):
    """
    Plot the geography of wind farms and weather stations; using Mercator projection. A lot of options allow
    to finetune the visualization, see parameter descriptions.

    :param include_center_wind_farm_zones: Whether to include center wind farm zones.
    :param show_center_wind_farm_zone_labels: Whether to show labels for center wind farm zones.
    :param color_center_wind_farm_zones: Optional dictionary mapping zone names to colors. Must include all zones if
        provided.
    :param include_other_wind_farm_zones: Whether to include other wind farm zones.
    :param show_other_wind_farm_zone_labels: Whether to show labels for other wind farm zones.
    :param include_other_wind_farm_zones_under_construction: Whether to include other wind farm zones under construction.
    :param include_weather_stations: Whether to include weather stations.
    :param show_weather_station_names: Whether to show weather station names.
    :param include_basemap: Whether to include a basemap
    :param include_seaborders: Whether to include seaborders
    :param include_grid: Whether to include grid lines
    :param zoom_extent: Zoom extent in kilometers, half-width/height from center.
    :param zoom_center: Tuple of (latitude, longitude) in WGS84 for the center of the zoom
    :param filepath: Optional path to save the plot as a PNG file.
    """
    # Load all data
    zones = gpd.read_file(FARMS_SHAPEFILE_PATH).to_crs(4326)
    stations_df = pd.read_csv(WEATHER_STATIONS_CSV_PATH)
    stations = gpd.GeoDataFrame(stations_df, geometry=[Point(xy) for xy in zip(stations_df.Longitude, stations_df.Latitude)], crs=4326)
    boundaries = gpd.read_file(BOUNDARIES_SHAPEFILE_PATH).to_crs(4326)

    zones = zones.to_crs(3857)
    stations = stations.to_crs(3857)
    boundaries = boundaries.to_crs(3857)

    cx, cy = zoom_center[1], zoom_center[0]
    cx, cy = gpd.GeoSeries([Point(cx, cy)], crs=4326).to_crs(3857).iloc[0].coords[0]
    d = zoom_extent * 1000
    clip_box = box(cx - d, cy - d, cx + d, cy + d)

    zones = zones.clip(clip_box)
    stations = stations.clip(clip_box)
    boundaries = boundaries.clip(clip_box)

    zones["color"] = zones["name"].map(lambda x: zone_styles.get(x, {}).get("color"))
    zones["display"] = zones["name"].map(lambda x: zone_styles.get(x, {}).get("Display"))
    zones["status"] = zones["name"].map(lambda x: zone_styles.get(x, {}).get("Status"))
    zones["homeland"] = zones["name"].map(is_center)

    center = zones[(zones["homeland"] == True) & (zones["status"] == "Production") & zones["display"].notna()]
    other = zones[(zones["homeland"] == False) & zones["display"].notna()]

    if not include_other_wind_farm_zones_under_construction:
        other = other[other["status"] != "Construction"]

    if color_center_wind_farm_zones:
        for name, col in color_center_wind_farm_zones.items():
            center.loc[center["name"] == name, "color"] = col

    fig, ax = plt.subplots(figsize=(12, 12))

    if include_center_wind_farm_zones:
        center.plot(ax=ax, color=center["color"], edgecolor="black", linewidth=0.8, zorder=3)
        if show_center_wind_farm_zone_labels:
            for _, r in center.iterrows():
                p = r.geometry.representative_point()
                ax.text(p.x, p.y, r["display"], fontsize=8, ha="center", va="center")

    if include_other_wind_farm_zones:
        other.plot(ax=ax, color=other["color"], edgecolor="black", linewidth=0.8, alpha=0.8, zorder=2)
        if show_other_wind_farm_zone_labels:
            for _, r in other.iterrows():
                p = r.geometry.representative_point()
                ax.text(p.x, p.y, r["display"], fontsize=8, ha="center", va="center")

    if include_weather_stations:
        stations.plot(ax=ax, color="red", markersize=50, zorder=4)

    if include_weather_stations and show_weather_station_names:
        for _, r in stations.iterrows():
            p = r.geometry
            ax.text(p.x, p.y, r["Station"], fontsize=8, ha="left", va="bottom")


    if include_seaborders:
        boundaries.plot(ax=ax, color="black", linewidth=1, zorder=5)

    # Style axes
    cx, cy = gpd.GeoSeries([Point(zoom_center[1], zoom_center[0])], crs=4326).to_crs(3857).iloc[0].coords[0]
    d = zoom_extent * 1000
    ax.set_xlim(cx - d, cx + d)
    ax.set_ylim(cy - d, cy + d)

    tick_deg = 0.25
    box_wgs = gpd.GeoSeries([box(cx - d, cy - d, cx + d, cy + d)], crs=3857).to_crs(4326).total_bounds
    minlon, minlat, maxlon, maxlat = box_wgs

    lon_ticks = np.arange(np.floor(minlon), np.ceil(maxlon), tick_deg)
    lat_ticks = np.arange(np.floor(minlat), np.ceil(maxlat), tick_deg)

    lon_ticks_3857 = gpd.GeoSeries([Point(lon, zoom_center[0]) for lon in lon_ticks], crs=4326).to_crs(3857).x
    lat_ticks_3857 = gpd.GeoSeries([Point(zoom_center[1], lat) for lat in lat_ticks], crs=4326).to_crs(3857).y

    ax.set_xticks(lon_ticks_3857)
    ax.set_yticks(lat_ticks_3857)
    ax.set_xticklabels([f"{lon:.2f}°E" for lon in lon_ticks])
    ax.set_yticklabels([f"{lat:.2f}°N" for lat in lat_ticks])

    if include_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Force plot extent to match clipping
    cx, cy = gpd.GeoSeries([Point(zoom_center[1], zoom_center[0])], crs=4326).to_crs(3857).iloc[0].coords[0]
    d = zoom_extent * 1000
    ax.set_xlim(cx - d, cx + d)
    ax.set_ylim(cy - d, cy + d)

    if include_basemap:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels)

    # North arrow
    ax.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.28), xycoords='axes fraction', textcoords='axes fraction',
                ha='center', va='center', fontsize=14, fontweight='bold',
                arrowprops=dict(arrowstyle='<|-', linewidth=2))

    # Scale bar
    scale_len_km = 20
    scale_len_m = scale_len_km * 1000
    cx, cy = gpd.GeoSeries([Point(zoom_center[1], zoom_center[0])], crs=4326).to_crs(3857).iloc[0].coords[0]
    x0 = cx - scale_len_m / 2
    x1 = cx + scale_len_m / 2
    y = cy - zoom_extent * 1000 * 0.9
    ax.plot([x0, x1], [y, y], color='black', linewidth=3)
    ax.text(cx, y - scale_len_m * 0.05, f'{scale_len_km} km', ha='center', va='top', fontsize=10)



    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Test the function
    plot_geography(filepath=path.join(_FILE_PATH, "Visualisations", "geography_plot_base.png"), zoom_extent=50)
