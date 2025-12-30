"""
Process raw wind data from KNMI and Meetnet Vlaamse Banken (MVB) into structured CSV files.

@author: Wouter Vermeulen
@date: 2025-12-29
"""

import numpy as np
import os
import os.path as path
import pandas as pd
import re

#%%
_FILE_PATH = path.dirname(__name__)
RAWDATA_PATH = path.join(_FILE_PATH, "DataRaw/WindData")
DATA_PATH = path.join(_FILE_PATH, "Data")

# Debug settings
pd.set_option('display.max_columns', None)

# Skip steps for wind data extraction
SKIP_DATA_EXTRACTION = False

#%%

def _gather_mvb_variable_files(input_dir: str) -> tuple[dict[str, dict[str, list[str]]], int]:
    """Gathers MVB wind variable files from the input folder.

    Returns a dictionary mapping variable names to lists of file paths, and the total number of files found.
    """
    stationcodes = {
        "Wandelaar": "0",
        "Westhinder": "7"
    }
    data_variables = {
        "WVC": "WindSpeedAverage",
        "WC3": "WindGust",
        "WRS": "WindDirection"
    }

    pattern_folder = re.compile(r"^([A-Za-z ]+)\s(\d{4})$")
    pattern_file = re.compile(r"^MP(\d)\.([WCRSV3]{3})_.*\.txt$")
    data_files = {}  # {station1: {"WindDirection": [file, file, ...], "WindSpeed": [file, file, ...], "WindGust": [file, file, ...]}, ...}

    n_files = 0
    for dirname in os.listdir(input_dir):
        folder_path = path.join(input_dir, dirname)
        match = pattern_folder.match(dirname)
        if not match:
            continue

        station, _ = match.groups()
        if station not in data_files:
            data_files[station] = {"WindDirection": [], "WindSpeedAverage": [], "WindGust": []}

        for fname in os.listdir(folder_path):
            match = pattern_file.match(fname)
            if not match:
                continue

            file_path = path.join(folder_path, fname)
            stationcode, variable = match.groups()
            if stationcodes[station] != stationcode:
                print(
                    f"Warning: station code mismatch for station {station} in file {fname}. Expected {stationcodes[station]}, got {stationcode}.")

            data_files[station][data_variables[variable]].append(file_path)
            n_files += 1
    return data_files, n_files


def _load_mvb_variable_files(file_list, value_column_name) -> pd.DataFrame:
    """Load and concatenate multiple files of the same wind variable."""
    dfs = []

    for file_path in file_list:
        df = pd.read_csv(file_path, skiprows=1, sep="\t", names=["DateTime (UTC)", value_column_name])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def _hourly_aggregation_wind_dataframe(x: pd.Series) -> pd.Series:
    """Aggregate 10-minute wind data from Meetnet Vlaamse Banken into hourly data, following KNMI guidelines to ensure dataset remains
    coherent with KNMI datasets."""
    if len(x) != 6:
        return pd.Series(
            {
                "WindSpeedHourlyAverage": np.nan,
                "WindDirection": np.nan,
                "WindGust": np.nan
            })

    return pd.Series({
        "WindSpeedHourlyAverage": x["WindSpeedAverage"].mean() if x["WindSpeedAverage"].count() == 6 else np.nan,
        "WindDirection": x["WindDirection"].iloc[-1] if x.index[-1].minute == 50 else np.nan,
        "WindGust": x["WindGust"].max() if x["WindGust"].count() == 6 else np.nan
    })


def extract_wind_data():

    # KNMI wind data
    KNMI_header = ["Station", "Date", "Time", "WindDirection", "WindSpeedHourlyAverage", "WindSpeedTenMinAverage", "WindGust", "Temperature", "DewPointTemperature", "AirPressure", "HorizontalView", "CloudCover", "RelativeHumidity", "WeatherCode", "IndicaterWeatherCode", "Fog", "Rain", "Snow", "Thunder", "IceFormation", "SeaSurfaceTemperature"]
    useful_columns = ["Station", "DateTime (UTC)", "WindDirection", "WindSpeedHourlyAverage", "WindGust"]

    print("Processing KNMI wind data...")
    borssele_wind_data = pd.read_csv(path.join(RAWDATA_PATH, "KNMI", "2021-2025 Borssele alpha wind.txt"), skiprows=26, sep=",", names=KNMI_header)
    borssele_wind_data = borssele_wind_data.assign(Station = "Borssele Alpha")

    europoint_wind_data_1 = pd.read_csv(path.join(RAWDATA_PATH, "KNMI", "2011-2020 Europoint wind.txt"), skiprows=26, sep=",", names=KNMI_header)
    europoint_wind_data_2 = pd.read_csv(path.join(RAWDATA_PATH, "KNMI", "2021-2025 Europoint wind.txt"), skiprows=26, sep=",", names=KNMI_header)
    europoint_wind_data_1 = europoint_wind_data_1.assign(Station = "Europoint")
    europoint_wind_data_2 = europoint_wind_data_2.assign(Station = "Europoint")

    knmi_wind_data = pd.concat([borssele_wind_data, europoint_wind_data_1, europoint_wind_data_2], ignore_index=True)
    knmi_wind_data["DateTime (UTC)"] = pd.to_datetime(knmi_wind_data["Date"].astype(str) + " " + (knmi_wind_data["Time"] - 1).astype(str) + ":0", format="%Y%m%d %H:%M")
    knmi_wind_data = knmi_wind_data[useful_columns]
    knmi_wind_data["WindDirection"] = knmi_wind_data["WindDirection"].str.strip().replace("", pd.NA)
    knmi_wind_data["WindSpeedHourlyAverage"] = knmi_wind_data["WindSpeedHourlyAverage"].str.strip().replace("", pd.NA)
    knmi_wind_data["WindGust"] = knmi_wind_data["WindGust"].str.strip().replace("", pd.NA)
    knmi_wind_data["WindDirection"] = knmi_wind_data["WindDirection"].astype("Int64")
    knmi_wind_data["WindSpeedHourlyAverage"] = knmi_wind_data["WindSpeedHourlyAverage"].astype("Float64") / 10  # Convert to m/s
    knmi_wind_data["WindGust"] = knmi_wind_data["WindGust"].astype("Float64") / 10  # Convert to m/s

    # Meetnet Vlaamse Banken wind data
    mvb_files, _ = _gather_mvb_variable_files(input_dir=path.join(RAWDATA_PATH, "MVB"))

    station_dfs = []
    for station, variables in mvb_files.items():
        print(f"Processing MVB wind data for station {station}...")

        mvb_winddirection_data = _load_mvb_variable_files(variables["WindDirection"], "WindDirection")
        mvb_windspeed_data = _load_mvb_variable_files(variables["WindSpeedAverage"], "WindSpeedAverage")
        mvb_windgust_data = _load_mvb_variable_files(variables["WindGust"], "WindGust")

        mvb_data_station = mvb_winddirection_data.merge(mvb_windspeed_data, on="DateTime (UTC)", how="outer").merge(mvb_windgust_data, on="DateTime (UTC)", how="outer")

        # Reindex MVB data to ensure continuous 10-minute intervals
        mvb_data_station["DateTime (UTC)"] = pd.to_datetime(mvb_data_station["DateTime (UTC)"], format="%Y-%m-%dT%H:%M:%S+00:00")
        mvb_data_station = mvb_data_station.set_index("DateTime (UTC)").sort_index()
        mvb_data_station = mvb_data_station[~mvb_data_station.index.duplicated(keep="last")]  # Remove duplicate timestamps (20xx-01-01T00:00:00+00:00 is present twice due to double entry in files)
        full_index = pd.date_range(
            start=mvb_data_station.index.min().floor("h"),
            end=mvb_data_station.index.max().ceil("h") - pd.Timedelta(minutes=10),
            freq="10min"
        )
        mvb_data_station = mvb_data_station.reindex(full_index)
        mvb_data_station.index.name = "DateTime (UTC)"

        # Compute hourly averages for MVB data to align with KNMI data
        mvb_data_station = mvb_data_station.resample("1h").apply(_hourly_aggregation_wind_dataframe)
        mvb_data_station.reset_index(inplace=True)

        mvb_data_station = mvb_data_station.assign(Station = station)
        station_dfs.append(mvb_data_station)

    mvb_wind_data = pd.concat(station_dfs, ignore_index=True)

    # Combine KNMI and MVB wind data, and export to CSV files
    wind_data = pd.concat([knmi_wind_data, mvb_wind_data], ignore_index=True)
    winddirection_df = wind_data.pivot(index="DateTime (UTC)", columns="Station", values="WindDirection")
    windspeed_df = wind_data.pivot(index="DateTime (UTC)", columns="Station", values="WindSpeedHourlyAverage")
    windgust_df = wind_data.pivot(index="DateTime (UTC)", columns="Station", values="WindGust")
    winddirection_df.to_csv(path.join(DATA_PATH, "Weather", "WindDirections.csv"), sep=";")
    windspeed_df.to_csv(path.join(DATA_PATH, "Weather", "WindSpeeds.csv"), sep=";")
    windgust_df.to_csv(path.join(DATA_PATH, "Weather", "WindGusts.csv"), sep=";")


if not SKIP_DATA_EXTRACTION:
    extract_wind_data()

#%%
def extract_station_coordinates():
    """Extract station coordinates and save to JSON file."""

    # MVB stations directly filled from https://meetnetvlaamsebanken.be/map
    coords_WGS84 = {
        "Europoint": (),
        "Borssele Alpha": (),
        "Wandelaar": (51.39416667, 3.04555556),
        "Westhinder": (51.38833333, 2.43777778)
    }

    # KNMI stations
    KNMI_stations = pd.read_csv(path.join(RAWDATA_PATH, "KNMI", "AWS_stationsmetadata.csv"), sep=",")
    KNMI_convert_names = {"Europoint": "Europlatform", "Borssele Alpha": "Borssele Alpha"}
    for station in ["Europoint", "Borssele Alpha"]:
        station_info = KNMI_stations[KNMI_stations["LOCATIE"] == KNMI_convert_names[station]]
        lat = station_info["POS_NB"].values[0]
        lon = station_info["POS_OL"].values[0]
        coords_WGS84[station] = (lat, lon)

    with open(path.join(DATA_PATH, "Geography", "WeatherStationCoordinates.csv"), "w") as f:
        f.write("Station,Latitude,Longitude")
        for station, (lat, lon) in coords_WGS84.items():
            f.write(f"\n{station},{lat},{lon}")


extract_station_coordinates()