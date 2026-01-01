# Folder contents: Wind Data

| File                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `WindDirections.csv`    | Processed wind direction data from weather stations                         |
| `WindSpeeds.csv`        | Processed wind speed data from weather stations                             |
| `WindGusts.csv`         | Processed wind gust data from weather stations                              |
| `WindDirectionsAvg.csv` | Average wind direction data from selected stations, used for the modelling. |


## `WindDirections.csv`
### Description
This file contains processed wind direction data collected from various weather stations. Each column represents a different measurement station, and the values indicate the prevailing wind direction in the last 10 minutes of each hour, expressed in degrees (1° to 360°).

### Sources:
* KNMI climatology weather station data https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens_Noordzee
* Meetnet Vlaamse Banken weather pole data https://meetnetvlaamsebanken.be/Download (needs account)

### Transformation of raw data
* KNMI data: `csv` files merged and filtered in `ProcessRawWindData.py` to only include a few stations
* MVB data: `csv` files merged in `ProcessRawWindData.py`. Agregated to match KNMI format (last 10 minutes of each hour).

### Structure

| Column                    | Format                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                         | 
|---------------------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DateTime (UTC)`          | `YYYY-MM-DD HH:MM:SS`    | The date and time (in UTC) of the data point. Data is hourly and describes the hour in the future: if this field is for example `2025-01-01 12:00:00` then the data it contains is for the period from 12h00 to 13h00.                                                                                                                                                                                                              |
| `<MeasurementStation<i>>` | `int` or `''` (no value) | One column for each measurement station. In each column is given the average wind direction in degrees (in the range `1-360`, where `90=east`, `180=south`, `270=west` and `360=north` - `0` indicates no wind and `990` indicates variable direction). Note this is the direction FROM WHICH the wind is blowing.<br/><br/>*Note: following the data from KNMI, the average is taken only from the ten last minutes of each hour.* |


## `WindSpeeds.csv`
### Description
This file contains hourly wind speed data collected from various weather stations. Each column represents a different measurement station, and the values indicate the maximum wind speed recorded during each hour at a height of 10 meters above ground level. The wind speeds are measured in meters per second (m/s).

### Sources:
* KNMI climatology weather station data https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens_Noordzee
* Meetnet Vlaamse Banken weather pole data https://meetnetvlaamsebanken.be/Download (needs account)

### Transformation of raw data
* KNMI data: `csv` files merged and filtered in `ProcessRawWindData.py` to only include a few stations
* MVB data: `csv` files merged in `ProcessRawWindData.py`. Agregated to match KNMI format (hourly averages).

### Structure
| Column                    | Format                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 
|---------------------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DateTime (UTC)`          | `YYYY-MM-DD HH:MM:SS`      | The date and time (in UTC) of the data point. Data is hourly and describes the hour in the future: if this field is for example `2025-01-01 12:00:00` then the data it contains is for the period from 12h00 to 13h00.                                                                                                                                                                                                                                                                                      |
| `<MeasurementStation<i>>` | `float` or `''` (no value) | One column for each measurement station. In each column is given the average (over the hour) wind speed (at height 10m), in `m/s`.<br/> <br/>*Note: following a KNMI reference (https://www.knmiprojects.nl/site/binaries/site-content/collections/documents/2006/01/01/h05-wind/Handboek_H05.pdf) and with the goal to present only very similar data : in hourly data average wind speed is the numerical average of the 10-minute averages (not the vector-average taking into account wind direction).* |


## `WindGusts.csv`

### Description
This file contains processed wind gust data collected from various weather stations. Each column represents a different measurement station, and the values indicate the maximum wind speed recorded during each hour. The data is provided in meters per second (m/s) and is based on 3-second averages.

### Sources:
* KNMI climatology weather station data https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens_Noordzee
* Meetnet Vlaamse Banken weather pole data https://meetnetvlaamsebanken.be/Download (needs account)

### Transformation of raw data
* KNMI data: `csv` files merged and filtered in `ProcessRawWindData.py` to only include a few stations
* MVB data: `csv` files merged in `ProcessRawWindData.py`. Agregated to match KNMI format (hourly maximums).
### Structure
| Column                    | Format                     | Description                                                                                                                                                                                                                                            | 
|---------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DateTime (UTC)`          | `YYYY-MM-DD HH:MM:SS`      | The date and time (in UTC) of the data point. Data is hourly and describes the hour in the future: if this field is for example `2025-01-01 12:00:00` then the data it contains is for the period from 12h00 to 13h00.                                 |
| `<MeasurementStation<i>>` | `float` or `''` (no value) | One column for each measurement station. In each column is given the maximum wind speed measured in the hour (at height 10m), in `m/s`.<br/> <br/>*Note: as measurements are all 3-second averages already, this means maximum 3-second-long average.* |

## `WindDirectionsAvg.csv`
### Description
This file contains processed wind direction data, an average of multiple others.

### Sources:
* See `WindDirections.csv`

### Transformation of raw data
* Created in `InvestigateWindSpeed.py` as an average of some columns of `WindDirections.csv`.

### Structure
| Column                 | Format                     | Description                                                                                                                                                                                                                                                                 | 
|------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DateTime (UTC)`       | `YYYY-MM-DD HH:MM:SS`      | The date and time (in UTC) of the data point. Data is hourly and describes the hour in the future: if this field is for example `2025-01-01 12:00:00` then the data it contains is for the period from 12h00 to 13h00.                                                      |
| `AverageWindDirection` | `float` or `''` (no value) | The average wind direction in degrees (in the range `1-360`, where `90=east`, `180=south`, `270=west` and `360=north`). Values `0` (no wind) and `990` (variable wind) have been transformed to `''` (no value). Note this is the direction FROM WHICH the wind is blowing. |


## `WindSpeedsAvg.csv`
### Description
This file contains processed wind speed data, an average of multiple others.

### Sources
* See `WindSpeeds.csv`

### Transformation of raw data
* Created in `InvestigateWindSpeed.py` as an average of some columns of `WindSpeeds.csv`.

### Structure
| Column             | Format                     | Description                                                                                                                                                                                                            | 
|--------------------|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DateTime (UTC)`   | `YYYY-MM-DD HH:MM:SS`      | The date and time (in UTC) of the data point. Data is hourly and describes the hour in the future: if this field is for example `2025-01-01 12:00:00` then the data it contains is for the period from 12h00 to 13h00. |
| `AverageWindSpeed` | `float` or `''` (no value) | The average wind speed (at height 10m), in `m/s`.                                                                                                                                                                      |