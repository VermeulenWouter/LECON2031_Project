# Folder contents: Geographical Data

| File Name                       | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `WeatherStationCoordinates.csv` | Coordinates of weather stations used for wind data collection.              |
| `OffshoreWindZones.shp`         | Shapefile containing offshore wind zone boundaries and metadata.            |
| `EEZBoundaries.shp`             | Shapefile containing Exclusive Economic Zone (EEZ) boundaries and metadata. |

## `WeatherStationCoordinates.csv`
### Description
This file contains the geographical coordinates (latitude and longitude) of various weather stations used for collecting wind data. The coordinates are provided in decimal degrees using the WSG84 coordinate system.

### Sources:
* KNMI weather station locations: https://dataplatform.knmi.nl/dataset/waarneemstations-csv-1-0 (csv, filtered in `ProcessRawWindData.py`)
* MVB weather station locations: https://meetnetvlaamsebanken.be/map (manual extraction, in `ProcessRawWindData.py`)

### Transformation of raw data
* KNMI data: `csv` file filtered in `ProcessRawWindData.py` to only include a few stations
* MVB data: manually extracted from the map and added in `ProcessRawWindData.py`

### Structure
| Column      | Data Type | Description                                                                                        |
|-------------|-----------|----------------------------------------------------------------------------------------------------|
| `Station`   | `String`  | The name or identifier of the weather station where the wind data was collected.                   |
| `Latitude`  | `Float`   | The latitude coordinate of the weather station in decimal degrees in the WSG84 coordinate system.  |
| `Longitude` | `Float`   | The longitude coordinate of the weather station in decimal degrees in the WSG84 coordinate system. |

## `OffshoreWindZones.shp`
*Must be together with the following files: `OffshoreWindZones.cst`, `OffshoreWindZones.dbf`, `OffshoreWindZones.prj` (projection), and `OffshoreWindZones.shx` (index).*

### Description
This shapefile contains the geographical boundaries of various offshore wind zones, together with some metadata about each zone.

### Source
* European Marine Observation and Data Network (EMODnet): https://emodnet.ec.europa.eu/geoviewer/# (downloaded as shapefile)

### Transformation of raw data
* None, used as is.

### Structure
The shapefile contains the following attributes for each offshore wind zone:

| Attribute    | Data Type | Description                                                                                                           |
|--------------|-----------|-----------------------------------------------------------------------------------------------------------------------|
| `country`    | `str`     | A full country name, in which the offshore wind zone is located.                                                      |
| `name`       | `str`     | The name of the offshore wind zone.                                                                                   |
| `n_turbines` | `float`   | The number of turbines in the offshore wind zone (should be whole value).                                             |
| `status`     | `str`     | The operational status of the offshore wind zone (e.g., `"Production"`, `"Construction"`, `"Approved"`, `"Planned"`). |
| `geometry`   | `Polygon` | The polygon geometry representing the boundaries of the offshore wind zone.                                           |
| `...`        | `...`     | Additional attributes may be present but are not used.                                                                |


## `EEZBoundaries.shp`
*Must be together with the following files: `EEZBoundaries.cst`, `EEZBoundaries.dbf`, `EEZBoundaries.qmd`, `EEZBoundaries.prj` (projection), and `EEZBoundaries.shx` (index).*

### Description
This shapefile contains the boundaries of Exclusive Economic Zones (EEZ) for various countries, along with relevant metadata.

### Source
* VLIZ Maritime Boundaries Geodatabase: https://www.vliz.be/nl/imis?dasid=8394&doiid=911 (via platform Marineregions: https://www.marineregions.org/downloads.php#marbound)

### Transformation of raw data
* None, used as is.

### Structure
No specific attributes are used in the project, though some might be present.
