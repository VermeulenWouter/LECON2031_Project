#  Folder contents: Weather Data

## `KNNMI/<year>-<year>_<weatherstation> wind.txt`
### Description
This file contains raw wind data from the Royal Netherlands Meteorological Institute (KNMI) for a specific weather station and year range. The data includes wind speed and direction measurements taken at regular intervals.

### Source
* Royal Netherlands Meteorological Institute (KNMI): https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens_Noordzee

### Structure
See the header of the specific file for detailed column descriptions.


## `AWS_stationsmetadata.csv`
### Description
This file contains metadata for weather stations used in the project, including station IDs, names, locations, and other relevant information.

### Source
* Royal Netherlands Meteorological Institute (KNMI): https://dataplatform.knmi.nl/dataset/waarneemstations-csv-1-0

### Structure
*Note only relevant columns are included below.*

| Column    | Data Type | Description                                            |
|-----------|-----------|--------------------------------------------------------|
| `LOCATIE` | `str`     | Name of the weather station.                           |
| `POS_NB`  | `float`   | Latitude of the weather station (in decimal degrees).  |
| `POS_OL`  | `float`   | Longitude of the weather station (in decimal degrees). |
| `HOOGTE`  | `float`   | Elevation of the weather station (in meters).          |


## `MVB/<weatherstation> <year>/MP<i>.WC3_<...>.txt`, `MVB/<weatherstation> <year>/MP<i>.WRS_<...>.txt`, `MVB/<weatherstation> <year>/MP<i>.WVC_<...>.txt`

### Description
These files contain raw wind data from the Meetnet Vlaamse Banken (MVB) for specific weather stations and years. The data includes wind speed (WVC), wind direction (WRC), and wind gusts (WC3).

### Source
* Meetnet Vlaamse Banken (MVB): https://meetnetvlaamsebanken.be/Download (account needed)

### Structure
See the files `MVB/<weatherstation> <year>/_Content_Description.txt` for column informtation.

