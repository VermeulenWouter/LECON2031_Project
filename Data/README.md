# Folder contents: Data

This folder contains various data files used in the project, including processed electricity production data and wind data.

*Note: unprocessed raw data files are stored in a separate folder, `DataRaw/`.*

## Subfolders

| Subfolder Name | Description                                                                                                                                                           |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Geography/`   | Contains geographical data files such as weather station coordinates and shapefiles for offshore wind zones and EEZ boundaries.                                       |
| `Weather/`     | Contains processed wind data files, including wind speeds and directions from various weather stations.                                                               |
| `Electricity/` | Contains processed electricity production data files for different areas and production units. Also contains a database with information on each production facility. |


## Data sources
* `Electricity/`: ENTSO-E Transparency Platform (https://transparency.entsoe.eu/ - needs account).
* `Weather/`: KNMI (https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens_Noordzee) and Meetnet Vlaamse Banken (https://meetnetvlaamsebanken.be/Download - needs account).
* `Geography/`
  * KNMI weather station locations: https://dataplatform.knmi.nl/dataset/waarneemstations-csv-1-0
  * MVB weather station locations: https://meetnetvlaamsebanken.be/map
  * Geographical shapefiles for offshore wind zones are sourced from EMODnet: https://emodnet.ec.europa.eu/geoviewer/#
  * Geographical shapefiles for EEZ boundaries: Flanders Marine Institute (https://www.marineregions.org/)