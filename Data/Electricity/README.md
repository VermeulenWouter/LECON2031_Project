# Folder contents: Electricity Data

This folder contains processed electricity data files used in the project, including installed capacity and actual generation data for various areas.

| File Name                                   | Description                                                                                            |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `<MapCode>_ProductionUnits.csv`             | Processed installed capacity data for one Area (per production unit). Also gives unit types and names. |
| `<MapCode>_GenerationPerProductionUnit.csv` | Processed actual generation data for one Area (per production unit).                                   |

## `BE_ProductionUnits.csv`
### Description
This file contains processed installed capacity data for one Area (Belgium in this case), detailing each production unit along with its type, name, and installed capacity from 2015 to 2023.

### Source
* ENTSO-E Transparency Platform: https://transparency.entsoe.eu/ (account needed)

### Transformation of raw data
Filtered to only include Belgian production unit production data, then restructured to have one row per production unit with relevant details.

### Structure
| Column                                | Data Type | Description                                                                          |
|---------------------------------------|-----------|--------------------------------------------------------------------------------------|
| `GenerationUnitCode`                  | `str`     | Unique identifier for each production unit (same as used in actual generation data). |
| `GenerationUnitName`                  | `str`     | Name of the production unit.                                                         |
| `GenerationUnitType`                  | `str`     | Type of production unit (e.g., `"Offshore Wind"`).                                   |
| `GenerationUnitInstalledCapacity(MW)` | `float`   | Installed capacity in MW.                                                            |
| `AreaCode`                            | `str`     | Code of the area (e.g., `"10YBE------------2"` for Belgium).                         |
| `AreaDisplayName`                     | `str`     | Display name of the area (e.g., `"Belgium (BE)"`).                                   |
| `AreaTypeCode`                        | `str`     | ?                                                                                    |
| `MapCode`                             | `str`     | Geographical location of the production unit (e.g., `"BE"` for Belgium).             |



## `BE_GenerationPerProductionUnit.csv`
### Description
This file contains processed actual generation data for one Area (Belgium in this case), detailing the actual generation output per production unit for each hour from 01/01/2015 to 30/11/2025.

### Source
* ENTSO-E Transparency Platform: https://transparency.entsoe.eu/ (account needed)

### Transformation of raw data
Filtered to only include Belgian production unit generation data, then restructured to have one column per production unit with timestamps as rows. Column names are formatted as `<GenerationUnitCode>_Production`, where `<GenerationUnitCode>` corresponds to the unique identifier of each production unit on which can be found more information in `BE_ProductionUnits.csv`.

### Structure
| Column                               | Data Type | Description                                                                                                    |
|--------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------|
| `DateTime (UTC)`                     | `str`     | Timestamp of the generation data (in UTC).                                                                     |
| `<GenerationUnitCode<1>>_Production` | `float`   | Actual generation output in MW for production unit with code `<GenerationUnitCode<1>>` at the given timestamp. |
| `<GenerationUnitCode<2>>_Production` | `float`   | Actual generation output in MW for production unit with code `<GenerationUnit<2>>` at the given timestamp.     |
| `...`                                | `...`     | ...                                                                                                            |
| `<GenerationUnitCode<N>>_Production` | `float`   | Actual generation output in MW for production unit with code `<GenerationUnit< N>>` at the given timestamp.    |
