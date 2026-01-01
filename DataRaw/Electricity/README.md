# Folder contents: Electricity Data

## `<year>_<month>_ActualGeneration.csv`
### Description
This file contains actual generation data for all units on the ENTSO-E Transparency Platform, detailing the actual generation output per generation unit for each hour, with for each entry also installed capacity, unit type, unit name, ...

*These files are too large to be stored in the repository (up to 600MB per month) and can be obtained from the ENTSO-E Transparency Platform directly.*

### Source
* ENTSO-E Transparency Platform: https://transparency.entsoe.eu/ (account needed)
* Documentation: https://transparencyplatform.zendesk.com/hc/en-us/articles/16648326220564-Actual-Generation-per-Generation-Unit-16-1-A

### Structure
| Column                                | Data Type | Description                                                                           |
|---------------------------------------|-----------|---------------------------------------------------------------------------------------|
| `DateTime (UTC)`                      | `str`     | Timestamp of the generation data (in UTC).                                            |
| `ResolutionCode`                      | `str`     | Time resolution of the data (should be `"PT60M"` for hourly data).                    |
| `GenerationUnitCode`                  | `str`     | Unique identifier for each production unit.                                           |
| `GenerationUnitName`                  | `str`     | Name of the production unit.                                                          |
| `GenerationUnitType`                  | `str`     | Type of production unit (e.g., `"Offshore Wind"`).                                    |
| `ActualGenerationOutput(MW)`          | `float`   | Actual generation output in MW.                                                       |
| `GenerationUnitInstalledCapacity(MW)` | `float`   | Installed capacity of the generation unit in MW.                                      |
| `AreaCode`                            | `str`     | Code of the area (e.g., `"10YBE------------2"` for Belgium).                          |
| `AreaDisplayName`                     | `str`     | Display name of the area (e.g., `"Belgium (BE)"`).                                    |
| `AreaTypeCode`                        | `str`     | ?                                                                                     |
| `MapCode`                             | `str`     | Geographical location of the production unit (e.g., `"BE"` for Belgium).              |
| `ActualConsumption(MW)`               | `float`   | Actual consumption in MW. Will always be zero since the data is for production units. |
| `UpdateTime (UTC)`                    | `str`     | Timestamp of the last update of the data (in UTC).                                    |

