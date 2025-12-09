
## How to run

#TODO

## Packages used

* `statsmodels` : VAR, unit root tests, BIC, AIC, Granger causality tests, ...
    * `statsmodels.tsa` : time series analysis
    * Documentation: https://www.statsmodels.org/stable/tsa.html


## Input Data

### Source
The raw data is sourced from the ENTSO-E Transparency Platform: https://transparency.entsoe.eu/. These files contain detailed information about electricity generation and installed capacity across various regions in Europe, but they are quite large and contain more information than needed for our specific analysis. A script (`ProcessRawData.py`) has been created to preprocess and filter the raw data, extracting only the relevant information for our analysis; only this processed data is included in the `Data/` directory.

### Files
* `<MapCode>_ProductionUnits.csv`: Processed installed capacity data for one Area (per production unit and for each year from 2015 to 2023).
    * Use: understand the installed capacity and have information about all production units in the area of interest.
    * `[GenerationUnitCode;GenerationUnitName;GenerationUnitType;GenerationUnitInstalledCapacity(MW);AreaCode;AreaDisplayName;AreaTypeCode;MapCode]` 
        * `GenerationUnitCode`: unique identifier for each production unit (same as used in actual generation data)
        * `GenerationUnitName`: name of the production unit
        * `GenerationUnitType`: type of production unit (e.g. `"Offshore Wind"`)
        * `GenerationUnitInstalledCapacity(MW)`: installed capacity in MW
        * `AreaCode`: code of the area (e.g. `"10YBE----------2"` for Belgium)
        * `AreaDisplayName`: display name of the area (e.g. `"Belgium (BE)"`)
        * `AreaTypeCode`: ?
        * `MapCode`: geographical location of the production unit (e.g. `"BE"` for Belgium)

* `<MapCode>_GenerationPerProductionUnit.csv`: Processed actual generation data for one Area (per production unit, for each hour from 01/01/2015 - 30/11/2025).
  * `["DateTime (UTC)", "GenerationUnitCode<1>_Production", "GenerationUnitCode<2>_Production", ..., "<GenerationUnitCode<N>_Production"]`
      * `DateTime (UTC)`: timestamp of the generation data (in UTC)
      * `<GenerationUnitCode<i>>`: actual generation output in MW for production unit with code `<GenerationUnitCode<i>>` at the given timestamp (for information about the production unit, refer to `<MapCode>_ProductionUnits.csv` file)


## Code Structure

| Filename            | Description                                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `ProcessRawData.py` | Script to preprocess raw data from ENTSO-E Transparency Platform and generate the processed data files in the `Data/` directory. |
| *#TODO*             | Actual project code.                                                                                                             |