## Packages

* `statsmodels` : VAR, unit root tests, BIC, AIC, Granger causality tests, ...
    * `statsmodels.tsa` : time series analysis
    * Documentation: https://www.statsmodels.org/stable/tsa.html


## Input Data

In `DataRaw/` directory.

### On installed capacity
Note sure necessary because also included in actual generation files, but could be useful for verification.
* https://www.belgianoffshoreplatform.be/en/projects/ gives installed generation capacity per offshore wind farm + their location
* `InstalledCapacity/<year>_PerProductionUnit.csv`: Installed capacity per production unit for each year (only includes units with capacity >= 100MW).
    * Use: understand the installed capacity of different production units in Belgium to allow to express actual production as a percentage of installed capacity (= utilization rate or capacity factor).
    * https://transparencyplatform.zendesk.com/hc/en-us/articles/16648452972180-Installed-Capacity-Per-Production-Unit-14-1-B
    * Filter on: `ControlArea="10YBE----------2" and Status="COMMISSIONED"`(Belgium, operational units)
        * `ControlArea`: control area of the production unit
        * `Status`: operational status of the production unit
    * Other columns of interest: `["EICCode", "Name", "Type", "InstalledCapacity"]`
        * `EICCode`: unique identifier for each production unit
        * `Name`: name of the production unit
        * `Type`: type of production unit (non-exhaustive list: `{"B01": "Biomass", "B04": "Fossil Gas", "B10": "Hydro Pumped Storage", "B14": "Nuclear", "B18": "Wind Offshore", "B25": "Energy storage"}`)
        * `Location`: geographical location of the production unit (should be `"Belgium"` for all filtered units)
        * `InstalledCapacity`: installed capacity in MW
* `InstalledCapacity/Aggregated.csv`: Aggregated installed generation capacity per type, year, and control area.
    * Use: verify the installed capacity data from the production unit level by comparing with aggregated data (and possible determine missing capacity from smaller units <100MW).
    * https://transparencyplatform.zendesk.com/hc/en-us/articles/16648300912916-Installed-Generation-Capacity-Aggregated-14-1-A
    * Filter on: `ControlArea="10YBE----------2 and Year=<year>"`(Belgium and by year of interest)
        * `ControlArea`: control area of the production unit
    * Other columns of interest: `["Year", "Type", "InstalledCapacity"]`
        * `ProductionType`: type of production unit
            * Of interest: `Wind Offshore`
        * `AreaMapCode`: geographical location of the production unit (can be )
            * Of interest: `BE`

### On actual generation
* `<year>_<month>_ActualGeneration`: Actual Generation per Generation Unit
    * Use: get the actual generation per production unit over time to calculate utilization rates (capacity factors) for different production units.
    * https://transparencyplatform.zendesk.com/hc/en-us/articles/16648326220564-Actual-Generation-per-Generation-Unit-16-1-A
    * Filter on: `AreaCode="10YBE----------2" and GenerationUnitType="B18"`(Belgium, offshore wind units)
    * Other columns of interest: `["DateTime (UTC)", "ResolutionCode", "GenerationUnitCode", "GenerationUnitName", "GenerationUnitType", "ActualGenerationOutput", "GenerationUnitInstalledCapacity(MW)"]`
        * `DateTime (UTC)`: timestamp of the generation data (in UTC)
        * `ResolutionCode`: time resolution of the data (should be `"PT60M"` for hourly data !)
        * `GenerationUnitCode`: unique identifier for each production unit
        * `GenerationUnitName`: name of the production unit
        * `GenerationUnitType`: type of production unit (should be `"Offshore Wind"` for offshore wind units)
        * `ActualGenerationOutput`: actual generation output in MW
        * `GenerationUnitInstalledCapacity(MW)`: installed capacity of the generation unit in MW

## Data to work with
As the files in the previous section are quite large, we have preprocessed and filtered them to only include the relevant data for our analysis. The processed data files are stored in the `Data/` directory.

* `<AreaMapCode>_ProductionUnits.csv`: Processed installed capacity data for one Area (per production unit and for each year from 2015 to 2023).
    * Use: understand the installed capacity and have information about all production units in the area of interest.
    * `["GenerationUnitCode", "GenerationUnitName", "GenerationUnitType", "InstalledCapacity", "Year commissioned"]` 
        * `GenerationUnitCode`: unique identifier for each production unit (same as used in actual generation data)
        * `GenerationUnitName`: name of the production unit
        * `GenerationUnitType`: type of production unit (e.g. `"Offshore Wind"`)
        * `InstalledCapacity`: installed capacity in MW
        * `Year commissioned`: year the production unit was commissioned (putative, based on first year with non-zero generation)

* `<AreaMapCode>_<year>_GenerationPerProductionUnit.csv`: Processed actual generation data for one Area (per production unit and for each hour (if hourly data) in the specified year).
  * `["DateTime (UTC)", "GenerationUnitCode<1>_Production", "GenerationUnitCode<2>_Production", ..., "<GenerationUnitCode<N>_Production"]`
      * `DateTime (UTC)`: timestamp of the generation data (in UTC)
      * `GenerationUnitCode<i>_Production`: actual generation output in MW for production unit with code `GenerationUnitCode<i>` at the given timestamp




