
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