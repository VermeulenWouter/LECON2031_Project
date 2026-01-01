
## Authors
* Wouter Vermeulen (wouter.vermeulen@student.uclouvain.be), NOMA 35342100

## How to run

Install required packages (see `requirements.txt`):
```bash
pip install -r requirements.txt
```

To follow the project timeline, run the scripts (in the given order!):
```bash
python ProcessProductionRawData.py  # Process raw ENTSO-E data (datafiles not included because too large (multiple gigabytes)) 
```

```bash
python ProcessRawWindData.py  # Process raw wind data from KNMI and MVB
```

```bash
python InvestigateWindDirection.py  # Investigate wind direction data and create final average wind direction file
# Note this script outputs in the console, but also creates a file `Data/Weather/WindDirectionsAvg.csv` with the final average wind direction data, and some plots in `Visualisations/WindPreprocessing/`
```

```bash
python InvestigateWindSpeed.py  # Investigate wind speed data and create final average wind speed file
# Note this script outputs in the console, but also creates a file `Data/Weather/WindSpeedsAvg.csv` with the final average wind speed data, and some plots in `Visualisations/WindPreprocessing/`
```

#TODO

## Packages used

* `statsmodels` : VAR, unit root tests, BIC, AIC, Granger causality tests, ...
    * `statsmodels.tsa` : time series analysis
    * Documentation: https://www.statsmodels.org/stable/tsa.html

#TODO

## Input Data

#TODO

### Source
The raw data is sourced from the ENTSO-E Transparency Platform: https://transparency.entsoe.eu/ (needs account). These files contain detailed information about electricity generation and installed capacity across various regions in Europe, but they are quite large and contain more information than needed for our specific analysis. A script (`ProcessRawData.py`) has been created to preprocess and filter the raw data, extracting only the relevant information for our analysis; only this processed data is included in the `Data/` directory.

Wind data is sourced both from KNMI (https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens_Noordzee) and Meetnet Vlaamse Banken (https://meetnetvlaamsebanken.be/Download - needs account).
KNMI weather station locations: https://dataplatform.knmi.nl/dataset/waarneemstations-csv-1-0
MVB weather station locations: https://meetnetvlaamsebanken.be/map

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

* The Netherlands don't seem to share individual generation => annoying because they have some wind farms very close to the Belgian ones ...


## Code Structure

| Filename            | Description                                                                                                                                                                                                                                                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ProcessRawData.py` | Script to preprocess raw data from ENTSO-E Transparency Platform and generate the processed data files in the `Data/` directory.                                                                                                                                                                                                    |
| `ProcessRawWindData.py` | Script to preprocess raw wind data from KNMI and MVB and generate some of the processed wind data files in the `Data/Weather/` directory.                                                                                                                                                                                           |
| `InvestigateWindDirection.py` | Script to preprocess the wind direction data (from `Data/Weather/WindDirections.csv`) and giving final data used for the model (stored in `Data/Weather/WindDirectionsAvg.csv`. Checks correlation of different stations, then also looks at properties (main direction, seasonality) of the obtained final average wind direction. |
| *#TODO*             | Actual project code.                                                                                                                                                                                                                                                                                                                |

### Localisation data
For a global view: https://windeurope.org/data/products/european-offshore-wind-farms-map-public/

KNMI stations: https://www.knmi.nl/kennis-en-datacentrum/uitleg/automatische-weerstations