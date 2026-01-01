
## About

### Authors
* Wouter Vermeulen ([wouter.vermeulen@student.uclouvain.be](mailto:wouter.vermeulen@student.uclouvain.be)), NOMA 35342100


## Getting started

### Prerequisites
* Python 3 (tested with Python 3.13)
* Required Python packages (see `requirements.txt`, and a short overview below)

### Installation

Clone the repository:
```bash
git clone https://github.com/VermeulenWouter/LECON2031_Project.git
cd LECON2031_Project
```

Install Python 3 (if not already installed): https://www.python.org/downloads/

Install required packages (see `requirements.txt`):
```bash
pip install -r requirements.txt
```

### Running the code
To follow the project timeline, run the scripts (in the given order!):
```bash
python PreProcessing/ProcessProductionRawData.py  # Process raw ENTSO-E data (datafiles not included because too large (multiple gigabytes)) 
```

```bash
python PreProcessing/ProcessRawWindData.py  # Process raw wind data from KNMI and MVB
```

```bash
python PreProcessing/InvestigateWindDirection.py  # Investigate wind direction data and create final average wind direction file
# Note this script outputs in the console, but also creates a file `Data/Weather/WindDirectionsAvg.csv` with the final average wind direction data, and some plots in `Visualisations/WindPreprocessing/`
```

```bash
python PreProcessing/InvestigateWindSpeed.py  # Investigate wind speed data and create final average wind speed file
# Note this script outputs in the console, but also creates a file `Data/Weather/WindSpeedsAvg.csv` with the final average wind speed data, and some plots in `Visualisations/WindPreprocessing/`
```

#TODO

## Key packages used

* `statsmodels` : VAR, unit root tests, BIC, AIC, Granger causality tests, ...
    * `statsmodels.tsa` : time series analysis
    * Documentation: https://www.statsmodels.org/stable/tsa.html
* `pandas` : data manipulation and analysis
    * Documentation: https://pandas.pydata.org/docs/
* `numpy` : numerical computing
    * Documentation: https://numpy.org/doc/
* `matplotlib` : data visualization
    * Documentation: https://matplotlib.org/stable/contents.html

#TODO


## Data sources
See `Data/README.md` for more information on data sources.


## Code Structure

# Folders
The main code files are located in the root directory of the project. The following table provides an overview of the main code files and folders and their purposes.

*For more details on the different folders and the files they contain, see the respective README files in those folders.*

| Name                          | Description                                                                                                                                                                                                                                                                                                                         |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DataRaw/`                    | Folder containing the raw data files used in the project. The pre-processed and filtered data files are stored in the `Data/` directory.                                                                                                                                                                                            |
| `Data/`                       | Folder containing various data files used in the model and/or analysis, including processed electricity production data and wind data.                                                                                                                                                                                              |
| `Visualisations/`             | Folder pre-created to contain visualizations and plots generated during the analysis.                                                                                                                                                                                                                                               |

| `PreProcessing/ProcessRawData.py`           | Script to preprocess raw data from ENTSO-E Transparency Platform and generate the processed data files in the `Data/` directory.                                                                                                                                                                                                    |
| `PreProcessing/ProcessRawWindData.py`       | Script to preprocess raw wind data from KNMI and MVB and generate some of the processed wind data files in the `Data/Weather/` directory.                                                                                                                                                                                           |
| `PreProcessing/InvestigateWindDirection.py` | Script to preprocess the wind direction data (from `Data/Weather/WindDirections.csv`) and giving final data used for the model (stored in `Data/Weather/WindDirectionsAvg.csv`. Checks correlation of different stations, then also looks at properties (main direction, seasonality) of the obtained final average wind direction. |
| `PreProcessing/InvestigateWindSpeed.py`     | Script to preprocess the wind speed data (from `Data/Weather/WindSpeeds.csv`) and giving final data (stored in `Data/Weather/WindSpeedsAvg.csv`. Checks correlation of different stations, then also looks at seasonality of the obtained final average wind speed.                                                                 |
| *#TODO*                       | Actual project code.                                                                                                                                                                                                                                                                                                                |
