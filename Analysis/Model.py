import os.path as path
import matplotlib.pyplot as plt
from Helper import load_timeseries_data


if __name__ == "__main__":
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    PRODUCTION_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_OffshoreWind_CapacityFactors_Deseasonalized.csv")
    PRODUCTIONUNITS_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_ProductionUnits.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "Model")

    df = load_timeseries_data(PRODUCTION_DATA_PATH)

    plt.plot(df.index, df["Nobelwind"], label='Nobelwind', marker="o", markersize=2, linestyle=None)
    plt.plot(df.index, df["Nobelwind_daily_season"]+df["Nobelwind_yearly_season"]+df["Nobelwind_deseason"], label='Nobelwind (with seasonal components)', marker="o", markersize=2, linestyle=":")
    plt.xlabel("DateTime (UTC)")
    plt.title("Nobelwind Capacity Factor Over Time")
    plt.ylabel("Capacity Factor [-]")
    plt.legend()
    plt.grid()
    plt.show()
