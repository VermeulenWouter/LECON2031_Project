import os.path as path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helper import *


#%%
if __name__ == "__main__":
    FIGS = False
    # %%
    _FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
    PRODUCTION_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_GenerationPerProductionUnit.csv")
    PRODUCTIONUNITS_DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity", "BE_ProductionUnits.csv")
    VISUALISATION_PATH = path.join(_FILE_PATH, "Visualisations", "ElectricityPreProcessing")

    df_allproductions = load_timeseries_data(PRODUCTION_DATA_PATH)
    df_generators = pd.read_csv(PRODUCTIONUNITS_DATA_PATH, sep=';')


    df_generators_ow = df_generators[df_generators['GenerationUnitType'] == "Wind Offshore"]
    df_productions_ow = df_allproductions[[*df_generators_ow["GenerationUnitCode"], "DateTime (UTC)"]]

    #%%
    if FIGS:
        fig, ax = plt.subplots(figsize=(12, 8), nrows=6, ncols=2)
        for i, col in enumerate(df_productions_ow.columns):
            if col != "DateTime (UTC)":
                generator_name = unitname_to_commonname[df_generators_ow[df_generators_ow["GenerationUnitCode"] == col]["GenerationUnitName"].values[0]]
                a, b = divmod(i-1, 2)
                ax[a][b].plot(df_productions_ow["DateTime (UTC)"], df_productions_ow[col], label=f"{generator_name} ({col})", color=farms_to_color.get(generator_name, None))
                ax[a][b].grid()
                ax[a][b].legend(loc="lower left")
                ax[a][b].set_xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2026-12-31"))
                i += 1

        plt.tight_layout()
        plt.savefig(path.join(VISUALISATION_PATH, "Stacked_Offshore_Wind_Generation.png"))
        plt.show()

    # CONCLUSION: Norther Offshore WP is present twice but for different time periods => Merge
    #%%

    # Union the two Norther Offshore WP columns
    norther_duplicates = df_generators_ow[df_generators_ow["GenerationUnitName"] == "Norther Offshore WP"]["GenerationUnitCode"].to_list()
    print(f"UnitGenerationCode for Northern: {norther_duplicates}")
    df_productions_ow = df_productions_ow.assign(**{"Norther Offshore WP - Merged": np.where(df_productions_ow[norther_duplicates[0]] == 0, df_productions_ow[norther_duplicates[1]], df_productions_ow[norther_duplicates[0]])})
    df_productions_ow = df_productions_ow.drop(columns=[norther_duplicates[0], norther_duplicates[1]])
    df_productions_ow.rename(columns={"Norther Offshore WP - Merged": norther_duplicates[0]}, inplace=True)

    # Rename columns to common names
    rename_dict = df_generators_ow[["GenerationUnitCode", "GenerationUnitName"]].set_index("GenerationUnitCode")["GenerationUnitName"].to_dict()
    rename_dict = {k: unitname_to_commonname[v] for k, v in rename_dict.items()}
    df_productions_ow.rename(columns=rename_dict, inplace=True)

    #%%
    # Convert generation data in MW to capacity factors
    df_cf_ow = df_productions_ow.copy()
    df_cf_ow = df_cf_ow.set_index("DateTime (UTC)")
    for col in df_cf_ow.columns:
        capacity_given = df_generators_ow[df_generators_ow["GenerationUnitName"] == commonname_to_unitname[col]]["GenerationUnitInstalledCapacity(MW)"].values[0]
        capacity_exp = df_cf_ow[col].max()
        if not capacity_exp <= capacity_given:
            print(f"WARNING: Capacity mismatch for {col}: Given={capacity_given}, Experimental={capacity_exp}")
        df_cf_ow[col] = df_cf_ow[col] / max(capacity_given, capacity_exp)


    if FIGS:
        fig, ax = plt.subplots(figsize=(12, 8), nrows=5, ncols=2)
        for i, col in enumerate(df_cf_ow.columns):
            b, a = divmod(i - 1, 5)
            ax[a][b].plot(df_cf_ow[col], label=col, color=farms_to_color.get(col, None))
            ax[a][b].grid()
            ax[a][b].legend(loc="lower left")
            ax[a][b].set_xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2026-12-31"))
            i += 1

        plt.tight_layout()
        plt.savefig(path.join(VISUALISATION_PATH, "Stacked_Offshore_Wind_CapacityFactors.png"))
        plt.show()

