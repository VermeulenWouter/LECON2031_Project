"""
Processes raw data files for generation and installed capacity.

@author: Wouter Vermeulen
@date: 2025-12-10
"""

import os
import os.path as path
import pandas as pd
import re
from typing import List, Optional


#%%

_FILE_PATH = path.dirname(path.abspath(path.dirname(__file__)))
# RAWDATA_PATH = "D:/Temp projet econometrics"  # Because files too big
RAWDATA_PATH = path.join(_FILE_PATH, "DataRaw")
DATA_PATH = path.join(_FILE_PATH, "Data", "Electricity")

# Debug settings
DEBUG = False
pd.set_option('display.max_columns', None)

#%%
def _safe_rename(src: str, dst: str) -> bool:
    """
    Safely rename a file: check the ´dest´ filename doesn't already exist.
    Also catch errors (simply report).

    :param src: the src file name
    :param dst: the new filename
    :return: True if rename was success, False if not
    """
    # If src and dst are the same path, nothing to do
    if src == dst:
        return False
    if os.path.exists(dst):
        # Do not overwrite; skip
        print(f"Skipping rename, destination exists: '{dst}' (source: '{src}')")
        return False
    try:
        os.rename(src, dst)
        print(f"Renamed: '{src}' -> '{dst}'")
        return True
    except OSError as e:
        print(f"Failed to rename '{src}' -> '{dst}': {e}")
        return False

def rename_raw_input_files() -> Optional[List[str]]:
    """Renames input files in the DataRaw folder to a standardized format.

    Behavior:
    - DataRaw/InstalledCapacity/<year>_InstalledCapacityProductionUnit_...csv
      -> DataRaw/InstalledCapacity/<year>_PerProductionUnit.csv
    - DataRaw/Electricity/<year>_<month>_ActualGenerationOutputPerGenerationUnit_...csv
      -> DataRaw/Electricity/<year>_<month>_ActualGeneration.csv

    The function will not overwrite existing files. If a target name already exists it will skip that file
    and report it as skipped. Returns a list with the original file paths that were renamed, or None if
    no files were changed.
    """
    # Resolve DataRaw directory relative to this script
    changed: List[str] = []

    # inst_dir = os.path.join(RAWDATA_PATH, "InstalledCapacity")
    # for fname in os.listdir(inst_dir):
    #     fpath = os.path.join(inst_dir, fname)
    #
    #     m = re.match(r"^(?P<year>\d{4})_InstalledCapacityProductionUnit_.*\.csv$", fname, flags=re.IGNORECASE)
    #     if not m:
    #         continue
    #     year = m.group('year')
    #     new_name = f"{year}_PerProductionUnit.csv"
    #     new_path = os.path.join(inst_dir, new_name)
    #     if _safe_rename(fpath, new_path):
    #         changed.append(fpath)

    # 2) Electricity
    gen_dir = os.path.join(RAWDATA_PATH, "Electricity")
    for fname in os.listdir(gen_dir):
        fpath = os.path.join(gen_dir, fname)

        m = re.match(r"^(?P<year>\d{4})_(?P<month>\d{2})_ActualGenerationOutputPerGenerationUnit_.*\.csv$", fname)
        if not m:
            continue
        year = m.group('year')
        month = m.group('month')
        new_name = f"{year}_{month}_ActualGeneration.csv"
        new_path = os.path.join(gen_dir, new_name)
        if _safe_rename(fpath, new_path):
            changed.append(fpath)

    if not changed:
        return None
    return changed


rename_raw_input_files()
#%%

def extract_generation_data(output_directory: str = DATA_PATH, control_area: str | None = None, generation_type: str | None = None):
    """Extracts generation data from input files and saves them to the output directory.

    Either 'control_area' or 'generation_type' must be provided (or both).
    Two files are output:
    - "<output_directory>/<AreaMapCode>_<GenerationType>_GenerationPerProductionUnit.csv"
    - "<output_directory>/<AreaMapCode>_<GenerationType>_GenerationUnitSummary.csv"
    If either 'control_area' or 'generation_type' is None, it is omitted from the filename. If both are none,
    the prefix "GenerationData" is used.

    <GenerationType> = `generation_type` except for: {"Wind Offwhore": "WO"}

    This function overwrites existing files in the output directory!
    """
    os.makedirs(output_directory, exist_ok=True)

    generation_types = {"Wind Offshore": "WO"}
    generationtype = generation_types.get(generation_type, generation_type)

    # Gather input files
    input_dir = path.join(RAWDATA_PATH, "Electricity")
    pattern = re.compile(r"^(\d{4})_(\d{2})_ActualGeneration\.csv$")
    actual_generation_files = {}  # {year_1: [file_month_1, file_month_2, ...], ...}

    n_files = 0
    for fname in os.listdir(input_dir):
        full_path = os.path.join(input_dir, fname)

        match = pattern.match(fname)
        if not match:
            continue

        year, month = match.groups()
        if year not in actual_generation_files:
            actual_generation_files[year] = []
        actual_generation_files[year].append(full_path)
        n_files += 1

    # Concatenate all files (work per month then per year to avoid too large lists (one month ~500MB))
    print(f"Processing files:")
    dfs_per_year = []
    i = 1
    for year in actual_generation_files.keys():
        dfs_per_month = []
        for f in actual_generation_files[year]:
            print(f"\t[{i:3}/{n_files}] {f}")
            df_month = pd.read_csv(f, sep="\t")

            # Keep only one control area if specified
            if control_area is not None:
                df_month = df_month[df_month["AreaCode"] == control_area]
                if df_month.empty:
                    print(f"No rows found for AreaCode={control_area} in file {f}. Skipping...")
                    continue

                # Check resolution code
                # if not (df_month["ResolutionCode"] == "PT60M").all():
                #    raise ValueError(f"File {f} contains non-PT60M rows for control area {control_area}.")

            # TODO check the netherlands because all are "other"
            if generation_type is not None:
                df_month = df_month[df_month["GenerationUnitType"] == generation_type]
                if df_month.empty:
                    print(f"No rows found for GenerationUnitType={generation_type} in file {f}. Skipping...")
                    continue

                # Check resolution code
                # TODO check why some files have non-PT60M rows
                # if not (df_month["ResolutionCode"] == "PT60M").all():
                #    raise ValueError(f"File {f} contains non-PT60M rows for GenerationUnitType {generation_type}.")


            # Already delete columns not used later
            unused_cols = ["ActualConsumption(MW)", "UpdateTime(UTC)"]
            df_month = df_month.drop(columns=unused_cols)

            dfs_per_month.append(df_month)
            i += 1

        df_year = pd.concat(dfs_per_month)
        del dfs_per_month[:]

        if df_year.empty:
            print(f"No rows found for AreaCode={control_area} in year={year}. Check if 'control_area' already existed at that time...")
            return

        dfs_per_year.append(df_year)

    df = pd.concat(dfs_per_year)
    del dfs_per_year[:]

    # Parse columns
    df["DateTime (UTC)"] = pd.to_datetime(df["DateTime (UTC)"], errors='coerce')

    # Check metadata is constant for each generation unit
    print("\nChecking metadata consistency per GenerationUnitCode...")
    metadata_cols = [
        "GenerationUnitName",
        "GenerationUnitType",
        "GenerationUnitInstalledCapacity(MW)",
        "AreaCode",
        "AreaDisplayName",
        "AreaTypeCode",
        "MapCode",
        "ResolutionCode"
    ]

    for unit, group in df.groupby("GenerationUnitCode"):
        inconsistent = False
        for col in metadata_cols:
            unique_vals = group[col].dropna().unique()
            if len(unique_vals) > 1:
                if not inconsistent:
                    print(f"Inconsistency found for GenerationUnitCode: {unit}")
                    inconsistent = True
                print(f"\t* Column '{col}' has multiple values: {unique_vals}")

    # Extract metadata per generation unit and save to separate file
    generation_unit_summary = (
        df.groupby("GenerationUnitCode")[metadata_cols]
        .first()
        .reset_index()
    )

    if control_area is not None:
        mapcode = generation_unit_summary["MapCode"].iloc[0]
        filename_prefix = f"{mapcode}_{generationtype}" if generationtype is not None else f"{mapcode}"
    else:
        filename_prefix = f"{generationtype}" if generationtype is not None else "GenerationData"
    generation_unit_summary.to_csv(path.join(output_directory, f"{filename_prefix}_ProductionUnits.csv"), index=False, sep=";")

    # Restructure DataFrame (losing 'metadata') and output to separate file
    print("\nRestructuring generation data and saving to file...")
    df = df.drop(columns=metadata_cols)

    df = df.pivot(
        index="DateTime (UTC)",
        columns="GenerationUnitCode",
        values="ActualGenerationOutput(MW)"
    ).sort_index()

    # Make sure output is continuously each hour (values for missing hours become NaN)
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h"
    )

    df = df.reindex(full_index)
    df.index.name = "DateTime (UTC)"

    df.to_csv(path.join(output_directory, f"{filename_prefix}_GenerationPerProductionUnit.csv"), sep=";")


extract_generation_data(control_area="10YBE----------2")
# extract_generation_data(generation_type="Wind Offshore")

# extract_generation_data(control_area="10YNL----------L")
