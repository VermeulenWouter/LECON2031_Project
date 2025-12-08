import os

def rename_input_files():
    """Renames input files to a standardized format. To be run once before processing.

    InstalledCapacity/<year>_InstalledCapacityProductionUnit_14.1.B.csv ->  InstalledCapacity/<year>_PerProductionUnit.csv
    InstalledCapacity/InstalledGenerationCapacityAggregated_14.A.r3.csv -> InstalledCapacity/Aggregated.csv


    """
    pass


def extract_installed_capacity_data(input_directory, output_directory, control_area: str = "10YBE----------2"):
    """Extracts installed capacity data from input files and saves them to the output directory.

    Outputs to a csv file with each production unit's installed capacity per year between 2015-2025 (capacity = 0 if not present in a certain year).
    """
    pass

    # Save to "<output_directory>/<AreaMapCode>_InstalledCapacityPerProductionUnit.csv"


def extract_generation_data(input_directory, output_directory, control_area: str = "10YBE----------2"):
    """Extracts generation data from input files and saves them to the output directory.

    Outputs to one csv file per year (2015-2025) with each production unit's generation per hour (generation = 0 if not present in a certain year).
    """
    pass

    # Save to "<output_directory>/<AreaMapCode>_<year>_GenerationPerProductionUnit.csv"
