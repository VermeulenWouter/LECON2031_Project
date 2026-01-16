import pandas as pd



# Debug settings
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def correlation_matrix(df, columns):
    """Compute the correlation matrix for the given columns in the DataFrame."""
    C = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for i, column_i in enumerate(columns):
        for j, column_j in enumerate(columns):
            if i <= j:
                corr_ij = df[column_i].corr(df[column_j])
                C.at[column_i, column_j] = corr_ij
                C.at[column_j, column_i] = corr_ij
    return C


def load_timeseries_data(path: str) -> pd.DataFrame:
    """Load time series data from a CSV file, filtering by date range."""
    df = pd.read_csv(path, sep=";", parse_dates=['DateTime (UTC)'])
    df = df[df["DateTime (UTC)"] > pd.Timestamp("2014-12-31 23:59:59")]
    df = df[df["DateTime (UTC)"] < pd.Timestamp("2025-11-15 23:59:59")]
    return df


farms_to_color = {
    "Mermaid": "#98df8a",
    "Northwester 2": "#c5b0d5",
    "Nobelwind": "#ffbb78",
    "Belwind": "#ff9896",
    "Seastar": "#9467bd",
    "Northwind": "#8c564b",
    "Rentel": "#e377c2",
    "Thorntonbank SW": "#7f7f7f",
    "Thorntonbank NE": "#bcbd22",
    "Norther": "#17becf",
    "Princess Elisabeth 1": "#393b79",
    "Princess Elisabeth 2": "#5254a3",
    "Princess Elisabeth 3": "#6b6ecf"
}

# Mapping done manually after investigating the data
unitname_to_commonname = {
    "Nobelwind Offshore Windpark": "Nobelwind",
    "Rentel Offshore WP": "Rentel",
    "Norther Offshore WP": "Norther",
    "Northwester 2": "Northwester_2",
    "Mermaid Offshore WP": "Mermaid",
    "Seastar Offshore WP": "Seastar",
    "Belwind Phase 1": "Belwind",
    "Northwind": "Northwind",
    "Thorntonbank - C-Power - Area NE": "Thorntonbank_NE",
    "Thorntonbank - C-Power - Area SW": "Thorntonbank_SW"
}

commonname_to_unitname = {v: k for k, v in unitname_to_commonname.items()}