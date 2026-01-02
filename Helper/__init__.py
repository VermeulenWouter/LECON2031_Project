import pandas as pd



# Debug settings
pd.set_option('display.max_columns', None)


def load_timeseries_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=['DateTime (UTC)'])
    df = df[df["DateTime (UTC)"] > pd.Timestamp("2014-12-31 23:59:59")]
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
    "Northwester 2": "Northwester 2",
    "Mermaid Offshore WP": "Mermaid",
    "Seastar Offshore WP": "Seastar",
    "Belwind Phase 1": "Belwind",
    "Northwind": "Northwind",
    "Thorntonbank - C-Power - Area NE": "Thorntonbank NE",
    "Thorntonbank - C-Power - Area SW": "Thorntonbank SW"
}

commonname_to_unitname = {v: k for k, v in unitname_to_commonname.items()}