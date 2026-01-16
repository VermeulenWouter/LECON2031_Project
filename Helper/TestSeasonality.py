import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def test_seasonality(df, column, freq: str = "daily"):
    """
    Runs a daily (that is, with a cycle period of one day) or yearly (that is, with a cycle period of one year) seasonality test using OLS with hour dummies, extracts the estimated 24-hour seasonal pattern.

    :param df: dataframe with datetime index
    :param column: column name existing in the df to test
    :param freq: 'daily' or 'yearly' to indicate which seasonality to test
    :return: (estimated_pattern: pd.Series indexed by hour 0-23 (or month 1-11), f_test: statsmodels F-test result, model: statsmodels regression results)
    """
    df2 = df[[column]].dropna().copy()
    df2["hour" if freq == "daily" else "month"] = df2.index.hour if freq == "daily" else df2.index.month

    model = smf.ols(f"{column} ~ C(hour)" if freq == "daily" else f"{column} ~ C(month)", data=df2).fit()

    terms = [name for name in model.params.index if name.startswith("C(hour)[T.") or name.startswith("C(month)[T.")]
    constraint = " = 0, ".join(terms) + " = 0"
    f_test = model.f_test(constraint)

    if freq == "daily":
        idx = np.arange(24)
        pattern = [model.params["Intercept"] if h == 0 else model.params.get(f"C(hour)[T.{h}]", 0.0) + model.params["Intercept"] for h in idx]
    else:
        idx = range(1, 13)
        pattern = [model.params["Intercept"] if m == 1 else model.params.get(f"C(month)[T.{m}]", 0.0) + model.params["Intercept"] for m in idx]

    pattern = pd.Series(pattern, index=idx, name="estimated_pattern")
    return pattern, f_test, model


def remove_given_seasonalities(df, columns, daily_pattern, yearly_pattern):
    """
    Removes given daily and yearly seasonal components from the dataframe for the specified columns.

    :param df: the dataframe with datetime index
    :param columns: list of column names for which to remove seasonalities
    :param daily_pattern: dict {col: daily_pattern_series}
    :param yearly_pattern: dict {col: yearly_pattern_series}
    :return: new dataframe with original values, seasonal components, and deseasonalized values. Columns added:
        - {col}_daily_season
        - {col}_yearly_season
        - {col}_deseason
    """

    out = df.copy()

    # Extract hour and month once
    out["hour"] = out.index.hour
    out["month"] = out.index.month

    for col in columns:
        # Only map seasonal components from first valid index onward
        daily = out["hour"].map(daily_pattern[col])
        yearly = out["month"].map(yearly_pattern[col])
        first_valid = out[col].first_valid_index()
        daily = daily.where(out.index >= first_valid)
        yearly = yearly.where(out.index >= first_valid)
        out[f"{col}_daily_season"] = daily
        out[f"{col}_yearly_season"] = yearly

        # Remove both seasonal components
        out[f"{col}_sadjusted"] = out[col] - daily - yearly
    return out