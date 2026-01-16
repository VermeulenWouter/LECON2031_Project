"""
Forecasting functions for VAR, AR, and ARX models with error metrics.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg


def add_lags(df, cols, p):
    """
    Add lagged versions of columns in `cols` up to lag p.
    """
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        for lag in range(1, p + 1):
            df[f"{col}.L{lag}"] = df[col].shift(lag)

    return df


def ar_forecast(results, window_values, h):
    """
    Make h-step ahead forecast using fitted AutoReg model.

    :param results: a fitted AutoReg results object
    :param window_values: last p values of the time series (1D array-like)
    :param h: forecast horizon
    :return: the h-step ahead forecast
    """
    params = results.params
    intercept = params.iloc[0]
    coefs = params.iloc[1:]
    p = len(coefs)

    # Start with the last p values
    history = list(window_values[-p:])

    for _ in range(h):
        yhat = intercept + np.dot(coefs, history[-p:])
        history.append(yhat)

    return history[-1]


def arx_forecast(results, y_window, X_window, h, p):
    """
    results: fitted AutoReg with exog
    y_window: last p values of y (1D)
    X_window: last row of exog (1D, length k)
    p: lag order
    """
    params = results.params.values

    intercept = params[0]
    ar_coefs = params[1:1+p]
    exog_coefs = params[1+p:]

    y_hist = list(y_window[-p:])
    X_last = X_window[-1]  # shape (k,)

    for _ in range(h):
        y_lags = y_hist[-p:]
        yhat = intercept + np.dot(ar_coefs, y_lags) + np.dot(exog_coefs, X_last)
        y_hist.append(yhat)

    return y_hist[-1]


def forecast_model(df, model_type, target_col, cols, lag_order, cutoff_date, horizons, seasonal_component=None):
    """
    Forecasting function to get error estimates for VAR, AR, and ARX models.

    :param df: a pandas DataFrame with a DateTimeIndex
    :param model_type: AR, ARX or VAR. ARX is used when emulating a VAR but only forecasting one target variable.
    :param target_col: the target column to forecast and for which to measure errors
    :param cols: the columns to use as predictors (for VAR, all columns including target_col; for ARX, exogenous columns)
    :param lag_order: the lag order to use in the models
    :param cutoff_date: the date to split train and test data (call this function multiple times to do rolling forecasts)
    :param horizons: the forecast horizons to evaluate (in number of time steps)
    :param seasonal_component: if provided, this seasonal component will be added back to forecasts and true values before error calculation

    :return: the forecasts and error metrics (RMSE and MAE) for each horizon
    """
    train = df.loc[:cutoff_date]
    test = df.loc[cutoff_date:]

    if model_type == "VAR":
        model = VAR(train[cols], freq="h")
        results = model.fit(lag_order)
        if not results.is_stable():
            raise ValueError("Fitted VAR model is not stable.")
    elif model_type == "AR":
        model = AutoReg(train[target_col], lags=lag_order, trend="c")
        results = model.fit()
    elif model_type == "ARX":
        model = AutoReg(train[target_col], lags=lag_order, exog=train[cols], trend="c")
        results = model.fit()
    else:
        raise ValueError("model_type must be 'VAR', 'AR', or 'ARX'")

    forecasts = {h: [] for h in horizons}
    index = test.index
    data = pd.concat([train, test])

    for t in range(len(test)):
        window_end = len(train) + t
        window = data.iloc[window_end - lag_order : window_end]

        for h in horizons:
            if model_type == "VAR":
                f = results.forecast(window[cols].values, steps=h)[-1]
                f = pd.Series(f, index=cols)[target_col]
            elif model_type == "AR":
                f = ar_forecast(results, window[target_col].values, h)
            elif model_type == "ARX":
                X_window = window[cols].values  # shape (lag_order, k)
                f = arx_forecast(results, window[target_col].values, X_window, h, lag_order)

            forecasts[h].append(f)

    # Conver to series and add back seasonal component if provided
    for h in horizons:
        forecasts[h] = pd.Series(forecasts[h], index=index, name=target_col)

    if seasonal_component is not None:
        for h in horizons:
            forecasts[h] = forecasts[h] + seasonal_component.loc[index]

    metrics = {}
    for h in horizons:
        pred = forecasts[h]
        true = test[target_col].loc[pred.index]
        if seasonal_component is not None:
            true = true + seasonal_component.loc[pred.index]

        metrics[h] = {
            "RMSE": float(np.sqrt(np.mean((true - pred)**2))),
            "MAE": float(np.mean(np.abs(true - pred)))
        }

    return forecasts, metrics
