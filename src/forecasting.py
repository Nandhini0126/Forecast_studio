"""
forecasting.py
--------------
The two baselines, the machine-learning model, and the error metrics.

Why baselines matter: a model is only "good" if it beats a dumb rule. If a
Random Forest can't beat "assume next month looks like last month", the Random
Forest is not worth using. Always compare against a baseline -- examiners ask.
"""

import numpy as np

from config import RANDOM_STATE, HORIZON
from features import feature_columns


# ---- error metrics -----------------------------------------------------------
def mae(actual, pred):
    """Mean Absolute Error: on average, how many units off are we?"""
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual, pred):
    """Root Mean Squared Error: like MAE but punishes big misses harder."""
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(actual, pred):
    """Mean Absolute Percentage Error: average error as a % of actual sales."""
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def all_metrics(actual, pred):
    return {"MAE": round(mae(actual, pred), 2),
            "RMSE": round(rmse(actual, pred), 2),
            "MAPE_%": round(mape(actual, pred), 2)}


# ---- baselines ---------------------------------------------------------------
def seasonal_naive_weekly(train, horizon=HORIZON):
    """
    'Next weeks look like the last observed week.' We take the final 7 days of
    training and repeat them across the horizon. Because 7 days line up by
    weekday, this respects the weekly pattern.
    """
    last7 = train["sales"].to_numpy()[-7:]
    return np.array([last7[i % 7] for i in range(horizon)])


def seasonal_naive_yearly(series_full, test):
    """
    'This period looks like the same dates last year.' For each test date we look
    up the value 364 days earlier (364 = 52 weeks, so the weekday matches too).
    """
    lookup = series_full.set_index("date")["sales"]
    preds = []
    for d in test["date"]:
        prior = d - np.timedelta64(364, "D")
        preds.append(lookup.get(prior, np.nan))
    return np.array(preds, dtype=float)


# ---- machine-learning model --------------------------------------------------
def train_ml_model(train, test):
    """
    A Random Forest that learns the relationship between the features (calendar +
    lagged past) and sales. It is NOT told the answer for the test days -- it
    predicts them from features that only use information >=28 days old.
    """
    cols = feature_columns()
    # import here to avoid import-time issues in environments without sklearn
    # import here to avoid import-time issues in environments without sklearn
    try:
        # use dynamic import to avoid some linters complaining when sklearn
        # isn't installed in the environment
        ensemble = __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"])
        RandomForestRegressor = getattr(ensemble, "RandomForestRegressor")
    except Exception as e:  # pragma: no cover - environment-specific
        raise ImportError("scikit-learn is required to train the ML model") from e

    model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[cols], train["sales"])
    preds = model.predict(test[cols])
    return model, preds
