"""
features.py
-----------
Loads the data, keeps ONE store-item series, builds forecasting features, and
splits by TIME (never randomly).

The single most important idea in this whole project: to forecast 28 days ahead,
every feature for a given day must be computable using only information from at
least 28 days earlier. That is why all our lag features are >= 28. It guarantees
we never accidentally peek at the future ("data leakage"), which is the mistake
that makes a forecasting model look great in testing and fail in real life.
"""

import pandas as pd

from config import DATA_PATH, STORE, ITEM, HORIZON, LAGS


def load_series():
    """Read the CSV, keep one store+item, return a clean daily series."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Boolean filtering: keep rows where store AND item match.
    s = df[(df["store_id"] == STORE) & (df["item_id"] == ITEM)].copy()
    s = s.sort_values("date").reset_index(drop=True)

    if len(s) == 0:
        raise ValueError(f"No rows for {STORE}/{ITEM} -- check your CSV is the full file.")
    return s


def add_features(s):
    """
    Turn the raw series into a table of features the model can learn from.

    Two kinds of features:
    1. Calendar features (weekday, month, ...): these are known for ANY future
       date, so they're always safe to use.
    2. Lag features (sales 28, 35, 42, 364 days ago): the recent past. All lags
       are >= the 28-day horizon, so they stay leakage-safe (see module note).
    """
    s = s.copy()

    # --- calendar features (always known in advance) ---
    s["weekday"] = s["date"].dt.weekday          # 0=Mon ... 6=Sun
    s["month"] = s["date"].dt.month
    s["day"] = s["date"].dt.day
    s["dayofyear"] = s["date"].dt.dayofyear
    s["is_weekend"] = (s["weekday"] >= 5).astype(int)

    # --- lag features (the past, shifted by >= horizon) ---
    for lag in LAGS:
        s[f"lag_{lag}"] = s["sales"].shift(lag)

    # --- rolling summary of the past, also shifted by the horizon ---
    # .shift(HORIZON) first guarantees the window ends >=28 days ago (no leakage),
    # then we average the 7 days before that -> "what sales looked like ~a month ago".
    shifted = s["sales"].shift(HORIZON)
    s["roll_mean_28"] = shifted.rolling(7).mean()
    s["roll_std_28"] = shifted.rolling(7).std()

    # price and promo are planned in advance by the business, so we may use them.
    # (They're already columns in the data.)

    # Early rows can't have a 364-day lag, so drop rows with missing features.
    s = s.dropna().reset_index(drop=True)
    return s


def feature_columns():
    """The exact list of columns the model trains on (target excluded)."""
    cols = ["weekday", "month", "day", "dayofyear", "is_weekend",
            "price", "promo", "roll_mean_28", "roll_std_28"]
    cols += [f"lag_{lag}" for lag in LAGS]
    return cols


def train_test_split_by_time(s):
    """
    Hold out the LAST `HORIZON` days as the test set; everything before is train.

    This is the time-series version of a train/test split: we always test on the
    most RECENT period, because in real life you forecast the future from the past
    -- never the other way round.
    """
    test = s.iloc[-HORIZON:].copy()
    train = s.iloc[:-HORIZON].copy()
    return train, test
