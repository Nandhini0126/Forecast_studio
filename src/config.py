"""
config.py
---------
Every setting in one place. Change a value here, not scattered through the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Put retail_sales.csv in the data/ folder next to this project.
DATA_PATH = ROOT / "data" / "retail_sales.csv"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42

# Which single series we forecast (one store + one item).
STORE = "store_1"
ITEM = "item_1"

# How many days ahead we predict.
HORIZON = 28

# Lag features. KEY RULE: every lag must be >= HORIZON (28). Why: to forecast a
# day 28 steps ahead, we can only use information from >=28 days earlier -- using
# a 7-day lag would need data we won't have yet at forecast time (that's leakage).
# 364 = 52 weeks, i.e. "same day, same weekday, last year".
LAGS = [28, 35, 42, 364]
