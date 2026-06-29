"""
app.py
------
Backend for the forecasting website. On startup it loads the dataset ONCE into
memory. For whichever store-item the user picks, it trains the same Random Forest
from the main project and returns the history, the 28-day forecast, the baseline,
and the error metrics for the chart to draw.

Run:  cd web && uvicorn app:app --reload     (first start takes a few seconds to
load the data file)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATA_PATH, HORIZON
from features import add_features, train_test_split_by_time
from forecasting import seasonal_naive_weekly, seasonal_naive_yearly, train_ml_model, all_metrics
HERE = Path(__file__).resolve().parent

# ---- load the dataset once, with memory-friendly dtypes ----------------------
# category dtype for the id columns saves a lot of memory on 4.5M rows.
DATA = pd.read_csv(
    DATA_PATH,
    usecols=["date", "store_id", "item_id", "sales", "price", "promo"],
    dtype={"store_id": "category", "item_id": "category"},
    parse_dates=["date"],
)

_cache = {}   # remember trained results per (store, item) so repeats are instant

app = FastAPI(title="Sales Forecasting")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Pick(BaseModel):
    store: str
    item: str


@app.get("/api/series")
def list_series():
    """Dropdown options: the available stores and items."""
    return {
        "stores": sorted(DATA["store_id"].cat.categories.tolist()),
        "items": sorted(DATA["item_id"].cat.categories.tolist()),
    }


@app.post("/api/forecast")
def forecast(pick: Pick):
    key = (pick.store, pick.item)
    if key in _cache:
        return _cache[key]

    s = DATA[(DATA["store_id"] == pick.store) & (DATA["item_id"] == pick.item)]
    s = s.sort_values("date").reset_index(drop=True)
    if len(s) < 400:
        return {"error": f"Not enough history for {pick.store}/{pick.item}."}

    feat = add_features(s)
    train, test = train_test_split_by_time(feat)
    actual = test["sales"].to_numpy()

    weekly = seasonal_naive_weekly(train)
    yearly = seasonal_naive_yearly(s, test)
    model, rf = train_ml_model(train, test)

    from features import feature_columns
    importances = sorted(zip(feature_columns(), model.feature_importances_),
                         key=lambda x: -x[1])[:5]

    recent = s.iloc[-90:]
    result = {
        "history": {
            "dates": recent["date"].dt.strftime("%Y-%m-%d").tolist(),
            "sales": recent["sales"].astype(float).tolist(),
        },
        "test_dates": test["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": actual.astype(float).tolist(),
        "rf": [round(float(v), 1) for v in rf],
        "weekly": [round(float(v), 1) for v in weekly],
        "metrics": {
            "rf": all_metrics(actual, rf),
            "weekly": all_metrics(actual, weekly),
            "yearly": all_metrics(actual, yearly),
        },
        "top_features": [[c, round(float(v), 3)] for c, v in importances],
        "horizon": HORIZON,
    }
    _cache[key] = result
    return result


# --------------- NEW ENDPOINTS ------------------------------------------------

class ComparePick(BaseModel):
    """Request body for /api/compare – up to 4 store-item pairs."""
    pairs: list  # list of {"store": "store_1", "item": "item_1"}


@app.get("/api/summary")
def summary():
    """Dataset-level summary statistics."""
    total_rows = len(DATA)
    date_min = DATA["date"].min()
    date_max = DATA["date"].max()
    num_stores = DATA["store_id"].cat.categories.size
    num_items = DATA["item_id"].cat.categories.size
    avg_daily_sales = round(float(DATA["sales"].mean()), 1)
    total_sales = int(DATA["sales"].sum())

    return {
        "total_rows": total_rows,
        "date_range": [
            date_min.strftime("%Y-%m-%d"),
            date_max.strftime("%Y-%m-%d"),
        ],
        "num_stores": num_stores,
        "num_items": num_items,
        "avg_daily_sales": avg_daily_sales,
        "total_sales": total_sales,
    }


@app.get("/api/eda")
def eda(store: str = Query(...), item: str = Query(...)):
    """Exploratory-data-analysis payload for a single store-item series."""
    s = DATA[(DATA["store_id"] == store) & (DATA["item_id"] == item)]
    s = s.sort_values("date").reset_index(drop=True)

    if s.empty:
        return {"error": f"No data for {store}/{item}."}

    num_days = int(len(s))
    date_range = [
        s["date"].min().strftime("%Y-%m-%d"),
        s["date"].max().strftime("%Y-%m-%d"),
    ]
    avg_sales = round(float(s["sales"].mean()), 1)
    std_sales = round(float(s["sales"].std()), 1)
    min_sales = int(s["sales"].min())
    max_sales = int(s["sales"].max())

    # --- weekday aggregation (Mon=0 … Sun=6) ---------------------------------
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_means = s.groupby(s["date"].dt.dayofweek)["sales"].mean()
    weekday_values = [round(float(weekday_means.get(i, 0)), 1) for i in range(7)]

    # --- monthly aggregation ---------------------------------------------------
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_means = s.groupby(s["date"].dt.month)["sales"].mean()
    month_values = [round(float(month_means.get(i, 0)), 1) for i in range(1, 13)]

    # Spread = max - min of the aggregated means
    weekday_spread = round(max(weekday_values) - min(weekday_values), 1)
    month_spread = round(max(month_values) - min(month_values), 1)
    dominant_seasonality = "WEEKLY" if weekday_spread >= month_spread else "YEARLY"

    # --- timeline downsampled to weekly averages ------------------------------
    weekly = s.set_index("date").resample("W-MON")["sales"].mean().dropna()
    timeline = {
        "dates": weekly.index.strftime("%Y-%m-%d").tolist(),
        "sales": [round(float(v), 1) for v in weekly.values],
    }

    return {
        "store": store,
        "item": item,
        "num_days": num_days,
        "date_range": date_range,
        "avg_sales": avg_sales,
        "std_sales": std_sales,
        "min_sales": min_sales,
        "max_sales": max_sales,
        "by_weekday": {"labels": weekday_labels, "values": weekday_values},
        "by_month": {"labels": month_labels, "values": month_values},
        "weekday_spread": weekday_spread,
        "month_spread": month_spread,
        "dominant_seasonality": dominant_seasonality,
        "timeline": timeline,
    }


@app.post("/api/compare")
def compare(body: ComparePick):
    """Forecast comparison for up to 4 store-item pairs."""
    pairs = body.pairs[:4]  # hard cap at 4
    results = []

    for p in pairs:
        store = p.get("store", "") if isinstance(p, dict) else ""
        item = p.get("item", "") if isinstance(p, dict) else ""

        # Reuse the existing forecast() logic & cache
        pick = Pick(store=store, item=item)
        resp = forecast(pick)

        if "error" in resp:
            results.append({"store": store, "item": item, "error": resp["error"]})
            continue

        results.append({
            "store": store,
            "item": item,
            "metrics": resp["metrics"],
            "test_dates": resp["test_dates"],
            "actual": resp["actual"],
            "rf": resp["rf"],
            "weekly": resp["weekly"],
        })

    return results


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
