"""
main.py
-------
Runs the whole forecasting project end to end:

  1. EDA: plot the series, find which seasonality (weekly vs yearly) dominates
  2. Build features + time-based split (last 28 days held out)
  3. Baselines: weekly seasonal-naive and yearly seasonal-naive
  4. ML model: Random Forest on calendar + lag features
  5. Compare all three on the 28-day test window, save a forecast plot

Run from the project root:   python src/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, STORE, ITEM, HORIZON
from features import (load_series, add_features, train_test_split_by_time)
from forecasting import (seasonal_naive_weekly, seasonal_naive_yearly,
                         train_ml_model, all_metrics)


def line(t):
    print("\n" + "=" * 68 + "\n" + t + "\n" + "=" * 68)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------- 1. EDA
    line(f"1. EDA  ({STORE} / {ITEM})")
    s = load_series()
    print(f"{len(s)} daily rows, {s['date'].min().date()} to {s['date'].max().date()}")

    by_weekday = s.groupby(s["date"].dt.weekday)["sales"].mean()
    by_month = s.groupby(s["date"].dt.month)["sales"].mean()
    # "Which seasonality is stronger?" = which grouping spreads the averages more.
    weekday_spread = by_weekday.max() - by_weekday.min()
    month_spread = by_month.max() - by_month.min()
    print(f"Weekday avg spread: {weekday_spread:.1f}  |  Month avg spread: {month_spread:.1f}")
    dominant = "WEEKLY" if weekday_spread >= month_spread else "YEARLY"
    print(f"-> {dominant} seasonality dominates. That justifies the baseline choice.")

    # EDA plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(s["date"], s["sales"], lw=0.6, color="#185FA5"); ax[0].set_title("Sales over time")
    by_weekday.plot(kind="bar", ax=ax[1], color="#1D9E75")
    ax[1].set_title("Avg sales by weekday (0=Mon)"); ax[1].set_xlabel("")
    by_month.plot(kind="bar", ax=ax[2], color="#BA7517")
    ax[2].set_title("Avg sales by month"); ax[2].set_xlabel("")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "01_eda.png", dpi=120); plt.close()

    # ------------------------------------------------- 2. features + split
    line("2. FEATURES + TIME-BASED SPLIT")
    feat = add_features(s)
    train, test = train_test_split_by_time(feat)
    print(f"Train: {len(train)} rows up to {train['date'].max().date()}")
    print(f"Test : {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    print("All features use data >=28 days old -> no future leakage.")

    actual = test["sales"].to_numpy()

    # -------------------------------------------------------- 3. baselines
    line("3. BASELINES (the bar to beat)")
    wk = seasonal_naive_weekly(train)
    yr = seasonal_naive_yearly(s, test)
    m_wk, m_yr = all_metrics(actual, wk), all_metrics(actual, yr)
    print("Weekly seasonal-naive:", m_wk)
    print("Yearly seasonal-naive:", m_yr)

    # ---------------------------------------------------------- 4. ML model
    line("4. MACHINE-LEARNING MODEL (Random Forest)")
    model, ml = train_ml_model(train, test)
    m_ml = all_metrics(actual, ml)
    print("Random Forest:", m_ml)

    # top features the model relied on
    cols = train[["weekday", "month", "day", "dayofyear", "is_weekend",
                  "price", "promo", "roll_mean_28", "roll_std_28",
                  "lag_28", "lag_35", "lag_42", "lag_364"]].columns
    importances = sorted(zip(cols, model.feature_importances_),
                         key=lambda x: -x[1])[:5]
    print("Top 5 features:", [(c, round(v, 3)) for c, v in importances])

    # ------------------------------------------------------- 5. compare + plot
    line("5. RESULT")
    best_base = min(m_wk["MAE"], m_yr["MAE"])
    verdict = ("BEATS" if m_ml["MAE"] < best_base else "does NOT beat")
    print(f"Random Forest MAE {m_ml['MAE']} vs best baseline MAE {best_base}")
    print(f"-> The model {verdict} the baseline.")
    print("Report this honestly. If it doesn't beat the baseline, that is itself a")
    print("valid finding -- it means the simple rule is hard to improve on here.")

    # Forecast plot: recent history + the 28-day test actual vs forecasts
    recent = s.iloc[-90:]
    plt.figure(figsize=(11, 4.5))
    plt.plot(recent["date"], recent["sales"], color="#888780", lw=0.9, label="history")
    plt.plot(test["date"], actual, color="#042C53", lw=2, marker="o", ms=3, label="actual")
    plt.plot(test["date"], ml, color="#D85A30", lw=2, label="Random Forest")
    plt.plot(test["date"], wk, color="#1D9E75", lw=1.5, ls="--", label="weekly naive")
    plt.axvline(test["date"].iloc[0], color="#B4B2A9", ls=":", lw=1)
    plt.title(f"28-day forecast vs actual ({STORE}/{ITEM})")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_forecast.png", dpi=120); plt.close()

    line("DONE")
    print(f"Plots saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
