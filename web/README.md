# Sales Forecasting — Interactive Website

A web front end for the forecasting model. Pick a store and item; the page shows
the recent history, the model's 28-day forecast vs what actually happened, a
"repeat last week" baseline, and the error metrics.

## How it's wired (the thing to understand)

A browser can't run scikit-learn, so this is two pieces:

```
  Browser (dropdowns + chart)  --HTTP-->  FastAPI backend  -->  Random Forest
   web/static/index.html                  web/app.py            (trained per series)
```

When you pick a series, the backend trains the same model from the main project on
that series and returns the numbers; the page draws them with Chart.js. The model
is the real work — the website is a screen around it.

## Run it

```bash
# from the project root, with your venv active:
pip install -r requirements.txt -r web/requirements.txt

# make sure data/retail_sales.csv is the COMPLETE file (see main README)
cd web
uvicorn app:app --reload
# open http://127.0.0.1:8000   (first start takes a few seconds to load the data)
```

## Using it in Antigravity

Open the project, run the command above once so the site works, then direct the
agent for changes — naming the exact file each time. Examples:
- "In web/static/index.html only, add a dropdown to choose the forecast horizon."
- "In web/app.py, add a /api/forecast endpoint variant that returns feature
  importances as a separate chart." Keep the existing /api endpoints stable.

## Honest caveat

The website is presentation, not new ML — same Random Forest underneath. A viva
will test the model: why split by time, what data leakage is, why compare to a
baseline. Those answers are in the main project README's "Viva defense" section.
Read that, not just this page.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/series` | the store and item options |
| POST | `/api/forecast` | history, 28-day forecast, baselines, metrics, top features |
