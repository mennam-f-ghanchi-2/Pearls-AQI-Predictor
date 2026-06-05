# AQI Predictor — Karachi

> A real-time air quality intelligence system powered by multi-station aggregation, ML model comparison, and SHAP explainability.

[Live App](https://pearls-aqi-predictor-kci3mkvihenbkucn5azjmp.streamlit.app/)


---

## What This Project Does

- Fetches and averages AQI data from **multiple Karachi stations** via the AQICN API
- Runs a **feature pipeline** to compute pollutant readings and store them in MongoDB
- **Backfills 90 days** of synthetic-varied historical data for model training
- Trains **4 regression models** and registers all of them to Hopsworks Model Registry
- Serves a **dark-themed Streamlit dashboard** with live AQI, 7-day history, pollutant charts, 5-day forecast, EDA, and SHAP feature importance

---

##  Architecture

```
AQICN API → MongoDB → 4 ML Models → Hopsworks Registry → Streamlit App
```

---

## Project Structure

```
karachi-aqi-predictor/
├── feature_pipeline.py       # Step 1  — fetch + store features hourly
├── backfill.py               # One-time — generate 90d of training data
├── training_pipeline.py      # Step 2+3 — train + register 4 models
├── streamlit_run_app.py      # Dashboard
├── test_connection.py        # Verify env before running
├── requirements.txt
└── .env                      # Your secrets (not committed)
```

---

##  Setup

### Prerequisites

- Python 3.10+
- Free AQICN API token → [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/)
- MongoDB Atlas cluster (free tier works) → [mongodb.com/atlas](https://www.mongodb.com/atlas)
- Hopsworks account (free tier) → [hopsworks.ai](https://www.hopsworks.ai)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Your `.env` File

```env
AQICN_TOKEN=your_token_here
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=your_project_name
```

| Variable | Description |
|---|---|
| `AQICN_TOKEN` | Your AQICN API token |
| `MONGO_URI` | MongoDB Atlas connection string |
| `HOPSWORKS_API_KEY` | From Hopsworks account settings |
| `HOPSWORKS_PROJECT` | Your Hopsworks project name |

---

## Run Order

### Step 0 — Test Connections

Run this first to verify all credentials and external services work.

```bash
python test_connection.py
```

Checks: AQICN_TOKEN and MONGO_URI are set, MongoDB ping + insert/delete, live AQICN API response.

---

### Step 1A — Backfill Historical Data *(run once)*

Generates 90 days × 24 hours = **2,160 records** in MongoDB, based on today's live station readings with realistic variation applied.

```bash
python backfill.py
```

- Fetches current multi-station average as a base
- Applies hour-of-day, weekday, seasonal, and random noise factors
- Saves in batches of 168 records to the MongoDB `features` collection

---

### Step 1B — Feature Pipeline *(run hourly)*

Fetches live data, averages two Karachi stations, computes features, and appends one record to MongoDB.

```bash
python feature_pipeline.py
```

- Averages AQI and pollutants from `karachi` + `@401143` stations
- Computes `aqi_change_rate` by diffing against the last stored value
- Saves timestamp, all pollutants, and time features to MongoDB

> **Tip:** Schedule this with cron or a GitHub Actions workflow to keep data fresh.

---

### Step 2 — Train Models

Loads all MongoDB records, trains 4 regression models, evaluates each with RMSE / MAE / R², and pushes all 4 to Hopsworks Model Registry.

```bash
python training_pipeline.py
```

- Only uses feature columns with more than 10 non-null values
- Applies `StandardScaler` for Ridge; tree models use raw features
- The best model (lowest RMSE) is flagged in the registry description

---

### Step 3 — Launch the Dashboard

```bash
streamlit run streamlit_run_app.py
```

---

## 🤖 Models

Four models are trained and compared on every run:

| Model | Type | Notes |
|---|---|---|
| Random Forest | Tree-based | `n_estimators=100`, `max_depth=10` |
| Ridge Regression | Linear | `alpha=1.0`, scaled inputs |
| XGBoost | Gradient boosting | `lr=0.1`, `max_depth=6` |
| LightGBM | Gradient boosting | `lr=0.1`, `max_depth=6` |

All 4 are saved to Hopsworks. The model with the lowest RMSE is marked as best.

### Input Features

```
```
pm25  pm10  o3  no2  so2  co
temperature  humidity  wind_speed  pressure
hour  day_of_week  month  is_weekend  aqi_change_rate
## 📊 Model Performance

Four regression models were trained and registered to Hopsworks Model Registry:

| Model | MAE | R² | RMSE |
|---|---|---|---|
| **Ridge Regression** ⭐ | **14.08** | **0.5754** | **14.59** |
| Random Forest | 14.20 | 0.5433 | 15.13 |
| LightGBM | 14.18 | 0.5357 | 15.26 |
| XGBoost | 14.17 | 0.5294 | 15.36 |

> ✅ **Best Model:** Ridge Regression — lowest MAE and RMSE, highest R²  
> All models are registered in Hopsworks. The best model is used for inference in the Streamlit app.

---

## 📊 Dashboard

The Streamlit app includes:

- **Live AQI hero card** — color-coded category and change rate indicator
- **Pollutant metric cards** — PM2.5, PM10, O3, NO2, SO2, CO, temperature, humidity, wind, pressure
- **5-day forecast** — using the best registered model from Hopsworks
- **7-day historical trend** — Plotly chart, color-banded by AQI severity
- **Pollutant breakdown** — bar chart of current pollutant levels
- **EDA section** — hourly patterns, weekday vs weekend, AQI distribution
- **SHAP feature importance** — TreeExplainer chart showing what drives predictions
- **AQI scale reference** — color legend from 0 (Good) to 500 (Hazardous)

The app uses `streamlit-autorefresh` to pull new data periodically without a manual reload.

---

## 📸 App Screenshots

### Exploratory Data Analysis — Hourly & Weekly Patterns

![AQI by Hour and Day of Week](eda_hourly_weekly.png)

> AQI peaks in the early morning hours (0–4am) and drops mid-day. Thursday shows the highest average AQI among weekdays.

### AQI Distribution & Weekday vs Weekend

![AQI Distribution and Weekend Comparison](eda_distribution_weekend.png)

> Most AQI readings fall in the 60–80 range. Weekday AQI averages around 74.5 — weekend data was not present in this sample window.

### SHAP Feature Importance

![SHAP Feature Importance](eda_shap.png)

> **PM2.5 dominates predictions** with a SHAP value of 11.755 — far ahead of `day_of_week` (2.747) and `month` (1.257). Weather features like temperature, humidity, and wind speed had negligible impact on this dataset.

---

## 🎨 AQI Scale

| Range | Category | Who's at risk |
|---|---|---|
| 0–50 | 🟢 Good | Air quality is satisfactory |
| 51–100 | 🟡 Moderate | Acceptable for most people |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Sensitive people at risk |
| 151–200 | 🔴 Unhealthy | Everyone may be affected |
| 201–300 | 🟣 Very Unhealthy | Health alert for everyone |
| 301–500 | ⚫ Hazardous | Emergency conditions |

---

## 📦 Requirements

```
streamlit
plotly
pandas
numpy
pymongo
requests
python-dotenv
joblib
scikit-learn
xgboost
lightgbm
shap
matplotlib
pyarrow
hopsworks==3.7.0
streamlit-autorefresh
```


