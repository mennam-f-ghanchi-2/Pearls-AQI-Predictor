# ============================================================
# STEP 2 + 3: Training Pipeline + Model Registry
# What this script does:
#   1. Fetches features from MongoDB
#   2. Trains 4 models: Random Forest, Ridge, XGBoost, LightGBM
#   3. Evaluates all 4 with RMSE, MAE, R²
#   4. Saves ALL artifacts locally FIRST (before any network calls)
#   5. Uploads all 4 models to Hopsworks Model Registry
#   6. Marks the best model clearly
#
# Key fix: feature_cols.pkl, aqi_model.pkl, and scaler.pkl are
# saved to disk before the Hopsworks connection is attempted.
# This guarantees the Streamlit app always has consistent artifacts
# regardless of whether the registry upload succeeds or fails.
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Hopsworks
import hopsworks

load_dotenv()

MONGO_URI         = os.getenv("MONGO_URI")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")


# ============================================================
# FUNCTION 1: Load features from MongoDB
# ============================================================
def load_features_from_mongodb():
    print("📦 Loading features from MongoDB...")

    client     = MongoClient(MONGO_URI)
    collection = client["aqi_db"]["features"]
    records    = list(collection.find({"aqi": {"$ne": None}}))
    client.close()

    if len(records) == 0:
        raise Exception("No data found! Run backfill.py first.")

    df = pd.DataFrame(records)
    df = df.drop(columns=["_id"], errors="ignore")

    print(f"✅ Loaded {len(df)} records from MongoDB")
    return df


# ============================================================
# FUNCTION 2: Prepare features and target
# ============================================================
def prepare_data(df):
    print("\n🔧 Preparing data for training...")

    all_possible_features = [
        "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "wind_speed", "pressure",
        "hour", "day_of_week", "month", "is_weekend",
        "aqi_change_rate"
    ]

    # Only use columns that have enough non-null data
    available_cols = []
    for col in all_possible_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 10:
                available_cols.append(col)

    print(f"   Using features: {available_cols}")

    df_work = df[available_cols + ["aqi"]].copy()

    # Fill missing values with median
    for col in available_cols:
        df_work[col] = df_work[col].fillna(df_work[col].median())

    df_clean = df_work.dropna(subset=["aqi"])
    print(f"   Clean records: {len(df_clean)}")

    X = df_clean[available_cols]
    y = df_clean["aqi"]

    return X, y, available_cols


# ============================================================
# FUNCTION 3: Evaluate a model
# ============================================================
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae  = mean_absolute_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)

    print(f"\n   📊 {name}:")
    print(f"      RMSE : {rmse:.4f}  (lower is better)")
    print(f"      MAE  : {mae:.4f}  (lower is better)")
    print(f"      R²   : {r2:.4f}  (closer to 1.0 is better)")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ============================================================
# FUNCTION 4: Train all 4 models
# ============================================================
def train_all_models(X, y):
    print("\n🚀 Training 4 models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set : {len(X_train)} records")
    print(f"   Test set     : {len(X_test)} records")

    # Scale features (needed for Ridge)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = []

    # --- Model 1: Random Forest ---
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest", rf, X_test, y_test)
    results.append({
        "name":         "random_forest",
        "label":        "Random Forest",
        "model":        rf,
        "metrics":      rf_metrics,
        "needs_scaler": False,
        "X_test":       X_test,
    })

    # --- Model 2: Ridge Regression ---
    print("\n📈 Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_metrics = evaluate_model("Ridge Regression", ridge, X_test_scaled, y_test)
    results.append({
        "name":         "ridge_regression",
        "label":        "Ridge Regression",
        "model":        ridge,
        "metrics":      ridge_metrics,
        "needs_scaler": True,
        "X_test":       X_test_scaled,
    })

    # --- Model 3: XGBoost ---
    print("\n⚡ Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_metrics = evaluate_model("XGBoost", xgb, X_test, y_test)
    results.append({
        "name":         "xgboost",
        "label":        "XGBoost",
        "model":        xgb,
        "metrics":      xgb_metrics,
        "needs_scaler": False,
        "X_test":       X_test,
    })

    # --- Model 4: LightGBM ---
    print("\n💡 Training LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_metrics = evaluate_model("LightGBM", lgbm, X_test, y_test)
    results.append({
        "name":         "lightgbm",
        "label":        "LightGBM",
        "model":        lgbm,
        "metrics":      lgbm_metrics,
        "needs_scaler": False,
        "X_test":       X_test,
    })

    # --- Compare all 4 ---
    print("\n" + "=" * 55)
    print("🏆 MODEL COMPARISON")
    print("=" * 55)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 55)
    for r in results:
        m = r["metrics"]
        print(f"{r['label']:<22} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")

    # Pick best model (lowest RMSE)
    best = min(results, key=lambda x: x["metrics"]["rmse"])
    print(f"\n🥇 Best model: {best['label']} (RMSE: {best['metrics']['rmse']:.4f})")

    return results, best, scaler


# ============================================================
# FUNCTION 5: Save ALL models to Hopsworks
#
# Critical fix: all local artifacts (aqi_model.pkl,
# feature_cols.pkl, scaler.pkl, model_info.txt) are written
# to disk BEFORE the Hopsworks connection is attempted.
# Previously these saves were inside the try/except block,
# meaning a network failure would silently skip them and leave
# the Streamlit app with stale or mismatched artifacts.
# ============================================================
def save_all_to_registry(results, best, scaler, feature_cols):

    os.makedirs("model_artifacts", exist_ok=True)

    # ── Step A: Save all local artifacts unconditionally ──────
    # These are written before any network call so the Streamlit
    # app always has a consistent, in-sync set of files.

    joblib.dump(scaler, "model_artifacts/scaler.pkl")
    print("   ✅ Scaler saved         → model_artifacts/scaler.pkl")

    joblib.dump(feature_cols, "model_artifacts/feature_cols.pkl")
    print(f"   ✅ Feature list saved   → model_artifacts/feature_cols.pkl")
    print(f"      Features ({len(feature_cols)}): {feature_cols}")

    # Save the best model as the primary file the Streamlit app loads
    joblib.dump(best["model"], "model_artifacts/aqi_model.pkl")
    print(f"   ✅ Best model saved     → model_artifacts/aqi_model.pkl ({best['label']})")

    with open("model_artifacts/model_info.txt", "w") as f:
        f.write(f"Model type   : {best['name']}\n")
        f.write(f"Is best      : True\n")
        f.write(f"Trained at   : {datetime.utcnow()}\n")
        f.write(f"RMSE         : {best['metrics']['rmse']:.4f}\n")
        f.write(f"MAE          : {best['metrics']['mae']:.4f}\n")
        f.write(f"R2           : {best['metrics']['r2']:.4f}\n")
        f.write(f"Needs scaler : {best['needs_scaler']}\n")
        f.write(f"Features     : {feature_cols}\n")
        f.write(f"Note         : THIS IS THE BEST MODEL\n")
    print("   ✅ Model info saved     → model_artifacts/model_info.txt")

    # ── Step B: Attempt Hopsworks upload ──────────────────────
    # A failure here does NOT affect the local artifacts above.
    print("\n☁️  Connecting to Hopsworks...")
    try:
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        mr = project.get_model_registry()
    except Exception as e:
        print(f"\n⚠️  Could not connect to Hopsworks: {e}")
        print("   Local artifacts were saved successfully.")
        print("   Registry upload skipped. Check HOPSWORKS_API_KEY,")
        print("   HOPSWORKS_PROJECT, and network access to c.app.hopsworks.ai.")
        return

    print(f"\n📤 Uploading all 4 models to Hopsworks registry...")

    for r in results:
        is_best = (r["name"] == best["name"])
        tag     = " ⭐ BEST" if is_best else ""
        print(f"\n   Uploading {r['label']}{tag}...")

        # Temporarily overwrite aqi_model.pkl with this model for upload
        joblib.dump(r["model"], "model_artifacts/aqi_model.pkl")

        with open("model_artifacts/model_info.txt", "w") as f:
            f.write(f"Model type   : {r['name']}\n")
            f.write(f"Is best      : {is_best}\n")
            f.write(f"Trained at   : {datetime.utcnow()}\n")
            f.write(f"RMSE         : {r['metrics']['rmse']:.4f}\n")
            f.write(f"MAE          : {r['metrics']['mae']:.4f}\n")
            f.write(f"R2           : {r['metrics']['r2']:.4f}\n")
            f.write(f"Needs scaler : {r['needs_scaler']}\n")
            f.write(f"Features     : {feature_cols}\n")
            if is_best:
                f.write(f"Note         : THIS IS THE BEST MODEL\n")

        registry_name = f"aqi_{r['name']}"
        description   = f"AQI {r['label']} model"
        if is_best:
            description += " - BEST PERFORMING MODEL"

        hops_model = mr.python.create_model(
            name=registry_name,
            metrics={
                "rmse": round(r["metrics"]["rmse"], 4),
                "mae":  round(r["metrics"]["mae"],  4),
                "r2":   round(r["metrics"]["r2"],   4),
            },
            description=description,
        )
        hops_model.save("model_artifacts")
        print(f"   ✅ {r['label']} uploaded as '{registry_name}'")

    # Restore the best model as the canonical aqi_model.pkl
    joblib.dump(best["model"], "model_artifacts/aqi_model.pkl")
    print(f"\n✅ All 4 models uploaded to Hopsworks!")
    print(f"   aqi_model.pkl restored to best model: {best['label']}")
    print(f"   Go to Hopsworks → Model Registry to see all 4 models.")


# ============================================================
# MAIN
# ============================================================
def run_training_pipeline():
    print("=" * 55)
    print("🚀 Starting Training Pipeline — 4 Models")
    print("=" * 55)

    df                    = load_features_from_mongodb()
    X, y, feature_cols    = prepare_data(df)
    results, best, scaler = train_all_models(X, y)
    save_all_to_registry(results, best, scaler, feature_cols)

    print("\n" + "=" * 55)
    print("✅ Training Pipeline Complete!")
    print(f"   Models trained : 4")
    print(f"   Best model     : {best['label']}")
    print(f"   Best RMSE      : {best['metrics']['rmse']:.4f}")
    print(f"   Best R²        : {best['metrics']['r2']:.4f}")
    print("=" * 55)
    print("\n👉 Check Hopsworks Model Registry to see all 4 models!")


if __name__ == "__main__":
    run_training_pipeline()