# ============================================================
# STEP 1: Feature Pipeline
# What this script does:
#   1. Fetches AQI + pollutant data from AQICN API
#   2. Computes useful features from that raw data
#   3. Saves the features into MongoDB
# ============================================================

import os
import requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------------------------------------------------------
# Load your secret keys from the .env file
# ---------------------------------------------------------
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("AQICN_TOKEN")   # Your AQICN API token
MONGO_URI   = os.getenv("MONGO_URI")     # Your MongoDB connection string

# ---------------------------------------------------------
# CHANGE THIS to your city name (e.g. "karachi", "london")
# ---------------------------------------------------------
CITY = "karachi"

# ============================================================
# FUNCTION 1: Fetch raw data from AQICN API
# ============================================================
def fetch_raw_data(city: str) -> dict:
    """
    Calls the AQICN API and returns raw air quality data.
    Returns a dictionary with AQI and pollutant readings.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={OPENWEATHER_API_KEY}"
    
    print(f"📡 Fetching AQI data for: {city}")
    response = requests.get(url)
    
    # Check if the API call worked
    if response.status_code != 200:
        raise Exception(f"API call failed with status: {response.status_code}")
    
    data = response.json()
    
    # Check if AQICN returned a valid response
    if data["status"] != "ok":
        raise Exception(f"AQICN API error: {data.get('data', 'Unknown error')}")
    
    print("✅ Raw data fetched successfully!")
    return data["data"]   # Return just the 'data' part


# ============================================================
# FUNCTION 2: Compute features from raw data
# ============================================================
def compute_features(raw: dict) -> dict:
    """
    Takes raw API data and extracts + computes useful features.
    These features are what the ML model will learn from.
    """
    now = datetime.utcnow()
    
    # --- Helper: safely get a pollutant value ---
    # Some cities may not have all pollutants, so we use .get() safely
    iaqi = raw.get("iaqi", {})
    
    def get_val(key):
        return iaqi.get(key, {}).get("v", None)  # Returns None if missing
    
    features = {
        # === Identity ===
        "city":        CITY,
        "timestamp":   now,                        # When we fetched this
        
        # === Target variable (what we want to predict) ===
        "aqi":         raw.get("aqi", None),       # The AQI value right now
        
        # === Pollutants (model inputs) ===
        "pm25":        get_val("pm25"),  # Fine particles (most important!)
        "pm10":        get_val("pm10"),  # Coarse particles
        "o3":          get_val("o3"),    # Ozone
        "no2":         get_val("no2"),   # Nitrogen dioxide
        "so2":         get_val("so2"),   # Sulfur dioxide
        "co":          get_val("co"),    # Carbon monoxide
        
        # === Weather features ===
        "temperature": get_val("t"),     # Temperature in °C
        "humidity":    get_val("h"),     # Humidity %
        "wind_speed":  get_val("w"),     # Wind speed
        "pressure":    get_val("p"),     # Atmospheric pressure
        
        # === Time-based features ===
        # These help the model learn daily/weekly patterns
        "hour":        now.hour,          # 0–23 (rush hour vs midnight)
        "day_of_week": now.weekday(),     # 0=Monday, 6=Sunday
        "month":       now.month,         # 1–12 (seasonal patterns)
        "is_weekend":  int(now.weekday() >= 5),  # 1 if Saturday/Sunday
    }
    
    print(f"🔧 Features computed. AQI = {features['aqi']}")
    return features


# ============================================================
# FUNCTION 3: Save features to MongoDB
# ============================================================
def save_to_mongodb(features: dict):
    """
    Connects to MongoDB and inserts the feature record.
    Each call to this function adds one new row to the database.
    """
    client = MongoClient(MONGO_URI)
    
    db         = client["aqi_db"]        # Database name
    collection = db["features"]          # Collection (like a table) name
    
    result = collection.insert_one(features)
    
    print(f"💾 Saved to MongoDB with ID: {result.inserted_id}")
    client.close()


# ============================================================
# FUNCTION 4: Compute AQI change rate (needs last saved record)
# ============================================================
def get_aqi_change_rate(current_aqi: float) -> float:
    """
    Looks at the last saved AQI value in MongoDB,
    and computes how much AQI has changed since then.
    This is a derived feature — useful for the model.
    """
    try:
        client     = MongoClient(MONGO_URI)
        collection = client["aqi_db"]["features"]
        
        # Get the most recent record for this city
        last_record = collection.find_one(
            {"city": CITY, "aqi": {"$ne": None}},
            sort=[("timestamp", -1)]   # Sort by newest first
        )
        client.close()
        
        if last_record and last_record.get("aqi"):
            change_rate = current_aqi - last_record["aqi"]
            print(f"📈 AQI change rate: {change_rate:+.1f}")
            return change_rate
        else:
            return 0.0   # No previous record found
            
    except Exception as e:
        print(f"⚠️  Could not compute change rate: {e}")
        return 0.0


# ============================================================
# MAIN: Run the full pipeline
# ============================================================
def run_pipeline():
    print("=" * 50)
    print("🚀 Starting Feature Pipeline")
    print("=" * 50)
    
    # Step 1: Fetch raw data
    raw = fetch_raw_data(CITY)
    
    # Step 2: Compute features
    features = compute_features(raw)
    
    # Step 3: Add AQI change rate (derived feature)
    if features["aqi"]:
        features["aqi_change_rate"] = get_aqi_change_rate(features["aqi"])
    else:
        features["aqi_change_rate"] = 0.0
    
    # Step 4: Save to MongoDB
    save_to_mongodb(features)
    
    print("=" * 50)
    print("✅ Pipeline complete!")
    print("=" * 50)
    
    return features


# Run when you execute: python feature_pipeline.py
if __name__ == "__main__":
    result = run_pipeline()
    
    # Print a nice summary
    print("\n📊 Summary of saved record:")
    for key, val in result.items():
        if key not in ["_id"]:
            print(f"   {key:20s}: {val}")