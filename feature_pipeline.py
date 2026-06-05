# ============================================================
# STEP 1: Feature Pipeline
# What this script does:
#   1. Fetches AQI + pollutant data from multiple stations
#   2. Computes useful features from that averaged data
#   3. Saves the features into MongoDB
# ============================================================

import os
import requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("AQICN_TOKEN")   
MONGO_URI   = os.getenv("MONGO_URI")     

CITY = "karachi"

def fetch_raw_data(city: str) -> dict:
    """
    Fetches and averages data from multiple Karachi stations 
    to create a true city-wide representation.
    """
    print("📡 Fetching and aggregating multi-station AQI data...")
    stations = ["karachi", "@401143"]
    valid_data = []
    
    for st_id in stations:
        url = f"https://api.waqi.info/feed/{st_id}/?token={OPENWEATHER_API_KEY}"
        try:
            r = requests.get(url, timeout=5).json()
            if r.get("status") == "ok":
                valid_data.append(r["data"])
        except:
            continue
            
    if not valid_data:
        raise Exception("API call failed for all stations.")

    # Average the AQI
    avg_aqi = int(sum(d.get("aqi", 0) for d in valid_data if type(d.get("aqi")) in [int, float]) / len(valid_data))
    
    # Average the pollutants
    avg_iaqi = {}
    for key in ["pm25", "pm10", "o3", "no2", "so2", "co", "t", "h", "w", "p"]:
        vals = [d.get("iaqi", {}).get(key, {}).get("v") for d in valid_data if d.get("iaqi", {}).get(key)]
        vals = [v for v in vals if v is not None]
        if vals:
            avg_iaqi[key] = {"v": round(sum(vals) / len(vals), 1)}

    print(f"✅ Data aggregated! True City Average AQI: {avg_aqi}")
    
    master_data = valid_data[0].copy()
    master_data["aqi"] = avg_aqi
    master_data["iaqi"] = avg_iaqi
    
    return master_data

def compute_features(raw: dict) -> dict:
    now = datetime.utcnow()
    iaqi = raw.get("iaqi", {})
    
    def get_val(key):
        return iaqi.get(key, {}).get("v", None)  
    
    features = {
        "city":        CITY,
        "timestamp":   now,                       
        "aqi":         raw.get("aqi", None),       
        "pm25":        get_val("pm25"),  
        "pm10":        get_val("pm10"),  
        "o3":          get_val("o3"),    
        "no2":         get_val("no2"),   
        "so2":         get_val("so2"),   
        "co":          get_val("co"),    
        "temperature": get_val("t"),     
        "humidity":    get_val("h"),     
        "wind_speed":  get_val("w"),     
        "pressure":    get_val("p"),     
        "hour":        now.hour,          
        "day_of_week": now.weekday(),     
        "month":       now.month,         
        "is_weekend":  int(now.weekday() >= 5),  
    }
    
    print(f"🔧 Features computed. AQI = {features['aqi']}")
    return features

def save_to_mongodb(features: dict):
    client = MongoClient(MONGO_URI)
    db         = client["aqi_db"]        
    collection = db["features"]          
    result = collection.insert_one(features)
    print(f"💾 Saved to MongoDB with ID: {result.inserted_id}")
    client.close()

def get_aqi_change_rate(current_aqi: float) -> float:
    try:
        client     = MongoClient(MONGO_URI)
        collection = client["aqi_db"]["features"]
        last_record = collection.find_one(
            {"city": CITY, "aqi": {"$ne": None}},
            sort=[("timestamp", -1)]   
        )
        client.close()
        
        if last_record and last_record.get("aqi"):
            change_rate = current_aqi - last_record["aqi"]
            print(f"📈 AQI change rate: {change_rate:+.1f}")
            return change_rate
        else:
            return 0.0   
            
    except Exception as e:
        print(f"⚠️  Could not compute change rate: {e}")
        return 0.0

def run_pipeline():
    print("=" * 50)
    print("🚀 Starting Feature Pipeline")
    print("=" * 50)
    
    raw = fetch_raw_data(CITY)
    features = compute_features(raw)
    
    if features["aqi"]:
        features["aqi_change_rate"] = get_aqi_change_rate(features["aqi"])
    else:
        features["aqi_change_rate"] = 0.0
    
    save_to_mongodb(features)
    
    print("=" * 50)
    print("✅ Pipeline complete!")
    print("=" * 50)
    
    return features

if __name__ == "__main__":
    result = run_pipeline()
    print("\n📊 Summary of saved record:")
    for key, val in result.items():
        if key not in ["_id"]:
            print(f"   {key:20s}: {val}")