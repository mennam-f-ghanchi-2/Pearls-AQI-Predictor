# ============================================================
# BACKFILL SCRIPT
# What this script does:
#   Fetches historical AQI data for past dates and saves
#   them all to MongoDB so we have enough data to train
#   our ML model properly.
#
# AQICN free API gives us past 7 days of hourly data.
# We will loop through each hour and save it as a record.
# ============================================================

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import time

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
MONGO_URI   = os.getenv("MONGO_URI")
CITY        = "karachi"

# ============================================================
# FUNCTION 1: Fetch current AQI data (same as feature pipeline)
# ============================================================
def fetch_raw_data():
    """Fetch current AQI data from AQICN API"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    data = response.json()

    if data["status"] != "ok":
        return None

    return data["data"]


# ============================================================
# FUNCTION 2: Build a feature record from raw data + a timestamp
# ============================================================
def build_feature_record(raw, timestamp):
    """
    Builds a feature dictionary from raw API data.
    We pass a custom timestamp so we can simulate past dates.
    """
    iaqi = raw.get("iaqi", {})

    def get_val(key):
        return iaqi.get(key, {}).get("v", None)

    return {
        "city":            CITY,
        "timestamp":       timestamp,
        "aqi":             raw.get("aqi", None),
        "pm25":            get_val("pm25"),
        "pm10":            get_val("pm10"),
        "o3":              get_val("o3"),
        "no2":             get_val("no2"),
        "so2":             get_val("so2"),
        "co":              get_val("co"),
        "temperature":     get_val("t"),
        "humidity":        get_val("h"),
        "wind_speed":      get_val("w"),
        "pressure":        get_val("p"),
        "hour":            timestamp.hour,
        "day_of_week":     timestamp.weekday(),
        "month":           timestamp.month,
        "is_weekend":      int(timestamp.weekday() >= 5),
        "aqi_change_rate": 0.0,   # set to 0 for backfill records
        "is_backfill":     True,  # mark as backfill so we know later
    }


# ============================================================
# FUNCTION 3: Add synthetic variation to simulate past data
# ============================================================
def add_variation(record, hour, day_offset):
    """
    Since AQICN free API only gives current data,
    we add realistic variation to simulate different hours/days.

    Real AQI patterns:
    - Higher in morning rush hour (7-9am) and evening (5-8pm)
    - Lower at night and midday
    - Slightly higher on weekdays vs weekends
    - Random noise to make it realistic
    """
    import random
    import math

    base_aqi = record["aqi"]
    if base_aqi is None:
        return record

    # Time of day pattern (rush hour effect)
    hour_factor = 1.0 + 0.3 * math.sin((hour - 8) * math.pi / 12)

    # Day of week pattern (weekdays more polluted)
    day_factor = 1.1 if record["is_weekend"] == 0 else 0.9

    # Random noise ±15%
    noise = random.uniform(0.85, 1.15)

    # Seasonal variation (higher in winter months)
    month = record["month"]
    season_factor = 1.2 if month in [11, 12, 1, 2] else 1.0

    new_aqi = base_aqi * hour_factor * day_factor * noise * season_factor
    new_aqi = max(0, round(new_aqi, 1))  # AQI can't be negative

    record["aqi"] = new_aqi

    # Also vary pollutants slightly
    for key in ["pm25", "pm10", "o3", "no2"]:
        if record.get(key) is not None:
            record[key] = round(record[key] * noise * hour_factor, 2)

    # Compute AQI change rate vs previous record
    record["aqi_change_rate"] = round(random.uniform(-15, 15), 2)

    return record


# ============================================================
# FUNCTION 4: Save a batch of records to MongoDB
# ============================================================
def save_batch_to_mongodb(records):
    """Save a list of feature records to MongoDB at once"""
    if not records:
        return

    client     = MongoClient(MONGO_URI)
    collection = client["aqi_db"]["features"]
    result     = collection.insert_many(records)
    client.close()

    return len(result.inserted_ids)


# ============================================================
# FUNCTION 5: Check how many records already exist
# ============================================================
def count_existing_records():
    client     = MongoClient(MONGO_URI)
    collection = client["aqi_db"]["features"]
    count      = collection.count_documents({"city": CITY})
    client.close()
    return count


# ============================================================
# MAIN: Run the backfill
# ============================================================
def run_backfill(days_back=90):
    """
    Generates synthetic historical data for the past N days.
    Default is 90 days (3 months) — good enough to train a model.

    For each day, we create 24 records (one per hour).
    Total records = days_back × 24
    Example: 90 days × 24 hours = 2,160 records
    """
    print("=" * 55)
    print("🔄 Starting Backfill")
    print(f"   Generating data for past {days_back} days")
    print(f"   Total records to create: {days_back * 24}")
    print("=" * 55)

    # First fetch current real data to use as base
    print("\n📡 Fetching current AQI data as base...")
    raw = fetch_raw_data()

    if raw is None:
        print("❌ Could not fetch AQI data. Check your AQICN_TOKEN.")
        return

    current_aqi = raw.get("aqi", 100)
    print(f"✅ Current AQI in {CITY}: {current_aqi}")

    # Check existing records
    existing = count_existing_records()
    print(f"📊 Existing records in MongoDB: {existing}")

    # Generate records for each hour of each past day
    now         = datetime.utcnow()
    all_records = []
    total_days  = 0

    print(f"\n⏳ Generating {days_back} days of hourly data...")

    for day_offset in range(days_back, 0, -1):
        day_records = []

        for hour in range(24):
            # Create a timestamp for this past hour
            timestamp = now - timedelta(days=day_offset, hours=(23 - hour))

            # Build base feature record
            record = build_feature_record(raw, timestamp)

            # Add realistic variation
            record = add_variation(record, hour, day_offset)

            day_records.append(record)

        all_records.extend(day_records)
        total_days += 1

        # Save in batches of 7 days (168 records) to avoid memory issues
        if len(all_records) >= 168:
            saved = save_batch_to_mongodb(all_records)
            print(f"   💾 Saved batch: {total_days} days done ({total_days * 24} records)")
            all_records = []

    # Save any remaining records
    if all_records:
        save_batch_to_mongodb(all_records)

    # Final count
    final_count = count_existing_records()

    print("\n" + "=" * 55)
    print("✅ Backfill Complete!")
    print(f"   Records in MongoDB now: {final_count}")
    print(f"   Ready to train your ML model!")
    print("=" * 55)
    print("\n👉 Next step: run python training_pipeline.py")


if __name__ == "__main__":
    # Change days_back to however many days you want
    # 90 days = good enough for training
    # 180 days = even better
    run_backfill(days_back=90)