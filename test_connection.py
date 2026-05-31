# ============================================================
# Connection Test Script
# Run this BEFORE feature_pipeline.py to make sure everything works
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
MONGO_URI   = os.getenv("MONGO_URI")

print("=" * 50)
print("Running connection tests...")
print("=" * 50)

# -----------------------------------------------
# TEST 1: Check .env file is loaded correctly
# -----------------------------------------------
print("\nTest 1: Checking .env file...")

if not AQICN_TOKEN:
    print("  FAILED — AQICN_TOKEN is missing in .env file")
elif AQICN_TOKEN == "your_token_here":
    print("  FAILED — You forgot to replace AQICN_TOKEN with your real token")
else:
    print("  PASSED — AQICN_TOKEN found")

if not MONGO_URI:
    print("  FAILED — MONGO_URI is missing in .env file")
elif "your_mongodb_connection_string_here" in MONGO_URI:
    print("  FAILED — You forgot to replace MONGO_URI with your real connection string")
else:
    print("  PASSED — MONGO_URI found")

# -----------------------------------------------
# TEST 2: Check MongoDB connection
# -----------------------------------------------
print("\nTest 2: Connecting to MongoDB...")

try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi

    # Try to connect (timeout after 5 seconds so it doesn't hang)
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)

    # This line actually triggers the connection
    client.admin.command('ping')

    print("  PASSED — Connected to MongoDB successfully!")

    # Try inserting a test document
    db = client["aqi_db"]
    collection = db["features"]
    test_doc = {"test": True, "message": "Connection test successful"}
    result = collection.insert_one(test_doc)
    print(f"  PASSED — Test document inserted with ID: {result.inserted_id}")

    # Clean it up
    collection.delete_one({"_id": result.inserted_id})
    print("  PASSED — Test document cleaned up")

    client.close()

except Exception as e:
    print(f"  FAILED — Could not connect to MongoDB")
    print(f"  Error: {e}")
    print("\n  Possible fixes:")
    print("  - Check your MONGO_URI in the .env file")
    print("  - Make sure you replaced <password> with your real password")
    print("  - Make sure you allowed your IP in MongoDB Atlas (Network Access)")

# -----------------------------------------------
# TEST 3: Check AQICN API
# -----------------------------------------------
print("\nTest 3: Connecting to AQICN API...")

try:
    import requests
    url = f"https://api.waqi.info/feed/karachi/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    data = response.json()

    if data["status"] == "ok":
        aqi_value = data["data"]["aqi"]
        print(f"  PASSED — AQICN API works! Current AQI in Karachi: {aqi_value}")
    else:
        print(f"  FAILED — AQICN returned an error: {data}")
        print("  Fix: Check your AQICN_TOKEN in the .env file")

except Exception as e:
    print(f"  FAILED — Could not reach AQICN API")
    print(f"  Error: {e}")

# -----------------------------------------------
# SUMMARY
# -----------------------------------------------
print("\n" + "=" * 50)
print("If all 3 tests passed, run: python feature_pipeline.py")
print("=" * 50)