import pandas as pd
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_RAINFALL = 29.9      # Rainfall in mm
INPUT_WATERLEVEL = 12.0    # Water Level in meters
INPUT_SOIL = 65.2          # Soil Moisture %
INPUT_ELEVATION = 15.0     # Elevation in meters

# Options: "Manila", "Marikina", "Pasig", "Quezon City"
INPUT_LOCATION = "Marikina" 

MODEL_FILE = "Random_Forest_model.pkl"

# ==========================================
# 2. LOAD MODEL
# ==========================================
try:
    model = joblib.load(MODEL_FILE)
    print(f"✅ Loaded {MODEL_FILE}")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{MODEL_FILE}'. Run train_and_save.py first.")
    exit()

# ==========================================
# 3. PREPARE DATA
# ==========================================
print(f"\n--- SCENARIO: {INPUT_LOCATION} ---")
print(f"Rain: {INPUT_RAINFALL}mm | Water: {INPUT_WATERLEVEL}m | Soil: {INPUT_SOIL}%")

# Create the basic dictionary
input_data = {
    'Rainfall_mm': [INPUT_RAINFALL],
    'WaterLevel_m': [INPUT_WATERLEVEL],
    'SoilMoisture_pct': [INPUT_SOIL],
    'Elevation_m': [INPUT_ELEVATION]
}

# --- CRITICAL FIX: MATCHING YOUR DATASET EXACTLY ---
# Your dataset only has: Manila, Marikina, Pasig, Quezon City.
# "Manila" was dropped during training (drop_first=True), so it doesn't get a column.
# The model ONLY expects columns for the others.

# These are the columns the model EXPECTS to see:
model_expected_cities = ["Marikina", "Pasig", "Quezon City"]

for city in model_expected_cities:
    col_name = f"Location_{city}"
    
    # Logic: If user picked this city, put 1. Otherwise 0.
    # If user picked "Manila", ALL of these will be 0 (which is correct).
    if INPUT_LOCATION.lower() == city.lower():
        input_data[col_name] = [1]
    else:
        input_data[col_name] = [0]

# Convert to DataFrame
df_input = pd.DataFrame(input_data)

# ==========================================
# 4. PREDICT & REPORT
# ==========================================
try:
    # We predict!
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0]

    print("\n" + "="*30)
    if prediction == 1:
        confidence = probability[1] * 100
        print(f"🚨 RESULT: FLOOD WARNING (Confidence: {confidence:.1f}%)")
    else:
        confidence = probability[0] * 100
        print(f"✅ RESULT: SAFE (Safety Confidence: {confidence:.1f}%)")
    print("="*30 + "\n")

except ValueError as e:
    print("\n⚠️ ERROR: Column mismatch.")
    print(f"Details: {e}")