import numpy as np
import pandas as pd
from scipy.stats import skew, rankdata

print("Initializing 1-Year Dataset Generation (1-min intervals)...")

# 1. TIME SETUP
timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=525600, freq="1min")
total_mins = len(timestamps)
hour_of_day = timestamps.hour + timestamps.minute / 60.0
month_of_year = timestamps.month

# 2. MALAYSIAN CLIMATE BASELINES
# Monsoon weighting: Nov-Jan and May-Jul
monsoon_weight = np.isin(month_of_year, [11, 12, 1, 5, 6, 7]).astype(float)

base_flow = 5.0 + (monsoon_weight * 2.0)
base_dist = 196.0 - (monsoon_weight * 3.0)
base_ntu = 15.0 + (monsoon_weight * 5.0)

# Diurnal temperature: peak around 2 PM, lowest around 4 AM
base_temp = 28.0 + 5.0 * np.sin((hour_of_day - 8) * 2 * np.pi / 24)

# 3. EVENT PROFILES
surge_profile = np.concatenate([
    np.linspace(0, 1, 15),
    np.ones(75),
    np.linspace(1, 0, 120)
])

h_rain_profile = np.concatenate([
    np.linspace(0, 1, 10),
    np.ones(30),
    np.linspace(1, 0, 45)
])

n_rain_profile = np.concatenate([
    np.linspace(0, 1, 25),
    np.ones(40),
    np.linspace(1, 0, 60)
])

# 4. EVENT TRIGGERING
np.random.seed(42)

surge_triggers = np.zeros(total_mins)
h_rain_triggers = np.zeros(total_mins)
n_rain_triggers = np.zeros(total_mins)

# Daily event check
for i in range(1440, total_mins - 1440, 1440):
    prob_mult = 3.0 if monsoon_weight[i] == 1 else 1.0

    if np.random.rand() < (0.05 * prob_mult):
        surge_triggers[i + np.random.randint(0, 720)] = 1.2 + np.random.rand() * 0.3
    elif np.random.rand() < (0.15 * prob_mult):
        h_rain_triggers[i + np.random.randint(0, 720)] = 0.8 + np.random.rand() * 0.3
    elif np.random.rand() < (0.30 * prob_mult):
        n_rain_triggers[i + np.random.randint(0, 720)] = 0.4 + np.random.rand() * 0.3

# 5. CREATE CONTINUOUS WATER EVENT SIGNAL
surge_signal = np.convolve(surge_triggers, surge_profile, mode="same")
h_rain_signal = np.convolve(h_rain_triggers, h_rain_profile, mode="same")
n_rain_signal = np.convolve(n_rain_triggers, n_rain_profile, mode="same")

total_water_event = surge_signal + h_rain_signal + n_rain_signal
total_water_event = np.clip(total_water_event, 0, 1.5)

# Temperature drops 20 minutes before event peak
temp_drop_signal = np.roll(total_water_event, shift=-20)
temp_drop_signal = np.clip(temp_drop_signal, 0, 1.0)

# 6. APPLY EVENT EFFECTS TO SENSOR PARAMETERS
flow = base_flow + (total_water_event * 18.0)
distance = base_dist - (total_water_event * 50.0)
turbidity = base_ntu + (total_water_event * 380.0)
temperature = base_temp - (temp_drop_signal * 8.0)

# 7. ADD SENSOR NOISE
flow += np.random.normal(0, np.abs(flow * 0.04))
distance += np.random.normal(0, np.abs(distance * 0.02))
turbidity += np.random.normal(0, np.abs(turbidity * 0.05))
temperature += np.random.normal(0, 0.3)

# 8. CLAMP TO PROJECT SENSOR RANGE
flow = np.clip(flow, 3.0, 28.91)
distance = np.clip(distance, 127.99, 200.0)
temperature = np.clip(temperature, 16.60, 38.06)
turbidity = np.clip(turbidity, 5.0, 597.23)

# 9. SKEWNESS BALANCING FUNCTION
def balance_skewness_preserve_order(data, target_min, target_max, blend=0.20):
    """
    This function balances skewness while preserving the original time-order pattern.
    It uses rank-based balancing and rescales the data to the target sensor range.
    """
    data = data.astype(float)

    x_min = np.min(data)
    x_max = np.max(data)
    x_scaled = (data - x_min) / (x_max - x_min)

    ranks = (rankdata(data, method="average") - 1) / (len(data) - 1)

    balanced = (1 - blend) * ranks + blend * x_scaled
    balanced = (balanced - np.min(balanced)) / (np.max(balanced) - np.min(balanced))

    return balanced * (target_max - target_min) + target_min

print("\nOptimizing statistical constraints...")

flow = balance_skewness_preserve_order(flow, 3.0, 28.91, blend=0.20)
distance = balance_skewness_preserve_order(distance, 127.99, 200.0, blend=0.20)
temperature = balance_skewness_preserve_order(temperature, 16.60, 38.06, blend=0.20)
turbidity = balance_skewness_preserve_order(turbidity, 5.0, 597.23, blend=0.20)

# 10. EXPORT CSV
df = pd.DataFrame({
    "Timestamp": timestamps,
    "Flow_L_min": np.round(flow, 2),
    "Distance_cm": np.round(distance, 2),
    "Temp_C": np.round(temperature, 2),
    "Turbidity_NTU": np.round(turbidity, 2)
})

csv_name = "malaysia_1year_surge_datetime.csv"
df.to_csv(csv_name, index=False)

# 11. PRINT STATISTICAL VERIFICATION
print("\n" + "=" * 65)
print("DATASET GENERATION COMPLETE")
print("=" * 65)
print(f"Total Rows (1-min intervals): {len(df)}")

print("\n--- STATISTICAL VERIFICATION ---")
print(f"{'Sensor':<15} | {'Min':>8} | {'Max':>8} | {'Skewness':>10}")
print("-" * 65)

for col in ["Flow_L_min", "Distance_cm", "Temp_C", "Turbidity_NTU"]:
    s = skew(df[col])
    status = "Pass" if abs(s) < 0.5 else "Check"
    print(f"{col:<15} | {df[col].min():>8.2f} | {df[col].max():>8.2f} | {s:>10.4f} ({status})")

print(f"\nDataset successfully saved to: {csv_name}")
