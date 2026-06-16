import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # To save the scaler

# 1. Load the dataset
# Features: Flow, Distance, Temperature, Turbidity
df = pd.read_csv('malaysia_1year_surge_datetime.csv')
features = ['Flow_L_min', 'Distance_cm', 'Temp_C', 'Turbidity_NTU']
data = df[features].values

# 2. Chronological Splitting (80/10/10)
# Preserving the time-series sequence is critical
n = len(data)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

# SLICE THE DATA
# [start : end]
train_raw = data[:train_end]
val_raw = data[train_end:val_end]
test_raw = data[val_end:]

# 3. Max-Min Normalization (Fit ONLY on Training Set)
# This prevents data leakage from future timestamps
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)  #use max and min of traning set

train_scaled = scaler.transform(train_raw) #apply the scale (use train max and min value ) to validate and test 
val_scaled = scaler.transform(val_raw)
test_scaled = scaler.transform(test_raw)

# 4. Multivariate Windowing Function
# lookback = 15 samples (15 minutes)
# horizon = 5 samples (predicting the state 5 minutes into the future)
def create_multivariate_windows(data, lookback=15, horizon=5):
    X, y = [], []
    # Ensure the loop stops so the target exists for the final window
    for i in range(len(data) - lookback - horizon + 1):
        # Input: All 4 sensors for the last 15 minutes
        X.append(data[i : i + lookback, :])
        
        # Output: All 4 sensors at the specific +5 minute mark
        # Index i + lookback is the first minute after the window.
        # Index i + lookback + horizon - 1 is exactly 5 minutes ahead.
        y.append(data[i + lookback + horizon - 1, :]) 
        
    return np.array(X), np.array(y)

# 5. Generate Tensors
X_train, y_train = create_multivariate_windows(train_scaled)
X_val, y_val     = create_multivariate_windows(val_scaled)
X_test, y_test   = create_multivariate_windows(test_scaled)

# 6. Verification
print(f"Data Preparation Complete (80/10/10 Split)")
print(f"Train: X {X_train.shape} -> y {y_train.shape}")
print(f"Val:   X {X_val.shape} -> y {y_val.shape}")
print(f"Test:  X {X_test.shape} -> y {y_test.shape}")


# 7. Save the Tensors into a single compressed file
# This is very efficient for the 1GB+ of data you might have
np.savez_compressed(
    'water_surge_processed_data.npz', 
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# 8. SAVE THE SCALER (CRITICAL!)
# You MUST save this to de-normalize your 8-bit results later 
# on the DE10-Nano to get real units (L/min, cm, etc.)
joblib.dump(scaler, 'min_max_scaler.pkl')

print("Files Generated Successfully:")
print("- water_surge_processed_data.npz (The Tensors)")
print("- min_max_scaler.pkl (The Normalization Math)")