import os
import numpy as np
import joblib
import sys
import time
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 0. LOGGER SETUP ---
class BaselineLogger(object):
    def __init__(self, filename="h5_sequential_performance_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect output to the log file
sys.stdout = BaselineLogger("h5_sequential_performance_log.txt")
print(f"HEAVYWEIGHT (.H5) SEQUENTIAL PERFORMANCE REPORT")
print(f"Timestamp: {time.ctime()}")
print("-" * 70)

# --- 1. LOAD MODEL AND DATA ---
h5_path = 'best_water_surge_lstm.h5'
print(f"\n[SECTION 1: MODEL SPECIFICATIONS]")
if os.path.exists(h5_path):
    h5_size = os.path.getsize(h5_path) / 1024
    print(f"{'Model Format':<30} | {'Size (KB)':<15}")
    print("-" * 50)
    print(f"{'Original Keras (.h5)':<30} | {h5_size:>10.2f} KB")
else:
    print("Error: .h5 model not found.")
    sys.exit()

# Load test data and scaler from your data directory
data_package = np.load("water_surge_processed_data.npz")
scaler = joblib.load('min_max_scaler.pkl')
model = load_model(h5_path)

X_test = data_package['X_test']
y_test = data_package['y_test']
num_samples = len(X_test)

# --- 2. SEQUENTIAL INFERENCE (MIRRORING TFLITE METHOD) ---
print("\n[SECTION 2: SEQUENTIAL INFERENCE PERFORMANCE]")
print(f"Running inference on {num_samples} samples individually...")

y_pred_list = []
start_time = time.time()

for i in range(num_samples):
    # Select a single sample and keep the 3D shape (1, 15, 4)
    sample = X_test[i:i+1]
    
    # Perform prediction
    # training=False ensures dropout/batchnorm are in inference mode
    prediction = model(sample, training=False)
    y_pred_list.append(prediction.numpy()[0])

end_time = time.time()
y_pred_normalized = np.array(y_pred_list)

# Calculations
total_inference_time = end_time - start_time
avg_latency = (total_inference_time / num_samples) * 1000
throughput = num_samples / total_inference_time

print(f"Average Inference Latency: {avg_latency:.2f} ms")
print(f"Inference Throughput:      {throughput:.2f} predictions/sec")

# --- 3. ACCURACY EVALUATION ---
print("\n[SECTION 3: PREDICTIVE ACCURACY]")
# De-normalize
y_pred_real = scaler.inverse_transform(y_pred_normalized)
y_test_real = scaler.inverse_transform(y_test)

sensor_names = ['Flow Rate (L/min)', 'Distance (cm)', 'Temp (C)', 'Turbidity (NTU)']

print("\n" + "="*70)
print(f"{'SENSOR':<20} | {'RMSE':<10} | {'MAE':<10} | {'R2 SCORE'}")
print("-" * 70)

total_r2 = 0
for i in range(len(sensor_names)):
    actual = y_test_real[:, i]
    pred = y_pred_real[:, i]
    
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    total_r2 += r2
    
    print(f"{sensor_names[i]:<20} | {rmse:<10.4f} | {mae:<10.4f} | {r2:.4f}")

system_avg_r2 = total_r2 / 4
print("-" * 70)
print(f"{'SYSTEM AVERAGE':<20} | {'':<10} | {'':<10} | {system_avg_r2:.4f}")
print("="*70)

print(f"\nReport saved to 'h5_sequential_performance_log.txt'")