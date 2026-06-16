import os
import numpy as np
import joblib
import sys
import time
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 0. LOGGER SETUP ---
class FinalLogger(object):
    def __init__(self, filename="final_optimization_log.txt"):
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
sys.stdout = FinalLogger("final_optimization_log.txt")
print(f"COMPREHENSIVE MODEL EVALUATION REPORT")
print(f"Timestamp: {time.ctime()}")
print("-" * 70)

# --- 1. PHYSICAL SIZE COMPARISON ---
h5_path = 'best_water_surge_lstm.h5'
tflite_path = 'best_water_surge_lstm_float16_builtin.tflite'

print("\n[SECTION 1: STORAGE EFFICIENCY]")
if os.path.exists(h5_path) and os.path.exists(tflite_path):
    h5_size = os.path.getsize(h5_path) / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024
    reduction = ((h5_size - tflite_size) / h5_size) * 100

    print(f"{'Model Format':<30} | {'Size (KB)':<15}")
    print("-" * 50)
    print(f"{'Original Keras (.h5)':<30} | {h5_size:>10.2f} KB")
    print(f"{'Graph-Optimized TFLite (.tflite)':<30} | {tflite_size:>10.2f} KB")
    print("-" * 50)
    print(f"Total Model Size Reduction: {reduction:.2f}%")
else:
    print("Error: Models not found. Run the conversion script first.")

# --- 2. ACCURACY EVALUATION (FLOAT16 GO) ---
print("\n[SECTION 2: PREDICTIVE ACCURACY]")
data = np.load('water_surge_processed_data.npz')
X_test = data['X_test'].astype(np.float32)
y_test = data['y_test']
scaler = joblib.load('min_max_scaler.pkl')

# Allocate Interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

y_pred_go = []
start_time = time.time()

for i in range(len(X_test)):
    interpreter.set_tensor(input_details['index'], X_test[i:i+1])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    y_pred_go.append(output_data[0])

end_time = time.time()
y_pred_go = np.array(y_pred_go)

# De-normalize
y_pred_real = scaler.inverse_transform(y_pred_go)
y_test_real = scaler.inverse_transform(y_test)

sensor_names = ['Flow Rate (L/min)', 'Distance (cm)', 'Temp (C)', 'Turbidity (NTU)']

print("\n" + "="*70)
print(f"{'SENSOR':<20} | {'RMSE':<10} | {'MAE':<10} | {'R2 SCORE'}")
print("-" * 70)

total_r2 = 0
for i in range(len(sensor_names)):
    rmse = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
    mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
    r2 = r2_score(y_test_real[:, i], y_pred_real[:, i])
    total_r2 += r2
    print(f"{sensor_names[i]:<20} | {rmse:<10.4f} | {mae:<10.4f} | {r2:.4f}")

system_avg = total_r2 / 4
print("-" * 70)
print(f"{'SYSTEM AVERAGE':<20} | {'':<10} | {'':<10} | {system_avg:.4f}")
print("="*70)

# --- 3. PERFORMANCE METRICS ---
print("\n[SECTION 3: EDGE PERFORMANCE]")
avg_latency = (end_time - start_time) / len(X_test) * 1000
print(f"Average Inference Latency: {avg_latency:.2f} ms")
print(f"Inference through-put:    {1000/avg_latency:.1f} predictions/sec")
print(f"\nReport saved to 'final_optimization_log.txt'")