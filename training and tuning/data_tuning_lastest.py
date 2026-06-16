import os
import time
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# --- 0. LOGGER SETUP (Saves Terminal Output to File) ---
class Logger(object):
    def __init__(self, filename="tuning_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Initialize Logger
sys.stdout = Logger()

# --- 1. DATA LOADING ---
print("START: Loading preprocessed data...")
data_package = np.load('water_surge_processed_data.npz')
X_train, y_train = data_package['X_train'], data_package['y_train']
X_val, y_val = data_package['X_val'], data_package['y_val']
print(f"DONE: Data Loaded. Training: {X_train.shape} | Validation: {X_val.shape}")

# --- 2. DYNAMIC MODEL BUILDER ---
def build_model(hp):
    # Retrieve hyperparameters for this specific trial
    n_layers = hp.Int('num_layers', 1, 1)
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    # EXPLICIT LOGGING: Prints at the beginning of each trial
    print("\n" + "-"*30)
    print(f" BUILDING MODEL FOR TRIAL ")
    print(f" Layers: {n_layers} | Learning Rate: {lr}")
    
    model = keras.Sequential()
    for i in range(n_layers):
        units = hp.Int(f'units_{i}', 32, 128, step=32)
        drop = hp.Float(f'dropout_{i}', 0.1, 0.4, step=0.1)
        print(f" > Layer {i+1}: Units={units}, Dropout={drop:.1f}")
        
        is_last = (i == n_layers - 1)
        if i == 0:
            model.add(layers.LSTM(units=units, input_shape=(15, 4), return_sequences=not is_last))
        else:
            model.add(layers.LSTM(units=units, return_sequences=not is_last))
        model.add(layers.Dropout(drop))

    model.add(layers.Dense(4))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    print("-" * 30)
    return model

# --- 3. TUNER INITIALIZATION ---
tuner = kt.RandomSearch(
    build_model, objective='val_loss', max_trials=10,
    directory='fyp_logs', project_name='surge_optimization'
)

# Early stopping is the 'referee' that manages epochs
#stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
stop_early = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)
#    restore_best_weights=True
# --- 4. EXECUTE SEARCH ---
print("\n>>> STARTING HYPERPARAMETER SEARCH (All output stored in tuning_log.txt)")
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), 
             callbacks=[stop_early], verbose=2)

# --- 5. DETAILED RESULTS TABULATION (Simplified / No Time Calculation) ---
print("\n" + "="*85)
print(f"{'TRIAL ID':<10} | {'STOP EPOCH':<10} | {'BEST MSE':<12} | {'HYPERPARAMETERS'}")
print("-" * 85)

all_trials = tuner.oracle.trials
best_tid = ""
min_loss = float('inf')

for tid, trial in all_trials.items():
    # 1. Safely get the score (Validation MSE)
    score = trial.score if trial.score is not None else 999.0
    
    # 2. Safely get the best epoch (where it stopped)
    best_epoch = trial.best_step + 1 if trial.best_step is not None else "N/A"
    
    # 3. Get Hyperparameters
    hps = trial.hyperparameters.values
    u = hps.get('units_0', '??')
    lr = hps.get('learning_rate', '??')
    dr = hps.get('dropout_0', '??')
    
    config_str = f"LR:{lr} | Units:{u} | Drop:{dr}"
    
    print(f"{tid:<10} | {best_epoch:<10} | {score:<12.6f} | {config_str}")
    
    # Keep track of the winner
    if score < min_loss and score != 999.0:
        min_loss = score
        best_tid = tid

print("="*85)

# --- 6. SAVE BEST MODEL ---
if best_tid:
    print(f"WINNER: Trial {best_tid} with MSE: {min_loss:.6f}")
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('best_water_surge_lstm.h5')
    print("SUCCESS: 'best_water_surge_lstm.h5' saved.")
else:
    print("WARNING: No successful trials found to save.")