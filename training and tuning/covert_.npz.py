import numpy as np

print("Loading .npz data...")
data = np.load('water_surge_processed_data.npz')

# Flatten X_test from (N, 15, 4) to (N, 60)
X_flat = data['X_test'].reshape(len(data['X_test']), -1)
y_true = data['y_test']

# Combine them into one giant array of 64 columns, and save as CSV
combined_data = np.hstack((X_flat, y_true))
np.savetxt("de10_test_data.csv", combined_data, delimiter=",", fmt="%.6f")
print("Successfully exported 'de10_test_data.csv' for C++ Evaluation!")