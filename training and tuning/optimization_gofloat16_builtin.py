import sys
import time
import tensorflow as tf
import numpy as np

# --- 0. LOGGER SETUP ---
class Logger(object):
    def __init__(self, filename="optimization_go_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger("optimization_go_log.txt")

print(f"PHASE 4: GRAPH OPTIMIZATION + FLOAT16 BUILT-INS ONLY - {time.ctime()}")
print("TensorFlow version:", tf.__version__)


# --- 1. LOAD THE MODEL ---
print("Loading rolled/stateless LSTM model...")

model = tf.keras.models.load_model(
    "best_water_surge_lstm.h5",
    compile=False
)

print("Model loaded successfully.")


# --- 2. CONVERSION SETUP ---
print("Configuring converter...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ========================================================
# GRAPH OPTIMIZATION + FLOAT16 WEIGHT CONVERSION
# ========================================================
# This enables graph optimization such as constant folding.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# This stores suitable weights in Float16 to reduce model size.
# Usually, input/output still remain Float32.
converter.target_spec.supported_types = [tf.float16]

# ========================================================
# IMPORTANT:
# Use TFLITE_BUILTINS only.
# Do NOT use SELECT_TF_OPS because your DE10-Nano C++ runtime
# only links normal libtensorflow-lite.a, not Flex delegate.
# ========================================================
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]


# --- 3. CONVERT TO TFLITE ---
try:
    print("Converting to TFLite...")

    tflite_model = converter.convert()

    output_file = "best_water_surge_lstm_float16_builtin.tflite"

    with open(output_file, "wb") as f:
        f.write(tflite_model)

    print(f"✅ SUCCESS: Saved as '{output_file}'")


    # --- 4. VERIFY TFLITE MODEL ---
    print("\nVerifying TFLite model with Python interpreter...")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nInput details:")
    print(input_details)

    print("\nOutput details:")
    print(output_details)

    print("\nInput dtype :", input_details[0]["dtype"])
    print("Output dtype:", output_details[0]["dtype"])

    if input_details[0]["dtype"] != np.float32:
        print("⚠️ WARNING: Input is not Float32. C++ code needs modification.")
    else:
        print("✅ Input is Float32. C++ input code is compatible.")

    if output_details[0]["dtype"] != np.float32:
        print("⚠️ WARNING: Output is not Float32. C++ code needs modification.")
    else:
        print("✅ Output is Float32. C++ output code is compatible.")

except Exception as e:
    print(f"❌ CONVERSION FAILED: {str(e)}")


print(f"COMPLETED AT {time.ctime()}")