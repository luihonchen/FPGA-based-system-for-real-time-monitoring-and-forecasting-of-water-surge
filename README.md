# FPGA-Based System for Real-Time Monitoring and Forecasting of Water Surge

This project develops an FPGA-enabled real-time water surge monitoring and forecasting system using the DE10-Nano SoC board. The system acquires sensor data through FPGA hardware, transfers the readings to the HPS through Avalon-MM registers, and performs 5-minute water surge prediction using an optimized TensorFlow Lite LSTM model.

---

## Training and Tuning LSTM Model

| File                                           | Description                                                                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `1min_data.py`                                 | Generates one year of synthetic water surge data with a 1-minute sampling interval based on Malaysian weather conditions.        |
| `malaysia_1year_surge_datetime.csv`            | Output dataset containing one year of synthetic sensor data.                                                                     |
| `data_preprocessing.py`                        | Preprocesses the synthetic dataset by splitting, normalizing, and converting the data into LSTM time-series windows.             |
| `min_max_scaler.pkl`                           | Saved Min-Max scaler used for normalization and inverse transformation.                                                          |
| `water_surge_processed_data.npz`               | Processed dataset containing training, validation, and testing arrays for LSTM model development.                                |
| `data_tuning_latest.py`                        | Performs hyperparameter tuning using KerasTuner and trains the LSTM model using the best hyperparameter combination.             |
| `best_water_surge_lstm.h5`                     | Trained full-precision LSTM model saved in H5 format.                                                                            |
| `test_acc_h5.py`                               | Evaluates the H5 model in terms of prediction accuracy, model size, inference latency, and throughput.                           |
| `optimization_gofloat16_builtin.py`            | Converts the trained H5 model into a lightweight TensorFlow Lite model using graph optimization and Float16 weight conversion.   |
| `best_water_surge_lstm_float16_builtin.tflite` | Optimized lightweight TensorFlow Lite LSTM model for embedded deployment.                                                        |
| `test_size_accuracy_tflite.py`                 | Compares the H5 and TensorFlow Lite models in terms of file size, accuracy, inference latency, and throughput.                   |
| `convert_.npz.py`                        | Converts the processed NPZ testing dataset into CSV format so that it can be read by the HPS without requiring Python libraries. |
| `de10_test_data.csv`                           | CSV testing dataset used for evaluating the TensorFlow Lite model on the DE10-Nano HPS.                                          |

---

## FPGA Development

| File                   | Description                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `DE10_NANO_SoC_GHRD.v` | Top-level DE10-Nano GHRD Verilog file, modified to include the custom sensor acquisition module and HPS-related IP connections. |
| `top_sensors.v`        | Integrates all sensor driver modules and stores the processed sensor readings into Avalon-MM registers.                         |
| `adc_tur_dri.v`        | ADC driver used to read turbidity sensor data through the DE10-Nano onboard ADC.                                                |
| `temp_dri.v`           | DS18B20 temperature sensor driver using the 1-Wire communication protocol.                                                      |
| `flow_rate_dri.v`      | Flow rate sensor driver that counts pulse signals and converts them into flow rate values.                                      |
| `ultrasonic_dri.v`     | Ultrasonic sensor driver used to measure water level distance based on echo pulse duration.                                     |

---

## HPS Code

| File                        | Description                                                                                                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `libtensorflow-lite.a`      | TensorFlow Lite static library version 2.4.1 used to run the optimized LSTM model on the DE10-Nano HPS.                                                                                      |
| `avalon_mm_hps.cpp`         | Reads real-time sensor values from FPGA Avalon-MM registers, including sensor readings, FPGA acquisition latency, and ADC voltage.                                                           |
| `dataset_test_accuracy.cpp` | Evaluates the TensorFlow Lite model on the HPS using the test dataset and calculates accuracy, latency, and throughput.                                                                      |
| `run_lstm_prediction.cpp`   | Main real-time application that reads sensor data from FPGA registers, performs LSTM prediction, compares values with warning thresholds, controls alerts, and uploads data to the database. |

---

## Overall System Flow

1. Generate synthetic water surge data using `1min_data.py`.
2. Preprocess the dataset using `data_preprocessing.py`.
3. Train and tune the LSTM model using `data_tuning_latest.py`.
4. Evaluate the trained H5 model using `test_acc_h5.py`.
5. Optimize the H5 model into TensorFlow Lite format using `optimization_gofloat16_builtin.py`.
6.Evaluate TensorFlow Lite using `test_size_accuracy_tflite.py`.
7. Convert the processed testing dataset into CSV format using `convert_.npz.py`.
8. Implement FPGA sensor acquisition using Verilog sensor drivers.
9. Read FPGA sensor data on the HPS through Avalon-MM registers.
10. Run real-time TensorFlow Lite LSTM prediction on the HPS.
11. Compare raw and predicted values with warning thresholds.
12. Upload data to the database and trigger warning alerts when necessary.
