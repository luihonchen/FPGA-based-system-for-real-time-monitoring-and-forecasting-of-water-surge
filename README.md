# FPGA-based-system-for-real-time-monitoring-and-forecasting-of-water-surge

# Training and tuning LSTM
1min_data.py                      -> file to generate synthetic data
**output= 
malaysia_1year_surge_datetime.csv -> one year synthetic data

data_preprocessing.py             -> preprocess the synthetic data spliting, normalized and windowing
**output= 
min_max_scaler.pkl                -> normalized data
water_surge_processed_data.npz    ->full dataset and the expected output

data_tuning_latest.py             -> auto tuning using KerasTuner and train the model using best hyperparamter
**output= 
best_water_surge_lstm.h5          -> full weight LSTM model

test_acc_h5.py                    -> test the h5 file accuracy, file size and latency

optimization_gofloat16_builtin.py -> optimize the file using GO and convert to TFLite file
**output= 
best_water_surge_lstm_float16_builtin -> lightweight LSTM model

test_size_accuracy_tflite.py      -> compare the file size of h5 and tflite and test the tflite file accuracy, latency.

convert_.npz.py                   -> convert the .npz file (need python library to read) to .csv file which can can read by the HPS no need contain the python library.
*output= 
de10_test_data.csv                 ->data set that can be used in HPS

# FPGA development
