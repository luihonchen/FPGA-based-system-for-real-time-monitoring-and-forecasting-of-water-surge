# FPGA-based-system-for-real-time-monitoring-and-forecasting-of-water-surge

1min_data.py           -> file to generate synthetic data

data_preprocessing.py  -> preprocess the synthetic data spliting, normalized and windowing

data_tuning_latest.py  -> auto tuning using KerasTuner and train the model using best hyperparamter

test_acc_h5.py          -> test the h5 file accuracy, file size and latency

optimization_gofloat16_builtin.py -> optimize the file using GO and convert to TFLite file

test_size_accuracy_tflite.py -> compare the file size of h5 and tflite and test the tflite file accuracy, latency.
