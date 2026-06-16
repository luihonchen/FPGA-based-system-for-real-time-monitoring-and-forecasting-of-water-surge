#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <unistd.h>
#include <memory>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace std;

const int TIME_STEPS = 15;
const int FEATURES = 4;
const int OUTPUTS = 4;
const int INPUT_SIZE = TIME_STEPS * FEATURES;
const int TOTAL_COLUMNS = INPUT_SIZE + OUTPUTS;

// Scaling constants from your scaler
const float MIN_VALS[4] = {3.73f, 8.27f, 17.56f, 24.65f};
const float MAX_VALS[4] = {16.05f, 21.81f, 33.06f, 183.88f};

const string SENSOR_NAMES[4] = {
    "Flow Rate (L/min)",
    "Distance (cm)",
    "Temp (C)",
    "Turbidity (NTU)"
};

bool parse_csv_line(const string& line, vector<float>& values) {
    values.clear();

    stringstream ss(line);
    string token;

    while (getline(ss, token, ',')) {
        if (token.empty()) {
            return false;
        }

        try {
            values.push_back(stof(token));
        } catch (...) {
            return false;
        }
    }

    return true;
}

float inverse_scale(float scaled_value, int index) {
    return scaled_value * (MAX_VALS[index] - MIN_VALS[index]) + MIN_VALS[index];
}

int main() {
    cout << "=================================================" << endl;
    cout << "  DE10-NANO REGRESSION PERFORMANCE EVALUATION    " << endl;
    cout << "=================================================" << endl;

    string model_path = "../LSTM/best_water_surge_lstm_float16_builtin.tflite";
    string csv_path = "de10_test_data.csv";

    cout << "[MODEL] " << model_path << endl;
    cout << "[CSV]   " << csv_path << endl;

    // ============================================================
    // 1. Load TFLite model
    // ============================================================
    unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (!model) {
        cerr << "[FATAL] Failed to load model!" << endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<tflite::Interpreter> interpreter;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter) {
        cerr << "[FATAL] Failed to build interpreter!" << endl;
        return -1;
    }

    interpreter->SetNumThreads(1);

    if (interpreter->ResizeInputTensor(interpreter->inputs()[0],
                                       {1, TIME_STEPS, FEATURES}) != kTfLiteOk) {
        cerr << "[FATAL] Failed to resize input tensor!" << endl;
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "[FATAL] Failed to allocate tensors!" << endl;
        return -1;
    }

    TfLiteTensor* input_info = interpreter->tensor(interpreter->inputs()[0]);
    TfLiteTensor* output_info = interpreter->tensor(interpreter->outputs()[0]);

    if (input_info == nullptr || output_info == nullptr) {
        cerr << "[FATAL] Input or output tensor is null!" << endl;
        return -1;
    }

    cout << "[DEBUG] Input tensor type  : " << input_info->type << endl;
    cout << "[DEBUG] Output tensor type : " << output_info->type << endl;

    if (input_info->type != kTfLiteFloat32) {
        cerr << "[FATAL] Input tensor is not Float32!" << endl;
        return -1;
    }

    if (output_info->type != kTfLiteFloat32) {
        cerr << "[FATAL] Output tensor is not Float32!" << endl;
        return -1;
    }

    // ============================================================
    // 2. Open CSV
    // Your de10_test_data.csv has no header:
    // 60 input columns + 4 target columns = 64 columns
    // ============================================================
    ifstream test_file(csv_path.c_str());

    if (!test_file.is_open()) {
        cerr << "[FATAL] Could not open " << csv_path << endl;
        return -1;
    }

    // Metrics accumulators
    double sum_squared_error[OUTPUTS] = {0, 0, 0, 0};
    double sum_absolute_error[OUTPUTS] = {0, 0, 0, 0};
    double sum_true[OUTPUTS] = {0, 0, 0, 0};
    double sum_true_sq[OUTPUTS] = {0, 0, 0, 0};

    double total_inference_time_ms = 0.0;
    int total_inferences = 0;
    int skipped_rows = 0;

    string line;

    cout << "[SYSTEM] Running massive batch inference on ARM CPU..." << endl;

    // ============================================================
    // 3. Run inference for every CSV row
    // ============================================================
    while (getline(test_file, line)) {
        if (line.empty()) {
            continue;
        }

        vector<float> values;

        if (!parse_csv_line(line, values)) {
            skipped_rows++;
            continue;
        }

        if ((int)values.size() != TOTAL_COLUMNS) {
            cerr << "[WARNING] Skipped row. Columns = "
                 << values.size() << ", expected = " << TOTAL_COLUMNS << endl;
            skipped_rows++;
            continue;
        }

        // IMPORTANT:
        // Get input pointer inside the loop before writing input.
        float* input_tensor = interpreter->typed_input_tensor<float>(0);

        if (input_tensor == nullptr) {
            cerr << "[ERROR] Failed to get input tensor pointer." << endl;
            skipped_rows++;
            continue;
        }

        // Fill 60 input values
        for (int i = 0; i < INPUT_SIZE; i++) {
            input_tensor[i] = values[i];
        }

        // Read 4 target values
        float y_test_scaled[OUTPUTS];

        for (int i = 0; i < OUTPUTS; i++) {
            y_test_scaled[i] = values[INPUT_SIZE + i];
        }

        // Debug first sample BEFORE Invoke
        if (total_inferences == 0) {
            cout << "========== DE10 FIRST SAMPLE DEBUG BEFORE INVOKE ==========" << endl;

            cout << "Input first 8 values:" << endl;
            for (int k = 0; k < 8; k++) {
                cout << fixed << setprecision(6) << input_tensor[k] << " ";
            }
            cout << endl;

            cout << "Target scaled:" << endl;
            for (int k = 0; k < OUTPUTS; k++) {
                cout << fixed << setprecision(6) << y_test_scaled[k] << " ";
            }
            cout << endl;
        }

        // Measure only Invoke latency
        auto start_time = chrono::high_resolution_clock::now();

        if (interpreter->Invoke() != kTfLiteOk) {
            cerr << "[ERROR] Invoke failed at sample "
                 << total_inferences << endl;
            skipped_rows++;
            continue;
        }

        auto end_time = chrono::high_resolution_clock::now();

        double elapsed_ms =
            chrono::duration<double, milli>(end_time - start_time).count();

        total_inference_time_ms += elapsed_ms;

        // Get output AFTER Invoke
        TfLiteTensor* out_tensor =
            interpreter->tensor(interpreter->outputs()[0]);

        if (out_tensor == nullptr ||
            out_tensor->type != kTfLiteFloat32 ||
            out_tensor->data.f == nullptr) {
            cerr << "[ERROR] Invalid output tensor after Invoke." << endl;
            skipped_rows++;
            continue;
        }

        float* output_ptr = out_tensor->data.f;

        // IMPORTANT:
        // Copy output immediately.
        float y_pred_scaled[OUTPUTS];

        for (int i = 0; i < OUTPUTS; i++) {
            y_pred_scaled[i] = output_ptr[i];
        }

        // Debug first sample prediction
        if (total_inferences == 0) {
            cout << "Prediction scaled:" << endl;
            for (int k = 0; k < OUTPUTS; k++) {
                cout << fixed << setprecision(8) << y_pred_scaled[k] << " ";
            }
            cout << endl;
            cout << "===========================================================" << endl;
        }

        // Inverse transform and update metrics
        for (int i = 0; i < OUTPUTS; i++) {
            double y_true_real = inverse_scale(y_test_scaled[i], i);
            double y_pred_real = inverse_scale(y_pred_scaled[i], i);

            double diff = y_true_real - y_pred_real;

            sum_squared_error[i] += diff * diff;
            sum_absolute_error[i] += fabs(diff);
            sum_true[i] += y_true_real;
            sum_true_sq[i] += y_true_real * y_true_real;
        }

        total_inferences++;
    }

    test_file.close();

    if (total_inferences == 0) {
        cerr << "[FATAL] No valid samples found!" << endl;
        return -1;
    }

    // ============================================================
    // 4. Accuracy metrics
    // ============================================================
    cout << "\n======================================================================" << endl;
    cout << left << setw(20) << "SENSOR"
         << " | " << setw(10) << "RMSE"
         << " | " << setw(10) << "MAE"
         << " | " << "R2 SCORE" << endl;
    cout << "----------------------------------------------------------------------" << endl;

    double total_r2 = 0.0;

    for (int i = 0; i < OUTPUTS; i++) {
        double mse = sum_squared_error[i] / total_inferences;
        double rmse = sqrt(mse);
        double mae = sum_absolute_error[i] / total_inferences;

        double mean_true = sum_true[i] / total_inferences;
        double ss_tot =
            sum_true_sq[i] - total_inferences * mean_true * mean_true;

        double r2 = 0.0;

        if (fabs(ss_tot) > 1e-12) {
            r2 = 1.0 - (sum_squared_error[i] / ss_tot);
        }

        total_r2 += r2;

        cout << left << setw(20) << SENSOR_NAMES[i]
             << " | " << fixed << setprecision(4) << setw(10) << rmse
             << " | " << fixed << setprecision(4) << setw(10) << mae
             << " | " << fixed << setprecision(4) << r2 << endl;
    }

    double system_avg_r2 = total_r2 / OUTPUTS;

    cout << "----------------------------------------------------------------------" << endl;
    cout << left << setw(20) << "SYSTEM AVERAGE"
         << " | " << setw(10) << ""
         << " | " << setw(10) << ""
         << " | " << fixed << setprecision(4) << system_avg_r2 << endl;
    cout << "======================================================================" << endl;

    // ============================================================
    // 5. Edge performance
    // ============================================================
    double avg_latency = total_inference_time_ms / total_inferences;
    double throughput = 1000.0 / avg_latency;

    cout << "\n[SECTION 3: EDGE PERFORMANCE (ARM Cortex-A9)]" << endl;
    cout << "Total Predictions Made    : " << total_inferences << endl;
    cout << "Skipped Rows              : " << skipped_rows << endl;
    cout << "Average Inference Latency : "
         << fixed << setprecision(2) << avg_latency << " ms" << endl;
    cout << "Inference Throughput      : "
         << fixed << setprecision(1) << throughput
         << " predictions/sec" << endl;

    cout << "[SYSTEM] Evaluation finished successfully." << endl;

    cout.flush();
    cerr.flush();

    // Avoid destructor cleanup crash on DE10
    _exit(0);
}