#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <exception>
#include <memory>
#include <chrono>   // Added for time tracking
#include <ctime>

//database
#include <algorithm>
#include <cstdlib>
#include <iomanip>

//memory mapping
#include <fcntl.h>
#include <sys/mman.h>

// DE10-Nano Lightweight HPS-to-FPGA bridge physical memory address
#define HW_REGS_BASE (0xFF200000)
#define HW_REGS_SPAN (0x00200000)
#define HW_REGS_MASK (HW_REGS_SPAN - 1)

// Qsys Base Addresses
#define SENSOR_BASE_OFFSET (0x00000000) // Base for the 4 sensors
#define BUZZER_PIO_OFFSET  (0x00000020)
#define BUTTON_PIO_OFFSET  (0x00005000) 

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace std;

// Scaling constants
//const float MIN_VALS[4] = {3.73f, 8.27f, 17.56f, 24.65f};
//const float MAX_VALS[4] = {16.05f, 21.81f, 33.06f, 183.88f};
//const float THRESHOLDS[4] = {0.0f, 4.0f, 0.0f, 0.0f};

// Scaling constants from latest min_max_scaler.pkl
// Feature order: Flow_L_min, Distance_cm, Temp_C, Turbidity_NTU
const float MIN_VALS[4] = {3.00f, 133.55f, 16.60f, 5.00f};
const float MAX_VALS[4] = {28.43f, 200.00f, 38.06f, 597.23f};

// Water surge thresholds
const float FLOW_SURGE_THRESHOLD = 20.0f;      // Flow > 20 L/min
const float DIST_SURGE_THRESHOLD = 160.0f;     // Distance < 160 cm
const float TEMP_RAIN_THRESHOLD  = 28.0f;      // Temperature drops during rain
const float NTU_SURGE_THRESHOLD  = 300.0f;     // Turbidity > 300 NTU

//database
string make_firebase_key(string time_str) {
    replace(time_str.begin(), time_str.end(), ' ', '_');
    replace(time_str.begin(), time_str.end(), ':', '-');
    replace(time_str.begin(), time_str.end(), '/', '-');
    return time_str;
}

bool upload_prediction_to_firebase(
    const string& time_str,
    const vector<float>& raw,
    const vector<float>& pred_real,
    const string& sensor_status,
    const string& prediction_status
) {
    string key_str = make_firebase_key(time_str);

    ostringstream json;
    json << fixed << setprecision(4);

    json << "{";
    json << "\"Timestamp\":\"" << time_str << "\",";
    json << "\"Raw_FlowRate_Lmin\":" << raw[0] << ",";
    json << "\"Raw_Distance_cm\":" << raw[1] << ",";
    json << "\"Raw_Temperature_C\":" << raw[2] << ",";
    json << "\"Raw_Turbidity_NTU\":" << raw[3] << ",";
    json << "\"Pred_FlowRate_Lmin\":" << pred_real[0] << ",";
    json << "\"Pred_Distance_cm\":" << pred_real[1] << ",";
    json << "\"Pred_Temperature_C\":" << pred_real[2] << ",";
    json << "\"Pred_Turbidity_NTU\":" << pred_real[3] << ",";
    json << "\"Sensor_Status\":\"" << sensor_status << "\",";
    json << "\"Prediction_Status\":\"" << prediction_status << "\"";
    json << "}";

    string firebase_url =
        "https://fyp-project-781fc-default-rtdb.asia-southeast1.firebasedatabase.app/"
        "prediction_log/" + key_str + ".json";

    string curl_command =
        "curl -s -S -m 10 -X PUT "
        "-H \"Content-Type: application/json\" "
        "-d '" + json.str() + "' "
        "\"" + firebase_url + "\" "
        "> /dev/null 2>&1";

    int result = system(curl_command.c_str());

    return result == 0;
}

//SHOW WINDOW

void print_lstm_window(const deque<vector<float>>& window) {
    cout << "================ LSTM 15-STEP WINDOW ================" << endl;
    cout << fixed << setprecision(4);

    cout << left
         << setw(8)  << "Step"
         << setw(12) << "Flow"
         << setw(12) << "Distance"
         << setw(12) << "Temp"
         << setw(12) << "NTU"
         << endl;

    cout << "------------------------------------------------------" << endl;

    for (int i = 0; i < window.size(); i++) {
        cout << left
             << setw(8)  << i
             << setw(12) << window[i][0]
             << setw(12) << window[i][1]
             << setw(12) << window[i][2]
             << setw(12) << window[i][3]
             << endl;
    }

    cout << "======================================================" << endl;
}


int main() {
    cout << "[SYSTEM] Loading TFLite model..." << endl;

    unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("../LSTM/best_water_surge_lstm_float16_builtin.tflite");

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

    if (interpreter->ResizeInputTensor(interpreter->inputs()[0], {1, 15, 4}) != kTfLiteOk) {
        cerr << "[FATAL] Failed to resize input tensor!" << endl;
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "[FATAL] Failed to allocate tensors!" << endl;
        return -1;
    }

    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    if (input_tensor == nullptr) {
        cerr << "[FATAL] Failed to get input tensor pointer." << endl;
        return -1;
    }

    deque<vector<float>> window(15, vector<float>(4, 0.0f));

    cout << "[SYSTEM] Initializing Master FPGA Memory Map..." << endl;
    int fd;
    void *virtual_base;
    
    if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) {
        cerr << "[FATAL] could not open /dev/mem. Did you run as root?" << endl;
        return -1;
    }
    
    virtual_base = mmap( NULL, HW_REGS_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HW_REGS_BASE );
    
    if( virtual_base == MAP_FAILED ) {
        cerr << "[FATAL] mmap() failed..." << endl;
        close( fd );
        return -1;
    }
    
    // --- MAP SENSORS ---
    void *sensor_addr = (uint8_t *)virtual_base + (SENSOR_BASE_OFFSET & HW_REGS_MASK);
    volatile uint32_t *distance_ptr    = (uint32_t *)((uint8_t *)sensor_addr + 0x00);
    volatile uint32_t *temperature_ptr = (uint32_t *)((uint8_t *)sensor_addr + 0x04);
    volatile uint32_t *ntu_ptr         = (uint32_t *)((uint8_t *)sensor_addr + 0x08);
    volatile uint32_t *flow_ptr        = (uint32_t *)((uint8_t *)sensor_addr + 0x0C);

    // --- MAP BUZZER & BUTTON ---
    volatile uint32_t *buzzer_addr = (uint32_t *)((uint8_t *)virtual_base + ((BUZZER_PIO_OFFSET) & (HW_REGS_MASK)));
    volatile uint32_t *button_addr = (uint32_t *)((uint8_t *)virtual_base + ((BUTTON_PIO_OFFSET) & (HW_REGS_MASK))); 

    *buzzer_addr = 0; // Ensure buzzer is OFF at startup

    cout << "[SYSTEM] Pipeline initialized. System Armed." << endl;

    bool is_muted = false; 

    while (true) {
        bool is_alarm_active = false; 

                // Start a high-precision stopwatch at the exact start of the loop
        auto loop_start_time = std::chrono::steady_clock::now();

        // =========================================================
        // 1. GET CURRENT TIME & READ SENSORS DIRECTLY FROM FPGA
        // =========================================================
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        char time_buffer[80];
        std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
        string time_str(time_buffer);

// =========================================================
//End-to-end prediction latency =
//sensor reading + scaling + LSTM window update + TFLite inference
//+ inverse scaling + threshold checking + terminal output
//+ buzzer control + CSV logging
//Excluded = Firebase upload + 60-second waiting delay
// =========================================================
//start- end to end latency
auto e2e_start_time = std::chrono::steady_clock::now();


        vector<float> raw(4, 0.0f);
        raw[0] = (float)(*flow_ptr & 0xFFFF) / 10.0f;       // Flow
        raw[1] = (float)(*distance_ptr & 0xFFFF);           // Distance
        raw[2] = (float)(*temperature_ptr & 0xFFFF) / 16.0f;// Temperature
        raw[3] = (float)(*ntu_ptr & 0xFFFF);                // NTU

        cout << "------------------------------------------------" << endl;
        cout << "[DATA] Time Collected : " << time_str << endl;
        cout << "[DATA] Raw Readings   : Flow: " << raw[0]
             << " | Dist: " << raw[1]
             << " | Temp: " << raw[2]
             << " | NTU: " << raw[3] << endl;
        cout << "------------------------------------------------" << endl;

        // =========================================================
        // 2. DATA SCALING & SLIDING WINDOW
        // =========================================================
        vector<float> scaled(4, 0.0f);
        for (int i = 0; i < 4; i++) {
            scaled[i] = (raw[i] - MIN_VALS[i]) / (MAX_VALS[i] - MIN_VALS[i]);
            if (scaled[i] > 1.0f) scaled[i] = 1.0f;
            if (scaled[i] < 0.0f) scaled[i] = 0.0f;
        }

        window.pop_front();
        window.push_back(scaled);

	//show window
	print_lstm_window(window);

        for (int t = 0; t < 15; t++) {
            for (int f = 0; f < 4; f++) {
                input_tensor[(t * 4) + f] = window[t][f];
            }
        }

        // =========================================================
        // 3. AI PREDICTION
        // =========================================================

//start& end- inference latency
cout << "[SYSTEM] Invoking AI Model..." << endl;

auto infer_start_time = std::chrono::steady_clock::now();

if (interpreter->Invoke() != kTfLiteOk) {
    cerr << "[ERROR] Model Invoke failed!" << endl;
    sleep(10);
    continue;
}

auto infer_end_time = std::chrono::steady_clock::now();

double inference_latency_ms =
    std::chrono::duration<double, std::milli>(infer_end_time - infer_start_time).count();



        TfLiteTensor* out_tensor_after =
            interpreter->tensor(interpreter->outputs()[0]);

        if (out_tensor_after == nullptr) {
            cerr << "[ERROR] Output tensor is null after Invoke." << endl;
            sleep(10);
            continue;
        }

        if (out_tensor_after->type != kTfLiteFloat32) {
            cerr << "[ERROR] Output tensor is not Float32 after Invoke." << endl;
            sleep(10);
            continue;
        }

        if (out_tensor_after->data.f == nullptr) {
            cerr << "[ERROR] Output tensor data pointer is null." << endl;
            sleep(10);
            continue;
        }

        float* output_tensor = out_tensor_after->data.f;
                vector<float> pred_real(4, 0.0f);
                for (int i = 0; i < 4; i++) {
                    pred_real[i] = output_tensor[i] * (MAX_VALS[i] - MIN_VALS[i]) + MIN_VALS[i];
        }

        cout << "[SUCCESS] PREDICTION | Flow: " << pred_real[0]
             << " | Dist: " << pred_real[1]
             << " | Temp: " << pred_real[2]
             << " | NTU: " << pred_real[3] << endl;

        // =========================================================
        // 4. ALARM LOGIC & FIREBASE UPLOAD
        // =========================================================
bool sensor_warning =
    raw[0] >= FLOW_SURGE_THRESHOLD &&
    raw[1] <= DIST_SURGE_THRESHOLD &&
    raw[3] >= NTU_SURGE_THRESHOLD;

// Optional supporting condition
bool sensor_rain_support =
    raw[2] <= TEMP_RAIN_THRESHOLD;

bool prediction_warning =
    pred_real[0] >= FLOW_SURGE_THRESHOLD &&
    pred_real[1] <= DIST_SURGE_THRESHOLD &&
    pred_real[3] >= NTU_SURGE_THRESHOLD;

// Optional supporting condition
bool prediction_rain_support =
    pred_real[2] <= TEMP_RAIN_THRESHOLD;


        string sensor_status = sensor_warning ? "Warning" : "Normal";
        string prediction_status = prediction_warning ? "Warning" : "Normal";

        cout << "================ STATUS CHECK ================" << endl;
        cout << "[SENSOR STATUS]     " << sensor_status << endl;
        cout << "[PREDICTION STATUS] " << prediction_status << endl;

cout << "[SENSOR CHECK] "
     << "Flow: " << raw[0] << " >= " << FLOW_SURGE_THRESHOLD
     << " | Dist: " << raw[1] << " <= " << DIST_SURGE_THRESHOLD
     << " | NTU: " << raw[3] << " >= " << NTU_SURGE_THRESHOLD
     << " | Temp support: " << raw[2] << " <= " << TEMP_RAIN_THRESHOLD
     << endl;

cout << "[PRED CHECK]   "
     << "Flow: " << pred_real[0] << " >= " << FLOW_SURGE_THRESHOLD
     << " | Dist: " << pred_real[1] << " <= " << DIST_SURGE_THRESHOLD
     << " | NTU: " << pred_real[3] << " >= " << NTU_SURGE_THRESHOLD
     << " | Temp support: " << pred_real[2] << " <= " << TEMP_RAIN_THRESHOLD
     << endl;


        is_alarm_active = sensor_warning || prediction_warning;

        if (is_alarm_active) {
            cout << "[ALARM STATUS] ACTIVE" << endl;
        } else {
            cout << "[ALARM STATUS] NORMAL" << endl;
        }

        cout << "==============================================" << endl;

        uint32_t keys_state = *button_addr;
        bool key0_pressed = ((keys_state & 0x01) == 0); 

        if (!is_alarm_active) {
            is_muted = false; 
            *buzzer_addr = 0; 
            cout << "[STATUS] Normal condition." << endl;
        } 
        else {
            if (sensor_warning) cout << "[ALARM] Raw sensor readings exceeded thresholds!" << endl;
            if (prediction_warning) cout << "[ALARM] Predicted readings exceeded thresholds!" << endl;

            if (key0_pressed && !is_muted) {
                is_muted = true;
                cout << "[SYSTEM] Operator pressed KEY[0]. Buzzer MUTED." << endl;
            }

            if (is_muted) {
                *buzzer_addr = 0; 
            } else {
                *buzzer_addr = 1; 
            }
        }

ofstream pred_log("predictions_5min.csv", ios::app);
if (pred_log.is_open()) {
    pred_log << time_str << ","
             << raw[0] << "," << raw[1] << "," << raw[2] << "," << raw[3] << ","
             << pred_real[0] << "," << pred_real[1] << "," << pred_real[2] << "," << pred_real[3] << ","
             << sensor_status << "," << prediction_status << "\n";
    pred_log.close();
}

// =========================================================
// END-TO-END PREDICTION LATENCY END
// Excludes Firebase upload and 60-second waiting delay
// =========================================================
auto e2e_end_time = std::chrono::steady_clock::now();

double end_to_end_latency_ms =
    std::chrono::duration<double, std::milli>(e2e_end_time - e2e_start_time).count();

cout << fixed << setprecision(4);
cout << "[LATENCY] HPS TFLite inference latency: "
     << inference_latency_ms << " ms" << endl;

cout << "[LATENCY] End-to-end prediction latency: "
     << end_to_end_latency_ms << " ms" << endl;

cout << "[LATENCY SCOPE] Included: FPGA register read, scaling, LSTM window update, "
     << "TFLite inference, inverse scaling, threshold checking, terminal output, "
     << "buzzer control, and CSV logging." << endl;

cout << "[LATENCY SCOPE] Excluded: Firebase upload and 60-second waiting delay." << endl;

// Firebase upload is after latency measurement, so it is excluded
bool uploaded = upload_prediction_to_firebase(time_str, raw, pred_real, sensor_status, prediction_status);
if (uploaded) {
    cout << "[SYSTEM] Data uploaded to Firebase." << endl;
} else {
    cerr << "[WARNING] Firebase upload failed." << endl;
}
        // =========================================================
        // 5. RESPONSIVE POLLING (COMPENSATED FOR EXECUTION TIME)
        // =========================================================
        
        // Stop the stopwatch and calculate how long the AI and Firebase took
        auto loop_end_time = std::chrono::steady_clock::now();
        int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - loop_start_time).count();

        // Subtract the execution time from the 60 seconds (60000 ms)
        int wait_time_ms = 60000 - elapsed_ms;
        
        // Failsafe: If the internet hangs and takes longer than 60 seconds, don't sleep at all
        if (wait_time_ms < 0) {
            wait_time_ms = 0;
        }

        int poll_interval_ms = 100;  

        for (int w = 0; w < wait_time_ms; w += poll_interval_ms) {
            if (is_alarm_active && !is_muted) {
                uint32_t current_keys = *button_addr;
                if ((current_keys & 0x01) == 0) {
                    is_muted = true;
                    *buzzer_addr = 0; 
                    cout << "\n[SYSTEM] Operator pressed KEY[0]. Buzzer MUTED." << endl;
                }
            }
            usleep(poll_interval_ms * 1000); 
        }

}
    // Clean up memory before exiting (Though the while loop is infinite)
    munmap(virtual_base, HW_REGS_SPAN);
    close(fd);
    return 0;
}