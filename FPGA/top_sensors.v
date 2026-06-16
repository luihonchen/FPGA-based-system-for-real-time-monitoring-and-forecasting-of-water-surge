module top_sensors ( 
    input  wire 			CLOCK_50,    
    input  wire 			[0:0]  KEY,          // Added KEY[0] as Reset
	 output wire 			[7:0] LED,  
	 
	 //ultrasonic
    input  wire        GPIO_0_AH14, 
    output wire        GPIO_0_AH13, 
	 
	 //temp
	 inout  wire        GPIO_0_D11, // DS18B20 Data Line (PIN_D11)
	 
	 //Turbidity
	 // ADC 物理引脚 (硬核连接至 DE10-Nano 的 J14 插座)
    output wire        ADC_CONVST,  // PIN_U9
    output wire        ADC_SCK,     // PIN_W10
    output wire        ADC_SDI,     // PIN_V10
    input  wire        ADC_SDO,     // PIN_AC4
	 
	 //Flow rate
	 input  wire        GPIO_0_AE24,   // 水流脉冲输入 (接引脚 PIN_V12)
	 
	 
	 //Avolan MM
	 input  wire [2:0]  avs_address,   // 3-bit address (Reg 0 to Reg 4)
    input  wire        avs_read,      // Read enable from HPS
    output reg  [31:0] avs_readdata  // Data sent back to HPS
       
);

//************************ULTRASONIC******************************//
    wire [15:0] distance;

	wire        distance_valid;
    // Instantiate modified Ultrasonic Module
    ultrasonic_dri ultra_inst
	 (
        .clk(CLOCK_50),
        .rst_n(KEY[0]),        // Pass reset signal
        .echo(GPIO_0_AH14),
        .trig(GPIO_0_AH13),
        .distance(distance),
		  .distance_valid(distance_valid)
    );
	 
//********************************TEMPERATURE ********************************************************//
    // Internal wires connecting to the 1-Wire controller
    wire [15:0] current_temperature;
    wire        temp_valid;

    // Instantiate the 1-Wire Driver
    temp_dri temp_inst (
        .clk_50(CLOCK_50),
        .reset_n(KEY[0]),          // Active-low reset button
        .dq(GPIO_0_D11),           // Physical 1-Wire connection
        .temp_data(current_temperature),
        .data_valid(temp_valid)
    );
	 
//********************************TURBIDITY ********************************************************// 

    // 内部连线与寄存器
    wire [11:0] raw_adc_data;
    wire        adc_ready;
    reg         adc_start;
    reg [15:0]  current_ntu;
    reg [21:0]  trigger_timer;      // 用于产生 10Hz 采样的计数器

    // =========================================================
    // 1. 定时产生自动采样触发脉冲 (约每秒采样 10 次)
    // =========================================================
    always @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0]) begin
            trigger_timer <= 22'd0;
            adc_start     <= 1'b0;
        end else begin
            if (trigger_timer < 22'd4_999_999) begin // 5M * 20ns = 0.1s
                trigger_timer <= trigger_timer + 1'b1;
                adc_start     <= 1'b0;
            end else begin
                trigger_timer <= 22'd0;
                adc_start     <= 1'b1;  // 发出单个周期的启动高脉冲
            end
        end
    end
// =========================================================
// Turbidity NTU calibration for DE10-Nano onboard ADC
// ADC range: 0V to 4.096V = 0 to 4095
//
// Mapping:
// Voltage > 3.63V        -> 0 NTU
// Clear water 3.63V      -> 5 NTU
// Stirred soil 2.20V     -> 500 NTU
// Very muddy 0.30V       -> 1000 NTU
// Voltage < 0.30V        -> clamp at 1000 NTU
// =========================================================

localparam [11:0] CLEAR_ADC   = 12'd3630;  // 3.63V
localparam [11:0] DIRTY_ADC   = 12'd2200;  // 2.20V
localparam [11:0] EXTREME_ADC = 12'd300;   // 0.30V

localparam [15:0] CLEAR_NTU   = 16'd5;
localparam [15:0] DIRTY_NTU   = 16'd500;
localparam [15:0] MAX_NTU     = 16'd1000;

reg [31:0] ntu_calc;

always @(posedge CLOCK_50 or negedge KEY[0]) begin
    if (!KEY[0]) begin
        current_ntu <= 16'd0;
        ntu_calc    <= 32'd0;
    end else if (adc_ready) begin

        // Higher than clear-water voltage = clearer than reference
        if (raw_adc_data > CLEAR_ADC) begin
            current_ntu <= 16'd0;
        end

        // Between 3.63V and 2.20V
        // 3.63V -> 5 NTU
        // 2.20V -> 500 NTU
        else if (raw_adc_data >= DIRTY_ADC) begin
            ntu_calc = 32'd5 +
                       ((32'd3630 - {20'd0, raw_adc_data}) * (32'd500 - 32'd5)) /
                       (32'd3630 - 32'd2200);

            current_ntu <= ntu_calc[15:0];
        end

        // Between 2.20V and 0.30V
        // 2.20V -> 500 NTU
        // 0.30V -> 1000 NTU
        else if (raw_adc_data > EXTREME_ADC) begin
            ntu_calc = 32'd500 +
                       ((32'd2200 - {20'd0, raw_adc_data}) * (32'd1000 - 32'd500)) /
                       (32'd2200 - 32'd300);

            current_ntu <= ntu_calc[15:0];
        end

        // Lower than 0.30V = extremely muddy / sensor saturation
        else begin
            current_ntu <= MAX_NTU;
        end
    end
end
    // =========================================================
    // 2. 浊度 NTU 计算与 LED 灯条显示逻辑 (修复竞争冒险隐患)
    // =========================================================
//    always @(posedge CLOCK_50 or negedge KEY[0]) begin
//        if (!KEY[0]) begin
//            current_ntu <= 16'd0;
//            //LED         <= 8'b0000_0000;
//        end else if (adc_ready) begin
//            // 浊度分段校准逻辑
//            if (raw_adc_data >= 12'd4000) 
//                current_ntu <= 16'd0;       // 清澈水质
//            else if (raw_adc_data <= 12'd2500) 
//                current_ntu <= 16'd3000;    // 极度浑浊
//            else 
//                current_ntu <= (12'd4000 - raw_adc_data) * 2; // 线性近似

//            // LED 条形图（Bar Graph）累加显示逻辑
//            // 使用条件拼接或级联判断，确保高亮灯时低位灯保持点亮
//            LED[0] <= (current_ntu > 16'd10);
//            LED[1] <= (current_ntu > 16'd500);
//            LED[2] <= (current_ntu > 16'd1000);
//            LED[3] <= (current_ntu > 16'd1500);
//            LED[4] <= (current_ntu > 16'd2000);
//            LED[5] <= (current_ntu > 16'd2500);
//            LED[6] <= (current_ntu > 16'd2800);
//            LED[7] <= (current_ntu >= 16'd3000);
 //       end
  //  end

    // =========================================================
    // 3. 例化之前编写的固定通道 0 的 LTC2308 驱动核
    // =========================================================
    adc_tur_dri tur_inst (
        .clk         (CLOCK_50),
        .rst_n       (KEY[0]),
        .adc_ch      (3'd0),            // 默认探测通道 0
        .start       (adc_start),       // 定时触发信号
        .adc_data    (raw_adc_data),    // 12位原始读数
        .data_valid  (adc_ready),       // 转换结束标志
        
        // 绑定物理硬件引脚
        .adc_convst  (ADC_CONVST),
        .adc_sck     (ADC_SCK),
        .adc_sdi     (ADC_SDI),
        .adc_sdo     (ADC_SDO)
    );

//******************************FLOW RATE****************************************//
    wire [15:0] flow_val; // 内部流速值 (L/min * 10)
    wire [31:0] total_p;
	 wire        flow_updated;

    // 例化核心处理模块
    flow_rate_dri flow_inst (
        .clk            (CLOCK_50),
        .rst_n          (KEY[0]),
        .pulse_in       (GPIO_0_AE24),
        .flow_rate_lmin (flow_val),
        .total_pulses   (total_p),
		  .flow_updated   (flow_updated)
    );

    // LED 条形图逻辑 (量程 1-30 L/min) 
//    always @(posedge CLOCK_50) begin
//        LED[0] <= (flow_val >= 16'd10);  // > 1.0 L/min
//        LED[1] <= (flow_val >= 16'd40);  // > 4.0 L/min
//        LED[2] <= (flow_val >= 16'd80);  // > 8.0 L/min
//        LED[3] <= (flow_val >= 16'd120); // > 12.0 L/min
//        LED[4] <= (flow_val >= 16'd160); // > 16.0 L/min
//        LED[5] <= (flow_val >= 16'd200); // > 20.0 L/min
//        LED[6] <= (flow_val >= 16'd250); // > 25.0 L/min
//        LED[7] <= (flow_val >= 16'd300); // > 30.0 L/min (满量程)
//    end

//*****************************FPGA ACQUISITION LATENCY*************************//
// Measure full FPGA acquisition latency
// Start: adc_start
// Stop : distance_valid + temp_valid + adc_ready + flow_updated are all received
//
// CLOCK_50 = 50 MHz
// 1 cycle = 20 ns
// latency_us = acquisition_latency_count / 50
// latency_ms = acquisition_latency_count / 50000

reg [31:0] acquisition_latency_counter;
reg [31:0] acquisition_latency_count;
reg        acquisition_measuring;

reg distance_done;
reg temperature_done;
reg turbidity_done;
reg flow_done;

wire all_sensor_done;

assign all_sensor_done = distance_done &&
                         
                         turbidity_done;
                        //  && flow_done&&temperature_done 

always @(posedge CLOCK_50 or negedge KEY[0]) begin
    if (!KEY[0]) begin
        acquisition_latency_counter <= 32'd0;
        acquisition_latency_count   <= 32'd0;
        acquisition_measuring       <= 1'b0;

        distance_done    <= 1'b0;
        temperature_done <= 1'b0;
        turbidity_done   <= 1'b0;
        flow_done        <= 1'b0;
    end else begin

        // Start latency measurement when ADC sampling starts
        if (adc_start && !acquisition_measuring) begin
            acquisition_latency_counter <= 32'd0;
            acquisition_measuring       <= 1'b1;

            distance_done    <= 1'b0;
            temperature_done <= 1'b0;
            turbidity_done   <= 1'b0;
            flow_done        <= 1'b0;
        end

        // Count while waiting for all sensor update signals
        else if (acquisition_measuring && !all_sensor_done) begin
            acquisition_latency_counter <= acquisition_latency_counter + 1'b1;

            if (distance_valid) begin
                distance_done <= 1'b1;
            end

            if (temp_valid) begin
                temperature_done <= 1'b1;
            end

            if (adc_ready) begin
                turbidity_done <= 1'b1;
            end

            if (flow_updated) begin
                flow_done <= 1'b1;
            end
        end

        // Stop counter when all sensor values are updated
        else if (acquisition_measuring && all_sensor_done) begin
            acquisition_latency_count <= acquisition_latency_counter;
            acquisition_measuring     <= 1'b0;
        end
    end
end

//*****************************LED LOGIC*****************************************//


    // 8-bit register to hold the state of all 8 LEDs
    reg [7:0] led_status;

    always @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0]) begin
            led_status <= 8'b0000_0000;
        end else if (temp_valid) begin
            
            // Default: turn off all LEDs when new data arrives
            led_status <= 8'b0000_0000;
            
				//Ultrasonic
            if (distance <= 16'd60) //>=60cm
				begin
                     led_status[0] <= 1'b1;
            end 
				
				else if (distance >= 16'd60) //>=60cm
				begin
                led_status[1] <= 1'b1;
            end 
				
				// Temperature Range Logic (Threshold = Temp * 16)
				if (current_temperature <= 16'd576) //<=36oC
				begin
                led_status[2] <= 1'b1;
            end 
				
				else if (current_temperature >= 16'd576) //>=36oC
				begin
                led_status[3] <= 1'b1;
            end 
				
				//Turbidity 1500=1500ntu
				if (current_ntu <= 16'd1500) //0ntu clean
				begin
               led_status[4] <= 1'b1;
            end 
				
				else if (current_ntu >= 16'd1500) 
				begin
               led_status[5] <= 1'b1;
            end 
				
				//Flow rate 120=12.0L/min
				if (flow_val <= 16'd120) // < 12.0 L/min
				begin
               led_status[6] <= 1'b1;
            end 
				else if (flow_val >= 16'd120)// > 12.0 L/min
				begin
                led_status[7] <= 1'b1;
            end
        end
    end

    // Map the internal register to the physical LED output pins
    assign LED = led_status;

//******************************** AVALON-MM READ LOGIC (NEW)***********************************************//

    always @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0]) begin
            avs_readdata <= 32'd0;
        end else if (avs_read) begin
				case (avs_address)
    3'd0: avs_readdata <= {16'd0, distance};                    // Reg 0: Ultrasonic
    3'd1: avs_readdata <= {16'd0, current_temperature};         // Reg 1: Temperature
    3'd2: avs_readdata <= {16'd0, current_ntu};                 // Reg 2: Turbidity NTU
    3'd3: avs_readdata <= {16'd0, flow_val};                    // Reg 3: Flow Rate
    3'd4: avs_readdata <= total_p;                              // Reg 4: Flow Pulses
    3'd5: avs_readdata <= {20'd0, raw_adc_data};                // Reg 5: Raw ADC
    3'd6: avs_readdata <= acquisition_latency_count;            // Reg 6: FPGA acquisition latency
    default: avs_readdata <= 32'hDEADBEEF;
endcase
			end
	 end
	 
endmodule