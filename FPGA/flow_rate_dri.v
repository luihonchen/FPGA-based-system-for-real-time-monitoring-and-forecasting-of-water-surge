module flow_rate_dri (
    input  wire        clk,           // 50MHz 系统时钟
    input  wire        rst_n,         // 异步复位 (低电平有效)
    input  wire        pulse_in,      // 传感器信号输入 (黄线，需上拉至3.3V)
    output reg [15:0]  flow_rate_lmin,// 实时流速 (L/min * 10, 例如 15 表示 1.5 L/min)
    output reg [31:0]  total_pulses,   // 累计脉冲总数，用于计算总流量
	 
	 
	  output reg         flow_updated      // 1-clock pulse when flow_rate_lmin updates
);

    // 1. 信号同步与上升沿检测
    reg [2:0] pulse_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pulse_reg <= 3'b0;
        else pulse_reg <= {pulse_reg[1:0], pulse_in};
    end
    wire rising_edge = (pulse_reg[1] && !pulse_reg[2]);

    // 2. 1秒定时闸门 (50,000,000 个周期)
    reg [25:0] timer;
    wire tick_1s = (timer == 26'd49_999_999);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) timer <= 26'd0;
        else timer <= tick_1s ? 26'd0 : timer + 1'b1;
    end

    // 3. 频率统计与流速转换
    reg [15:0] count_per_sec;
    
//    always @(posedge clk or negedge rst_n) begin
//        if (!rst_n) begin
//            count_per_sec  <= 16'd0;
//            flow_rate_lmin <= 16'd0;
//            total_pulses   <= 32'd0;
//        end else begin
//            if (rising_edge) begin
//                count_per_sec <= count_per_sec + 1'b1;
//                total_pulses  <= total_pulses + 1'b1; // 累计脉冲 
//            end
//
//            if (tick_1s) begin
//                // 根据方程 F = 10Q 
//                // Q (L/min) = F (Hz) / 10
//                // 为了保留1位小数 (Q*10)，直接取 F 的值即可
//                // 例如：F = 50Hz -> Q = 5.0 L/min -> 输出 50
//                flow_rate_lmin <= count_per_sec; 
//                count_per_sec  <= 16'd0;
//            end
//        end
//    end
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        count_per_sec  <= 16'd0;
        flow_rate_lmin <= 16'd0;
        total_pulses   <= 32'd0;
        flow_updated   <= 1'b0;
    end else begin
        flow_updated <= 1'b0;   // default low, only pulse high for 1 clock

        if (rising_edge) begin
            count_per_sec <= count_per_sec + 1'b1;
            total_pulses  <= total_pulses + 1'b1;
        end

        if (tick_1s) begin
            flow_rate_lmin <= count_per_sec;
            count_per_sec  <= 16'd0;
            flow_updated   <= 1'b1;  // flow value updated
        end
    end
end
endmodule

