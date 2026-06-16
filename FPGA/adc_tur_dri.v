module adc_tur_dri (
    // FPGA 系统接口
    input  wire        clk,        // 系统时钟 (建议 50MHz)
    input  wire        rst_n,      // 异步复位
    input  wire [2:0]  adc_ch,     // 读取通道 (此处已被锁定为CH0)
    input  wire        start,      // 启动单次转换脉冲
    output reg [11:0]  adc_data,   // 转换后的12位数据
    output reg         data_valid, // 数据有效标志位

    // ADC 器件硬件接口
    output reg         adc_convst, // CONVST
    output reg         adc_sck,    // SCK
    output reg         adc_sdi,    // SDI
    input  wire        adc_sdo     // SDO
);

    // 状态机定义
    localparam S_IDLE    = 3'd0;
    localparam S_CONV    = 3'd1;
    localparam S_WAIT    = 3'd2;
    localparam S_SPI     = 3'd3;
    localparam S_DONE    = 3'd4;

    reg [2:0] state;
    reg [5:0] sck_cnt;      // 记录移位个数
    reg [7:0] wait_cnt;     // 等待转换完成的计数器
    reg [11:0] shift_reg;   // 数据移位寄存器
    
    // =======================================================
    // 固定使用 Channel 0 的配置字
    // S/D=1, O/S=0, S1=0, S0=0, UNI=1, SLP=0 -> 6'b100010
    // =======================================================
    wire [5:0] config_word = 6'b100010;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            adc_convst <= 1'b0;
            adc_sck    <= 1'b0;
            adc_sdi    <= 1'b0;
            data_valid <= 1'b0;
            wait_cnt   <= 8'd0;
            sck_cnt    <= 6'd0;
            shift_reg  <= 12'd0;
            adc_data   <= 12'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    data_valid <= 1'b0;
                    if (start) begin
                        adc_convst <= 1'b1; // 拉高启动转换
                        state      <= S_CONV;
                    end
                end

                S_CONV: begin
                    adc_convst <= 1'b0;     // 完成上升沿触发，拉低
                    wait_cnt   <= 8'd0;
                    state      <= S_WAIT;
                end

                S_WAIT: begin
                    // 等待内部 SAR 转换完成 (t_CONV max = 1.6us)
                    // 50MHz 时钟下，100个周期为 2.0us，安全裕量充足
                    if (wait_cnt < 8'd100) 
                        wait_cnt <= wait_cnt + 1'b1;
                    else begin
                        sck_cnt <= 6'd0;
                        adc_sck <= 1'b0;
                        state   <= S_SPI;
                    end
                end

                S_SPI: begin
                    // 产生移位时钟 SCK (通过状态机翻转，实际 SCK 频率为 clk/2 = 25MHz)
                    if (sck_cnt < 6'd12) begin
                        adc_sck <= ~adc_sck; 
                        
                        if (adc_sck == 1'b1) begin // 即将变为下降沿
                            // 发送 6-bit 配置字 (前6个时钟周期)
                            if (sck_cnt < 6'd6)
                                adc_sdi <= config_word[5 - sck_cnt];
                            else
                                adc_sdi <= 1'b0;
                        end else begin // 即将变为上升沿
                            // 并在上升沿锁存串行输入的 SDO 数据
                            shift_reg <= {shift_reg[10:0], adc_sdo};
                            sck_cnt   <= sck_cnt + 1'b1;
                        end
                    end else begin
                        adc_sck <= 1'b0;
                        adc_sdi <= 1'b0;
                        state   <= S_DONE;
                    end
                end

                S_DONE: begin
                    adc_data   <= shift_reg;
                    data_valid <= 1'b1; // 输出有效脉冲
                    state      <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule