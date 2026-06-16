// Driver module: temp_dri.v
// Handles DS18B20 protocol with 50MHz input clock
//****************************************************************************************//

module temp_dri(
    input              clk_50,     // 50MHz Clock input
    input              reset_n,    // Active-low reset
    inout              dq,         // 1-Wire Bus
    output reg [15:0]  temp_data,  // 16-bit raw temperature
    output reg         data_valid  // Single-cycle pulse when data is ready
);

// Commands
localparam  ROM_SKIP_CMD = 8'hcc,
            CONVERT_CMD  = 8'h44,
            READ_TEMP    = 8'hbe;

// States
localparam  init=3'd1, rom_skip=3'd2, wr_byte=3'd3, 
            temp_convert=3'd4, delay=3'd5, rd_temp=3'd6, rd_byte=3'd7;

reg [4:0]  cnt;
reg        clk_1us;
reg [19:0] cnt_1us;
reg [2:0]  cur_state, next_state;
reg [3:0]  flow_cnt, wr_cnt, cmd_cnt;
reg [4:0]  rd_cnt, bit_width;
reg [7:0]  wr_data;
reg [15:0] rd_data;
reg        init_done, st_done, cnt_1us_en, dq_out;

assign dq = dq_out;

// 1MHz Clock Generation from 50MHz
always @ (posedge clk_50 or negedge reset_n) begin
    if (!reset_n) begin
        cnt <= 5'd0; clk_1us <= 1'b0;
    end else if(cnt < 5'd24) begin
        cnt <= cnt + 1'b1;
    end else begin
        cnt <= 5'd0; clk_1us <= ~clk_1us;
    end
end

// Microsecond Counter
always @ (posedge clk_1us or negedge reset_n) begin
    if (!reset_n) cnt_1us <= 20'b0;
    else if (cnt_1us_en) cnt_1us <= cnt_1us + 1'b1;
    else cnt_1us <= 20'b0;
end

// FSM Transitions
always @ (posedge clk_1us or negedge reset_n) begin
    if(!reset_n) cur_state <= init;
    else cur_state <= next_state;
end

// FSM Logic
always @(*) begin
    case(cur_state)
        init:         next_state = init_done ? rom_skip : init;
        rom_skip:     next_state = st_done ? wr_byte : rom_skip;
        wr_byte: begin
            if(st_done) begin
                case(cmd_cnt)
                    4'd1: next_state = temp_convert;
                    4'd2: next_state = delay;
                    4'd3: next_state = rd_temp;
                    4'd4: next_state = rd_byte;
                    default: next_state = temp_convert;
                endcase
            end else next_state = wr_byte;
        end
        temp_convert: next_state = st_done ? wr_byte : temp_convert;
        delay:        next_state = st_done ? init : delay;
        rd_temp:      next_state = st_done ? wr_byte : rd_temp;
        rd_byte:      next_state = st_done ? init : rd_byte;
        default:      next_state = init;
    endcase
end

// Protocol Timing Logic
always @ (posedge clk_1us or negedge reset_n) begin
    if(!reset_n) begin
        flow_cnt <= 4'd0; init_done <= 1'b0; cnt_1us_en <= 1'b1;
        dq_out <= 1'bZ; st_done <= 1'b0; rd_cnt <= 5'd0; 
        wr_cnt <= 4'd0; cmd_cnt <= 4'd0; data_valid <= 1'b0;
    end else begin
        st_done <= 1'b0;
        data_valid <= 1'b0; 
        case (next_state)
            init: begin
                init_done <= 1'b0;
                case(flow_cnt)
                    4'd1: begin
                        cnt_1us_en <= 1'b1;
                        if(cnt_1us < 20'd500) dq_out <= 1'b0;
                        else begin dq_out <= 1'bz; cnt_1us_en <= 1'b0; flow_cnt <= 4'd2; end
                    end
                    4'd2: begin
                        cnt_1us_en <= 1'b1;
                        if(cnt_1us < 20'd30) dq_out <= 1'bz;
                        else flow_cnt <= 4'd3;
                    end
                    4'd3: if(!dq) flow_cnt <= 4'd4;
                    4'd4: if(cnt_1us == 20'd500) begin
                        cnt_1us_en <= 1'b0; init_done <= 1'b1; flow_cnt <= 4'd0;
                    end
                    default: flow_cnt <= 4'd1;
                endcase
            end
            wr_byte: begin
                if(wr_cnt <= 4'd7) begin
                    case(flow_cnt)
                        4'd0: begin dq_out <= 1'b0; cnt_1us_en <= 1'b1; flow_cnt <= 4'd1; end
                        4'd1: flow_cnt <= 4'd2;
                        4'd2: begin
                            if(cnt_1us < 20'd60) dq_out <= wr_data[wr_cnt];
                            else if(cnt_1us < 20'd63) dq_out <= 1'bz;
                            else flow_cnt <= 4'd3;
                        end
                        4'd3: begin flow_cnt <= 4'd0; cnt_1us_en <= 1'b0; wr_cnt <= wr_cnt + 1'b1; end
                    endcase
                end else begin 
                    st_done <= 1'b1; 
                    wr_cnt <= 4'd0; 
                    cmd_cnt <= (cmd_cnt == 4'd4) ? 4'd1 : cmd_cnt + 1'b1; 
                end
            end
            rom_skip:     begin wr_data <= ROM_SKIP_CMD; st_done <= 1'b1; end
            temp_convert: begin wr_data <= CONVERT_CMD; st_done <= 1'b1; end
            delay: begin
                cnt_1us_en <= 1'b1;
                // Wait 750ms for 12-bit conversion
                if(cnt_1us == 20'd750000) begin st_done <= 1'b1; cnt_1us_en <= 1'b0; end
            end
            rd_temp: begin wr_data <= READ_TEMP; bit_width <= 5'd16; st_done <= 1'b1; end
            rd_byte: begin
                if(rd_cnt < bit_width) begin
                    case(flow_cnt)
                        4'd0: begin cnt_1us_en <= 1'b1; dq_out <= 1'b0; flow_cnt <= 4'd1; end
                        4'd1: begin
                            dq_out <= 1'bz;
                            if(cnt_1us == 20'd14) begin rd_data <= {dq, rd_data[15:1]}; flow_cnt <= 4'd2; end
                        end
                        4'd2: if(cnt_1us > 20'd64) begin 
                            flow_cnt <= 4'd0; rd_cnt <= rd_cnt + 1'b1; cnt_1us_en <= 1'b0; 
                        end
                    endcase
                end else begin
                    temp_data <= rd_data;
                    data_valid <= 1'b1; 
                    st_done <= 1'b1;
                    rd_cnt <= 5'd0;
                end
            end
        endcase
    end
end

endmodule