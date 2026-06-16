module ultrasonic_dri (
    input  wire        clk,
    input  wire        rst_n,      // Added active-low reset
    input  wire        echo,
    output wire        trig,
    output reg [15:0]  distance = 0,
	 
	 
	 output reg         distance_valid     // 1-clock pulse when distance updates
);

    reg [31:0] cnt_us = 0;
    reg        _trig = 0;

    // Counters
    reg [9:0]  cnt_one_us = 0;
    reg [9:0]  cnt_ten_us = 0;
    reg [21:0] cnt_eighty_ms = 0;

    // Synchronizer for Echo Pin
    reg echo_reg1, echo_reg2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            echo_reg1 <= 1'b0;
            echo_reg2 <= 1'b0;
        end else begin
            echo_reg1 <= echo;
            echo_reg2 <= echo_reg1;
        end
    end

    wire one_us    = (cnt_one_us == 0);
    wire ten_us    = (cnt_ten_us == 0);
    wire eighty_ms = (cnt_eighty_ms == 0);

    assign trig = _trig;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_one_us    <= 10'd50;
            cnt_ten_us    <= 10'd500;
            cnt_eighty_ms <= 22'd4000000;
            cnt_us        <= 32'd0;
            _trig         <= 1'b0;
            distance      <= 16'd0;
				distance_valid <= 1'b0; //add
        end else begin
		  
		  distance_valid <= 1'b0;
            // decrement counter values
            cnt_one_us    <= (one_us    ? 50      : cnt_one_us)    - 1;
            cnt_ten_us    <= (ten_us    ? 500     : cnt_ten_us)    - 1;
            cnt_eighty_ms <= (eighty_ms ? 4000000 : cnt_eighty_ms) - 1;

            // Trigger pulse logic
            if (eighty_ms) begin
                _trig <= 1'b1;
            end
            
            if (ten_us && _trig) begin
                _trig <= 1'b0;
            end

            // Distance measurement logic
            if (one_us) begin
                if (echo_reg2) begin
                    cnt_us <= cnt_us + 1;
                end
                else if (cnt_us > 0) begin
                    distance <= cnt_us / 58;
                    cnt_us <= 0;
						  distance_valid <= 1'b1;   // distance value updated
                end
            end
        end
    end

endmodule