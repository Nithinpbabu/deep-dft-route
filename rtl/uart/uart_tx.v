// =============================================================================
// Module Name  : uart_tx
// Description  : Lightweight Synthesizable UART Transmitter (Smoke Test Design)
// Project      : DFT-Aware Physical Design Hotspot Prediction
// License      : Apache License 2.0 / MIT
// Provenance   : Intentionally small smoke test design for Stage 1 flow verification.
//                For final dataset, externally sourced open RTL with recorded provenance is preferred.
// =============================================================================

`default_nettype none

module uart_tx #(
    parameter integer CLK_FREQ = 50000000,
    parameter integer BAUDRATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       tx_start,
    input  wire [7:0] tx_data,
    output reg        tx_out,
    output reg        tx_busy,
    output reg        tx_done
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUDRATE;

    localparam [2:0] STATE_IDLE  = 3'b000;
    localparam [2:0] STATE_START = 3'b001;
    localparam [2:0] STATE_DATA  = 3'b010;
    localparam [2:0] STATE_STOP  = 3'b011;
    localparam [2:0] STATE_DONE  = 3'b100;

    reg [2:0]  state;
    reg [15:0] clk_count;
    reg [2:0]  bit_index;
    reg [7:0]  data_buf;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= STATE_IDLE;
            tx_out    <= 1'b1;
            tx_busy   <= 1'b0;
            tx_done   <= 1'b0;
            clk_count <= 16'd0;
            bit_index <= 3'd0;
            data_buf  <= 8'd0;
        end else begin
            tx_done <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    tx_out    <= 1'b1;
                    tx_busy   <= 1'b0;
                    clk_count <= 16'd0;
                    bit_index <= 3'd0;
                    if (tx_start) begin
                        tx_busy  <= 1'b1;
                        data_buf <= tx_data;
                        state    <= STATE_START;
                    end
                end

                STATE_START: begin
                    tx_out <= 1'b0;
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 16'd0;
                        state     <= STATE_DATA;
                    end
                end

                STATE_DATA: begin
                    tx_out <= data_buf[bit_index];
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 16'd0;
                        if (bit_index < 7) begin
                            bit_index <= bit_index + 1'b1;
                        end else begin
                            bit_index <= 3'd0;
                            state     <= STATE_STOP;
                        end
                    end
                end

                STATE_STOP: begin
                    tx_out <= 1'b1;
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1'b1;
                    end else begin
                        clk_count <= 16'd0;
                        state     <= STATE_DONE;
                    end
                end

                STATE_DONE: begin
                    tx_done <= 1'b1;
                    tx_busy <= 1'b0;
                    state   <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
