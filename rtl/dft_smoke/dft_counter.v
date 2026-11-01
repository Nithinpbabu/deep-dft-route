// ==============================================================================
// Design: dft_counter
// Description: Minimal 16-bit sequential counter with synchronous load and enable.
//              Created as a synthesizable smoke test for OpenROAD native DFT validation.
//
// Flip-Flop Count: 16 sequential storage elements
// Interface:
//   - clk   : Primary system clock
//   - rst_n : Active-low asynchronous reset
//   - enable: Synchronous counter increment enable
//   - load  : Synchronous parallel data load enable
//   - d     : 16-bit parallel data load bus
//   - q     : 16-bit registered counter output
//
// License: MIT
// Copyright (c) 2026 DFT-Aware Physical Design Hotspot Prediction Project
// ==============================================================================

`timescale 1ns / 1ps

module dft_counter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire        load,
    input  wire [15:0] d,
    output reg  [15:0] q
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q <= 16'h0000;
        end else if (load) begin
            q <= d;
        end else if (enable) begin
            q <= q + 16'h0001;
        end
    end

endmodule
