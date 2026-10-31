# Multi-Target TinyUNet Routing Label Model Limitations

## 1. Empty and Sparse Channels
- **`metal1_wirelength_density` is all-zero ($0.0000$)**:
  In the Nangate45 PDK with OpenROAD standard cell placement and routing, signal routing is strictly carried on Metal 2 (horizontal) and Metal 3 (vertical) and higher. Metal 1 is reserved inside standard cells for local pin connections and power rails (`VDD`/`VSS`). Therefore, inter-cell routing labels produce zero wirelength on Metal 1.
  - Pearson and Spearman rank correlations are undefined (NaN) on `metal1_wirelength_density` due to zero target variance.
  - The model correctly predicts near-zero outputs across this channel.
  - In macro-averaged metrics, this channel is reported and isolated.

## 2. Multi-Target vs. Single-Target Tradeoff on Primary Target (`total_wirelength_density`)
- Predicting all 8 channels simultaneously forces the network's convolutional feature backbone to share representation capacity across horizontal wirelength, vertical wirelength, via count, and layer-specific distributions.
- See `metrics_per_channel.csv` and Stage 5d summary report for quantitative comparison against single-target Stage 5c baseline.

## 3. Dataset Scope
- Evaluated on `expanded_v002` (36 Nangate45 runs across `uart_tx`, `dft_counter`, and `spi`). Generalization to larger macros and different technology nodes (e.g. Sky130hd) requires additional floorplan evaluations.
