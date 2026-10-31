# Stage 4c Limitations & Generalization Disclaimers

## Overview
- **Dataset ID**: `uart_nangate45_smoke_v001`
- **Design / Platform**: `uart_tx` / `nangate45`
- **Target Channel**: `total_wirelength_density`
- **Evaluation Purpose**: Verification of metric computation and heuristic baseline pipeline readiness.

## Methodological Boundaries & Limitations

1. **Non-Learning Heuristics Only**:
   - None of the predictors evaluated in Stage 4c are trained or learned models.
   - All weights (e.g. in `simple_weighted_sum`) are arbitrarily specified heuristic coefficients based on EDA domain intuitions, NOT optimization against loss functions.

2. **Single Design & Tiny Sample Count**:
   - The dataset contains only 6 runs from a single small benchmark design (`uart_tx`).
   - Across the 6 runs, there are only **3 unique physical layouts** because timing slack invariance caused identical placement/routing solutions for clock periods 3.0 ns and 5.0 ns.

3. **Split Identity Warning**:
   - Under the `smoke_config_holdout` split policy, the validation sample (`uart_tx_nangate45_u65_clk3p0`) and test sample (`uart_tx_nangate45_u65_clk5p0`) are **identical physical layouts** with zero difference in either features or routing labels.
   - Any metric performance reported on this test split is strictly code-path smoke verification and must NEVER be cited as empirical evidence of generalization.

4. **Per-Sample Normalization Constraint**:
   - Each heuristic predictor is normalized to $[0.0, 1.0]$ using per-sample maximum scaling: $\hat{y} = x / \max(x)$.
   - While this matches the per-sample normalization of the ground-truth routing labels extracted from DEF in Stage 2b, it assumes knowledge of the peak density within each layout tile.

5. **Hotspot Sparsity & Tie Resolution**:
   - Routing congestion and cell placements in `uart_tx` occupy less than $10\%$ of spatial bins. Top-k thresholds at $5\%$ and $10\%$ operate on sparse regions where small pixel shifts can cause sharp drops in IoU and Dice scores.

## Baseline Summary (Evaluation Partition: `val`)
- **Best Baseline by MAE**: `cell_density` (MAE = 0.0155)
- **Worst Baseline by MAE**: `sequential_cell_density` (MAE = 0.0212)
- **Best Baseline by Top-5% Recall**: `simple_weighted_sum` (Recall@5% = 66.83%)
