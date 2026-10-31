# Stage 4d Classical ML Baselines: Limitations & Generalization Disclaimers

## Overview
- **Evaluation Purpose**: Classical machine learning pipeline validation on grid-cell tabular formulation.
- **Formulation**: Each $64 \times 64$ grid cell is an independent training row ($4,096$ rows per sample).
- **Target Channel**: `total_wirelength_density` (normalized post-route routing label).
- **Evaluation Partition**: `VAL`

## Methodological Boundaries & Limitations

1. **Tiny Dataset & Duplicate Layouts**:
   - The training set contains only 4 samples ($16,384$ rows) drawn from only **2 unique physical layouts** (`u45` and `u55`).
   - The validation sample (`u65_clk3p0`) and test sample (`u65_clk5p0`) are **identical physical layouts** with $0.0$ difference across all input and target channels.
   - Consequently, test set metrics replicate validation metrics with zero variance.

2. **Smoke-Only Pipeline Validation**:
   - These models and metrics are evaluated solely to verify feature matrix construction, model fitting, metric logging, and coefficient extraction.
   - They MUST NOT be cited or interpreted as scientific evidence of machine learning generalization on unseen physical designs.

3. **Spatial Independence Assumption in Tabular Regressors**:
   - Standard tabular models (Linear, Ridge, Lasso, Trees) treat each grid cell independently, discarding spatial 2D neighborhood context (receptive fields) except for coordinate channels (`physical/normalized_x`, `physical/normalized_y`).
   - Fully Convolutional Networks (FCNs / U-Nets) in Stage 5 will capture 2D spatial receptive fields natively.

## Comparison to Stage 4c Heuristic Reference Floor
- **Stage 4c Heuristic Reference (`simple_weighted_sum`)**:
  - Validation MAE: `0.0155`
  - Validation Recall@5%: `66.83%`
- **Stage 4d Best Classical ML Model**:
  - Best by MAE: `random_forest_regressor` (MAE = 0.0113)
  - Best by Recall@5%: `random_forest_regressor` (Recall@5% = 89.76%)
