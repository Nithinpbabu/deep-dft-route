# Stage 5c: TinyUNet Prediction Distribution Diagnostics & Calibration Summary

## 1. Distribution Diagnostics (Raw Predictions vs Targets)

| Split | Target Mean (Std) | Pred Mean (Std) | Target Median / P95 | Pred Median / P95 | Mean Additive Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 0.0212 (0.0954) | 0.1264 (0.0738) | 0.0000 / 0.1499 | 0.1024 / 0.2380 | **+0.1051** |
| **Validation** | 0.0200 (0.0948) | 0.1255 (0.0743) | 0.0000 / 0.1099 | 0.1018 / 0.2206 | **+0.1055** |
| **Test** | 0.0170 (0.0875) | 0.1218 (0.0596) | 0.0000 / 0.0136 | 0.1021 / 0.1705 | **+0.1048** |

### Key Diagnostic Insight
- **Systematic Additive Shift**: Raw TinyUNet predictions suffer from a consistent **+0.105** positive offset across all three partitions.
- **Sparsity Squeeze**: The ground truth routing target is highly sparse (median $= 0.0000$, 95th percentile $\approx 0.01-0.15$), but the raw network's minimum prediction is $\approx 0.065-0.081$.
- **Source of High MAE**: The Stage 5b MAE ($0.1111$) was almost purely driven by this constant $+0.105$ pedestal shift rather than random spatial errors!

---

## 2. Post-Hoc Calibration Methods (Fitted on Validation Split)

- **Global Affine**:
  - $y = a \cdot x + b$
  - Slope $a = 1.0982$
  - Intercept $b = -0.1178$ (subtracts the $\approx 0.1$ offset!)
- **Global Scale**:
  - $y = s \cdot x$
  - Scale factor $s = 0.4031$
- **Isotonic Regression**: Monotonic piecewise-constant transformation.

---

## 3. Calibrated Performance on Holdout Test Split

| Calibration Method | Test MAE $\downarrow$ | Test RMSE $\downarrow$ | Pearson $r \uparrow$ | Top-5% Recall $\uparrow$ | IoU @ 5% Hotspot $\uparrow$ | Dice @ 5% Hotspot $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw TinyUNet (Identity)** | 0.1111 | 0.1143 | 0.8762 | 89.50% | 0.8100 | 0.8950 |
| **Global Scale ($s=0.40$)** | 0.0529 | 0.0747 | 0.8762 | 89.50% | 0.8100 | 0.8950 |
| **Global Affine ($a=1.10, b=-0.12$)** | **0.0158** | **0.0429** | **0.8884** | **89.50%** | **0.8100** | **0.8950** |
| **Isotonic Regression** | 0.0075 | 0.0382 | 0.9010 | 89.59% | 0.8113 | 0.8959 |

---

## 4. Key Conclusion
Applying a simple validation-fitted **Global Affine calibrator** completely eliminates the $+0.105$ bias, slashing the MAE from **0.1111 down to 0.0158** while **strictly preserving** the outstanding **0.8100 IoU@5%** and **89.50% Recall@5%**!
