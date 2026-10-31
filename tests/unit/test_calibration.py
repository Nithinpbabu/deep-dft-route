"""
Unit tests for post-hoc calibration module in dft_hotspot.ml.calibration.
"""

from __future__ import annotations

import unittest
import numpy as np

from dft_hotspot.ml.calibration import (
    GlobalAffineCalibrator,
    GlobalScaleCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    fit_calibrators,
)
from dft_hotspot.evaluation.metrics import mae, rmse


class TestCalibration(unittest.TestCase):
    """Test suite for post-hoc prediction calibrators."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_identity_calibrator(self) -> None:
        """Verifies identity passthrough."""
        cal = IdentityCalibrator()
        preds = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        cal.fit(preds, preds)
        out = cal.predict(preds)
        np.testing.assert_allclose(out, preds)
        self.assertEqual(cal.get_params()["method"], "identity")

    def test_affine_calibrator_synthetic(self) -> None:
        """Verifies affine least-squares recovery of slope and intercept."""
        # True relation: target = 0.5 * pred - 0.05
        preds = np.linspace(0.1, 0.9, 50, dtype=np.float32)
        targets = 0.5 * preds - 0.05

        cal = GlobalAffineCalibrator().fit(preds, targets)
        params = cal.get_params()
        self.assertAlmostEqual(params["slope_a"], 0.5, places=4)
        self.assertAlmostEqual(params["intercept_b"], -0.05, places=4)

        out = cal.predict(preds)
        np.testing.assert_allclose(out, targets, atol=1e-5)
        # Check clipping bounds
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

    def test_scale_calibrator_synthetic(self) -> None:
        """Verifies global scale through the origin."""
        preds = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        targets = 0.5 * preds

        cal = GlobalScaleCalibrator().fit(preds, targets)
        self.assertAlmostEqual(cal.get_params()["scale_s"], 0.5, places=4)

        out = cal.predict(preds)
        np.testing.assert_allclose(out, targets, atol=1e-5)

    def test_calibration_improves_biased_mae(self) -> None:
        """Verifies that affine calibration dramatically improves MAE on shifted predictions."""
        # Simulates the TinyUNet shift: predictions are shifted upward by +0.10
        targets = np.random.uniform(0.0, 0.2, 100).astype(np.float32)
        preds = targets + 0.10  # Systematic additive bias

        mae_raw = mae(preds, targets)
        self.assertAlmostEqual(mae_raw, 0.10, places=3)

        cal = GlobalAffineCalibrator().fit(preds, targets)
        cal_preds = cal.predict(preds)

        mae_cal = mae(cal_preds, targets)
        self.assertLess(mae_cal, 0.01)  # Error dramatically reduced!

    def test_constant_prediction_edge_case(self) -> None:
        """Verifies safe handling of constant predictions without zero division."""
        preds = np.full(50, 0.5, dtype=np.float32)
        targets = np.linspace(0.0, 1.0, 50, dtype=np.float32)

        # Affine
        cal_aff = GlobalAffineCalibrator().fit(preds, targets)
        out_aff = cal_aff.predict(preds)
        self.assertAlmostEqual(float(out_aff[0]), float(np.mean(targets)), places=4)

        # Scale
        cal_scale = GlobalScaleCalibrator().fit(preds, targets)
        out_scale = cal_scale.predict(preds)
        self.assertTrue(np.all(np.isfinite(out_scale)))

        # Isotonic
        cal_iso = IsotonicCalibrator().fit(preds, targets)
        out_iso = cal_iso.predict(preds)
        self.assertAlmostEqual(float(out_iso[0]), float(np.mean(targets)), places=4)

    def test_fit_calibrators_suite(self) -> None:
        """Verifies fit_calibrators returns all expected calibrators."""
        preds = np.random.uniform(0.1, 0.8, (2, 16, 16)).astype(np.float32)
        targets = np.random.uniform(0.0, 0.5, (2, 16, 16)).astype(np.float32)

        cal_dict = fit_calibrators(preds, targets)
        self.assertIn("identity", cal_dict)
        self.assertIn("global_affine", cal_dict)
        self.assertIn("global_scale", cal_dict)
        self.assertIn("isotonic", cal_dict)


if __name__ == "__main__":
    unittest.main()
