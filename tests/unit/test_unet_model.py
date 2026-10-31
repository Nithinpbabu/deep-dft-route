"""
Unit tests for TinyUNet model and training routines in dft_hotspot.ml.unet and train_unet.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from dft_hotspot.ml.train_unet import evaluate_model, set_seed, train_epoch, train_unet
from dft_hotspot.ml.unet import TinyUNet


class DictDataset(torch.utils.data.Dataset):
    """Simple dictionary dataset for test mocking."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, ids: list[str]) -> None:
        self.x = x
        self.y = y
        self.ids = ids

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {
            "input": self.x[idx],
            "target": self.y[idx],
            "sample_id": self.ids[idx],
        }


@unittest.skipUnless(_HAS_TORCH, "PyTorch required for TinyUNet tests")
class TestTinyUNet(unittest.TestCase):
    """Test suite for TinyUNet architecture and training execution."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_model_forward_shapes(self) -> None:
        """Verifies (B, 12, 64, 64) -> (B, 1, 64, 64) tensor output."""
        model = TinyUNet(in_channels=12, out_channels=1, base_channels=16, final_activation="sigmoid")
        self.assertGreater(model.count_parameters(), 100_000)

        x = torch.randn(2, 12, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2, 1, 64, 64]))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_activations(self) -> None:
        """Verifies relu, sigmoid, and none activation options."""
        m_relu = TinyUNet(in_channels=12, out_channels=1, final_activation="relu")
        m_none = TinyUNet(in_channels=12, out_channels=1, final_activation="none")

        x = torch.randn(1, 12, 16, 16)
        out_relu = m_relu(x)
        self.assertTrue(torch.all(out_relu >= 0.0))

        out_none = m_none(x)
        self.assertEqual(out_none.shape, torch.Size([1, 1, 16, 16]))

    def test_training_smoke_loop(self) -> None:
        """Verifies train_unet completes epochs and produces valid checkpoints and metrics."""
        set_seed(42)
        device = torch.device("cpu")

        # Mock 4 samples with 16x16 grids
        x_train = torch.rand(4, 12, 16, 16)
        y_train = torch.rand(4, 1, 16, 16)
        train_ds = DictDataset(x_train, y_train, ["t0", "t1", "t2", "t3"])
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

        x_val = torch.rand(2, 12, 16, 16)
        y_val = torch.rand(2, 1, 16, 16)
        val_ds = DictDataset(x_val, y_val, ["v0", "v1"])
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = TinyUNet(in_channels=12, out_channels=1, base_channels=8, final_activation="sigmoid")

        summary = train_unet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=self.output_dir,
            device=device,
            learning_rate=1e-2,
            max_epochs=3,
            early_stopping_patience=2,
            seed=42,
        )

        self.assertIn("best_epoch", summary)
        self.assertTrue((self.output_dir / "best_model.pt").exists())
        self.assertTrue((self.output_dir / "training_history.csv").exists())

        # Test evaluation routine
        criterion = nn.MSELoss()
        val_loss, metrics, preds = evaluate_model(model, val_loader, criterion, device)
        self.assertGreater(val_loss, 0.0)
        self.assertIn("mae", metrics)
        self.assertIn("v0", preds)
        self.assertEqual(preds["v0"].shape, (16, 16))

    def test_multichannel_forward_and_eval(self) -> None:
        """Verifies 8-channel forward pass and evaluation with empty channel handling."""
        model = TinyUNet(in_channels=12, out_channels=8, base_channels=16, final_activation="none")
        x = torch.randn(2, 12, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([2, 8, 64, 64]))

        # Test evaluation with 8 target channels, including one all-zero (empty) channel
        ch_names = [f"ch_{i}" for i in range(8)]
        y_val = torch.rand(2, 8, 16, 16)
        y_val[:, 4, :, :] = 0.0  # Make ch_4 empty (simulating metal1)
        x_val = torch.randn(2, 12, 16, 16)

        val_ds = DictDataset(x_val, y_val, ["v0", "v1"])
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        device = torch.device("cpu")
        criterion = nn.MSELoss()
        val_loss, metrics, preds = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            clamp_output=True,
            channel_names=ch_names,
        )

        self.assertIn("mae", metrics)
        self.assertIn("empty_channels", metrics)
        self.assertIn("ch_4", metrics["empty_channels"])
        self.assertIn("per_channel", metrics)
        self.assertIn("ch_4", metrics["per_channel"])
        self.assertEqual(preds["v0"].shape, (8, 16, 16))
        self.assertTrue(np.all(preds["v0"] >= 0.0) and np.all(preds["v0"] <= 1.0))

    def test_multichannel_training_smoke_loop(self) -> None:
        """Verifies 8-channel training smoke loop."""
        set_seed(42)
        device = torch.device("cpu")

        x_train = torch.rand(4, 12, 16, 16)
        y_train = torch.rand(4, 8, 16, 16)
        train_ds = DictDataset(x_train, y_train, ["t0", "t1", "t2", "t3"])
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

        x_val = torch.rand(2, 12, 16, 16)
        y_val = torch.rand(2, 8, 16, 16)
        val_ds = DictDataset(x_val, y_val, ["v0", "v1"])
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = TinyUNet(in_channels=12, out_channels=8, base_channels=8, final_activation="none")

        summary = train_unet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=self.output_dir,
            device=device,
            learning_rate=1e-2,
            max_epochs=2,
            early_stopping_patience=2,
            seed=42,
        )

        self.assertIn("best_epoch", summary)
        self.assertTrue((self.output_dir / "best_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
