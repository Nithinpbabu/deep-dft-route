"""
Unit tests for configuration manager, YAML parsing, and PDK validation.
"""

import unittest
import tempfile
from pathlib import Path
from dft_hotspot.config import ConfigManager

class TestConfigManager(unittest.TestCase):

    def test_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            ConfigManager.load_config("non_existent_config.yaml")

    def test_valid_config_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test.yaml"
            config_file.write_text("grid:\n  height: 64\n  width: 64\n", encoding="utf-8")

            cfg = ConfigManager.load_config(config_file)
            self.assertEqual(cfg["grid"]["height"], 64)
            self.assertEqual(cfg["grid"]["width"], 64)

    def test_validate_pdk_missing_files(self):
        pdk_cfg = {
            "name": "nangate45",
            "liberty_file": "non_existent/tech.lib",
            "tech_lef": "non_existent/tech.lef",
            "stdcell_lef": "non_existent/std.lef"
        }
        valid, missing = ConfigManager.validate_pdk_files(pdk_cfg)
        self.assertFalse(valid)
        self.assertEqual(len(missing), 3)

    def test_validate_pdk_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lib = tmp_path / "test.lib"
            t_lef = tmp_path / "test_tech.lef"
            m_lef = tmp_path / "test_macro.lef"
            lib.touch()
            t_lef.touch()
            m_lef.touch()

            pdk_cfg = {
                "name": "test_pdk",
                "liberty_file": str(lib),
                "tech_lef": str(t_lef),
                "stdcell_lef": str(m_lef)
            }
            valid, missing = ConfigManager.validate_pdk_files(pdk_cfg, workspace_root=tmp_path)
            self.assertTrue(valid)
            self.assertEqual(len(missing), 0)

if __name__ == "__main__":
    unittest.main()
