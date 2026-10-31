"""
Unit tests to verify package importability and logger initialization.
"""

import unittest

class TestPackageImports(unittest.TestCase):

    def test_package_import(self):
        import dft_hotspot
        self.assertEqual(dft_hotspot.__version__, "0.1.0")

    def test_logger_setup(self):
        from dft_hotspot.utils import setup_logger
        logger = setup_logger("test_logger")
        self.assertEqual(logger.name, "test_logger")

    def test_config_manager_import(self):
        from dft_hotspot.config import ConfigManager
        self.assertIsNotNone(ConfigManager)

if __name__ == "__main__":
    unittest.main()
