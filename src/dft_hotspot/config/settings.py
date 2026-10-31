import os
from pathlib import Path
import yaml
from typing import Dict, Any, List, Tuple

class ConfigManager:
    """Manages YAML configuration loading and validation for physical design flows."""

    @staticmethod
    def load_config(config_path: str | Path) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    @staticmethod
    def validate_pdk_files(pdk_cfg: Dict[str, Any], workspace_root: Path | None = None) -> Tuple[bool, List[str]]:
        """
        Validates whether required PDK files specified in configuration exist on disk.
        Returns (is_valid, list_of_missing_files).
        """
        if workspace_root is None:
            workspace_root = Path.cwd()

        required_keys = ["liberty_file", "tech_lef", "stdcell_lef"]
        missing = []

        for key in required_keys:
            val = pdk_cfg.get(key)
            if not val:
                missing.append(f"Key '{key}' missing from pdk config")
                continue

            # Support environment variable expansion if present
            expanded_val = os.path.expandvars(str(val))
            pdk_path = Path(expanded_val)
            if not pdk_path.is_absolute():
                pdk_path = workspace_root / pdk_path

            if not pdk_path.exists():
                missing.append(f"{key}: '{val}' (resolved: '{pdk_path}')")

        return (len(missing) == 0, missing)

    @staticmethod
    def validate_flow_config(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates required structure of a flow configuration.
        """
        errors = []
        if "design" not in cfg:
            errors.append("Missing required top-level section 'design'")
        else:
            design = cfg["design"]
            if "top_module" not in design:
                errors.append("Missing 'design.top_module'")
            if "rtl_files" not in design or not design["rtl_files"]:
                errors.append("Missing or empty 'design.rtl_files'")

        if "pdk" not in cfg:
            errors.append("Missing required top-level section 'pdk'")

        if "flow" not in cfg:
            errors.append("Missing required top-level section 'flow'")

        return (len(errors) == 0, errors)
