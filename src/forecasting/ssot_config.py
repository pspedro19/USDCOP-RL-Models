"""
Forecasting SSOT Config Loader
==============================

Typed access to config/forecasting_ssot.yaml.
Cached singleton — loaded once per process.

Usage:
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    cfg = ForecastingSSOTConfig.load()
    print(cfg.get_feature_columns())  # tuple of 21 feature names
    print(cfg.get_model_ids())        # tuple of 9 model IDs

Contract: CTR-FORECAST-SSOT-001
Version: 1.0.0
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

# Default config path (relative to project root)
_DEFAULT_CONFIG = "config/forecasting_ssot.yaml"


def _find_project_root() -> Path:
    """Walk up from this file to find project root (contains pyproject.toml)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").is_file():
            return p
        p = p.parent
    # Fallback: Airflow Docker layout
    airflow_root = Path("/opt/airflow")
    if airflow_root.exists():
        return airflow_root
    return Path(__file__).resolve().parent.parent.parent


class ForecastingSSOTConfig:
    """Loads and provides typed access to forecasting_ssot.yaml."""

    _instance: Optional["ForecastingSSOTConfig"] = None
    _instance_path: Optional[str] = None

    def __init__(self, raw: dict, config_path: str):
        self._raw = raw
        self._config_path = config_path
        self._validate()

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ForecastingSSOTConfig":
        """Load config, cached after first call per path."""
        if path is None:
            resolved = _find_project_root() / _DEFAULT_CONFIG
        else:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = _find_project_root() / resolved

        resolved_str = str(resolved)

        # Return cached if same path
        if cls._instance is not None and cls._instance_path == resolved_str:
            return cls._instance

        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        instance = cls(raw, resolved_str)
        cls._instance = instance
        cls._instance_path = resolved_str
        return instance

    @classmethod
    def reset(cls):
        """Clear cached instance (for testing)."""
        cls._instance = None
        cls._instance_path = None

    def _validate(self):
        """Basic validation on load."""
        features = self._raw.get("features", {})
        cols = features.get("columns", [])
        expected_count = features.get("count", 21)
        if len(cols) != expected_count:
            raise ValueError(
                f"Feature count mismatch: declared {expected_count}, "
                f"found {len(cols)} in columns list"
            )

        models = self._raw.get("models", {})
        if not models:
            raise ValueError("No models defined in config")

        horizons = self._raw.get("horizons", {}).get("values", [])
        if not horizons:
            raise ValueError("No horizons defined in config")

    # ---------------------------------------------------------------
    # Feature accessors
    # ---------------------------------------------------------------

    def get_feature_columns(self) -> Tuple[str, ...]:
        """Return the 21 SSOT feature column names."""
        return tuple(self._raw["features"]["columns"])

    def get_feature_count(self) -> int:
        return self._raw["features"]["count"]

    # ---------------------------------------------------------------
    # Model accessors
    # ---------------------------------------------------------------

    def get_model_ids(self) -> Tuple[str, ...]:
        """Return all model IDs (e.g., ridge, bayesian_ridge, ...)."""
        return tuple(self._raw["models"].keys())

    def get_model_def(self, model_id: str) -> Dict[str, Any]:
        """Return full model definition dict."""
        return dict(self._raw["models"][model_id])

    def get_model_type(self, model_id: str) -> str:
        """Return model type: linear, boosting, or hybrid."""
        return self._raw["models"][model_id]["type"]

    def get_model_params(self, model_id: str) -> Dict[str, Any]:
        """Return model-specific params dict."""
        return dict(self._raw["models"][model_id].get("params", {}))

    def get_param_translation(self, model_id: str) -> Dict[str, str]:
        """Return param name translation map (e.g., CatBoost)."""
        return dict(self._raw["models"][model_id].get("param_translation", {}))

    # ---------------------------------------------------------------
    # Horizon accessors
    # ---------------------------------------------------------------

    def get_horizons(self) -> Tuple[int, ...]:
        """Return all horizon values."""
        return tuple(self._raw["horizons"]["values"])

    def get_horizon_category(self, horizon: int) -> str:
        """Return category for a horizon: short, medium, or long."""
        cats = self._raw["horizons"]["categories"]
        for cat_name, cat_horizons in cats.items():
            if horizon in cat_horizons:
                return cat_name
        return "medium"

    def get_horizon_config(self, horizon: int) -> Dict[str, Any]:
        """Return horizon-specific hyperparameters."""
        cat = self.get_horizon_category(horizon)
        return dict(self._raw["horizons"]["configs"][cat])

    # ---------------------------------------------------------------
    # Data source accessors
    # ---------------------------------------------------------------

    def get_data_source(self, name: str) -> Dict[str, Any]:
        """Return data source config (ohlcv, macro, monitoring)."""
        return dict(self._raw["data_sources"][name])

    def get_macro_column_mapping(self) -> Dict[str, str]:
        """Return macro DB/parquet column -> feature name mapping."""
        return dict(self._raw["data_sources"]["macro"]["column_mapping"])

    def get_macro_lag_days(self) -> int:
        return self._raw["data_sources"]["macro"].get("lag_days", 1)

    # ---------------------------------------------------------------
    # Walk-forward accessors
    # ---------------------------------------------------------------

    def get_wf_config(self) -> Dict[str, Any]:
        """Return walk-forward validation config."""
        return dict(self._raw["walk_forward"])

    # ---------------------------------------------------------------
    # Track accessors
    # ---------------------------------------------------------------

    def get_track_config(self, track: str) -> Dict[str, Any]:
        """Return track config (h1 or h5)."""
        return dict(self._raw["tracks"][track])

    def get_track_model_ids(self, track: str) -> Tuple[str, ...]:
        """Return model IDs for a specific track."""
        track_cfg = self._raw["tracks"][track]
        models = track_cfg.get("models", "all")
        if models == "all":
            return self.get_model_ids()
        return tuple(models)

    # ---------------------------------------------------------------
    # Path accessors
    # ---------------------------------------------------------------

    def get_path(self, name: str) -> str:
        """Return a named path from the paths section."""
        return self._raw["paths"][name]

    # ---------------------------------------------------------------
    # Training accessors
    # ---------------------------------------------------------------

    def get_training_start_date(self) -> str:
        return self._raw["training"]["start_date"]

    def get_warmup_bars(self) -> int:
        return self._raw["training"].get("warmup_bars", 50)

    # ---------------------------------------------------------------
    # Raw access
    # ---------------------------------------------------------------

    @property
    def raw(self) -> dict:
        """Raw YAML dict for advanced access."""
        return self._raw

    @property
    def config_path(self) -> str:
        return self._config_path
