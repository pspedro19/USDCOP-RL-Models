"""
Configuration Loader - DEPRECATED
==================================

DEPRECATED: This module is deprecated. Use src/shared/config_loader.py instead:
    from shared.config_loader import ConfigLoader, get_config

This file is maintained for backward compatibility only.
The canonical ConfigLoader is in src/shared/config_loader.py.
"""

import warnings

# Re-export from canonical location
from shared.config_loader import ConfigLoader, get_config, load_feature_config

# Show deprecation warning on import
warnings.warn(
    "src.config.loader is deprecated. Use shared.config_loader instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["ConfigLoader", "get_config", "load_feature_config"]
