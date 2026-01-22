"""
Feature Flags Manager
=====================

Runtime feature flag management with hot reload support.
Allows enabling/disabling features without deployment.

Features:
- JSON config file-based flags
- Hot reload with configurable interval (default 60s)
- Rollout percentage support for gradual rollouts
- Thread-safe singleton pattern
- Default values for missing flags

Usage:
    from services.shared.feature_flags import get_feature_flags

    flags = get_feature_flags()

    if flags.is_enabled("shadow_mode"):
        # Execute shadow mode logic
        pass

    # With default value
    if flags.is_enabled("new_feature", default=False):
        # Feature code
        pass
"""

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default config path (relative to project root)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "feature_flags.json"


@dataclass
class FeatureFlag:
    """
    Represents a single feature flag.

    Attributes:
        name: Unique identifier for the flag
        enabled: Whether the flag is currently enabled
        description: Human-readable description of the flag
        created_at: When the flag was created
        updated_at: When the flag was last updated
        rollout_percentage: Percentage of requests that should see this flag (0-100)
    """
    name: str
    enabled: bool = False
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rollout_percentage: int = 100

    def is_enabled_for_rollout(self) -> bool:
        """Check if flag is enabled considering rollout percentage."""
        if not self.enabled:
            return False
        if self.rollout_percentage >= 100:
            return True
        if self.rollout_percentage <= 0:
            return False
        return random.randint(1, 100) <= self.rollout_percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "rollout_percentage": self.rollout_percentage,
        }


class FeatureFlags:
    """
    Feature flags manager with hot reload support.

    Loads flags from a JSON configuration file and periodically
    reloads to pick up changes without restart.

    Thread-safe implementation using locks.

    Attributes:
        config_path: Path to the JSON config file
        reload_interval: Seconds between reload checks (default 60)
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        reload_interval: int = 60,
    ):
        """
        Initialize the feature flags manager.

        Args:
            config_path: Path to JSON config file. Uses default if not provided.
            reload_interval: Seconds between automatic reloads (default 60)
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.reload_interval = reload_interval
        self._flags: Dict[str, FeatureFlag] = {}
        self._last_reload: float = 0.0
        self._lock = threading.RLock()
        self._config_mtime: float = 0.0

        # Initial load
        self._load_flags()

    def _load_flags(self) -> None:
        """
        Load flags from the JSON configuration file.

        Creates default config if file doesn't exist.
        """
        with self._lock:
            try:
                if not self.config_path.exists():
                    logger.warning(
                        f"Feature flags config not found at {self.config_path}, "
                        "using empty defaults"
                    )
                    self._flags = {}
                    self._last_reload = time.time()
                    return

                # Check if file was modified
                current_mtime = self.config_path.stat().st_mtime
                if current_mtime == self._config_mtime and self._flags:
                    # No changes, skip reload
                    self._last_reload = time.time()
                    return

                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                now = datetime.now(timezone.utc)
                new_flags: Dict[str, FeatureFlag] = {}

                for flag_name, flag_config in config.get("flags", {}).items():
                    if isinstance(flag_config, bool):
                        # Simple format: "flag_name": true/false
                        new_flags[flag_name] = FeatureFlag(
                            name=flag_name,
                            enabled=flag_config,
                            description="",
                            created_at=now,
                            updated_at=now,
                        )
                    elif isinstance(flag_config, dict):
                        # Full format with metadata
                        new_flags[flag_name] = FeatureFlag(
                            name=flag_name,
                            enabled=flag_config.get("enabled", False),
                            description=flag_config.get("description", ""),
                            created_at=now,
                            updated_at=now,
                            rollout_percentage=flag_config.get("rollout_percentage", 100),
                        )

                self._flags = new_flags
                self._config_mtime = current_mtime
                self._last_reload = time.time()

                logger.info(
                    f"Loaded {len(self._flags)} feature flags from {self.config_path}"
                )

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in feature flags config: {e}")
            except Exception as e:
                logger.error(f"Error loading feature flags: {e}")

    def _maybe_reload(self) -> None:
        """
        Reload flags if the reload interval has elapsed.

        Only reloads if the config file has been modified.
        """
        current_time = time.time()
        if current_time - self._last_reload >= self.reload_interval:
            self._load_flags()

    def is_enabled(self, flag_name: str, default: bool = False) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_name: Name of the flag to check
            default: Default value if flag doesn't exist

        Returns:
            True if flag is enabled, False otherwise
        """
        self._maybe_reload()

        with self._lock:
            flag = self._flags.get(flag_name)
            if flag is None:
                return default
            return flag.is_enabled_for_rollout()

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """
        Get a specific feature flag by name.

        Args:
            flag_name: Name of the flag

        Returns:
            FeatureFlag if found, None otherwise
        """
        self._maybe_reload()

        with self._lock:
            return self._flags.get(flag_name)

    def get_all(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags.

        Returns:
            Dictionary of flag_name -> FeatureFlag
        """
        self._maybe_reload()

        with self._lock:
            return dict(self._flags)

    def get_all_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all flags as a dictionary suitable for JSON serialization.

        Returns:
            Dictionary of flag_name -> flag_details
        """
        flags = self.get_all()
        return {name: flag.to_dict() for name, flag in flags.items()}

    def reload(self) -> int:
        """
        Force reload flags from config file.

        Returns:
            Number of flags loaded
        """
        self._config_mtime = 0.0  # Force reload
        self._load_flags()
        return len(self._flags)

    def set_flag(self, flag_name: str, enabled: bool) -> None:
        """
        Temporarily set a flag value (in-memory only).

        Note: This does not persist to the config file.
        Use for testing or runtime overrides.

        Args:
            flag_name: Name of the flag
            enabled: New enabled state
        """
        with self._lock:
            if flag_name in self._flags:
                self._flags[flag_name].enabled = enabled
                self._flags[flag_name].updated_at = datetime.now(timezone.utc)
            else:
                self._flags[flag_name] = FeatureFlag(
                    name=flag_name,
                    enabled=enabled,
                    description="Runtime override",
                )


# Singleton instance
_feature_flags_instance: Optional[FeatureFlags] = None
_instance_lock = threading.Lock()


def get_feature_flags(
    config_path: Optional[Path] = None,
    reload_interval: int = 60,
) -> FeatureFlags:
    """
    Get the singleton FeatureFlags instance.

    Creates the instance on first call. Subsequent calls return
    the existing instance (ignoring parameters).

    Args:
        config_path: Path to JSON config file (only used on first call)
        reload_interval: Seconds between reloads (only used on first call)

    Returns:
        FeatureFlags singleton instance
    """
    global _feature_flags_instance

    if _feature_flags_instance is None:
        with _instance_lock:
            if _feature_flags_instance is None:
                _feature_flags_instance = FeatureFlags(
                    config_path=config_path,
                    reload_interval=reload_interval,
                )

    return _feature_flags_instance


def reset_feature_flags() -> None:
    """
    Reset the singleton instance.

    Useful for testing or reinitializing with different config.
    """
    global _feature_flags_instance

    with _instance_lock:
        _feature_flags_instance = None
