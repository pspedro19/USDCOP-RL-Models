"""
Unified Configuration System
============================
Single source of truth for all configuration across the trading system.
Supports YAML files, environment variables, and runtime configuration.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """System-wide configuration settings"""
    
    # Environment
    env: str = "development"
    debug: bool = False
    
    # Paths
    project_root: Path = None
    config_dir: Path = None
    data_dir: Path = None
    logs_dir: Path = None
    
    # Database
    db_type: str = "sqlite"
    db_path: str = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading"
    db_user: str = None
    db_password: str = None
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = None
    
    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = False
    metrics_port: int = 9090
    
    # Trading
    default_symbol: str = "USDCOP"
    default_timeframe: str = "M5"
    
    def __post_init__(self):
        """Initialize derived paths"""
        if self.project_root is None:
            # Navigate up from src/core/config to project root
            self.project_root = Path(__file__).parent.parent.parent.parent
        
        if self.config_dir is None:
            self.config_dir = self.project_root / "configs"
        
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        
        if self.db_path is None and self.db_type == "sqlite":
            self.db_path = str(self.data_dir / "trading.db")
        
        if self.log_file is None:
            self.log_file = str(self.logs_dir / "trading.log")
        
        # Create directories if they don't exist
        for dir_path in [self.config_dir, self.data_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class UnifiedConfigLoader:
    """
    Unified configuration loader for all components.
    Implements singleton pattern for consistent configuration access.
    """
    
    _instance = None
    _configs: Dict[str, Dict[str, Any]] = {}
    _system_config: SystemConfig = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self._load_env()
            self._system_config = SystemConfig()
            self._load_all_configs()
            self.initialized = True
            logger.info("Unified configuration system initialized")
    
    def _load_env(self):
        """Load environment variables from .env file"""
        env_files = [
            Path(".env"),
            Path(".env.local"),
            self._get_project_root() / ".env"
        ]
        
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file}")
                break
    
    def _get_project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent.parent.parent
    
    def _load_all_configs(self):
        """Load all YAML configuration files"""
        config_files = [
            "mt5_config.yaml",
            "usdcop_config.yaml",
            "quality_thresholds.yaml",
            "dashboard_config.yaml"
        ]
        
        for config_file in config_files:
            self._load_yaml_config(config_file)
    
    def _load_yaml_config(self, filename: str) -> bool:
        """Load a single YAML configuration file"""
        try:
            config_path = self._system_config.config_dir / filename
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Replace environment variables
            config_data = self._replace_env_vars(config_data)
            
            # Store by name without extension
            config_name = filename.replace('.yaml', '').replace('.yml', '')
            self._configs[config_name] = config_data
            
            logger.info(f"Loaded configuration: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return False
    
    def _replace_env_vars(self, config: Any) -> Any:
        """Recursively replace environment variables in configuration"""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check for environment variable syntax: ${VAR_NAME}
            if config.startswith("${") and config.endswith("}"):
                env_var = config[2:-1]
                value = os.getenv(env_var)
                if value is None:
                    logger.warning(f"Environment variable {env_var} not found")
                    return config
                # Try to parse as JSON for complex types
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Try to convert to appropriate type
                    if value.lower() in ('true', 'false'):
                        return value.lower() == 'true'
                    try:
                        return int(value)
                    except ValueError:
                        try:
                            return float(value)
                        except ValueError:
                            return value
            # Check for inline environment variables: text${VAR}text
            elif "${" in config and "}" in config:
                import re
                pattern = r'\$\{([^}]+)\}'
                
                def replacer(match):
                    env_var = match.group(1)
                    return os.getenv(env_var, match.group(0))
                
                return re.sub(pattern, replacer, config)
        return config
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            name: Configuration name (without .yaml extension)
            
        Returns:
            Configuration dictionary or empty dict if not found
        """
        if name not in self._configs:
            # Try to load if not already loaded
            if self._load_yaml_config(f"{name}.yaml"):
                return self._configs.get(name, {})
            logger.warning(f"Configuration '{name}' not found")
            return {}
        return self._configs[name].copy()
    
    def get_nested_config(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., "mt5.connection.timeout")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        if not parts:
            return default
        
        # First part should be config name
        config_name = parts[0]
        config = self.get_config(config_name)
        
        # Navigate through nested structure
        result = config
        for part in parts[1:]:
            if isinstance(result, dict) and part in result:
                result = result[part]
            else:
                return default
        
        return result
    
    def set_config(self, name: str, config: Dict[str, Any]):
        """Set or update a configuration"""
        self._configs[name] = config
        logger.info(f"Updated configuration: {name}")
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return self._configs.copy()
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        return self._system_config
    
    def reload(self):
        """Reload all configurations"""
        self._configs.clear()
        self._load_all_configs()
        logger.info("Configurations reloaded")
    
    def reload_config(self, name: str) -> bool:
        """Reload a specific configuration"""
        return self._load_yaml_config(f"{name}.yaml")
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations into one.
        Later configs override earlier ones.
        """
        result = {}
        for name in config_names:
            config = self.get_config(name)
            result = self._deep_merge(result, config)
        return result
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def validate_config(self, name: str, schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against a schema.
        Returns list of validation errors.
        """
        errors = []
        config = self.get_config(name)
        
        # Simple validation - can be extended with jsonschema
        for key, expected_type in schema.items():
            if key not in config:
                errors.append(f"Missing required key: {key}")
            elif not isinstance(config[key], expected_type):
                errors.append(f"Invalid type for {key}: expected {expected_type.__name__}")
        
        return errors
    
    def export_config(self, name: str, filepath: Path):
        """Export configuration to file"""
        config = self.get_config(name)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                yaml.safe_dump(config, f, default_flow_style=False)
            elif filepath.suffix == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Exported configuration to {filepath}")


# Singleton instance
config_loader = UnifiedConfigLoader()


# Convenience functions for easy access
def get_config(name: str = None) -> Any:
    """
    Get configuration by name or all configurations.
    
    Args:
        name: Configuration name or None for all configs
        
    Returns:
        Configuration dict or all configurations
    """
    if name:
        return config_loader.get_config(name)
    return config_loader.get_all_configs()


def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return config_loader.get_system_config()


def get_nested_config(path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation"""
    return config_loader.get_nested_config(path, default)


def reload_configs():
    """Reload all configurations"""
    config_loader.reload()


# Environment-specific configuration helpers
def is_production() -> bool:
    """Check if running in production environment"""
    return get_system_config().env.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return get_system_config().env.lower() == "development"


def is_debug() -> bool:
    """Check if debug mode is enabled"""
    return get_system_config().debug