"""
Configuration loader for USDCOP Trading RL System
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader with environment variable support"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        
        # Load environment variables
        load_dotenv()
        
    def load_yaml(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if use_cache and filename in self.config_cache:
            return self.config_cache[filename]
            
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Replace environment variables
            content = self._replace_env_vars(content)
            
            # Parse YAML
            config = yaml.safe_load(content)
            
            if use_cache:
                self.config_cache[filename] = config
                
            logger.info(f"Loaded configuration: {filename}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration {filename}: {e}")
            raise
    
    def load_json(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load JSON configuration file"""
        if use_cache and filename in self.config_cache:
            return self.config_cache[filename]
            
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Replace environment variables
            content = self._replace_env_vars(content)
            
            # Parse JSON
            config = json.loads(content)
            
            if use_cache:
                self.config_cache[filename] = config
                
            logger.info(f"Loaded configuration: {filename}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration {filename}: {e}")
            raise
    
    def _replace_env_vars(self, content: str) -> str:
        """Replace environment variables in configuration content"""
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default_value:
                return default_value
            else:
                logger.warning(f"Environment variable {var_name} not found, using empty string")
                return ""
        
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_var, content)
    
    def get_config(self, config_type: str) -> Dict[str, Any]:
        """Get configuration by type"""
        config_mapping = {
            'mt5': 'mt5_config.yaml',
            'usdcop': 'usdcop_config.yaml',
            'quality': 'quality_thresholds.yaml',
            'dashboard': 'dashboard_config.yaml'
        }
        
        if config_type not in config_mapping:
            raise ValueError(f"Unknown config type: {config_type}")
            
        filename = config_mapping[config_type]
        return self.load_yaml(filename)
    
    def get_setting(self, config_type: str, key_path: str, 
                   default: Any = None) -> Any:
        """Get a specific setting from configuration"""
        try:
            config = self.get_config(config_type)
            
            # Navigate through nested keys
            keys = key_path.split('.')
            value = config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
                    
            return value
            
        except Exception as e:
            logger.warning(f"Error getting setting {key_path} from {config_type}: {e}")
            return default
    
    def reload_config(self, filename: Optional[str] = None):
        """Reload configuration files"""
        if filename:
            if filename in self.config_cache:
                del self.config_cache[filename]
            logger.info(f"Reloaded configuration: {filename}")
        else:
            self.config_cache.clear()
            logger.info("Reloaded all configurations")
    
    def validate_config(self, config: Dict[str, Any], 
                       required_keys: list) -> bool:
        """Validate configuration has required keys"""
        missing_keys = []
        
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
                
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
            
        return True
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to file"""
        file_path = self.config_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, 
                          allow_unicode=True, indent=2)
                
            logger.info(f"Saved configuration: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving configuration {filename}: {e}")
            raise

# Global config loader instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def get_config(config_type: str) -> Dict[str, Any]:
    """Get configuration by type"""
    return get_config_loader().get_config(config_type)

def get_setting(config_type: str, key_path: str, default: Any = None) -> Any:
    """Get a specific setting from configuration"""
    return get_config_loader().get_setting(config_type, key_path, default)
