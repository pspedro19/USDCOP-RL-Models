"""Dependency Handler for USDCOP Airflow DAGs

This module handles missing dependencies gracefully to allow Airflow to start
even when heavy ML dependencies are not available.
"""

import logging
import warnings
from typing import Optional, Any

logger = logging.getLogger(__name__)

class MissingDependencyHandler:
    """Handles missing dependencies gracefully"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.is_available = False
        self.module = None
        self._load_module()
    
    def _load_module(self):
        """Try to load the module"""
        try:
            if self.module_name == 'scipy':
                import scipy
                self.module = scipy
            elif self.module_name == 'gymnasium':
                import gymnasium
                self.module = gymnasium
            elif self.module_name == 'torch':
                import torch
                self.module = torch
            elif self.module_name == 'stable_baselines3':
                import stable_baselines3
                self.module = stable_baselines3
            elif self.module_name == 'sklearn':
                import sklearn
                self.module = sklearn
            else:
                self.module = __import__(self.module_name)
            
            self.is_available = True
            logger.info(f"Successfully loaded {self.module_name}")
            
        except ImportError as e:
            self.is_available = False
            logger.warning(f"Module {self.module_name} not available: {e}")
            logger.info(f"DAGs using {self.module_name} will be disabled or use fallback implementations")
    
    def get_module(self) -> Optional[Any]:
        """Get the module if available"""
        return self.module if self.is_available else None
    
    def require_module(self, fallback_message: str = None):
        """Require the module or raise informative error"""
        if not self.is_available:
            message = fallback_message or f"{self.module_name} is required but not available. Install it with: pip install {self.module_name}"
            raise ImportError(message)
        return self.module

# Global dependency handlers - lazy loaded
_handlers_cache = {}

def get_handler(module_name: str) -> MissingDependencyHandler:
    """Get or create a dependency handler lazily"""
    if module_name not in _handlers_cache:
        _handlers_cache[module_name] = MissingDependencyHandler(module_name)
    return _handlers_cache[module_name]

# Convenience functions for getting handlers
def get_scipy_handler() -> MissingDependencyHandler:
    return get_handler('scipy')

def get_gymnasium_handler() -> MissingDependencyHandler:
    return get_handler('gymnasium')

def get_torch_handler() -> MissingDependencyHandler:
    return get_handler('torch')

def get_stable_baselines3_handler() -> MissingDependencyHandler:
    return get_handler('stable_baselines3')

def get_sklearn_handler() -> MissingDependencyHandler:
    return get_handler('sklearn')

def check_ml_dependencies() -> dict:
    """Check status of all ML dependencies"""
    dependencies = ['scipy', 'gymnasium', 'torch', 'stable_baselines3', 'sklearn']
    return {dep: get_handler(dep).is_available for dep in dependencies}

def log_dependency_status():
    """Log the status of all dependencies"""
    status = check_ml_dependencies()
    logger.info("ML Dependencies Status:")
    for dep, available in status.items():
        status_str = "✓ Available" if available else "✗ Missing"
        logger.info(f"  {dep}: {status_str}")

    if not all(status.values()):
        logger.warning("Some ML dependencies are missing. Some DAGs may be disabled.")
        logger.info("To install missing dependencies, run: pip install -r requirements.txt")