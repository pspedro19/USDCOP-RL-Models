# backend/src/config/__init__.py
"""
Configuration module for USD/COP Forecasting Pipeline.

Centralizes all environment variables and configuration settings.
"""

from .settings import (
    Settings,
    get_settings,
    # Path constants
    BASE_DIR,
    PROJECT_ROOT,
    BACKEND_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    LOGS_DIR,
)

__all__ = [
    "Settings",
    "get_settings",
    "BASE_DIR",
    "PROJECT_ROOT",
    "BACKEND_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
]
