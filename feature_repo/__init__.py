"""
Feast Feature Repository for USDCOP Trading
============================================
This package contains Feast feature definitions for the trading system.

Files:
- feature_store.yaml: Feast configuration (registry, stores, provider)
- features.py: Entity, FeatureView, and FeatureService definitions

Usage:
    # Apply feature definitions
    cd feature_repo
    feast apply

    # Materialize features to online store
    feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

    # Start feature server
    feast serve -h 0.0.0.0 -p 6566

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
"""

from .features import (
    bar_entity,
    technical_features,
    macro_features,
    state_features,
    observation_15d,
    FEAST_FEATURE_ORDER,
)

__all__ = [
    "bar_entity",
    "technical_features",
    "macro_features",
    "state_features",
    "observation_15d",
    "FEAST_FEATURE_ORDER",
]
