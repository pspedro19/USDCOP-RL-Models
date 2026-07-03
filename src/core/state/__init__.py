"""
Core State Management for USD/COP Trading System
=================================================

State tracking and management for RL trading models.

Author: Pedro @ Lean Tech Solutions / Claude Code
Version: 19.0.0
Date: 2025-01-07
"""

from .state_tracker import (
    ModelState,
    StateTracker,
    create_state_tracker
)

__all__ = [
    'ModelState',
    'StateTracker',
    'create_state_tracker',
]
