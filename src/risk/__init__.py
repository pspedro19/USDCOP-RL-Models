"""
Risk Management Module
======================

Safety layer for USD/COP trading system.
Provides kill switches, daily limits, and cooldown mechanisms
to prevent catastrophic losses.
"""

from .risk_manager import RiskManager, RiskLimits

__all__ = ['RiskManager', 'RiskLimits']
