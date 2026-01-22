"""
Regression Tests Package - SSOT Compliance
===========================================
Tests to verify that Single Source of Truth (SSOT) contracts are properly
maintained across the codebase.

These tests ensure that:
1. Action enum values are correct and immutable (SELL=0, HOLD=1, BUY=2)
2. Feature order has exactly 15 elements in the correct order
3. Trading flags are properly respected (TRADING_ENABLED, KILL_SWITCH)
4. No duplicate definitions exist in the codebase

Run with: pytest tests/regression/ -v
"""
