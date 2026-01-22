"""
USD/COP Trading System - Notifications Module
==============================================

This module provides notification capabilities for the trading system,
with Slack as the primary notification channel.

Components:
    - SlackClient: Async client for sending Slack notifications
    - AlertSeverity: Enum for categorizing alert severity levels
    - SlackMessage: Dataclass for structured Slack messages
    - get_slack_client: Helper function to get the singleton instance
    - send_slack_alert: Synchronous wrapper for non-async contexts

Events Supported:
    - Model promotions/rollbacks
    - Trading alerts (drawdown, losses)
    - System health issues
    - Feature drift detection
    - Kill switch activation

Usage:
    # Async usage
    from src.shared.notifications import SlackClient, AlertSeverity, get_slack_client

    client = get_slack_client()
    await client.send_alert(
        title="Alert Title",
        message="Alert description",
        severity=AlertSeverity.WARNING,
        fields={"Key": "Value"}
    )

    # Sync usage (for non-async contexts like Airflow DAGs)
    from src.shared.notifications import send_slack_alert

    send_slack_alert(
        title="Alert Title",
        message="Alert description",
        severity="warning"
    )

    # Specialized notifications
    await client.notify_model_promotion(
        model_id="ppo_v20_20260115",
        from_stage="staging",
        to_stage="production",
        promoted_by="operator@company.com",
        metrics={"sharpe": 1.5, "win_rate": 0.55}
    )

    await client.notify_rollback(
        from_model="ppo_v20",
        to_model="ppo_v19",
        reason="Performance degradation",
        initiated_by="operator@company.com"
    )

    await client.notify_kill_switch(
        reason="5% drawdown exceeded",
        activated_by="system"
    )

    await client.notify_drift_detected(
        feature="rsi_9",
        psi_value=0.25,
        threshold=0.2
    )

Author: Trading Operations Team
Version: 1.0.0
Date: 2026-01-17
"""

from .slack_client import (
    # Main classes
    SlackClient,
    SlackMessage,
    AlertSeverity,
    # Helper functions
    get_slack_client,
    send_slack_alert,
)

__all__ = [
    # Main classes
    "SlackClient",
    "SlackMessage",
    "AlertSeverity",
    # Helper functions
    "get_slack_client",
    "send_slack_alert",
]

__version__ = "1.0.0"
