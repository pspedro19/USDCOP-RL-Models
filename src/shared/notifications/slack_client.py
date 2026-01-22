"""
Slack Notification Client
=========================
P1-1: Real-time notifications for trading events

Events:
- Model promotions/rollbacks
- Trading alerts (drawdown, losses)
- System health issues
- Drift detection
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SlackMessage:
    channel: str
    text: str
    blocks: Optional[List[Dict]] = None
    thread_ts: Optional[str] = None


class SlackClient:
    """
    Async Slack client for trading notifications.
    """

    SEVERITY_EMOJI = {
        AlertSeverity.INFO: "ℹ️",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.ERROR: "🔴",
        AlertSeverity.CRITICAL: "🚨",
    }

    SEVERITY_COLOR = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ff9800",
        AlertSeverity.ERROR: "#f44336",
        AlertSeverity.CRITICAL: "#9c27b0",
    }

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        fields: Optional[Dict[str, str]] = None,
        actions: Optional[List[Dict]] = None,
    ):
        """
        Send formatted alert to Slack.

        Args:
            title: Alert title
            message: Alert description
            severity: Severity level
            fields: Key-value pairs to display
            actions: Action buttons
        """
        if not self.webhook_url:
            logger.warning("Slack webhook not configured, skipping notification")
            return

        emoji = self.SEVERITY_EMOJI.get(severity, "")
        color = self.SEVERITY_COLOR.get(severity, "#808080")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]

        if fields:
            field_blocks = []
            for key, value in fields.items():
                field_blocks.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}"
                })

            # Slack allows max 10 fields, 2 per row
            for i in range(0, len(field_blocks), 2):
                blocks.append({
                    "type": "section",
                    "fields": field_blocks[i:i+2]
                })

        if actions:
            blocks.append({
                "type": "actions",
                "elements": actions
            })

        # Add timestamp footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} COT"
                }
            ]
        })

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks
                }
            ]
        }

        try:
            session = await self._get_session()
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")
                else:
                    logger.debug(f"Slack notification sent: {title}")
        except Exception as e:
            logger.error(f"Slack notification error: {e}")

    async def notify_model_promotion(
        self,
        model_id: str,
        from_stage: str,
        to_stage: str,
        promoted_by: str,
        metrics: Optional[Dict] = None,
    ):
        """Notify model promotion event."""
        fields = {
            "Model": model_id,
            "Transition": f"{from_stage} → {to_stage}",
            "Promoted By": promoted_by,
        }

        if metrics:
            fields["Sharpe"] = f"{metrics.get('sharpe', 'N/A'):.2f}"
            fields["Win Rate"] = f"{metrics.get('win_rate', 0) * 100:.1f}%"

        await self.send_alert(
            title=f"Model Promoted to {to_stage.upper()}",
            message=f"Model `{model_id}` has been promoted.",
            severity=AlertSeverity.INFO,
            fields=fields,
            actions=[
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View in MLflow"},
                    "url": f"http://mlflow:5000/#/models/{model_id}"
                }
            ]
        )

    async def notify_rollback(
        self,
        from_model: str,
        to_model: str,
        reason: str,
        initiated_by: str,
    ):
        """Notify model rollback event."""
        await self.send_alert(
            title="MODEL ROLLBACK",
            message=f"Production model rolled back due to: {reason}",
            severity=AlertSeverity.WARNING,
            fields={
                "Previous Model": from_model,
                "New Model": to_model,
                "Initiated By": initiated_by,
                "Reason": reason,
            }
        )

    async def notify_kill_switch(self, reason: str, activated_by: str):
        """Notify kill switch activation."""
        await self.send_alert(
            title="🔴 KILL SWITCH ACTIVATED",
            message="ALL TRADING HAS BEEN STOPPED",
            severity=AlertSeverity.CRITICAL,
            fields={
                "Reason": reason,
                "Activated By": activated_by,
                "Action": "All positions closed",
            },
            actions=[
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Dashboard"},
                    "url": "https://dashboard.internal/operations"
                }
            ]
        )

    async def notify_drift_detected(
        self,
        feature: str,
        psi_value: float,
        threshold: float = 0.2,
    ):
        """Notify feature drift detection."""
        await self.send_alert(
            title="Feature Drift Detected",
            message=f"Feature `{feature}` shows significant drift.",
            severity=AlertSeverity.WARNING,
            fields={
                "Feature": feature,
                "PSI Value": f"{psi_value:.3f}",
                "Threshold": f"{threshold:.3f}",
                "Status": "⚠️ Above threshold" if psi_value > threshold else "✅ Within limits",
            }
        )

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Global instance
_slack_client: Optional[SlackClient] = None


def get_slack_client() -> SlackClient:
    """Get or create global Slack client."""
    global _slack_client
    if _slack_client is None:
        _slack_client = SlackClient()
    return _slack_client


# Sync wrapper for non-async contexts
def send_slack_alert(title: str, message: str, severity: str = "info", **kwargs):
    """Synchronous wrapper for sending Slack alerts."""
    client = get_slack_client()
    severity_enum = AlertSeverity(severity)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(
        client.send_alert(title, message, severity_enum, **kwargs)
    )
