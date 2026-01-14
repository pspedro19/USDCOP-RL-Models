"""
Alert Service - Notification System
====================================
Sends alerts via multiple channels (Slack, Email, etc.)

SOLID Principles:
- Single Responsibility: Each notifier handles one channel
- Open/Closed: New channels via registration
- Liskov Substitution: All notifiers are interchangeable
- Interface Segregation: Minimal notifier interface
- Dependency Inversion: Depends on Notifier protocol

Design Patterns:
- Strategy Pattern: Different notification strategies
- Observer Pattern: Multiple observers for alerts
- Chain of Responsibility: Process through multiple channels

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from contracts.backtest_contracts import (
    Alert,
    AlertSeverity,
    ValidationReport,
    ValidationResult,
    BacktestMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS & INTERFACES
# =============================================================================

class Notifier(Protocol):
    """Protocol for notification channels"""

    def send(self, alert: Alert) -> bool:
        """Send alert. Returns True if successful."""
        ...

    @property
    def channel_name(self) -> str:
        """Name of the notification channel"""
        ...


# =============================================================================
# ABSTRACT BASE NOTIFIER
# =============================================================================

class AbstractNotifier(ABC):
    """Abstract base class for notifiers"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def send(self, alert: Alert) -> bool:
        """Send alert with error handling"""
        if not self.enabled:
            logger.debug(f"{self.channel_name} notifier disabled, skipping")
            return False

        try:
            return self._send_impl(alert)
        except Exception as e:
            logger.error(f"Failed to send via {self.channel_name}: {e}")
            return False

    @abstractmethod
    def _send_impl(self, alert: Alert) -> bool:
        """Implementation-specific send logic"""
        pass

    @property
    @abstractmethod
    def channel_name(self) -> str:
        pass


# =============================================================================
# CONCRETE NOTIFIERS
# =============================================================================

class SlackNotifier(AbstractNotifier):
    """
    Slack notification via webhook.

    Requires SLACK_WEBHOOK_URL environment variable.
    """

    def __init__(self, webhook_url: Optional[str] = None, enabled: bool = True):
        super().__init__(enabled)
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set, Slack notifications disabled")
            self.enabled = False

    @property
    def channel_name(self) -> str:
        return "slack"

    def _send_impl(self, alert: Alert) -> bool:
        """Send to Slack via webhook"""
        import requests

        # Build Slack message
        color = self._get_color(alert.severity)
        emoji = self._get_emoji(alert.severity)

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": self._build_fields(alert),
                    "footer": f"Backtest Validation | {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                }
            ]
        }

        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            logger.info(f"Slack alert sent: {alert.title}")
            return True
        else:
            logger.error(f"Slack send failed: {response.status_code} - {response.text}")
            return False

    def _get_color(self, severity: AlertSeverity) -> str:
        return {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ffcc00",   # Yellow
            AlertSeverity.CRITICAL: "#ff0000",  # Red
        }.get(severity, "#808080")

    def _get_emoji(self, severity: AlertSeverity) -> str:
        return {
            AlertSeverity.INFO: ":white_check_mark:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }.get(severity, ":bell:")

    def _build_fields(self, alert: Alert) -> List[Dict]:
        fields = []

        if alert.model_id:
            fields.append({
                "title": "Model",
                "value": alert.model_id,
                "short": True,
            })

        if alert.metrics:
            for key, value in alert.metrics.items():
                if isinstance(value, float):
                    value = f"{value:.2f}"
                fields.append({
                    "title": key.replace("_", " ").title(),
                    "value": str(value),
                    "short": True,
                })

        return fields


class EmailNotifier(AbstractNotifier):
    """
    Email notification via SMTP.

    Requires SMTP_* environment variables.
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        super().__init__(enabled)
        self.smtp_host = smtp_host or os.environ.get("SMTP_HOST")
        self.smtp_port = smtp_port or int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.environ.get("SMTP_USER")
        self.smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")
        self.from_email = from_email or os.environ.get("ALERT_FROM_EMAIL", "alerts@trading.local")
        self.to_emails = to_emails or os.environ.get("ALERT_TO_EMAILS", "").split(",")

        if not self.smtp_host or not self.to_emails:
            logger.warning("SMTP not configured, email notifications disabled")
            self.enabled = False

    @property
    def channel_name(self) -> str:
        return "email"

    def _send_impl(self, alert: Alert) -> bool:
        """Send email via SMTP"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg["From"] = self.from_email
        msg["To"] = ", ".join(self.to_emails)

        # Plain text version
        text = f"{alert.title}\n\n{alert.message}"
        if alert.metrics:
            text += "\n\nMetrics:\n"
            for k, v in alert.metrics.items():
                text += f"  {k}: {v}\n"

        # HTML version
        html = self._build_html(alert)

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        # Send
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.smtp_user and self.smtp_password:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.from_email, self.to_emails, msg.as_string())

        logger.info(f"Email alert sent to {self.to_emails}")
        return True

    def _build_html(self, alert: Alert) -> str:
        severity_colors = {
            AlertSeverity.INFO: "#28a745",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
        }
        color = severity_colors.get(alert.severity, "#6c757d")

        metrics_html = ""
        if alert.metrics:
            metrics_html = "<table style='border-collapse: collapse; margin: 10px 0;'>"
            for k, v in alert.metrics.items():
                metrics_html += f"<tr><td style='padding: 5px; border: 1px solid #ddd;'>{k}</td>"
                metrics_html += f"<td style='padding: 5px; border: 1px solid #ddd;'>{v}</td></tr>"
            metrics_html += "</table>"

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 15px;">
                <h2 style="color: {color}; margin: 0;">{alert.title}</h2>
                <p style="color: #666;">{alert.message}</p>
                {metrics_html}
                <p style="color: #999; font-size: 12px;">
                    Model: {alert.model_id or 'N/A'} |
                    {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC
                </p>
            </div>
        </body>
        </html>
        """


class LogNotifier(AbstractNotifier):
    """
    Log notification for development/testing.

    Always enabled, logs to standard logging.
    """

    @property
    def channel_name(self) -> str:
        return "log"

    def _send_impl(self, alert: Alert) -> bool:
        """Log the alert"""
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
        }.get(alert.severity, logger.info)

        log_func(
            f"ALERT [{alert.severity.value}] {alert.title}: {alert.message} "
            f"(model={alert.model_id}, metrics={alert.metrics})"
        )

        return True


class WebhookNotifier(AbstractNotifier):
    """
    Generic webhook notification.

    Sends JSON payload to any HTTP endpoint.
    """

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None, enabled: bool = True):
        super().__init__(enabled)
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    @property
    def channel_name(self) -> str:
        return "webhook"

    def _send_impl(self, alert: Alert) -> bool:
        """Send to webhook"""
        import requests

        payload = {
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "model_id": alert.model_id,
            "metrics": alert.metrics,
            "timestamp": alert.created_at.isoformat(),
            "source": alert.source,
        }

        response = requests.post(
            self.webhook_url,
            json=payload,
            headers=self.headers,
            timeout=10,
        )

        return response.status_code < 400


# =============================================================================
# ALERT SERVICE - Chain of Responsibility
# =============================================================================

class AlertService:
    """
    Central alert service managing multiple notification channels.

    Implements Chain of Responsibility pattern for multi-channel dispatch.

    Usage:
        service = AlertService()
        service.register_notifier(SlackNotifier())
        service.register_notifier(EmailNotifier())

        alert = Alert(
            title="Model Degradation Detected",
            message="Sharpe ratio dropped below threshold",
            severity=AlertSeverity.WARNING,
        )

        service.send_alert(alert)
    """

    def __init__(self, default_notifiers: bool = True):
        self._notifiers: Dict[str, AbstractNotifier] = {}

        if default_notifiers:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default notifiers"""
        # Always log
        self.register_notifier(LogNotifier())

        # Try Slack
        try:
            slack = SlackNotifier()
            if slack.enabled:
                self.register_notifier(slack)
        except Exception:
            pass

        # Try Email
        try:
            email = EmailNotifier()
            if email.enabled:
                self.register_notifier(email)
        except Exception:
            pass

    def register_notifier(self, notifier: AbstractNotifier) -> None:
        """Register a notification channel"""
        self._notifiers[notifier.channel_name] = notifier
        logger.info(f"Registered notifier: {notifier.channel_name}")

    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Send alert to specified channels.

        Args:
            alert: Alert to send
            channels: List of channel names (None = all registered)

        Returns:
            Dict of channel -> success status
        """
        results = {}

        target_channels = channels or list(self._notifiers.keys())

        for channel in target_channels:
            if channel in self._notifiers:
                success = self._notifiers[channel].send(alert)
                results[channel] = success
            else:
                logger.warning(f"Unknown channel: {channel}")
                results[channel] = False

        return results

    def send_validation_alert(
        self,
        report: ValidationReport,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Send alert based on validation report.

        Auto-determines severity from report results.
        """
        # Determine severity
        if report.overall_result == ValidationResult.FAILED:
            severity = AlertSeverity.CRITICAL
            title = f"Model Validation FAILED: {report.model_id}"
        elif report.overall_result == ValidationResult.DEGRADED:
            severity = AlertSeverity.WARNING
            title = f"Model Degradation Detected: {report.model_id}"
        else:
            severity = AlertSeverity.INFO
            title = f"Model Validation Passed: {report.model_id}"

        # Build message
        message_parts = [
            f"Validation Result: {report.overall_result.value}",
            f"Checks Passed: {report.passed_checks}/{len(report.checks)}",
        ]

        if report.critical_failures:
            message_parts.append("\nCritical Failures:")
            for check in report.critical_failures:
                message_parts.append(f"  - {check.check_name}: {check.message}")

        # Build metrics dict
        metrics = {}
        if report.backtest_result.metrics:
            m = report.backtest_result.metrics
            metrics = {
                "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown": f"{m.max_drawdown_pct:.1%}",
                "win_rate": f"{m.win_rate:.1%}",
                "total_trades": m.total_trades,
                "total_pnl": f"${m.total_pnl_usd:.2f}",
            }

        alert = Alert(
            title=title,
            message="\n".join(message_parts),
            severity=severity,
            model_id=report.model_id,
            metrics=metrics,
            validation_report=report,
            channels=channels or ["slack", "log"],
        )

        return self.send_alert(alert, channels)

    def list_channels(self) -> List[str]:
        """List registered channels"""
        return list(self._notifiers.keys())


# =============================================================================
# ALERT BUILDER - Fluent API
# =============================================================================

class AlertBuilder:
    """
    Builder for creating alerts with fluent API.

    Usage:
        alert = (AlertBuilder()
            .with_title("Model Degradation")
            .with_message("Sharpe dropped below 1.0")
            .with_severity(AlertSeverity.WARNING)
            .for_model("ppo_model")
            .with_metrics({"sharpe": 0.8})
            .build())
    """

    def __init__(self):
        self._title: str = "Alert"
        self._message: str = ""
        self._severity: AlertSeverity = AlertSeverity.INFO
        self._source: str = "backtest_validation"
        self._model_id: Optional[str] = None
        self._metrics: Dict[str, Any] = {}
        self._channels: List[str] = ["log"]

    def with_title(self, title: str) -> "AlertBuilder":
        self._title = title
        return self

    def with_message(self, message: str) -> "AlertBuilder":
        self._message = message
        return self

    def with_severity(self, severity: AlertSeverity) -> "AlertBuilder":
        self._severity = severity
        return self

    def for_model(self, model_id: str) -> "AlertBuilder":
        self._model_id = model_id
        return self

    def with_metrics(self, metrics: Dict[str, Any]) -> "AlertBuilder":
        self._metrics = metrics
        return self

    def to_channels(self, channels: List[str]) -> "AlertBuilder":
        self._channels = channels
        return self

    def critical(self) -> "AlertBuilder":
        self._severity = AlertSeverity.CRITICAL
        return self

    def warning(self) -> "AlertBuilder":
        self._severity = AlertSeverity.WARNING
        return self

    def info(self) -> "AlertBuilder":
        self._severity = AlertSeverity.INFO
        return self

    def build(self) -> Alert:
        return Alert(
            title=self._title,
            message=self._message,
            severity=self._severity,
            source=self._source,
            model_id=self._model_id,
            metrics=self._metrics,
            channels=self._channels,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global service instance
_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """Get or create global alert service"""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service


def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    model_id: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    channels: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Convenience function to send an alert"""
    alert = Alert(
        title=title,
        message=message,
        severity=severity,
        model_id=model_id,
        metrics=metrics or {},
        channels=channels or ["log"],
    )

    return get_alert_service().send_alert(alert, channels)
