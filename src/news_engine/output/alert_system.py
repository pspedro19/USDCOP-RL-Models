"""
Alert System (SDD-06 §4)
==========================
Detects breaking news and extreme market events.
Optional Slack webhook notification.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import requests

from src.news_engine.config import AlertConfig
from src.news_engine.models import EnrichedArticle

logger = logging.getLogger(__name__)


class AlertSystem:
    """Breaking news alert system."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.cfg = config or AlertConfig()

    def check_articles(self, articles: list[EnrichedArticle]) -> list[dict]:
        """Check articles for alert-worthy conditions.

        Returns:
            List of alert dicts with {type, severity, message, article_url}.
        """
        if not self.cfg.enabled:
            return []

        alerts = []
        for article in articles:
            if article.is_breaking:
                alert = {
                    "type": "breaking_news",
                    "severity": "high",
                    "message": f"[BREAKING] {article.raw.title}",
                    "article_url": article.raw.url,
                    "source": article.raw.source_id,
                    "sentiment": article.sentiment_score,
                    "category": article.category,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                alerts.append(alert)

            # Extreme GDELT tone
            if article.raw.gdelt_tone is not None:
                if article.raw.gdelt_tone <= self.cfg.gdelt_tone_threshold:
                    alert = {
                        "type": "extreme_negative_tone",
                        "severity": "high",
                        "message": (
                            f"[TONE ALERT] Extreme negative tone "
                            f"({article.raw.gdelt_tone:.1f}): {article.raw.title}"
                        ),
                        "article_url": article.raw.url,
                        "gdelt_tone": article.raw.gdelt_tone,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    alerts.append(alert)

        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts")
            self._send_notifications(alerts)

        return alerts

    def check_volume_spike(
        self,
        current_count: int,
        historical_mean: float,
        historical_std: float,
    ) -> Optional[dict]:
        """Check if current article volume is a spike."""
        if historical_std <= 0:
            return None

        z_score = (current_count - historical_mean) / historical_std
        if z_score >= self.cfg.volume_spike_threshold:
            alert = {
                "type": "volume_spike",
                "severity": "medium",
                "message": (
                    f"[VOLUME SPIKE] {current_count} articles "
                    f"(z={z_score:.1f}, mean={historical_mean:.0f})"
                ),
                "current_count": current_count,
                "z_score": round(z_score, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.warning(alert["message"])
            self._send_notifications([alert])
            return alert
        return None

    def _send_notifications(self, alerts: list[dict]) -> None:
        """Send alert notifications via Slack webhook."""
        if not self.cfg.slack_webhook:
            return

        for alert in alerts:
            try:
                payload = {
                    "text": f"*{alert.get('type', 'alert').upper()}*\n{alert['message']}",
                }
                requests.post(
                    self.cfg.slack_webhook,
                    json=payload,
                    timeout=5,
                )
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
