# -*- coding: utf-8 -*-
"""
Dead-Letter-Queue (DLQ) Service
===============================
Persistent storage and retry mechanism for failed data extractions.

Contract: CTR-L0-DLQ-001

This service provides:
1. Persistence of failed extraction attempts
2. Automatic retry with exponential backoff
3. Failure analytics and alerting
4. Manual review queue for repeated failures

Usage:
    from services.dlq_service import DeadLetterQueueService

    dlq = DeadLetterQueueService()

    # Save failed extraction
    dlq.save_failed_extraction(
        source='fred',
        variable='FEDFUNDS',
        error='ConnectionTimeout',
        payload={'series_id': 'FEDFUNDS', 'start': '2024-01-01'}
    )

    # Retry all dead letters
    results = dlq.retry_dead_letters()

    # Get failure stats
    stats = dlq.get_failure_stats()

Version: 1.0.0
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DLQStatus(Enum):
    """Status of a dead letter entry."""
    PENDING = "pending"           # Awaiting retry
    RETRYING = "retrying"         # Currently being retried
    RESOLVED = "resolved"         # Successfully retried
    FAILED_PERMANENT = "failed_permanent"  # Max retries exceeded
    MANUAL_REVIEW = "manual_review"  # Requires human intervention


@dataclass
class DeadLetterEntry:
    """A single dead letter queue entry."""
    id: str
    source: str
    variable: str
    error_message: str
    error_type: str
    payload: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    max_retries: int = 5
    status: DLQStatus = DLQStatus.PENDING
    next_retry_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.next_retry_at is None:
            self.next_retry_at = self._calculate_next_retry()

    def _calculate_next_retry(self) -> datetime:
        """Calculate next retry time with exponential backoff."""
        # Exponential backoff: 2^retry_count minutes, max 24 hours
        backoff_minutes = min(2 ** self.retry_count, 1440)
        return datetime.utcnow() + timedelta(minutes=backoff_minutes)

    def can_retry(self) -> bool:
        """Check if this entry can be retried."""
        if self.status in [DLQStatus.RESOLVED, DLQStatus.FAILED_PERMANENT, DLQStatus.MANUAL_REVIEW]:
            return False
        if self.retry_count >= self.max_retries:
            return False
        if self.next_retry_at and datetime.utcnow() < self.next_retry_at:
            return False
        return True

    def increment_retry(self, error: Optional[str] = None):
        """Increment retry count and update next retry time."""
        self.retry_count += 1
        self.updated_at = datetime.utcnow()

        if error:
            self.error_message = error

        if self.retry_count >= self.max_retries:
            self.status = DLQStatus.FAILED_PERMANENT
            self.next_retry_at = None
        else:
            self.status = DLQStatus.PENDING
            self.next_retry_at = self._calculate_next_retry()

    def mark_resolved(self, notes: Optional[str] = None):
        """Mark entry as successfully resolved."""
        self.status = DLQStatus.RESOLVED
        self.updated_at = datetime.utcnow()
        self.next_retry_at = None
        self.resolution_notes = notes or "Retry successful"

    def mark_manual_review(self, reason: str):
        """Mark entry as requiring manual review."""
        self.status = DLQStatus.MANUAL_REVIEW
        self.updated_at = datetime.utcnow()
        self.next_retry_at = None
        self.resolution_notes = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.next_retry_at:
            data['next_retry_at'] = self.next_retry_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeadLetterEntry':
        """Deserialize from dictionary."""
        data = data.copy()
        data['status'] = DLQStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('next_retry_at'):
            data['next_retry_at'] = datetime.fromisoformat(data['next_retry_at'])
        return cls(**data)


@dataclass
class RetryResult:
    """Result of a retry attempt."""
    entry_id: str
    success: bool
    error: Optional[str] = None
    data: Any = None


class DeadLetterQueueService:
    """
    Manages the dead-letter queue for failed extractions.

    Uses file-based storage for simplicity and portability.
    Can be extended to use PostgreSQL or Redis for production.
    """

    DEFAULT_DLQ_DIR = Path('/opt/airflow/data/dlq')
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_RETENTION_DAYS = 30

    def __init__(
        self,
        dlq_dir: Optional[Path] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ):
        """
        Initialize DLQ service.

        Args:
            dlq_dir: Directory for DLQ storage
            max_retries: Maximum retry attempts per entry
            retention_days: Days to keep resolved/failed entries
        """
        self.dlq_dir = dlq_dir or self._get_default_dlq_dir()
        self.max_retries = max_retries
        self.retention_days = retention_days

        # Ensure directory exists
        self.dlq_dir.mkdir(parents=True, exist_ok=True)

        # Initialize in-memory index for fast lookups
        self._entries: Dict[str, DeadLetterEntry] = {}
        self._load_entries()

    def _get_default_dlq_dir(self) -> Path:
        """Get default DLQ directory based on environment."""
        # Check for Airflow context
        if os.environ.get('AIRFLOW_HOME'):
            return Path(os.environ['AIRFLOW_HOME']) / 'data' / 'dlq'

        # Check for project root
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / 'data' / 'dlq'

    def _load_entries(self):
        """Load all DLQ entries from disk."""
        try:
            for entry_file in self.dlq_dir.glob('*.json'):
                try:
                    with open(entry_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        entry = DeadLetterEntry.from_dict(data)
                        self._entries[entry.id] = entry
                except Exception as e:
                    logger.warning("Failed to load DLQ entry %s: %s", entry_file, e)

            logger.info("Loaded %d DLQ entries from disk", len(self._entries))
        except Exception as e:
            logger.error("Failed to load DLQ entries: %s", e)

    def _save_entry(self, entry: DeadLetterEntry):
        """Save a single entry to disk."""
        entry_path = self.dlq_dir / f"{entry.id}.json"
        try:
            with open(entry_path, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save DLQ entry %s: %s", entry.id, e)

    def _generate_id(self, source: str, variable: str, error: str) -> str:
        """Generate unique ID for an entry."""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        hash_input = f"{source}:{variable}:{error}:{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def save_failed_extraction(
        self,
        source: str,
        variable: str,
        error: str,
        payload: Dict[str, Any],
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeadLetterEntry:
        """
        Save a failed extraction to the DLQ.

        Args:
            source: Data source (e.g., 'fred', 'investing')
            variable: Variable name that failed
            error: Error message
            payload: Original extraction payload for retry
            error_type: Type of error (e.g., 'ConnectionError')
            metadata: Additional context

        Returns:
            The created DeadLetterEntry
        """
        entry_id = self._generate_id(source, variable, error)
        now = datetime.utcnow()

        entry = DeadLetterEntry(
            id=entry_id,
            source=source,
            variable=variable,
            error_message=error,
            error_type=error_type or type(error).__name__,
            payload=payload,
            created_at=now,
            updated_at=now,
            max_retries=self.max_retries,
            metadata=metadata or {},
        )

        self._entries[entry_id] = entry
        self._save_entry(entry)

        logger.warning(
            "[DLQ] Saved failed extraction: source=%s, variable=%s, error=%s, id=%s",
            source, variable, error[:100], entry_id
        )

        return entry

    def get_pending_entries(self) -> List[DeadLetterEntry]:
        """Get all entries that are ready for retry."""
        return [
            entry for entry in self._entries.values()
            if entry.can_retry()
        ]

    def get_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """Get a specific entry by ID."""
        return self._entries.get(entry_id)

    def retry_dead_letters(
        self,
        retry_fn: Optional[Callable[[DeadLetterEntry], RetryResult]] = None,
        source_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[RetryResult]:
        """
        Retry all pending dead letters.

        Args:
            retry_fn: Function to call for each retry. If None, uses default extractor.
            source_filter: Only retry entries from this source
            limit: Maximum entries to retry in one batch

        Returns:
            List of RetryResult objects
        """
        pending = self.get_pending_entries()

        if source_filter:
            pending = [e for e in pending if e.source == source_filter]

        pending = pending[:limit]
        results = []

        logger.info("[DLQ] Starting retry of %d entries", len(pending))

        for entry in pending:
            entry.status = DLQStatus.RETRYING
            self._save_entry(entry)

            try:
                if retry_fn:
                    result = retry_fn(entry)
                else:
                    result = self._default_retry(entry)

                if result.success:
                    entry.mark_resolved(f"Retry successful: {result.data}")
                    logger.info("[DLQ] Entry %s resolved successfully", entry.id)
                else:
                    entry.increment_retry(result.error)
                    logger.warning(
                        "[DLQ] Entry %s retry failed (attempt %d/%d): %s",
                        entry.id, entry.retry_count, entry.max_retries, result.error
                    )

                results.append(result)

            except Exception as e:
                entry.increment_retry(str(e))
                results.append(RetryResult(
                    entry_id=entry.id,
                    success=False,
                    error=str(e),
                ))
                logger.error("[DLQ] Entry %s retry exception: %s", entry.id, e)

            finally:
                self._save_entry(entry)

        return results

    def _default_retry(self, entry: DeadLetterEntry) -> RetryResult:
        """Default retry implementation using ExtractorRegistry."""
        try:
            # Import here to avoid circular dependencies
            from extractors.registry import ExtractorRegistry

            registry = ExtractorRegistry()
            result = registry.extract_variable(
                entry.variable,
                **entry.payload
            )

            if result.success:
                return RetryResult(
                    entry_id=entry.id,
                    success=True,
                    data=f"Extracted {len(result.data) if result.data is not None else 0} records"
                )
            else:
                return RetryResult(
                    entry_id=entry.id,
                    success=False,
                    error=result.error or "Unknown extraction error"
                )

        except ImportError:
            return RetryResult(
                entry_id=entry.id,
                success=False,
                error="ExtractorRegistry not available"
            )
        except Exception as e:
            return RetryResult(
                entry_id=entry.id,
                success=False,
                error=str(e)
            )

    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics for monitoring."""
        total = len(self._entries)
        by_status = {}
        by_source = {}
        by_variable = {}
        by_error_type = {}

        for entry in self._entries.values():
            # By status
            status = entry.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # By source
            by_source[entry.source] = by_source.get(entry.source, 0) + 1

            # By variable
            by_variable[entry.variable] = by_variable.get(entry.variable, 0) + 1

            # By error type
            by_error_type[entry.error_type] = by_error_type.get(entry.error_type, 0) + 1

        # Calculate averages
        retry_counts = [e.retry_count for e in self._entries.values()]
        avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0

        return {
            'total_entries': total,
            'by_status': by_status,
            'by_source': by_source,
            'by_variable': dict(sorted(by_variable.items(), key=lambda x: -x[1])[:10]),
            'by_error_type': by_error_type,
            'avg_retry_count': round(avg_retries, 2),
            'pending_count': by_status.get('pending', 0),
            'failed_permanent_count': by_status.get('failed_permanent', 0),
            'resolved_count': by_status.get('resolved', 0),
        }

    def cleanup_old_entries(self) -> int:
        """
        Remove old resolved/failed entries beyond retention period.

        Returns:
            Number of entries removed
        """
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        removed = 0

        entries_to_remove = []
        for entry_id, entry in self._entries.items():
            if entry.status in [DLQStatus.RESOLVED, DLQStatus.FAILED_PERMANENT]:
                if entry.updated_at < cutoff:
                    entries_to_remove.append(entry_id)

        for entry_id in entries_to_remove:
            entry_path = self.dlq_dir / f"{entry_id}.json"
            try:
                if entry_path.exists():
                    entry_path.unlink()
                del self._entries[entry_id]
                removed += 1
            except Exception as e:
                logger.error("Failed to remove DLQ entry %s: %s", entry_id, e)

        if removed > 0:
            logger.info("[DLQ] Cleaned up %d old entries", removed)

        return removed

    def mark_for_manual_review(self, entry_id: str, reason: str) -> bool:
        """
        Mark an entry for manual review.

        Args:
            entry_id: The entry ID
            reason: Why manual review is needed

        Returns:
            True if successful
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return False

        entry.mark_manual_review(reason)
        self._save_entry(entry)
        logger.info("[DLQ] Entry %s marked for manual review: %s", entry_id, reason)
        return True

    def get_manual_review_queue(self) -> List[DeadLetterEntry]:
        """Get all entries requiring manual review."""
        return [
            entry for entry in self._entries.values()
            if entry.status == DLQStatus.MANUAL_REVIEW
        ]

    def resolve_manually(self, entry_id: str, notes: str) -> bool:
        """
        Manually resolve an entry.

        Args:
            entry_id: The entry ID
            notes: Resolution notes

        Returns:
            True if successful
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return False

        entry.mark_resolved(f"Manual resolution: {notes}")
        self._save_entry(entry)
        logger.info("[DLQ] Entry %s manually resolved: %s", entry_id, notes)
        return True


# Convenience singleton for DAG usage
_dlq_instance: Optional[DeadLetterQueueService] = None


def get_dlq_service() -> DeadLetterQueueService:
    """Get the DLQ service singleton."""
    global _dlq_instance
    if _dlq_instance is None:
        _dlq_instance = DeadLetterQueueService()
    return _dlq_instance
