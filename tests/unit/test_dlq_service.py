# -*- coding: utf-8 -*-
"""
Unit Tests for Dead-Letter-Queue Service
=========================================

Tests for:
- DeadLetterQueueService
- Entry lifecycle (create, retry, resolve)
- Exponential backoff
- Cleanup and retention

Contract: CTR-L0-DLQ-001
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
DAGS_PATH = PROJECT_ROOT / 'airflow' / 'dags'

for path in [str(DAGS_PATH), str(PROJECT_ROOT / 'src')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from services.dlq_service import (
    DeadLetterQueueService,
    DeadLetterEntry,
    DLQStatus,
    RetryResult,
    get_dlq_service,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dlq_dir():
    """Create a temporary directory for DLQ storage."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dlq_service(temp_dlq_dir):
    """Create a DLQ service with temporary storage."""
    return DeadLetterQueueService(
        dlq_dir=temp_dlq_dir,
        max_retries=3,
        retention_days=7,
    )


@pytest.fixture
def sample_entry():
    """Create a sample dead letter entry."""
    now = datetime.utcnow()
    return DeadLetterEntry(
        id='test-entry-001',
        source='fred',
        variable='FEDFUNDS',
        error_message='Connection timeout',
        error_type='TimeoutError',
        payload={'series_id': 'FEDFUNDS', 'start': '2024-01-01'},
        created_at=now,
        updated_at=now,
        retry_count=0,
        max_retries=3,
    )


# =============================================================================
# DEAD LETTER ENTRY TESTS
# =============================================================================

class TestDeadLetterEntry:
    """Tests for DeadLetterEntry dataclass."""

    def test_entry_creation(self, sample_entry):
        """Test basic entry creation."""
        assert sample_entry.id == 'test-entry-001'
        assert sample_entry.source == 'fred'
        assert sample_entry.variable == 'FEDFUNDS'
        assert sample_entry.status == DLQStatus.PENDING
        assert sample_entry.retry_count == 0

    def test_can_retry_initial(self, sample_entry):
        """Test that new entry can be retried."""
        assert sample_entry.can_retry() is True

    def test_can_retry_after_max(self, sample_entry):
        """Test that entry cannot be retried after max retries."""
        sample_entry.retry_count = 3
        sample_entry.status = DLQStatus.FAILED_PERMANENT
        assert sample_entry.can_retry() is False

    def test_can_retry_resolved(self, sample_entry):
        """Test that resolved entry cannot be retried."""
        sample_entry.status = DLQStatus.RESOLVED
        assert sample_entry.can_retry() is False

    def test_increment_retry(self, sample_entry):
        """Test retry count increment."""
        original_count = sample_entry.retry_count
        sample_entry.increment_retry('New error')

        assert sample_entry.retry_count == original_count + 1
        assert sample_entry.error_message == 'New error'
        assert sample_entry.next_retry_at is not None

    def test_increment_retry_max_reached(self, sample_entry):
        """Test that max retries marks entry as failed permanent."""
        sample_entry.retry_count = 2  # One below max
        sample_entry.increment_retry()

        assert sample_entry.status == DLQStatus.FAILED_PERMANENT
        assert sample_entry.next_retry_at is None

    def test_mark_resolved(self, sample_entry):
        """Test marking entry as resolved."""
        sample_entry.mark_resolved('Fixed by retry')

        assert sample_entry.status == DLQStatus.RESOLVED
        assert 'Fixed by retry' in sample_entry.resolution_notes
        assert sample_entry.next_retry_at is None

    def test_mark_manual_review(self, sample_entry):
        """Test marking entry for manual review."""
        sample_entry.mark_manual_review('API key expired')

        assert sample_entry.status == DLQStatus.MANUAL_REVIEW
        assert 'API key expired' in sample_entry.resolution_notes

    def test_exponential_backoff(self, sample_entry):
        """Test that backoff increases exponentially."""
        backoffs = []
        for i in range(5):
            sample_entry.retry_count = i
            backoff = sample_entry._calculate_next_retry()
            backoffs.append((backoff - datetime.utcnow()).total_seconds())

        # Each backoff should be roughly double the previous
        for i in range(1, len(backoffs)):
            assert backoffs[i] >= backoffs[i-1]

    def test_serialization(self, sample_entry):
        """Test entry serialization to dict."""
        data = sample_entry.to_dict()

        assert 'id' in data
        assert 'source' in data
        assert 'status' in data
        assert data['status'] == 'pending'  # String, not enum
        assert 'created_at' in data

    def test_deserialization(self, sample_entry):
        """Test entry deserialization from dict."""
        data = sample_entry.to_dict()
        restored = DeadLetterEntry.from_dict(data)

        assert restored.id == sample_entry.id
        assert restored.source == sample_entry.source
        assert restored.status == sample_entry.status


# =============================================================================
# DLQ SERVICE TESTS
# =============================================================================

class TestDeadLetterQueueService:
    """Tests for DeadLetterQueueService."""

    def test_service_initialization(self, dlq_service, temp_dlq_dir):
        """Test service initializes correctly."""
        assert dlq_service.dlq_dir == temp_dlq_dir
        assert dlq_service.max_retries == 3
        assert dlq_service.retention_days == 7

    def test_save_failed_extraction(self, dlq_service):
        """Test saving a failed extraction."""
        entry = dlq_service.save_failed_extraction(
            source='fred',
            variable='FEDFUNDS',
            error='Connection timeout',
            payload={'series_id': 'FEDFUNDS'},
        )

        assert entry.id is not None
        assert entry.source == 'fred'
        assert entry.variable == 'FEDFUNDS'
        assert entry.status == DLQStatus.PENDING

        # Check file was created
        entry_file = dlq_service.dlq_dir / f"{entry.id}.json"
        assert entry_file.exists()

    def test_get_pending_entries(self, dlq_service):
        """Test retrieving pending entries."""
        # Add some entries
        dlq_service.save_failed_extraction(
            source='fred', variable='VAR1', error='Error 1', payload={}
        )
        dlq_service.save_failed_extraction(
            source='investing', variable='VAR2', error='Error 2', payload={}
        )

        pending = dlq_service.get_pending_entries()

        assert len(pending) == 2
        assert all(e.status == DLQStatus.PENDING for e in pending)

    def test_get_entry_by_id(self, dlq_service):
        """Test retrieving a specific entry."""
        entry = dlq_service.save_failed_extraction(
            source='fred', variable='FEDFUNDS', error='Error', payload={}
        )

        retrieved = dlq_service.get_entry(entry.id)

        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.variable == entry.variable

    def test_retry_dead_letters_success(self, dlq_service):
        """Test successful retry of dead letters."""
        entry = dlq_service.save_failed_extraction(
            source='fred', variable='FEDFUNDS', error='Error', payload={}
        )

        # Mock successful retry
        def mock_retry(e):
            return RetryResult(entry_id=e.id, success=True, data='10 records')

        results = dlq_service.retry_dead_letters(retry_fn=mock_retry)

        assert len(results) == 1
        assert results[0].success is True

        # Entry should be resolved
        updated = dlq_service.get_entry(entry.id)
        assert updated.status == DLQStatus.RESOLVED

    def test_retry_dead_letters_failure(self, dlq_service):
        """Test failed retry of dead letters."""
        entry = dlq_service.save_failed_extraction(
            source='fred', variable='FEDFUNDS', error='Error', payload={}
        )

        # Mock failed retry
        def mock_retry(e):
            return RetryResult(entry_id=e.id, success=False, error='Still failing')

        results = dlq_service.retry_dead_letters(retry_fn=mock_retry)

        assert len(results) == 1
        assert results[0].success is False

        # Entry should still be pending with incremented retry
        updated = dlq_service.get_entry(entry.id)
        assert updated.status == DLQStatus.PENDING
        assert updated.retry_count == 1

    def test_retry_source_filter(self, dlq_service):
        """Test retry with source filter."""
        dlq_service.save_failed_extraction(
            source='fred', variable='VAR1', error='Error', payload={}
        )
        dlq_service.save_failed_extraction(
            source='investing', variable='VAR2', error='Error', payload={}
        )

        def mock_retry(e):
            return RetryResult(entry_id=e.id, success=True)

        results = dlq_service.retry_dead_letters(
            retry_fn=mock_retry,
            source_filter='fred',
        )

        assert len(results) == 1

    def test_get_failure_stats(self, dlq_service):
        """Test failure statistics."""
        dlq_service.save_failed_extraction(
            source='fred', variable='VAR1', error='Error', payload={},
            error_type='TimeoutError'
        )
        dlq_service.save_failed_extraction(
            source='fred', variable='VAR2', error='Error', payload={},
            error_type='ConnectionError'
        )
        dlq_service.save_failed_extraction(
            source='investing', variable='VAR3', error='Error', payload={},
            error_type='TimeoutError'
        )

        stats = dlq_service.get_failure_stats()

        assert stats['total_entries'] == 3
        assert stats['by_source']['fred'] == 2
        assert stats['by_source']['investing'] == 1
        assert stats['by_error_type']['TimeoutError'] == 2
        assert stats['pending_count'] == 3

    def test_cleanup_old_entries(self, dlq_service):
        """Test cleanup of old resolved entries."""
        # Create an old resolved entry
        entry = dlq_service.save_failed_extraction(
            source='fred', variable='OLD', error='Error', payload={}
        )
        entry.status = DLQStatus.RESOLVED
        entry.updated_at = datetime.utcnow() - timedelta(days=10)  # Older than retention
        dlq_service._save_entry(entry)

        # Create a recent entry
        recent = dlq_service.save_failed_extraction(
            source='fred', variable='RECENT', error='Error', payload={}
        )

        removed = dlq_service.cleanup_old_entries()

        assert removed == 1
        assert dlq_service.get_entry(entry.id) is None
        assert dlq_service.get_entry(recent.id) is not None

    def test_mark_for_manual_review(self, dlq_service):
        """Test marking entry for manual review."""
        entry = dlq_service.save_failed_extraction(
            source='fred', variable='FEDFUNDS', error='Error', payload={}
        )

        success = dlq_service.mark_for_manual_review(entry.id, 'API key expired')

        assert success is True
        updated = dlq_service.get_entry(entry.id)
        assert updated.status == DLQStatus.MANUAL_REVIEW

    def test_get_manual_review_queue(self, dlq_service):
        """Test getting manual review queue."""
        entry1 = dlq_service.save_failed_extraction(
            source='fred', variable='VAR1', error='Error', payload={}
        )
        entry2 = dlq_service.save_failed_extraction(
            source='fred', variable='VAR2', error='Error', payload={}
        )

        dlq_service.mark_for_manual_review(entry1.id, 'Needs attention')

        queue = dlq_service.get_manual_review_queue()

        assert len(queue) == 1
        assert queue[0].id == entry1.id

    def test_resolve_manually(self, dlq_service):
        """Test manual resolution."""
        entry = dlq_service.save_failed_extraction(
            source='fred', variable='FEDFUNDS', error='Error', payload={}
        )
        dlq_service.mark_for_manual_review(entry.id, 'Needs review')

        success = dlq_service.resolve_manually(entry.id, 'Fixed by updating API key')

        assert success is True
        updated = dlq_service.get_entry(entry.id)
        assert updated.status == DLQStatus.RESOLVED
        assert 'Fixed by updating API key' in updated.resolution_notes

    def test_persistence_across_instances(self, temp_dlq_dir):
        """Test that entries persist across service instances."""
        # Create entry with first instance
        service1 = DeadLetterQueueService(dlq_dir=temp_dlq_dir)
        entry = service1.save_failed_extraction(
            source='fred', variable='FEDFUNDS', error='Error', payload={}
        )
        entry_id = entry.id

        # Create new instance and verify entry exists
        service2 = DeadLetterQueueService(dlq_dir=temp_dlq_dir)
        retrieved = service2.get_entry(entry_id)

        assert retrieved is not None
        assert retrieved.id == entry_id
        assert retrieved.variable == 'FEDFUNDS'


# =============================================================================
# RETRY RESULT TESTS
# =============================================================================

class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_success_result(self):
        """Test successful retry result."""
        result = RetryResult(
            entry_id='test-123',
            success=True,
            data='Extracted 50 records',
        )

        assert result.success is True
        assert result.error is None
        assert 'records' in result.data

    def test_failure_result(self):
        """Test failed retry result."""
        result = RetryResult(
            entry_id='test-123',
            success=False,
            error='Connection refused',
        )

        assert result.success is False
        assert result.error is not None
        assert result.data is None


# =============================================================================
# DLQ STATUS TESTS
# =============================================================================

class TestDLQStatus:
    """Tests for DLQStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        assert DLQStatus.PENDING.value == 'pending'
        assert DLQStatus.RETRYING.value == 'retrying'
        assert DLQStatus.RESOLVED.value == 'resolved'
        assert DLQStatus.FAILED_PERMANENT.value == 'failed_permanent'
        assert DLQStatus.MANUAL_REVIEW.value == 'manual_review'

    def test_status_serialization(self):
        """Test status enum serialization."""
        assert DLQStatus('pending') == DLQStatus.PENDING
        assert DLQStatus('resolved') == DLQStatus.RESOLVED


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
