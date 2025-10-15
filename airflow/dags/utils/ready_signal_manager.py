"""
USDCOP Trading System - Ready Signal Coordination Manager
========================================================

Provides intelligent ready signal management for coordinating data processing
and WebSocket handover between pipeline stages.

COORDINATION FEATURES:
- Ready signal file management in /data/ready-signals/
- Metadata tracking for data freshness and processing status
- Integration with backup system for status coordination
- WebSocket handover status tracking
- Cross-pipeline synchronization support

SIGNAL TYPES:
- data_ready: Raw data available for processing
- processed_ready: Processed data ready for feature engineering
- features_ready: Features ready for RL training
- model_ready: Model ready for serving
- websocket_ready: WebSocket handover complete
"""

import os
import json
import fcntl
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .datetime_handler import UnifiedDatetimeHandler
from .backup_manager import BackupManager

logger = logging.getLogger(__name__)

class SignalStatus(Enum):
    """Enumeration of signal statuses"""
    PENDING = "pending"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class SignalType(Enum):
    """Enumeration of signal types"""
    DATA_READY = "data_ready"
    PROCESSED_READY = "processed_ready"
    FEATURES_READY = "features_ready"
    MODEL_READY = "model_ready"
    WEBSOCKET_READY = "websocket_ready"
    BACKUP_READY = "backup_ready"
    QUALITY_CHECK_READY = "quality_check_ready"

@dataclass
class ReadySignal:
    """Data class representing a ready signal"""
    signal_id: str
    signal_type: SignalType
    status: SignalStatus
    pipeline_run_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    data_metadata: Optional[Dict] = None
    processing_metadata: Optional[Dict] = None
    dependencies: Optional[List[str]] = None
    error_message: Optional[str] = None

class ReadySignalManager:
    """
    Manages ready signals for coordinating USDCOP trading pipeline stages.
    Provides thread-safe operations with file locking and status tracking.
    """

    def __init__(self,
                 signals_path: str = None,
                 default_expiry_minutes: int = 60,
                 cleanup_interval_minutes: int = 30,
                 backup_manager: BackupManager = None):
        """
        Initialize ready signal manager.

        Args:
            signals_path: Directory for signal files (default: /data/ready-signals/)
            default_expiry_minutes: Default signal expiry time in minutes
            cleanup_interval_minutes: Cleanup interval for expired signals
            backup_manager: BackupManager instance for integration
        """
        # Signal storage configuration
        self.signals_path = signals_path or self._get_default_signals_path()
        self.default_expiry_minutes = default_expiry_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes

        # Initialize storage
        self._setup_signal_storage()

        # Dependencies
        self.datetime_handler = UnifiedDatetimeHandler()
        self.backup_manager = backup_manager

        # Threading locks for thread-safe operations
        self._signal_locks = {}
        self._global_lock = threading.Lock()

        # Start cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()

        logger.info(f"ReadySignalManager initialized - Path: {self.signals_path}")

    def _get_default_signals_path(self) -> str:
        """Get default signals path with fallback options."""
        preferred_paths = ["/data/ready-signals", "/tmp/ready-signals", "./ready-signals"]

        for path in preferred_paths:
            try:
                os.makedirs(path, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(path, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"Using signals path: {path}")
                return path
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot use signals path {path}: {e}")
                continue

        raise RuntimeError("No suitable signals directory found")

    def _setup_signal_storage(self):
        """Setup signal storage directories."""
        try:
            os.makedirs(self.signals_path, exist_ok=True)

            # Create subdirectories for organization
            subdirs = ["active", "completed", "failed", "temp"]
            for subdir in subdirs:
                os.makedirs(os.path.join(self.signals_path, subdir), exist_ok=True)

            logger.info(f"✅ Signal storage initialized: {self.signals_path}")

        except Exception as e:
            logger.error(f"❌ Failed to setup signal storage: {e}")
            raise

    def create_ready_signal(self,
                           signal_type: Union[SignalType, str],
                           pipeline_run_id: str,
                           data_metadata: Optional[Dict] = None,
                           processing_metadata: Optional[Dict] = None,
                           dependencies: Optional[List[str]] = None,
                           expiry_minutes: Optional[int] = None) -> str:
        """
        Create a new ready signal.

        Args:
            signal_type: Type of signal to create
            pipeline_run_id: Associated pipeline run ID
            data_metadata: Metadata about the data (record count, date range, etc.)
            processing_metadata: Metadata about processing (duration, resources, etc.)
            dependencies: List of signal IDs this signal depends on
            expiry_minutes: Custom expiry time in minutes

        Returns:
            Signal ID of the created signal
        """
        try:
            # Convert string to enum if needed
            if isinstance(signal_type, str):
                signal_type = SignalType(signal_type)

            # Generate unique signal ID
            signal_id = f"{signal_type.value}_{pipeline_run_id}_{uuid.uuid4().hex[:8]}"

            # Calculate expiry time
            expiry_minutes = expiry_minutes or self.default_expiry_minutes
            expires_at = datetime.now() + timedelta(minutes=expiry_minutes)

            # Create signal object
            signal = ReadySignal(
                signal_id=signal_id,
                signal_type=signal_type,
                status=SignalStatus.READY,
                pipeline_run_id=pipeline_run_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                expires_at=expires_at,
                data_metadata=data_metadata or {},
                processing_metadata=processing_metadata or {},
                dependencies=dependencies or []
            )

            # Save signal to filesystem
            self._save_signal(signal)

            logger.info(f"✅ Created ready signal: {signal_id} (type: {signal_type.value})")
            return signal_id

        except Exception as e:
            logger.error(f"❌ Error creating ready signal: {e}")
            raise

    def check_ready_status(self, signal_id: str) -> Optional[Dict]:
        """
        Check the status of a ready signal.

        Args:
            signal_id: ID of the signal to check

        Returns:
            Signal status dictionary or None if not found
        """
        try:
            signal = self._load_signal(signal_id)
            if not signal:
                return None

            # Check if signal has expired
            if signal.expires_at and datetime.now() > signal.expires_at:
                if signal.status not in [SignalStatus.COMPLETED, SignalStatus.FAILED]:
                    signal.status = SignalStatus.EXPIRED
                    signal.updated_at = datetime.now()
                    self._save_signal(signal)

            # Check dependency status if signal is pending
            if signal.status == SignalStatus.PENDING and signal.dependencies:
                self._check_dependencies(signal)

            return {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "status": signal.status.value,
                "pipeline_run_id": signal.pipeline_run_id,
                "created_at": signal.created_at.isoformat(),
                "updated_at": signal.updated_at.isoformat(),
                "expires_at": signal.expires_at.isoformat() if signal.expires_at else None,
                "data_metadata": signal.data_metadata,
                "processing_metadata": signal.processing_metadata,
                "dependencies": signal.dependencies,
                "error_message": signal.error_message
            }

        except Exception as e:
            logger.error(f"❌ Error checking signal status: {e}")
            return None

    def update_signal_status(self,
                            signal_id: str,
                            status: Union[SignalStatus, str],
                            processing_metadata: Optional[Dict] = None,
                            error_message: Optional[str] = None) -> bool:
        """
        Update the status of a ready signal.

        Args:
            signal_id: ID of the signal to update
            status: New status for the signal
            processing_metadata: Updated processing metadata
            error_message: Error message if status is FAILED

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Convert string to enum if needed
            if isinstance(status, str):
                status = SignalStatus(status)

            signal = self._load_signal(signal_id)
            if not signal:
                logger.warning(f"Signal not found for update: {signal_id}")
                return False

            # Update signal
            signal.status = status
            signal.updated_at = datetime.now()

            if processing_metadata:
                signal.processing_metadata.update(processing_metadata)

            if error_message:
                signal.error_message = error_message

            # Save updated signal
            self._save_signal(signal)

            # Move signal to appropriate directory based on status
            self._organize_signal_by_status(signal)

            logger.info(f"✅ Updated signal {signal_id} status to {status.value}")
            return True

        except Exception as e:
            logger.error(f"❌ Error updating signal status: {e}")
            return False

    def update_last_processed(self,
                             signal_type: Union[SignalType, str],
                             pipeline_run_id: str,
                             last_processed_timestamp: datetime,
                             records_processed: int = 0) -> bool:
        """
        Update the last processed timestamp for a signal type.

        Args:
            signal_type: Type of signal
            pipeline_run_id: Pipeline run ID
            last_processed_timestamp: Last processed timestamp
            records_processed: Number of records processed

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Convert string to enum if needed
            if isinstance(signal_type, str):
                signal_type = SignalType(signal_type)

            # Find most recent signal of this type for the pipeline run
            signals = self.get_signals_by_type(signal_type, pipeline_run_id)
            if not signals:
                logger.warning(f"No signals found for type {signal_type.value}, run {pipeline_run_id}")
                return False

            # Update the most recent signal
            latest_signal = max(signals, key=lambda x: x['created_at'])
            signal_id = latest_signal['signal_id']

            processing_metadata = {
                "last_processed_timestamp": self.datetime_handler.ensure_timezone_aware(
                    last_processed_timestamp).isoformat(),
                "records_processed": records_processed,
                "processed_at": datetime.now().isoformat()
            }

            return self.update_signal_status(
                signal_id,
                SignalStatus.PROCESSING,
                processing_metadata=processing_metadata
            )

        except Exception as e:
            logger.error(f"❌ Error updating last processed: {e}")
            return False

    def get_signals_by_type(self,
                           signal_type: Union[SignalType, str],
                           pipeline_run_id: Optional[str] = None,
                           status_filter: Optional[Union[SignalStatus, str]] = None) -> List[Dict]:
        """
        Get all signals of a specific type.

        Args:
            signal_type: Type of signals to retrieve
            pipeline_run_id: Filter by pipeline run ID (optional)
            status_filter: Filter by status (optional)

        Returns:
            List of signal dictionaries
        """
        try:
            # Convert string to enum if needed
            if isinstance(signal_type, str):
                signal_type = SignalType(signal_type)
            if isinstance(status_filter, str):
                status_filter = SignalStatus(status_filter)

            signals = []

            # Search in all signal directories
            search_dirs = ["active", "completed", "failed"]
            for subdir in search_dirs:
                signals_dir = os.path.join(self.signals_path, subdir)
                if not os.path.exists(signals_dir):
                    continue

                for filename in os.listdir(signals_dir):
                    if filename.endswith('.json'):
                        signal = self._load_signal_from_file(
                            os.path.join(signals_dir, filename)
                        )
                        if not signal:
                            continue

                        # Apply filters
                        if signal.signal_type != signal_type:
                            continue

                        if pipeline_run_id and signal.pipeline_run_id != pipeline_run_id:
                            continue

                        if status_filter and signal.status != status_filter:
                            continue

                        # Convert to dictionary
                        signal_dict = self.check_ready_status(signal.signal_id)
                        if signal_dict:
                            signals.append(signal_dict)

            # Sort by creation time (newest first)
            signals.sort(key=lambda x: x['created_at'], reverse=True)

            logger.info(f"Found {len(signals)} signals of type {signal_type.value}")
            return signals

        except Exception as e:
            logger.error(f"❌ Error getting signals by type: {e}")
            return []

    def wait_for_signal(self,
                       signal_id: str,
                       timeout_seconds: int = 300,
                       check_interval_seconds: int = 5) -> Optional[Dict]:
        """
        Wait for a signal to become ready or completed.

        Args:
            signal_id: Signal ID to wait for
            timeout_seconds: Maximum wait time in seconds
            check_interval_seconds: Check interval in seconds

        Returns:
            Signal status dictionary when ready/completed, None on timeout
        """
        try:
            start_time = time.time()

            while time.time() - start_time < timeout_seconds:
                signal_status = self.check_ready_status(signal_id)
                if not signal_status:
                    logger.warning(f"Signal {signal_id} not found")
                    return None

                status = SignalStatus(signal_status['status'])

                if status in [SignalStatus.READY, SignalStatus.COMPLETED]:
                    logger.info(f"✅ Signal {signal_id} is {status.value}")
                    return signal_status

                if status in [SignalStatus.FAILED, SignalStatus.EXPIRED]:
                    logger.error(f"❌ Signal {signal_id} failed: {status.value}")
                    return signal_status

                # Wait before next check
                time.sleep(check_interval_seconds)

            logger.warning(f"⏰ Timeout waiting for signal {signal_id}")
            return None

        except Exception as e:
            logger.error(f"❌ Error waiting for signal: {e}")
            return None

    def cleanup_expired_signals(self) -> int:
        """
        Clean up expired signals.

        Returns:
            Number of signals cleaned up
        """
        try:
            cleaned_count = 0
            current_time = datetime.now()

            # Check all active signals
            active_dir = os.path.join(self.signals_path, "active")
            if os.path.exists(active_dir):
                for filename in os.listdir(active_dir):
                    if filename.endswith('.json'):
                        signal = self._load_signal_from_file(
                            os.path.join(active_dir, filename)
                        )
                        if not signal:
                            continue

                        # Check if expired
                        if signal.expires_at and current_time > signal.expires_at:
                            if signal.status not in [SignalStatus.COMPLETED, SignalStatus.FAILED]:
                                signal.status = SignalStatus.EXPIRED
                                signal.updated_at = current_time
                                self._save_signal(signal)
                                self._organize_signal_by_status(signal)
                                cleaned_count += 1
                                logger.info(f"🗑️  Expired signal: {signal.signal_id}")

            logger.info(f"✅ Cleanup completed: {cleaned_count} signals expired")
            return cleaned_count

        except Exception as e:
            logger.error(f"❌ Error during signal cleanup: {e}")
            return 0

    def get_websocket_handover_status(self, pipeline_run_id: str) -> Dict:
        """
        Get WebSocket handover status for a pipeline run.

        Args:
            pipeline_run_id: Pipeline run ID to check

        Returns:
            WebSocket handover status dictionary
        """
        try:
            # Get all WebSocket signals for this run
            websocket_signals = self.get_signals_by_type(
                SignalType.WEBSOCKET_READY,
                pipeline_run_id
            )

            status = {
                "pipeline_run_id": pipeline_run_id,
                "handover_complete": False,
                "handover_timestamp": None,
                "signals_count": len(websocket_signals),
                "latest_signal": None
            }

            if websocket_signals:
                # Get the latest signal
                latest_signal = websocket_signals[0]  # Already sorted by creation time
                status["latest_signal"] = latest_signal

                # Check if handover is complete
                signal_status = SignalStatus(latest_signal['status'])
                if signal_status == SignalStatus.COMPLETED:
                    status["handover_complete"] = True
                    status["handover_timestamp"] = latest_signal['updated_at']

            return status

        except Exception as e:
            logger.error(f"❌ Error getting WebSocket handover status: {e}")
            return {"error": str(e)}

    def integrate_with_backup(self, backup_name: str, pipeline_run_id: str) -> str:
        """
        Create a ready signal when backup is complete.

        Args:
            backup_name: Name of the completed backup
            pipeline_run_id: Associated pipeline run ID

        Returns:
            Signal ID of the backup ready signal
        """
        try:
            # Get backup metadata if backup manager is available
            data_metadata = {}
            if self.backup_manager:
                metadata = self.backup_manager.get_backup_metadata(backup_name)
                if metadata:
                    data_metadata = {
                        "backup_name": backup_name,
                        "backup_metadata": metadata
                    }

            # Create backup ready signal
            signal_id = self.create_ready_signal(
                signal_type=SignalType.BACKUP_READY,
                pipeline_run_id=pipeline_run_id,
                data_metadata=data_metadata,
                processing_metadata={
                    "backup_created_at": datetime.now().isoformat()
                }
            )

            logger.info(f"✅ Created backup ready signal: {signal_id} for backup: {backup_name}")
            return signal_id

        except Exception as e:
            logger.error(f"❌ Error integrating with backup: {e}")
            raise

    def _save_signal(self, signal: ReadySignal):
        """Save signal to filesystem with file locking."""
        signal_path = self._get_signal_path(signal)

        # Use file locking for thread safety
        with self._get_signal_lock(signal.signal_id):
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(signal_path), exist_ok=True)

                # Convert to dictionary and save
                signal_dict = asdict(signal)

                # Convert datetime objects to ISO format
                for key, value in signal_dict.items():
                    if isinstance(value, datetime):
                        signal_dict[key] = value.isoformat()
                    elif isinstance(value, (SignalType, SignalStatus)):
                        signal_dict[key] = value.value

                with open(signal_path, 'w') as f:
                    json.dump(signal_dict, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"❌ Error saving signal {signal.signal_id}: {e}")
                raise

    def _load_signal(self, signal_id: str) -> Optional[ReadySignal]:
        """Load signal from filesystem."""
        # Search in all possible locations
        search_dirs = ["active", "completed", "failed", "temp"]

        for subdir in search_dirs:
            signal_path = os.path.join(self.signals_path, subdir, f"{signal_id}.json")
            if os.path.exists(signal_path):
                return self._load_signal_from_file(signal_path)

        return None

    def _load_signal_from_file(self, file_path: str) -> Optional[ReadySignal]:
        """Load signal from a specific file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            for date_field in ['created_at', 'updated_at', 'expires_at']:
                if data.get(date_field):
                    data[date_field] = datetime.fromisoformat(data[date_field])

            # Convert enum strings back to enums
            data['signal_type'] = SignalType(data['signal_type'])
            data['status'] = SignalStatus(data['status'])

            return ReadySignal(**data)

        except Exception as e:
            logger.error(f"❌ Error loading signal from {file_path}: {e}")
            return None

    def _get_signal_path(self, signal: ReadySignal) -> str:
        """Get filesystem path for a signal based on its status."""
        if signal.status in [SignalStatus.COMPLETED]:
            subdir = "completed"
        elif signal.status in [SignalStatus.FAILED, SignalStatus.EXPIRED]:
            subdir = "failed"
        else:
            subdir = "active"

        return os.path.join(self.signals_path, subdir, f"{signal.signal_id}.json")

    def _organize_signal_by_status(self, signal: ReadySignal):
        """Move signal file to appropriate directory based on status."""
        try:
            current_path = None
            new_path = self._get_signal_path(signal)

            # Find current file location
            search_dirs = ["active", "completed", "failed", "temp"]
            for subdir in search_dirs:
                potential_path = os.path.join(self.signals_path, subdir, f"{signal.signal_id}.json")
                if os.path.exists(potential_path):
                    current_path = potential_path
                    break

            # Move file if needed
            if current_path and current_path != new_path:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(current_path, new_path)

        except Exception as e:
            logger.error(f"❌ Error organizing signal {signal.signal_id}: {e}")

    def _get_signal_lock(self, signal_id: str) -> threading.Lock:
        """Get or create a lock for a specific signal."""
        with self._global_lock:
            if signal_id not in self._signal_locks:
                self._signal_locks[signal_id] = threading.Lock()
            return self._signal_locks[signal_id]

    def _check_dependencies(self, signal: ReadySignal):
        """Check if signal dependencies are satisfied."""
        if not signal.dependencies:
            return

        all_ready = True
        for dep_signal_id in signal.dependencies:
            dep_status = self.check_ready_status(dep_signal_id)
            if not dep_status or dep_status['status'] not in ['ready', 'completed']:
                all_ready = False
                break

        if all_ready:
            signal.status = SignalStatus.READY
            signal.updated_at = datetime.now()
            self._save_signal(signal)

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval_minutes * 60):
                try:
                    self.cleanup_expired_signals()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def shutdown(self):
        """Shutdown the signal manager and cleanup threads."""
        try:
            self._stop_cleanup.set()
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)

            logger.info("✅ ReadySignalManager shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# Convenience functions
def get_ready_signal_manager(backup_manager: BackupManager = None) -> ReadySignalManager:
    """Get a ready signal manager instance with standard configuration."""
    return ReadySignalManager(backup_manager=backup_manager)


# Example usage and testing
if __name__ == "__main__":
    # Test ready signal functionality
    try:
        signal_mgr = ReadySignalManager()

        # Create test signal
        signal_id = signal_mgr.create_ready_signal(
            signal_type=SignalType.DATA_READY,
            pipeline_run_id="test_run_001",
            data_metadata={
                "record_count": 1000,
                "date_range": ["2024-01-15T08:00:00", "2024-01-15T14:00:00"]
            },
            processing_metadata={
                "processing_duration_seconds": 45.2,
                "memory_usage_mb": 128.5
            }
        )

        print(f"Created signal: {signal_id}")

        # Check signal status
        status = signal_mgr.check_ready_status(signal_id)
        print(f"Signal status: {status}")

        # Update signal status
        success = signal_mgr.update_signal_status(
            signal_id,
            SignalStatus.PROCESSING,
            processing_metadata={"started_processing_at": datetime.now().isoformat()}
        )
        print(f"Status update success: {success}")

        # List signals
        signals = signal_mgr.get_signals_by_type(SignalType.DATA_READY)
        print(f"Found {len(signals)} data ready signals")

        # Test WebSocket handover status
        handover_status = signal_mgr.get_websocket_handover_status("test_run_001")
        print(f"WebSocket handover status: {handover_status}")

        print("✅ ReadySignalManager test completed successfully!")

    except Exception as e:
        print(f"❌ ReadySignalManager test failed: {e}")
    finally:
        signal_mgr.shutdown()