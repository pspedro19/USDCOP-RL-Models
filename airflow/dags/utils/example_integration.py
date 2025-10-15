#!/usr/bin/env python3
"""
USDCOP Trading System - Backup Management Integration Example
===========================================================

This example demonstrates how to use the three new backup management utilities
together for comprehensive data management and coordination:

1. BackupManager - For data backup and recovery
2. ReadySignalManager - For pipeline coordination
3. GapDetector - For data quality and gap analysis

INTEGRATION WORKFLOW:
1. Detect gaps in existing data
2. Plan incremental updates to fill gaps
3. Coordinate data processing with ready signals
4. Create backups with comprehensive metadata
5. Validate backup integrity and completeness
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import the new utilities
from backup_manager import BackupManager
from ready_signal_manager import ReadySignalManager, SignalType, SignalStatus
from gap_detector import GapDetector
from db_manager import DatabaseManager
from datetime_handler import UnifiedDatetimeHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedBackupManager:
    """
    Integrated backup management system that coordinates all three utilities
    for comprehensive data management.
    """

    def __init__(self, enable_s3: bool = False, s3_bucket: str = None):
        """
        Initialize integrated backup manager.

        Args:
            enable_s3: Whether to enable S3 storage
            s3_bucket: S3 bucket name for remote storage
        """
        # Initialize utilities
        self.db_manager = DatabaseManager()
        self.datetime_handler = UnifiedDatetimeHandler()
        self.backup_manager = BackupManager(enable_s3=enable_s3, s3_bucket=s3_bucket)
        self.signal_manager = ReadySignalManager(backup_manager=self.backup_manager)
        self.gap_detector = GapDetector(db_manager=self.db_manager)

        logger.info("✅ Integrated backup manager initialized")

    def perform_comprehensive_backup_workflow(self,
                                            data_source: str = "database",
                                            pipeline_run_id: str = None,
                                            start_date: datetime = None,
                                            end_date: datetime = None) -> Dict[str, Any]:
        """
        Perform a comprehensive backup workflow with gap detection and coordination.

        Args:
            data_source: Source of data ("database", "api", "file")
            pipeline_run_id: Pipeline run identifier
            start_date: Start date for data analysis
            end_date: End date for data analysis

        Returns:
            Workflow results dictionary
        """
        try:
            workflow_start = datetime.now()
            pipeline_run_id = pipeline_run_id or f"backup_workflow_{workflow_start.strftime('%Y%m%d_%H%M%S')}"

            logger.info(f"🚀 Starting comprehensive backup workflow: {pipeline_run_id}")

            # Create initial ready signal
            initial_signal_id = self.signal_manager.create_ready_signal(
                signal_type=SignalType.DATA_READY,
                pipeline_run_id=pipeline_run_id,
                data_metadata={"workflow_type": "comprehensive_backup"},
                processing_metadata={"started_at": workflow_start.isoformat()}
            )

            workflow_results = {
                "workflow_id": pipeline_run_id,
                "started_at": workflow_start.isoformat(),
                "initial_signal_id": initial_signal_id,
                "steps": {}
            }

            # Step 1: Analyze existing data and detect gaps
            logger.info("📊 Step 1: Analyzing existing data and detecting gaps...")
            gap_analysis = self._analyze_data_gaps(start_date, end_date, pipeline_run_id)
            workflow_results["steps"]["gap_analysis"] = gap_analysis

            # Step 2: Plan incremental updates if gaps found
            if gap_analysis["gaps_detected"] > 0:
                logger.info("🔄 Step 2: Planning incremental updates for gap filling...")
                update_plan = self._plan_incremental_updates(gap_analysis, pipeline_run_id)
                workflow_results["steps"]["update_plan"] = update_plan

                # Create ready signal for gap filling
                gap_fill_signal_id = self.signal_manager.create_ready_signal(
                    signal_type=SignalType.PROCESSED_READY,
                    pipeline_run_id=pipeline_run_id,
                    data_metadata={"gaps_to_fill": gap_analysis["gaps_detected"]},
                    dependencies=[initial_signal_id]
                )
                workflow_results["gap_fill_signal_id"] = gap_fill_signal_id

            # Step 3: Load or simulate data for backup
            logger.info("💾 Step 3: Loading data for backup...")
            backup_data = self._load_backup_data(data_source, start_date, end_date)

            if backup_data.empty:
                logger.warning("No data available for backup")
                workflow_results["steps"]["backup"] = {"status": "skipped", "reason": "no_data"}
            else:
                # Step 4: Create comprehensive backup
                logger.info("🗄️ Step 4: Creating comprehensive backup...")
                backup_result = self._create_comprehensive_backup(backup_data, pipeline_run_id)
                workflow_results["steps"]["backup"] = backup_result

                # Step 5: Validate backup integrity
                logger.info("✅ Step 5: Validating backup integrity...")
                validation_result = self._validate_backup_integrity(backup_result, pipeline_run_id)
                workflow_results["steps"]["validation"] = validation_result

                # Create backup ready signal
                backup_signal_id = self.signal_manager.integrate_with_backup(
                    backup_result["backup_name"], pipeline_run_id
                )
                workflow_results["backup_signal_id"] = backup_signal_id

            # Step 6: Finalize workflow
            logger.info("🏁 Step 6: Finalizing workflow...")
            workflow_results = self._finalize_workflow(workflow_results, pipeline_run_id)

            logger.info(f"✅ Comprehensive backup workflow completed: {pipeline_run_id}")
            return workflow_results

        except Exception as e:
            logger.error(f"❌ Error in comprehensive backup workflow: {e}")
            # Update signal to failed status
            if initial_signal_id:
                self.signal_manager.update_signal_status(
                    initial_signal_id, SignalStatus.FAILED, error_message=str(e)
                )
            return {"error": str(e), "workflow_id": pipeline_run_id}

    def _analyze_data_gaps(self, start_date: datetime, end_date: datetime, run_id: str) -> Dict[str, Any]:
        """Analyze data gaps in the specified period."""
        try:
            # Use default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            if end_date is None:
                end_date = datetime.now()

            # Get existing data from database
            existing_data = self.db_manager.get_latest_market_data(limit=10000)

            if existing_data.empty:
                logger.warning("No existing data found for gap analysis")
                return {
                    "gaps_detected": 0,
                    "analysis_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                    "message": "No existing data found"
                }

            # Detect gaps
            timestamp_col = "datetime" if "datetime" in existing_data.columns else "timestamp"
            gaps = self.gap_detector.detect_gaps(
                existing_data[timestamp_col], start_date, end_date, business_hours_only=True
            )

            # Analyze gap patterns
            gap_analysis = self.gap_detector.analyze_gap_patterns(gaps)

            # Validate data completeness
            completeness = self.gap_detector.validate_data_completeness(
                existing_data, start_date, end_date, business_hours_only=True
            )

            return {
                "gaps_detected": len(gaps),
                "analysis_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "gap_patterns": gap_analysis,
                "completeness": completeness,
                "gaps": [
                    {
                        "gap_id": gap.gap_id,
                        "start_time": gap.start_time.isoformat(),
                        "end_time": gap.end_time.isoformat(),
                        "severity": gap.severity.value,
                        "missing_points": gap.missing_points
                    } for gap in gaps
                ]
            }

        except Exception as e:
            logger.error(f"Error analyzing data gaps: {e}")
            return {"error": str(e)}

    def _plan_incremental_updates(self, gap_analysis: Dict, run_id: str) -> Dict[str, Any]:
        """Plan incremental updates to fill detected gaps."""
        try:
            if gap_analysis.get("gaps_detected", 0) == 0:
                return {"status": "no_gaps", "message": "No gaps to fill"}

            # Calculate missing periods from gaps
            missing_periods = []
            for gap in gap_analysis.get("gaps", []):
                start_time = datetime.fromisoformat(gap["start_time"])
                end_time = datetime.fromisoformat(gap["end_time"])
                missing_periods.append((start_time, end_time))

            # Generate incremental ranges
            incremental_ranges = self.gap_detector.get_incremental_ranges(
                missing_periods, max_points_per_call=1000, priority_business_hours=True
            )

            return {
                "status": "planned",
                "missing_periods": len(missing_periods),
                "incremental_ranges": len(incremental_ranges),
                "total_expected_points": sum(r.expected_points for r in incremental_ranges),
                "estimated_duration_minutes": sum(r.estimated_duration_minutes for r in incremental_ranges),
                "ranges": [
                    {
                        "range_id": r.range_id,
                        "start_time": r.start_time.isoformat(),
                        "end_time": r.end_time.isoformat(),
                        "expected_points": r.expected_points,
                        "priority": r.priority
                    } for r in incremental_ranges
                ]
            }

        except Exception as e:
            logger.error(f"Error planning incremental updates: {e}")
            return {"error": str(e)}

    def _load_backup_data(self, data_source: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load data for backup from specified source."""
        try:
            if data_source == "database":
                # Load from database
                data = self.db_manager.get_latest_market_data(limit=1000)
                if not data.empty and start_date and end_date:
                    # Filter by date range if available
                    timestamp_col = "datetime" if "datetime" in data.columns else "timestamp"
                    if timestamp_col in data.columns:
                        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                        data = data[
                            (data[timestamp_col] >= start_date) &
                            (data[timestamp_col] <= end_date)
                        ]
                return data

            elif data_source == "simulation":
                # Create simulated data for demonstration
                return self._create_simulated_data(start_date, end_date)

            else:
                logger.warning(f"Unsupported data source: {data_source}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading backup data: {e}")
            return pd.DataFrame()

    def _create_simulated_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create simulated market data for demonstration."""
        try:
            # Use default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(hours=6)
            if end_date is None:
                end_date = datetime.now()

            # Generate timestamps for business hours only
            timestamps = self.datetime_handler.generate_expected_timestamps(
                start_date, end_date, interval_minutes=5,
                business_hours_only=True, exclude_holidays=True
            )

            if not timestamps:
                return pd.DataFrame()

            # Create simulated price data
            base_price = 4200.0
            data = []

            for i, timestamp in enumerate(timestamps):
                # Simple price simulation with random walk
                price_change = (i * 0.1) + (i % 10 - 5) * 0.05
                current_price = base_price + price_change

                data.append({
                    'timestamp': timestamp,
                    'open': current_price - 0.1,
                    'high': current_price + 0.2,
                    'low': current_price - 0.2,
                    'close': current_price,
                    'volume': 1000 + (i % 100) * 10
                })

            df = pd.DataFrame(data)
            logger.info(f"Created simulated data: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error creating simulated data: {e}")
            return pd.DataFrame()

    def _create_comprehensive_backup(self, data: pd.DataFrame, run_id: str) -> Dict[str, Any]:
        """Create a comprehensive backup with metadata."""
        try:
            backup_name = f"comprehensive_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Calculate date range
            timestamp_col = "datetime" if "datetime" in data.columns else "timestamp"
            if timestamp_col in data.columns:
                timestamps = pd.to_datetime(data[timestamp_col])
                date_range = (timestamps.min(), timestamps.max())
            else:
                date_range = None

            # Create backup
            save_results = self.backup_manager.save_backup(
                df=data,
                backup_name=backup_name,
                description=f"Comprehensive backup for run {run_id}",
                pipeline_run_id=run_id,
                date_range=date_range,
                storage_types=["local"]  # Add "s3" if S3 is enabled
            )

            # Get backup metadata
            metadata = self.backup_manager.get_backup_metadata(backup_name)

            return {
                "backup_name": backup_name,
                "save_results": save_results,
                "metadata": metadata,
                "records_backed_up": len(data),
                "date_range": {
                    "start": date_range[0].isoformat() if date_range else None,
                    "end": date_range[1].isoformat() if date_range else None
                }
            }

        except Exception as e:
            logger.error(f"Error creating comprehensive backup: {e}")
            return {"error": str(e)}

    def _validate_backup_integrity(self, backup_result: Dict, run_id: str) -> Dict[str, Any]:
        """Validate backup integrity and completeness."""
        try:
            backup_name = backup_result.get("backup_name")
            if not backup_name:
                return {"error": "No backup name provided"}

            # Validate backup integrity
            validation_results = self.backup_manager.validate_backup_integrity(backup_name)

            # Load backup to verify it can be restored
            restored_data = self.backup_manager.load_backup(backup_name)
            restoration_success = restored_data is not None and not restored_data.empty

            return {
                "backup_name": backup_name,
                "validation_results": validation_results,
                "restoration_test": {
                    "success": restoration_success,
                    "restored_records": len(restored_data) if restored_data is not None else 0
                }
            }

        except Exception as e:
            logger.error(f"Error validating backup integrity: {e}")
            return {"error": str(e)}

    def _finalize_workflow(self, workflow_results: Dict, run_id: str) -> Dict[str, Any]:
        """Finalize the workflow and update signals."""
        try:
            workflow_end = datetime.now()
            workflow_duration = (workflow_end - datetime.fromisoformat(workflow_results["started_at"])).total_seconds()

            # Update initial signal to completed
            if "initial_signal_id" in workflow_results:
                self.signal_manager.update_signal_status(
                    workflow_results["initial_signal_id"],
                    SignalStatus.COMPLETED,
                    processing_metadata={
                        "completed_at": workflow_end.isoformat(),
                        "duration_seconds": workflow_duration
                    }
                )

            # Add workflow summary
            workflow_results.update({
                "completed_at": workflow_end.isoformat(),
                "duration_seconds": workflow_duration,
                "status": "completed",
                "summary": {
                    "gaps_detected": workflow_results.get("steps", {}).get("gap_analysis", {}).get("gaps_detected", 0),
                    "backup_created": "backup" in workflow_results.get("steps", {}),
                    "validation_passed": workflow_results.get("steps", {}).get("validation", {}).get("validation_results", {}).get("is_valid", False)
                }
            })

            return workflow_results

        except Exception as e:
            logger.error(f"Error finalizing workflow: {e}")
            workflow_results["finalization_error"] = str(e)
            return workflow_results

    def get_backup_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the backup system."""
        try:
            # Get backup manager status
            local_backups = self.backup_manager.list_backups("local")

            # Get signal manager status
            recent_signals = self.signal_manager.get_signals_by_type(SignalType.BACKUP_READY)

            # Get database statistics
            db_stats = self.db_manager.get_market_data_stats()

            return {
                "backup_storage": {
                    "local_backups": len(local_backups),
                    "latest_backup": local_backups[0]["metadata"].get("created_at") if local_backups else None
                },
                "signal_coordination": {
                    "recent_backup_signals": len(recent_signals),
                    "latest_signal": recent_signals[0]["created_at"] if recent_signals else None
                },
                "database_status": db_stats,
                "system_health": "operational"
            }

        except Exception as e:
            logger.error(f"Error getting backup system status: {e}")
            return {"error": str(e)}

    def cleanup_old_backups_and_signals(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old backups and signals."""
        try:
            cleanup_results = {}

            # Clean up old backups
            backup_cleanup = self.backup_manager.cleanup_old_backups(max_age_days=max_age_days)
            cleanup_results["backups_cleaned"] = backup_cleanup

            # Clean up expired signals
            signal_cleanup = self.signal_manager.cleanup_expired_signals()
            cleanup_results["signals_cleaned"] = signal_cleanup

            # Clean up database records
            db_cleanup = self.db_manager.cleanup_old_data()
            cleanup_results["database_cleanup"] = db_cleanup

            logger.info(f"✅ Cleanup completed: {cleanup_results}")
            return cleanup_results

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}

    def shutdown(self):
        """Shutdown all components gracefully."""
        try:
            self.signal_manager.shutdown()
            self.db_manager.close()
            logger.info("✅ Integrated backup manager shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Example usage and demonstration
def main():
    """Demonstrate the integrated backup management system."""
    try:
        logger.info("🚀 Starting integrated backup management demonstration...")

        # Initialize integrated backup manager
        integrated_manager = IntegratedBackupManager(enable_s3=False)

        # Example 1: Comprehensive backup workflow
        logger.info("\n" + "="*60)
        logger.info("EXAMPLE 1: Comprehensive Backup Workflow")
        logger.info("="*60)

        workflow_results = integrated_manager.perform_comprehensive_backup_workflow(
            data_source="simulation",
            start_date=datetime.now() - timedelta(hours=6),
            end_date=datetime.now()
        )

        logger.info(f"Workflow results: {workflow_results.get('status', 'unknown')}")

        # Example 2: System status check
        logger.info("\n" + "="*60)
        logger.info("EXAMPLE 2: System Status Check")
        logger.info("="*60)

        system_status = integrated_manager.get_backup_system_status()
        logger.info(f"System status: {system_status.get('system_health', 'unknown')}")

        # Example 3: Cleanup demonstration
        logger.info("\n" + "="*60)
        logger.info("EXAMPLE 3: Cleanup Operations")
        logger.info("="*60)

        cleanup_results = integrated_manager.cleanup_old_backups_and_signals(max_age_days=1)
        logger.info(f"Cleanup results: {cleanup_results}")

        logger.info("\n✅ All examples completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
    finally:
        if 'integrated_manager' in locals():
            integrated_manager.shutdown()


if __name__ == "__main__":
    main()