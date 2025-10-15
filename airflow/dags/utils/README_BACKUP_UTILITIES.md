# USDCOP Trading System - Backup Management Utilities

## Overview

This document describes the comprehensive backup management utilities created for the USDCOP trading system. These utilities provide intelligent backup management, ready signal coordination, and gap detection for incremental updates.

## Created Utilities

### 1. BackupManager (`backup_manager.py`)
**Main backup management class with comprehensive data storage and validation capabilities.**

#### Key Features:
- **Dual Storage Support**: Local filesystem (`/data/backups/`) and optional S3 storage
- **Metadata Tracking**: Comprehensive metadata for last_updated, record_count, date_range, completeness
- **Compression & Optimization**: Parquet format with snappy compression for optimal storage
- **Integrity Validation**: Data hash verification and completeness scoring
- **Backup Organization**: Automatic directory structure with data, metadata, temp, and incremental folders

#### Core Methods:
- `check_backup_exists(backup_name, storage_type)` - Check if backup exists
- `load_backup(backup_name, storage_type)` - Load backup data
- `save_backup(df, backup_name, description, ...)` - Save with comprehensive metadata
- `get_backup_metadata(backup_name, storage_type)` - Retrieve backup metadata
- `validate_backup_integrity(backup_name, storage_type)` - Validate backup integrity
- `list_backups(storage_type)` - List all available backups
- `cleanup_old_backups(max_age_days, max_count, storage_type)` - Cleanup old backups

#### Example Usage:
```python
from utils.backup_manager import BackupManager

# Initialize backup manager
backup_mgr = BackupManager(enable_s3=False)

# Save backup with metadata
results = backup_mgr.save_backup(
    df=market_data,
    backup_name="usdcop_data_20240115",
    description="Market data for analysis",
    pipeline_run_id="run_001"
)

# Load backup
data = backup_mgr.load_backup("usdcop_data_20240115")

# Validate integrity
validation = backup_mgr.validate_backup_integrity("usdcop_data_20240115")
```

### 2. ReadySignalManager (`ready_signal_manager.py`)
**Ready signal coordination system for WebSocket handover and pipeline synchronization.**

#### Key Features:
- **Signal File Management**: Organized signals in `/data/ready-signals/` with active/completed/failed directories
- **Metadata Tracking**: Data freshness, processing status, and WebSocket handover coordination
- **Dependency Management**: Signal dependencies and sequential processing
- **Thread-Safe Operations**: File locking and concurrent access protection
- **Automatic Cleanup**: Background thread for expired signal cleanup

#### Signal Types:
- `DATA_READY` - Raw data available for processing
- `PROCESSED_READY` - Processed data ready for feature engineering
- `FEATURES_READY` - Features ready for RL training
- `MODEL_READY` - Model ready for serving
- `WEBSOCKET_READY` - WebSocket handover complete
- `BACKUP_READY` - Backup operation complete

#### Core Methods:
- `create_ready_signal(signal_type, pipeline_run_id, ...)` - Create new ready signal
- `check_ready_status(signal_id)` - Check signal status
- `update_signal_status(signal_id, status, ...)` - Update signal status
- `update_last_processed(signal_type, pipeline_run_id, ...)` - Update processing timestamp
- `get_signals_by_type(signal_type, pipeline_run_id, ...)` - Query signals by type
- `wait_for_signal(signal_id, timeout_seconds, ...)` - Wait for signal completion
- `get_websocket_handover_status(pipeline_run_id)` - Get WebSocket handover status
- `integrate_with_backup(backup_name, pipeline_run_id)` - Create backup ready signal

#### Example Usage:
```python
from utils.ready_signal_manager import ReadySignalManager, SignalType, SignalStatus

# Initialize signal manager
signal_mgr = ReadySignalManager()

# Create ready signal
signal_id = signal_mgr.create_ready_signal(
    signal_type=SignalType.DATA_READY,
    pipeline_run_id="run_001",
    data_metadata={"record_count": 1000}
)

# Check status
status = signal_mgr.check_ready_status(signal_id)

# Update status
signal_mgr.update_signal_status(signal_id, SignalStatus.PROCESSING)
```

### 3. GapDetector (`gap_detector.py`)
**Intelligent gap detection for incremental updates with business hours awareness.**

#### Key Features:
- **Business Hours Awareness**: Colombian market hours (8:00 AM - 12:55 PM COT, Monday-Friday)
- **5-Minute Interval Detection**: Precise gap detection for 5-minute forex data
- **Holiday Handling**: Colombian market holiday exclusion
- **Incremental Range Calculation**: Optimal API call batching for gap filling
- **Gap Classification**: Severity assessment and gap type categorization

#### Business Rules:
- Colombian market hours: 8:00 AM - 12:55 PM COT (Monday-Friday)
- Expected intervals: 5 minutes
- Colombian holidays excluded from gap detection
- Weekends excluded from business hour calculations

#### Core Methods:
- `detect_gaps(timestamps, start_time, end_time, ...)` - Detect gaps in timestamp data
- `calculate_missing_periods(start_time, end_time, existing_data, ...)` - Calculate missing periods
- `get_incremental_ranges(missing_periods, max_points_per_call, ...)` - Generate optimized ranges
- `analyze_gap_patterns(gaps, lookback_days)` - Analyze gap patterns for insights
- `validate_data_completeness(df, start_time, end_time, ...)` - Validate data completeness

#### Gap Severity Levels:
- **LOW**: 15 minutes or less
- **MEDIUM**: 1 hour or less
- **HIGH**: 4 hours or less
- **CRITICAL**: More than 4 hours

#### Example Usage:
```python
from utils.gap_detector import GapDetector

# Initialize gap detector
gap_detector = GapDetector(interval_minutes=5)

# Detect gaps
gaps = gap_detector.detect_gaps(
    timestamps=data['timestamp'],
    start_time=start_date,
    end_time=end_date,
    business_hours_only=True
)

# Calculate missing periods
missing_periods = gap_detector.calculate_missing_periods(
    start_time=start_date,
    end_time=end_date,
    existing_data=current_data
)

# Get incremental update ranges
ranges = gap_detector.get_incremental_ranges(
    missing_periods=missing_periods,
    max_points_per_call=1000
)
```

## Integration Example

The utilities are designed to work together seamlessly. See `example_integration.py` for a comprehensive demonstration:

```python
from utils import BackupManager, ReadySignalManager, GapDetector

# Initialize all utilities
backup_mgr = BackupManager()
signal_mgr = ReadySignalManager(backup_manager=backup_mgr)
gap_detector = GapDetector()

# Comprehensive workflow
workflow_results = perform_comprehensive_backup_workflow(
    data_source="database",
    pipeline_run_id="workflow_001",
    start_date=datetime.now() - timedelta(days=1),
    end_date=datetime.now()
)
```

## Directory Structure

```
/data/
├── backups/
│   ├── data/           # Backup parquet files
│   ├── metadata/       # Backup metadata JSON files
│   ├── temp/           # Temporary processing files
│   └── incremental/    # Incremental backup files
└── ready-signals/
    ├── active/         # Active signals
    ├── completed/      # Completed signals
    ├── failed/         # Failed signals
    └── temp/           # Temporary signal files
```

## Dependencies Integration

### Database Manager Integration
- Ready signals stored in PostgreSQL `ready_signals` table
- Backup metadata tracked in `backup_metadata` table
- Data gaps recorded in `data_gaps` table

### Datetime Handler Integration
- All utilities use `UnifiedDatetimeHandler` for timezone consistency
- Business hours calculated in Colombian Time (COT)
- Holiday awareness for Colombian market

### Existing Pipeline Integration
- Compatible with current L0-L6 pipeline structure
- Supports existing logging and monitoring systems
- Integrates with current database schema

## Configuration Options

### Backup Manager Configuration
```python
BackupManager(
    local_backup_path="/data/backups",
    s3_bucket="usdcop-backups",
    s3_prefix="trading-data",
    compression_level=6,
    enable_s3=False
)
```

### Ready Signal Manager Configuration
```python
ReadySignalManager(
    signals_path="/data/ready-signals",
    default_expiry_minutes=60,
    cleanup_interval_minutes=30
)
```

### Gap Detector Configuration
```python
GapDetector(
    interval_minutes=5,
    tolerance_minutes=2,
    max_api_calls_per_batch=100
)
```

## Error Handling and Logging

All utilities include comprehensive error handling and logging:

- **Graceful Degradation**: Fallback paths for permission issues
- **Detailed Logging**: Executive-style logging with clear metrics
- **Error Recovery**: Automatic retry mechanisms where appropriate
- **Status Tracking**: Comprehensive status reporting for monitoring

## Security Considerations

- **File Permissions**: Proper file permission handling with fallback directories
- **Data Validation**: Input validation and sanitization
- **Access Control**: Thread-safe operations with proper locking
- **Credential Management**: Environment variable based configuration

## Performance Optimizations

- **Efficient Storage**: Parquet format with compression
- **Batch Processing**: Optimized batch sizes for API calls
- **Memory Management**: Chunked processing for large datasets
- **Concurrent Operations**: Thread-safe parallel processing

## Monitoring and Maintenance

### Health Checks
- Backup integrity validation
- Signal coordination status
- Gap detection accuracy

### Cleanup Operations
- Automatic expired signal cleanup
- Configurable backup retention policies
- Database maintenance integration

### Metrics and Reporting
- Backup completeness scores
- Signal processing times
- Gap detection statistics

## Troubleshooting

### Common Issues
1. **Permission Denied**: Check directory permissions or use fallback paths
2. **Missing Dependencies**: Ensure all required Python packages are installed
3. **Database Connection**: Verify PostgreSQL connection parameters
4. **S3 Access**: Check AWS credentials and bucket permissions

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Advanced S3 Integration**: Multi-region backup support
2. **Real-time Monitoring**: WebSocket-based status updates
3. **Machine Learning**: Predictive gap detection
4. **API Integration**: Direct TwelveData API gap filling
5. **Dashboard Integration**: Visual backup and signal monitoring

## Conclusion

These backup management utilities provide a robust, enterprise-grade solution for data management in the USDCOP trading system. They handle the complexities of timezone awareness, business rules, and coordination between pipeline stages while maintaining data integrity and operational efficiency.