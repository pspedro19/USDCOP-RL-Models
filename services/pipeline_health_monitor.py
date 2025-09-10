"""
Pipeline Health Monitoring Service
==================================

Comprehensive health monitoring for L0-L6 pipeline stages with:
- Real-time status tracking
- Data flow metrics
- Processing time monitoring  
- Error rate tracking
- Airflow DAG integration
- System health indicators
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor
import redis.asyncio as redis
import boto3
from botocore.client import Config

@dataclass
class PipelineStageHealth:
    """Health status for a single pipeline stage"""
    stage: str
    status: str  # HEALTHY, WARNING, ERROR, UNKNOWN
    last_run: Optional[datetime]
    processing_time: Optional[float]  # seconds
    records_processed: int
    error_count: int
    error_rate: float
    data_completeness: float
    quality_score: float
    next_scheduled: Optional[datetime]
    dag_status: str
    
@dataclass  
class DataFlowMetrics:
    """Data flow metrics between pipeline stages"""
    source_stage: str
    target_stage: str
    records_transferred: int
    transfer_time: float
    data_quality_delta: float
    last_transfer: datetime

@dataclass
class SystemHealthIndicators:
    """Overall system health indicators"""
    overall_status: str
    pipeline_availability: float
    average_processing_time: float
    total_error_rate: float
    data_freshness: float  # hours since last L0 data
    storage_usage: Dict[str, float]
    active_dags: int
    failed_dags: int

class PipelineHealthMonitor:
    """
    Comprehensive pipeline health monitoring service
    
    Provides real-time monitoring of:
    - L0-L6 pipeline stage health
    - Data flow between stages
    - Processing times and error rates
    - Airflow DAG status
    - System resource usage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections
        self._db_pool = None
        self._redis_client = None
        self._s3_client = None
        
        # Pipeline stages
        self.pipeline_stages = [
            'L0_ACQUIRE', 'L1_STANDARDIZE', 'L2_PREPARE', 
            'L3_FEATURE', 'L4_RLREADY', 'L5_SERVING', 'L6_BACKTEST'
        ]
        
        # Monitoring intervals
        self.health_check_interval = config.get('health_check_interval', 30)  # seconds
        self.metrics_retention_hours = config.get('metrics_retention_hours', 168)  # 1 week
        
        # Cache for recent metrics
        self._metrics_cache = {}
        self._last_update = {}
        
    async def initialize(self):
        """Initialize connections and start monitoring"""
        try:
            # Initialize database connection
            await self._init_database()
            
            # Initialize Redis connection
            await self._init_redis()
            
            # Initialize S3 client
            await self._init_s3()
            
            # Create monitoring tables
            await self._create_monitoring_tables()
            
            self.logger.info("Pipeline health monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline health monitor: {e}")
            raise
    
    async def _init_database(self):
        """Initialize database connection pool"""
        db_config = self.config['database']
        self._db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        redis_config = self.config['redis']
        self._redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config.get('password'),
            decode_responses=True
        )
        await self._redis_client.ping()
    
    async def _init_s3(self):
        """Initialize S3/MinIO client"""
        s3_config = self.config['s3']
        self._s3_client = boto3.client(
            's3',
            endpoint_url=s3_config['endpoint'],
            aws_access_key_id=s3_config['access_key'],
            aws_secret_access_key=s3_config['secret_key'],
            config=Config(signature_version='s3v4')
        )
    
    async def _create_monitoring_tables(self):
        """Create monitoring tables if they don't exist"""
        with self._db_pool.getconn() as conn:
            with conn.cursor() as cur:
                # Pipeline stage health history
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_health_history (
                        id SERIAL PRIMARY KEY,
                        stage VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        status VARCHAR(20) NOT NULL,
                        processing_time FLOAT,
                        records_processed INTEGER,
                        error_count INTEGER,
                        error_rate FLOAT,
                        data_completeness FLOAT,
                        quality_score FLOAT,
                        dag_status VARCHAR(20),
                        metadata JSONB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_pipeline_health_stage_timestamp 
                    ON pipeline_health_history (stage, timestamp DESC);
                """)
                
                # Data flow metrics
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS data_flow_metrics (
                        id SERIAL PRIMARY KEY,
                        source_stage VARCHAR(20) NOT NULL,
                        target_stage VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        records_transferred INTEGER,
                        transfer_time FLOAT,
                        data_quality_delta FLOAT,
                        metadata JSONB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_data_flow_timestamp 
                    ON data_flow_metrics (timestamp DESC);
                """)
                
                # System health indicators
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_health_history (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        overall_status VARCHAR(20) NOT NULL,
                        pipeline_availability FLOAT,
                        average_processing_time FLOAT,
                        total_error_rate FLOAT,
                        data_freshness FLOAT,
                        storage_usage JSONB,
                        active_dags INTEGER,
                        failed_dags INTEGER,
                        metadata JSONB
                    );
                """)
                
                conn.commit()
                self.logger.info("Monitoring tables created successfully")
    
    async def get_pipeline_health(self) -> List[PipelineStageHealth]:
        """Get current health status for all pipeline stages"""
        health_status = []
        
        for stage in self.pipeline_stages:
            try:
                stage_health = await self._check_stage_health(stage)
                health_status.append(stage_health)
                
                # Cache the result
                self._metrics_cache[f"health_{stage}"] = asdict(stage_health)
                self._last_update[f"health_{stage}"] = time.time()
                
            except Exception as e:
                self.logger.error(f"Error checking health for {stage}: {e}")
                # Create error status
                health_status.append(PipelineStageHealth(
                    stage=stage,
                    status="ERROR",
                    last_run=None,
                    processing_time=None,
                    records_processed=0,
                    error_count=1,
                    error_rate=1.0,
                    data_completeness=0.0,
                    quality_score=0.0,
                    next_scheduled=None,
                    dag_status="ERROR"
                ))
        
        return health_status
    
    async def _check_stage_health(self, stage: str) -> PipelineStageHealth:
        """Check health for a specific pipeline stage"""
        
        # Get DAG status from Airflow
        dag_status = await self._get_airflow_dag_status(stage)
        
        # Get latest audit data
        audit_data = await self._get_latest_audit_data(stage)
        
        # Get processing metrics
        processing_metrics = await self._get_processing_metrics(stage)
        
        # Get data completeness and quality
        quality_metrics = await self._get_quality_metrics(stage)
        
        # Determine overall status
        status = self._determine_stage_status(
            dag_status, audit_data, processing_metrics, quality_metrics
        )
        
        return PipelineStageHealth(
            stage=stage,
            status=status,
            last_run=processing_metrics.get('last_run'),
            processing_time=processing_metrics.get('processing_time'),
            records_processed=audit_data.get('total_records', 0),
            error_count=audit_data.get('violations_count', 0),
            error_rate=audit_data.get('violations_count', 0) / max(audit_data.get('total_records', 1), 1),
            data_completeness=quality_metrics.get('completeness_rate', 0.0),
            quality_score=self._calculate_quality_score(audit_data, quality_metrics),
            next_scheduled=processing_metrics.get('next_scheduled'),
            dag_status=dag_status.get('state', 'unknown')
        )
    
    async def _get_airflow_dag_status(self, stage: str) -> Dict[str, Any]:
        """Get DAG status from Airflow API"""
        try:
            airflow_config = self.config['airflow']
            dag_id = f"usdcop_m5__{stage.lower()}"
            
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(
                    airflow_config['username'], 
                    airflow_config['password']
                )
                
                url = f"{airflow_config['base_url']}/api/v1/dags/{dag_id}/dagRuns"
                params = {'limit': 1, 'order_by': '-execution_date'}
                
                async with session.get(url, auth=auth, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('dag_runs'):
                            return data['dag_runs'][0]
                    
                    return {'state': 'unknown', 'execution_date': None}
                    
        except Exception as e:
            self.logger.error(f"Error getting Airflow status for {stage}: {e}")
            return {'state': 'error', 'execution_date': None}
    
    async def _get_latest_audit_data(self, stage: str) -> Dict[str, Any]:
        """Get latest audit data for a pipeline stage"""
        try:
            # Map stage to audit file
            audit_files = {
                'L0_ACQUIRE': 'l0_audit.json',
                'L1_STANDARDIZE': 'l1_audit.json', 
                'L2_PREPARE': 'l2_audit.json',
                'L3_FEATURE': 'l3_audit.json',
                'L4_RLREADY': 'l4_audit.json',
                'L5_SERVING': 'l5_audit.json',
                'L6_BACKTEST': 'l6_audit.json'
            }
            
            audit_file = audit_files.get(stage, f'{stage.lower()}_audit.json')
            audit_path = Path(self.config['project_root']) / audit_file
            
            if audit_path.exists():
                with open(audit_path, 'r') as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error reading audit data for {stage}: {e}")
            return {}
    
    async def _get_processing_metrics(self, stage: str) -> Dict[str, Any]:
        """Get processing time and execution metrics"""
        try:
            with self._db_pool.getconn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            timestamp as last_run,
                            processing_time,
                            records_processed
                        FROM pipeline_health_history 
                        WHERE stage = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (stage,))
                    
                    result = cur.fetchone()
                    if result:
                        return dict(result)
                    
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting processing metrics for {stage}: {e}")
            return {}
    
    async def _get_quality_metrics(self, stage: str) -> Dict[str, Any]:
        """Get data quality metrics for a pipeline stage"""
        try:
            # Get from S3 bucket for the stage
            stage_bucket_map = {
                'L0_ACQUIRE': '00-l0-ds-usdcop-acquire',
                'L1_STANDARDIZE': '01-l1-ds-usdcop-standardize',
                'L2_PREPARE': '02-l2-ds-usdcop-prepare',
                'L3_FEATURE': '03-l3-ds-usdcop-feature',
                'L4_RLREADY': '04-l4-ds-usdcop-rlready',
                'L5_SERVING': '05-l5-ds-usdcop-serving'
            }
            
            bucket = stage_bucket_map.get(stage)
            if not bucket:
                return {}
                
            # List recent objects to check data freshness
            response = self._s3_client.list_objects_v2(
                Bucket=bucket,
                MaxKeys=100
            )
            
            if 'Contents' not in response:
                return {'completeness_rate': 0.0}
            
            # Calculate metrics based on object count and timestamps
            objects = response['Contents']
            latest_object = max(objects, key=lambda x: x['LastModified'])
            
            data_freshness = (datetime.now(latest_object['LastModified'].tzinfo) - 
                            latest_object['LastModified']).total_seconds() / 3600  # hours
            
            return {
                'completeness_rate': min(len(objects) / 100, 1.0),  # Normalize to 0-1
                'data_freshness': data_freshness,
                'object_count': len(objects)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality metrics for {stage}: {e}")
            return {'completeness_rate': 0.0}
    
    def _determine_stage_status(self, dag_status: Dict, audit_data: Dict, 
                               processing_metrics: Dict, quality_metrics: Dict) -> str:
        """Determine overall status for a pipeline stage"""
        
        # Check for critical failures
        if dag_status.get('state') == 'failed':
            return 'ERROR'
            
        if audit_data.get('status') == 'FAILED':
            return 'ERROR'
            
        # Check for warnings
        if dag_status.get('state') in ['up_for_retry', 'up_for_reschedule']:
            return 'WARNING'
            
        if audit_data.get('status') == 'WARNING':
            return 'WARNING'
            
        if quality_metrics.get('completeness_rate', 0) < 0.5:
            return 'WARNING'
            
        # Check data freshness (warning if > 2 hours old)
        if quality_metrics.get('data_freshness', 0) > 2:
            return 'WARNING'
            
        # Healthy if all checks pass
        if (dag_status.get('state') == 'success' and 
            audit_data.get('status') in ['PASSED', 'PASS'] and
            quality_metrics.get('completeness_rate', 0) > 0.8):
            return 'HEALTHY'
            
        return 'UNKNOWN'
    
    def _calculate_quality_score(self, audit_data: Dict, quality_metrics: Dict) -> float:
        """Calculate a composite quality score (0-100)"""
        try:
            score_components = []
            
            # Audit status component (40%)
            if audit_data.get('status') == 'PASSED':
                score_components.append(40)
            elif audit_data.get('status') == 'WARNING':
                score_components.append(25)
            else:
                score_components.append(0)
            
            # Data completeness component (30%)
            completeness = quality_metrics.get('completeness_rate', 0)
            score_components.append(completeness * 30)
            
            # Data freshness component (20%)
            freshness = quality_metrics.get('data_freshness', 24)  # hours
            freshness_score = max(0, (6 - min(freshness, 6)) / 6 * 20)  # Best within 6 hours
            score_components.append(freshness_score)
            
            # Error rate component (10%)
            violations_count = audit_data.get('violations_count', 0)
            total_records = audit_data.get('total_records', 1)
            error_rate = violations_count / max(total_records, 1)
            error_score = max(0, (1 - error_rate) * 10)
            score_components.append(error_score)
            
            return round(sum(score_components), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    async def get_data_flow_metrics(self) -> List[DataFlowMetrics]:
        """Get data flow metrics between pipeline stages"""
        flow_metrics = []
        
        # Define pipeline flow
        flow_pairs = [
            ('L0_ACQUIRE', 'L1_STANDARDIZE'),
            ('L1_STANDARDIZE', 'L2_PREPARE'),
            ('L2_PREPARE', 'L3_FEATURE'),
            ('L3_FEATURE', 'L4_RLREADY'),
            ('L4_RLREADY', 'L5_SERVING'),
            ('L5_SERVING', 'L6_BACKTEST')
        ]
        
        for source, target in flow_pairs:
            try:
                metrics = await self._calculate_flow_metrics(source, target)
                flow_metrics.append(metrics)
            except Exception as e:
                self.logger.error(f"Error calculating flow metrics {source} -> {target}: {e}")
        
        return flow_metrics
    
    async def _calculate_flow_metrics(self, source_stage: str, target_stage: str) -> DataFlowMetrics:
        """Calculate data flow metrics between two stages"""
        
        # Get latest processing data for both stages
        source_metrics = await self._get_processing_metrics(source_stage)
        target_metrics = await self._get_processing_metrics(target_stage)
        
        # Get latest audit data
        source_audit = await self._get_latest_audit_data(source_stage)
        target_audit = await self._get_latest_audit_data(target_stage)
        
        # Calculate transfer metrics
        source_records = source_audit.get('total_records', 0)
        target_records = target_audit.get('total_records', 0)
        
        # Estimate transfer time (difference in processing times)
        source_time = source_metrics.get('last_run')
        target_time = target_metrics.get('last_run')
        transfer_time = 0.0
        
        if source_time and target_time:
            transfer_time = (target_time - source_time).total_seconds()
        
        # Calculate quality delta
        source_quality = self._calculate_quality_score(source_audit, {})
        target_quality = self._calculate_quality_score(target_audit, {})
        quality_delta = target_quality - source_quality
        
        return DataFlowMetrics(
            source_stage=source_stage,
            target_stage=target_stage,
            records_transferred=min(source_records, target_records),
            transfer_time=max(transfer_time, 0),
            data_quality_delta=quality_delta,
            last_transfer=target_time or datetime.now()
        )
    
    async def get_system_health_indicators(self) -> SystemHealthIndicators:
        """Get overall system health indicators"""
        
        # Get pipeline health status
        pipeline_health = await self.get_pipeline_health()
        
        # Calculate availability (% of stages that are healthy/warning)
        healthy_stages = sum(1 for stage in pipeline_health 
                           if stage.status in ['HEALTHY', 'WARNING'])
        availability = healthy_stages / len(pipeline_health) if pipeline_health else 0
        
        # Calculate average processing time
        processing_times = [stage.processing_time for stage in pipeline_health 
                          if stage.processing_time is not None]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate total error rate
        total_errors = sum(stage.error_count for stage in pipeline_health)
        total_records = sum(stage.records_processed for stage in pipeline_health)
        error_rate = total_errors / max(total_records, 1)
        
        # Get data freshness (hours since last L0 data)
        l0_stage = next((s for s in pipeline_health if s.stage == 'L0_ACQUIRE'), None)
        data_freshness = 0.0
        if l0_stage and l0_stage.last_run:
            data_freshness = (datetime.now() - l0_stage.last_run).total_seconds() / 3600
        
        # Get storage usage
        storage_usage = await self._get_storage_usage()
        
        # Get DAG statistics
        dag_stats = await self._get_dag_statistics()
        
        # Determine overall status
        error_stages = sum(1 for stage in pipeline_health if stage.status == 'ERROR')
        warning_stages = sum(1 for stage in pipeline_health if stage.status == 'WARNING')
        
        if error_stages > 0:
            overall_status = 'ERROR'
        elif warning_stages > 0:
            overall_status = 'WARNING'
        elif healthy_stages == len(pipeline_health):
            overall_status = 'HEALTHY'
        else:
            overall_status = 'UNKNOWN'
        
        return SystemHealthIndicators(
            overall_status=overall_status,
            pipeline_availability=availability * 100,
            average_processing_time=avg_processing_time,
            total_error_rate=error_rate * 100,
            data_freshness=data_freshness,
            storage_usage=storage_usage,
            active_dags=dag_stats.get('active', 0),
            failed_dags=dag_stats.get('failed', 0)
        )
    
    async def _get_storage_usage(self) -> Dict[str, float]:
        """Get storage usage statistics"""
        try:
            usage = {}
            
            # Get usage for each pipeline bucket
            bucket_map = {
                'L0': '00-l0-ds-usdcop-acquire',
                'L1': '01-l1-ds-usdcop-standardize', 
                'L2': '02-l2-ds-usdcop-prepare',
                'L3': '03-l3-ds-usdcop-feature',
                'L4': '04-l4-ds-usdcop-rlready',
                'L5': '05-l5-ds-usdcop-serving'
            }
            
            total_size = 0
            for stage, bucket in bucket_map.items():
                try:
                    response = self._s3_client.list_objects_v2(Bucket=bucket)
                    if 'Contents' in response:
                        size_bytes = sum(obj['Size'] for obj in response['Contents'])
                        size_mb = size_bytes / (1024 * 1024)  # Convert to MB
                        usage[stage] = round(size_mb, 2)
                        total_size += size_mb
                    else:
                        usage[stage] = 0.0
                except Exception as e:
                    self.logger.warning(f"Could not get size for bucket {bucket}: {e}")
                    usage[stage] = 0.0
            
            usage['total'] = round(total_size, 2)
            return usage
            
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {e}")
            return {}
    
    async def _get_dag_statistics(self) -> Dict[str, int]:
        """Get DAG statistics from Airflow"""
        try:
            airflow_config = self.config['airflow']
            
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(
                    airflow_config['username'],
                    airflow_config['password']
                )
                
                url = f"{airflow_config['base_url']}/api/v1/dags"
                params = {'only_active': True}
                
                async with session.get(url, auth=auth, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        dags = data.get('dags', [])
                        
                        active = len([d for d in dags if d.get('is_active', False)])
                        paused = len([d for d in dags if d.get('is_paused', False)])
                        
                        return {
                            'total': len(dags),
                            'active': active,
                            'paused': paused,
                            'failed': 0  # Would need separate call to get failed runs
                        }
            
            return {'total': 0, 'active': 0, 'paused': 0, 'failed': 0}
            
        except Exception as e:
            self.logger.error(f"Error getting DAG statistics: {e}")
            return {'total': 0, 'active': 0, 'paused': 0, 'failed': 0}
    
    async def record_metrics(self):
        """Record current metrics to database"""
        try:
            # Get all current metrics
            pipeline_health = await self.get_pipeline_health()
            data_flow_metrics = await self.get_data_flow_metrics()
            system_health = await self.get_system_health_indicators()
            
            with self._db_pool.getconn() as conn:
                with conn.cursor() as cur:
                    
                    # Record pipeline health
                    for health in pipeline_health:
                        cur.execute("""
                            INSERT INTO pipeline_health_history 
                            (stage, status, processing_time, records_processed, 
                             error_count, error_rate, data_completeness, 
                             quality_score, dag_status, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            health.stage,
                            health.status,
                            health.processing_time,
                            health.records_processed,
                            health.error_count,
                            health.error_rate,
                            health.data_completeness,
                            health.quality_score,
                            health.dag_status,
                            json.dumps({
                                'last_run': health.last_run.isoformat() if health.last_run else None,
                                'next_scheduled': health.next_scheduled.isoformat() if health.next_scheduled else None
                            })
                        ))
                    
                    # Record data flow metrics
                    for flow in data_flow_metrics:
                        cur.execute("""
                            INSERT INTO data_flow_metrics
                            (source_stage, target_stage, records_transferred,
                             transfer_time, data_quality_delta, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            flow.source_stage,
                            flow.target_stage,
                            flow.records_transferred,
                            flow.transfer_time,
                            flow.data_quality_delta,
                            json.dumps({
                                'last_transfer': flow.last_transfer.isoformat() if flow.last_transfer else None
                            })
                        ))
                    
                    # Record system health
                    cur.execute("""
                        INSERT INTO system_health_history
                        (overall_status, pipeline_availability, average_processing_time,
                         total_error_rate, data_freshness, storage_usage, 
                         active_dags, failed_dags, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        system_health.overall_status,
                        system_health.pipeline_availability,
                        system_health.average_processing_time,
                        system_health.total_error_rate,
                        system_health.data_freshness,
                        json.dumps(system_health.storage_usage),
                        system_health.active_dags,
                        system_health.failed_dags,
                        json.dumps({})
                    ))
                    
                    conn.commit()
            
            # Store in Redis for real-time access
            await self._redis_client.setex(
                'pipeline_health_current',
                300,  # 5 minutes TTL
                json.dumps([asdict(h) for h in pipeline_health], default=str)
            )
            
            await self._redis_client.setex(
                'system_health_current',
                300,
                json.dumps(asdict(system_health), default=str)
            )
            
            self.logger.info("Metrics recorded successfully")
            
        except Exception as e:
            self.logger.error(f"Error recording metrics: {e}")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.logger.info("Starting pipeline health monitoring...")
        
        while True:
            try:
                await self.record_metrics()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def cleanup_old_metrics(self):
        """Clean up old metrics data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
            
            with self._db_pool.getconn() as conn:
                with conn.cursor() as cur:
                    # Clean up old pipeline health records
                    cur.execute("""
                        DELETE FROM pipeline_health_history 
                        WHERE timestamp < %s
                    """, (cutoff_time,))
                    
                    # Clean up old data flow metrics
                    cur.execute("""
                        DELETE FROM data_flow_metrics 
                        WHERE timestamp < %s
                    """, (cutoff_time,))
                    
                    # Clean up old system health records
                    cur.execute("""
                        DELETE FROM system_health_history 
                        WHERE timestamp < %s
                    """, (cutoff_time,))
                    
                    conn.commit()
                    
            self.logger.info(f"Cleaned up metrics older than {cutoff_time}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {e}")
    
    async def get_historical_metrics(self, stage: str, hours: int = 24) -> List[Dict]:
        """Get historical metrics for a pipeline stage"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self._db_pool.getconn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM pipeline_health_history 
                        WHERE stage = %s AND timestamp >= %s
                        ORDER BY timestamp ASC
                    """, (stage, cutoff_time))
                    
                    return [dict(row) for row in cur.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting historical metrics for {stage}: {e}")
            return []
    
    async def close(self):
        """Clean up resources"""
        try:
            if self._redis_client:
                await self._redis_client.aclose()
            if self._db_pool:
                self._db_pool.closeall()
                
            self.logger.info("Pipeline health monitor closed")
            
        except Exception as e:
            self.logger.error(f"Error closing pipeline health monitor: {e}")