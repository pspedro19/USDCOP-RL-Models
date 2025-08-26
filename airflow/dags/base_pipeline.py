"""
Base Pipeline Class
====================
Provides common functionality for all L0-L5 pipelines
Ensures seamless integration and data flow
"""

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import io
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pipeline_integration_config import (
    get_pipeline_config, 
    get_input_config, 
    get_output_config,
    format_path,
    SENSOR_CONFIG
)

logger = logging.getLogger(__name__)

class BasePipeline:
    """Base class for all pipeline layers"""
    
    def __init__(self, layer: str, context: Dict[str, Any]):
        """
        Initialize base pipeline
        
        Args:
            layer: Pipeline layer (L0, L1, L2, L3, L4, L5)
            context: Airflow context
        """
        self.layer = layer
        self.context = context
        self.execution_date = context.get('ds', datetime.now().strftime('%Y-%m-%d'))
        self.run_id = str(uuid.uuid4())
        
        # Get configurations
        self.config = get_pipeline_config(layer)
        self.input_config = get_input_config(layer)
        self.output_config = get_output_config(layer)
        
        # Initialize S3 hook
        self.s3_hook = S3Hook(aws_conn_id=self.config['minio_conn_id'])
        
        # Parse date components
        self.year, self.month, self.day = self.execution_date.split('-')
        
        logger.info(f"Initialized {layer} pipeline")
        logger.info(f"Execution date: {self.execution_date}")
        logger.info(f"Run ID: {self.run_id}")
    
    def wait_for_upstream(self) -> bool:
        """Wait for upstream pipeline to complete"""
        if not self.input_config:
            logger.info(f"{self.layer} has no upstream dependency")
            return True
        
        upstream_layer = self.input_config['upstream_layer']
        wait_pattern = self.input_config['wait_for_signal']
        upstream_bucket = self.input_config['upstream_bucket']
        
        # Format the wait pattern
        wait_key = format_path(wait_pattern, date=self.execution_date)
        
        logger.info(f"Waiting for {upstream_layer} completion")
        logger.info(f"Looking for signal: {wait_key} in bucket {upstream_bucket}")
        
        # Check for signal
        prefix = wait_key.split('*')[0] if '*' in wait_key else wait_key
        keys = self.s3_hook.list_keys(bucket_name=upstream_bucket, prefix=prefix)
        
        if keys:
            logger.info(f"Found {upstream_layer} READY signal: {keys[0]}")
            return True
        
        # Fallback: Check if data exists
        logger.warning(f"No READY signal found, checking for {upstream_layer} data")
        return self.check_upstream_data_exists()
    
    def check_upstream_data_exists(self) -> bool:
        """Check if upstream data exists (fallback when no READY signal)"""
        if not self.input_config:
            return False
        
        input_pattern = self.input_config['input_pattern']
        upstream_bucket = self.input_config['upstream_bucket']
        
        # Format the input pattern
        input_key = format_path(
            input_pattern,
            date=self.execution_date,
            year=self.year,
            month=self.month,
            day=self.day
        )
        
        prefix = input_key.split('*')[0] if '*' in input_key else input_key
        keys = self.s3_hook.list_keys(bucket_name=upstream_bucket, prefix=prefix)
        
        if keys:
            logger.info(f"Found upstream data: {keys[0]}")
            return True
        
        return False
    
    def load_upstream_data(self) -> pd.DataFrame:
        """Load data from upstream pipeline"""
        if not self.input_config:
            raise ValueError(f"{self.layer} has no upstream to load from")
        
        upstream_bucket = self.input_config['upstream_bucket']
        input_pattern = self.input_config['input_pattern']
        
        # Build search patterns
        search_patterns = [
            format_path(
                input_pattern,
                date=self.execution_date,
                year=self.year,
                month=self.month,
                day=self.day,
                upstream_dag=self.input_config['upstream_dag']
            )
        ]
        
        # Add alternative patterns
        if self.layer == 'L2':
            # L2 reads from L1
            search_patterns.extend([
                f"l1-standardize/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/standardized_*.parquet",
                f"l1-standardize/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/*.parquet",
            ])
        elif self.layer == 'L3':
            # L3 reads from L2
            search_patterns.extend([
                f"l2-prepare/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/prepared_*.parquet",
                f"l2-prepare/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/*.parquet",
            ])
        elif self.layer == 'L4':
            # L4 reads from L3
            search_patterns.extend([
                f"l3-feature/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/features_*.parquet",
                f"l3-feature/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/*.parquet",
            ])
        elif self.layer == 'L5':
            # L5 reads from L4
            search_patterns.extend([
                f"l4-rlready/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/rlready_*.parquet",
                f"l4-rlready/market=usdcop/timeframe=m5/year={self.year}/month={self.month}/day={self.day}/*.parquet",
            ])
        
        logger.info(f"Searching for upstream data in bucket: {upstream_bucket}")
        
        df = None
        source_file = None
        
        for pattern in search_patterns:
            logger.info(f"Trying pattern: {pattern}")
            
            try:
                if '*' in pattern:
                    prefix = pattern.split('*')[0]
                    keys = self.s3_hook.list_keys(bucket_name=upstream_bucket, prefix=prefix)
                    
                    if keys:
                        # Filter for data files
                        data_keys = [k for k in keys if k.endswith(('.parquet', '.csv'))]
                        
                        if data_keys:
                            # Take the latest file
                            key = sorted(data_keys)[-1]
                            logger.info(f"Found data file: {key}")
                            
                            # Load the file
                            file_obj = self.s3_hook.get_key(key, bucket_name=upstream_bucket)
                            content = file_obj.get()['Body'].read()
                            
                            if key.endswith('.parquet'):
                                df = pd.read_parquet(io.BytesIO(content))
                            else:
                                df = pd.read_csv(io.BytesIO(content))
                            
                            source_file = key
                            logger.info(f"Loaded {len(df)} rows from {key}")
                            break
            except Exception as e:
                logger.warning(f"Error with pattern {pattern}: {e}")
                continue
        
        if df is None:
            raise ValueError(f"No upstream data found for {self.layer}")
        
        logger.info(f"Successfully loaded upstream data: {len(df)} records")
        self.context['ti'].xcom_push(key='upstream_source', value=source_file)
        
        return df
    
    def save_output(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> str:
        """Save processed data to output bucket"""
        output_bucket = self.output_config['bucket']
        output_pattern = self.output_config['output_pattern']
        ready_pattern = self.output_config['ready_signal']
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Format output path
        output_key = format_path(
            output_pattern,
            date=self.execution_date,
            year=self.year,
            month=self.month,
            day=self.day,
            timestamp=timestamp,
            run_id=self.run_id
        )
        
        # Save Parquet
        logger.info(f"Saving to: {output_bucket}/{output_key}")
        
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buffer, compression='snappy')
        buffer.seek(0)
        
        self.s3_hook.load_bytes(
            bytes_data=buffer.getvalue(),
            key=output_key,
            bucket_name=output_bucket,
            replace=True
        )
        logger.info(f"Saved {len(df)} records to {output_key}")
        
        # Save CSV backup
        csv_key = output_key.replace('.parquet', '.csv')
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        self.s3_hook.load_string(
            string_data=csv_buffer.getvalue(),
            key=csv_key,
            bucket_name=output_bucket,
            replace=True
        )
        logger.info(f"Saved CSV backup: {csv_key}")
        
        # Save metadata
        if metadata:
            metadata_key = output_key.replace('.parquet', '_metadata.json')
            self.s3_hook.load_string(
                string_data=json.dumps(metadata, indent=2),
                key=metadata_key,
                bucket_name=output_bucket,
                replace=True
            )
            logger.info(f"Saved metadata: {metadata_key}")
        
        # Create READY signal
        ready_key = format_path(
            ready_pattern,
            date=self.execution_date,
            run_id=self.run_id
        )
        
        ready_data = {
            'status': 'READY',
            'timestamp': datetime.now().isoformat(),
            'layer': self.layer,
            'run_id': self.run_id,
            'records': len(df),
            'output_file': output_key
        }
        
        self.s3_hook.load_string(
            string_data=json.dumps(ready_data, indent=2),
            key=ready_key,
            bucket_name=output_bucket,
            replace=True
        )
        logger.info(f"Created READY signal: {ready_key}")
        
        return output_key
    
    def create_sensor(self):
        """Create S3KeySensor for upstream dependency"""
        if self.layer not in SENSOR_CONFIG:
            return None
        
        config = SENSOR_CONFIG[self.layer]
        
        return S3KeySensor(
            task_id=config['sensor_id'],
            bucket_name=config['bucket_name'],
            bucket_key=config['bucket_key'](self.execution_date),
            wildcard_match=config['wildcard_match'],
            aws_conn_id=self.config['minio_conn_id'],
            timeout=config['timeout'],
            poke_interval=config['poke_interval'],
            soft_fail=config['soft_fail']
        )
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        # This should be overridden by each pipeline
        logger.info(f"Running basic validation for {self.layer}")
        
        if df.empty:
            logger.error("Data is empty")
            return False
        
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return True
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data - to be overridden by each pipeline"""
        raise NotImplementedError(f"Process method must be implemented for {self.layer}")
    
    def run(self):
        """Main pipeline execution"""
        logger.info(f"{'='*70}")
        logger.info(f"{self.layer} PIPELINE EXECUTION")
        logger.info(f"{'='*70}")
        
        try:
            # Wait for upstream
            if self.input_config:
                if not self.wait_for_upstream():
                    raise ValueError(f"Upstream {self.input_config['upstream_layer']} not ready")
                
                # Load upstream data
                df = self.load_upstream_data()
            else:
                # L0 loads from external source
                df = self.load_external_data()
            
            # Validate input
            if not self.validate_data(df):
                raise ValueError(f"Input validation failed for {self.layer}")
            
            # Process data
            processed_df = self.process(df)
            
            # Validate output
            if not self.validate_data(processed_df):
                raise ValueError(f"Output validation failed for {self.layer}")
            
            # Save output
            output_key = self.save_output(processed_df)
            
            logger.info(f"{'='*70}")
            logger.info(f"{self.layer} PIPELINE COMPLETE")
            logger.info(f"Output: {output_key}")
            logger.info(f"Records: {len(processed_df)}")
            logger.info(f"{'='*70}")
            
            return {
                'layer': self.layer,
                'run_id': self.run_id,
                'records': len(processed_df),
                'output': output_key
            }
            
        except Exception as e:
            logger.error(f"{self.layer} pipeline failed: {e}")
            raise
    
    def load_external_data(self) -> pd.DataFrame:
        """Load data from external source (L0 only)"""
        raise NotImplementedError("load_external_data must be implemented for L0")