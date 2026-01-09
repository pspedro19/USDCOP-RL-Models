"""
Pipeline Integration Configuration
===================================
Provides configuration for pipeline layers L0-L5
"""

from datetime import timedelta
from typing import Dict, Any, Optional

# Sensor configuration for each layer
SENSOR_CONFIG = {
    'L0': {
        'timeout': 300,
        'poke_interval': 30,
        'mode': 'poke'
    },
    'L1': {
        'timeout': 600,
        'poke_interval': 60,
        'mode': 'poke'
    },
    'L2': {
        'timeout': 600,
        'poke_interval': 60,
        'mode': 'poke'
    },
    'L3': {
        'timeout': 900,
        'poke_interval': 60,
        'mode': 'poke'
    },
    'L4': {
        'timeout': 1200,
        'poke_interval': 60,
        'mode': 'poke'
    },
    'L5': {
        'timeout': 1800,
        'poke_interval': 60,
        'mode': 'poke'
    }
}

# Pipeline configurations for each layer
PIPELINE_CONFIGS = {
    'L0': {
        'name': 'L0_Acquire',
        'description': 'Raw data acquisition from data sources',
        'bucket': '00-raw-usdcop-marketdata',
        'max_retries': 3,
        'retry_delay': 300
    },
    'L1': {
        'name': 'L1_Standardize',
        'description': 'Data standardization and quality checks',
        'bucket': '01-l1-ds-usdcop-standardize',
        'max_retries': 2,
        'retry_delay': 300
    },
    'L2': {
        'name': 'L2_Prepare',
        'description': 'Data preparation and cleaning',
        'bucket': '02-l2-ds-usdcop-prepared',
        'max_retries': 2,
        'retry_delay': 300
    },
    'L3': {
        'name': 'L3_Feature',
        'description': 'Feature engineering',
        'bucket': '03-l3-ds-usdcop-features',
        'max_retries': 2,
        'retry_delay': 300
    },
    'L4': {
        'name': 'L4_RLReady',
        'description': 'RL-ready dataset preparation',
        'bucket': '04-l4-ds-usdcop-rlready',
        'max_retries': 2,
        'retry_delay': 300
    },
    'L5': {
        'name': 'L5_Serving',
        'description': 'Model training and serving',
        'bucket': '05-l5-ds-usdcop-serving',
        'max_retries': 2,
        'retry_delay': 300
    }
}

# Input configurations for each layer
INPUT_CONFIGS = {
    'L0': {
        'source': None,  # External data sources
        'format': 'various'
    },
    'L1': {
        'source': 'L0',
        'bucket': '00-raw-usdcop-marketdata',
        'format': 'csv'
    },
    'L2': {
        'source': 'L1',
        'bucket': '01-l1-ds-usdcop-standardize',
        'format': 'parquet'
    },
    'L3': {
        'source': 'L2',
        'bucket': '02-l2-ds-usdcop-prepared',
        'format': 'parquet'
    },
    'L4': {
        'source': 'L3',
        'bucket': '03-l3-ds-usdcop-features',
        'format': 'parquet'
    },
    'L5': {
        'source': 'L4',
        'bucket': '04-l4-ds-usdcop-rlready',
        'format': 'parquet'
    }
}

# Output configurations for each layer
OUTPUT_CONFIGS = {
    'L0': {
        'bucket': '00-raw-usdcop-marketdata',
        'format': 'csv',
        'partitioning': 'date'
    },
    'L1': {
        'bucket': '01-l1-ds-usdcop-standardize',
        'format': 'parquet',
        'partitioning': 'date'
    },
    'L2': {
        'bucket': '02-l2-ds-usdcop-prepared',
        'format': 'parquet',
        'partitioning': 'date'
    },
    'L3': {
        'bucket': '03-l3-ds-usdcop-features',
        'format': 'parquet',
        'partitioning': 'date'
    },
    'L4': {
        'bucket': '04-l4-ds-usdcop-rlready',
        'format': 'parquet',
        'partitioning': 'run_id'
    },
    'L5': {
        'bucket': '05-l5-ds-usdcop-serving',
        'format': 'mixed',  # Models, configs, etc.
        'partitioning': 'run_id'
    }
}

def get_pipeline_config(layer: str) -> Dict[str, Any]:
    """
    Get pipeline configuration for a specific layer
    
    Args:
        layer: Pipeline layer (L0, L1, L2, L3, L4, L5)
        
    Returns:
        Pipeline configuration dictionary
    """
    if layer not in PIPELINE_CONFIGS:
        raise ValueError(f"Invalid layer: {layer}. Must be one of {list(PIPELINE_CONFIGS.keys())}")
    
    return PIPELINE_CONFIGS[layer]

def get_input_config(layer: str) -> Optional[Dict[str, Any]]:
    """
    Get input configuration for a specific layer
    
    Args:
        layer: Pipeline layer (L0, L1, L2, L3, L4, L5)
        
    Returns:
        Input configuration dictionary or None for L0
    """
    if layer not in INPUT_CONFIGS:
        raise ValueError(f"Invalid layer: {layer}. Must be one of {list(INPUT_CONFIGS.keys())}")
    
    return INPUT_CONFIGS[layer]

def get_output_config(layer: str) -> Dict[str, Any]:
    """
    Get output configuration for a specific layer
    
    Args:
        layer: Pipeline layer (L0, L1, L2, L3, L4, L5)
        
    Returns:
        Output configuration dictionary
    """
    if layer not in OUTPUT_CONFIGS:
        raise ValueError(f"Invalid layer: {layer}. Must be one of {list(OUTPUT_CONFIGS.keys())}")
    
    return OUTPUT_CONFIGS[layer]

def format_path(pattern: str, **kwargs) -> str:
    """
    Format a path pattern with provided values
    
    Args:
        pattern: Path pattern with placeholders
        **kwargs: Values to substitute in the pattern
        
    Returns:
        Formatted path string
        
    Examples:
        >>> format_path("data/date={date}/file.parquet", date="2024-01-01")
        'data/date=2024-01-01/file.parquet'
    """
    return pattern.format(**kwargs)

def get_bucket_for_layer(layer: str) -> str:
    """
    Get the MinIO bucket name for a specific layer
    
    Args:
        layer: Pipeline layer (L0, L1, L2, L3, L4, L5)
        
    Returns:
        Bucket name
    """
    config = get_pipeline_config(layer)
    return config['bucket']

def get_connection_id() -> str:
    """
    Get the Airflow connection ID for MinIO
    
    Returns:
        Connection ID string
    """
    return 'minio_conn'

# Default DAG arguments for all pipelines
DEFAULT_DAG_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# timedelta imported at top of file

# Export all functions and constants
__all__ = [
    'SENSOR_CONFIG',
    'PIPELINE_CONFIGS',
    'INPUT_CONFIGS',
    'OUTPUT_CONFIGS',
    'get_pipeline_config',
    'get_input_config',
    'get_output_config',
    'format_path',
    'get_bucket_for_layer',
    'get_connection_id',
    'DEFAULT_DAG_ARGS'
]