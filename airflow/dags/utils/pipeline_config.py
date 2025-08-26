"""
Pipeline Configuration Loader
==============================
Dynamically loads bucket names and configuration from pipeline_dataflow.yml
This ensures all DAGs use consistent bucket names with prefixes.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Singleton class to load and cache pipeline configuration"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            # Find the config file relative to the DAGs folder
            dag_folder = Path(__file__).parent.parent
            config_path = dag_folder.parent / 'configs' / 'pipeline_dataflow.yml'
            
            if not config_path.exists():
                # Fallback to environment variable
                config_path = Path(os.environ.get('PIPELINE_CONFIG_PATH', 
                                                  '/opt/airflow/configs/pipeline_dataflow.yml'))
            
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                
            logger.info(f"Loaded pipeline configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline configuration: {e}")
            # Fallback to hardcoded values if config file is not available
            self._config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration if YAML file is not available"""
        return {
            'buckets': {
                'l0_acquire': '00-raw-usdcop-marketdata',  # UNIFIED L0 BUCKET - PRODUCTION ONLY
                'l1_standardize': '01-l1-ds-usdcop-standardize',
                'l2_prepare': '02-l2-ds-usdcop-prepare',
                'l3_feature': '03-l3-ds-usdcop-feature',
                'l4_rlready': '04-l4-ds-usdcop-rlready',
                'l5_serving': '05-l5-ds-usdcop-serving',
                'common_models': '99-common-trading-models',
                'common_reports': '99-common-trading-reports',
                'common_backups': '99-common-trading-backups'
            },
            'quality_gates': {
                'L0': {'min_records': 50, 'required_columns': ['time', 'open', 'high', 'low', 'close']},
                'L1': {'completeness_min': 0.98, 'max_gap_bars': 1, 'required_bars_per_episode': 60},
                'L2': {'outliers_max_pct': 0.01, 'missing_features_max': 0},
                'L3': {'feature_count_min': 20, 'correlation_max': 0.99}
            },
            'connections': {
                'L0_to_L1': {'wait_timeout': 3600, 'poke_interval': 300},
                'L1_to_L2': {'wait_timeout': 3600, 'poke_interval': 300},
                'L2_to_L3': {'wait_timeout': 3600, 'poke_interval': 300}
            }
        }
    
    def get_bucket(self, layer: str) -> str:
        """
        Get bucket name for a specific layer
        
        Args:
            layer: Layer identifier (e.g., 'l0_acquire', 'l1_standardize', etc.)
        
        Returns:
            Bucket name with prefix
        """
        return self._config['buckets'].get(layer, f'unknown-{layer}')
    
    def get_bucket_for_dag(self, dag_id: str) -> Dict[str, str]:
        """
        Get input and output buckets for a specific DAG
        
        Args:
            dag_id: DAG identifier (e.g., 'usdcop_m5__01_l0_acquire')
        
        Returns:
            Dictionary with 'input' and 'output' bucket names
        """
        buckets = {}
        
        # Map DAG ID to layer
        if 'l0_acquire' in dag_id or '_01_l0_' in dag_id:
            buckets['output'] = self.get_bucket('l0_acquire')
            buckets['input'] = None  # L0 has no input bucket
            
        elif 'l1_standardize' in dag_id or '_02_l1_' in dag_id:
            buckets['input'] = self.get_bucket('l0_acquire')
            buckets['output'] = self.get_bucket('l1_standardize')
            
        elif 'l2_prepare' in dag_id or '_03_l2_' in dag_id:
            buckets['input'] = self.get_bucket('l1_standardize')
            buckets['output'] = self.get_bucket('l2_prepare')
            
        elif 'l3_feature' in dag_id or '_04_l3_' in dag_id:
            buckets['input'] = self.get_bucket('l2_prepare')
            buckets['output'] = self.get_bucket('l3_feature')
            
        elif 'l4_rlready' in dag_id or '_05_l4_' in dag_id:
            buckets['input'] = self.get_bucket('l3_feature')
            buckets['output'] = self.get_bucket('l4_rlready')
            
        elif 'l5_serving' in dag_id or '_06_l5_' in dag_id:
            buckets['input'] = self.get_bucket('l4_rlready')
            buckets['output'] = self.get_bucket('l5_serving')
        
        return buckets
    
    def get_quality_gates(self, layer: str) -> Dict[str, Any]:
        """Get quality gate configuration for a specific layer"""
        return self._config.get('quality_gates', {}).get(layer, {})
    
    def get_connection_config(self, connection: str) -> Dict[str, Any]:
        """Get connection configuration (timeouts, intervals, etc.)"""
        return self._config.get('connections', {}).get(connection, {})
    
    def get_pipeline_info(self, dag_id: str) -> Optional[Dict[str, Any]]:
        """Get complete pipeline information for a DAG"""
        pipelines = self._config.get('pipelines', {})
        return pipelines.get(dag_id)


# Convenience functions for DAGs
def get_bucket_config(dag_id: str) -> Dict[str, str]:
    """
    Get bucket configuration for a DAG
    
    Usage in DAG:
        from utils.pipeline_config import get_bucket_config
        
        buckets = get_bucket_config('usdcop_m5__01_l0_acquire')
        BUCKET_OUTPUT = buckets['output']
    """
    config = PipelineConfig()
    return config.get_bucket_for_dag(dag_id)


def get_bucket(layer: str) -> str:
    """
    Get bucket name for a specific layer
    
    Usage in DAG:
        from utils.pipeline_config import get_bucket
        
        BUCKET_OUTPUT = get_bucket('l0_acquire')
    """
    config = PipelineConfig()
    return config.get_bucket(layer)


def get_quality_gates(layer: str) -> Dict[str, Any]:
    """
    Get quality gates for a layer
    
    Usage in DAG:
        from utils.pipeline_config import get_quality_gates
        
        quality = get_quality_gates('L1')
        min_completeness = quality.get('completeness_min', 0.98)
    """
    config = PipelineConfig()
    return config.get_quality_gates(layer)


# For backward compatibility - these will be set dynamically
# DAGs can import these directly if needed
config = PipelineConfig()

# Export bucket names as constants for easy import
BUCKET_L0_ACQUIRE = config.get_bucket('l0_acquire')
BUCKET_L1_STANDARDIZE = config.get_bucket('l1_standardize')
BUCKET_L2_PREPARE = config.get_bucket('l2_prepare')
BUCKET_L3_FEATURE = config.get_bucket('l3_feature')
BUCKET_L4_RLREADY = config.get_bucket('l4_rlready')
BUCKET_L5_SERVING = config.get_bucket('l5_serving')
BUCKET_MODELS = config.get_bucket('common_models')
BUCKET_REPORTS = config.get_bucket('common_reports')
BUCKET_BACKUPS = config.get_bucket('common_backups')