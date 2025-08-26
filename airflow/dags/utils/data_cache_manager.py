"""
Data Cache Manager for USDCOP Trading Pipeline
===============================================
Manages data caching to avoid unnecessary API calls
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

class DataCacheManager:
    """
    Manages cached data to avoid redundant API calls
    
    Features:
    - Check if data already exists for a date range
    - Track data completeness and quality
    - Determine if refresh is needed
    - Manage cache metadata
    """
    
    def __init__(self, cache_dir: str = "/opt/airflow/data_cache"):
        """Initialize cache manager"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_cache_key(self, source: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache key for a data request"""
        key_string = f"{source}_{start_date.isoformat()}_{end_date.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def check_data_exists(
        self, 
        source: str, 
        start_date: datetime, 
        end_date: datetime,
        min_completeness: float = 95.0
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Check if data exists in cache with sufficient quality
        
        Returns:
            Tuple of (exists, file_path, metadata)
        """
        cache_key = self.get_cache_key(source, start_date, end_date)
        
        if cache_key in self.metadata:
            cache_info = self.metadata[cache_key]
            
            # Check if file still exists
            cache_file = self.cache_dir / cache_info['filename']
            if not cache_file.exists():
                logging.warning(f"Cache file missing: {cache_file}")
                del self.metadata[cache_key]
                self._save_metadata()
                return False, None, None
            
            # Check data freshness (re-fetch if older than 7 days for recent data)
            cached_time = datetime.fromisoformat(cache_info['cached_at'])
            if end_date > datetime.now() - timedelta(days=30):  # Recent data
                if datetime.now() - cached_time > timedelta(days=7):
                    logging.info(f"Cache expired for recent data: {cache_key}")
                    return False, None, None
            
            # Check completeness
            completeness = cache_info.get('completeness', 0)
            if completeness >= min_completeness:
                logging.info(f"✅ Cache hit: {source} {start_date.date()} to {end_date.date()}")
                logging.info(f"   Completeness: {completeness:.1f}%")
                logging.info(f"   Records: {cache_info.get('record_count', 0):,}")
                return True, str(cache_file), cache_info
            else:
                logging.info(f"Cache exists but incomplete: {completeness:.1f}% < {min_completeness}%")
                return False, None, None
        
        return False, None, None
    
    def save_to_cache(
        self,
        data: pd.DataFrame,
        source: str,
        start_date: datetime,
        end_date: datetime,
        completeness: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save data to cache with metadata"""
        cache_key = self.get_cache_key(source, start_date, end_date)
        filename = f"{source}_{cache_key}.parquet"
        cache_file = self.cache_dir / filename
        
        # Save data
        data.to_parquet(cache_file, compression='snappy')
        
        # Update metadata
        cache_info = {
            'filename': filename,
            'source': source,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'cached_at': datetime.now().isoformat(),
            'record_count': len(data),
            'completeness': completeness,
            'file_size_mb': cache_file.stat().st_size / (1024 * 1024),
        }
        
        if metadata:
            cache_info.update(metadata)
        
        self.metadata[cache_key] = cache_info
        self._save_metadata()
        
        logging.info(f"💾 Saved to cache: {source} {start_date.date()} to {end_date.date()}")
        logging.info(f"   File: {filename}")
        logging.info(f"   Records: {len(data):,}")
        logging.info(f"   Completeness: {completeness:.1f}%")
        
        return str(cache_file)
    
    def load_from_cache(self, file_path: str) -> pd.DataFrame:
        """Load data from cache file"""
        return pd.read_parquet(file_path)
    
    def get_missing_periods(
        self,
        source: str,
        start_date: datetime,
        end_date: datetime,
        batch_days: int = 15
    ) -> List[Tuple[datetime, datetime]]:
        """
        Determine which periods need to be fetched
        
        Returns list of (start, end) tuples for missing periods
        """
        missing_periods = []
        current = start_date
        
        while current < end_date:
            batch_end = min(current + timedelta(days=batch_days), end_date)
            
            # Check if this period exists in cache
            exists, _, _ = self.check_data_exists(source, current, batch_end)
            
            if not exists:
                missing_periods.append((current, batch_end))
            
            current = batch_end
        
        if missing_periods:
            logging.info(f"📊 Missing periods for {source}:")
            for start, end in missing_periods:
                logging.info(f"   - {start.date()} to {end.date()}")
        else:
            logging.info(f"✅ All data cached for {source} from {start_date.date()} to {end_date.date()}")
        
        return missing_periods
    
    def cleanup_old_cache(self, days_to_keep: int = 30):
        """Remove cache files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        removed_count = 0
        
        for cache_key, cache_info in list(self.metadata.items()):
            cached_at = datetime.fromisoformat(cache_info['cached_at'])
            if cached_at < cutoff_date:
                # Remove file
                cache_file = self.cache_dir / cache_info['filename']
                if cache_file.exists():
                    cache_file.unlink()
                
                # Remove metadata
                del self.metadata[cache_key]
                removed_count += 1
        
        if removed_count > 0:
            self._save_metadata()
            logging.info(f"🧹 Cleaned up {removed_count} old cache files")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_files = len(self.metadata)
        total_size_mb = sum(info.get('file_size_mb', 0) for info in self.metadata.values())
        total_records = sum(info.get('record_count', 0) for info in self.metadata.values())
        
        sources = {}
        for info in self.metadata.values():
            source = info['source']
            if source not in sources:
                sources[source] = {'count': 0, 'records': 0, 'size_mb': 0}
            sources[source]['count'] += 1
            sources[source]['records'] += info.get('record_count', 0)
            sources[source]['size_mb'] += info.get('file_size_mb', 0)
        
        return {
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'total_records': total_records,
            'sources': sources,
            'cache_dir': str(self.cache_dir)
        }