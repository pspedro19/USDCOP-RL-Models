"""
State Persistence
================
Save and restore system state during shutdown and restart.
"""
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class StatePersistence:
    """Manages persistence and restoration of system state."""
    
    def __init__(self, storage_path: str = "state", redis_client=None):
        """Initialize state persistence."""
        self.storage_path = Path(storage_path)
        self.redis_client = redis_client
        self.state_file = self.storage_path / "system_state.json"
        self.backup_file = self.storage_path / "system_state.backup.json"
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def save_state(self, state_data: Dict[str, Any], service_name: str = "system") -> bool:
        """
        Save system state to persistent storage.
        
        Args:
            state_data: State data to persist
            service_name: Name of the service saving state
            
        Returns:
            True if state was saved successfully
        """
        try:
            # Add metadata
            state_with_metadata = {
                'service': service_name,
                'timestamp': datetime.utcnow().isoformat() + "Z",
                'version': '1.0',
                'data': state_data
            }
            
            # Save to Redis if available
            if self.redis_client:
                await self._save_to_redis(service_name, state_with_metadata)
            
            # Save to file
            await self._save_to_file(state_with_metadata)
            
            logger.info(f"State saved successfully for service: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state for service {service_name}: {e}")
            return False
    
    async def load_state(self, service_name: str = "system") -> Optional[Dict[str, Any]]:
        """
        Load system state from persistent storage.
        
        Args:
            service_name: Name of the service to load state for
            
        Returns:
            State data if found, None otherwise
        """
        try:
            # Try Redis first
            if self.redis_client:
                state = await self._load_from_redis(service_name)
                if state:
                    logger.info(f"State loaded from Redis for service: {service_name}")
                    return state
            
            # Fall back to file
            state = await self._load_from_file(service_name)
            if state:
                logger.info(f"State loaded from file for service: {service_name}")
                return state
            
            logger.warning(f"No state found for service: {service_name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load state for service {service_name}: {e}")
            return None
    
    async def _save_to_redis(self, service_name: str, state_data: Dict[str, Any]):
        """Save state to Redis."""
        try:
            key = f"state:{service_name}"
            await self.redis_client.setex(
                key,
                7 * 24 * 60 * 60,  # 7 days TTL
                json.dumps(state_data, default=str)
            )
            
            # Add to state index
            index_key = "state:index"
            await self.redis_client.zadd(
                index_key,
                {service_name: time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to save state to Redis: {e}")
            raise
    
    async def _save_to_file(self, state_data: Dict[str, Any]):
        """Save state to file with backup."""
        try:
            # Create backup of existing state file
            if self.state_file.exists():
                import shutil
                shutil.copy2(self.state_file, self.backup_file)
            
            # Write new state file
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to save state to file: {e}")
            raise
    
    async def _load_from_redis(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Load state from Redis."""
        try:
            key = f"state:{service_name}"
            state_json = await self.redis_client.get(key)
            
            if state_json:
                return json.loads(state_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load state from Redis: {e}")
            return None
    
    async def _load_from_file(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Load state from file."""
        try:
            if not self.state_file.exists():
                return None
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Check if this is the state for the requested service
            if state_data.get('service') == service_name:
                return state_data.get('data', {})
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load state from file: {e}")
            return None
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of all persisted states."""
        try:
            summary = {}
            
            # Get Redis states
            if self.redis_client:
                index_key = "state:index"
                services = await self.redis_client.zrange(index_key, 0, -1)
                
                for service in services:
                    state = await self._load_from_redis(service)
                    if state:
                        summary[service] = {
                            'timestamp': state.get('timestamp'),
                            'version': state.get('version'),
                            'source': 'redis'
                        }
            
            # Get file state
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    service = state_data.get('service', 'unknown')
                    summary[service] = {
                        'timestamp': state_data.get('timestamp'),
                        'version': state_data.get('version'),
                        'source': 'file'
                    }
                except Exception as e:
                    logger.warning(f"Failed to read file state: {e}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get state summary: {e}")
            return {}
    
    async def cleanup_old_states(self, max_age_days: int = 7):
        """Clean up old state data."""
        try:
            if not self.redis_client:
                return
            
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            index_key = "state:index"
            
            # Get old services
            old_services = await self.redis_client.zrangebyscore(
                index_key, 0, cutoff_time
            )
            
            # Remove old states
            cleaned_count = 0
            for service in old_services:
                key = f"state:{service}"
                await self.redis_client.delete(key)
                await self.redis_client.zrem(index_key, service)
                cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old state entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
    
    def get_state_file_path(self) -> Path:
        """Get the path to the state file."""
        return self.state_file
    
    def get_backup_file_path(self) -> Path:
        """Get the path to the backup file."""
        return self.backup_file

# Global state persistence instance
_global_state_persistence: Optional[StatePersistence] = None

def get_state_persistence(storage_path: str = None) -> StatePersistence:
    """Get the global state persistence instance."""
    global _global_state_persistence
    
    if _global_state_persistence is None:
        storage_path = storage_path or "state"
        _global_state_persistence = StatePersistence(storage_path)
    
    return _global_state_persistence

async def save_system_state(state_data: Dict[str, Any], service_name: str = "system") -> bool:
    """Save system state using the global instance."""
    persistence = get_state_persistence()
    return await persistence.save_state(state_data, service_name)

async def load_system_state(service_name: str = "system") -> Optional[Dict[str, Any]]:
    """Load system state using the global instance."""
    persistence = get_state_persistence()
    return await persistence.load_state(service_name)
