"""
SAGA State Store
================
Dual persistence layer: Redis (primary) + SQLite (fallback)
"""

import os
import json
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .types import SagaStatus, StepStatus, SagaTransaction, SagaEvent, ResourceLock

logger = logging.getLogger(__name__)


class SagaStore:
    """Dual persistence store for SAGA state"""
    
    def __init__(self, redis_url: Optional[str] = None, db_path: str = "data/saga.db"):
        self.redis = None
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis = redis.from_url(redis_url)
                self.redis.ping()
                logger.info("Redis connection established for SAGA store")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to SQLite")
                self.redis = None
        
        # Initialize SQLite
        self._init_sqlite()
        logger.info(f"SAGA store initialized with {'Redis + SQLite' if self.redis else 'SQLite only'}")
    
    def _init_sqlite(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # SAGA transactions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS sagas(
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                context TEXT,
                metadata TEXT
            )
        """)
        
        # SAGA steps table
        c.execute("""
            CREATE TABLE IF NOT EXISTS saga_steps(
                saga_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                dependencies TEXT,
                compensation TEXT,
                critical BOOLEAN DEFAULT FALSE,
                metadata TEXT,
                FOREIGN KEY (saga_id) REFERENCES sagas(id)
            )
        """)
        
        # SAGA events table
        c.execute("""
            CREATE TABLE IF NOT EXISTS saga_events(
                saga_id TEXT NOT NULL,
                name TEXT NOT NULL,
                step TEXT,
                correlation_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                payload TEXT,
                source TEXT DEFAULT 'saga_system',
                severity TEXT DEFAULT 'info',
                FOREIGN KEY (saga_id) REFERENCES sagas(id)
            )
        """)
        
        # Resource locks table
        c.execute("""
            CREATE TABLE IF NOT EXISTS resource_locks(
                resource_id TEXT PRIMARY KEY,
                saga_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (saga_id) REFERENCES sagas(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def put_saga(self, saga: SagaTransaction) -> bool:
        """Store SAGA transaction"""
        try:
            if self.redis:
                # Store in Redis
                key = f"saga:{saga.id}"
                self.redis.setex(
                    key,
                    timedelta(hours=24),  # TTL for cleanup
                    json.dumps(saga.dict(), default=str)
                )
            
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT INTO sagas VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET 
                    status=excluded.status, 
                    updated_at=excluded.updated_at,
                    retry_count=excluded.retry_count
            """, (
                saga.id, saga.name, saga.correlation_id, saga.status.value,
                saga.started_at.isoformat(), saga.updated_at.isoformat(),
                saga.completed_at.isoformat() if saga.completed_at else None,
                saga.retry_count, saga.max_retries,
                json.dumps(saga.context), json.dumps(saga.metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store SAGA {saga.id}: {e}")
            return False
    
    def get_saga(self, saga_id: str) -> Optional[SagaTransaction]:
        """Retrieve SAGA transaction"""
        try:
            # Try Redis first
            if self.redis:
                key = f"saga:{saga_id}"
                data = self.redis.get(key)
                if data:
                    return SagaTransaction(**json.loads(data))
            
            # Fallback to SQLite
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("SELECT * FROM sagas WHERE id = ?", (saga_id,))
            row = c.fetchone()
            conn.close()
            
            if row:
                return SagaTransaction(
                    id=row[0], name=row[1], correlation_id=row[2],
                    status=SagaStatus(row[3]), started_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    retry_count=row[7], max_retries=row[8],
                    context=json.loads(row[9]) if row[9] else {},
                    metadata=json.loads(row[10]) if row[10] else {}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve SAGA {saga_id}: {e}")
            return None
    
    def update_saga_status(self, saga_id: str, status: SagaStatus, **kwargs) -> bool:
        """Update SAGA status and optional fields"""
        try:
            saga = self.get_saga(saga_id)
            if not saga:
                return False
            
            # Update fields
            saga.status = status
            saga.updated_at = datetime.utcnow()
            for key, value in kwargs.items():
                if hasattr(saga, key):
                    setattr(saga, key, value)
            
            # Store updated SAGA
            return self.put_saga(saga)
            
        except Exception as e:
            logger.error(f"Failed to update SAGA {saga_id} status: {e}")
            return False
    
    def append_step(self, saga_id: str, step_name: str, status: StepStatus, **kwargs) -> bool:
        """Append step status"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            now = datetime.utcnow().isoformat()
            c.execute("""
                INSERT INTO saga_steps VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """, (
                saga_id, step_name, status.value, now,
                kwargs.get('completed_at', None),
                kwargs.get('retry_count', 0),
                kwargs.get('max_retries', 3),
                json.dumps(kwargs.get('dependencies', [])),
                kwargs.get('compensation', None),
                kwargs.get('critical', False),
                json.dumps(kwargs.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to append step for SAGA {saga_id}: {e}")
            return False
    
    def append_event(self, event: SagaEvent) -> bool:
        """Append SAGA event"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT INTO saga_events VALUES(?,?,?,?,?,?,?,?)
            """, (
                event.saga_id, event.name, event.step, event.correlation_id,
                event.ts.isoformat(), json.dumps(event.payload),
                event.source, event.severity
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event for SAGA {event.saga_id}: {e}")
            return False
    
    def acquire_lock(self, resource_id: str, saga_id: str, ttl_seconds: int = 300) -> bool:
        """Acquire resource lock"""
        try:
            if self.redis:
                # Try Redis lock first
                lock_key = f"lock:{resource_id}"
                if self.redis.set(lock_key, saga_id, ex=ttl_seconds, nx=True):
                    return True
            
            # Fallback to SQLite
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if lock exists and is expired
            c.execute("""
                SELECT expires_at FROM resource_locks 
                WHERE resource_id = ? AND expires_at > ?
            """, (resource_id, datetime.utcnow().isoformat()))
            
            if c.fetchone():
                conn.close()
                return False  # Lock still valid
            
            # Acquire lock
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            c.execute("""
                INSERT OR REPLACE INTO resource_locks VALUES(?,?,?,?,?)
            """, (
                resource_id, saga_id, datetime.utcnow().isoformat(),
                expires_at.isoformat(), "{}"
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire lock for resource {resource_id}: {e}")
            return False
    
    def release_lock(self, resource_id: str, saga_id: str) -> bool:
        """Release resource lock"""
        try:
            if self.redis:
                # Release Redis lock
                lock_key = f"lock:{resource_id}"
                if self.redis.get(lock_key) == saga_id.encode():
                    self.redis.delete(lock_key)
            
            # Release SQLite lock
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                DELETE FROM resource_locks 
                WHERE resource_id = ? AND saga_id = ?
            """, (resource_id, saga_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to release lock for resource {resource_id}: {e}")
            return False
    
    def get_active_sagas(self) -> List[SagaTransaction]:
        """Get all active SAGA transactions"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT * FROM sagas 
                WHERE status IN (?, ?, ?)
                ORDER BY started_at DESC
            """, (SagaStatus.PENDING.value, SagaStatus.IN_PROGRESS.value, SagaStatus.COMPENSATING.value))
            
            rows = c.fetchall()
            conn.close()
            
            sagas = []
            for row in rows:
                try:
                    saga = SagaTransaction(
                        id=row[0], name=row[1], correlation_id=row[2],
                        status=SagaStatus(row[3]), started_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5]),
                        completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        retry_count=row[7], max_retries=row[8],
                        context=json.loads(row[9]) if row[9] else {},
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    sagas.append(saga)
                except Exception as e:
                    logger.warning(f"Failed to parse SAGA row: {e}")
                    continue
            
            return sagas
            
        except Exception as e:
            logger.error(f"Failed to get active SAGAs: {e}")
            return []
    
    def cleanup_expired_locks(self) -> int:
        """Clean up expired resource locks"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                DELETE FROM resource_locks 
                WHERE expires_at <= ?
            """, (datetime.utcnow().isoformat(),))
            
            deleted_count = c.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} expired locks")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired locks: {e}")
            return 0
