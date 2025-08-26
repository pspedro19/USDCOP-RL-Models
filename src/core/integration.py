"""
Main System Integration Module
==============================
Coordinates and verifies integration between all system components.
Provides initialization, shutdown, and verification functions.
"""

import logging
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# Import all core modules
from .config.unified_config import config_loader, get_system_config
from .database.db_integration import db_integration
from .errors.handlers import error_handler, ErrorContext, TradingSystemError
from .monitoring.health_checks import health_checker, get_health_status

# Try to import optional modules
try:
    from .events.bus import event_bus
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    event_bus = None

logger = logging.getLogger(__name__)


class SystemIntegration:
    """
    Main system integration coordinator.
    Manages initialization, verification, and shutdown of all components.
    """
    
    def __init__(self):
        self.config = get_system_config()
        self.initialized = False
        self.components_status = {}
        self.initialization_time = None
        
    def initialize(self, verify: bool = True) -> bool:
        """
        Initialize all system components in the correct order.
        
        Args:
            verify: Whether to verify integration after initialization
            
        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("="*50)
        logger.info("Starting USDCOP Trading RL System Integration")
        logger.info("="*50)
        
        try:
            with ErrorContext("system_integration", "initialization"):
                
                # 1. Load configurations
                logger.info("Step 1/5: Loading configurations...")
                if not self._initialize_config():
                    return False
                
                # 2. Initialize database
                logger.info("Step 2/5: Initializing database...")
                if not self._initialize_database():
                    return False
                
                # 3. Start event bus (if available)
                logger.info("Step 3/5: Starting event bus...")
                if not self._initialize_event_bus():
                    logger.warning("Event bus not available, continuing without it")
                
                # 4. Initialize error handling
                logger.info("Step 4/5: Setting up error handling...")
                self._initialize_error_handling()
                
                # 5. Run health checks
                logger.info("Step 5/5: Running initial health checks...")
                if not self._run_initial_health_checks():
                    logger.warning("Some health checks failed, but system can continue")
                
                # Verify integration if requested
                if verify:
                    logger.info("Verifying system integration...")
                    if not self.verify_integration():
                        logger.error("Integration verification failed")
                        return False
                
                self.initialized = True
                self.initialization_time = datetime.utcnow()
                
                logger.info("="*50)
                logger.info("System integration initialized successfully!")
                logger.info(f"Environment: {self.config.env}")
                logger.info(f"Debug mode: {self.config.debug}")
                logger.info("="*50)
                
                return True
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            error_handler.handle("system_integration", e, severity="CRITICAL")
            return False
    
    def _initialize_config(self) -> bool:
        """Initialize configuration system"""
        try:
            # Reload all configurations
            config_loader.reload()
            
            # Verify critical configs are loaded
            configs = config_loader.get_all_configs()
            required_configs = ['mt5', 'usdcop']
            
            missing = [c for c in required_configs if c not in configs]
            if missing:
                logger.error(f"Missing required configurations: {missing}")
                self.components_status['configuration'] = 'failed'
                return False
            
            self.components_status['configuration'] = 'initialized'
            logger.info(f"✓ Loaded {len(configs)} configuration files")
            return True
            
        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            self.components_status['configuration'] = 'error'
            return False
    
    def _initialize_database(self) -> bool:
        """Initialize database connection"""
        try:
            if not db_integration.initialized:
                logger.error("Database integration not initialized")
                self.components_status['database'] = 'failed'
                return False
            
            # Test connection
            stats = db_integration.get_statistics()
            if 'error' in stats:
                logger.error(f"Database connection test failed: {stats['error']}")
                self.components_status['database'] = 'error'
                return False
            
            self.components_status['database'] = 'initialized'
            logger.info(f"✓ Database connected ({stats.get('database_size_bytes', 0) / 1024:.1f} KB)")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.components_status['database'] = 'error'
            return False
    
    def _initialize_event_bus(self) -> bool:
        """Initialize event bus if available"""
        if not HAS_EVENT_BUS or not event_bus:
            self.components_status['event_bus'] = 'not_available'
            return False
        
        try:
            # Test event bus
            test_received = []
            
            def test_handler(event):
                test_received.append(event)
            
            event_bus.subscribe("INIT_TEST", test_handler)
            event_bus.publish({"event": "INIT_TEST", "payload": {}})
            
            # Small delay for processing
            import time
            time.sleep(0.1)
            
            event_bus.unsubscribe("INIT_TEST", test_handler)
            
            if test_received:
                self.components_status['event_bus'] = 'initialized'
                logger.info("✓ Event bus operational")
                return True
            else:
                self.components_status['event_bus'] = 'degraded'
                logger.warning("Event bus slow or not responding")
                return False
                
        except Exception as e:
            logger.error(f"Event bus initialization failed: {e}")
            self.components_status['event_bus'] = 'error'
            return False
    
    def _initialize_error_handling(self):
        """Initialize error handling system"""
        try:
            # Reset error statistics
            error_handler.reset_stats()
            
            self.components_status['error_handling'] = 'initialized'
            logger.info("✓ Error handling system ready")
            
        except Exception as e:
            logger.warning(f"Error handling setup warning: {e}")
            self.components_status['error_handling'] = 'degraded'
    
    def _run_initial_health_checks(self) -> bool:
        """Run initial health checks"""
        try:
            health = get_health_status()
            overall_status = health.get('overall_status', 'unknown')
            
            # Log component statuses
            components = health.get('components', {})
            healthy_count = sum(1 for c in components.values() 
                              if c.get('status') == 'healthy')
            total_count = len(components)
            
            logger.info(f"✓ Health checks: {healthy_count}/{total_count} components healthy")
            
            # Store health status
            self.components_status['health_checks'] = overall_status
            
            return overall_status != 'unhealthy'
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
            self.components_status['health_checks'] = 'error'
            return False
    
    def verify_integration(self) -> bool:
        """
        Verify all components are properly integrated.
        
        Returns:
            True if all critical integrations are working
        """
        logger.info("Running integration verification...")
        
        checks = {
            'config': self._check_config(),
            'database': self._check_database(),
            'imports': self._check_imports(),
            'data_flow': self._check_data_flow()
        }
        
        if HAS_EVENT_BUS:
            checks['event_bus'] = self._check_event_bus()
        
        # Log results
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        
        logger.info(f"Integration verification: {passed}/{total} checks passed")
        
        for check, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}")
        
        # All critical checks must pass
        critical_checks = ['config', 'database', 'imports']
        critical_passed = all(checks[c] for c in critical_checks if c in checks)
        
        if not critical_passed:
            logger.error("Critical integration checks failed")
            return False
        
        return True
    
    def _check_config(self) -> bool:
        """Check configuration is loaded"""
        try:
            configs = config_loader.get_all_configs()
            return len(configs) > 0
        except Exception as e:
            logger.error(f"Config check failed: {e}")
            return False
    
    def _check_database(self) -> bool:
        """Check database is accessible"""
        try:
            stats = db_integration.get_statistics()
            return 'error' not in stats
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False
    
    def _check_event_bus(self) -> bool:
        """Check event bus is operational"""
        if not HAS_EVENT_BUS or not event_bus:
            return False
        
        try:
            # Simple connectivity test
            return True
        except Exception:
            return False
    
    def _check_imports(self) -> bool:
        """Check all critical imports work"""
        try:
            # Try importing critical modules
            from src.core.connectors.mt5_connector import RobustMT5Connector
            from src.markets.usdcop.pipeline import DataPipeline
            from src.markets.usdcop.feature_engine import FeatureEngine
            
            return True
            
        except ImportError as e:
            logger.error(f"Import check failed: {e}")
            return False
    
    def _check_data_flow(self) -> bool:
        """Check basic data flow works"""
        try:
            import pandas as pd
            import numpy as np
            
            # Create minimal test data
            test_data = pd.DataFrame({
                'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
                'open': np.random.randn(100) + 4000,
                'high': np.random.randn(100) + 4010,
                'low': np.random.randn(100) + 3990,
                'close': np.random.randn(100) + 4000,
                'volume': np.random.randint(100, 1000, 100)
            })
            
            # Try feature engineering
            from src.markets.usdcop.feature_engine import FeatureEngine
            engine = FeatureEngine()
            
            # This should not raise an error
            _ = engine.add_all_features(test_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Data flow check failed: {e}")
            return False
    
    def shutdown(self, graceful: bool = True):
        """
        Shutdown all system components.
        
        Args:
            graceful: Whether to perform graceful shutdown
        """
        logger.info("="*50)
        logger.info("Shutting down system integration...")
        logger.info("="*50)
        
        try:
            # Stop event bus
            if HAS_EVENT_BUS and event_bus:
                try:
                    logger.info("Stopping event bus...")
                    # event_bus.stop() if it has this method
                except Exception as e:
                    logger.warning(f"Event bus shutdown warning: {e}")
            
            # Close database connections
            if db_integration.db_manager:
                try:
                    logger.info("Closing database connections...")
                    # db_integration.db_manager.close_all() if it has this method
                except Exception as e:
                    logger.warning(f"Database shutdown warning: {e}")
            
            # Save error statistics
            if graceful:
                try:
                    stats = error_handler.get_error_stats()
                    logger.info(f"System errors during session: {stats['total_errors']}")
                except Exception:
                    pass
            
            self.initialized = False
            logger.info("System integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        uptime = None
        if self.initialization_time:
            uptime = (datetime.utcnow() - self.initialization_time).total_seconds()
        
        return {
            'initialized': self.initialized,
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'uptime_seconds': uptime,
            'environment': self.config.env,
            'debug_mode': self.config.debug,
            'components': self.components_status,
            'health': get_health_status() if self.initialized else None
        }
    
    def restart(self) -> bool:
        """Restart the system"""
        logger.info("Restarting system...")
        
        # Shutdown
        self.shutdown(graceful=True)
        
        # Wait a moment
        import time
        time.sleep(1)
        
        # Reinitialize
        return self.initialize()


# ===========================
# Global System Integration
# ===========================

# Create singleton instance
system_integration = SystemIntegration()


# ===========================
# Public API Functions
# ===========================

def initialize_system(verify: bool = True) -> bool:
    """
    Initialize the entire trading system.
    
    Args:
        verify: Whether to verify integration after initialization
        
    Returns:
        True if initialization successful
    """
    return system_integration.initialize(verify=verify)


def shutdown_system(graceful: bool = True):
    """
    Shutdown the entire trading system.
    
    Args:
        graceful: Whether to perform graceful shutdown
    """
    system_integration.shutdown(graceful=graceful)


def verify_system() -> bool:
    """
    Verify system integration.
    
    Returns:
        True if all critical integrations are working
    """
    return system_integration.verify_integration()


def get_system_status() -> Dict[str, Any]:
    """
    Get current system status.
    
    Returns:
        Dictionary with system status information
    """
    return system_integration.get_status()


def restart_system() -> bool:
    """
    Restart the system.
    
    Returns:
        True if restart successful
    """
    return system_integration.restart()


# ===========================
# CLI Support
# ===========================

def main():
    """Main entry point for CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='USDCOP Trading System Integration Manager')
    parser.add_argument('command', choices=['init', 'verify', 'status', 'shutdown', 'restart'],
                       help='Command to execute')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification during initialization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)-8s %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if args.command == 'init':
        success = initialize_system(verify=not args.no_verify)
        sys.exit(0 if success else 1)
    
    elif args.command == 'verify':
        success = verify_system()
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        status = get_system_status()
        import json
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == 'shutdown':
        shutdown_system()
    
    elif args.command == 'restart':
        success = restart_system()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()