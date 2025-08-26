"""
Fallback Strategies
==================
Fallback strategies for circuit breaker patterns.
"""

import logging
from typing import Any, Callable, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Available fallback strategies"""
    CACHE = "cache"
    SIMULATOR = "simulator"
    DEGRADED = "degraded"
    RETRY = "retry"
    FAIL_FAST = "fail_fast"
    CUSTOM = "custom"


class FallbackManager:
    """Manages fallback strategies for circuit breakers"""
    
    def __init__(self):
        self._strategies: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}
        
        # Register default strategies
        self._register_default_strategies()
        
        logger.info("Fallback manager initialized")
    
    def _register_default_strategies(self):
        """Register default fallback strategies"""
        self.register_strategy(FallbackStrategy.CACHE, self._cache_strategy)
        self.register_strategy(FallbackStrategy.SIMULATOR, self._simulator_strategy)
        self.register_strategy(FallbackStrategy.DEGRADED, self._degraded_strategy)
        self.register_strategy(FallbackStrategy.RETRY, self._retry_strategy)
        self.register_strategy(FallbackStrategy.FAIL_FAST, self._fail_fast_strategy)
    
    def register_strategy(self, name: str, strategy: Callable):
        """Register a fallback strategy"""
        self._strategies[name] = strategy
        logger.info(f"Registered fallback strategy: {name}")
    
    def get_strategy(self, name: str) -> Optional[Callable]:
        """Get a fallback strategy by name"""
        return self._strategies.get(name)
    
    def execute(self, strategy_name: str, *args, **kwargs) -> Any:
        """Execute a registered strategy directly"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # If strategy is a simple function without context
        if strategy in [self._cache_strategy, self._simulator_strategy, 
                        self._degraded_strategy, self._retry_strategy, 
                        self._fail_fast_strategy]:
            # These are internal strategies that need context
            return self.execute_fallback(strategy_name, {}, lambda: None)
        else:
            # User-registered simple functions
            return strategy(*args, **kwargs)
    
    def execute_fallback(self, strategy_name: str, context: Dict[str, Any], 
                        original_func: Callable, *args, **kwargs) -> Any:
        """Execute a fallback strategy"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            logger.warning(f"Fallback strategy '{strategy_name}' not found, using fail_fast")
            strategy = self._fail_fast_strategy
        
        try:
            logger.info(f"Executing fallback strategy: {strategy_name}")
            return strategy(context, original_func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback strategy '{strategy_name}' failed: {e}")
            # Last resort: fail fast
            return self._fail_fast_strategy(context, original_func, *args, **kwargs)
    
    def _cache_strategy(self, context: Dict[str, Any], original_func: Callable, 
                       *args, **kwargs) -> Any:
        """Cache-based fallback strategy"""
        cache_key = f"{original_func.__name__}:{hash(str(args) + str(kwargs))}"
        
        if cache_key in self._cache:
            logger.info(f"Returning cached result for {original_func.__name__}")
            return self._cache[cache_key]
        
        # No cache available, try to execute with reduced expectations
        logger.warning(f"No cache available for {original_func.__name__}, using degraded mode")
        return self._degraded_strategy(context, original_func, *args, **kwargs)
    
    def _simulator_strategy(self, context: Dict[str, Any], original_func: Callable, 
                           *args, **kwargs) -> Any:
        """Simulator-based fallback strategy"""
        logger.info(f"Using simulator fallback for {original_func.__name__}")
        
        # This would typically return simulated data
        # For now, return a placeholder
        if "get_historical_data" in original_func.__name__:
            return self._get_simulated_historical_data(*args, **kwargs)
        elif "get_realtime_data" in original_func.__name__:
            return self._get_simulated_realtime_data(*args, **kwargs)
        else:
            return self._get_simulated_generic_data(*args, **kwargs)
    
    def _degraded_strategy(self, context: Dict[str, Any], original_func: Callable, 
                          *args, **kwargs) -> Any:
        """Degraded functionality fallback strategy"""
        logger.info(f"Using degraded fallback for {original_func.__name__}")
        
        # Return minimal functionality
        if "get_historical_data" in original_func.__name__:
            return self._get_degraded_historical_data(*args, **kwargs)
        else:
            return self._get_degraded_generic_data(*args, **kwargs)
    
    def _retry_strategy(self, context: Dict[str, Any], original_func: Callable, 
                       *args, **kwargs) -> Any:
        """Retry-based fallback strategy"""
        max_retries = context.get("max_retries", 3)
        retry_delay = context.get("retry_delay", 1.0)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Retry attempt {attempt + 1} for {original_func.__name__}")
                return original_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All retries failed, fall back to simulator
        logger.error(f"All retry attempts failed for {original_func.__name__}, using simulator")
        return self._simulator_strategy(context, original_func, *args, **kwargs)
    
    def _fail_fast_strategy(self, context: Dict[str, Any], original_func: Callable, 
                           *args, **kwargs) -> Any:
        """Fail-fast fallback strategy"""
        logger.error(f"Fail-fast fallback for {original_func.__name__}")
        raise RuntimeError(f"Service {original_func.__name__} unavailable and no fallback available")
    
    def _get_simulated_historical_data(self, *args, **kwargs):
        """Get simulated historical data"""
        import pandas as pd
        import numpy as np
        
        # Create simulated OHLC data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = {
            'open': np.random.uniform(3800, 4200, 100),
            'high': np.random.uniform(3800, 4200, 100),
            'low': np.random.uniform(3800, 4200, 100),
            'close': np.random.uniform(3800, 4200, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }
        
        df = pd.DataFrame(data, index=dates)
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _get_simulated_realtime_data(self, *args, **kwargs):
        """Get simulated realtime data"""
        import pandas as pd
        import numpy as np
        
        # Create single simulated bar
        data = {
            'open': np.random.uniform(3800, 4200),
            'high': np.random.uniform(3800, 4200),
            'low': np.random.uniform(3800, 4200),
            'close': np.random.uniform(3800, 4200),
            'volume': np.random.uniform(1000, 5000),
            'timestamp': pd.Timestamp.now()
        }
        
        data['high'] = max(data['open'], data['high'], data['low'], data['close'])
        data['low'] = min(data['open'], data['high'], data['low'], data['close'])
        
        return data
    
    def _get_simulated_generic_data(self, *args, **kwargs):
        """Get generic simulated data"""
        return {"status": "simulated", "data": "fallback_data"}
    
    def _get_degraded_historical_data(self, *args, **kwargs):
        """Get degraded historical data (minimal)"""
        import pandas as pd
        
        # Return minimal data structure
        return pd.DataFrame({
            'open': [4000],
            'high': [4000],
            'low': [4000],
            'close': [4000],
            'volume': [0]
        }, index=[pd.Timestamp.now()])
    
    def _get_degraded_generic_data(self, *args, **kwargs):
        """Get degraded generic data"""
        return {"status": "degraded", "data": "minimal_data"}
    
    def set_cache(self, key: str, value: Any):
        """Set a cache value"""
        self._cache[key] = value
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cache value"""
        return self._cache.get(key)
    
    def clear_cache(self):
        """Clear all cache"""
        self._cache.clear()
        logger.info("Cache cleared")


# Global instance
_global_fallback_manager: Optional[FallbackManager] = None


def get_global_fallback_manager() -> FallbackManager:
    """Get the global fallback manager"""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = FallbackManager()
    return _global_fallback_manager


def register_fallback_strategy(name: str, strategy: Callable) -> None:
    """Register a fallback strategy in the global manager"""
    get_global_fallback_manager().register_strategy(name, strategy)


def execute_fallback(strategy_name: str, context: Dict[str, Any], 
                    original_func: Callable, *args, **kwargs) -> Any:
    """Execute a fallback strategy using the global manager"""
    return get_global_fallback_manager().execute_fallback(
        strategy_name, context, original_func, *args, **kwargs
    )
