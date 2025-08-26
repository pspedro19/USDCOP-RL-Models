"""
Data Sources Module for USDCOP Trading Pipeline
"""

# Try to import each client separately to avoid cascade failures
try:
    from .mt5_client import MT5DataClient
except ImportError as e:
    import logging
    logging.warning(f"MT5DataClient not available: {e}")
    MT5DataClient = None

try:
    from .twelvedata_client import TwelveDataClient
except ImportError as e:
    import logging
    logging.warning(f"TwelveDataClient not available: {e}")
    TwelveDataClient = None

__all__ = ['MT5DataClient', 'TwelveDataClient']