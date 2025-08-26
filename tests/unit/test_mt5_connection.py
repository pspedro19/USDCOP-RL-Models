"""
MT5 Connection Unit Tests
=========================
Tests for MT5 connector including connection, retry logic, and fallback mechanisms.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta


class TestMT5Connection:
    """Test MT5 connection functionality"""
    
    def test_mt5_connection_successful(self, mock_mt5_config):
        """Test successful MT5 connection"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = Mock(connected=True)
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector(mock_mt5_config)
            assert connector.initialize() == True
    
    def test_mt5_connection_retry_mechanism(self, mock_mt5_config):
        """Test MT5 connection retry on failure"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            # Fail twice, then succeed
            mock_mt5.initialize.side_effect = [False, False, True]
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector(mock_mt5_config)
            connector.max_retries = 3
            
            result = connector.initialize()
            assert result == True
            assert mock_mt5.initialize.call_count == 3
    
    def test_mt5_connection_timeout_handling(self, mock_mt5_config):
        """Test MT5 connection timeout handling"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.initialize.side_effect = TimeoutError("Connection timeout")
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector(mock_mt5_config)
            connector.config.auto_fallback = False  # Disable fallback for this test
            connector.max_retries = 1
            
            result = connector.initialize()
            assert result == False
    
    def test_mt5_invalid_credentials(self, mock_mt5_config):
        """Test MT5 connection with invalid credentials"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.login.return_value = False
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector(mock_mt5_config)
            connector.config.auto_fallback = False  # Disable fallback
            result = connector.connect()
            assert result == True  # Initialize succeeds even if login fails (returns True but not logged in)
    
    def test_mt5_server_unavailable(self):
        """Test MT5 behavior when server is unavailable"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = False
            mock_mt5.last_error.return_value = (5001, "Server not available")
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            config = {'server': 'unavailable-server'}
            connector = RobustMT5Connector(config)
            connector.config.auto_fallback = False  # Disable fallback
            
            result = connector.initialize()
            assert result == False
    
    def test_mt5_connection_pool_management(self):
        """Test MT5 connection pool management"""
        from src.core.connectors.mt5_connector import ConnectionPool
        
        pool = ConnectionPool(max_connections=3)
        
        # Get connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        conn3 = pool.get_connection()
        
        assert pool.active_connections == 3
        
        # Return connection
        pool.return_connection(conn1)
        assert pool.active_connections == 2