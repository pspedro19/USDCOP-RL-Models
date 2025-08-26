"""
Order Execution Module
=====================
Handles market orders, limit orders, and order management with slippage tracking.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    volume: float
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic: int = 0


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_volume: Optional[float] = None
    commission: Optional[float] = None
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    slippage: Optional[float] = None


class OrderExecutor:
    """
    Handles order execution with different brokers and execution methods.
    """
    
    def __init__(self, connector=None, simulation_mode: bool = True):
        """
        Initialize order executor.
        
        Args:
            connector: MT5 or other broker connector
            simulation_mode: If True, simulate orders without real execution
        """
        self.connector = connector
        self.simulation_mode = simulation_mode
        self.orders = {}
        self.execution_history = []
        self.next_order_id = 1
        
    def execute_market_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> Dict[str, Any]:
        """
        Execute market order.
        
        Args:
            symbol: Trading symbol
            volume: Order volume
            order_type: 'BUY' or 'SELL'
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            comment: Order comment
            magic: Magic number for identification
            
        Returns:
            Execution result dictionary
        """
        try:
            if self.simulation_mode:
                return self._simulate_market_order(
                    symbol, volume, order_type, stop_loss, take_profit, comment, magic
                )
            
            if not self.connector:
                return {
                    'success': False,
                    'error_message': 'No connector available'
                }
            
            # Real execution with MT5
            import MetaTrader5 as mt5
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {
                    'success': False,
                    'error_message': f'Failed to get tick for {symbol}'
                }
            
            # Determine price
            price = tick.ask if order_type.upper() == 'BUY' else tick.bid
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss or 0,
                "tp": take_profit or 0,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error_message': f'Order failed: {result.retcode}',
                    'retcode': result.retcode
                }
            
            # Calculate slippage
            slippage = abs(result.price - price) if result.price else 0
            
            execution_result = {
                'success': True,
                'order_id': str(result.order),
                'executed_price': result.price,
                'executed_volume': result.volume,
                'commission': getattr(result, 'commission', 0),
                'execution_time': datetime.now(),
                'slippage': slippage,
                'retcode': result.retcode
            }
            
            # Store execution
            self.execution_history.append(execution_result)
            
            logger.info(f"Market order executed: {symbol} {order_type} {volume} @ {result.price}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def execute_limit_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> Dict[str, Any]:
        """
        Execute limit order.
        
        Args:
            symbol: Trading symbol
            volume: Order volume
            order_type: 'BUY' or 'SELL'
            price: Limit price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            comment: Order comment
            magic: Magic number
            
        Returns:
            Execution result dictionary
        """
        try:
            if self.simulation_mode:
                return self._simulate_limit_order(
                    symbol, volume, order_type, price, stop_loss, take_profit, comment, magic
                )
            
            if not self.connector:
                return {
                    'success': False,
                    'error_message': 'No connector available'
                }
            
            # Real execution with MT5
            import MetaTrader5 as mt5
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY_LIMIT if order_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT,
                "price": price,
                "sl": stop_loss or 0,
                "tp": take_profit or 0,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error_message': f'Limit order failed: {result.retcode}',
                    'retcode': result.retcode
                }
            
            # Store order
            order_info = {
                'order_id': str(result.order),
                'symbol': symbol,
                'volume': volume,
                'type': order_type,
                'price': price,
                'status': 'PENDING',
                'timestamp': datetime.now()
            }
            
            self.orders[str(result.order)] = order_info
            
            logger.info(f"Limit order placed: {symbol} {order_type} {volume} @ {price}")
            return {
                'success': True,
                'order_id': str(result.order),
                'status': 'PENDING'
            }
            
        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify existing order.
        
        Args:
            order_id: Order ID to modify
            new_price: New price (for pending orders)
            new_sl: New stop loss
            new_tp: New take profit
            
        Returns:
            Modification result
        """
        try:
            if self.simulation_mode:
                return self._simulate_order_modification(order_id, new_price, new_sl, new_tp)
            
            if not self.connector:
                return {
                    'success': False,
                    'error_message': 'No connector available'
                }
            
            # Real modification with MT5
            import MetaTrader5 as mt5
            
            # Get order info
            orders = mt5.orders_get(ticket=int(order_id))
            if not orders:
                return {
                    'success': False,
                    'error_message': f'Order {order_id} not found'
                }
            
            order = orders[0]
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": int(order_id),
                "price": new_price or order.price_open,
                "sl": new_sl or order.sl,
                "tp": new_tp or order.tp,
            }
            
            # Check request
            result = mt5.order_check(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error_message': f'Order check failed: {result.retcode}'
                }
            
            # Send modification
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error_message': f'Order modification failed: {result.retcode}'
                }
            
            logger.info(f"Order {order_id} modified successfully")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Order modification failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result
        """
        try:
            if self.simulation_mode:
                return self._simulate_order_cancellation(order_id)
            
            if not self.connector:
                return {
                    'success': False,
                    'error_message': 'No connector available'
                }
            
            # Real cancellation with MT5
            import MetaTrader5 as mt5
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error_message': f'Order cancellation failed: {result.retcode}'
                }
            
            # Remove from tracking
            if order_id in self.orders:
                del self.orders[order_id]
            
            logger.info(f"Order {order_id} cancelled successfully")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Check order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status information
        """
        try:
            if self.simulation_mode:
                return self._simulate_order_status(order_id)
            
            if not self.connector:
                return {
                    'success': False,
                    'error_message': 'No connector available'
                }
            
            # Real status check with MT5
            import MetaTrader5 as mt5
            
            # Check pending orders
            orders = mt5.orders_get(ticket=int(order_id))
            if orders:
                order = orders[0]
                return {
                    'order_id': order_id,
                    'state': 'PENDING',
                    'filled_volume': 0,
                    'remaining_volume': order.volume_current,
                    'price': order.price_open
                }
            
            # Check positions (filled orders)
            positions = mt5.positions_get(ticket=int(order_id))
            if positions:
                position = positions[0]
                return {
                    'order_id': order_id,
                    'state': 'FILLED',
                    'filled_volume': position.volume,
                    'remaining_volume': 0,
                    'price': position.price_open
                }
            
            # Check history
            deals = mt5.history_deals_get(ticket=int(order_id))
            if deals:
                deal = deals[0]
                return {
                    'order_id': order_id,
                    'state': 'FILLED',
                    'filled_volume': deal.volume,
                    'remaining_volume': 0,
                    'price': deal.price
                }
            
            return {
                'order_id': order_id,
                'state': 'NOT_FOUND'
            }
            
        except Exception as e:
            logger.error(f"Order status check failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions (emergency function).
        
        Returns:
            Result of mass position closure
        """
        try:
            if self.simulation_mode:
                return {'closed': 0, 'message': 'Simulation mode - no real positions'}
            
            if not self.connector:
                return {
                    'success': False,
                    'error_message': 'No connector available'
                }
            
            # Real position closure with MT5
            import MetaTrader5 as mt5
            
            positions = mt5.positions_get()
            if not positions:
                return {'closed': 0, 'message': 'No positions to close'}
            
            closed_count = 0
            for position in positions:
                try:
                    # Prepare close request
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "position": position.ticket,
                        "deviation": 20,
                        "magic": position.magic,
                        "comment": "Emergency close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to close position {position.ticket}: {e}")
            
            return {
                'closed': closed_count,
                'total_positions': len(positions),
                'message': f'Closed {closed_count} of {len(positions)} positions'
            }
            
        except Exception as e:
            logger.error(f"Mass position closure failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    # Simulation methods
    def _simulate_market_order(self, symbol, volume, order_type, sl, tp, comment, magic):
        """Simulate market order execution"""
        import random
        
        # Simulate price with small random variation
        base_price = 4000  # USDCOP base price
        price = base_price + random.uniform(-10, 10)
        
        # Simulate slippage
        slippage = random.uniform(0, 2)
        if order_type.upper() == 'BUY':
            executed_price = price + slippage
        else:
            executed_price = price - slippage
        
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        result = {
            'success': True,
            'order_id': order_id,
            'executed_price': executed_price,
            'executed_volume': volume,
            'commission': volume * 0.0001,  # Simulate commission
            'execution_time': datetime.now(),
            'slippage': slippage
        }
        
        self.execution_history.append(result)
        logger.info(f"SIMULATED: Market order {symbol} {order_type} {volume} @ {executed_price:.2f}")
        
        return result
    
    def _simulate_limit_order(self, symbol, volume, order_type, price, sl, tp, comment, magic):
        """Simulate limit order placement"""
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        order_info = {
            'order_id': order_id,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'price': price,
            'status': 'PENDING',
            'timestamp': datetime.now()
        }
        
        self.orders[order_id] = order_info
        logger.info(f"SIMULATED: Limit order {symbol} {order_type} {volume} @ {price:.2f}")
        
        return {
            'success': True,
            'order_id': order_id,
            'status': 'PENDING'
        }
    
    def _simulate_order_modification(self, order_id, new_price, new_sl, new_tp):
        """Simulate order modification"""
        if order_id in self.orders:
            if new_price:
                self.orders[order_id]['price'] = new_price
            logger.info(f"SIMULATED: Order {order_id} modified")
            return {'success': True}
        else:
            return {'success': False, 'error_message': 'Order not found'}
    
    def _simulate_order_cancellation(self, order_id):
        """Simulate order cancellation"""
        if order_id in self.orders:
            del self.orders[order_id]
            logger.info(f"SIMULATED: Order {order_id} cancelled")
            return {'success': True}
        else:
            return {'success': False, 'error_message': 'Order not found'}
    
    def _simulate_order_status(self, order_id):
        """Simulate order status check"""
        if order_id in self.orders:
            order = self.orders[order_id]
            return {
                'order_id': order_id,
                'state': 'PENDING',
                'filled_volume': 0,
                'remaining_volume': order['volume'],
                'price': order['price']
            }
        else:
            return {
                'order_id': order_id,
                'state': 'NOT_FOUND'
            }


class SlippageTracker:
    """
    Track and analyze order execution slippage.
    """
    
    def __init__(self):
        """Initialize slippage tracker"""
        self.executions = []
        
    def track_execution(
        self,
        expected_price: float,
        executed_price: float,
        volume: float,
        order_type: str
    ):
        """
        Track order execution for slippage analysis.
        
        Args:
            expected_price: Expected execution price
            executed_price: Actual execution price
            volume: Order volume
            order_type: Order type ('BUY' or 'SELL')
        """
        slippage = executed_price - expected_price
        if order_type.upper() == 'SELL':
            slippage = -slippage  # Negative slippage is bad for sells
            
        slippage_cost = abs(slippage) * volume
        
        execution = {
            'timestamp': datetime.now(),
            'expected_price': expected_price,
            'executed_price': executed_price,
            'volume': volume,
            'order_type': order_type,
            'slippage': slippage,
            'slippage_cost': slippage_cost
        }
        
        self.executions.append(execution)
        
        # Keep only last 1000 executions
        if len(self.executions) > 1000:
            self.executions = self.executions[-1000:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get slippage statistics"""
        if not self.executions:
            return {}
        
        slippages = [e['slippage'] for e in self.executions]
        costs = [e['slippage_cost'] for e in self.executions]
        
        return {
            'total_executions': len(self.executions),
            'average_slippage': sum(slippages) / len(slippages),
            'max_slippage': max(slippages),
            'min_slippage': min(slippages),
            'total_slippage_cost': sum(costs),
            'average_slippage_cost': sum(costs) / len(costs)
        }