"""
Technical Indicators Module
============================
Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from price data"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of closing prices
            period: RSI period (default: 14)

        Returns:
            Current RSI value (0-100)
        """
        try:
            if len(prices) < period + 1:
                logger.warning(f"Not enough data for RSI calculation: {len(prices)} < {period + 1}")
                return 50.0  # Neutral

            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram'
        """
        try:
            if len(prices) < slow + signal:
                logger.warning(f"Not enough data for MACD calculation")
                return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}

            # Calculate EMAs
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()

            # Calculate histogram
            histogram = macd_line - signal_line

            return {
                'macd': float(macd_line.iloc[-1]),
                'macd_signal': float(signal_line.iloc[-1]),
                'macd_histogram': float(histogram.iloc[-1])
            }

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of closing prices
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)

        Returns:
            Dictionary with 'bb_upper', 'bb_middle', 'bb_lower'
        """
        try:
            if len(prices) < period:
                logger.warning(f"Not enough data for Bollinger Bands calculation")
                current_price = float(prices.iloc[-1])
                return {
                    'bb_upper': current_price * 1.02,
                    'bb_middle': current_price,
                    'bb_lower': current_price * 0.98
                }

            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()

            # Calculate standard deviation
            std = prices.rolling(window=period).std()

            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            return {
                'bb_upper': float(upper_band.iloc[-1]),
                'bb_middle': float(middle_band.iloc[-1]),
                'bb_lower': float(lower_band.iloc[-1])
            }

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = float(prices.iloc[-1])
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98
            }

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> float:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            prices: Series of closing prices
            period: EMA period

        Returns:
            Current EMA value
        """
        try:
            if len(prices) < period:
                return float(prices.mean())

            ema = prices.ewm(span=period, adjust=False).mean()
            return float(ema.iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return float(prices.iloc[-1])

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float:
        """
        Calculate Average True Range (ATR).

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period (default: 14)

        Returns:
            Current ATR value
        """
        try:
            if len(high) < period + 1:
                # Fallback to simple range
                return float((high.iloc[-1] - low.iloc[-1]))

            # Calculate True Range
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate ATR as moving average of True Range
            atr = true_range.rolling(window=period).mean()

            return float(atr.iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return float(high.iloc[-1] - low.iloc[-1])

    @staticmethod
    def calculate_volatility(returns: pd.Series, period: int = 20) -> float:
        """
        Calculate historical volatility.

        Args:
            returns: Series of price returns
            period: Lookback period (default: 20)

        Returns:
            Annualized volatility
        """
        try:
            if len(returns) < period:
                return 1.0

            # Calculate rolling standard deviation
            volatility = returns.rolling(window=period).std()

            # Annualize (assuming 252 trading days)
            annualized_vol = float(volatility.iloc[-1]) * np.sqrt(252)

            return annualized_vol

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 1.0

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_short: int = 20,
        ema_long: int = 50,
        atr_period: int = 14
    ) -> Dict[str, float]:
        """
        Calculate all technical indicators at once.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume'
            rsi_period: RSI period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands std dev
            ema_short: Short EMA period
            ema_long: Long EMA period
            atr_period: ATR period

        Returns:
            Dictionary with all calculated indicators
        """
        indicators = {}

        try:
            # Ensure we have required columns
            required_cols = ['close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Have: {df.columns}")
                return indicators

            close = df['close']

            # RSI
            indicators['rsi'] = TechnicalIndicators.calculate_rsi(close, rsi_period)

            # MACD
            macd_values = TechnicalIndicators.calculate_macd(
                close, macd_fast, macd_slow, macd_signal
            )
            indicators.update(macd_values)

            # Bollinger Bands
            bb_values = TechnicalIndicators.calculate_bollinger_bands(
                close, bb_period, bb_std
            )
            indicators.update(bb_values)

            # EMAs
            indicators['ema_20'] = TechnicalIndicators.calculate_ema(close, ema_short)
            indicators['ema_50'] = TechnicalIndicators.calculate_ema(close, ema_long)

            # ATR (if we have OHLC data)
            if all(col in df.columns for col in ['high', 'low']):
                indicators['atr'] = TechnicalIndicators.calculate_atr(
                    df['high'], df['low'], close, atr_period
                )

            # Volatility
            returns = close.pct_change()
            indicators['volatility'] = TechnicalIndicators.calculate_volatility(returns)

            # Volume analysis (if available)
            if 'volume' in df.columns:
                indicators['avg_volume'] = float(df['volume'].tail(20).mean())
                indicators['volume'] = float(df['volume'].iloc[-1])

            logger.debug(f"Calculated {len(indicators)} technical indicators")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators
