"""
Signal Generator Service
=========================
Generates trading signals from model predictions and market data.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

from ..models.signal_schema import TradingSignal, SignalAction, SignalMetadata
from ..config import get_config
from .inference_service import InferenceService

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals from predictions and market data"""

    def __init__(self, inference_service: InferenceService):
        """
        Initialize signal generator.

        Args:
            inference_service: Inference service instance
        """
        self.inference_service = inference_service
        self.config = get_config()
        self.signals_generated = 0

    def generate_signal(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Optional[Dict[str, float]] = None
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from market data.

        Args:
            market_data: Current market data (OHLCV)
            technical_indicators: Optional pre-calculated technical indicators

        Returns:
            TradingSignal if valid signal generated, None otherwise
        """
        start_time = time.time()

        try:
            # Merge technical indicators into market data
            if technical_indicators:
                market_data.update(technical_indicators)

            # Get model prediction
            action, confidence, inference_latency = self.inference_service.predict_action(
                market_data
            )

            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                logger.debug(
                    f"Signal confidence {confidence:.3f} below threshold "
                    f"{self.config.confidence_threshold}"
                )
                # Force HOLD for low confidence
                action = SignalAction.HOLD

            # Get current price
            current_price = market_data.get('close', market_data.get('price', 0))

            if current_price <= 0:
                logger.error("Invalid current price")
                return None

            # Calculate risk levels (ATR-based)
            risk_levels = self._calculate_risk_levels(
                current_price=current_price,
                action=action,
                market_data=market_data
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(
                action=action,
                market_data=market_data,
                confidence=confidence
            )

            # Calculate position size
            position_size = self._calculate_position_size(
                confidence=confidence,
                market_data=market_data
            )

            # Calculate risk/reward ratio
            risk = abs(current_price - risk_levels['stop_loss'])
            reward = abs(risk_levels['take_profit'] - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0.0

            # Validate risk/reward
            if risk_reward_ratio < self.config.min_risk_reward_ratio:
                logger.debug(
                    f"Risk/reward ratio {risk_reward_ratio:.2f} below minimum "
                    f"{self.config.min_risk_reward_ratio}"
                )
                # Force HOLD for poor risk/reward
                action = SignalAction.HOLD

            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000

            # Create signal metadata
            metadata = SignalMetadata(
                atr=market_data.get('atr'),
                volatility=market_data.get('volatility'),
                trend_direction=self._determine_trend(market_data),
                market_regime=self._determine_market_regime(market_data),
                volume_profile='normal'
            )

            # Create signal
            signal = TradingSignal(
                symbol=market_data.get('symbol', 'USDCOP'),
                action=action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=risk_levels['stop_loss'],
                take_profit=risk_levels['take_profit'],
                position_size=position_size,
                risk_reward_ratio=risk_reward_ratio,
                model_version=self.config.model_version,
                model_type=self.config.model_type,
                reasoning=reasoning,
                technical_factors=self._extract_technical_factors(market_data),
                latency_ms=total_latency,
                metadata=metadata
            )

            self.signals_generated += 1
            logger.info(
                f"Signal generated: {action.value} @ {current_price:.2f} "
                f"(confidence={confidence:.3f}, latency={total_latency:.2f}ms)"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None

    def _calculate_risk_levels(
        self,
        current_price: float,
        action: SignalAction,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels using ATR.

        Args:
            current_price: Current market price
            action: Trading action
            market_data: Market data containing ATR

        Returns:
            Dictionary with 'stop_loss' and 'take_profit'
        """
        # Get ATR (Average True Range)
        atr = market_data.get('atr', current_price * 0.01)  # Default to 1% if not available

        # Calculate SL/TP distances
        sl_distance = atr * self.config.atr_multiplier_sl
        tp_distance = atr * self.config.atr_multiplier_tp

        if action == SignalAction.BUY:
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        elif action == SignalAction.SELL:
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        else:  # HOLD
            stop_loss = current_price * 0.98  # 2% buffer
            take_profit = current_price * 1.02

        return {
            'stop_loss': max(stop_loss, 0.01),  # Ensure positive
            'take_profit': max(take_profit, 0.01)
        }

    def _calculate_position_size(
        self,
        confidence: float,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate position size based on confidence and volatility.

        Args:
            confidence: Signal confidence (0-1)
            market_data: Market data

        Returns:
            Position size as percentage of capital (0-1)
        """
        # Base position size
        base_size = self.config.position_size_pct

        # Adjust by confidence
        adjusted_size = base_size * confidence

        # Adjust by volatility (if available)
        volatility = market_data.get('volatility', 1.0)
        if volatility > 2.0:  # High volatility
            adjusted_size *= 0.5
        elif volatility < 0.5:  # Low volatility
            adjusted_size *= 1.2

        # Cap at maximum
        return min(adjusted_size, self.config.max_position_size_pct)

    def _generate_reasoning(
        self,
        action: SignalAction,
        market_data: Dict[str, Any],
        confidence: float
    ) -> List[str]:
        """
        Generate human-readable reasoning for the signal.

        Args:
            action: Trading action
            market_data: Market data with indicators
            confidence: Signal confidence

        Returns:
            List of reasoning strings
        """
        reasoning = []

        # Model confidence
        reasoning.append(f"Model confidence: {confidence:.1%}")

        # RSI analysis
        rsi = market_data.get('rsi')
        if rsi is not None:
            if rsi < self.config.rsi_oversold:
                reasoning.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.config.rsi_overbought:
                reasoning.append(f"RSI overbought ({rsi:.1f})")
            else:
                reasoning.append(f"RSI neutral ({rsi:.1f})")

        # MACD analysis
        macd = market_data.get('macd')
        macd_signal = market_data.get('macd_signal')
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                reasoning.append("MACD bullish cross")
            else:
                reasoning.append("MACD bearish cross")

        # Trend analysis
        ema_20 = market_data.get('ema_20')
        ema_50 = market_data.get('ema_50')
        if ema_20 is not None and ema_50 is not None:
            if ema_20 > ema_50:
                reasoning.append("Uptrend (EMA 20 > EMA 50)")
            else:
                reasoning.append("Downtrend (EMA 20 < EMA 50)")

        # Bollinger Bands
        close = market_data.get('close')
        bb_upper = market_data.get('bb_upper')
        bb_lower = market_data.get('bb_lower')
        if all([close, bb_upper, bb_lower]):
            if close >= bb_upper:
                reasoning.append("Price at upper Bollinger Band")
            elif close <= bb_lower:
                reasoning.append("Price at lower Bollinger Band")

        # Volume analysis
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        if volume > avg_volume * 1.5:
            reasoning.append("High volume detected")
        elif volume < avg_volume * 0.5:
            reasoning.append("Low volume detected")

        return reasoning

    def _extract_technical_factors(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract technical indicator values for signal record.

        Args:
            market_data: Market data with indicators

        Returns:
            Dictionary of technical factors
        """
        factors = {}

        # Extract key indicators
        indicator_keys = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ema_20', 'ema_50', 'atr', 'volatility'
        ]

        for key in indicator_keys:
            if key in market_data:
                factors[key] = float(market_data[key])

        return factors

    def _determine_trend(self, market_data: Dict[str, Any]) -> str:
        """
        Determine current trend direction.

        Args:
            market_data: Market data with indicators

        Returns:
            Trend direction string
        """
        ema_20 = market_data.get('ema_20')
        ema_50 = market_data.get('ema_50')

        if ema_20 is not None and ema_50 is not None:
            if ema_20 > ema_50 * 1.001:  # 0.1% threshold
                return "uptrend"
            elif ema_20 < ema_50 * 0.999:
                return "downtrend"

        return "sideways"

    def _determine_market_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Determine current market regime.

        Args:
            market_data: Market data with indicators

        Returns:
            Market regime string
        """
        volatility = market_data.get('volatility', 1.0)
        atr = market_data.get('atr', 0)
        close = market_data.get('close', 1)

        # Calculate ATR percentage
        atr_pct = (atr / close * 100) if close > 0 else 0

        if atr_pct > 1.5:
            return "high_volatility"
        elif atr_pct < 0.5:
            return "low_volatility"
        else:
            return "normal"

    def get_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics"""
        return {
            'signals_generated': self.signals_generated,
            'confidence_threshold': self.config.confidence_threshold,
            'min_risk_reward': self.config.min_risk_reward_ratio,
            'position_size_pct': self.config.position_size_pct
        }

    def reset_stats(self):
        """Reset signal generation statistics"""
        self.signals_generated = 0
        logger.info("Signal generation statistics reset")
