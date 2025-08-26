"""
Signal Generator Module
======================
Generates trading signals from model predictions with confidence thresholds and ensemble methods.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    """Signal generation result with metadata"""
    signal: Signal
    confidence: float
    price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    metadata: Optional[Dict[str, Any]] = None


class SignalGenerator:
    """
    Generates trading signals from model predictions.
    
    Converts numerical model outputs to actionable trading signals
    with confidence thresholds and filtering.
    """
    
    def __init__(
        self,
        model: Any = None,
        confidence_threshold: float = 0.6,
        action_mapping: Optional[Dict[int, Signal]] = None
    ):
        """
        Initialize signal generator.
        
        Args:
            model: Trained model for predictions
            confidence_threshold: Minimum confidence for signal generation
            action_mapping: Custom mapping from action integers to signals
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.action_mapping = action_mapping or {
            0: Signal.BUY,
            1: Signal.SELL, 
            2: Signal.HOLD
        }
        
        self.signal_history = []
        
    def generate_signal(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        current_price: Optional[float] = None
    ) -> str:
        """
        Generate trading signal from current market data.
        
        Args:
            data: Market data or features for prediction
            current_price: Current market price
            
        Returns:
            Signal string ('BUY', 'SELL', 'HOLD')
        """
        try:
            if self.model is None:
                logger.warning("No model available, returning HOLD")
                return Signal.HOLD.value
                
            # Prepare input data
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return Signal.HOLD.value
                features = data.iloc[-1].values if len(data.shape) == 2 else data.values
            else:
                features = data
                
            # Get model prediction
            action, info = self.model.predict(features)
            
            # Extract action if it's an array
            if hasattr(action, '__len__') and len(action) > 0:
                action = action[0]
                
            # Check confidence if available
            confidence = info.get('confidence', 1.0) if info else 1.0
            
            if confidence < self.confidence_threshold:
                logger.debug(f"Low confidence {confidence:.3f}, returning HOLD")
                return Signal.HOLD.value
                
            # Map action to signal
            signal = self.action_mapping.get(int(action), Signal.HOLD)
            
            # Store signal history
            result = SignalResult(
                signal=signal,
                confidence=confidence,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                metadata={'action': int(action), 'model_info': info}
            )
            self.signal_history.append(result)
            
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
                
            logger.info(f"Generated signal: {signal.value} (confidence: {confidence:.3f})")
            return signal.value
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return Signal.HOLD.value
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """Get statistics about generated signals"""
        if not self.signal_history:
            return {}
            
        signals = [s.signal.value for s in self.signal_history]
        confidences = [s.confidence for s in self.signal_history]
        
        return {
            'total_signals': len(signals),
            'buy_signals': signals.count('BUY'),
            'sell_signals': signals.count('SELL'),
            'hold_signals': signals.count('HOLD'),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'last_signal': signals[-1] if signals else None,
            'last_confidence': confidences[-1] if confidences else None
        }
    
    def clear_history(self):
        """Clear signal history"""
        self.signal_history.clear()


class EnsembleSignalGenerator:
    """
    Generates signals from multiple models using ensemble methods.
    
    Combines predictions from multiple models to create more robust signals.
    """
    
    def __init__(
        self,
        models: List[Any],
        ensemble_method: str = "majority_vote",
        confidence_threshold: float = 0.6,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble signal generator.
        
        Args:
            models: List of trained models
            ensemble_method: Method for combining predictions ('majority_vote', 'weighted_average')
            confidence_threshold: Minimum confidence for signal generation
            weights: Weights for weighted ensemble (must sum to 1)
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.confidence_threshold = confidence_threshold
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights
            
        self.generators = [
            SignalGenerator(model, confidence_threshold)
            for model in models
        ]
        
    def generate_signal(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        current_price: Optional[float] = None
    ) -> str:
        """
        Generate ensemble signal from multiple models.
        
        Args:
            data: Market data or features for prediction
            current_price: Current market price
            
        Returns:
            Signal string ('BUY', 'SELL', 'HOLD')
        """
        try:
            if not self.models:
                return Signal.HOLD.value
                
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for i, generator in enumerate(self.generators):
                try:
                    # Get individual prediction
                    if isinstance(data, pd.DataFrame):
                        features = data.iloc[-1].values if len(data.shape) == 2 else data.values
                    else:
                        features = data
                        
                    action, info = self.models[i].predict(features)
                    if hasattr(action, '__len__') and len(action) > 0:
                        action = action[0]
                        
                    confidence = info.get('confidence', 1.0) if info else 1.0
                    
                    predictions.append(int(action))
                    confidences.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Model {i} failed: {e}")
                    predictions.append(2)  # HOLD
                    confidences.append(0.0)
            
            # Ensemble prediction
            if self.ensemble_method == "majority_vote":
                final_action = self._majority_vote(predictions)
                final_confidence = np.mean(confidences)
                
            elif self.ensemble_method == "weighted_average":
                final_action = self._weighted_average(predictions, confidences)
                final_confidence = np.average(confidences, weights=self.weights)
                
            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
            
            # Check confidence threshold
            if final_confidence < self.confidence_threshold:
                logger.debug(f"Ensemble confidence {final_confidence:.3f} below threshold")
                return Signal.HOLD.value
                
            # Map to signal
            signal = {0: Signal.BUY, 1: Signal.SELL, 2: Signal.HOLD}.get(
                final_action, Signal.HOLD
            )
            
            logger.info(f"Ensemble signal: {signal.value} (confidence: {final_confidence:.3f})")
            return signal.value
            
        except Exception as e:
            logger.error(f"Error generating ensemble signal: {e}")
            return Signal.HOLD.value
    
    def _majority_vote(self, predictions: List[int]) -> int:
        """Determine action by majority vote"""
        votes = {0: 0, 1: 0, 2: 0}
        for pred in predictions:
            if pred in votes:
                votes[pred] += 1
                
        return max(votes, key=votes.get)
    
    def _weighted_average(self, predictions: List[int], confidences: List[float]) -> int:
        """Determine action by weighted average based on confidence"""
        if not predictions:
            return 2  # HOLD
            
        # Weight votes by confidence
        weighted_votes = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for pred, conf, weight in zip(predictions, confidences, self.weights):
            if pred in weighted_votes:
                weighted_votes[pred] += conf * weight
                
        return max(weighted_votes, key=weighted_votes.get)
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get statistics about ensemble performance"""
        stats = {}
        for i, generator in enumerate(self.generators):
            stats[f'model_{i}'] = generator.get_signal_stats()
            
        return {
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.models),
            'weights': self.weights,
            'model_stats': stats
        }


# Utility functions for signal analysis
def analyze_signal_performance(signals: List[SignalResult], returns: List[float]) -> Dict[str, float]:
    """
    Analyze signal performance against actual returns.
    
    Args:
        signals: List of signal results
        returns: Corresponding market returns
        
    Returns:
        Performance metrics dictionary
    """
    if len(signals) != len(returns) or not signals:
        return {}
        
    # Calculate signal returns
    signal_returns = []
    for signal, ret in zip(signals, returns):
        if signal.signal == Signal.BUY:
            signal_returns.append(ret)
        elif signal.signal == Signal.SELL:
            signal_returns.append(-ret)
        else:  # HOLD
            signal_returns.append(0.0)
    
    signal_returns = np.array(signal_returns)
    
    # Calculate metrics
    total_return = np.sum(signal_returns)
    avg_return = np.mean(signal_returns)
    volatility = np.std(signal_returns)
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Win rate
    winning_trades = np.sum(signal_returns > 0)
    total_trades = np.sum([s.signal != Signal.HOLD for s in signals])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'average_return': avg_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'winning_trades': winning_trades
    }


def signal_correlation(signals1: List[SignalResult], signals2: List[SignalResult]) -> float:
    """Calculate correlation between two signal series"""
    if len(signals1) != len(signals2) or not signals1:
        return 0.0
        
    # Convert signals to numerical values
    signal_map = {Signal.BUY: 1, Signal.SELL: -1, Signal.HOLD: 0}
    
    values1 = [signal_map[s.signal] for s in signals1]
    values2 = [signal_map[s.signal] for s in signals2]
    
    return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0