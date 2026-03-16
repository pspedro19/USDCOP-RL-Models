"""
Technical Analysis Engine (Multi-Agent System — Phase 1)
=========================================================
Pre-computes ALL indicators using pandas-ta — LLM only interprets.

Indicators: Ichimoku, SuperTrend, MACD, RSI, Bollinger, ATR, Fibonacci, S/R.
Outputs: TechnicalAnalysisReport with bias, signals, scenarios, watch list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# NumPy 2.x removed several aliases that pandas-ta still references.
# Patch them before pandas-ta is imported (happens lazily in _compute_* methods).
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = np.int_
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "complex"):
    np.complex = np.complex128
if not hasattr(np, "object"):
    np.object = np.object_
if not hasattr(np, "str"):
    np.str = np.str_

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IchimokuReading:
    tenkan: Optional[float] = None
    kijun: Optional[float] = None
    senkou_a: Optional[float] = None
    senkou_b: Optional[float] = None
    chikou: Optional[float] = None
    price_vs_cloud: str = "neutral"  # above, below, inside
    tk_cross: str = "none"           # bullish, bearish, none
    cloud_color: str = "neutral"     # green, red, neutral

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuperTrendReading:
    value: Optional[float] = None
    direction: str = "neutral"       # bullish, bearish
    flip_detected: bool = False
    distance_pct: Optional[float] = None
    acting_as: str = "none"          # support, resistance

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MACDReading:
    macd_line: Optional[float] = None
    signal_line: Optional[float] = None
    histogram: Optional[float] = None
    histogram_direction: str = "flat"  # expanding, contracting, flat
    cross: str = "none"                # bullish, bearish, none
    divergence: str = "none"           # bullish, bearish, none

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FibonacciLevels:
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    direction: str = "none"   # retracement_up, retracement_down
    levels: dict = field(default_factory=dict)  # {0.236: price, 0.382: ..., 0.5: ..., 0.618: ...}
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SupportResistance:
    key_supports: list[float] = field(default_factory=list)
    key_resistances: list[float] = field(default_factory=list)
    no_trade_zone: tuple = (0.0, 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["no_trade_zone"] = list(self.no_trade_zone)
        return d


@dataclass
class TradingScenario:
    direction: str = "long"     # long, short
    entry_condition: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    targets: list[float] = field(default_factory=list)
    risk_reward: Optional[float] = None
    confidence: str = "medium"  # high, medium, low
    profile: str = "swing"     # scalp, intraday, swing

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TechnicalAnalysisReport:
    current_price: float = 0.0
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    volatility_regime: str = "normal"  # low, normal, high

    ichimoku: IchimokuReading = field(default_factory=IchimokuReading)
    supertrend: SuperTrendReading = field(default_factory=SuperTrendReading)
    macd: MACDReading = field(default_factory=MACDReading)
    rsi: Optional[float] = None
    bollinger_width: Optional[float] = None
    bollinger_position: str = "middle"  # upper, middle, lower

    dominant_bias: str = "neutral"      # bullish, bearish, neutral
    bias_confidence: float = 0.0        # 0.0 to 1.0
    bullish_signals: list[str] = field(default_factory=list)
    bearish_signals: list[str] = field(default_factory=list)

    fibonacci: FibonacciLevels = field(default_factory=FibonacciLevels)
    support_resistance: SupportResistance = field(default_factory=SupportResistance)

    scenarios: list[TradingScenario] = field(default_factory=list)
    watch_list: list[str] = field(default_factory=list)
    timeframe: str = "1d"

    def to_dict(self) -> dict:
        d = {
            "current_price": self.current_price,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "volatility_regime": self.volatility_regime,
            "ichimoku": self.ichimoku.to_dict(),
            "supertrend": self.supertrend.to_dict(),
            "macd": self.macd.to_dict(),
            "rsi": self.rsi,
            "bollinger_width": self.bollinger_width,
            "bollinger_position": self.bollinger_position,
            "dominant_bias": self.dominant_bias,
            "bias_confidence": self.bias_confidence,
            "bullish_signals": self.bullish_signals,
            "bearish_signals": self.bearish_signals,
            "fibonacci": self.fibonacci.to_dict(),
            "support_resistance": self.support_resistance.to_dict(),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "watch_list": self.watch_list,
            "timeframe": self.timeframe,
        }
        return d


# ---------------------------------------------------------------------------
# Confluence weights
# ---------------------------------------------------------------------------

CONFLUENCE_WEIGHTS = {
    "ichimoku": 2.0,
    "supertrend": 2.0,
    "macd_cross": 1.5,
    "rsi": 1.0,
    "bollinger": 0.5,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TechnicalAnalysisEngine:
    """Pre-compute all technical indicators — LLM only interprets."""

    def analyze(self, df: pd.DataFrame, timeframe: str = "1d") -> TechnicalAnalysisReport:
        """Run full technical analysis on OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close] (optionally volume).
                Must be sorted by time ascending.
            timeframe: Label for the timeframe (e.g., "5m", "15m", "1h", "4h", "1d").

        Returns:
            TechnicalAnalysisReport with all indicators + bias + scenarios.
        """
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for TA ({len(df)} rows, need >= 30)")
            return TechnicalAnalysisReport(timeframe=timeframe)

        df = df.copy()
        # Ensure lowercase column names
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return TechnicalAnalysisReport(timeframe=timeframe)

        current_price = float(df["close"].iloc[-1])
        report = TechnicalAnalysisReport(current_price=current_price, timeframe=timeframe)

        # Compute indicators
        try:
            self._compute_atr(df, report)
            self._compute_ichimoku(df, report)
            self._compute_supertrend(df, report)
            self._compute_macd(df, report)
            self._compute_rsi(df, report)
            self._compute_bollinger(df, report)
            self._compute_fibonacci(df, report)
            self._compute_support_resistance(df, report)
            self._determine_bias(report)
            self._generate_scenarios(df, report)
            self._generate_watch_list(report)
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}", exc_info=True)

        return report

    # ------------------------------------------------------------------
    # Individual indicator computations
    # ------------------------------------------------------------------

    def _compute_atr(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Compute Average True Range."""
        try:
            import pandas_ta as ta
            atr_s = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr_s is not None and not atr_s.empty:
                report.atr = float(atr_s.iloc[-1])
                report.atr_pct = round(report.atr / report.current_price * 100, 3)

                # Volatility regime from ATR percentile
                atr_pctl = atr_s.rank(pct=True).iloc[-1]
                if atr_pctl > 0.8:
                    report.volatility_regime = "high"
                elif atr_pctl < 0.2:
                    report.volatility_regime = "low"
                else:
                    report.volatility_regime = "normal"
        except Exception as e:
            logger.warning(f"ATR computation failed: {e}")

    def _compute_ichimoku(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Compute Ichimoku Cloud indicators."""
        try:
            import pandas_ta as ta
            ichi = ta.ichimoku(df["high"], df["low"], df["close"])
            if ichi is None or len(ichi) < 2:
                return

            ichi_df, ichi_span = ichi[0], ichi[1]

            tenkan = float(ichi_df.iloc[-1].get("ITS_9", np.nan))
            kijun = float(ichi_df.iloc[-1].get("IKS_26", np.nan))
            senkou_a = float(ichi_df.iloc[-1].get("ISA_9", np.nan))
            senkou_b = float(ichi_df.iloc[-1].get("ISB_26", np.nan))

            reading = IchimokuReading(
                tenkan=tenkan if not np.isnan(tenkan) else None,
                kijun=kijun if not np.isnan(kijun) else None,
                senkou_a=senkou_a if not np.isnan(senkou_a) else None,
                senkou_b=senkou_b if not np.isnan(senkou_b) else None,
            )

            # Price vs cloud
            price = report.current_price
            if reading.senkou_a is not None and reading.senkou_b is not None:
                cloud_top = max(reading.senkou_a, reading.senkou_b)
                cloud_bottom = min(reading.senkou_a, reading.senkou_b)
                if price > cloud_top:
                    reading.price_vs_cloud = "above"
                elif price < cloud_bottom:
                    reading.price_vs_cloud = "below"
                else:
                    reading.price_vs_cloud = "inside"

                reading.cloud_color = "green" if reading.senkou_a > reading.senkou_b else "red"

            # TK cross
            if reading.tenkan is not None and reading.kijun is not None:
                prev_tenkan = float(ichi_df.iloc[-2].get("ITS_9", np.nan))
                prev_kijun = float(ichi_df.iloc[-2].get("IKS_26", np.nan))
                if not np.isnan(prev_tenkan) and not np.isnan(prev_kijun):
                    if reading.tenkan > reading.kijun and prev_tenkan <= prev_kijun:
                        reading.tk_cross = "bullish"
                    elif reading.tenkan < reading.kijun and prev_tenkan >= prev_kijun:
                        reading.tk_cross = "bearish"

            report.ichimoku = reading
        except Exception as e:
            logger.warning(f"Ichimoku computation failed: {e}")

    def _compute_supertrend(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Compute SuperTrend indicator."""
        try:
            import pandas_ta as ta
            st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
            if st is None or st.empty:
                return

            st_cols = [c for c in st.columns if "SUPERTd" in c]
            st_val_cols = [c for c in st.columns if "SUPERT_" in c and "SUPERTd" not in c
                          and "SUPERTl" not in c and "SUPERTs" not in c]

            if st_cols and st_val_cols:
                direction_val = float(st[st_cols[0]].iloc[-1])
                st_value = float(st[st_val_cols[0]].iloc[-1])

                prev_direction = float(st[st_cols[0]].iloc[-2]) if len(st) > 1 else direction_val

                reading = SuperTrendReading(
                    value=st_value,
                    direction="bullish" if direction_val > 0 else "bearish",
                    flip_detected=(direction_val != prev_direction),
                    distance_pct=round(
                        abs(report.current_price - st_value) / report.current_price * 100, 3
                    ),
                    acting_as="support" if direction_val > 0 else "resistance",
                )
                report.supertrend = reading
        except Exception as e:
            logger.warning(f"SuperTrend computation failed: {e}")

    def _compute_macd(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Compute MACD with divergence detection."""
        try:
            import pandas_ta as ta
            macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd_df is None or macd_df.empty:
                return

            macd_cols = list(macd_df.columns)
            macd_line = float(macd_df[macd_cols[0]].iloc[-1])
            signal_line = float(macd_df[macd_cols[1]].iloc[-1])
            histogram = float(macd_df[macd_cols[2]].iloc[-1])

            prev_macd = float(macd_df[macd_cols[0]].iloc[-2])
            prev_signal = float(macd_df[macd_cols[1]].iloc[-2])
            prev_hist = float(macd_df[macd_cols[2]].iloc[-2])

            # Histogram direction
            if abs(histogram) > abs(prev_hist):
                hist_dir = "expanding"
            elif abs(histogram) < abs(prev_hist):
                hist_dir = "contracting"
            else:
                hist_dir = "flat"

            # Cross detection
            cross = "none"
            if macd_line > signal_line and prev_macd <= prev_signal:
                cross = "bullish"
            elif macd_line < signal_line and prev_macd >= prev_signal:
                cross = "bearish"

            # Simple divergence detection (price vs MACD direction over last 10 bars)
            divergence = "none"
            if len(df) >= 10:
                price_trend = df["close"].iloc[-1] - df["close"].iloc[-10]
                macd_trend = macd_line - float(macd_df[macd_cols[0]].iloc[-10])
                if price_trend > 0 and macd_trend < 0:
                    divergence = "bearish"
                elif price_trend < 0 and macd_trend > 0:
                    divergence = "bullish"

            report.macd = MACDReading(
                macd_line=macd_line,
                signal_line=signal_line,
                histogram=histogram,
                histogram_direction=hist_dir,
                cross=cross,
                divergence=divergence,
            )
        except Exception as e:
            logger.warning(f"MACD computation failed: {e}")

    def _compute_rsi(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Compute RSI using Wilder's smoothing."""
        try:
            import pandas_ta as ta
            rsi = ta.rsi(df["close"], length=14)
            if rsi is not None and not rsi.empty:
                report.rsi = round(float(rsi.iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"RSI computation failed: {e}")

    def _compute_bollinger(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Compute Bollinger Bands."""
        try:
            import pandas_ta as ta
            bb = ta.bbands(df["close"], length=20, std=2.0)
            if bb is None or bb.empty:
                return

            bb_cols = list(bb.columns)
            # BBL, BBM, BBU, BBB, BBP typically
            bb_lower = float(bb[bb_cols[0]].iloc[-1])
            bb_upper = float(bb[bb_cols[2]].iloc[-1])
            bb_width = float(bb[bb_cols[3]].iloc[-1]) if len(bb_cols) > 3 else None

            report.bollinger_width = bb_width

            price = report.current_price
            if price > bb_upper:
                report.bollinger_position = "upper"
            elif price < bb_lower:
                report.bollinger_position = "lower"
            else:
                report.bollinger_position = "middle"
        except Exception as e:
            logger.warning(f"Bollinger computation failed: {e}")

    def _compute_fibonacci(self, df: pd.DataFrame, report: TechnicalAnalysisReport) -> None:
        """Auto-detect swing high/low and compute Fibonacci retracement levels."""
        try:
            lookback = min(50, len(df))
            recent = df.tail(lookback)

            swing_high = float(recent["high"].max())
            swing_low = float(recent["low"].min())
            high_idx = recent["high"].idxmax()
            low_idx = recent["low"].idxmin()

            diff = swing_high - swing_low
            if diff < 1e-6:
                return

            # Determine direction based on which swing came first
            if high_idx > low_idx:
                # Swing low first, then high — uptrend, compute retracement down
                direction = "retracement_down"
                levels = {
                    0.0: swing_high,
                    0.236: swing_high - diff * 0.236,
                    0.382: swing_high - diff * 0.382,
                    0.500: swing_high - diff * 0.500,
                    0.618: swing_high - diff * 0.618,
                    0.786: swing_high - diff * 0.786,
                    1.0: swing_low,
                }
            else:
                # Swing high first, then low — downtrend, compute retracement up
                direction = "retracement_up"
                levels = {
                    0.0: swing_low,
                    0.236: swing_low + diff * 0.236,
                    0.382: swing_low + diff * 0.382,
                    0.500: swing_low + diff * 0.500,
                    0.618: swing_low + diff * 0.618,
                    0.786: swing_low + diff * 0.786,
                    1.0: swing_high,
                }

            # Round levels
            levels = {k: round(v, 2) for k, v in levels.items()}

            # Find nearest support and resistance
            price = report.current_price
            supports = sorted([v for v in levels.values() if v < price], reverse=True)
            resistances = sorted([v for v in levels.values() if v > price])

            report.fibonacci = FibonacciLevels(
                swing_high=swing_high,
                swing_low=swing_low,
                direction=direction,
                levels=levels,
                nearest_support=supports[0] if supports else None,
                nearest_resistance=resistances[0] if resistances else None,
            )
        except Exception as e:
            logger.warning(f"Fibonacci computation failed: {e}")

    def _compute_support_resistance(
        self, df: pd.DataFrame, report: TechnicalAnalysisReport
    ) -> None:
        """Find support/resistance levels from price action."""
        try:
            lookback = min(100, len(df))
            recent = df.tail(lookback)

            highs = recent["high"].values
            lows = recent["low"].values
            price = report.current_price

            # Simple pivot-based S/R
            supports = []
            resistances = []

            # Use recent swing lows as supports, swing highs as resistances
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
                   lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                    supports.append(float(lows[i]))

                if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
                   highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                    resistances.append(float(highs[i]))

            # Deduplicate close levels (within 0.1% of each other)
            supports = self._deduplicate_levels(supports, threshold_pct=0.1)
            resistances = self._deduplicate_levels(resistances, threshold_pct=0.1)

            # Keep only levels near current price (within 5%)
            supports = [s for s in supports if abs(s - price) / price < 0.05]
            resistances = [r for r in resistances if abs(r - price) / price < 0.05]

            # Sort
            supports.sort(reverse=True)
            resistances.sort()

            # No-trade zone: nearest S and R within 0.3% of price
            nearest_s = supports[0] if supports else price * 0.997
            nearest_r = resistances[0] if resistances else price * 1.003

            report.support_resistance = SupportResistance(
                key_supports=supports[:5],
                key_resistances=resistances[:5],
                no_trade_zone=(round(nearest_s, 2), round(nearest_r, 2)),
            )
        except Exception as e:
            logger.warning(f"S/R computation failed: {e}")

    @staticmethod
    def _deduplicate_levels(levels: list[float], threshold_pct: float = 0.1) -> list[float]:
        """Merge levels that are within threshold_pct of each other."""
        if not levels:
            return []
        levels = sorted(levels)
        merged = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - merged[-1]) / max(merged[-1], 1e-6) * 100 > threshold_pct:
                merged.append(lvl)
            else:
                merged[-1] = (merged[-1] + lvl) / 2  # average
        return [round(m, 2) for m in merged]

    # ------------------------------------------------------------------
    # Bias determination (confluence voting)
    # ------------------------------------------------------------------

    def _determine_bias(self, report: TechnicalAnalysisReport) -> None:
        """Determine dominant bias using weighted confluence voting."""
        bullish_score = 0.0
        bearish_score = 0.0
        bullish_signals = []
        bearish_signals = []

        # Ichimoku
        w = CONFLUENCE_WEIGHTS["ichimoku"]
        ichi = report.ichimoku
        if ichi.price_vs_cloud == "above":
            bullish_score += w
            bullish_signals.append("Precio por encima de la nube Ichimoku")
        elif ichi.price_vs_cloud == "below":
            bearish_score += w
            bearish_signals.append("Precio por debajo de la nube Ichimoku")
        if ichi.tk_cross == "bullish":
            bullish_score += w * 0.5
            bullish_signals.append("Cruce TK alcista (Ichimoku)")
        elif ichi.tk_cross == "bearish":
            bearish_score += w * 0.5
            bearish_signals.append("Cruce TK bajista (Ichimoku)")

        # SuperTrend
        w = CONFLUENCE_WEIGHTS["supertrend"]
        st = report.supertrend
        if st.direction == "bullish":
            bullish_score += w
            bullish_signals.append("SuperTrend alcista")
        elif st.direction == "bearish":
            bearish_score += w
            bearish_signals.append("SuperTrend bajista")
        if st.flip_detected:
            msg = "Cambio de SuperTrend detectado"
            if st.direction == "bullish":
                bullish_signals.append(msg)
            else:
                bearish_signals.append(msg)

        # MACD cross
        w = CONFLUENCE_WEIGHTS["macd_cross"]
        macd = report.macd
        if macd.cross == "bullish":
            bullish_score += w
            bullish_signals.append("Cruce MACD alcista")
        elif macd.cross == "bearish":
            bearish_score += w
            bearish_signals.append("Cruce MACD bajista")
        if macd.histogram and macd.histogram > 0 and macd.histogram_direction == "expanding":
            bullish_score += w * 0.3
            bullish_signals.append("Histograma MACD expandiendo (positivo)")
        elif macd.histogram and macd.histogram < 0 and macd.histogram_direction == "expanding":
            bearish_score += w * 0.3
            bearish_signals.append("Histograma MACD expandiendo (negativo)")
        if macd.divergence == "bullish":
            bullish_signals.append("Divergencia alcista MACD")
            bullish_score += w * 0.5
        elif macd.divergence == "bearish":
            bearish_signals.append("Divergencia bajista MACD")
            bearish_score += w * 0.5

        # RSI
        w = CONFLUENCE_WEIGHTS["rsi"]
        if report.rsi is not None:
            if report.rsi > 70:
                bearish_score += w
                bearish_signals.append(f"RSI sobrecomprado ({report.rsi:.1f})")
            elif report.rsi < 30:
                bullish_score += w
                bullish_signals.append(f"RSI sobrevendido ({report.rsi:.1f})")
            elif report.rsi > 50:
                bullish_score += w * 0.3
            else:
                bearish_score += w * 0.3

        # Bollinger
        w = CONFLUENCE_WEIGHTS["bollinger"]
        if report.bollinger_position == "upper":
            bearish_score += w
            bearish_signals.append("Precio en banda superior de Bollinger")
        elif report.bollinger_position == "lower":
            bullish_score += w
            bullish_signals.append("Precio en banda inferior de Bollinger")

        # Compute bias
        total = bullish_score + bearish_score
        if total < 0.1:
            report.dominant_bias = "neutral"
            report.bias_confidence = 0.0
        elif bullish_score > bearish_score * 1.3:
            report.dominant_bias = "bullish"
            report.bias_confidence = round(bullish_score / total, 2)
        elif bearish_score > bullish_score * 1.3:
            report.dominant_bias = "bearish"
            report.bias_confidence = round(bearish_score / total, 2)
        else:
            report.dominant_bias = "neutral"
            report.bias_confidence = round(0.5 - abs(bullish_score - bearish_score) / (2 * total), 2)

        report.bullish_signals = bullish_signals
        report.bearish_signals = bearish_signals

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def _generate_scenarios(
        self, df: pd.DataFrame, report: TechnicalAnalysisReport
    ) -> None:
        """Generate 2-3 trading scenarios with R:R calculations."""
        scenarios = []
        price = report.current_price
        atr = report.atr or (price * 0.01)  # fallback 1%

        # Scenario 1: Bias-aligned
        if report.dominant_bias == "bullish":
            entry = price
            sl = price - atr * 1.5
            tp1 = price + atr * 2
            tp2 = price + atr * 3
            rr = round((tp1 - entry) / (entry - sl), 2) if entry > sl else None
            scenarios.append(TradingScenario(
                direction="long",
                entry_condition="Compra en retroceso a SMA20 o soporte cercano",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                targets=[round(tp1, 2), round(tp2, 2)],
                risk_reward=rr,
                confidence="high" if report.bias_confidence > 0.7 else "medium",
                profile="swing",
            ))
        elif report.dominant_bias == "bearish":
            entry = price
            sl = price + atr * 1.5
            tp1 = price - atr * 2
            tp2 = price - atr * 3
            rr = round((entry - tp1) / (sl - entry), 2) if sl > entry else None
            scenarios.append(TradingScenario(
                direction="short",
                entry_condition="Venta en rebote a resistencia o SMA20",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                targets=[round(tp1, 2), round(tp2, 2)],
                risk_reward=rr,
                confidence="high" if report.bias_confidence > 0.7 else "medium",
                profile="swing",
            ))

        # Scenario 2: Counter-trend (always include for completeness)
        if report.dominant_bias == "bullish":
            # Short scenario on rejection at resistance
            if report.support_resistance.key_resistances:
                resist = report.support_resistance.key_resistances[0]
                sl = resist + atr
                tp = resist - atr * 2
                rr = round((resist - tp) / (sl - resist), 2) if sl > resist else None
                scenarios.append(TradingScenario(
                    direction="short",
                    entry_condition=f"Venta en rechazo de resistencia {resist:.0f}",
                    entry_price=round(resist, 2),
                    stop_loss=round(sl, 2),
                    targets=[round(tp, 2)],
                    risk_reward=rr,
                    confidence="low",
                    profile="swing",
                ))
        elif report.dominant_bias == "bearish":
            if report.support_resistance.key_supports:
                support = report.support_resistance.key_supports[0]
                sl = support - atr
                tp = support + atr * 2
                rr = round((tp - support) / (support - sl), 2) if support > sl else None
                scenarios.append(TradingScenario(
                    direction="long",
                    entry_condition=f"Compra en rebote de soporte {support:.0f}",
                    entry_price=round(support, 2),
                    stop_loss=round(sl, 2),
                    targets=[round(tp, 2)],
                    risk_reward=rr,
                    confidence="low",
                    profile="swing",
                ))

        # Scenario 3: Breakout
        if report.support_resistance.key_resistances:
            resist = report.support_resistance.key_resistances[0]
            entry = round(resist + atr * 0.1, 2)
            sl = round(resist - atr * 0.5, 2)
            tp = round(resist + atr * 2, 2)
            rr = round((tp - entry) / (entry - sl), 2) if entry > sl else None
            scenarios.append(TradingScenario(
                direction="long",
                entry_condition=f"Compra en ruptura de {resist:.0f}",
                entry_price=entry,
                stop_loss=sl,
                targets=[tp],
                risk_reward=rr,
                confidence="medium",
                profile="swing",
            ))

        report.scenarios = scenarios[:3]

    def _generate_watch_list(self, report: TechnicalAnalysisReport) -> None:
        """Generate list of key levels and events to monitor."""
        items = []

        if report.supertrend.flip_detected:
            items.append(f"SuperTrend cambio a {report.supertrend.direction}")

        if report.macd.cross != "none":
            items.append(f"Cruce MACD {report.macd.cross}")

        if report.macd.divergence != "none":
            items.append(f"Divergencia {report.macd.divergence} en MACD")

        if report.ichimoku.tk_cross != "none":
            items.append(f"Cruce TK {report.ichimoku.tk_cross}")

        if report.rsi is not None:
            if report.rsi > 65:
                items.append(f"RSI acercandose a sobrecompra ({report.rsi:.1f})")
            elif report.rsi < 35:
                items.append(f"RSI acercandose a sobreventa ({report.rsi:.1f})")

        if report.fibonacci.nearest_support:
            items.append(f"Soporte Fibonacci en {report.fibonacci.nearest_support:.0f}")
        if report.fibonacci.nearest_resistance:
            items.append(f"Resistencia Fibonacci en {report.fibonacci.nearest_resistance:.0f}")

        if report.volatility_regime == "high":
            items.append("Volatilidad alta — usar stops mas amplios")
        elif report.volatility_regime == "low":
            items.append("Volatilidad baja — posible ruptura inminente")

        report.watch_list = items[:8]
