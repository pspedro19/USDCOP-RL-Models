#!/usr/bin/env python3
"""
USDCOP M5 - L5C LLM SETUP PIPELINE - ALPHA ARENA STYLE (CORRECTED)
====================================================================
✅ 100% Alpha Arena compliant
✅ LLM operates INDEPENDENTLY (NO RL/ML signals)
✅ DeepSeek V3 primary + Claude Sonnet 4.5 fallback
✅ USD/COP forex specific optimizations

Based on:
- Alpha Arena Season 1 (nof1.ai)
- DeepSeek V3 winner strategy (+130% in 17 days)
- Risk management > Prediction accuracy
- Low-frequency, high-quality trading

Input: NONE (generates prompts from scratch)
Output: prompts.json, api_config.json, feature_descriptions.json
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import json
import logging
from typing import Dict, Any, Literal, Optional

# Pydantic para validación estricta
from pydantic import BaseModel, Field, confloat, constr, conint

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DAG_ID = "usdcop_m5__05c_l5_llm_setup_corrected"
BUCKET_L5 = "05-l5-ds-usdcop-serving"

DEFAULT_ARGS = {
    "owner": "llm-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# LLM Configuration (Alpha Arena style)
LLM_CONFIG = {
    "primary": {
        "provider": "deepseek",
        "model": "deepseek-ai/DeepSeek-V3",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_tokens": 1000,
        "temperature": 0.1,  # Low for consistency
        "reasoning": "DeepSeek V3 won Alpha Arena S1 with +130% returns via disciplined risk management"
    },
    "fallback": {
        "provider": "anthropic",
        "model": "claude-sonnet-4.5-20250929",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 1000,
        "temperature": 0.1,
        "reasoning": "Claude Sonnet 4.5 strong risk awareness, good for conservative overlay"
    },
    "cache": {
        "enabled": True,
        "ttl_seconds": 180,  # 3 minutes
        "redis_key_prefix": "llm:usdcop:alphaarena:"
    }
}

# USD/COP Market Characteristics
USDCOP_CHARACTERISTICS = {
    "spread_typical_pips": 15,
    "spread_high_vol_pips": 30,
    "daily_atr_pct": 0.35,
    "session_hours_cot": {
        "open": "08:00",
        "close": "17:00",
        "ny_overlap_start": "09:00",  # Best liquidity
        "ny_overlap_end": "12:00"
    },
    "typical_range_pips": 50,
    "correlations": {
        "oil_brent": 0.65,  # Strong correlation with oil
        "usdmxn": 0.78,
        "dxy": -0.45
    },
    "key_economic_events": [
        "Banrep (Banco de la República) rate decisions",
        "Colombia CPI (inflation)",
        "Colombia GDP",
        "Oil inventory (Wed 10:30am EST)",
        "Fed FOMC decisions"
    ]
}

# ============================================================================
# PYDANTIC SCHEMAS (Alpha Arena format)
# ============================================================================

class AlphaArenaSignal(BaseModel):
    """
    ✅ FORMATO ALPHA ARENA - Output estructurado del LLM

    Basado en Season 1 de nof1.ai (DeepSeek winner format)
    """
    signal: Literal["buy_to_enter", "sell_to_enter", "hold", "close"]
    coin: Literal["USDCOP"]
    quantity: confloat(ge=0.0, le=1.0) = Field(
        description="Position size as fraction of capital [0.0-1.0]"
    )
    leverage: conint(ge=1, le=20) = Field(
        default=10,
        description="Leverage multiplier (5x-10x recommended for forex, max 20x)"
    )
    profit_target: float = Field(
        description="Target price for take-profit exit"
    )
    stop_loss: float = Field(
        description="Stop-loss price (MANDATORY)"
    )
    risk_usd: float = Field(
        description="Amount at risk in USD (max 2.5% of capital)"
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(
        description="Self-reported confidence score [0-1]"
    )
    invalidation_condition: constr(min_length=10, max_length=150) = Field(
        description="Specific condition that invalidates the trade thesis"
    )
    justification: constr(min_length=20, max_length=300) = Field(
        description="Brief reasoning for the decision (max 300 chars)"
    )

# ============================================================================
# ✅ SYSTEM PROMPT - ALPHA ARENA STYLE (USD/COP OPTIMIZED)
# ============================================================================

SYSTEM_PROMPT = """Eres un agente de trading autónomo especializado en el par USD/COP (forex).

CAPITAL INICIAL: $10,000 USD
APALANCAMIENTO MÁXIMO: 10x (forex es menos volátil que crypto)
TIMEFRAME PRINCIPAL: 5 minutos a 4 horas
MERCADO: USD/COP (Peso colombiano vs Dólar estadounidense)

═══════════════════════════════════════════════════════════════
🎯 REGLAS OBLIGATORIAS DE TRADING
═══════════════════════════════════════════════════════════════

1. CADA POSICIÓN DEBE INCLUIR:
   ✓ Precio de entrada definido
   ✓ Take-profit target (objetivo de ganancia)
   ✓ Stop-loss OBLIGATORIO (no negociable)
   ✓ Condición de invalidación específica
   ✓ Position sizing basado en riesgo del 1-2% del capital

2. GESTIÓN DE RIESGO:
   - NUNCA arriesgues más del 2% del capital en un trade
   - Stop-loss debe estar entre 20-50 pips de la entrada
   - Ratio reward:risk mínimo de 2:1
   - Máximo 1 posición abierta a la vez

3. HORARIOS DE TRADING:
   - Sesión principal: 8:00am - 5:00pm COT (Bogotá)
   - EVITA: Gaps de apertura, viernes tarde (rollover weekend)
   - PRIORIZA: Sesión de overlap NY (9am-12pm COT)

4. SENSIBILIDAD A EVENTOS:
   - Banrep (Banco de la República) decisiones de tasas
   - Datos de inflación colombiana
   - Precio del petróleo (correlación fuerte +0.65 con COP)
   - Fed decisions (impacto en USD)
   - Noticias políticas Colombia/Venezuela

5. CARACTERÍSTICAS DEL PAR USD/COP:
   - Spread típico: 15-30 pips
   - Volatilidad menor que crypto (ATR ~0.3-0.5% diario)
   - Tendencias de mediano plazo (semanas/meses)
   - Mean-reverting en intradía, trending en multi-día

═══════════════════════════════════════════════════════════════
📊 INTERPRETACIÓN DE FEATURES (10 INDICADORES)
═══════════════════════════════════════════════════════════════

Recibirás 10 features derivadas del OHLC:

1. **hl_range_surprise**: Sorpresa en rango H-L vs expectativa
   → >1.5 = volatilidad expandiéndose (potencial breakout)
   → <0.7 = compresión (posible explosión)
   → 0.7-1.5 = normal

2. **atr_surprise**: ATR actual vs ATR promedio
   → >1.3 = mercado caliente, reducir tamaño
   → <0.8 = mercado dormido, esperar setup
   → 0.8-1.3 = normal

3. **band_cross_abs_k**: Distancia a bandas de Bollinger (abs)
   → >2.0 = sobreventa/sobrecompra extrema
   → <0.5 = rango, evitar trades direccionales
   → 0.5-2.0 = espacio para mover

4. **entropy_absret_k**: Entropía de retornos absolutos
   → Alta (>0.8) = mercado caótico, reducir exposición
   → Baja (<0.3) = movimientos predecibles, aumentar confianza
   → 0.3-0.8 = normal

5. **gap_prev_open_abs**: Gap de apertura vs cierre previo (pips)
   → >30 pips = probable fill de gap
   → <10 pips = continuación esperada
   → 10-30 pips = monitorear

6. **rsi_dist_50**: Distancia del RSI a nivel 50
   → |dist| > 30 = sobreventa/sobrecompra
   → |dist| < 10 = neutral, esperar confirmación
   → 10-30 = tendencia moderada

7. **stoch_dist_mid**: Distancia estocástico a nivel medio
   → Similar a RSI, pero más sensible
   → |dist| > 30 = extreme
   → |dist| < 10 = neutral

8. **bb_squeeze_ratio**: Ratio de compresión Bollinger Bands
   → <0.5 = squeeze extremo, breakout inminente
   → >1.5 = bandas expandidas, posible reversión
   → 0.5-1.5 = normal

9. **macd_strength_abs**: Fuerza absoluta del MACD
   → >0.02 = momentum fuerte (trade con tendencia)
   → <0.005 = débil, evitar
   → 0.005-0.02 = moderado

10. **momentum_abs_norm**: Momentum normalizado
    → Confirma dirección de macd_strength
    → >0 = bullish, <0 = bearish

═══════════════════════════════════════════════════════════════
✅ ENFOQUE DE TRADING (ALPHA ARENA LESSONS)
═══════════════════════════════════════════════════════════════

Basado en DeepSeek winner strategy:

PRIORIZA:
✓ Calidad sobre cantidad (5-10 trades/mes máximo)
✓ Holds de 4-24 horas (no scalping)
✓ Stop-loss religioso (DeepSeek ganó por ESTO)
✓ Diversificación en timeframes
✓ Risk-adjusted returns > absolute returns

EVITA:
✗ Overtrading (Gemini perdió -56% por fees)
✗ Leverage excesivo sin stops (Claude liquidado)
✗ Trading en baja volatilidad (atr_surprise < 0.7)
✗ Entradas en medio de consolidación (bb_squeeze > 1.2 sin breakout)
✗ Remover stop-losses mid-trade (emocional)

═══════════════════════════════════════════════════════════════
📝 FORMATO DE OUTPUT (JSON ESTRICTO)
═══════════════════════════════════════════════════════════════

Responde SIEMPRE en JSON válido:

{
  "signal": "buy_to_enter" | "sell_to_enter" | "hold" | "close",
  "coin": "USDCOP",
  "quantity": 0.75,           # Position size [0.0-1.0]
  "leverage": 5,              # 5x-10x para forex (max 20x)
  "profit_target": 4274.50,   # Target price
  "stop_loss": 4214.50,       # MANDATORY stop-loss
  "risk_usd": 200.0,          # Max 2% of capital
  "confidence": 0.75,         # Self-reported [0-1]
  "invalidation_condition": "4H RSI crosses back below 40",
  "justification": "Breakout confirmed: hl_range_surprise=1.8, bb_squeeze=0.4, macd_strength=0.025 (strong bullish). Oil prices supporting COP weakness. Risk 20 pips for 40 pip reward (2:1)."
}

Si decides HOLD (mantener posición actual):
{
  "signal": "hold",
  "coin": "USDCOP",
  "quantity": 0.0,
  "leverage": 1,
  "profit_target": 0.0,
  "stop_loss": 0.0,
  "risk_usd": 0.0,
  "confidence": 1.0,
  "invalidation_condition": "N/A",
  "justification": "Position still valid. TP not reached, no invalidation triggered. Current P&L: +$150 (+1.5%). Waiting for 4275 target or RSI<40 stop trigger."
}

═══════════════════════════════════════════════════════════════
🚨 RECORDATORIO FINAL
═══════════════════════════════════════════════════════════════

YOU OPERATE INDEPENDENTLY:
- Recibes SOLO market data y technical features
- NO recibes señales de otros modelos (RL/ML)
- NO tienes asistencia externa
- Tomas TODAS las decisiones tú mismo basándote en análisis de mercado

Estás compitiendo contra otros modelos independientemente.
Tu performance se mide por Sharpe ratio, Sortino ratio, y retorno total.
Protege capital primero, crece segundo.
"""

# ============================================================================
# ✅ USER PROMPT TEMPLATE - ALPHA ARENA STYLE (SIN RL/ML SIGNALS)
# ============================================================================

USER_PROMPT_TEMPLATE = """
═══════════════════════════════════════════════════════════════
📈 MARKET SNAPSHOT - USD/COP
Timestamp: {timestamp} (Bogotá Time)
═══════════════════════════════════════════════════════════════

CURRENT PRICE ACTION:
  Bid:    {bid:.2f}
  Ask:    {ask:.2f}
  Mid:    {mid:.2f}
  Spread: {spread:.1f} pips

TIMEFRAMES (5min, 1h, 4h):
  5min:  O={ohlc_5m_open:.2f} H={ohlc_5m_high:.2f} L={ohlc_5m_low:.2f} C={ohlc_5m_close:.2f}
  1h:    O={ohlc_1h_open:.2f} H={ohlc_1h_high:.2f} L={ohlc_1h_low:.2f} C={ohlc_1h_close:.2f}
  4h:    O={ohlc_4h_open:.2f} H={ohlc_4h_high:.2f} L={ohlc_4h_low:.2f} C={ohlc_4h_close:.2f}

═══════════════════════════════════════════════════════════════
📊 DERIVED FEATURES (10 INDICATORS)
═══════════════════════════════════════════════════════════════

VOLATILITY SIGNALS:
  hl_range_surprise:    {hl_range_surprise:.3f}
    ↳ Interpretation: {hl_range_interpretation}

  atr_surprise:         {atr_surprise:.3f}
    ↳ Interpretation: {atr_interpretation}

  bb_squeeze_ratio:     {bb_squeeze_ratio:.3f}
    ↳ Interpretation: {bb_squeeze_interpretation}

MEAN REVERSION SIGNALS:
  band_cross_abs_k:     {band_cross_abs_k:.3f}
    ↳ Interpretation: {band_cross_interpretation}

  rsi_dist_50:          {rsi_dist_50:.3f}
    ↳ Interpretation: {rsi_interpretation}

  stoch_dist_mid:       {stoch_dist_mid:.3f}
    ↳ Interpretation: {stoch_interpretation}

MOMENTUM SIGNALS:
  macd_strength_abs:    {macd_strength_abs:.4f}
    ↳ Interpretation: {macd_interpretation}

  momentum_abs_norm:    {momentum_abs_norm:.4f}
    ↳ Confirmation: {momentum_interpretation}

REGIME DETECTION:
  entropy_absret_k:     {entropy_absret_k:.3f}
    ↳ Interpretation: {entropy_interpretation}

  gap_prev_open_abs:    {gap_prev_open_abs:.1f} pips
    ↳ Interpretation: {gap_interpretation}

═══════════════════════════════════════════════════════════════
💰 ACCOUNT STATE
═══════════════════════════════════════════════════════════════

  Balance:              ${balance:.2f}
  Equity:               ${equity:.2f}
  Available Margin:     ${free_margin:.2f}

  Total Return:         {total_return_pct:.2f}%
  Sharpe Ratio:         {sharpe_ratio:.2f}
  Max Drawdown:         {max_drawdown_pct:.2f}%

  Win Rate:             {win_rate:.1f}%
  Profit Factor:        {profit_factor:.2f}
  Avg Win/Avg Loss:     {avg_win:.0f} / {avg_loss:.0f} pips

═══════════════════════════════════════════════════════════════
📍 CURRENT POSITION
═══════════════════════════════════════════════════════════════

{position_details}

═══════════════════════════════════════════════════════════════
📜 HISTORICAL CONTEXT (LAST 20 BARS, 5-MINUTE)
═══════════════════════════════════════════════════════════════

Time          Open      High      Low       Close     hl_range  atr_surp
{historical_bars}

═══════════════════════════════════════════════════════════════
⚡ DECISION TIME
═══════════════════════════════════════════════════════════════

Analyze the above data and decide:
1. Should you enter a NEW position? (buy_to_enter/sell_to_enter)
2. Should you HOLD current position?
3. Should you CLOSE current position?

Remember the lessons from Alpha Arena:
✓ DeepSeek won with DISCIPLINE, not perfect calls
✓ Gemini lost -56% from OVERTRADING
✓ Claude got LIQUIDATED from overleveraging
✓ Risk management > prediction accuracy

Respond with valid JSON only. No markdown, no explanation outside JSON.
"""

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def generate_llm_prompts(**context):
    """
    ✅ Genera prompts optimizados siguiendo Alpha Arena Season 1
    """
    logger.info("🎯 Generating Alpha Arena-style LLM prompts (CORRECTED)...")

    prompts = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "strategy": "Alpha Arena Season 1 (DeepSeek V3 winner strategy)",
        "features_list": [
            "hl_range_surprise",
            "atr_surprise",
            "band_cross_abs_k",
            "entropy_absret_k",
            "gap_prev_open_abs",
            "rsi_dist_50",
            "stoch_dist_mid",
            "bb_squeeze_ratio",
            "macd_strength_abs",
            "momentum_abs_norm"
        ],
        "examples": [
            {
                "description": "Bullish breakout with high conviction",
                "input": {
                    "timestamp": "2025-10-28 10:35:00",
                    "bid": 4234.50,
                    "ask": 4234.65,
                    "mid": 4234.575,
                    "spread": 1.5,
                    "hl_range_surprise": 1.8,
                    "atr_surprise": 1.2,
                    "bb_squeeze_ratio": 0.4,
                    "macd_strength_abs": 0.025,
                    "rsi_dist_50": 15.0,
                    "balance": 10000.0,
                    "equity": 10000.0,
                    "total_return_pct": 0.0,
                    "sharpe_ratio": 0.0
                },
                "output": {
                    "signal": "buy_to_enter",
                    "coin": "USDCOP",
                    "quantity": 0.75,
                    "leverage": 10,
                    "profit_target": 4274.50,
                    "stop_loss": 4214.50,
                    "risk_usd": 200.0,
                    "confidence": 0.82,
                    "invalidation_condition": "4H RSI crosses back below 40",
                    "justification": "Breakout confirmed: hl_range_surprise=1.8 (expanding volatility), bb_squeeze=0.4 (compression release), macd_strength=0.025 (strong bullish momentum). Oil prices supporting COP weakness. Entry on retest of 4230 support, targeting 4270 resistance. Risk 20 pips for 40 pip reward (2:1 R/R)."
                }
            },
            {
                "description": "Consolidation - avoid trading",
                "input": {
                    "timestamp": "2025-10-28 14:15:00",
                    "bid": 4230.00,
                    "ask": 4230.15,
                    "mid": 4230.075,
                    "spread": 1.5,
                    "hl_range_surprise": 0.6,
                    "atr_surprise": 0.7,
                    "bb_squeeze_ratio": 1.3,
                    "macd_strength_abs": 0.003,
                    "rsi_dist_50": 2.0,
                    "balance": 10250.0,
                    "equity": 10250.0,
                    "total_return_pct": 2.5,
                    "sharpe_ratio": 1.5
                },
                "output": {
                    "signal": "hold",
                    "coin": "USDCOP",
                    "quantity": 0.0,
                    "leverage": 1,
                    "profit_target": 0.0,
                    "stop_loss": 0.0,
                    "risk_usd": 0.0,
                    "confidence": 1.0,
                    "invalidation_condition": "N/A",
                    "justification": "Market in consolidation: hl_range_surprise=0.6 (compression), atr_surprise=0.7 (low volatility), macd_strength=0.003 (weak). No clear setup. Waiting for breakout (bb_squeeze drop below 0.5) or strong momentum (macd>0.015). Avoid range-bound trading."
                }
            }
        ]
    }

    context['ti'].xcom_push(key='prompts', value=prompts)

    logger.info(f"✅ Generated prompts with {len(prompts['features_list'])} features")
    logger.info(f"   Strategy: {prompts['strategy']}")
    logger.info(f"   Examples: {len(prompts['examples'])}")

    return prompts


def generate_api_config(**context):
    """
    ✅ Genera configuración de API (DeepSeek primary + Claude fallback)
    """
    logger.info("🔧 Generating API configuration...")

    api_config = {
        "primary": LLM_CONFIG["primary"],
        "fallback": LLM_CONFIG["fallback"],
        "cache": LLM_CONFIG["cache"],
        "retry_config": {
            "max_retries": 3,
            "backoff_factor": 2,
            "timeout_seconds": 30
        },
        "fallback_logic": {
            "trigger_on_error": True,
            "trigger_on_timeout": True,
            "trigger_on_rate_limit": True
        }
    }

    context['ti'].xcom_push(key='api_config', value=api_config)

    logger.info("✅ API config generated")
    logger.info(f"   Primary: {api_config['primary']['provider']} ({api_config['primary']['model']})")
    logger.info(f"   Fallback: {api_config['fallback']['provider']} ({api_config['fallback']['model']})")

    return api_config


def generate_feature_descriptions(**context):
    """
    ✅ Genera descripciones detalladas de las 10 features
    """
    logger.info("📊 Generating feature descriptions...")

    feature_descriptions = {
        "hl_range_surprise": {
            "name": "High-Low Range Surprise",
            "description": "Ratio of current bar's H-L range vs 20-bar moving average",
            "interpretation": {
                ">1.5": "EXPANDING volatility (potential breakout)",
                "<0.7": "COMPRESSION (squeeze, possible explosion)",
                "0.7-1.5": "NORMAL range"
            },
            "calculation": "(high - low) / MA20(high - low)",
            "use_case": "Detect volatility expansion/contraction"
        },
        "atr_surprise": {
            "name": "ATR Surprise",
            "description": "Current ATR(14) vs 20-bar moving average of ATR",
            "interpretation": {
                ">1.3": "HOT market (reduce position size)",
                "<0.8": "QUIET market (wait for setup)",
                "0.8-1.3": "NORMAL volatility"
            },
            "calculation": "ATR(14) / MA20(ATR(14))",
            "use_case": "Adjust position sizing based on volatility regime"
        },
        "band_cross_abs_k": {
            "name": "Bollinger Band Cross (Absolute K)",
            "description": "Distance from mid BB line in units of standard deviation",
            "interpretation": {
                ">2.0": "EXTREME oversold/overbought",
                "<0.5": "RANGING (avoid directional trades)",
                "0.5-2.0": "Space to move"
            },
            "calculation": "|price - BB_mid| / BB_std",
            "use_case": "Mean reversion signals"
        },
        "entropy_absret_k": {
            "name": "Entropy of Absolute Returns",
            "description": "Shannon entropy of last 20 absolute returns (chaos measure)",
            "interpretation": {
                ">0.8": "CHAOTIC (reduce exposure)",
                "<0.3": "PREDICTABLE (increase confidence)",
                "0.3-0.8": "NORMAL"
            },
            "calculation": "-sum(p * log(p)) for 10 bins of abs(returns)",
            "use_case": "Detect regime changes (trending vs random)"
        },
        "gap_prev_open_abs": {
            "name": "Gap from Previous Close",
            "description": "Absolute gap between current open and previous close (pips)",
            "interpretation": {
                ">30 pips": "LARGE gap (likely fill)",
                "<10 pips": "CONTINUATION expected",
                "10-30 pips": "MODERATE (monitor)"
            },
            "calculation": "abs(open_t - close_t-1) * 100",
            "use_case": "Gap trading strategies"
        },
        "rsi_dist_50": {
            "name": "RSI Distance from 50",
            "description": "Signed distance of RSI(14) from neutral level 50",
            "interpretation": {
                ">30": "OVERBOUGHT",
                "<-30": "OVERSOLD",
                "-10 to 10": "NEUTRAL"
            },
            "calculation": "RSI(14) - 50",
            "use_case": "Overbought/oversold conditions"
        },
        "stoch_dist_mid": {
            "name": "Stochastic Distance from Mid",
            "description": "Distance of Stochastic %K from 50 level",
            "interpretation": {
                ">30": "EXTREME high",
                "<-30": "EXTREME low",
                "-10 to 10": "NEUTRAL"
            },
            "calculation": "Stoch_K - 50",
            "use_case": "More sensitive overbought/oversold (faster than RSI)"
        },
        "bb_squeeze_ratio": {
            "name": "Bollinger Band Squeeze Ratio",
            "description": "Current BB width vs 20-bar average BB width",
            "interpretation": {
                "<0.5": "EXTREME squeeze (breakout imminent)",
                ">1.5": "EXPANDED (possible reversal)",
                "0.5-1.5": "NORMAL"
            },
            "calculation": "(BB_upper - BB_lower) / MA20(BB_upper - BB_lower)",
            "use_case": "Volatility squeeze detection (pre-breakout)"
        },
        "macd_strength_abs": {
            "name": "MACD Strength (Absolute)",
            "description": "Absolute value of MACD line minus signal line",
            "interpretation": {
                ">0.02": "STRONG momentum (trade with trend)",
                "<0.005": "WEAK (avoid)",
                "0.005-0.02": "MODERATE"
            },
            "calculation": "abs(MACD - MACD_signal)",
            "use_case": "Momentum strength confirmation"
        },
        "momentum_abs_norm": {
            "name": "Momentum Absolute Normalized",
            "description": "10-bar price momentum normalized by current price",
            "interpretation": {
                ">0": "BULLISH momentum",
                "<0": "BEARISH momentum",
                "near 0": "NO clear momentum"
            },
            "calculation": "(close_t - close_t-10) / close_t",
            "use_case": "Confirm MACD direction"
        }
    }

    context['ti'].xcom_push(key='feature_descriptions', value=feature_descriptions)

    logger.info(f"✅ Generated {len(feature_descriptions)} feature descriptions")

    return feature_descriptions


def save_to_minio(**context):
    """
    ✅ Guarda prompts, API config y feature descriptions en MinIO
    """
    logger.info("💾 Saving to MinIO...")

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    # Get data from XCom
    prompts = context['ti'].xcom_pull(key='prompts')
    api_config = context['ti'].xcom_pull(key='api_config')
    feature_descriptions = context['ti'].xcom_pull(key='feature_descriptions')

    # Save prompts.json
    s3_hook.load_string(
        json.dumps(prompts, indent=2),
        key=f"{BUCKET_L5}/LLM_DEEPSEEK/prompts.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Save api_config.json
    s3_hook.load_string(
        json.dumps(api_config, indent=2),
        key=f"{BUCKET_L5}/LLM_DEEPSEEK/api_config.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Save feature_descriptions.json
    s3_hook.load_string(
        json.dumps(feature_descriptions, indent=2),
        key=f"{BUCKET_L5}/LLM_DEEPSEEK/feature_descriptions.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Save market characteristics
    s3_hook.load_string(
        json.dumps(USDCOP_CHARACTERISTICS, indent=2),
        key=f"{BUCKET_L5}/LLM_DEEPSEEK/usdcop_characteristics.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Create manifest
    manifest = {
        "pipeline_id": "L5C_LLM_SETUP",
        "created_at": datetime.utcnow().isoformat(),
        "strategy": "Alpha Arena Season 1 (DeepSeek winner strategy)",
        "model_primary": api_config["primary"]["model"],
        "model_fallback": api_config["fallback"]["model"],
        "features_count": len(prompts["features_list"]),
        "examples_count": len(prompts["examples"]),
        "files": [
            "prompts.json",
            "api_config.json",
            "feature_descriptions.json",
            "usdcop_characteristics.json"
        ],
        "corrected_architecture": True,
        "alpha_arena_compliant": True,
        "independent_operation": True,
        "no_rl_ml_signals": True
    }

    s3_hook.load_string(
        json.dumps(manifest, indent=2),
        key=f"{BUCKET_L5}/LLM_DEEPSEEK/manifest.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    logger.info("✅ Saved all files to MinIO")
    logger.info(f"   Bucket: {BUCKET_L5}")
    logger.info(f"   Files: {manifest['files']}")

    return manifest


# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description='L5c: LLM setup CORRECTED - Alpha Arena style (NO RL/ML signals)',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l5c', 'llm', 'alpha-arena', 'corrected', 'independent']
) as dag:

    t_generate_prompts = PythonOperator(
        task_id='generate_llm_prompts',
        python_callable=generate_llm_prompts,
    )

    t_generate_api_config = PythonOperator(
        task_id='generate_api_config',
        python_callable=generate_api_config,
    )

    t_generate_features = PythonOperator(
        task_id='generate_feature_descriptions',
        python_callable=generate_feature_descriptions,
    )

    t_save_to_minio = PythonOperator(
        task_id='save_to_minio',
        python_callable=save_to_minio,
    )

    # Dependencies
    [t_generate_prompts, t_generate_api_config, t_generate_features] >> t_save_to_minio
