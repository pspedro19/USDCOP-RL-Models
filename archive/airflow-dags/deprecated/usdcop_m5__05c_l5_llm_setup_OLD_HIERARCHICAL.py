#!/usr/bin/env python3
"""
USDCOP M5 - L5C LLM SETUP PIPELINE - ALPHA ARENA STRATEGY
==========================================================
Configura estrategia LLM basada en Alpha Arena (nof1.ai):
✅ System prompt profesional de trading autónomo
✅ User prompt con estructura completa de mercado
✅ Pydantic schemas estrictos (formato Alpha Arena)
✅ DeepSeek V3.1 primary + Claude Sonnet 4.5 fallback
✅ Redis cache para optimización

**Estrategia basada en:**
- Alpha Arena Season 1 (nof1.ai)
- DeepSeek V3.1 ganador (+130% en 17 días)
- Risk management > Prediction accuracy
- Low-frequency, high-quality trading

Input: Market context templates
Output: prompts.json, api_config.json, cache_config.json
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import json
import logging
from typing import Dict, Any, Literal, Optional

# ✅ Pydantic para validación estricta (formato Alpha Arena)
from pydantic import BaseModel, Field, confloat, constr, conint

logger = logging.getLogger(__name__)

# ============================================================================
# ✅ ALPHA ARENA: PYDANTIC SCHEMAS
# ============================================================================

class MarketState(BaseModel):
    """Estado actual de mercado (snapshot)"""
    timestamp: str
    price: float
    ema_20: float
    macd: float
    rsi_7: float
    rsi_14: float
    volume: float
    volatility: float
    trend: Literal["bullish", "bearish", "neutral"]

class AccountInfo(BaseModel):
    """Información de cuenta y performance"""
    total_return_pct: float
    available_cash: float
    total_equity: float
    sharpe_ratio: Optional[float] = None
    open_positions: int = 0

class SignalInput(BaseModel):
    """Señales de RL y ML models"""
    rl_signal: Literal["long", "short", "flat"]
    rl_confidence: confloat(ge=0.0, le=1.0)
    ml_signal: Literal["long", "short", "flat"]
    ml_confidence: confloat(ge=0.0, le=1.0)

class AlphaArenaSignal(BaseModel):
    """
    ✅ FORMATO ALPHA ARENA - Output estructurado del LLM

    Basado en Season 1 de nof1.ai:
    - signal: Acción a tomar
    - coin: Activo (USDCOP en nuestro caso)
    - quantity: Tamaño de posición [0.0-1.0]
    - leverage: 1x-20x (típicamente 10x-15x)
    - profit_target: Precio de salida objetivo
    - stop_loss: Precio de stop loss (OBLIGATORIO)
    - risk_usd: Monto en riesgo (USD)
    - confidence: Score autorreportado [0-1]
    - invalidation_condition: Trigger de salida predefinido
    - justification: Razonamiento breve (max 240 chars)
    """
    signal: Literal["buy_to_enter", "sell_to_enter", "hold", "close"]
    coin: Literal["USDCOP"]
    quantity: confloat(ge=0.0, le=1.0) = Field(
        description="Position size as fraction of capital [0.0-1.0]"
    )
    leverage: conint(ge=1, le=20) = Field(
        default=10,
        description="Leverage multiplier (1x-20x, typically 10x-15x)"
    )
    profit_target: float = Field(
        description="Target price for take-profit exit"
    )
    stop_loss: float = Field(
        description="Stop-loss price (MANDATORY)"
    )
    risk_usd: float = Field(
        description="Amount at risk in USD"
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(
        description="Self-reported confidence score [0-1]"
    )
    invalidation_condition: constr(min_length=10, max_length=100) = Field(
        description="Specific condition that invalidates the trade thesis"
    )
    justification: constr(min_length=20, max_length=240) = Field(
        description="Brief reasoning for the decision"
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

DAG_ID = "usdcop_m5__05c_l5_llm_setup"
BUCKET_L5 = "usdcop-l5-serving"

DEFAULT_ARGS = {
    "owner": "llm-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# LLM Configuration (Alpha Arena inspired)
LLM_CONFIG = {
    "primary": {
        "provider": "deepseek",
        "model": "deepseek-ai/DeepSeek-V3",
        "api_key_env": "DEEPSEEK_API_KEY",
        "max_tokens": 500,
        "temperature": 0.1,
        "reasoning": "DeepSeek V3.1 won Alpha Arena S1 with +130% returns via disciplined risk management"
    },
    "fallback": {
        "provider": "anthropic",
        "model": "claude-sonnet-4.5-20250929",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 500,
        "temperature": 0.1,
        "reasoning": "Claude Sonnet 4.5 achieved +25% in Alpha Arena S1, strong risk awareness"
    },
    "cache": {
        "enabled": True,
        "ttl_seconds": 180,  # 3 minutes (Alpha Arena uses 2-3 min cycles)
        "redis_key_prefix": "llm:usdcop:alphaarena:"
    }
}

# ============================================================================
# ✅ ALPHA ARENA: SYSTEM PROMPT (PRODUCTION-GRADE)
# ============================================================================

SYSTEM_PROMPT = """You are an autonomous trading agent operating on USD/COP perpetual futures.

You begin with $10,000 capital and must generate high-quality, disciplined trading decisions.

═══════════════════════════════════════════════════════════════════════
🎯 CRITICAL RULES (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════════════

1. **EVERY POSITION MUST INCLUDE:**
   - A defined entry price
   - A profit target (take-profit level)
   - A stop-loss level (MANDATORY, never remove)
   - An invalidation condition (specific market trigger)
   - Position sizing based on risk tolerance

2. **LEVERAGE & RISK:**
   - Use leverage of 10x-15x appropriately (max 20x in high-conviction)
   - Never exceed 20% of capital at risk in a single trade
   - Stop-losses must be ABOVE liquidation prices
   - Calculate risk_usd = position_size × distance_to_stop_loss

3. **TRADING DISCIPLINE:**
   - Quality over quantity: Fewer trades, higher conviction
   - Never override stop-losses once set
   - Consider transaction fees in every decision
   - Risk-adjusted returns > absolute returns

4. **MARKET CONTEXT:**
   - Trading window: 8:00 AM - 12:55 PM COT (Mon-Fri)
   - If spread > 10 bps, reduce position size by 50%
   - If volatility > 2%, consider reducing leverage
   - Diversification reduces risk but not required

5. **POSITION SIZING FORMULA:**
   - Base size = confidence × available_capital × 0.40
   - Adjusted size = base_size × (1 - spread_penalty) × (1 - volatility_penalty)
   - Leverage applied: notional_exposure = adjusted_size × leverage

═══════════════════════════════════════════════════════════════════════
📊 DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════════════

**Inputs you receive:**
1. Current market state (price, EMA, MACD, RSI, volume, volatility)
2. Intraday series (last 10 bars, 5-minute intervals)
3. Account information (returns, Sharpe, cash, open positions)
4. Signals from RL and ML models (side + confidence)

**Your output (JSON only):**
{
  "signal": "buy_to_enter" | "sell_to_enter" | "hold" | "close",
  "coin": "USDCOP",
  "quantity": 0.0-1.0,
  "leverage": 10-20,
  "profit_target": <price>,
  "stop_loss": <price>,
  "risk_usd": <amount>,
  "confidence": 0.0-1.0,
  "invalidation_condition": "<specific trigger>",
  "justification": "<brief reasoning, max 240 chars>"
}

═══════════════════════════════════════════════════════════════════════
✅ BEST PRACTICES (ALPHA ARENA SEASON 1 LESSONS)
═══════════════════════════════════════════════════════════════════════

**From DeepSeek V3.1 (Winner, +130%):**
- Low-frequency trading (17 trades in 10 days)
- Average hold time: 49 hours
- Profit/Loss ratio: 6.7:1
- Leverage: 10-15x with strict stops
- Diversified across all 6 assets (we have 1, so focus on quality)

**Common failure modes to AVOID:**
❌ Over-trading (Gemini: 64 trades, -56% due to fees)
❌ Excessive leverage without stops (Claude: liquidated at 20x)
❌ Ignoring transaction costs (fees erode profits)
❌ Removing stop-losses mid-trade (emotional override)
❌ Trading against clear trend without invalidation

**Key insight:** "You don't need perfect market calls—just solid risk controls."

═══════════════════════════════════════════════════════════════════════
🚨 REPORT FORMAT (STRICT)
═══════════════════════════════════════════════════════════════════════

SIDE | COIN | LEVERAGE | NOTIONAL | EXIT PLAN | RISK_USD | CONFIDENCE

Example:
LONG | USDCOP | 15x | $6,000 | TP: 4,150 / SL: 4,010 / INV: RSI<40 | $400 | 0.75

Always respond with valid JSON matching AlphaArenaSignal schema.
"""

# ============================================================================
# ✅ ALPHA ARENA: USER PROMPT TEMPLATE
# ============================================================================

USER_PROMPT_TEMPLATE = """
═══════════════════════════════════════════════════════════════════════
📈 CURRENT MARKET STATE (USD/COP)
═══════════════════════════════════════════════════════════════════════

**Timestamp:** {timestamp} (COT)
**Current Price:** ${price:.2f}
**EMA(20):** ${ema_20:.2f}
**MACD:** {macd:.6f}
**RSI(7):** {rsi_7:.1f}
**RSI(14):** {rsi_14:.1f}
**Volume:** {volume:,.0f}
**Volatility (5-bar):** {volatility:.4f}
**Trend:** {trend}

═══════════════════════════════════════════════════════════════════════
📊 INTRADAY SERIES (OLDEST → NEWEST, Last 10 bars)
═══════════════════════════════════════════════════════════════════════

{intraday_series}

═══════════════════════════════════════════════════════════════════════
💰 ACCOUNT INFORMATION
═══════════════════════════════════════════════════════════════════════

**Total Return:** {total_return_pct:+.2f}%
**Available Cash:** ${available_cash:,.2f}
**Total Equity:** ${total_equity:,.2f}
**Sharpe Ratio:** {sharpe_ratio:.3f}
**Open Positions:** {open_positions}

═══════════════════════════════════════════════════════════════════════
🤖 MODEL SIGNALS (RL & ML)
═══════════════════════════════════════════════════════════════════════

**RL Model:**
  - Signal: {rl_signal}
  - Confidence: {rl_confidence:.2f}

**ML Model:**
  - Signal: {ml_signal}
  - Confidence: {ml_confidence:.2f}

═══════════════════════════════════════════════════════════════════════
⚡ YOUR DECISION (JSON ONLY)
═══════════════════════════════════════════════════════════════════════

Based on the above market state, account info, and model signals, generate your trading decision.

**CRITICAL REMINDERS:**
1. Always include stop_loss (non-negotiable)
2. Position size must reflect confidence and risk
3. Justify your decision with specific evidence
4. If RL and ML disagree → reduce confidence or go flat
5. If outside trading window (8AM-12:55PM COT) → hold or close

Respond with valid JSON matching this structure:
{
  "signal": "buy_to_enter" | "sell_to_enter" | "hold" | "close",
  "coin": "USDCOP",
  "quantity": 0.75,
  "leverage": 15,
  "profit_target": 4150.0,
  "stop_loss": 4010.0,
  "risk_usd": 400.0,
  "confidence": 0.75,
  "invalidation_condition": "RSI drops below 40 on 4H timeframe",
  "justification": "Strong bullish momentum, RL/ML consensus, RSI not overbought, tight stops"
}
"""

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def generate_llm_prompts(**context):
    """
    ✅ Genera prompts optimizados siguiendo Alpha Arena Season 1
    """
    logger.info("🎯 Generating Alpha Arena-style LLM prompts...")

    prompts = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "strategy": "Alpha Arena Season 1 (DeepSeek V3.1 winner strategy)",
        "examples": [
            {
                "description": "Bullish consensus with moderate confidence",
                "input": {
                    "timestamp": "2025-10-28 10:35:00",
                    "price": 4025.50,
                    "ema_20": 4018.30,
                    "macd": 0.0023,
                    "rsi_7": 62.5,
                    "rsi_14": 58.3,
                    "volume": 125000,
                    "volatility": 0.0145,
                    "trend": "bullish",
                    "intraday_series": "Price series: [4010, 4012, 4018, 4020, 4022, 4023, 4024, 4025, 4025, 4026]",
                    "total_return_pct": 5.2,
                    "available_cash": 8500.0,
                    "total_equity": 10520.0,
                    "sharpe_ratio": 1.82,
                    "open_positions": 0,
                    "rl_signal": "long",
                    "rl_confidence": 0.78,
                    "ml_signal": "long",
                    "ml_confidence": 0.72
                },
                "expected_output": {
                    "signal": "buy_to_enter",
                    "coin": "USDCOP",
                    "quantity": 0.70,
                    "leverage": 15,
                    "profit_target": 4150.0,
                    "stop_loss": 4000.0,
                    "risk_usd": 595.0,
                    "confidence": 0.75,
                    "invalidation_condition": "Price breaks below EMA20 with volume surge",
                    "justification": "RL/ML consensus, bullish trend confirmed, MACD positive, RSI healthy range, tight 25bp stop"
                }
            },
            {
                "description": "Disagreement between models - go flat",
                "input": {
                    "timestamp": "2025-10-28 11:45:00",
                    "price": 4052.80,
                    "ema_20": 4048.10,
                    "macd": -0.0012,
                    "rsi_7": 71.2,
                    "rsi_14": 68.5,
                    "volume": 98000,
                    "volatility": 0.0235,
                    "trend": "neutral",
                    "intraday_series": "Price series: [4050, 4052, 4055, 4053, 4051, 4052, 4053, 4052, 4051, 4053]",
                    "total_return_pct": 12.5,
                    "available_cash": 7200.0,
                    "total_equity": 11250.0,
                    "sharpe_ratio": 2.15,
                    "open_positions": 1,
                    "rl_signal": "short",
                    "rl_confidence": 0.55,
                    "ml_signal": "long",
                    "ml_confidence": 0.62
                },
                "expected_output": {
                    "signal": "hold",
                    "coin": "USDCOP",
                    "quantity": 0.0,
                    "leverage": 10,
                    "profit_target": 4052.80,
                    "stop_loss": 4052.80,
                    "risk_usd": 0.0,
                    "confidence": 0.30,
                    "invalidation_condition": "N/A - staying flat",
                    "justification": "RL/ML disagree, RSI overbought, volatility elevated (2.35%), MACD turning negative. Prudent to wait."
                }
            },
            {
                "description": "High volatility - reduce leverage",
                "input": {
                    "timestamp": "2025-10-28 12:15:00",
                    "price": 4010.20,
                    "ema_20": 4025.50,
                    "macd": -0.0045,
                    "rsi_7": 35.8,
                    "rsi_14": 42.1,
                    "volume": 185000,
                    "volatility": 0.0312,
                    "trend": "bearish",
                    "intraday_series": "Price series: [4050, 4045, 4038, 4030, 4025, 4020, 4015, 4012, 4010, 4011]",
                    "total_return_pct": 3.8,
                    "available_cash": 9100.0,
                    "total_equity": 10380.0,
                    "sharpe_ratio": 1.45,
                    "open_positions": 0,
                    "rl_signal": "short",
                    "rl_confidence": 0.82,
                    "ml_signal": "short",
                    "ml_confidence": 0.79
                },
                "expected_output": {
                    "signal": "sell_to_enter",
                    "coin": "USDCOP",
                    "quantity": 0.45,
                    "leverage": 10,
                    "profit_target": 3950.0,
                    "stop_loss": 4030.0,
                    "risk_usd": 364.0,
                    "confidence": 0.80,
                    "invalidation_condition": "RSI rebounds above 50 or price reclaims EMA20",
                    "justification": "Strong short consensus, oversold RSI, MACD negative. Reduced leverage (10x) due to 3.1% volatility."
                }
            }
        ]
    }

    context['ti'].xcom_push(key='prompts', value=prompts)

    logger.info("✅ Alpha Arena prompts generated with 3 examples")
    return prompts

def validate_deepseek_api(**context):
    """
    ✅ Valida conectividad con DeepSeek API (primary)
    """
    logger.info("🔍 Validating DeepSeek API...")

    api_key = os.getenv('DEEPSEEK_API_KEY')

    if not api_key:
        logger.warning("⚠️ DEEPSEEK_API_KEY not set, will use fallback (Claude)")
        context['ti'].xcom_push(key='deepseek_available', value=False)
        return {"status": "unavailable", "fallback": "claude"}

    try:
        # Test with simple inference
        import requests

        test_prompt = """Test: USD/COP at 4025. Respond with JSON:
        {"signal": "hold", "coin": "USDCOP", "quantity": 0.0, "leverage": 10,
         "profit_target": 4025.0, "stop_loss": 4025.0, "risk_usd": 0.0,
         "confidence": 0.5, "invalidation_condition": "test", "justification": "test response"}"""

        response = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'system', 'content': 'You are a trading assistant.'},
                    {'role': 'user', 'content': test_prompt}
                ],
                'max_tokens': 200,
                'temperature': 0.1
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ DeepSeek API test successful")
            context['ti'].xcom_push(key='deepseek_available', value=True)

            return {
                "status": "available",
                "model": "deepseek-chat",
                "test_response": result['choices'][0]['message']['content'][:100]
            }
        else:
            logger.error(f"❌ DeepSeek API returned {response.status_code}")
            context['ti'].xcom_push(key='deepseek_available', value=False)
            return {"status": "failed", "error": f"HTTP {response.status_code}", "fallback": "claude"}

    except Exception as e:
        logger.error(f"❌ DeepSeek API validation failed: {e}")
        context['ti'].xcom_push(key='deepseek_available', value=False)
        return {"status": "failed", "error": str(e), "fallback": "claude"}

def validate_claude_api(**context):
    """
    ✅ Valida conectividad con Claude API (fallback)
    """
    logger.info("🔍 Validating Claude API (fallback)...")

    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        logger.warning("⚠️ ANTHROPIC_API_KEY not set")
        context['ti'].xcom_push(key='claude_available', value=False)
        return {"status": "unavailable"}

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        # Test inference
        test_prompt = """Test: USD/COP at 4025. Respond with JSON:
        {"signal": "hold", "coin": "USDCOP", "quantity": 0.0, "leverage": 10,
         "profit_target": 4025.0, "stop_loss": 4025.0, "risk_usd": 0.0,
         "confidence": 0.5, "invalidation_condition": "test", "justification": "test response"}"""

        response = client.messages.create(
            model=LLM_CONFIG['fallback']['model'],
            max_tokens=200,
            temperature=0.1,
            system="You are a trading assistant.",
            messages=[{
                "role": "user",
                "content": test_prompt
            }]
        )

        logger.info(f"✅ Claude API test successful")
        context['ti'].xcom_push(key='claude_available', value=True)

        return {
            "status": "available",
            "model": LLM_CONFIG['fallback']['model'],
            "test_response": response.content[0].text[:100]
        }

    except Exception as e:
        logger.error(f"❌ Claude API validation failed: {e}")
        context['ti'].xcom_push(key='claude_available', value=False)
        return {"status": "failed", "error": str(e)}

def setup_redis_cache(**context):
    """
    ✅ Configura Redis para cachear respuestas LLM (Alpha Arena: 2-3 min cycles)
    """
    logger.info("🔧 Setting up Redis cache for LLM...")

    try:
        import redis

        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'usdcop-redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD', 'redis123'),
            decode_responses=True
        )

        # Test connection
        r.ping()

        # Set test key
        test_key = f"{LLM_CONFIG['cache']['redis_key_prefix']}test"
        r.setex(test_key, 60, json.dumps({"test": "ok", "strategy": "alpha_arena"}))

        # Verify
        cached = r.get(test_key)
        assert json.loads(cached)['test'] == 'ok'

        logger.info("✅ Redis cache setup successful (TTL: 180s for Alpha Arena cycles)")

        return {
            "status": "configured",
            "redis_host": os.getenv('REDIS_HOST', 'usdcop-redis'),
            "cache_ttl": LLM_CONFIG['cache']['ttl_seconds'],
            "strategy": "alpha_arena_season1"
        }

    except Exception as e:
        logger.error(f"❌ Redis cache setup failed: {e}")
        return {"status": "failed", "error": str(e)}

def save_llm_config(**context):
    """
    ✅ Guarda configuración completa en MinIO (Alpha Arena strategy)
    """
    logger.info("💾 Saving Alpha Arena LLM configuration to MinIO...")

    s3_hook = S3Hook(aws_conn_id='minio_conn')
    run_id = f"L5c_AlphaArena_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    base_path = f"llm_config/{run_id}/"

    # Gather results
    prompts = context['ti'].xcom_pull(key='prompts')
    deepseek_result = context['ti'].xcom_pull(task_ids='validate_deepseek_api')
    claude_result = context['ti'].xcom_pull(task_ids='validate_claude_api')
    redis_result = context['ti'].xcom_pull(task_ids='setup_redis_cache')

    # API Config
    api_config = {
        "strategy": "Alpha Arena Season 1",
        "winner_model": "DeepSeek V3.1 (+130% in 17 days)",
        "primary": {
            **LLM_CONFIG['primary'],
            "available": context['ti'].xcom_pull(key='deepseek_available')
        },
        "fallback": {
            **LLM_CONFIG['fallback'],
            "available": context['ti'].xcom_pull(key='claude_available')
        },
        "validation_results": {
            "deepseek": deepseek_result,
            "claude": claude_result
        }
    }

    # Cache Config
    cache_config = {
        **LLM_CONFIG['cache'],
        "redis_status": redis_result,
        "note": "Alpha Arena uses 2-3 min inference cycles, cache TTL set to 180s"
    }

    # Upload to MinIO
    s3_hook.load_string(
        string_data=json.dumps(prompts, indent=2),
        key=f"{base_path}prompts.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    s3_hook.load_string(
        string_data=json.dumps(api_config, indent=2),
        key=f"{base_path}api_config.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    s3_hook.load_string(
        string_data=json.dumps(cache_config, indent=2),
        key=f"{base_path}cache_config.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Pydantic schema for reference
    schema_doc = {
        "AlphaArenaSignal": AlphaArenaSignal.schema(),
        "MarketState": MarketState.schema(),
        "AccountInfo": AccountInfo.schema(),
        "SignalInput": SignalInput.schema(),
        "note": "These schemas enforce Alpha Arena Season 1 output format"
    }

    s3_hook.load_string(
        string_data=json.dumps(schema_doc, indent=2),
        key=f"{base_path}pydantic_schemas.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Manifest
    manifest = {
        "model_id": "LLM_ALPHAARENA_v1.0",
        "model_type": "LLM",
        "algorithm": "Alpha Arena S1 (DeepSeek V3 primary + Claude 4.5 fallback)",
        "strategy_code": "LLM_ALPHAARENA",
        "strategy_origin": "nof1.ai Alpha Arena Season 1",
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "primary_provider": "deepseek",
        "primary_available": context['ti'].xcom_pull(key='deepseek_available'),
        "fallback_provider": "anthropic",
        "fallback_available": context['ti'].xcom_pull(key='claude_available'),
        "cache_enabled": cache_config.get('enabled', False),
        "gates_passed": True,
        "key_principles": [
            "Risk management > Prediction accuracy",
            "Low-frequency, high-quality trading",
            "Mandatory stop-losses on every position",
            "Position sizing linked to confidence",
            "10x-15x leverage with strict risk controls"
        ],
        "alpha_arena_results": {
            "deepseek_v3": {"return": 130, "trades": 17, "avg_hold_hours": 49, "profit_loss_ratio": 6.7},
            "claude_sonnet_4.5": {"return": 25, "trades": 9, "win_rate": 0.38, "profit_loss_ratio": 2.1}
        }
    }

    s3_hook.load_string(
        string_data=json.dumps(manifest, indent=2),
        key=f"{base_path}manifest.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    # Save to _meta/l5c_latest.json
    s3_hook.load_string(
        string_data=json.dumps({
            "run_id": run_id,
            "layer": "l5c",
            "strategy": "alpha_arena_season1",
            "updated_at": datetime.utcnow().isoformat()
        }, indent=2),
        key="_meta/l5c_latest.json",
        bucket_name=BUCKET_L5,
        replace=True
    )

    logger.info(f"✅ Alpha Arena LLM config saved: {run_id}")

    return manifest

# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description='L5c: Alpha Arena LLM strategy setup (DeepSeek V3 + Claude fallback)',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l5c', 'llm', 'alpha-arena', 'deepseek', 'production']
) as dag:

    t_generate_prompts = PythonOperator(
        task_id='generate_prompts',
        python_callable=generate_llm_prompts,
    )

    t_validate_deepseek = PythonOperator(
        task_id='validate_deepseek_api',
        python_callable=validate_deepseek_api,
    )

    t_validate_claude = PythonOperator(
        task_id='validate_claude_api',
        python_callable=validate_claude_api,
    )

    t_setup_cache = PythonOperator(
        task_id='setup_redis_cache',
        python_callable=setup_redis_cache,
    )

    t_save_config = PythonOperator(
        task_id='save_llm_config',
        python_callable=save_llm_config,
    )

    # Dependencies
    t_generate_prompts >> [t_validate_deepseek, t_validate_claude, t_setup_cache] >> t_save_config
