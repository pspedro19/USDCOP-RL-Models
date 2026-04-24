"""
Agent State Schema (Multi-Agent System — Phase 3)
===================================================
Typed state for the LangGraph multi-agent analysis graph.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class MasterAnalysisState(TypedDict, total=False):
    """Full state flowing through the LangGraph analysis pipeline.

    All fields are optional (total=False) since they're populated
    at different stages of the graph.
    """
    # --- Inputs (set by the entry node) ---
    symbol: str
    week_start: str
    week_end: str
    iso_year: int
    iso_week: int

    # Pre-loaded data (serialized as dicts to avoid DataFrame in state)
    ohlcv_daily: list[dict]        # Daily OHLCV rows
    ohlcv_m5: list[dict]           # 5-min OHLCV rows (for multi-timeframe)
    macro_data: list[dict]         # Macro indicators rows
    cop_prices: list[dict]         # USDCOP close {date, close}
    gdelt_articles: list[dict]     # Raw GDELT articles
    events_calendar: list[dict]    # Economic events

    # --- Pre-loaded data (Phase 0: MacroDataPreprocessor) ---
    macro_digest: dict | None           # MacroDigest from preprocessor

    # --- Agent outputs ---
    ta_report: dict | None
    mtf_analysis: dict | None
    news_intelligence: dict | None
    macro_regime: dict | None
    fx_context: dict | None
    political_bias_analysis: dict | None  # Phase 3: Bias detection

    # --- Synthesis (Reflection loop) ---
    synthesis_draft: str | None
    synthesis_critique: str | None
    synthesis_quality: float | None
    synthesis_revision: int
    final_report: str | None

    # --- Injected dependencies (not serialized, used by nodes) ---
    _llm_client: object | None          # Phase 1: Injected LLMClient

    # --- Metadata ---
    # Annotated with operator.add so parallel agents can append concurrently
    execution_log: Annotated[list[str], operator.add]
    cost_tracking: dict
    errors: Annotated[list[str], operator.add]
