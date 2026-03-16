"""
Agent State Schema (Multi-Agent System — Phase 3)
===================================================
Typed state for the LangGraph multi-agent analysis graph.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

import pandas as pd


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
    macro_digest: Optional[dict]           # MacroDigest from preprocessor

    # --- Agent outputs ---
    ta_report: Optional[dict]
    mtf_analysis: Optional[dict]
    news_intelligence: Optional[dict]
    macro_regime: Optional[dict]
    fx_context: Optional[dict]
    political_bias_analysis: Optional[dict]  # Phase 3: Bias detection

    # --- Synthesis (Reflection loop) ---
    synthesis_draft: Optional[str]
    synthesis_critique: Optional[str]
    synthesis_quality: Optional[float]
    synthesis_revision: int
    final_report: Optional[str]

    # --- Injected dependencies (not serialized, used by nodes) ---
    _llm_client: Optional[object]          # Phase 1: Injected LLMClient

    # --- Metadata ---
    # Annotated with operator.add so parallel agents can append concurrently
    execution_log: Annotated[list[str], operator.add]
    cost_tracking: dict
    errors: Annotated[list[str], operator.add]
