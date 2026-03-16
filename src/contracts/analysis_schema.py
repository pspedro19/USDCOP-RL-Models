"""
Analysis Module Contract Schema (SDD-07/08)
=============================================
Dataclasses for weekly/daily analysis, macro snapshots, and chat.
Mirrors the database schema from migration 046.

Contract: CTR-ANALYSIS-SCHEMA-001
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Macro variable snapshot
# ---------------------------------------------------------------------------

@dataclass
class MacroSnapshot:
    """Technical indicators for a single macro variable on a single date."""
    snapshot_date: date
    variable_key: str
    variable_name: str
    value: float

    # Simple Moving Averages
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None

    # Bollinger Bands (20-period, 2 std)
    bollinger_upper_20: Optional[float] = None
    bollinger_lower_20: Optional[float] = None
    bollinger_width_20: Optional[float] = None

    # RSI (Wilder's, 14-period)
    rsi_14: Optional[float] = None

    # MACD (12, 26, 9)
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Rate of Change
    roc_5: Optional[float] = None
    roc_20: Optional[float] = None

    # Derived signals
    z_score_20: Optional[float] = None
    trend: Optional[str] = None       # above_sma20, below_sma20, golden_cross, death_cross
    signal: Optional[str] = None      # overbought, oversold, neutral, bb_squeeze

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert date to string
        if isinstance(d.get("snapshot_date"), date):
            d["snapshot_date"] = d["snapshot_date"].isoformat()
        return d

    def to_chart_point(self) -> dict:
        """Return minimal dict for Recharts frontend."""
        return {
            "date": self.snapshot_date.isoformat() if isinstance(self.snapshot_date, date) else str(self.snapshot_date),
            "value": self.value,
            "sma20": self.sma_20,
            "bb_upper": self.bollinger_upper_20,
            "bb_lower": self.bollinger_lower_20,
            "rsi": self.rsi_14,
        }


# ---------------------------------------------------------------------------
# Daily analysis
# ---------------------------------------------------------------------------

@dataclass
class DailyAnalysisRecord:
    """AI-generated daily analysis entry."""
    analysis_date: date
    iso_year: int
    iso_week: int
    day_of_week: int                       # 0=Mon, 4=Fri

    headline: Optional[str] = None
    summary_markdown: Optional[str] = None
    sentiment: Optional[str] = None        # bullish, bearish, neutral, mixed

    usdcop_close: Optional[float] = None
    usdcop_change_pct: Optional[float] = None
    usdcop_high: Optional[float] = None
    usdcop_low: Optional[float] = None

    h1_signal: dict = field(default_factory=dict)
    h5_status: dict = field(default_factory=dict)
    macro_publications: list = field(default_factory=list)
    economic_events: list = field(default_factory=list)
    news_highlights: list = field(default_factory=list)

    llm_model: Optional[str] = None
    llm_tokens_used: Optional[int] = None
    llm_cost_usd: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(d.get("analysis_date"), date):
            d["analysis_date"] = d["analysis_date"].isoformat()
        return d


# ---------------------------------------------------------------------------
# Weekly analysis
# ---------------------------------------------------------------------------

@dataclass
class WeeklyAnalysisRecord:
    """AI-generated weekly analysis report."""
    iso_year: int
    iso_week: int
    week_start: date
    week_end: date

    summary_markdown: Optional[str] = None
    headline: Optional[str] = None
    sentiment: Optional[str] = None        # bullish, bearish, neutral, mixed
    themes: list = field(default_factory=list)

    ohlcv_summary: dict = field(default_factory=dict)
    h5_signal: dict = field(default_factory=dict)
    h1_signals: list = field(default_factory=list)
    news_summary: dict = field(default_factory=dict)

    llm_model: Optional[str] = None
    llm_tokens_used: Optional[int] = None
    llm_cost_usd: Optional[float] = None
    generation_time_s: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("week_start", "week_end"):
            if isinstance(d.get(key), date):
                d[key] = d[key].isoformat()
        return d


# ---------------------------------------------------------------------------
# Chat message
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single chat message."""
    role: str          # user, assistant, system
    content: str
    session_id: str
    context_year: Optional[int] = None
    context_week: Optional[int] = None
    tokens_used: Optional[int] = None
    llm_model: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


# ---------------------------------------------------------------------------
# Weekly view (full JSON export for dashboard)
# ---------------------------------------------------------------------------

@dataclass
class WeeklyViewExport:
    """Complete weekly view exported as JSON for dashboard consumption.

    File: public/data/analysis/weekly_YYYY_WXX.json
    """
    weekly_summary: dict = field(default_factory=dict)
    daily_entries: list = field(default_factory=list)
    macro_snapshots: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    upcoming_events: list = field(default_factory=list)
    macro_charts: dict = field(default_factory=dict)
    news_context: dict = field(default_factory=dict)

    # Multi-Agent Analysis outputs (Phase 4 — optional, set by LangGraph pipeline)
    technical_analysis: Optional[dict] = None
    mtf_analysis: Optional[dict] = None
    news_intelligence: Optional[dict] = None
    macro_regime: Optional[dict] = None
    fx_context: Optional[dict] = None
    political_bias_analysis: Optional[dict] = None  # Phase 3: Bias detection
    quality_score: Optional[float] = None
    synthesis_markdown: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None optional fields to keep JSON clean
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Analysis index
# ---------------------------------------------------------------------------

@dataclass
class AnalysisIndexEntry:
    """Entry in the analysis_index.json file."""
    year: int
    week: int
    start: str        # ISO date
    end: str          # ISO date
    sentiment: Optional[str] = None
    headline: Optional[str] = None
    has_weekly: bool = False
    daily_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Key macro variables tracked
# ---------------------------------------------------------------------------

KEY_MACRO_VARIABLES = (
    "dxy",           # US Dollar Index
    "vix",           # CBOE Volatility Index
    "wti",           # WTI Crude Oil
    "embi_col",      # EMBI Colombia spread
    "ust10y",        # US 10Y Treasury yield
    "ust2y",         # US 2Y Treasury yield
    "ibr",           # Colombia interbank rate
    "tpm",           # Colombia policy rate
    "fedfunds",      # Fed Funds rate
    "gold",          # Gold price
    "brent",         # Brent crude
    "cpi_us",        # US CPI
    "cpi_col",       # Colombia CPI
)

DISPLAY_NAMES = {
    "dxy":       "DXY (Dollar Index)",
    "vix":       "VIX (Volatilidad)",
    "wti":       "WTI Petroleo",
    "embi_col":  "EMBI Colombia",
    "ust10y":    "US Treasury 10Y",
    "ust2y":     "US Treasury 2Y",
    "ibr":       "IBR (Interbancaria)",
    "tpm":       "Tasa Politica Monetaria",
    "fedfunds":  "Fed Funds Rate",
    "gold":      "Oro",
    "brent":     "Brent",
    "cpi_us":    "CPI (EEUU)",
    "cpi_col":   "CPI (Colombia)",
}


# ---------------------------------------------------------------------------
# Safe JSON helper (reuse from strategy_schema)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-Agent Analysis outputs (Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class TechnicalAnalysisOutput:
    """Technical analysis output from TA agent."""
    current_price: float = 0.0
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    volatility_regime: str = "normal"
    indicators: dict = field(default_factory=dict)
    dominant_bias: str = "neutral"
    bias_confidence: float = 0.0
    bullish_signals: list = field(default_factory=list)
    bearish_signals: list = field(default_factory=list)
    scenarios: list = field(default_factory=list)
    no_trade_zone: tuple = (0.0, 0.0)
    watch_list: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["no_trade_zone"] = list(self.no_trade_zone)
        return d


@dataclass
class NewsClusterOutput:
    """A news cluster output from News agent."""
    label: str = ""
    article_count: int = 0
    avg_sentiment: float = 0.0
    bias_distribution: dict = field(default_factory=dict)
    representative_titles: list = field(default_factory=list)
    narrative_summary: str = ""
    articles: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MacroRegimeOutput:
    """Macro regime output from Macro agent."""
    regime: dict = field(default_factory=dict)
    correlations: dict = field(default_factory=dict)
    granger_leaders: list = field(default_factory=list)
    zscore_alerts: list = field(default_factory=list)
    changepoints: list = field(default_factory=list)
    insights: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FXContextOutput:
    """FX context output from FX agent."""
    carry_trade: dict = field(default_factory=dict)
    oil_impact: dict = field(default_factory=dict)
    banrep: dict = field(default_factory=dict)
    risk_factors: list = field(default_factory=list)
    fx_narrative: str = ""
    cop_weekly_change_pct: Optional[float] = None
    cop_level: Optional[float] = None
    sensitivity_impacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PoliticalBiasOutput:
    """Political bias analysis output from Bias Detection agent (Phase 3)."""
    source_bias_distribution: dict = field(default_factory=dict)  # {left: N, center-left: N, ...}
    bias_diversity_score: float = 0.0                             # 0-1 (1 = balanced)
    factuality_distribution: dict = field(default_factory=dict)   # {high: N, mixed: N, low: N}
    cluster_bias_assessments: list = field(default_factory=list)   # [{cluster_label, bias_label, confidence}]
    flagged_articles: int = 0
    bias_narrative: str = ""
    total_analyzed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FullAnalysisReport:
    """Complete multi-agent analysis report."""
    technical: Optional[dict] = None
    mtf_analysis: Optional[dict] = None
    news: Optional[dict] = None
    macro: Optional[dict] = None
    fx: Optional[dict] = None
    markdown_report: str = ""
    quality_score: float = 0.0
    cost_usd: float = 0.0
    agent_times: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Safe JSON helper (reuse from strategy_schema)
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj):
    """Recursively replace Infinity/NaN with None."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, date) and not isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
