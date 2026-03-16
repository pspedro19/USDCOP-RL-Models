"""
Agent Tools (Multi-Agent System — Phase 3)
============================================
Wraps existing and new components as callable tools for LangGraph agents.
Each tool is a plain function that operates on data, not a LangChain Tool.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Data loading tools
# ---------------------------------------------------------------------------

def load_daily_ohlcv() -> pd.DataFrame:
    """Load USDCOP daily OHLCV from parquet."""
    path = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
    if not path.exists():
        logger.warning(f"Daily OHLCV not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.index.name == "time" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"])
        df = df.set_index("date").sort_index()
    return df


def load_m5_ohlcv(last_n_days: int = 60) -> pd.DataFrame:
    """Load USDCOP 5-min OHLCV from parquet (last N days for MTF analysis)."""
    path = PROJECT_ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet"
    if not path.exists():
        logger.warning(f"5-min OHLCV not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)

    # Filter to USD/COP
    if "symbol" in df.columns:
        df = df[df["symbol"] == "USD/COP"].copy()

    if df.index.name == "time" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

    # Last N days
    if not df.empty and last_n_days > 0:
        cutoff = df.index.max() - pd.Timedelta(days=last_n_days)
        df = df[df.index >= cutoff]

    return df


def load_macro_data() -> pd.DataFrame:
    """Load macro data from MACRO_DAILY_CLEAN.parquet."""
    path = PROJECT_ROOT / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
    if not path.exists():
        logger.warning(f"Macro data not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.set_index("fecha").sort_index()
    return df


def load_gdelt_articles(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Load GDELT articles from CSV, optionally filtered by date range."""
    path = PROJECT_ROOT / "data/news/gdelt_articles_historical.csv"
    if not path.exists():
        return []

    try:
        df = pd.read_csv(
            path,
            usecols=["date", "title", "source", "domain", "language"],
            dtype={"title": str, "source": str, "domain": str, "language": str},
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "title"])

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        df = df.sort_values("date", ascending=False)
        return df.to_dict("records")
    except Exception as e:
        logger.warning(f"Failed to load GDELT articles: {e}")
        return []


def load_gdelt_sentiment(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load GDELT daily sentiment CSV."""
    path = PROJECT_ROOT / "data/news/gdelt_daily_sentiment.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df
    except Exception as e:
        logger.warning(f"Failed to load GDELT sentiment: {e}")
        return pd.DataFrame()


def get_cop_series(daily_df: pd.DataFrame) -> pd.Series:
    """Extract USDCOP close price series from daily OHLCV."""
    if daily_df.empty:
        return pd.Series(dtype=float)
    if "close" in daily_df.columns:
        return daily_df["close"].dropna()
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Analysis tools (wrapping engines)
# ---------------------------------------------------------------------------

def run_technical_analysis(
    daily_df: pd.DataFrame,
    timeframe: str = "1d",
) -> dict:
    """Run full technical analysis on OHLCV data."""
    from src.analysis.technical_engine import TechnicalAnalysisEngine
    engine = TechnicalAnalysisEngine()
    report = engine.analyze(daily_df, timeframe=timeframe)
    return report.to_dict()


def run_multi_timeframe(
    m5_df: pd.DataFrame,
    daily_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Run multi-timeframe analysis."""
    from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
    analyzer = MultiTimeframeAnalyzer()
    timeframes = analyzer.aggregate_timeframes(m5_df)
    report = analyzer.analyze_all_timeframes(timeframes, daily_df=daily_df)
    return report.to_dict()


def run_macro_regime(
    macro_df: pd.DataFrame,
    cop_series: pd.Series,
    week_start: str,
    week_end: str,
) -> dict:
    """Run macro regime analysis."""
    from src.analysis.macro_regime import MacroRegimeEngine
    engine = MacroRegimeEngine()
    report = engine.analyze(macro_df, cop_series, week_start, week_end)
    return report.to_dict()


def run_news_intelligence(
    articles: list[dict],
    week_start: str,
    week_end: str,
    min_relevance: float = 0.3,
) -> dict:
    """Run news intelligence analysis."""
    from src.analysis.news_intelligence import NewsIntelligenceEngine
    engine = NewsIntelligenceEngine()
    report = engine.analyze(articles, week_start, week_end, min_relevance=min_relevance)
    return report.to_dict()


def run_fx_context(
    macro_df: pd.DataFrame,
    cop_series: pd.Series,
    week_start: str,
    week_end: str,
    events_calendar: Optional[list[dict]] = None,
) -> dict:
    """Run FX context analysis."""
    from src.analysis.fx_context import FXContextEngine
    engine = FXContextEngine()
    report = engine.analyze(macro_df, cop_series, week_start, week_end, events_calendar)
    return report.to_dict()


def compute_macro_snapshots(
    macro_df: pd.DataFrame,
    target_date: date,
    variables: Optional[list[str]] = None,
) -> dict:
    """Compute macro snapshots using existing MacroAnalyzer."""
    from src.analysis.macro_analyzer import MacroAnalyzer
    analyzer = MacroAnalyzer()
    snapshots = analyzer.compute_all_snapshots(macro_df, target_date, variables=variables)
    return {k: v.to_dict() for k, v in snapshots.items()}
