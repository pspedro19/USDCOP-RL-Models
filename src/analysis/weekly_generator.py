"""
Weekly Analysis Generator (SDD-07 §5)
========================================
Orchestrates the full analysis pipeline:
1. Load macro data from existing tables/parquets
2. Compute technical indicators (MacroAnalyzer)
3. Load OHLCV data for the week
4. Load model signals (H1, H5)
5. Load news context from GDELT data
6. Generate daily + weekly AI analysis via LLM
7. Generate charts (matplotlib PNGs)
8. Export JSON files for dashboard
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.chart_generator import generate_all_charts
from src.analysis.llm_client import LLMClient
from src.analysis.macro_analyzer import MacroAnalyzer
from src.analysis.prompt_templates import (
    DAILY_OUTPUT_SCHEMA,
    DAILY_TEMPLATE,
    DAILY_TEMPLATE_V2,
    SYSTEM_DAILY,
    SYSTEM_DAILY_V2,
    SYSTEM_WEEKLY,
    SYSTEM_WEEKLY_V2,
    WEEKLY_OUTPUT_SCHEMA,
    WEEKLY_TEMPLATE,
    WEEKLY_TEMPLATE_V2,
    WEEKLY_TEMPLATE_V3,
    build_events_section,
    build_macro_section,
    build_macro_table_v2,
    build_news_section,
    build_prior_week_section,
    build_regime_section,
    build_signal_section,
)
from src.analysis.sources import build_fuentes_section
from src.contracts.analysis_schema import (
    AnalysisIndexEntry,
    DailyAnalysisRecord,
    WeeklyViewExport,
    _sanitize_for_json,
)

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 8 daily-frequency variables for chart generation
CHART_VARIABLES = ("dxy", "vix", "wti", "embi_col", "ust10y", "ibr", "gold", "brent")


class WeeklyAnalysisGenerator:
    """Orchestrates the full weekly analysis pipeline.

    Supports three execution modes:
    - default: Dual-track (existing pipeline + LangGraph enrichment)
    - legacy: Existing pipeline only (--legacy-pipeline)
    - langgraph: LangGraph only (--langgraph-only)
    """

    def __init__(
        self,
        config_path: str | None = None,
        llm_client: LLMClient | None = None,
        dry_run: bool = False,
        mode: str = "default",
    ):
        # Load config
        config_path = config_path or str(PROJECT_ROOT / "config/analysis/weekly_analysis_ssot.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.macro_analyzer = MacroAnalyzer(
            sma_periods=self.config["macro"]["sma_periods"],
            bollinger_period=self.config["macro"]["bollinger_period"],
            bollinger_std=self.config["macro"]["bollinger_std"],
            rsi_period=self.config["macro"]["rsi_period"],
            macd_fast=self.config["macro"]["macd_fast"],
            macd_slow=self.config["macro"]["macd_slow"],
            macd_signal=self.config["macro"]["macd_signal"],
            roc_periods=self.config["macro"]["roc_periods"],
            lookback_days=self.config["macro"]["lookback_days"],
        )

        cache_dir = str(PROJECT_ROOT / self.config["llm"]["cache"]["cache_dir"])
        if llm_client is None:
            from src.analysis.llm_client import AzureOpenAIProvider
            azure_cfg = self.config["llm"].get("azure_openai", {})
            primary = AzureOpenAIProvider(
                deployment=azure_cfg.get("deployment", "gpt-4o-mini"),
                api_version=azure_cfg.get("api_version", "2024-12-01-preview"),
            )
            self.llm = LLMClient(primary=primary, cache_dir=cache_dir)
        else:
            self.llm = llm_client
        self.dry_run = dry_run
        self.mode = mode  # "default", "legacy", or "langgraph"
        self.export_dir = PROJECT_ROOT / self.config["export"]["output_dir"]

        # Lazy-loaded caches
        self._h5_trades_cache: dict | None = None
        self._h1_forward_cache: pd.DataFrame | None = None
        self._gdelt_sentiment_cache: pd.DataFrame | None = None
        self._all_articles_cache: pd.DataFrame | None = None
        self._calendar_cache: list | None = None
        self._charts_generated = 0
        self._llm_sentiment_scores: dict | None = None  # title_hash -> score

    def generate_for_date(self, target_date: date) -> DailyAnalysisRecord:
        """Generate daily analysis for a specific date."""
        iso_cal = target_date.isocalendar()
        logger.info(f"Generating daily analysis for {target_date}")

        # 1. Load macro data
        macro_df = self._load_macro_data()
        snapshots = self.macro_analyzer.compute_all_snapshots(
            macro_df, target_date,
            variables=self.config["macro"]["key_variables"],
        )

        # 2. Load OHLCV
        ohlcv = self._load_daily_ohlcv(target_date)

        # 3. Load signals
        h5_signal = self._load_h5_signal(iso_cal[0], iso_cal[1])
        h1_signal = self._load_h1_signals(target_date, target_date)

        # 4. Load news highlights
        news_highlights = self._load_news_highlights_for_day(target_date)

        # 5. Load upcoming events
        events_for_date = self._load_upcoming_events(target_date, window_days=1)

        # 6. Build LLM prompt
        signal_section = build_signal_section(h1_signal=h1_signal, h5_signal=h5_signal)
        news_context = {
            "article_count": len(news_highlights),
            "avg_sentiment": (
                np.mean([h.get("sentiment", h.get("tone", 0)) for h in news_highlights])
                if news_highlights else 0
            ),
        }
        news_section = build_news_section(news_context, highlights=news_highlights) if news_highlights else "- Sin noticias relevantes para hoy"
        events_section = build_events_section(events_for_date)

        use_v2 = self.config.get("prompts", {}).get("version") == "v2"
        use_structured = self.config.get("prompts", {}).get("structured_output", False) and use_v2

        if use_v2:
            macro_digest_text = self._compute_macro_digest_text(macro_df, target_date)
            user_prompt = DAILY_TEMPLATE_V2.format(
                date=target_date.isoformat(),
                close=ohlcv.get("close", "N/A"),
                change_pct=ohlcv.get("change_pct", "N/A"),
                low=ohlcv.get("low", "N/A"),
                high=ohlcv.get("high", "N/A"),
                macro_digest_text=macro_digest_text,
                signal_section=signal_section,
                news_section=news_section,
                events_section=events_section,
            )
            system_prompt = SYSTEM_DAILY_V2
        else:
            macro_section = build_macro_section(snapshots)
            user_prompt = DAILY_TEMPLATE.format(
                date=target_date.isoformat(),
                close=ohlcv.get("close", "N/A"),
                change_pct=ohlcv.get("change_pct", "N/A"),
                low=ohlcv.get("low", "N/A"),
                high=ohlcv.get("high", "N/A"),
                macro_section=macro_section,
                signal_section=signal_section,
                news_section=news_section,
                events_section=events_section,
            )
            system_prompt = SYSTEM_DAILY

        # 7. Generate with LLM
        if self.dry_run:
            content = f"[DRY RUN] Analisis diario para {target_date}"
            llm_result = {"content": content, "tokens_used": 0, "model": "dry_run", "cost_usd": 0}
            headline = "Analisis no disponible (dry run)"
            sentiment = "neutral"
        elif use_structured:
            cache_key = f"daily_v2_{target_date.isoformat()}"
            llm_result = self.llm.generate_structured(
                system_prompt, user_prompt,
                response_format=DAILY_OUTPUT_SCHEMA,
                max_tokens=self.config["generation"]["daily"]["max_tokens"],
                cache_key=cache_key,
            )
            parsed = llm_result.get("content", {})
            if isinstance(parsed, dict):
                headline = parsed.get("headline", self._extract_headline(
                    parsed.get("analysis_markdown", "")))
                sentiment = parsed.get("sentiment_label", self._detect_sentiment(
                    parsed.get("analysis_markdown", "")))
                llm_result["content"] = parsed.get("analysis_markdown", str(parsed))
            else:
                headline = self._extract_headline(str(parsed))
                sentiment = self._detect_sentiment(str(parsed))
                llm_result["content"] = str(parsed)
        else:
            cache_key = f"daily_{target_date.isoformat()}"
            llm_result = self.llm.generate(
                system_prompt, user_prompt,
                max_tokens=self.config["generation"]["daily"]["max_tokens"],
                cache_key=cache_key,
            )
            headline = self._extract_headline(llm_result["content"])
            sentiment = self._detect_sentiment(llm_result["content"])

        # 8. Build record
        record = DailyAnalysisRecord(
            analysis_date=target_date,
            iso_year=iso_cal[0],
            iso_week=iso_cal[1],
            day_of_week=target_date.weekday(),
            headline=headline,
            summary_markdown=llm_result["content"],
            sentiment=sentiment,
            usdcop_close=ohlcv.get("close"),
            usdcop_change_pct=ohlcv.get("change_pct"),
            usdcop_high=ohlcv.get("high"),
            usdcop_low=ohlcv.get("low"),
            h1_signal=h1_signal,
            h5_status=h5_signal,
            macro_publications=[snap.to_dict() for snap in snapshots.values()],
            economic_events=events_for_date,
            news_highlights=news_highlights,
            llm_model=llm_result.get("model"),
            llm_tokens_used=llm_result.get("tokens_used"),
            llm_cost_usd=llm_result.get("cost_usd"),
        )

        logger.info(f"Daily analysis generated: {target_date} ({llm_result.get('tokens_used', 0)} tokens)")
        return record

    def generate_for_week(
        self,
        iso_year: int,
        iso_week: int,
    ) -> WeeklyViewExport:
        """Generate full weekly analysis (daily entries + weekly summary + charts).

        Returns a WeeklyViewExport ready for JSON serialization.
        """
        start = date.fromisocalendar(iso_year, iso_week, 1)
        end = date.fromisocalendar(iso_year, iso_week, 5)  # Friday
        logger.info(f"Generating weekly analysis: {iso_year}-W{iso_week:02d} ({start} to {end})")

        start_time = time.time()

        # 0. Enrich articles with LLM batch sentiment (single call, cached)
        self._enrich_articles_with_llm_sentiment(iso_year, iso_week, start, end)

        # 1. Load macro data
        macro_df = self._load_macro_data()

        # 2. Generate daily entries (Mon-Fri)
        daily_entries = []
        current = start
        while current <= min(end, date.today()):
            if current.weekday() < 5:  # Mon-Fri
                try:
                    daily = self.generate_for_date(current)
                    daily_entries.append(daily.to_dict())
                except Exception as e:
                    logger.warning(f"Daily generation failed for {current}: {e}")
            current += timedelta(days=1)

        # 3. Compute macro snapshots for all chart variables
        all_vars = list(CHART_VARIABLES) + [
            v for v in self.config["macro"].get("key_variables", [])
            if v not in CHART_VARIABLES
        ]
        snapshots = self.macro_analyzer.compute_all_snapshots(
            macro_df, min(end, date.today()),
            variables=all_vars,
        )
        macro_snapshots_dict = {}
        for key, snap in snapshots.items():
            col = self.macro_analyzer._find_column(macro_df, key)
            if col:
                chart_data = self.macro_analyzer.get_chart_data(
                    macro_df[col].dropna(),
                    key, min(end, date.today()),
                )
            else:
                chart_data = []
            macro_snapshots_dict[key] = {
                **snap.to_dict(),
                "chart_data": chart_data,
            }

        # 4. Generate charts (matplotlib PNGs) for 8 variables
        week_label = f"{iso_year}-W{iso_week:02d}"
        charts_dir = str(self.export_dir / "charts")
        var_cols = {}
        for var_key in CHART_VARIABLES:
            col = self.macro_analyzer._find_column(macro_df, var_key)
            if col:
                var_cols[var_key] = col

        chart_paths = {}
        if self.config["charts"]["enabled"]:
            chart_paths = generate_all_charts(
                macro_df, var_cols, min(end, date.today()),
                charts_dir, week_label,
            )
            self._charts_generated += len(chart_paths)

        # Add PNG URLs to macro snapshots
        for var_key, path in chart_paths.items():
            if var_key in macro_snapshots_dict:
                macro_snapshots_dict[var_key]["png_url"] = (
                    f"/data/analysis/charts/macro_{var_key}_{week_label}.png"
                )

        # 5. Load signals + news + events
        h5_signal = self._load_h5_signal(iso_year, iso_week)
        h1_signals = self._load_h1_signals(start, end)
        news_context = self._load_news_context(start, end)
        upcoming_events = self._load_upcoming_events(end)

        # 5b. Load prior week JSON for V3 prompt continuity
        prior_json = self._load_prior_week_json(iso_year, iso_week)

        # 5c. Compute regime directly (bypasses LangGraph which may be unavailable)
        regime_data = self._compute_regime_direct(macro_df, start, end)

        # 6. Generate weekly summary via LLM
        ohlcv_summary = self._load_weekly_ohlcv(start, end)
        weekly_llm = self._generate_weekly_summary(
            iso_year, iso_week, start, end,
            snapshots, ohlcv_summary, daily_entries,
            h5_signal=h5_signal,
            h1_signals=h1_signals,
            news_context=news_context,
            upcoming_events=upcoming_events,
            prior_json=prior_json,
            regime_data=regime_data,
            macro_snapshots_dict=macro_snapshots_dict,
        )

        # 7. Build export
        generation_time = time.time() - start_time
        weekly_summary_dict = {
            "headline": weekly_llm.get("headline", ""),
            "markdown": weekly_llm.get("content", ""),
            "sentiment": weekly_llm.get("sentiment", "neutral"),
            "themes": weekly_llm.get("themes", []),
            "ohlcv": ohlcv_summary,
        }
        # V2 additions: scenarios, key_drivers, sentiment_score
        if weekly_llm.get("scenarios"):
            weekly_summary_dict["scenarios"] = weekly_llm["scenarios"]
        if weekly_llm.get("key_drivers"):
            weekly_summary_dict["key_drivers"] = weekly_llm["key_drivers"]
        if weekly_llm.get("sentiment_score") is not None:
            weekly_summary_dict["sentiment_score"] = weekly_llm["sentiment_score"]

        export = WeeklyViewExport(
            weekly_summary=weekly_summary_dict,
            daily_entries=daily_entries,
            macro_snapshots=macro_snapshots_dict,
            signals={
                "h5": h5_signal,
                "h1": h1_signals,
            },
            upcoming_events=upcoming_events,
            macro_charts={
                var_key: {
                    "png_url": macro_snapshots_dict.get(var_key, {}).get("png_url"),
                    "data": macro_snapshots_dict.get(var_key, {}).get("chart_data", []),
                }
                for var_key in var_cols
            },
            news_context=news_context,
        )

        # 7a. Attach regime data and prior outlook accuracy to export
        if regime_data:
            export.macro_regime = regime_data
        if prior_json:
            # Compute prior outlook accuracy if prior had scenarios and we have this week's OHLCV
            prior_accuracy = self._evaluate_prior_outlook(prior_json, ohlcv_summary)
            if prior_accuracy:
                if isinstance(export.weekly_summary, dict):
                    export.weekly_summary["prior_outlook_accuracy"] = prior_accuracy

        # 7b. LangGraph multi-agent enrichment (dual-track or langgraph-only)
        if self.mode != "legacy":
            self._enrich_with_langgraph(export, iso_year, iso_week, start, end)

        # 8. Save JSON export
        self._save_weekly_export(export, iso_year, iso_week, start, end, weekly_llm)

        logger.info(
            f"Weekly analysis complete: {week_label} "
            f"({len(daily_entries)} daily, {len(chart_paths)} charts, "
            f"{generation_time:.1f}s, ${self.llm.total_cost:.4f})"
        )

        return export

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_weekly_summary(
        self,
        iso_year: int,
        iso_week: int,
        start: date,
        end: date,
        snapshots: dict,
        ohlcv_summary: dict,
        daily_entries: list,
        h5_signal: dict | None = None,
        h1_signals: dict | None = None,
        news_context: dict | None = None,
        upcoming_events: list | None = None,
        prior_json: dict | None = None,
        regime_data: dict | None = None,
        macro_snapshots_dict: dict | None = None,
    ) -> dict:
        """Generate weekly summary via LLM (V1, V2 structured, or V3 enhanced)."""
        signal_section = build_signal_section(
            h1_signal=h1_signals or {},
            h5_signal=h5_signal or {},
        )
        news_highlights = (news_context or {}).get("highlights", [])
        news_section = build_news_section(news_context or {}, highlights=news_highlights)
        events_section = build_events_section(upcoming_events or [])

        ohlcv_text = (
            f"- Apertura: {ohlcv_summary.get('open', 'N/A')}\n"
            f"- Cierre: {ohlcv_summary.get('close', 'N/A')}\n"
            f"- Maximo: {ohlcv_summary.get('high', 'N/A')}\n"
            f"- Minimo: {ohlcv_summary.get('low', 'N/A')}\n"
            f"- Cambio: {ohlcv_summary.get('change_pct', 'N/A')}%"
        )

        prompt_version = self.config.get("prompts", {}).get("version", "v1")
        use_v3 = prompt_version == "v3"
        use_v2 = prompt_version == "v2"
        use_structured = self.config.get("prompts", {}).get("structured_output", False) and use_v2

        if use_v3:
            # V3 enhanced template with regime, prior week, enhanced macro table
            prior_week_section = build_prior_week_section(prior_json)
            regime_section = build_regime_section(regime_data)
            macro_table = build_macro_table_v2(macro_snapshots_dict or snapshots)

            # Build compressed daily summaries
            daily_summaries_lines = []
            for de in daily_entries:
                d_date = de.get("analysis_date", "?")
                d_headline = de.get("headline", "")
                d_sentiment = de.get("sentiment", "neutral")
                d_close = de.get("usdcop_close", "N/A")
                d_change = de.get("usdcop_change_pct", "N/A")
                daily_summaries_lines.append(
                    f"- **{d_date}**: {d_headline} | COP: {d_close} ({d_change}%) [{d_sentiment}]"
                )
            daily_summaries = "\n".join(daily_summaries_lines) if daily_summaries_lines else "- Sin resumenes diarios disponibles"

            # Compute next week label
            next_week = iso_week + 1
            next_year = iso_year
            # Handle year boundary (ISO weeks can be 52 or 53)
            try:
                date.fromisocalendar(next_year, next_week, 1)
            except ValueError:
                next_week = 1
                next_year = iso_year + 1
            next_week_label = f"Semana {next_week} de {next_year}"

            user_prompt = WEEKLY_TEMPLATE_V3.format(
                week=iso_week, year=iso_year,
                start=start.isoformat(), end=end.isoformat(),
                prior_week_section=prior_week_section,
                regime_section=regime_section,
                ohlcv_section=ohlcv_text,
                macro_table=macro_table,
                news_section=news_section,
                signal_section=signal_section,
                daily_summaries=daily_summaries,
                next_week_label=next_week_label,
            )
            system_prompt = SYSTEM_WEEKLY
        elif use_v2:
            macro_df = self._load_macro_data()
            macro_digest_text = self._compute_macro_digest_text(macro_df, end)
            user_prompt = WEEKLY_TEMPLATE_V2.format(
                week=iso_week, year=iso_year,
                start=start.isoformat(), end=end.isoformat(),
                ohlcv_section=ohlcv_text,
                macro_digest_text=macro_digest_text,
                signal_section=signal_section,
                news_section=news_section,
                events_section=events_section,
            )
            system_prompt = SYSTEM_WEEKLY_V2
        else:
            macro_section = build_macro_section(snapshots)
            user_prompt = WEEKLY_TEMPLATE.format(
                week=iso_week, year=iso_year,
                start=start.isoformat(), end=end.isoformat(),
                ohlcv_section=ohlcv_text,
                macro_section=macro_section,
                signal_section=signal_section,
                news_section=news_section,
                events_section=events_section,
            )
            system_prompt = SYSTEM_WEEKLY

        if self.dry_run:
            return {
                "content": f"[DRY RUN] Resumen semanal {iso_year}-W{iso_week:02d}",
                "headline": "Resumen no disponible (dry run)",
                "sentiment": "neutral",
                "themes": [],
            }

        if use_structured:
            cache_key = f"weekly_v2_{iso_year}_W{iso_week:02d}"
            result = self.llm.generate_structured(
                system_prompt, user_prompt,
                response_format=WEEKLY_OUTPUT_SCHEMA,
                max_tokens=self.config["generation"]["weekly"]["max_tokens"],
                cache_key=cache_key,
            )
            parsed = result.get("content", {})
            if isinstance(parsed, dict):
                return {
                    "content": parsed.get("analysis_markdown", str(parsed)),
                    "headline": parsed.get("headline", ""),
                    "sentiment": parsed.get("sentiment_label", "neutral"),
                    "themes": [
                        t.get("theme", "") if isinstance(t, dict) else str(t)
                        for t in parsed.get("themes", [])
                    ],
                    "scenarios": parsed.get("scenarios"),
                    "key_drivers": parsed.get("key_drivers", []),
                    "sentiment_score": parsed.get("sentiment_score"),
                }
            else:
                content = str(parsed)
                return {
                    "content": content,
                    "headline": self._extract_headline(content),
                    "sentiment": self._detect_sentiment(content),
                    "themes": self._extract_themes(content),
                }
        else:
            # V1 or V3 (both use markdown, not structured JSON)
            version_tag = "v3" if use_v3 else "v1"
            cache_key = f"weekly_{version_tag}_{iso_year}_W{iso_week:02d}"
            result = self.llm.generate(
                system_prompt, user_prompt,
                max_tokens=self.config["generation"]["weekly"]["max_tokens"],
                cache_key=cache_key,
            )
            content = result["content"]

            # V3: Append deterministic fuentes section after LLM generation
            if use_v3:
                macro_keys = list((macro_snapshots_dict or snapshots or {}).keys())
                fuentes = build_fuentes_section(
                    macro_keys=macro_keys,
                    news_highlights=news_highlights,
                )
                content = content + "\n\n" + fuentes

            return {
                "content": content,
                "headline": self._extract_headline(content),
                "sentiment": self._detect_sentiment(content),
                "themes": self._extract_themes(content),
            }

    def _compute_macro_digest_text(self, macro_df: pd.DataFrame, as_of: date) -> str:
        """Compute macro digest text for V2 chain-of-thought prompts.

        Uses MacroDataPreprocessor to generate structured digest, then formats
        as prompt-injection text for the LLM.
        """
        try:
            from src.analysis.macro_preprocessor import MacroDataPreprocessor

            preprocessor = MacroDataPreprocessor(analyzer=self.macro_analyzer)

            # Get COP close series from daily OHLCV
            cop_series = pd.Series(dtype=float)
            ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
            if ohlcv_path.exists():
                daily_df = pd.read_parquet(ohlcv_path)
                if isinstance(daily_df.index, pd.DatetimeIndex):
                    daily_df = daily_df.reset_index()
                if "time" in daily_df.columns:
                    daily_df["date"] = pd.to_datetime(daily_df["time"])
                    daily_df = daily_df.set_index("date").sort_index()
                if "close" in daily_df.columns:
                    cop_series = daily_df["close"].dropna()

            if macro_df.empty or cop_series.empty:
                return build_macro_section(
                    self.macro_analyzer.compute_all_snapshots(
                        macro_df, as_of,
                        variables=self.config["macro"]["key_variables"],
                    )
                )

            digest = preprocessor.compute_digest(macro_df, cop_series, as_of)
            return digest.to_prompt_text()

        except Exception as e:
            logger.warning(f"Macro digest computation failed, falling back to V1 format: {e}")
            snapshots = self.macro_analyzer.compute_all_snapshots(
                macro_df, as_of,
                variables=self.config["macro"]["key_variables"],
            )
            return build_macro_section(snapshots)

    def _enrich_with_langgraph(
        self,
        export: WeeklyViewExport,
        iso_year: int,
        iso_week: int,
        start: date,
        end: date,
    ) -> None:
        """Run LangGraph multi-agent pipeline and merge outputs into the export.

        This runs in parallel with (or instead of) the existing pipeline.
        Failures are non-blocking — the existing export remains valid.
        """
        try:
            from src.analysis.agent_graph import run_analysis_graph
        except ImportError as e:
            logger.warning(f"LangGraph not available (missing dependencies): {e}")
            return

        try:
            logger.info(f"Running LangGraph multi-agent analysis for {iso_year}-W{iso_week:02d}")
            lg_start = time.time()

            # Phase 1: Prepare pre-loaded data for LangGraph agents
            preloaded = self._prepare_preloaded_data(start, end)

            result = run_analysis_graph(
                iso_year=iso_year,
                iso_week=iso_week,
                dry_run=self.dry_run,
                preloaded_data=preloaded,
                llm_client=self.llm,
            )

            lg_time = time.time() - lg_start
            lg_errors = result.get("errors", [])

            # Extract agent outputs from LangGraph state
            export_dict = export.to_dict()

            if result.get("ta_report"):
                export_dict["technical_analysis"] = result["ta_report"]

            if result.get("mtf_analysis"):
                export_dict["mtf_analysis"] = result["mtf_analysis"]

            if result.get("news_intelligence"):
                export_dict["news_intelligence"] = result["news_intelligence"]

            if result.get("macro_regime"):
                export_dict["macro_regime"] = result["macro_regime"]

            if result.get("fx_context"):
                export_dict["fx_context"] = result["fx_context"]

            if result.get("synthesis_quality") is not None:
                export_dict["quality_score"] = result["synthesis_quality"]

            if result.get("final_report"):
                export_dict["synthesis_markdown"] = result["final_report"]
                # Also update weekly_summary.markdown so the header renders the richer synthesis
                if "weekly_summary" in export_dict and isinstance(export_dict["weekly_summary"], dict):
                    export_dict["weekly_summary"]["markdown"] = result["final_report"]
                # Update the export dataclass directly (to_dict creates a copy)
                if isinstance(export.weekly_summary, dict):
                    export.weekly_summary["markdown"] = result["final_report"]

            # Phase 3: Bias analysis
            if result.get("political_bias_analysis"):
                export_dict["political_bias_analysis"] = result["political_bias_analysis"]

            # Re-populate export fields from enriched dict
            for key in (
                "technical_analysis", "mtf_analysis", "news_intelligence",
                "macro_regime", "fx_context", "quality_score", "synthesis_markdown",
                "political_bias_analysis",
            ):
                if key in export_dict:
                    # WeeklyViewExport uses a flat dict, so we store in the dict
                    # that will be serialized. We set it as an attribute on the export.
                    setattr(export, key, export_dict[key])

            cost_info = result.get("cost_tracking", {})
            total_cost = cost_info.get("total_usd", 0)

            agents_ok = sum(
                1 for k in ("ta_report", "news_intelligence", "macro_regime", "fx_context")
                if result.get(k) is not None
            )

            logger.info(
                f"LangGraph enrichment complete: {agents_ok}/4 agents succeeded, "
                f"quality={result.get('synthesis_quality', 'N/A')}, "
                f"cost=${total_cost:.4f}, time={lg_time:.1f}s"
            )

            if lg_errors:
                logger.warning(f"LangGraph errors ({len(lg_errors)}): {lg_errors[:3]}")

        except Exception as e:
            logger.warning(f"LangGraph enrichment failed (non-blocking): {e}")

    def _prepare_preloaded_data(self, start: date, end: date) -> dict:
        """Prepare pre-loaded data dict for LangGraph agents.

        Converts DataFrames to list[dict] records for LangGraph state serialization.
        Reuses existing cached data loaders to avoid redundant file I/O.
        """
        preloaded = {}

        # 1. Daily OHLCV
        try:
            ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
            if ohlcv_path.exists():
                daily_df = pd.read_parquet(ohlcv_path)
                if isinstance(daily_df.index, pd.DatetimeIndex):
                    daily_df = daily_df.reset_index()
                if "time" in daily_df.columns:
                    daily_df["date"] = pd.to_datetime(daily_df["time"])
                    daily_df = daily_df.set_index("date").sort_index()
                preloaded["ohlcv_daily"] = daily_df.reset_index().to_dict("records")

                # COP series from daily OHLCV
                if "close" in daily_df.columns:
                    cop_series = daily_df["close"].dropna()
                    preloaded["cop_prices"] = [
                        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "close": float(v)}
                        for d, v in cop_series.items()
                    ]
        except Exception as e:
            logger.warning(f"Preload daily OHLCV failed: {e}")

        # 2. 5-min OHLCV (last 60 days for MTF analysis)
        try:
            m5_path = PROJECT_ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet"
            if m5_path.exists():
                m5_df = pd.read_parquet(m5_path)
                if "symbol" in m5_df.columns:
                    m5_df = m5_df[m5_df["symbol"] == "USD/COP"].copy()
                if isinstance(m5_df.index, pd.DatetimeIndex):
                    m5_df = m5_df.reset_index()
                if "time" in m5_df.columns:
                    m5_df["time"] = pd.to_datetime(m5_df["time"])
                    m5_df = m5_df.set_index("time").sort_index()
                if not m5_df.empty:
                    cutoff = m5_df.index.max() - pd.Timedelta(days=60)
                    m5_df = m5_df[m5_df.index >= cutoff]
                    preloaded["ohlcv_m5"] = m5_df.reset_index().to_dict("records")
        except Exception as e:
            logger.warning(f"Preload 5-min OHLCV failed: {e}")

        # 3. Macro data
        try:
            macro_df = self._load_macro_data()
            if not macro_df.empty:
                preloaded["macro_data"] = macro_df.reset_index().to_dict("records")
        except Exception as e:
            logger.warning(f"Preload macro data failed: {e}")

        # 4. GDELT articles for the week (include weekend for lag)
        try:
            articles_df = self._get_all_articles()
            if not articles_df.empty:
                news_end = end + timedelta(days=2)
                mask = (
                    (articles_df["date"] >= pd.Timestamp(start))
                    & (articles_df["date"] <= pd.Timestamp(news_end))
                )
                week_articles = articles_df[mask]
                if not week_articles.empty:
                    preloaded["gdelt_articles"] = week_articles.to_dict("records")
        except Exception as e:
            logger.warning(f"Preload articles failed: {e}")

        # 5. Macro digest (Phase 0: MacroDataPreprocessor)
        try:
            from src.analysis.macro_preprocessor import MacroDataPreprocessor

            pp_config = self.config.get("macro_preprocessor", {})
            if pp_config.get("enabled", True):
                preprocessor = MacroDataPreprocessor(analyzer=self.macro_analyzer)
                macro_df = self._load_macro_data()
                cop_series = pd.Series(dtype=float)
                if "cop_prices" in preloaded:
                    cop_df = pd.DataFrame(preloaded["cop_prices"])
                    cop_df["date"] = pd.to_datetime(cop_df["date"])
                    cop_series = cop_df.set_index("date")["close"].sort_index()

                if not macro_df.empty and not cop_series.empty:
                    digest = preprocessor.compute_digest(macro_df, cop_series, end)
                    preloaded["macro_digest"] = digest.to_dict()
                    logger.info(
                        f"Macro digest computed: {len(digest.top_movers)} top movers, "
                        f"{len(digest.anomalies)} anomalies"
                    )
        except Exception as e:
            logger.warning(f"Preload macro digest failed: {e}")

        logger.info(f"Pre-loaded data prepared: {list(preloaded.keys())}")
        return preloaded

    def _save_weekly_export(
        self,
        export: WeeklyViewExport,
        iso_year: int,
        iso_week: int,
        start: date,
        end: date,
        weekly_llm: dict,
    ) -> None:
        """Save weekly JSON export and update index."""
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Weekly file
        pattern = self.config["export"]["files"]["weekly_pattern"]
        filename = pattern.format(year=iso_year, week=iso_week)
        filepath = self.export_dir / filename

        data = _sanitize_for_json(export.to_dict())
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved weekly export: {filepath}")

        # Update index
        self._update_index(iso_year, iso_week, start, end, weekly_llm,
                           daily_count=len(export.daily_entries))

    def _update_index(
        self,
        iso_year: int,
        iso_week: int,
        start: date,
        end: date,
        weekly_llm: dict,
        daily_count: int = 0,
    ) -> None:
        """Update analysis_index.json with the new week."""
        index_path = self.export_dir / self.config["export"]["files"]["index"]

        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)
        else:
            index_data = {"weeks": []}

        # Remove existing entry for this week
        index_data["weeks"] = [
            w for w in index_data["weeks"]
            if not (w["year"] == iso_year and w["week"] == iso_week)
        ]

        # Add new entry
        entry = AnalysisIndexEntry(
            year=iso_year,
            week=iso_week,
            start=start.isoformat(),
            end=end.isoformat(),
            sentiment=weekly_llm.get("sentiment"),
            headline=weekly_llm.get("headline"),
            has_weekly=True,
            daily_count=daily_count,
        )
        index_data["weeks"].append(entry.to_dict())
        index_data["weeks"].sort(key=lambda w: (w["year"], w["week"]), reverse=True)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Prior week, regime, and outlook accuracy helpers
    # ------------------------------------------------------------------

    def _load_prior_week_json(self, iso_year: int, iso_week: int) -> dict | None:
        """Load the previous week's analysis JSON for V3 prompt continuity.

        Handles year boundaries (W01 -> previous year W52/W53).
        """
        prev_week = iso_week - 1
        prev_year = iso_year
        if prev_week < 1:
            prev_year = iso_year - 1
            # ISO year can have 52 or 53 weeks; try 53 first, fall back to 52
            for w in (53, 52):
                try:
                    date.fromisocalendar(prev_year, w, 1)
                    prev_week = w
                    break
                except ValueError:
                    continue
            else:
                prev_week = 52

        pattern = self.config["export"]["files"]["weekly_pattern"]
        filename = pattern.format(year=prev_year, week=prev_week)
        prior_path = self.export_dir / filename

        if prior_path.exists():
            try:
                with open(prior_path, encoding="utf-8") as f:
                    prior_json = json.load(f)
                logger.info(f"Loaded prior week JSON: {prior_path.name}")
                return prior_json
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load prior week JSON {prior_path}: {e}")
                return None
        else:
            logger.info(f"No prior week JSON found at {prior_path}")
            return None

    def _compute_regime_direct(
        self, macro_df: pd.DataFrame, start: date, end: date
    ) -> dict | None:
        """Compute macro regime directly (bypasses LangGraph).

        Calls MacroRegimeEngine.analyze() which needs macro_df, cop_series,
        week_start, week_end. Returns regime_data dict or None.
        """
        try:
            from src.analysis.macro_regime import MacroRegimeEngine

            regime_engine = MacroRegimeEngine()

            # Load COP close series from daily OHLCV
            ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
            if not ohlcv_path.exists():
                logger.warning("Daily OHLCV not found, skipping regime detection")
                return None

            daily_df = pd.read_parquet(ohlcv_path)
            if isinstance(daily_df.index, pd.DatetimeIndex):
                daily_df = daily_df.reset_index()
            if "time" in daily_df.columns:
                daily_df["date"] = pd.to_datetime(daily_df["time"])
                daily_df = daily_df.set_index("date").sort_index()
            if "close" not in daily_df.columns:
                logger.warning("No 'close' column in daily OHLCV, skipping regime detection")
                return None

            cop_series = daily_df["close"].dropna()
            if cop_series.empty:
                return None

            regime_report = regime_engine.analyze(
                macro_df=macro_df,
                cop_series=cop_series,
                week_start=start.isoformat(),
                week_end=end.isoformat(),
            )
            regime_data = regime_report.to_dict() if regime_report else None
            if regime_data:
                logger.info(
                    f"Regime detected: {regime_data.get('regime', {}).get('label', '?')} "
                    f"(confidence: {regime_data.get('regime', {}).get('confidence', 0):.0%})"
                )
            return regime_data

        except ImportError as e:
            logger.warning(f"Regime engine import failed (missing deps like hmmlearn/ruptures): {e}")
            return None
        except Exception as e:
            logger.warning(f"Regime detection failed (non-blocking): {e}")
            return None

    def _evaluate_prior_outlook(
        self, prior_json: dict, current_ohlcv: dict
    ) -> dict | None:
        """Evaluate accuracy of the prior week's outlook vs actual outcome.

        Returns a dict with accuracy metrics, or None if not enough data.
        """
        if not prior_json or not current_ohlcv:
            return None

        ws = prior_json.get("weekly_summary", {})
        scenarios = ws.get("scenarios", {})
        if not scenarios:
            return None

        actual_close = current_ohlcv.get("close")
        actual_change_pct = current_ohlcv.get("change_pct")
        if actual_close is None:
            return None

        result = {
            "actual_close": actual_close,
            "actual_change_pct": actual_change_pct,
        }

        # Check base scenario target
        base = scenarios.get("base", {})
        base_target = base.get("target")
        if base_target is not None and actual_close is not None:
            try:
                error_pct = abs(float(actual_close) - float(base_target)) / float(base_target) * 100
                result["base_target"] = base_target
                result["base_error_pct"] = round(error_pct, 2)
                result["base_hit"] = error_pct < 1.0  # Within 1% of target
            except (ValueError, ZeroDivisionError):
                pass

        # Check which scenario was closest
        bull_target = scenarios.get("bull", {}).get("target")
        bear_target = scenarios.get("bear", {}).get("target")

        targets = {}
        if base_target is not None:
            targets["base"] = float(base_target)
        if bull_target is not None:
            targets["bull"] = float(bull_target)
        if bear_target is not None:
            targets["bear"] = float(bear_target)

        if targets and actual_close is not None:
            closest = min(targets, key=lambda k: abs(targets[k] - float(actual_close)))
            result["closest_scenario"] = closest

        return result

    # ------------------------------------------------------------------
    # Signal / News / Events data loaders
    # ------------------------------------------------------------------

    def _load_h5_signal(self, iso_year: int, iso_week: int) -> dict:
        """Load H5 Smart Simple signal for a specific week from trade files."""
        if self._h5_trades_cache is None:
            self._h5_trades_cache = {}
            for suffix in ("", "_2025"):
                path = PROJECT_ROOT / f"usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11{suffix}.json"
                if path.exists():
                    try:
                        with open(path) as f:
                            data = json.load(f)
                        for trade in data.get("trades", []):
                            ts = trade.get("timestamp", "")
                            try:
                                dt = datetime.fromisoformat(ts)
                                iso = dt.date().isocalendar()
                                key = (iso[0], iso[1])
                                self._h5_trades_cache[key] = trade
                            except (ValueError, TypeError):
                                continue
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Failed to load H5 trades from {path}: {e}")

        trade = self._h5_trades_cache.get((iso_year, iso_week))
        if trade:
            return {
                "direction": trade.get("side", "N/A"),
                "confidence": trade.get("confidence_tier", "N/A"),
                "predicted_return": trade.get("pnl_pct"),
                "leverage": trade.get("leverage"),
                "exit_reason": trade.get("exit_reason"),
                "pnl_pct": trade.get("pnl_pct"),
            }
        return {"direction": "HOLD", "note": "Sin trade esta semana"}

    def _load_h1_signals(self, start: date, end: date) -> dict:
        """Load H1 daily forward forecast signals from bi_dashboard_unified.csv."""
        if self._h1_forward_cache is None:
            csv_path = PROJECT_ROOT / "usdcop-trading-dashboard/public/forecasting/bi_dashboard_unified.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    self._h1_forward_cache = df[
                        (df["view_type"] == "forward_forecast") &
                        (df["horizon_days"] == 1)
                    ].copy()
                except Exception as e:
                    logger.warning(f"Failed to load H1 forward data: {e}")
                    self._h1_forward_cache = pd.DataFrame()
            else:
                self._h1_forward_cache = pd.DataFrame()

        if self._h1_forward_cache.empty:
            return {"direction": "N/A", "note": "Datos no disponibles"}

        # Find the inference week that matches the target range
        target_iso = start.isocalendar()
        target_week_str = f"{target_iso[0]}-W{target_iso[1]:02d}"

        week_data = self._h1_forward_cache[
            self._h1_forward_cache["inference_week"] == target_week_str
        ]

        if week_data.empty:
            return {"direction": "N/A", "note": "Datos no disponibles para esta semana"}

        # Compute ensemble stats across all 9 models for H=1
        avg_da = float(week_data["direction_accuracy"].mean())
        avg_return = float(week_data["total_return"].mean())
        best_model = week_data.loc[week_data["direction_accuracy"].idxmax()]

        direction = "SHORT" if avg_return < 0 else "LONG"

        return {
            "direction": direction,
            "magnitude": round(abs(avg_return), 4),
            "avg_da": round(avg_da * 100, 1),
            "best_model": str(best_model.get("model_name", "N/A")),
            "models_count": len(week_data),
            "inference_week": target_week_str,
        }

    def _load_news_context(self, start: date, end: date) -> dict:
        """Load news context from all sources (GDELT + Google News + Investing.com + Colombia news).

        Combines GDELT sentiment summary with articles from all sources.
        """
        result = {
            "article_count": 0,
            "avg_sentiment": 0,
            "top_categories": {},
            "highlights": [],
            "source_breakdown": {},
        }

        # Load articles from ALL sources
        # Extend end by 2 days to capture weekend articles (GDELT indexes
        # with 1-2 day lag — most articles appear on Saturday/Sunday after
        # the business week ends on Friday).
        articles_df = self._get_all_articles()
        if not articles_df.empty:
            news_end = end + timedelta(days=2)  # Include Sat + Sun
            mask = (articles_df["date"] >= pd.Timestamp(start)) & (articles_df["date"] <= pd.Timestamp(news_end))
            week_articles = articles_df[mask]
            if not week_articles.empty:
                result["article_count"] = len(week_articles)

                # Compute avg_sentiment from article tones (hybrid FX rules)
                if "tone" in week_articles.columns:
                    non_zero_tones = week_articles["tone"][week_articles["tone"] != 0]
                    if len(non_zero_tones) > 0:
                        result["avg_sentiment"] = round(float(non_zero_tones.mean()), 3)
                        logger.info(
                            f"News sentiment: {len(non_zero_tones)}/{len(week_articles)} "
                            f"articles with non-zero tone, avg={result['avg_sentiment']}"
                        )

                # Fallback to GDELT sentiment CSV if article tones are all zero
                if result["avg_sentiment"] == 0:
                    sentiment_df = self._get_gdelt_sentiment()
                    if not sentiment_df.empty:
                        smask = (sentiment_df.index >= pd.Timestamp(start)) & (
                            sentiment_df.index <= pd.Timestamp(end)
                        )
                        week_sent = sentiment_df[smask]
                        if not week_sent.empty:
                            result["avg_sentiment"] = round(float(week_sent["tone_avg"].mean()), 3)

                # Source breakdown for context
                result["source_breakdown"] = (
                    week_articles["news_source"].value_counts().to_dict()
                )

                # Top highlights — round-robin across sources for diversity
                seen_titles = set()
                highlights = []
                source_groups = {}
                for _, row in week_articles.iterrows():
                    title = str(row.get("title", ""))
                    ns = str(row.get("news_source", ""))
                    if title and title.lower().strip() not in seen_titles and len(title) > 15:
                        seen_titles.add(title.lower().strip())
                        entry = {
                            "title": title[:200],
                            "source": str(row.get("source", row.get("news_source", "N/A"))),
                            "date": str(row.get("date", ""))[:10],
                            "news_source": ns,
                            "url": str(row.get("url", "")),
                            "sentiment": round(float(row.get("tone", row.get("sentiment_score", 0)) or 0), 2),
                        }
                        source_groups.setdefault(ns, []).append(entry)

                # Round-robin: pick 2-3 per source, cycling through sources
                source_keys = list(source_groups.keys())
                idx = dict.fromkeys(source_keys, 0)
                while len(highlights) < 10 and any(idx[k] < len(source_groups[k]) for k in source_keys):
                    for k in source_keys:
                        if idx[k] < len(source_groups[k]) and len(highlights) < 10:
                            highlights.append(source_groups[k][idx[k]])
                            idx[k] += 1
                result["highlights"] = highlights

        return result

    # Regex-based relevance patterns for FX-specific scoring (Phase 5 v2)
    # Score 3: Direct FX impact
    _SCORE3_PATTERNS = [
        re.compile(
            r"\b(fed\b|fomc|powell|tasas?\s+de\s+inter[eé]s|rate\s+hike|rate\s+cut"
            r"|banrep|banco\s+de\s+la\s+rep[uú]blica|tpm\b|ibr\b|pol[ií]tica\s+monetaria"
            r"|wti\b|brent\b|petr[oó]leo|crude\s+oil|opec"
            r"|dxy\b|d[oó]lar\s+index|dollar\s+strength"
            r"|embi\b|riesgo\s+pa[ií]s|credit\s+default"
            r"|usd/?cop|tipo\s+de\s+cambio|devaluaci[oó]n|revaluaci[oó]n"
            r"|peso\s+colombiano|d[oó]lar\s+colombi)",
            re.IGNORECASE,
        ),
    ]
    # Score 2: Indirect impact
    _SCORE2_PATTERNS = [
        re.compile(
            r"\b(exportaciones|importaciones|balanza\s+comercial"
            r"|colcap|bolsa\s+de\s+colombia|ecopetrol"
            r"|reforma\s+(tributaria|fiscal|pensional)"
            r"|pib\b|gdp\b|desempleo|unemployment"
            r"|emerging\s+markets|mercados?\s+emergentes|latam"
            r"|vix\b|volatilidad|risk[\s-]off|risk[\s-]on"
            r"|treasury|bonos\b|tes\b|deuda\s+p[uú]blica)",
            re.IGNORECASE,
        ),
    ]
    # Score 0: Noise exclusion
    _NOISE_PATTERNS = [
        re.compile(
            r"\b(f[uú]tbol|soccer|liga\s+betplay|champions|gol\b"
            r"|far[aá]ndula|celebridad|reality|netflix"
            r"|hor[oó]scopo|receta|cocina|novela)",
            re.IGNORECASE,
        ),
    ]

    # COP impact keyword patterns for bearish/bullish estimation
    _BEARISH_COP_PATTERNS = re.compile(
        r"\b(rate\s+hike|sube\s+tasas|hawkish|fed\s+endurece"
        r"|dxy\s+(sube|alza|fortalece)|dollar\s+strength"
        r"|embi\s+(sube|alza)|riesgo\s+pa[ií]s\s+(sube|alza|aumenta)"
        r"|petr[oó]leo\s+(cae|baja|desploma)|oil\s+(drop|fall|decline)"
        r"|wti\s+(cae|baja)|brent\s+(cae|baja)"
        r"|risk[\s-]off|aversion\s+(al\s+)?riesgo"
        r"|fuga\s+de\s+capitales|capital\s+outflow"
        r"|devaluaci[oó]n|peso\s+(cae|pierde|debilita))",
        re.IGNORECASE,
    )
    _BULLISH_COP_PATTERNS = re.compile(
        r"\b(rate\s+cut|baja\s+tasas|dovish|fed\s+suaviza"
        r"|dxy\s+(cae|baja|debilita)|dollar\s+weak"
        r"|embi\s+(cae|baja)|riesgo\s+pa[ií]s\s+(cae|baja|disminuye)"
        r"|petr[oó]leo\s+(sube|alza|repunta)|oil\s+(rise|rally|surge)"
        r"|wti\s+(sube|alza)|brent\s+(sube|alza)"
        r"|risk[\s-]on|apetito\s+(por\s+)?riesgo"
        r"|flujos?\s+de\s+inversi[oó]n|capital\s+inflow"
        r"|revaluaci[oó]n|peso\s+(sube|gana|fortalece))",
        re.IGNORECASE,
    )

    @staticmethod
    def _score_relevance(title: str, language: str) -> float:
        """Score article relevance to USDCOP trading (0-1).

        Uses regex-based pattern matching with 3 tiers:
        - Score 3 patterns (direct FX): +0.4
        - Score 2 patterns (indirect):  +0.2
        - Noise patterns: force 0.0
        - Spanish language bonus: +0.15
        """
        text = title.lower()

        # Check noise first — exclude irrelevant articles
        for pat in WeeklyAnalysisGenerator._NOISE_PATTERNS:
            if pat.search(text):
                return 0.0

        score = 0.0

        # Direct FX impact
        for pat in WeeklyAnalysisGenerator._SCORE3_PATTERNS:
            if pat.search(text):
                score += 0.4
                break

        # Indirect impact
        for pat in WeeklyAnalysisGenerator._SCORE2_PATTERNS:
            if pat.search(text):
                score += 0.2
                break

        # Prefer Spanish articles
        if language and str(language).lower().startswith("es"):
            score += 0.15

        return min(score, 1.0)

    @staticmethod
    def _estimate_cop_impact(title: str, summary: str = "") -> str:
        """Estimate COP impact direction from article text.

        Returns:
            'bearish_cop', 'bullish_cop', or 'neutral'
        """
        text = f"{title} {summary}".lower()

        bearish = bool(WeeklyAnalysisGenerator._BEARISH_COP_PATTERNS.search(text))
        bullish = bool(WeeklyAnalysisGenerator._BULLISH_COP_PATTERNS.search(text))

        if bearish and not bullish:
            return "bearish_cop"
        if bullish and not bearish:
            return "bullish_cop"
        return "neutral"

    def _load_news_highlights_for_day(self, target_date: date) -> list:
        """Load top news articles for a specific day from all sources.

        Also includes articles from the next day to capture GDELT's 1-day
        indexing lag (e.g., Friday articles often appear with Saturday dates).
        Uses relevance scoring to prioritize USDCOP-relevant articles.
        """
        articles_df = self._get_all_articles()
        if articles_df.empty:
            return []

        next_day = target_date + timedelta(days=1)
        mask = (articles_df["date"].dt.date == target_date) | (articles_df["date"].dt.date == next_day)
        day_articles = articles_df[mask]

        if day_articles.empty:
            return []

        # Score and sort by relevance
        candidates = []
        seen = set()
        for _, row in day_articles.iterrows():
            title = str(row.get("title", ""))
            if title and title.lower().strip() not in seen and len(title) > 15:
                seen.add(title.lower().strip())
                relevance = self._score_relevance(title, str(row.get("language", "")))
                candidates.append((relevance, row))

        # Sort by relevance descending, then ensure source diversity in top slots
        candidates.sort(key=lambda x: x[0], reverse=True)

        highlights = []
        seen_sources = set()
        # First pass: pick top from diverse sources
        for relevance, row in candidates:
            ns = str(row.get("news_source", ""))
            if ns not in seen_sources or len(highlights) >= 3:
                highlights.append({
                    "title": str(row.get("title", ""))[:200],
                    "source": str(row.get("source", row.get("news_source", "N/A"))),
                    "sentiment": round(float(row.get("tone", row.get("sentiment_score", row.get("gdelt_tone", 0))) or 0), 2),
                    "url": str(row.get("url", "")),
                    "news_source": ns,
                })
                seen_sources.add(ns)
                if len(highlights) >= 8:
                    break

        return highlights

    def _load_upcoming_events(self, reference_date: date, window_days: int = 14) -> list:
        """Load upcoming economic events from the calendar YAML."""
        if self._calendar_cache is None:
            cal_path = PROJECT_ROOT / "config/analysis/economic_calendar_2026.yaml"
            if cal_path.exists():
                try:
                    with open(cal_path) as f:
                        cal_data = yaml.safe_load(f)
                    self._calendar_cache = cal_data.get("events", [])
                except Exception as e:
                    logger.warning(f"Failed to load economic calendar: {e}")
                    self._calendar_cache = []
            else:
                self._calendar_cache = []

        upcoming = []
        end_date = reference_date + timedelta(days=window_days)
        for evt in self._calendar_cache:
            try:
                evt_date = date.fromisoformat(str(evt["date"]))
                if reference_date <= evt_date <= end_date:
                    upcoming.append({
                        "date": evt_date.isoformat(),
                        "event": evt["event"],
                        "country": evt.get("country", ""),
                        "impact_level": evt.get("impact_level", "medium"),
                    })
            except (ValueError, KeyError):
                continue

        # Sort by date
        upcoming.sort(key=lambda e: e["date"])
        return upcoming

    @staticmethod
    def _extract_themes(markdown: str) -> list:
        """Extract theme strings from LLM markdown headers."""
        themes = []
        for line in markdown.split("\n"):
            line = line.strip()
            # Match ## headers (section titles)
            if line.startswith("## ") and len(line) > 4:
                theme = line.lstrip("#").strip()
                # Skip generic headers
                if theme.lower() not in (
                    "resumen ejecutivo", "perspectiva", "conclusion",
                    "datos del dia", "indicadores macro clave",
                    "senales de modelos", "resumen de noticias",
                    "eventos economicos",
                ):
                    themes.append(theme)

        if not themes:
            # Fallback: extract bold phrases
            bold_pattern = re.compile(r"\*\*([^*]{5,60})\*\*")
            matches = bold_pattern.findall(markdown)
            themes = list(dict.fromkeys(matches))[:5]  # deduplicate, keep order

        return themes[:5]

    # ------------------------------------------------------------------
    # LLM batch sentiment enrichment
    # ------------------------------------------------------------------

    def _enrich_articles_with_llm_sentiment(
        self, iso_year: int, iso_week: int, start: date, end: date
    ) -> None:
        """Enrich all week's articles with hybrid sentiment scores.

        Phase 1: Apply FX impact rules to ALL week articles (fast, no models needed).
        Phase 2: Optionally run LLM batch scoring (single call, cached 7 days).
        Phase 3: Re-score with hybrid analyzer combining all available signals.

        Updates the tone column in self._all_articles_cache so all downstream
        consumers (_load_news_context, _load_news_highlights_for_day) see
        non-zero sentiment.
        """
        if self.dry_run:
            return

        try:
            from src.analysis.sentiment_analyzer import _title_hash, get_analyzer

            sentiment_cfg = self.config.get("sentiment", {})
            analyzer = get_analyzer(sentiment_cfg)

            # Load all articles (populates cache with tone=0 for CSV sources)
            articles_df = self._get_all_articles()
            if articles_df.empty:
                return

            news_end = end + timedelta(days=2)  # Include weekend lag
            mask = (articles_df["date"] >= pd.Timestamp(start)) & (
                articles_df["date"] <= pd.Timestamp(news_end)
            )
            week_idx = articles_df[mask].index

            if len(week_idx) == 0:
                return

            # Phase 1: LLM batch scoring (optional, cached)
            llm_batch_cfg = sentiment_cfg.get("llm_batch", {})
            if llm_batch_cfg.get("enabled", True) and self.llm is not None:
                article_list = []
                for idx in week_idx:
                    title = str(self._all_articles_cache.at[idx, "title"])
                    if title and len(title) > 10:
                        article_list.append({"title": title})

                if article_list:
                    cache_key = f"sentiment_batch_{iso_year}_W{iso_week:02d}"
                    try:
                        self._llm_sentiment_scores = analyzer.score_batch_with_llm(
                            article_list, self.llm, cache_key=cache_key
                        )
                        n_scored = len(self._llm_sentiment_scores)
                        logger.info(
                            f"LLM batch sentiment: scored {n_scored}/{len(article_list)} "
                            f"articles for {iso_year}-W{iso_week:02d}"
                        )
                    except Exception as e:
                        logger.warning(f"LLM batch scoring failed (continuing with FX rules): {e}")

            # Phase 2: Apply hybrid sentiment to ALL week articles
            # FX impact rules produce non-zero scores based on title keywords
            # (oil, Fed, BanRep, DXY, etc.) — no models or LLM needed.
            updated = 0
            for idx in week_idx:
                title = str(self._all_articles_cache.at[idx, "title"])
                if not title or len(title) < 10:
                    continue

                key = _title_hash(title)
                llm_score = self._llm_sentiment_scores.get(key) if self._llm_sentiment_scores else None
                existing_tone = self._all_articles_cache.at[idx, "tone"]
                gdelt_tone = existing_tone if existing_tone != 0 else None

                result = analyzer.analyze_single(
                    title=title,
                    gdelt_tone=gdelt_tone,
                    llm_score=llm_score,
                )
                new_score = result.fx_adjusted_score
                if new_score != 0:
                    self._all_articles_cache.at[idx, "tone"] = new_score
                    updated += 1

            if updated:
                logger.info(f"Hybrid sentiment: scored {updated}/{len(week_idx)} week articles")
            else:
                logger.info(f"Hybrid sentiment: no FX-relevant articles found in {len(week_idx)} articles")

        except Exception as e:
            logger.warning(f"Sentiment enrichment failed: {e}")

    # ------------------------------------------------------------------
    # Cached data accessors (lazy-loaded)
    # ------------------------------------------------------------------

    def _get_gdelt_sentiment(self) -> pd.DataFrame:
        """Load and cache GDELT daily sentiment CSV."""
        if self._gdelt_sentiment_cache is None:
            path = PROJECT_ROOT / "data/news/gdelt_daily_sentiment.csv"
            if path.exists():
                try:
                    df = pd.read_csv(path, parse_dates=["date"])
                    df = df.set_index("date").sort_index()
                    self._gdelt_sentiment_cache = df
                except Exception as e:
                    logger.warning(f"Failed to load GDELT sentiment: {e}")
                    self._gdelt_sentiment_cache = pd.DataFrame()
            else:
                self._gdelt_sentiment_cache = pd.DataFrame()
        return self._gdelt_sentiment_cache

    def _get_all_articles(self) -> pd.DataFrame:
        """Load and cache articles from ALL news sources (GDELT + Google News + Investing.com + Colombia news).

        Returns a unified DataFrame with columns: date, title, source, language, news_source, url.
        Deduplicates by title (case-insensitive, first occurrence wins).
        """
        if self._all_articles_cache is not None:
            return self._all_articles_cache

        frames: list[pd.DataFrame] = []

        # 1. GDELT articles CSV (128K+ articles, 2017-2026)
        gdelt_path = PROJECT_ROOT / "data/news/gdelt_articles_historical.csv"
        if gdelt_path.exists():
            try:
                df = pd.read_csv(
                    gdelt_path,
                    usecols=["date", "title", "url", "source", "domain", "language"],
                    dtype={"title": str, "url": str, "source": str, "domain": str, "language": str},
                )
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                # Use 'source' if available, fall back to 'domain'
                df["source"] = df["source"].fillna(df.get("domain", "GDELT"))
                df["news_source"] = "gdelt"
                if "url" not in df.columns:
                    df["url"] = ""
                frames.append(df[["date", "title", "source", "language", "news_source", "url"]])
                logger.debug(f"GDELT: loaded {len(df)} articles")
            except Exception as e:
                logger.warning(f"Failed to load GDELT articles: {e}")

        # 2. Google News JSON (13K+ articles, 2020-2026)
        gn_path = PROJECT_ROOT / "data/news/_google_news_full_extract_2020_2026.json"
        if gn_path.exists():
            try:
                with open(gn_path, encoding="utf-8") as f:
                    gn_data = json.load(f)
                if gn_data:
                    df = pd.DataFrame(gn_data)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["news_source"] = "google_news"
                    if "source" not in df.columns:
                        df["source"] = "Google News"
                    if "language" not in df.columns:
                        df["language"] = "en"
                    if "url" not in df.columns:
                        df["url"] = ""
                    frames.append(df[["date", "title", "source", "language", "news_source", "url"]])
                    logger.debug(f"Google News: loaded {len(df)} articles")
            except Exception as e:
                logger.warning(f"Failed to load Google News articles: {e}")

        # 3. Investing.com JSON (1.7K+ articles, 2020-2026)
        inv_path = PROJECT_ROOT / "data/news/_investing_full_extract_2020_2026.json"
        if inv_path.exists():
            try:
                with open(inv_path, encoding="utf-8") as f:
                    inv_data = json.load(f)
                if inv_data:
                    df = pd.DataFrame(inv_data)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["news_source"] = "investing"
                    df["source"] = "Investing.com"
                    if "language" not in df.columns:
                        df["language"] = "en"
                    if "url" not in df.columns:
                        df["url"] = ""
                    frames.append(df[["date", "title", "source", "language", "news_source", "url"]])
                    logger.debug(f"Investing.com: loaded {len(df)} articles")
            except Exception as e:
                logger.warning(f"Failed to load Investing.com articles: {e}")

        # 4. Colombia news historical CSV (800+ articles, recent weeks)
        col_path = PROJECT_ROOT / "data/news/colombia_news_historical.csv"
        if col_path.exists():
            try:
                df = pd.read_csv(col_path, dtype=str)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["news_source"] = "colombia_scraper"
                if "language" not in df.columns:
                    df["language"] = "es"
                if "url" not in df.columns:
                    df["url"] = ""
                frames.append(df[["date", "title", "source", "language", "news_source", "url"]])
                logger.debug(f"Colombia news: loaded {len(df)} articles")
            except Exception as e:
                logger.warning(f"Failed to load Colombia news: {e}")

        # 5. Live news_articles DB table (NewsEngine pipeline — enriched articles)
        try:
            import os

            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "usdcop_trading"),
                user=os.environ.get("POSTGRES_USER", "admin"),
                password=os.environ.get("POSTGRES_PASSWORD", ""),
            )
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT published_at AS date, title, source_id AS source,
                           url, language, sentiment_score, sentiment_label, category
                    FROM news_articles
                    ORDER BY published_at DESC
                """)
                rows = cur.fetchall()
            conn.close()
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
                df["date"] = df["date"].dt.tz_convert("America/Bogota").dt.tz_localize(None)
                df["news_source"] = df["source"]  # investing, portafolio, etc.
                if "url" not in df.columns:
                    df["url"] = ""
                # Preserve sentiment_score as "tone" for downstream compatibility
                if "sentiment_score" in df.columns:
                    df["tone"] = df["sentiment_score"].fillna(0.0)
                else:
                    df["tone"] = 0.0
                cols = ["date", "title", "source", "language", "news_source", "url", "tone"]
                frames.append(df[[c for c in cols if c in df.columns]])
                logger.info(f"DB news_articles: loaded {len(df)} articles (with sentiment)")
        except Exception as e:
            logger.debug(f"Could not load from news_articles DB: {e}")

        if not frames:
            self._all_articles_cache = pd.DataFrame()
            return self._all_articles_cache

        # Concatenate all sources
        combined = pd.concat(frames, ignore_index=True)
        # Ensure tone column always exists (CSV sources don't have it)
        if "tone" not in combined.columns:
            combined["tone"] = 0.0
        else:
            combined["tone"] = combined["tone"].fillna(0.0)
        combined = combined.dropna(subset=["date", "title"])

        # Deduplicate by normalized title (case-insensitive, keep first)
        combined["_title_norm"] = combined["title"].str.lower().str.strip()
        combined = combined.drop_duplicates(subset=["_title_norm"], keep="first")
        combined = combined.drop(columns=["_title_norm"])

        combined = combined.sort_values("date", ascending=False)

        source_counts = combined["news_source"].value_counts().to_dict()
        logger.info(f"All articles loaded: {len(combined)} total — {source_counts}")

        self._all_articles_cache = combined
        return self._all_articles_cache

    # ------------------------------------------------------------------
    # Core data loaders (unchanged)
    # ------------------------------------------------------------------

    def _load_macro_data(self) -> pd.DataFrame:
        """Load macro data from parquet file (local dev) or DB (production)."""
        macro_path = PROJECT_ROOT / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
        if macro_path.exists():
            df = pd.read_parquet(macro_path)
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"])
                df = df.set_index("fecha").sort_index()

            # Staleness warning
            if not df.empty:
                last_date = df.index.max()
                if hasattr(last_date, "date"):
                    last_date = last_date.date()
                days_stale = (date.today() - last_date).days
                if days_stale > 3:
                    logger.warning(
                        f"Macro data is {days_stale} days stale (last: {last_date}). "
                        f"Run L0 macro backfill to update."
                    )

            return df

        logger.warning("Macro parquet not found, returning empty DataFrame")
        return pd.DataFrame()

    def _load_daily_ohlcv(self, target_date: date) -> dict:
        """Load USDCOP OHLCV for a specific date."""
        ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
        if not ohlcv_path.exists():
            return {}

        df = pd.read_parquet(ohlcv_path)
        # Normalize: ensure 'date' column exists as date objects (not datetime64)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        # Find the date-like column
        date_col = None
        for col in ("date", "time", "Date"):
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            return {}

        df["_date"] = pd.to_datetime(df[date_col]).dt.date

        row = df[df["_date"] == target_date]
        if row.empty:
            return {}

        row = row.iloc[0]
        prev_rows = df[df["_date"] < target_date]
        prev_close = prev_rows.iloc[-1]["close"] if len(prev_rows) > 0 else None
        change_pct = round((row["close"] - prev_close) / prev_close * 100, 2) if prev_close else None

        return {
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "change_pct": change_pct,
        }

    def _load_weekly_ohlcv(self, start: date, end: date) -> dict:
        """Load USDCOP OHLCV summary for a week."""
        ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
        if not ohlcv_path.exists():
            return {}

        df = pd.read_parquet(ohlcv_path)
        # Normalize: ensure date column as date objects (not datetime64)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        date_col = None
        for col in ("date", "time", "Date"):
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            return {}

        df["_date"] = pd.to_datetime(df[date_col]).dt.date

        week_data = df[(df["_date"] >= start) & (df["_date"] <= end)]
        if week_data.empty:
            return self._load_weekly_ohlcv_from_db(start, end)

        open_price = float(week_data.iloc[0]["open"])
        close_price = float(week_data.iloc[-1]["close"])
        change_pct = round((close_price - open_price) / open_price * 100, 2)

        return {
            "open": open_price,
            "high": float(week_data["high"].max()),
            "low": float(week_data["low"].min()),
            "close": close_price,
            "change_pct": change_pct,
            "range_pct": round((float(week_data["high"].max()) - float(week_data["low"].min())) / open_price * 100, 2),
        }

    def _load_weekly_ohlcv_from_db(self, start: date, end: date) -> dict:
        """Fallback: aggregate daily OHLCV from 5-min bars in DB."""
        try:
            import os as _os

            import psycopg2
            db_url = _os.environ.get("DATABASE_URL")
            if not db_url:
                logger.debug("No DATABASE_URL set, cannot fall back to DB for OHLCV")
                return {}
            conn = psycopg2.connect(db_url)
            query = """
                SELECT
                    date_trunc('day', time AT TIME ZONE 'America/Bogota')::date as trade_date,
                    (array_agg(open ORDER BY time))[1] as open,
                    max(high) as high,
                    min(low) as low,
                    (array_agg(close ORDER BY time DESC))[1] as close
                FROM usdcop_m5_ohlcv
                WHERE symbol = 'USD/COP'
                  AND (time AT TIME ZONE 'America/Bogota')::date BETWEEN %s AND %s
                GROUP BY trade_date
                ORDER BY trade_date
            """
            df = pd.read_sql(query, conn, params=[start, end])
            conn.close()
            if df.empty:
                return {}
            open_price = float(df.iloc[0]["open"])
            close_price = float(df.iloc[-1]["close"])
            return {
                "open": open_price,
                "high": float(df["high"].max()),
                "low": float(df["low"].min()),
                "close": close_price,
                "change_pct": round((close_price - open_price) / open_price * 100, 2),
                "range_pct": round((float(df["high"].max()) - float(df["low"].min())) / open_price * 100, 2),
            }
        except Exception as e:
            logger.warning("DB fallback for weekly OHLCV failed: %s", e)
            return {}

    @staticmethod
    def _extract_headline(markdown: str) -> str:
        """Extract first meaningful line as headline."""
        for line in markdown.split("\n"):
            line = line.strip().lstrip("#").strip()
            if line and len(line) > 10:
                return line[:200]
        return "Analisis generado"

    @staticmethod
    def _detect_sentiment(text: str) -> str:
        """Simple keyword-based sentiment detection from generated text."""
        text_lower = text.lower()
        bullish_words = ["alcista", "fortalecimiento", "subio", "positivo", "ganancias", "recuperacion"]
        bearish_words = ["bajista", "debilitamiento", "cayo", "negativo", "perdidas", "caida"]

        bull_count = sum(1 for w in bullish_words if w in text_lower)
        bear_count = sum(1 for w in bearish_words if w in text_lower)

        if bull_count > bear_count + 1:
            return "bullish"
        elif bear_count > bull_count + 1:
            return "bearish"
        elif bull_count > 0 and bear_count > 0:
            return "mixed"
        return "neutral"
