"""
LangGraph Multi-Agent Analysis Graph (Phase 3)
================================================
StateGraph with 4 parallel domain agents + Reflection synthesizer.

Architecture:
  START → preprocess → [ta_agent, news_agent, macro_agent, fx_agent] → synthesizer → END

Each agent runs independently (graceful degradation on failure).
Synthesizer uses Reflection pattern: generate → evaluate → revise (max 3x).
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

import pandas as pd

from src.analysis.agent_state import MasterAnalysisState

logger = logging.getLogger(__name__)

# Maximum reflection iterations
MAX_REFLECTIONS = 3
MIN_QUALITY_SCORE = 0.8
DEGRADED_QUALITY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def preprocess_node(state: MasterAnalysisState) -> dict:
    """Load and prepare all data for the agents.

    If pre-loaded data is already injected (via run_analysis_graph preloaded_data),
    skip file loading and use the injected data directly.
    """
    from src.analysis.agent_tools import (
        load_daily_ohlcv, load_m5_ohlcv, load_macro_data,
        load_gdelt_articles, get_cop_series,
    )

    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    t0 = time.time()

    updates = {}

    # Check if data was pre-loaded (Phase 1: injected from weekly_generator)
    if state.get("ohlcv_daily"):
        log.append("Using pre-loaded data (skipping file loading)")
        log.append(f"Preprocess complete in {time.time() - t0:.1f}s (pre-loaded)")
        updates["execution_log"] = log
        updates["errors"] = errors
        return updates

    try:
        daily_df = load_daily_ohlcv()
        if not daily_df.empty:
            updates["ohlcv_daily"] = daily_df.reset_index().to_dict("records")
            cop = get_cop_series(daily_df)
            if not cop.empty:
                updates["cop_prices"] = [
                    {"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "close": float(v)}
                    for d, v in cop.items()
                ]
        log.append(f"Loaded daily OHLCV: {len(daily_df)} rows")
    except Exception as e:
        errors.append(f"preprocess/daily_ohlcv: {e}")

    try:
        m5_df = load_m5_ohlcv(last_n_days=60)
        if not m5_df.empty:
            updates["ohlcv_m5"] = m5_df.reset_index().to_dict("records")
        log.append(f"Loaded 5-min OHLCV: {len(m5_df)} rows")
    except Exception as e:
        errors.append(f"preprocess/m5_ohlcv: {e}")

    try:
        macro_df = load_macro_data()
        if not macro_df.empty:
            updates["macro_data"] = macro_df.reset_index().to_dict("records")
        log.append(f"Loaded macro data: {len(macro_df)} rows")
    except Exception as e:
        errors.append(f"preprocess/macro: {e}")

    try:
        articles = load_gdelt_articles(
            start_date=state.get("week_start"),
            end_date=state.get("week_end"),
        )
        updates["gdelt_articles"] = articles
        log.append(f"Loaded GDELT articles: {len(articles)}")
    except Exception as e:
        errors.append(f"preprocess/gdelt: {e}")

    log.append(f"Preprocess complete in {time.time() - t0:.1f}s")
    updates["execution_log"] = log
    updates["errors"] = errors
    return updates


def ta_agent_node(state: MasterAnalysisState) -> dict:
    """Technical Analysis agent: indicators + bias + scenarios."""
    from src.analysis.agent_tools import run_technical_analysis, run_multi_timeframe

    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    t0 = time.time()

    ta_report = None
    mtf_analysis = None

    try:
        # Daily TA
        daily_records = state.get("ohlcv_daily", [])
        if daily_records:
            daily_df = pd.DataFrame(daily_records)
            if "date" in daily_df.columns:
                daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
                daily_df = daily_df.dropna(subset=["date"])
                daily_df = daily_df.set_index("date").sort_index()
            ta_report = run_technical_analysis(daily_df, timeframe="1d")
            log.append(f"TA agent: daily analysis complete (bias={ta_report.get('dominant_bias')})")
    except Exception as e:
        errors.append(f"ta_agent/daily: {e}")
        logger.error(f"TA agent daily analysis failed: {e}", exc_info=True)

    try:
        # Multi-timeframe
        m5_records = state.get("ohlcv_m5", [])
        if m5_records:
            m5_df = pd.DataFrame(m5_records)
            if "time" in m5_df.columns:
                m5_df["time"] = pd.to_datetime(m5_df["time"])
                m5_df = m5_df.set_index("time").sort_index()

            daily_df_for_mtf = None
            if daily_records:
                daily_df_for_mtf = pd.DataFrame(daily_records)
                if "date" in daily_df_for_mtf.columns:
                    daily_df_for_mtf["date"] = pd.to_datetime(daily_df_for_mtf["date"], errors="coerce")
                    daily_df_for_mtf = daily_df_for_mtf.dropna(subset=["date"])
                    daily_df_for_mtf = daily_df_for_mtf.set_index("date").sort_index()

            mtf_analysis = run_multi_timeframe(m5_df, daily_df=daily_df_for_mtf)
            log.append(
                f"TA agent: MTF analysis complete "
                f"(alignment={mtf_analysis.get('alignment_label')})"
            )
    except Exception as e:
        errors.append(f"ta_agent/mtf: {e}")
        logger.error(f"TA agent MTF analysis failed: {e}", exc_info=True)

    log.append(f"TA agent done in {time.time() - t0:.1f}s")
    return {
        "ta_report": ta_report,
        "mtf_analysis": mtf_analysis,
        "execution_log": log,
        "errors": errors,
    }


def news_agent_node(state: MasterAnalysisState) -> dict:
    """News Intelligence agent: sentiment + clustering."""
    from src.analysis.agent_tools import run_news_intelligence

    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    t0 = time.time()

    news_intelligence = None

    try:
        articles = state.get("gdelt_articles", [])
        if articles:
            news_intelligence = run_news_intelligence(
                articles,
                state.get("week_start", ""),
                state.get("week_end", ""),
                min_relevance=0.2,
            )
            log.append(
                f"News agent: {news_intelligence.get('relevant_articles', 0)} relevant articles, "
                f"{len(news_intelligence.get('clusters', []))} clusters"
            )
        else:
            log.append("News agent: no articles available")
    except Exception as e:
        errors.append(f"news_agent: {e}")
        logger.error(f"News agent failed: {e}", exc_info=True)

    log.append(f"News agent done in {time.time() - t0:.1f}s")
    return {
        "news_intelligence": news_intelligence,
        "execution_log": log,
        "errors": errors,
    }


def macro_agent_node(state: MasterAnalysisState) -> dict:
    """Macro Regime agent: HMM + Granger + changepoints."""
    from src.analysis.agent_tools import run_macro_regime

    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    t0 = time.time()

    macro_regime = None

    try:
        macro_records = state.get("macro_data", [])
        cop_records = state.get("cop_prices", [])

        if macro_records and cop_records:
            macro_df = pd.DataFrame(macro_records)
            if "fecha" in macro_df.columns:
                macro_df["fecha"] = pd.to_datetime(macro_df["fecha"])
                macro_df = macro_df.set_index("fecha").sort_index()

            cop_df = pd.DataFrame(cop_records)
            cop_df["date"] = pd.to_datetime(cop_df["date"], format="%Y-%m-%d", errors="coerce")
            cop_df = cop_df.dropna(subset=["date"])
            cop_series = cop_df.set_index("date")["close"].sort_index()

            macro_regime = run_macro_regime(
                macro_df, cop_series,
                state.get("week_start", ""),
                state.get("week_end", ""),
            )
            log.append(
                f"Macro agent: regime={macro_regime.get('regime', {}).get('label')}, "
                f"{len(macro_regime.get('granger_leaders', []))} Granger leaders"
            )
        else:
            log.append("Macro agent: insufficient data")
    except Exception as e:
        errors.append(f"macro_agent: {e}")
        logger.error(f"Macro agent failed: {e}", exc_info=True)

    log.append(f"Macro agent done in {time.time() - t0:.1f}s")
    return {
        "macro_regime": macro_regime,
        "execution_log": log,
        "errors": errors,
    }


def fx_agent_node(state: MasterAnalysisState) -> dict:
    """FX Context agent: carry trade, oil, BanRep."""
    from src.analysis.agent_tools import run_fx_context

    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    t0 = time.time()

    fx_context = None

    try:
        macro_records = state.get("macro_data", [])
        cop_records = state.get("cop_prices", [])

        if macro_records:
            macro_df = pd.DataFrame(macro_records)
            if "fecha" in macro_df.columns:
                macro_df["fecha"] = pd.to_datetime(macro_df["fecha"])
                macro_df = macro_df.set_index("fecha").sort_index()

            cop_series = pd.Series(dtype=float)
            if cop_records:
                cop_df = pd.DataFrame(cop_records)
                cop_df["date"] = pd.to_datetime(cop_df["date"], format="%Y-%m-%d", errors="coerce")
                cop_df = cop_df.dropna(subset=["date"])
                cop_series = cop_df.set_index("date")["close"].sort_index()

            fx_context = run_fx_context(
                macro_df, cop_series,
                state.get("week_start", ""),
                state.get("week_end", ""),
                state.get("events_calendar", []),
            )
            log.append(f"FX agent: narrative generated ({len(fx_context.get('fx_narrative', ''))} chars)")
        else:
            log.append("FX agent: no macro data available")
    except Exception as e:
        errors.append(f"fx_agent: {e}")
        logger.error(f"FX agent failed: {e}", exc_info=True)

    log.append(f"FX agent done in {time.time() - t0:.1f}s")
    return {
        "fx_context": fx_context,
        "execution_log": log,
        "errors": errors,
    }


def synthesizer_node(state: MasterAnalysisState) -> dict:
    """Synthesis node: combine all agent outputs into a unified report.

    Uses Reflection pattern: generate → evaluate → revise (max 3x).
    """
    from src.analysis.prompt_templates import (
        build_ta_context, build_news_clusters_context,
        build_regime_context, build_fx_context_section,
        SYSTEM_SYNTHESIZER, EVALUATION_RUBRIC,
    )

    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    cost = state.get("cost_tracking", {"total_cost": 0, "total_tokens": 0})
    t0 = time.time()

    revision = state.get("synthesis_revision", 0)

    # Count available agent outputs
    available = sum(1 for k in ("ta_report", "news_intelligence", "macro_regime", "fx_context")
                    if state.get(k) is not None)

    if available == 0:
        log.append("Synthesizer: no agent outputs available — skipping LLM synthesis")
        return {
            "final_report": "[Sin datos suficientes para generar analisis]",
            "synthesis_quality": 0.0,
            "execution_log": log,
            "errors": errors,
        }

    # Build context sections dynamically from available agents
    sections = []
    if state.get("ta_report"):
        sections.append(build_ta_context(state["ta_report"]))
    if state.get("mtf_analysis"):
        sections.append(f"\n## Analisis Multi-Timeframe\nAlignment: {state['mtf_analysis'].get('alignment_label', 'N/A')}")
    if state.get("news_intelligence"):
        sections.append(build_news_clusters_context(state["news_intelligence"]))
    if state.get("macro_regime"):
        sections.append(build_regime_context(state["macro_regime"]))
    if state.get("fx_context"):
        sections.append(build_fx_context_section(state["fx_context"]))

    context = "\n\n---\n\n".join(sections)

    # Build synthesis prompt
    week_label = f"{state.get('iso_year', '')}-W{state.get('iso_week', 0):02d}"
    missing_agents = [k for k in ("ta_report", "news_intelligence", "macro_regime", "fx_context")
                      if state.get(k) is None]
    missing_note = ""
    if missing_agents:
        missing_note = f"\nNota: Los siguientes agentes no produjeron datos: {', '.join(missing_agents)}. Omite esas secciones."

    synthesis_prompt = (
        f"Genera el informe de analisis semanal para USD/COP, semana {week_label}.\n\n"
        f"## Datos de los agentes especializados\n{context}\n{missing_note}\n\n"
        f"Genera el informe completo siguiendo el formato del sistema."
    )

    # Generate synthesis (or revise if this is a revision)
    if revision > 0 and state.get("synthesis_critique"):
        synthesis_prompt = (
            f"Revisa el siguiente borrador basandote en la critica:\n\n"
            f"### Borrador Anterior\n{state.get('synthesis_draft', '')}\n\n"
            f"### Critica\n{state['synthesis_critique']}\n\n"
            f"### Datos Originales\n{context}\n\n"
            f"Genera una version mejorada."
        )

    try:
        from src.analysis.llm_client import LLMClient
        # Phase 1: Use injected LLM client (with API keys) if available
        llm = state.get("_llm_client") or LLMClient()
        result = llm.generate(
            SYSTEM_SYNTHESIZER,
            synthesis_prompt,
            max_tokens=3000,
            cache_key=f"synthesis_{week_label}_v{revision}",
        )
        draft = result.get("content", "")
        cost["total_cost"] = cost.get("total_cost", 0) + result.get("cost_usd", 0)
        cost["total_tokens"] = cost.get("total_tokens", 0) + result.get("tokens_used", 0)

        log.append(f"Synthesizer: draft v{revision} generated ({len(draft)} chars)")

        # Evaluate quality
        quality = _evaluate_quality(draft, llm, week_label, cost)
        log.append(f"Synthesizer: quality score = {quality:.2f}")

        # Determine quality threshold based on available agents
        threshold = MIN_QUALITY_SCORE if available >= 3 else DEGRADED_QUALITY_THRESHOLD

        if quality >= threshold or revision >= MAX_REFLECTIONS:
            # Accept
            return {
                "final_report": draft,
                "synthesis_draft": draft,
                "synthesis_quality": quality,
                "synthesis_revision": revision,
                "cost_tracking": cost,
                "execution_log": log,
                "errors": errors,
            }
        else:
            # Need revision — generate critique
            critique = _generate_critique(draft, llm, cost)
            log.append(f"Synthesizer: revision {revision + 1} needed (quality={quality:.2f} < {threshold})")

            return {
                "synthesis_draft": draft,
                "synthesis_critique": critique,
                "synthesis_quality": quality,
                "synthesis_revision": revision + 1,
                "cost_tracking": cost,
                "execution_log": log,
                "errors": errors,
            }

    except Exception as e:
        errors.append(f"synthesizer: {e}")
        logger.error(f"Synthesizer failed: {e}", exc_info=True)
        # Fallback: return raw context as final report
        return {
            "final_report": f"## Analisis Semanal {week_label}\n\n{context}",
            "synthesis_quality": 0.3,
            "synthesis_revision": revision,
            "cost_tracking": cost,
            "execution_log": log,
            "errors": errors,
        }


def _evaluate_quality(
    draft: str,
    llm: object,
    week_label: str,
    cost: dict,
) -> float:
    """Evaluate synthesis quality using rubric (Haiku-class task)."""
    from src.analysis.prompt_templates import EVALUATION_RUBRIC

    try:
        result = llm.generate(
            "Eres un evaluador de calidad de informes financieros. Responde SOLO con un JSON.",
            EVALUATION_RUBRIC.format(draft=draft),
            max_tokens=300,
            cache_key=f"eval_{week_label}_{hash(draft[:100])}",
        )
        content = result.get("content", "")
        cost["total_cost"] = cost.get("total_cost", 0) + result.get("cost_usd", 0)
        cost["total_tokens"] = cost.get("total_tokens", 0) + result.get("tokens_used", 0)

        # Parse score from response
        import json
        import re
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            scores = json.loads(json_match.group())
            # Average all numeric scores
            numeric = [v for v in scores.values() if isinstance(v, (int, float))]
            if numeric:
                return sum(numeric) / len(numeric)

    except Exception as e:
        logger.warning(f"Quality evaluation failed: {e}")

    return 0.7  # Default if evaluation fails


def _generate_critique(draft: str, llm: object, cost: dict) -> str:
    """Generate improvement critique for the draft."""
    try:
        result = llm.generate(
            "Eres un editor senior de informes financieros.",
            f"Critica el siguiente informe semanal de USD/COP y sugiere mejoras concretas:\n\n{draft[:2000]}",
            max_tokens=500,
            cache_key=f"critique_{hash(draft[:100])}",
        )
        cost["total_cost"] = cost.get("total_cost", 0) + result.get("cost_usd", 0)
        cost["total_tokens"] = cost.get("total_tokens", 0) + result.get("tokens_used", 0)
        return result.get("content", "")
    except Exception:
        return ""


def bias_agent_node(state: MasterAnalysisState) -> dict:
    """Political Bias Detection agent (Phase 3): source + cluster bias analysis."""
    log = state.get("execution_log", [])
    errors = state.get("errors", [])
    t0 = time.time()

    bias_output = None

    try:
        from src.analysis.bias_detector import PoliticalBiasDetector

        articles = state.get("gdelt_articles", [])
        news_intel = state.get("news_intelligence")

        if articles:
            detector = PoliticalBiasDetector()
            clusters = news_intel.get("clusters", []) if news_intel else []
            llm = state.get("_llm_client")

            bias_output = detector.analyze(
                articles=articles,
                clusters=clusters,
                llm_client=llm,
            )
            log.append(
                f"Bias agent: {bias_output.get('flagged_articles', 0)} flagged, "
                f"diversity={bias_output.get('bias_diversity_score', 0):.2f}"
            )
        else:
            log.append("Bias agent: no articles available")
    except Exception as e:
        errors.append(f"bias_agent: {e}")
        logger.error(f"Bias agent failed: {e}", exc_info=True)

    log.append(f"Bias agent done in {time.time() - t0:.1f}s")
    return {
        "political_bias_analysis": bias_output,
        "execution_log": log,
        "errors": errors,
    }


def finalizer_node(state: MasterAnalysisState) -> dict:
    """Final node: log completion."""
    log = state.get("execution_log", [])
    log.append("Pipeline complete")
    return {"execution_log": log}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_master_graph():
    """Build the LangGraph StateGraph for multi-agent analysis.

    Returns:
        Compiled LangGraph graph ready for .invoke().
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(MasterAnalysisState)

    # Add nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("ta_agent", ta_agent_node)
    graph.add_node("news_agent", news_agent_node)
    graph.add_node("macro_agent", macro_agent_node)
    graph.add_node("fx_agent", fx_agent_node)
    graph.add_node("bias_agent", bias_agent_node)    # Phase 3: 5th agent
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("finalizer", finalizer_node)

    # Entry point
    graph.set_entry_point("preprocess")

    # Preprocess → all 5 agents (parallel fan-out)
    graph.add_edge("preprocess", "ta_agent")
    graph.add_edge("preprocess", "news_agent")
    graph.add_edge("preprocess", "macro_agent")
    graph.add_edge("preprocess", "fx_agent")
    graph.add_edge("preprocess", "bias_agent")       # Phase 3

    # All agents → synthesizer (fan-in)
    graph.add_edge("ta_agent", "synthesizer")
    graph.add_edge("news_agent", "synthesizer")
    graph.add_edge("macro_agent", "synthesizer")
    graph.add_edge("fx_agent", "synthesizer")
    graph.add_edge("bias_agent", "synthesizer")      # Phase 3

    # Synthesizer → conditional: revise or finalize
    def should_revise(state: MasterAnalysisState) -> str:
        revision = state.get("synthesis_revision", 0)
        quality = state.get("synthesis_quality")
        final = state.get("final_report")

        if final is not None:
            return "finalizer"
        if revision >= MAX_REFLECTIONS:
            return "finalizer"
        return "synthesizer"

    graph.add_conditional_edges("synthesizer", should_revise, {
        "synthesizer": "synthesizer",
        "finalizer": "finalizer",
    })

    graph.add_edge("finalizer", END)

    return graph.compile()


def run_analysis_graph(
    iso_year: int,
    iso_week: int,
    dry_run: bool = False,
    preloaded_data: Optional[dict] = None,
    llm_client: Optional[object] = None,
) -> MasterAnalysisState:
    """Convenience function to run the full analysis graph.

    Args:
        iso_year: ISO year.
        iso_week: ISO week number.
        dry_run: If True, skip LLM calls.
        preloaded_data: Optional dict with pre-loaded DataFrames/articles
            (keys: ohlcv_daily, ohlcv_m5, macro_data, cop_prices,
             gdelt_articles, macro_digest). Skips file loading if provided.
        llm_client: Optional configured LLMClient instance (with API keys).
            If None, synthesizer creates a blank LLMClient (may fail without keys).

    Returns:
        Final state dict with all agent outputs.
    """
    start = date.fromisocalendar(iso_year, iso_week, 1)
    end = date.fromisocalendar(iso_year, iso_week, 5)

    initial_state: MasterAnalysisState = {
        "symbol": "USD/COP",
        "week_start": start.isoformat(),
        "week_end": end.isoformat(),
        "iso_year": iso_year,
        "iso_week": iso_week,
        "execution_log": [],
        "cost_tracking": {"total_cost": 0, "total_tokens": 0},
        "errors": [],
        "synthesis_revision": 0,
    }

    # Inject pre-loaded data (Phase 1: avoids redundant file loading)
    if preloaded_data:
        for key in ("ohlcv_daily", "ohlcv_m5", "macro_data", "cop_prices",
                     "gdelt_articles", "macro_digest", "events_calendar"):
            if key in preloaded_data and preloaded_data[key]:
                initial_state[key] = preloaded_data[key]

    # Inject LLM client (Phase 1: so synthesizer uses configured client)
    if llm_client:
        initial_state["_llm_client"] = llm_client

    graph = build_master_graph()
    result = graph.invoke(initial_state)
    return result
