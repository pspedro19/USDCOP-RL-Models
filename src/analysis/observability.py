"""
Observability Module (Phase 6)
================================
LangFuse integration for tracing every multi-agent analysis run.

Tracks:
- Cost per agent per execution
- Latency per agent
- Quality score from evaluator
- Revision count (Reflection iterations)
- Error rate per agent
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentTrace:
    """Trace data for a single agent execution."""
    agent_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalysisTrace:
    """Complete trace for a multi-agent analysis run."""
    trace_id: str = ""
    iso_year: int = 0
    iso_week: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_s: float = 0.0
    total_cost_usd: float = 0.0
    quality_score: Optional[float] = None
    revision_count: int = 0
    agents: dict[str, AgentTrace] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class AnalysisObserver:
    """Observability wrapper for multi-agent analysis runs.

    Supports LangFuse when configured, falls back to logging.
    """

    def __init__(self):
        self._langfuse = None
        self._current_trace: Optional[AnalysisTrace] = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Initialize LangFuse client if credentials are available."""
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.debug("LangFuse credentials not set, using log-only observability")
            return

        try:
            from langfuse import Langfuse
            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("LangFuse observability initialized")
        except ImportError:
            logger.debug("langfuse package not installed, using log-only observability")
        except Exception as e:
            logger.warning(f"Failed to initialize LangFuse: {e}")

    @contextmanager
    def trace_analysis(self, iso_year: int, iso_week: int):
        """Context manager for a complete analysis run."""
        import uuid

        trace = AnalysisTrace(
            trace_id=str(uuid.uuid4())[:8],
            iso_year=iso_year,
            iso_week=iso_week,
            start_time=time.time(),
        )
        self._current_trace = trace

        lf_trace = None
        if self._langfuse:
            try:
                lf_trace = self._langfuse.trace(
                    name=f"weekly_analysis_{iso_year}_W{iso_week:02d}",
                    metadata={
                        "iso_year": iso_year,
                        "iso_week": iso_week,
                    },
                )
            except Exception as e:
                logger.warning(f"LangFuse trace creation failed: {e}")

        try:
            yield trace
        finally:
            trace.end_time = time.time()
            trace.total_duration_s = trace.end_time - trace.start_time
            trace.total_cost_usd = sum(
                a.cost_usd for a in trace.agents.values()
            )

            # Log summary
            agents_ok = sum(1 for a in trace.agents.values() if a.success)
            agents_total = len(trace.agents)
            logger.info(
                f"Analysis trace {trace.trace_id}: "
                f"{agents_ok}/{agents_total} agents OK, "
                f"quality={trace.quality_score}, "
                f"revisions={trace.revision_count}, "
                f"cost=${trace.total_cost_usd:.4f}, "
                f"duration={trace.total_duration_s:.1f}s"
            )

            # Update LangFuse
            if lf_trace:
                try:
                    lf_trace.update(
                        output={
                            "agents_ok": agents_ok,
                            "agents_total": agents_total,
                            "quality_score": trace.quality_score,
                            "revision_count": trace.revision_count,
                            "total_cost_usd": trace.total_cost_usd,
                            "errors": trace.errors[:5],
                        },
                    )
                    self._langfuse.flush()
                except Exception as e:
                    logger.warning(f"LangFuse trace update failed: {e}")

            self._current_trace = None

    @contextmanager
    def trace_agent(self, agent_name: str):
        """Context manager for a single agent execution within a trace."""
        agent_trace = AgentTrace(
            agent_name=agent_name,
            start_time=time.time(),
        )

        lf_span = None
        if self._langfuse and self._current_trace:
            try:
                lf_span = self._langfuse.span(
                    name=agent_name,
                    trace_id=self._current_trace.trace_id,
                )
            except Exception:
                pass

        try:
            yield agent_trace
        except Exception as e:
            agent_trace.success = False
            agent_trace.error = str(e)
            raise
        finally:
            agent_trace.end_time = time.time()
            agent_trace.duration_s = agent_trace.end_time - agent_trace.start_time

            if self._current_trace:
                self._current_trace.agents[agent_name] = agent_trace
                if not agent_trace.success:
                    self._current_trace.errors.append(
                        f"{agent_name}: {agent_trace.error}"
                    )

            # Log agent result
            status = "OK" if agent_trace.success else f"FAILED: {agent_trace.error}"
            logger.info(
                f"  Agent {agent_name}: {status} "
                f"({agent_trace.duration_s:.1f}s, ${agent_trace.cost_usd:.4f})"
            )

            if lf_span:
                try:
                    lf_span.end(
                        output={
                            "success": agent_trace.success,
                            "error": agent_trace.error,
                            "cost_usd": agent_trace.cost_usd,
                            "tokens_in": agent_trace.tokens_in,
                            "tokens_out": agent_trace.tokens_out,
                        },
                    )
                except Exception:
                    pass

    def record_quality(self, score: float, revision_count: int) -> None:
        """Record quality score and revision count for the current trace."""
        if self._current_trace:
            self._current_trace.quality_score = score
            self._current_trace.revision_count = revision_count

    def record_agent_cost(
        self,
        agent_name: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record cost data for an agent in the current trace."""
        if self._current_trace and agent_name in self._current_trace.agents:
            agent = self._current_trace.agents[agent_name]
            agent.tokens_in += tokens_in
            agent.tokens_out += tokens_out
            agent.cost_usd += cost_usd


# Module-level singleton
_observer: Optional[AnalysisObserver] = None


def get_observer() -> AnalysisObserver:
    """Get or create the global observer singleton."""
    global _observer
    if _observer is None:
        _observer = AnalysisObserver()
    return _observer
