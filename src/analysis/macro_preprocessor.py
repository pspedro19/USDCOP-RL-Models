"""
Macro Data Preprocessor (Phase 0)
===================================
Transforms raw macro data into a structured digest JSON for LLM prompts
and LangGraph agents: 8 variable groups, anomalies, top movers, correlation shifts.

Reuses MacroAnalyzer internals for indicator computation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.analysis.macro_analyzer import MacroAnalyzer, _safe_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable group definitions with USDCOP impact direction
# ---------------------------------------------------------------------------

VARIABLE_GROUPS: dict[str, dict[str, str]] = {
    "commodities": {
        "wti": "negative",       # Higher oil → COP strengthens
        "brent": "negative",
        "gold": "negative",
    },
    "usd_strength": {
        "dxy": "positive",       # Stronger USD → COP weakens
        "ust10y": "positive",
        "ust2y": "positive",
    },
    "colombia_rates": {
        "ibr": "mixed",
        "tpm": "mixed",
    },
    "risk_sentiment": {
        "vix": "positive",       # Higher VIX → risk-off → COP weakens
        "embi_col": "positive",  # Higher spread → COP weakens
    },
    "inflation": {
        "cpi_us": "indirect",
        "cpi_col": "indirect",
    },
    "fed_policy": {
        "fedfunds": "positive",  # Higher rates → stronger USD → COP weakens
    },
}

# Pre-built causal mechanism text per group (injected into LLM prompt)
IMPACT_CHAINS: dict[str, str] = {
    "commodities": (
        "Petroleo (WTI/Brent) → exportaciones colombianas → oferta de USD → COP se aprecia. "
        "Oro → refugio global, correlacion inversa con USD."
    ),
    "usd_strength": (
        "DXY sube → USD se fortalece globalmente → COP se debilita. "
        "Treasuries suben → mayor diferencial atrae capital a USA → presion al COP."
    ),
    "colombia_rates": (
        "IBR/TPM mas altos → carry trade atractivo → flujos hacia Colombia → COP se fortalece. "
        "Recortes de BanRep → reduce carry → COP se debilita."
    ),
    "risk_sentiment": (
        "VIX sube → aversion al riesgo → capitales salen de emergentes → COP se debilita. "
        "EMBI sube → mayor prima de riesgo Colombia → COP se debilita."
    ),
    "inflation": (
        "CPI USA alto → Fed hawkish → tasas USA suben → USD se fortalece → COP se debilita. "
        "CPI Colombia alto → BanRep mantiene tasas altas → carry protege COP."
    ),
    "fed_policy": (
        "Fed Funds sube → diferencial vs Colombia baja → carry menos atractivo → COP presionado. "
        "Fed dovish → USD debil → COP se fortalece."
    ),
}


@dataclass
class VariableDigest:
    """Structured digest for a single macro variable."""
    variable_key: str
    group: str
    impact_direction: str
    current_value: float | None = None
    last_update: str | None = None

    # Changes
    change_1d_pct: float | None = None
    change_1w_pct: float | None = None
    change_1m_pct: float | None = None

    # Context
    z_score_20d: float | None = None
    z_score_60d: float | None = None
    percentile_252d: float | None = None
    range_90d_low: float | None = None
    range_90d_high: float | None = None
    trend_slope: float | None = None  # Linear regression slope over 20d

    # Flags
    daily_move_class: str | None = None   # "large_up", "large_down", "normal"
    is_anomaly: bool = False                  # |z_score_20d| > 2
    near_90d_high: bool = False               # Within 2% of 90d high
    near_90d_low: bool = False                # Within 2% of 90d low

    # Correlation with COP
    correlation_20d: float | None = None
    correlation_60d: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MacroDigest:
    """Complete macro digest for LLM prompts."""
    as_of_date: str
    groups: dict[str, list[dict]] = field(default_factory=dict)
    top_movers: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    correlations: dict[str, dict] = field(default_factory=dict)
    impact_chains: dict[str, str] = field(default_factory=lambda: dict(IMPACT_CHAINS))
    group_assessments: dict[str, str] = field(default_factory=dict)  # group → bullish/bearish/neutral

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_text(self) -> str:
        """Format digest as structured text for LLM prompt injection."""
        lines = [f"## Macro Digest (as of {self.as_of_date})\n"]

        # Anomalies first (most important)
        if self.anomalies:
            lines.append("### ⚠ Anomalias (|z-score| > 2)")
            for a in self.anomalies:
                z = a.get("z_score_20d", 0) or 0
                lines.append(
                    f"- **{a['variable_key'].upper()}**: z={z:+.1f} "
                    f"(valor: {a.get('current_value', 'N/A')}, "
                    f"cambio 1d: {_fmt_pct(a.get('change_1d_pct'))})"
                )

        # Top movers
        if self.top_movers:
            lines.append("\n### Top 5 Movers (por |z-score 20d|)")
            for m in self.top_movers[:5]:
                z = m.get("z_score_20d", 0) or 0
                lines.append(
                    f"- **{m['variable_key'].upper()}** [{m.get('group', '')}]: "
                    f"z={z:+.1f}, 1d={_fmt_pct(m.get('change_1d_pct'))}, "
                    f"1w={_fmt_pct(m.get('change_1w_pct'))}"
                )

        # Group assessments
        if self.group_assessments:
            lines.append("\n### Evaluacion por Grupo")
            for group, assessment in self.group_assessments.items():
                chain = self.impact_chains.get(group, "")
                lines.append(f"- **{group}**: {assessment}")
                if chain:
                    lines.append(f"  Mecanismo: {chain}")

        # Correlation shifts
        if self.correlations:
            lines.append("\n### Correlaciones con COP (20d vs 60d)")
            for var_key, corr in sorted(
                self.correlations.items(),
                key=lambda x: abs(x[1].get("shift") or 0),
                reverse=True,
            )[:5]:
                c20 = corr.get("correlation_20d")
                c60 = corr.get("correlation_60d")
                shift = corr.get("shift")
                if c20 is not None and c60 is not None:
                    lines.append(
                        f"- {var_key.upper()}: 20d={c20:+.2f}, 60d={c60:+.2f} "
                        f"(cambio: {shift:+.2f})"
                    )

        return "\n".join(lines)


class MacroDataPreprocessor:
    """Transform raw macro data into structured digest for LLM prompts.

    Reuses MacroAnalyzer internals for indicator computation.
    Adds: percentiles, correlation shifts, anomaly detection, group assessments.
    """

    def __init__(self, analyzer: MacroAnalyzer | None = None):
        self.analyzer = analyzer or MacroAnalyzer()

    def compute_digest(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        as_of_date: date,
    ) -> MacroDigest:
        """Compute full digest with per-variable stats + derived analytics.

        Args:
            macro_df: Macro DataFrame indexed by date (fecha).
            cop_series: USDCOP close price series indexed by date.
            as_of_date: Reference date for all computations.

        Returns:
            MacroDigest with groups, anomalies, top_movers, correlations.
        """
        digest = MacroDigest(as_of_date=as_of_date.isoformat())
        all_digests: list[VariableDigest] = []

        for group_name, variables in VARIABLE_GROUPS.items():
            group_entries = []

            for var_key, impact_dir in variables.items():
                col = self.analyzer._find_column(macro_df, var_key)
                if col is None:
                    continue

                series = macro_df[col].dropna()
                if series.empty or len(series) < 30:
                    continue

                vd = self._compute_variable_digest(
                    series, var_key, group_name, impact_dir,
                    cop_series, as_of_date,
                )
                if vd:
                    group_entries.append(vd.to_dict())
                    all_digests.append(vd)

            if group_entries:
                digest.groups[group_name] = group_entries

        # Top movers: sorted by |z_score_20d|
        digest.top_movers = sorted(
            [d.to_dict() for d in all_digests if d.z_score_20d is not None],
            key=lambda x: abs(x.get("z_score_20d", 0) or 0),
            reverse=True,
        )[:5]

        # Anomalies: |z_score_20d| > 2
        digest.anomalies = [
            d.to_dict() for d in all_digests
            if d.is_anomaly
        ]

        # Correlations with COP
        for vd in all_digests:
            if vd.correlation_20d is not None or vd.correlation_60d is not None:
                c20 = vd.correlation_20d or 0
                c60 = vd.correlation_60d or 0
                digest.correlations[vd.variable_key] = {
                    "correlation_20d": vd.correlation_20d,
                    "correlation_60d": vd.correlation_60d,
                    "shift": round(c20 - c60, 4) if c20 and c60 else None,
                    "expected_direction": vd.impact_direction,
                }

        # Group assessments
        digest.group_assessments = self._assess_groups(digest.groups)

        return digest

    def _compute_variable_digest(
        self,
        series: pd.Series,
        var_key: str,
        group: str,
        impact_direction: str,
        cop_series: pd.Series,
        as_of_date: date,
    ) -> VariableDigest | None:
        """Compute digest for a single variable."""
        try:
            # Slice to as_of_date
            s = series.loc[:pd.Timestamp(as_of_date)]
            if s.empty:
                return None

            current = float(s.iloc[-1])
            last_date = s.index[-1]

            # Changes
            change_1d = _pct_change(s, 1)
            change_1w = _pct_change(s, 5)
            change_1m = _pct_change(s, 21)

            # Z-scores
            z20 = self._z_score(s, 20)
            z60 = self._z_score(s, 60)

            # Percentile (1-year lookback)
            pct_252 = self._percentile(s, 252)

            # 90-day range
            s_90d = s.iloc[-90:] if len(s) >= 90 else s
            range_low = float(s_90d.min())
            range_high = float(s_90d.max())

            # Trend slope (20d linear regression)
            slope = self._trend_slope(s, 20)

            # Daily move classification
            daily_class = "normal"
            if change_1d is not None:
                if abs(change_1d) > 2.0:
                    daily_class = "large_up" if change_1d > 0 else "large_down"
                elif abs(change_1d) > 1.0:
                    daily_class = "moderate_up" if change_1d > 0 else "moderate_down"

            # Flags
            is_anomaly = abs(z20 or 0) > 2.0
            near_high = current >= range_high * 0.98 if range_high else False
            near_low = current <= range_low * 1.02 if range_low else False

            # Correlations with COP
            corr_20 = self._rolling_correlation(s, cop_series, 20, as_of_date)
            corr_60 = self._rolling_correlation(s, cop_series, 60, as_of_date)

            return VariableDigest(
                variable_key=var_key,
                group=group,
                impact_direction=impact_direction,
                current_value=_safe_float(current),
                last_update=str(last_date.date()) if hasattr(last_date, "date") else str(last_date),
                change_1d_pct=_safe_float(change_1d),
                change_1w_pct=_safe_float(change_1w),
                change_1m_pct=_safe_float(change_1m),
                z_score_20d=_safe_float(z20),
                z_score_60d=_safe_float(z60),
                percentile_252d=_safe_float(pct_252),
                range_90d_low=_safe_float(range_low),
                range_90d_high=_safe_float(range_high),
                trend_slope=_safe_float(slope),
                daily_move_class=daily_class,
                is_anomaly=is_anomaly,
                near_90d_high=near_high,
                near_90d_low=near_low,
                correlation_20d=_safe_float(corr_20),
                correlation_60d=_safe_float(corr_60),
            )

        except Exception as e:
            logger.warning(f"Failed to compute digest for {var_key}: {e}")
            return None

    @staticmethod
    def _z_score(series: pd.Series, window: int) -> float | None:
        """Compute z-score vs rolling mean/std."""
        if len(series) < window:
            return None
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        mean_val = rolling_mean.iloc[-1]
        std_val = rolling_std.iloc[-1]
        if std_val and std_val > 0:
            return round((float(series.iloc[-1]) - float(mean_val)) / float(std_val), 4)
        return None

    @staticmethod
    def _percentile(series: pd.Series, window: int) -> float | None:
        """Compute percentile rank within lookback window."""
        if len(series) < 10:
            return None
        lookback = series.iloc[-window:] if len(series) >= window else series
        current = float(series.iloc[-1])
        rank = (lookback < current).sum() / len(lookback)
        return round(float(rank) * 100, 1)

    @staticmethod
    def _trend_slope(series: pd.Series, window: int) -> float | None:
        """Linear regression slope over last N periods (normalized by mean)."""
        if len(series) < window:
            return None
        y = series.iloc[-window:].values.astype(float)
        x = np.arange(len(y), dtype=float)
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            return None
        y, x = y[mask], x[mask]
        mean_y = np.mean(y)
        if mean_y == 0:
            return None
        slope = np.polyfit(x, y, 1)[0]
        # Normalize by mean to get % per day
        return round((slope / mean_y) * 100, 4)

    @staticmethod
    def _rolling_correlation(
        var_series: pd.Series,
        cop_series: pd.Series,
        window: int,
        as_of_date: date,
    ) -> float | None:
        """Compute rolling correlation between variable and COP close."""
        try:
            # Align dates
            combined = pd.DataFrame({
                "var": var_series,
                "cop": cop_series,
            }).dropna()

            if len(combined) < window:
                return None

            combined = combined.loc[:pd.Timestamp(as_of_date)]
            if len(combined) < window:
                return None

            corr = combined["var"].iloc[-window:].corr(combined["cop"].iloc[-window:])
            if np.isnan(corr):
                return None
            return round(float(corr), 4)
        except Exception:
            return None

    def _assess_groups(self, groups: dict[str, list[dict]]) -> dict[str, str]:
        """Produce bullish/bearish/neutral assessment per group based on z-scores."""
        assessments = {}
        for group_name, entries in groups.items():
            if not entries:
                continue

            # Weighted z-score for the group
            z_scores = [e.get("z_score_20d") for e in entries if e.get("z_score_20d") is not None]
            if not z_scores:
                assessments[group_name] = "neutral (sin datos z-score)"
                continue

            avg_z = np.mean(z_scores)

            # Map z-score to COP impact using direction
            # For "positive" impact: high z → COP weakens (bearish for COP)
            # For "negative" impact: high z → COP strengthens (bullish for COP)
            directions = [e.get("impact_direction", "mixed") for e in entries]
            cop_impact_scores = []
            for z, d in zip(z_scores, directions):
                if d == "positive":
                    cop_impact_scores.append(z)       # Higher → COP weaker
                elif d == "negative":
                    cop_impact_scores.append(-z)      # Higher → COP stronger (inverse)
                else:
                    cop_impact_scores.append(0)

            cop_z = np.mean(cop_impact_scores) if cop_impact_scores else 0

            if cop_z > 0.5:
                label = "bearish para COP"
            elif cop_z < -0.5:
                label = "bullish para COP"
            else:
                label = "neutral"

            detail_parts = []
            for e in entries:
                z_val = e.get("z_score_20d")
                if z_val is not None:
                    detail_parts.append(f"{e['variable_key'].upper()} z={z_val:+.1f}")
            detail = ", ".join(detail_parts)

            assessments[group_name] = f"{label} ({detail})"

        return assessments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_change(series: pd.Series, periods: int) -> float | None:
    """Compute percentage change over N periods."""
    if len(series) <= periods:
        return None
    current = float(series.iloc[-1])
    past = float(series.iloc[-1 - periods])
    if past == 0:
        return None
    return round(((current - past) / abs(past)) * 100, 4)


def _fmt_pct(val: float | None) -> str:
    """Format percentage for display."""
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"
