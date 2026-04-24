"""
FX Context Engine (Multi-Agent System — Phase 2)
==================================================
USDCOP-specific domain knowledge:
- Carry trade analysis (IBR - Fed Funds differential)
- Oil impact estimation (WTI sensitivity)
- BanRep context (next meeting, rate expectation)
- Key risk identification
- Spanish narrative generation
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Historical sensitivities (pre-computed estimates)
# ---------------------------------------------------------------------------

# How much a 1% change in the variable affects USDCOP (in same direction)
# Positive = var up → COP weakens (USDCOP up)
# Negative = var up → COP strengthens (USDCOP down)
FX_SENSITIVITIES = {
    "wti": -0.15,       # Oil up → COP strengthens
    "dxy": +0.45,       # Dollar up → COP weakens
    "embi_col": +0.30,  # Risk up → COP weakens
    "vix": +0.20,       # Fear up → COP weakens
    "gold": -0.05,      # Gold up → mild COP strengthening
    "ust10y": +0.10,    # Yields up → COP weakens
}

# Column name lookup
COLUMN_ALIASES = {
    "dxy": ["dxy_close", "dxy", "FXRT_INDEX_DXY_USA_D_DXY"],
    "vix": ["vix_close", "vix", "FXRT_INDEX_VIX_USA_D_VIX", "VOLT_VIX_USA_D_VIX"],
    "wti": ["wti_close", "wti", "COMMODITY_OIL_WTI_USA_D_CL1", "COMM_OIL_WTI_GLB_D_WTI"],
    "embi_col": ["embi_close", "embi_col", "SPREAD_EMBI_COL_COL_D_EMBI", "CRSK_SPREAD_EMBI_COL_D_EMBI"],
    "ust10y": ["ust10y_close", "ust10y", "RATE_TREASURY_10Y_USA_D_GS10", "FINC_BOND_YIELD10Y_USA_D_UST10Y"],
    "ibr": ["ibr_close", "ibr", "RATE_IBR_COL_D_IBR", "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR"],
    "fedfunds": ["fedfunds_close", "fedfunds", "RATE_FEDFUNDS_USA_M_DFF", "POLR_PRIME_RATE_USA_D_PRIME"],
    "tpm": ["tpm_close", "tpm", "POLR_POLICY_RATE_COL_M_TPM"],
    "gold": ["gold_close", "gold", "COMMODITY_GOLD_USA_D_GC1", "COMM_METAL_GOLD_GLB_D_GOLD"],
    "brent": ["brent_close", "brent", "COMMODITY_OIL_BRENT_UK_D_BZ1", "COMM_OIL_BRENT_GLB_D_BRENT"],
}


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    """Find matching column for a macro variable key."""
    for alias in COLUMN_ALIASES.get(key, [key]):
        if alias in df.columns:
            return alias
    for col in df.columns:
        if key.lower() in col.lower():
            return col
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CarryTradeAnalysis:
    ibr_rate: float | None = None
    fed_funds_rate: float | None = None
    differential_pct: float | None = None  # IBR - FedFunds
    differential_trend: str = "stable"  # narrowing, widening, stable
    carry_attractiveness: str = "neutral"  # attractive, neutral, unattractive
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OilImpact:
    wti_current: float | None = None
    wti_weekly_change_pct: float | None = None
    brent_current: float | None = None
    estimated_cop_impact_pct: float | None = None
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BanRepContext:
    tpm_current: float | None = None
    ibr_current: float | None = None
    next_meeting: str | None = None    # ISO date or "unknown"
    rate_expectation: str = "hold"        # cut, hold, hike
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskFactor:
    factor: str = ""
    severity: str = "medium"  # low, medium, high
    direction: str = ""       # cop_weakening, cop_strengthening
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FXContextReport:
    carry_trade: CarryTradeAnalysis = field(default_factory=CarryTradeAnalysis)
    oil_impact: OilImpact = field(default_factory=OilImpact)
    banrep: BanRepContext = field(default_factory=BanRepContext)
    risk_factors: list[RiskFactor] = field(default_factory=list)
    fx_narrative: str = ""  # 3-5 sentence Spanish narrative
    cop_weekly_change_pct: float | None = None
    cop_level: float | None = None
    sensitivity_impacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "carry_trade": self.carry_trade.to_dict(),
            "oil_impact": self.oil_impact.to_dict(),
            "banrep": self.banrep.to_dict(),
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "fx_narrative": self.fx_narrative,
            "cop_weekly_change_pct": self.cop_weekly_change_pct,
            "cop_level": self.cop_level,
            "sensitivity_impacts": self.sensitivity_impacts,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FXContextEngine:
    """USD/COP FX-specific domain analysis."""

    def analyze(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        week_start: str,
        week_end: str,
        events_calendar: list[dict] | None = None,
    ) -> FXContextReport:
        """Full FX context analysis.

        Args:
            macro_df: Macro DataFrame (DatetimeIndex).
            cop_series: USDCOP close prices (DatetimeIndex).
            week_start: ISO date.
            week_end: ISO date.
            events_calendar: Optional list of upcoming economic events.

        Returns:
            FXContextReport.
        """
        report = FXContextReport()

        if cop_series is not None and not cop_series.empty:
            report.cop_level = round(float(cop_series.iloc[-1]), 2)
            # Weekly change
            start_ts = pd.Timestamp(week_start)
            end_ts = pd.Timestamp(week_end)
            week_data = cop_series[(cop_series.index >= start_ts) & (cop_series.index <= end_ts)]
            if len(week_data) >= 2:
                pct = (week_data.iloc[-1] - week_data.iloc[0]) / week_data.iloc[0] * 100
                report.cop_weekly_change_pct = round(float(pct), 3)

        try:
            self._analyze_carry_trade(macro_df, report)
        except Exception as e:
            logger.warning(f"Carry trade analysis failed: {e}")

        try:
            self._estimate_oil_impact(macro_df, week_start, week_end, report)
        except Exception as e:
            logger.warning(f"Oil impact estimation failed: {e}")

        try:
            self._banrep_context(macro_df, events_calendar, report)
        except Exception as e:
            logger.warning(f"BanRep context failed: {e}")

        try:
            self._compute_sensitivity_impacts(macro_df, week_start, week_end, report)
        except Exception as e:
            logger.warning(f"Sensitivity impacts failed: {e}")

        try:
            self._identify_risks(macro_df, cop_series, report)
        except Exception as e:
            logger.warning(f"Risk identification failed: {e}")

        self._build_fx_narrative(report)

        return report

    # ------------------------------------------------------------------
    # Carry Trade
    # ------------------------------------------------------------------

    def _analyze_carry_trade(
        self, macro_df: pd.DataFrame, report: FXContextReport
    ) -> None:
        """Analyze IBR - Fed Funds carry trade differential."""
        ibr_col = _find_col(macro_df, "ibr")
        fed_col = _find_col(macro_df, "fedfunds")

        ibr_rate = None
        fed_rate = None

        if ibr_col:
            ibr_series = macro_df[ibr_col].dropna()
            if not ibr_series.empty:
                ibr_rate = float(ibr_series.iloc[-1])

        if fed_col:
            fed_series = macro_df[fed_col].dropna()
            if not fed_series.empty:
                fed_rate = float(fed_series.iloc[-1])

        if ibr_rate is not None and fed_rate is not None:
            diff = ibr_rate - fed_rate

            # Trend: compare current diff vs 30 days ago
            trend = "stable"
            if ibr_col and fed_col:
                ibr_30 = macro_df[ibr_col].dropna().tail(30)
                fed_30 = macro_df[fed_col].dropna().tail(30)
                if len(ibr_30) >= 2 and len(fed_30) >= 2:
                    old_diff = float(ibr_30.iloc[0]) - float(fed_30.iloc[0])
                    if diff - old_diff > 0.25:
                        trend = "widening"
                    elif old_diff - diff > 0.25:
                        trend = "narrowing"

            # Attractiveness
            if diff > 3.0:
                attractiveness = "attractive"
            elif diff > 1.0:
                attractiveness = "neutral"
            else:
                attractiveness = "unattractive"

            interpretation = (
                f"Diferencial carry trade IBR-FedFunds: {diff:.2f}pp. "
                f"{'Atractivo para flujos hacia Colombia' if attractiveness == 'attractive' else 'Diferencial insuficiente para atraer carry trade'}."
            )

            report.carry_trade = CarryTradeAnalysis(
                ibr_rate=ibr_rate,
                fed_funds_rate=fed_rate,
                differential_pct=round(diff, 2),
                differential_trend=trend,
                carry_attractiveness=attractiveness,
                interpretation=interpretation,
            )

    # ------------------------------------------------------------------
    # Oil Impact
    # ------------------------------------------------------------------

    def _estimate_oil_impact(
        self,
        macro_df: pd.DataFrame,
        week_start: str,
        week_end: str,
        report: FXContextReport,
    ) -> None:
        """Estimate weekly oil impact on COP."""
        wti_col = _find_col(macro_df, "wti")
        brent_col = _find_col(macro_df, "brent")

        if wti_col:
            wti_series = macro_df[wti_col].dropna()
            if len(wti_series) >= 5:
                current = float(wti_series.iloc[-1])
                week_ago = float(wti_series.iloc[-5]) if len(wti_series) >= 5 else current
                weekly_change = (current - week_ago) / week_ago * 100

                # Estimated COP impact = WTI change * sensitivity
                cop_impact = weekly_change * FX_SENSITIVITIES["wti"]

                direction = "fortalecimiento" if cop_impact < 0 else "debilitamiento"
                interpretation = (
                    f"WTI cambio {weekly_change:+.1f}% esta semana. "
                    f"Impacto estimado en COP: {cop_impact:+.2f}% ({direction})."
                )

                report.oil_impact = OilImpact(
                    wti_current=round(current, 2),
                    wti_weekly_change_pct=round(weekly_change, 2),
                    estimated_cop_impact_pct=round(cop_impact, 3),
                    interpretation=interpretation,
                )

        if brent_col:
            brent_series = macro_df[brent_col].dropna()
            if not brent_series.empty:
                report.oil_impact.brent_current = round(float(brent_series.iloc[-1]), 2)

    # ------------------------------------------------------------------
    # BanRep Context
    # ------------------------------------------------------------------

    def _banrep_context(
        self,
        macro_df: pd.DataFrame,
        events_calendar: list[dict] | None,
        report: FXContextReport,
    ) -> None:
        """BanRep policy rate context."""
        tpm_col = _find_col(macro_df, "tpm")
        ibr_col = _find_col(macro_df, "ibr")

        tpm = None
        ibr = None

        if tpm_col:
            s = macro_df[tpm_col].dropna()
            if not s.empty:
                tpm = float(s.iloc[-1])

        if ibr_col:
            s = macro_df[ibr_col].dropna()
            if not s.empty:
                ibr = float(s.iloc[-1])

        # Find next BanRep meeting from calendar
        next_meeting = None
        if events_calendar:
            for evt in events_calendar:
                if "banrep" in str(evt.get("event", "")).lower():
                    next_meeting = evt.get("date")
                    break

        # Rate expectation inference
        expectation = "hold"
        interpretation = ""
        if tpm is not None and ibr is not None:
            if ibr < tpm - 0.5:
                expectation = "cut"
                interpretation = (
                    f"IBR ({ibr:.2f}%) esta por debajo del TPM ({tpm:.2f}%), "
                    f"sugiriendo expectativa de recorte de tasa."
                )
            elif ibr > tpm + 0.5:
                expectation = "hike"
                interpretation = (
                    f"IBR ({ibr:.2f}%) esta por encima del TPM ({tpm:.2f}%), "
                    f"sugiriendo expectativa de alza."
                )
            else:
                interpretation = (
                    f"IBR ({ibr:.2f}%) cerca del TPM ({tpm:.2f}%), "
                    f"sugiriendo expectativa de tasa sin cambios."
                )

        report.banrep = BanRepContext(
            tpm_current=tpm,
            ibr_current=ibr,
            next_meeting=next_meeting,
            rate_expectation=expectation,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Sensitivity Impacts
    # ------------------------------------------------------------------

    def _compute_sensitivity_impacts(
        self,
        macro_df: pd.DataFrame,
        week_start: str,
        week_end: str,
        report: FXContextReport,
    ) -> None:
        """Compute estimated COP impact from each macro variable's weekly change."""
        impacts = {}

        for key, sensitivity in FX_SENSITIVITIES.items():
            col = _find_col(macro_df, key)
            if col is None:
                continue

            series = macro_df[col].dropna()
            if len(series) < 5:
                continue

            current = float(series.iloc[-1])
            prev = float(series.iloc[-5]) if len(series) >= 5 else current
            if abs(prev) < 1e-10:
                continue

            weekly_change_pct = (current - prev) / abs(prev) * 100
            cop_impact = weekly_change_pct * sensitivity

            impacts[key] = {
                "current": round(current, 2),
                "weekly_change_pct": round(weekly_change_pct, 2),
                "sensitivity": sensitivity,
                "estimated_cop_impact_pct": round(cop_impact, 3),
            }

        report.sensitivity_impacts = impacts

    # ------------------------------------------------------------------
    # Risk Identification
    # ------------------------------------------------------------------

    def _identify_risks(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        report: FXContextReport,
    ) -> None:
        """Identify key risk factors for the week."""
        risks = []

        # VIX spike
        vix_col = _find_col(macro_df, "vix")
        if vix_col:
            vix = macro_df[vix_col].dropna()
            if not vix.empty and float(vix.iloc[-1]) > 25:
                risks.append(RiskFactor(
                    factor="VIX elevado",
                    severity="high" if float(vix.iloc[-1]) > 30 else "medium",
                    direction="cop_weakening",
                    description=f"VIX en {float(vix.iloc[-1]):.1f} — aversion al riesgo presiona EM",
                ))

        # EMBI spike
        embi_col = _find_col(macro_df, "embi_col")
        if embi_col:
            embi = macro_df[embi_col].dropna()
            if len(embi) >= 20:
                z = (embi.iloc[-1] - embi.tail(60).mean()) / max(embi.tail(60).std(), 1e-6)
                if z > 1.5:
                    risks.append(RiskFactor(
                        factor="EMBI Colombia elevado",
                        severity="high" if z > 2 else "medium",
                        direction="cop_weakening",
                        description=f"Riesgo pais Colombia por encima de norma (z={float(z):.1f})",
                    ))

        # Oil collapse
        if report.oil_impact.wti_weekly_change_pct is not None:
            if report.oil_impact.wti_weekly_change_pct < -5:
                risks.append(RiskFactor(
                    factor="Caida de petroleo",
                    severity="high",
                    direction="cop_weakening",
                    description=f"WTI cayo {report.oil_impact.wti_weekly_change_pct:.1f}% — negativo para COP",
                ))

        # DXY strength
        for key, impact in report.sensitivity_impacts.items():
            if key == "dxy" and impact.get("weekly_change_pct", 0) > 1.5:
                risks.append(RiskFactor(
                    factor="Fortalecimiento del dolar",
                    severity="medium",
                    direction="cop_weakening",
                    description=f"DXY subio {impact['weekly_change_pct']:.1f}% — presion sobre EM",
                ))

        # Carry trade narrowing
        if report.carry_trade.differential_trend == "narrowing":
            risks.append(RiskFactor(
                factor="Reduccion del carry",
                severity="medium",
                direction="cop_weakening",
                description="Diferencial de tasas se estrecha — menos atractivo para flujos",
            ))

        report.risk_factors = risks[:6]

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------

    def _build_fx_narrative(self, report: FXContextReport) -> None:
        """Build 3-5 sentence FX context narrative in Spanish."""
        parts = []

        # COP level and direction
        if report.cop_level and report.cop_weekly_change_pct is not None:
            direction = "se deprecio" if report.cop_weekly_change_pct > 0 else "se aprecio"
            parts.append(
                f"El USD/COP cerro la semana en {report.cop_level:.0f}, "
                f"el peso {direction} {abs(report.cop_weekly_change_pct):.2f}%."
            )

        # Carry trade
        if report.carry_trade.differential_pct is not None:
            parts.append(report.carry_trade.interpretation)

        # Oil
        if report.oil_impact.interpretation:
            parts.append(report.oil_impact.interpretation)

        # BanRep
        if report.banrep.interpretation:
            parts.append(report.banrep.interpretation)

        # Top risk
        if report.risk_factors:
            top_risk = report.risk_factors[0]
            parts.append(f"Riesgo principal: {top_risk.description}")

        report.fx_narrative = " ".join(parts[:5])
