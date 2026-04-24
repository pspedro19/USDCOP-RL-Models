"""
Macro Regime Engine (Multi-Agent System — Phase 1)
====================================================
Regime detection using HMM (hmmlearn) + changepoints (ruptures).
Granger causality to identify leading indicators for COP.
Reuses existing MacroAnalyzer for SMA/BB/RSI/MACD (no duplication).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    label: str = "transition"     # risk_on, transition, risk_off
    since: str | None = None   # ISO date when regime started
    confidence: float = 0.0
    transition_probabilities: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GrangerLeader:
    variable: str = ""
    optimal_lag: int = 1
    f_statistic: float = 0.0
    p_value: float = 1.0
    direction: str = ""  # "positive" or "negative" correlation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Changepoint:
    date: str = ""
    variable: str = ""
    direction: str = ""  # "up" or "down"
    magnitude: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ZScoreAlert:
    variable: str = ""
    variable_name: str = ""
    z_score: float = 0.0
    direction: str = ""  # "extreme_high" or "extreme_low"
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MacroRegimeReport:
    regime: RegimeState = field(default_factory=RegimeState)
    granger_leaders: list[GrangerLeader] = field(default_factory=list)
    changepoints: list[Changepoint] = field(default_factory=list)
    correlations: dict = field(default_factory=dict)  # var -> rolling weekly corr with COP
    zscore_alerts: list[ZScoreAlert] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)  # Spanish-language insights

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.to_dict(),
            "granger_leaders": [g.to_dict() for g in self.granger_leaders],
            "changepoints": [c.to_dict() for c in self.changepoints],
            "correlations": self.correlations,
            "zscore_alerts": [z.to_dict() for z in self.zscore_alerts],
            "insights": self.insights,
        }


# ---------------------------------------------------------------------------
# Key variables and their expected COP impact
# ---------------------------------------------------------------------------

MACRO_COP_IMPACT = {
    "dxy": ("positive", "DXY sube → COP se debilita"),
    "vix": ("positive", "VIX sube → COP se debilita (risk-off)"),
    "wti": ("negative", "Petroleo sube → COP se fortalece"),
    "embi_col": ("positive", "EMBI sube → COP se debilita (riesgo pais)"),
    "ust10y": ("positive", "Treasury sube → COP se debilita (carry)"),
    "ibr": ("negative", "IBR sube → COP se fortalece (carry inverso)"),
    "gold": ("negative", "Oro sube → COP tiende a fortalecerse"),
    "brent": ("negative", "Brent sube → COP se fortalece"),
}

# Column name lookup (same logic as MacroAnalyzer)
COLUMN_ALIASES = {
    "dxy": ["dxy_close", "dxy", "FXRT_INDEX_DXY_USA_D_DXY"],
    "vix": ["vix_close", "vix", "FXRT_INDEX_VIX_USA_D_VIX"],
    "wti": ["wti_close", "wti", "COMMODITY_OIL_WTI_USA_D_CL1"],
    "embi_col": ["embi_close", "embi_col", "SPREAD_EMBI_COL_COL_D_EMBI"],
    "ust10y": ["ust10y_close", "ust10y", "RATE_TREASURY_10Y_USA_D_GS10"],
    "ibr": ["ibr_close", "ibr", "RATE_IBR_COL_D_IBR"],
    "gold": ["gold_close", "gold", "COMMODITY_GOLD_USA_D_GC1"],
    "brent": ["brent_close", "brent", "COMMODITY_OIL_BRENT_UK_D_BZ1"],
}


def _find_column(df: pd.DataFrame, key: str) -> str | None:
    """Find the best matching column for a macro variable key."""
    for alias in COLUMN_ALIASES.get(key, [key]):
        if alias in df.columns:
            return alias
    # Fuzzy match
    for col in df.columns:
        if key.lower() in col.lower():
            return col
    return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MacroRegimeEngine:
    """Regime detection, Granger causality, changepoints, correlations."""

    def analyze(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        week_start: str,
        week_end: str,
    ) -> MacroRegimeReport:
        """Full macro regime analysis.

        Args:
            macro_df: Macro indicators DataFrame (DatetimeIndex, multi-column).
            cop_series: USDCOP close Series (DatetimeIndex).
            week_start: ISO date string.
            week_end: ISO date string.

        Returns:
            MacroRegimeReport.
        """
        report = MacroRegimeReport()

        if macro_df.empty or cop_series.empty:
            logger.warning("Insufficient data for macro regime analysis")
            return report

        try:
            self._fit_hmm_regime(macro_df, cop_series, report)
        except Exception as e:
            logger.warning(f"HMM regime detection failed: {e}")

        try:
            self._test_granger_causality(macro_df, cop_series, report)
        except Exception as e:
            logger.warning(f"Granger causality test failed: {e}")

        try:
            self._detect_changepoints(macro_df, cop_series, week_start, week_end, report)
        except Exception as e:
            logger.warning(f"Changepoint detection failed: {e}")

        try:
            self._compute_correlations(macro_df, cop_series, report)
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")

        try:
            self._compute_zscore_alerts(macro_df, week_end, report)
        except Exception as e:
            logger.warning(f"Z-score alerts failed: {e}")

        self._generate_insights(report)

        return report

    # ------------------------------------------------------------------
    # HMM Regime Detection
    # ------------------------------------------------------------------

    def _fit_hmm_regime(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        report: MacroRegimeReport,
    ) -> None:
        """Fit 3-state HMM (Risk-On / Transition / Risk-Off) on COP returns."""
        from hmmlearn.hmm import GaussianHMM

        # Compute daily log returns
        returns = np.log(cop_series / cop_series.shift(1)).dropna()
        if len(returns) < 60:
            logger.warning(f"Insufficient returns for HMM ({len(returns)} < 60)")
            return

        # Use last 252 trading days
        returns = returns.tail(252)
        X = returns.values.reshape(-1, 1)

        # Fit 3-state HMM
        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)

        # Get hidden states
        states = model.predict(X)
        current_state = int(states[-1])

        # Label states by mean return: highest = risk_on, lowest = risk_off
        state_means = model.means_.flatten()
        sorted_states = np.argsort(state_means)
        state_labels = {int(sorted_states[0]): "risk_off", int(sorted_states[1]): "transition", int(sorted_states[2]): "risk_on"}

        label = state_labels.get(current_state, "transition")

        # Find when current regime started
        regime_start_idx = len(states) - 1
        while regime_start_idx > 0 and states[regime_start_idx - 1] == current_state:
            regime_start_idx -= 1

        regime_start_date = returns.index[regime_start_idx]

        # Transition probabilities for current state
        trans_probs = {}
        for i, lbl in state_labels.items():
            trans_probs[lbl] = round(float(model.transmat_[current_state, i]), 3)

        # Confidence = stationary probability of current state
        # Approximate from last N observations
        state_counts = np.bincount(states[-20:], minlength=3)
        confidence = round(float(state_counts[current_state] / max(state_counts.sum(), 1)), 2)

        report.regime = RegimeState(
            label=label,
            since=str(regime_start_date)[:10],
            confidence=confidence,
            transition_probabilities=trans_probs,
        )

    # ------------------------------------------------------------------
    # Granger Causality
    # ------------------------------------------------------------------

    def _test_granger_causality(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        report: MacroRegimeReport,
    ) -> None:
        """Test which macro variables Granger-cause COP movements."""
        from statsmodels.tsa.stattools import grangercausalitytests

        cop_returns = cop_series.pct_change().dropna()
        leaders = []

        for key in MACRO_COP_IMPACT:
            col = _find_column(macro_df, key)
            if col is None:
                continue

            var_series = macro_df[col].dropna()
            var_returns = var_series.pct_change().dropna()

            # Align
            combined = pd.concat([cop_returns, var_returns], axis=1, join="inner").dropna()
            combined.columns = ["cop", "var"]

            if len(combined) < 50:
                continue

            # Test lags 1-5
            best_lag = 1
            best_p = 1.0
            best_f = 0.0

            try:
                results = grangercausalitytests(combined[["cop", "var"]], maxlag=5, verbose=False)
                for lag, res in results.items():
                    p_val = res[0]["ssr_ftest"][1]
                    f_stat = res[0]["ssr_ftest"][0]
                    if p_val < best_p:
                        best_p = p_val
                        best_f = f_stat
                        best_lag = lag
            except Exception:
                continue

            if best_p < 0.1:  # Relaxed threshold for reporting
                direction = MACRO_COP_IMPACT.get(key, ("unknown", ""))[0]
                leaders.append(GrangerLeader(
                    variable=key,
                    optimal_lag=int(best_lag),
                    f_statistic=float(round(best_f, 2)),
                    p_value=float(round(best_p, 4)),
                    direction=direction,
                ))

        # Sort by p-value
        leaders.sort(key=lambda x: x.p_value)
        report.granger_leaders = leaders[:5]

    # ------------------------------------------------------------------
    # Changepoint Detection
    # ------------------------------------------------------------------

    def _detect_changepoints(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        week_start: str,
        week_end: str,
        report: MacroRegimeReport,
    ) -> None:
        """Detect structural breaks using PELT algorithm (ruptures)."""
        import ruptures

        changepoints = []

        # Detect changepoints in COP
        cop_vals = cop_series.dropna().tail(120).values
        if len(cop_vals) >= 30:
            try:
                algo = ruptures.Pelt(model="rbf", min_size=10).fit(cop_vals)
                bkps = algo.predict(pen=10)
                cop_index = cop_series.dropna().tail(120).index

                for bkp in bkps[:-1]:  # Last is always len(data)
                    if bkp < len(cop_index):
                        bkp_date = str(cop_index[bkp])[:10]
                        # Direction based on mean before vs after
                        before = cop_vals[max(0, bkp - 10):bkp].mean()
                        after = cop_vals[bkp:min(bkp + 10, len(cop_vals))].mean()
                        changepoints.append(Changepoint(
                            date=bkp_date,
                            variable="USDCOP",
                            direction="up" if after > before else "down",
                            magnitude=float(round(abs(after - before), 2)),
                        ))
            except Exception as e:
                logger.debug(f"COP changepoint detection failed: {e}")

        # Detect changepoints in key macro variables
        for key in ("dxy", "vix", "wti", "embi_col"):
            col = _find_column(macro_df, key)
            if col is None:
                continue

            vals = macro_df[col].dropna().tail(120).values
            if len(vals) < 30:
                continue

            try:
                algo = ruptures.Pelt(model="rbf", min_size=10).fit(vals)
                bkps = algo.predict(pen=10)
                idx = macro_df[col].dropna().tail(120).index

                for bkp in bkps[:-1]:
                    if bkp < len(idx):
                        bkp_date = str(idx[bkp])[:10]
                        before = vals[max(0, bkp - 5):bkp].mean()
                        after = vals[bkp:min(bkp + 5, len(vals))].mean()
                        changepoints.append(Changepoint(
                            date=bkp_date,
                            variable=key.upper(),
                            direction="up" if after > before else "down",
                            magnitude=float(round(abs(after - before) / max(abs(before), 1e-6) * 100, 2)),
                        ))
            except Exception:
                continue

        # Filter to recent changepoints (within 30 days of week_end)
        try:
            end = pd.Timestamp(week_end)
            recent = [cp for cp in changepoints if (end - pd.Timestamp(cp.date)).days <= 30]
            recent.sort(key=lambda x: x.date, reverse=True)
            report.changepoints = recent[:8]
        except Exception:
            report.changepoints = changepoints[:8]

    # ------------------------------------------------------------------
    # Rolling Correlations
    # ------------------------------------------------------------------

    def _compute_correlations(
        self,
        macro_df: pd.DataFrame,
        cop_series: pd.Series,
        report: MacroRegimeReport,
    ) -> None:
        """Compute rolling weekly correlations between macro vars and COP."""
        cop_returns = cop_series.pct_change().dropna()
        correlations = {}

        for key in MACRO_COP_IMPACT:
            col = _find_column(macro_df, key)
            if col is None:
                continue

            var_returns = macro_df[col].pct_change().dropna()
            combined = pd.concat([cop_returns, var_returns], axis=1, join="inner").dropna()
            combined.columns = ["cop", "var"]

            if len(combined) < 20:
                continue

            # 20-day rolling correlation
            rolling_corr = combined["cop"].rolling(20).corr(combined["var"])
            if not rolling_corr.empty:
                current_corr = float(rolling_corr.iloc[-1])
                avg_corr = float(rolling_corr.tail(60).mean())
                correlations[key] = {
                    "current": round(current_corr, 3) if not np.isnan(current_corr) else None,
                    "avg_60d": round(avg_corr, 3) if not np.isnan(avg_corr) else None,
                    "expected_direction": MACRO_COP_IMPACT[key][0],
                }

        report.correlations = correlations

    # ------------------------------------------------------------------
    # Z-Score Alerts
    # ------------------------------------------------------------------

    def _compute_zscore_alerts(
        self,
        macro_df: pd.DataFrame,
        week_end: str,
        report: MacroRegimeReport,
    ) -> None:
        """Find variables with |z-score| > 2 (extreme readings)."""
        alerts = []

        for key, (direction, interpretation) in MACRO_COP_IMPACT.items():
            col = _find_column(macro_df, key)
            if col is None:
                continue

            series = macro_df[col].dropna()
            if len(series) < 60:
                continue

            # Z-score of last value relative to 60-day window
            window = series.tail(60)
            mean = window.mean()
            std = window.std()
            if std < 1e-10:
                continue

            z = (series.iloc[-1] - mean) / std

            if abs(z) > 2.0:
                z_dir = "extreme_high" if z > 0 else "extreme_low"
                display_name = {
                    "dxy": "DXY", "vix": "VIX", "wti": "WTI",
                    "embi_col": "EMBI Colombia", "ust10y": "Treasury 10Y",
                    "ibr": "IBR", "gold": "Oro", "brent": "Brent",
                }.get(key, key)

                alerts.append(ZScoreAlert(
                    variable=key,
                    variable_name=display_name,
                    z_score=round(float(z), 2),
                    direction=z_dir,
                    interpretation=interpretation,
                ))

        # Sort by absolute z-score
        alerts.sort(key=lambda x: abs(x.z_score), reverse=True)
        report.zscore_alerts = alerts

    # ------------------------------------------------------------------
    # Insights Generation (Spanish)
    # ------------------------------------------------------------------

    def _generate_insights(self, report: MacroRegimeReport) -> None:
        """Generate Spanish-language insights from analysis results."""
        insights = []

        # Regime insight
        regime_labels = {
            "risk_on": "apetito por riesgo (Risk-On)",
            "risk_off": "aversion al riesgo (Risk-Off)",
            "transition": "transicion entre regimenes",
        }
        label = regime_labels.get(report.regime.label, report.regime.label)
        insights.append(
            f"El regimen actual es {label} "
            f"(confianza {report.regime.confidence:.0%}), "
            f"activo desde {report.regime.since or 'N/A'}."
        )

        # Granger leaders
        if report.granger_leaders:
            leaders_str = ", ".join(
                f"{g.variable.upper()} (lag={g.optimal_lag}d, p={g.p_value:.3f})"
                for g in report.granger_leaders[:3]
            )
            insights.append(f"Variables lideres para COP: {leaders_str}.")

        # Z-score alerts
        for alert in report.zscore_alerts[:3]:
            direction_es = "extremadamente alto" if alert.direction == "extreme_high" else "extremadamente bajo"
            insights.append(
                f"{alert.variable_name} esta {direction_es} (z={alert.z_score:+.1f}): "
                f"{alert.interpretation}"
            )

        # Changepoints
        recent_cps = [cp for cp in report.changepoints if cp.variable == "USDCOP"]
        if recent_cps:
            cp = recent_cps[0]
            dir_es = "al alza" if cp.direction == "up" else "a la baja"
            insights.append(
                f"Cambio estructural detectado en USDCOP {dir_es} "
                f"cerca del {cp.date} (magnitud: {cp.magnitude:.0f} pesos)."
            )

        # Correlation anomalies
        for key, corr_data in report.correlations.items():
            expected = corr_data.get("expected_direction")
            current = corr_data.get("current")
            if current is not None and expected:
                if (expected == "positive" and current < -0.3) or \
                   (expected == "negative" and current > 0.3):
                    insights.append(
                        f"Anomalia: correlacion {key.upper()}-COP ({current:+.2f}) "
                        f"va contra la relacion historica esperada ({expected})."
                    )

        report.insights = insights[:8]
