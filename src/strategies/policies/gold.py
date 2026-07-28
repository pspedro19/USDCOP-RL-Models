"""XAU/USD — ``gold_trend_simple`` as a first-class policy (BL-47 R7).

Frozen source of truth (``config/strategy_manifests/xauusd.yaml``,
code_hash ``bc494207ee2916d8``): 2-of-3 SMA vote {63,126,252}, binary, sized
by vol targeting 0.10 / rv20 with floor 0.06 and cap 1.5, executed shift(1).

Per-bar reproduction, NOT a re-derivation: the vectorised legacy path applies
the same three comparisons and the same arithmetic to each row. The rolling
means and the realized vol are FEATURES (computed by the feature layer with
the same frozen indicator code); the policy only consumes them, which is
exactly the split §5 prescribes (shared features vs policy-local decision).

DIVERGENCIA DECLARADA (ver ``config/policies/gold_trend_simple.yaml``): the
repo has TWO producers for this strategy_id and they do NOT agree on the
regime multiplier. ``apply_regime_multiplier`` makes the choice explicit
instead of hiding it; the spec pins the PUBLISHER path (multiplier OFF).
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.policy import PolicyContext, StrategyDecision
from src.contracts.rule_trace import RuleTrace
from src.strategies.policies.base import CodedPolicy


class GoldTrendSimplePolicy(CodedPolicy):
    """2-of-3 SMA vote × vol-targeted size, long-flat."""

    required = ("close", "sma_63", "sma_126", "sma_252", "realized_vol_20")
    optional_defaults = {"regime_risk_mult": 1.0}

    def decide(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        windows = [int(w) for w in self.params["vote_windows"]]
        min_votes = int(self.params["min_votes"])
        target_vol = float(self.params["target_vol"])
        vol_floor = float(self.params["vol_floor"])
        size_cap = float(self.params["size_cap"])
        apply_regime = bool(self.params["apply_regime_multiplier"])

        close = self._get(snapshot, "close")
        entries = []
        votes = 0
        for window in windows:
            sma = self._get(snapshot, f"sma_{window}")
            fired = close > sma
            votes += int(fired)
            entries.append(
                self._entry(
                    f"close_above_sma_{window}",
                    f"Cierre sobre SMA{window}",
                    {"close": close, f"sma_{window}": sma},
                    fired,
                    "CLOSE_ABOVE_SMA" if fired else "CLOSE_BELOW_SMA",
                    {f"sma_{window}": sma},
                )
            )

        in_market = votes >= min_votes
        entries.append(
            self._entry(
                "sma_vote_threshold",
                f"Votos SMA >= {min_votes}",
                {"sma_votes": votes},
                in_market,
                "SMA_VOTES_GE_MIN" if in_market else "SMA_VOTES_LT_MIN",
                {"min_votes": min_votes},
            )
        )

        # Vol targeting — el sizing determinista del manifiesto congelado.
        # clip(lower=vol_floor) ANTES de dividir, cap DESPUÉS: el mismo orden
        # que ``(TARGET_VOL / rv.clip(lower=0.06)).clip(upper=MAX_SIZE)``.
        realized_vol = self._get(snapshot, "realized_vol_20")
        size = target_vol / max(realized_vol, vol_floor)
        if apply_regime:
            size = min(max(size * self._get(snapshot, "regime_risk_mult"), 0.0), size_cap)
        else:
            size = min(size, size_cap)

        exposure = (1.0 if in_market else 0.0) * size
        direction = "LONG" if exposure > 0.0 else "FLAT"
        if in_market:
            reason_codes = ("SMA_VOTES_GE_MIN",)
            trace = RuleTrace(rules=tuple(entries), winning_rule_id="sma_vote_threshold")
        else:
            reason_codes = (self.default_reason_code,)
            exposure = self.default_exposure
            direction = self.default_direction
            trace = RuleTrace(rules=tuple(entries), fallback_applied=True)

        components = {
            "close": close,
            "sma_votes": votes,
            "realized_vol_20": realized_vol,
            "vol_target_size": size,
            "target_exposure": exposure,
        }
        if apply_regime:
            components["regime_risk_mult"] = self._get(snapshot, "regime_risk_mult")

        return self._decision(
            context,
            direction=direction,
            exposure=exposure,
            reason_codes=reason_codes,
            components=components,
            trace=trace,
        )
