"""BTC/USDT — ``btc_hodl_b1`` as a first-class policy (BL-47 R7).

Frozen source of truth (``config/strategy_manifests/btcusdt.yaml``,
code_hash ``3b9b3c6dd1b9a70d``): intent constante 1.0 (HODL) × vol targeting
0.30 / rv20 (floor 0.30) × multiplicador de régimen, cap spot 1.0, shift(1).

The constitution's §3 point in code form: the honest baseline IS the
strategy. It is `rule_based` and not "no strategy" precisely because the
sizing layer is a deterministic policy — and this is what makes it a policy
rather than a declarative rule: ``target_vol / realized_vol`` is arithmetic,
which the whitelist DSL does not (and should not) express.

Per-bar reproduction of ``src/btc_strategy/strategies.py::build_positions``
with ``intent_hodl``: intent 1.0, size = clip(base × regime_mult, 0, 1),
position = clip(intent × size, 0, 1) — same order of operations, same clips.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.policy import PolicyContext, StrategyDecision
from src.contracts.rule_trace import RuleTrace
from src.strategies.policies.base import CodedPolicy


class BtcHodlB1Policy(CodedPolicy):
    """HODL intent (constante) × exposición vol-targeted spot-only [0, 1]."""

    # regime_risk_mult es OPCIONAL con default 1.0 porque eso es EXACTAMENTE lo
    # que hace el código congelado (``df.get("regime_risk_mult", 1.0)``) y
    # porque ``build_daily_features`` del track BTC NO emite esa columna: el
    # pipeline real corre con multiplicador 1.0. Declararla requerida cambiaría
    # el comportamiento (fallo cerrado donde el legacy opera).
    required = ("realized_vol_20",)
    optional_defaults = {"regime_risk_mult": 1.0}

    def decide(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        intent = float(self.params["intent"])
        target_vol = float(self.params["target_vol"])
        vol_floor = float(self.params["vol_floor"])
        max_exposure = float(self.params["max_exposure"])
        min_exposure = float(self.params["min_exposure"])

        realized_vol = self._get(snapshot, "realized_vol_20")
        regime_mult = self._get(snapshot, "regime_risk_mult")

        base = target_vol / max(realized_vol, vol_floor)
        size = min(max(base * regime_mult, min_exposure), max_exposure)
        exposure = min(max(intent * size, min_exposure), max_exposure)

        # HODL: la intención NO es condicional — se traza como tal (result=True
        # siempre) para que el panel muestre por qué la exposición es la que es
        # sin que el frontend tenga que inferirlo (invariante 7).
        entries = (
            self._entry(
                "hodl_intent",
                "Intención HODL (siempre en mercado)",
                {"intent": intent},
                True,
                "HODL_ALWAYS_IN",
                {"intent": intent},
            ),
            self._entry(
                "vol_target_size",
                "Exposición vol-targeted (spot cap 1.0)",
                {"realized_vol_20": realized_vol, "regime_risk_mult": regime_mult},
                exposure > 0.0,
                "VOL_TARGET_SIZED" if exposure > 0.0 else "SIZE_ZERO",
                {"target_vol": target_vol, "vol_floor": vol_floor,
                 "max_exposure": max_exposure},
            ),
        )

        if exposure > 0.0:
            direction = "LONG"
            reason_codes = ("HODL_ALWAYS_IN", "VOL_TARGET_SIZED")
            trace = RuleTrace(rules=entries, winning_rule_id="vol_target_size")
        else:
            direction = self.default_direction
            exposure = self.default_exposure
            reason_codes = (self.default_reason_code,)
            trace = RuleTrace(rules=entries, fallback_applied=True)

        return self._decision(
            context,
            direction=direction,
            exposure=exposure,
            reason_codes=reason_codes,
            components={
                "intent": intent,
                "realized_vol_20": realized_vol,
                "regime_risk_mult": regime_mult,
                "vol_target_size": size,
                "target_exposure": exposure,
            },
            trace=trace,
        )
