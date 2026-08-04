"""XAU/USD — ``gold_dynamic_exit`` como la primera policy STATEFUL (BL-45 §15.2).

Es la estrategia que la spec cita como la que **exige** estado, y hasta ahora vivía
sólo en `scripts/analysis/gold_dynamic_exit.py`: un simulador de investigación cuyo
estado eran variables locales de un bucle (`in_trade`, `size`, `hi_close`, `trail_px`).
Mientras estuvo así, `PolicyContext.state` era un campo sin ningún consumidor — el mismo
"mecanismo correcto sin llamador" del que salió BL-16.

**Qué se porta y qué no.** Se porta la **decisión**: qué exposición sostener en cada
barra y por qué salir. NO se porta la contabilidad de PnL del simulador (costes, swap,
acumulación de retornos): eso es ejecución, y el contrato de estrategia separa señal de
ejecución a propósito. Portar ambas cosas juntas habría convertido una migración en un
segundo motor de backtest.

**El estado, explícito.** Cuatro campos en `context.state`, y ninguno derivable de la
barra actual:

* ``in_trade``  — si hay posición abierta;
* ``size``      — el sizing FIJADO EN LA ENTRADA, que no se recalcula mientras dure el
  trade (recalcularlo cada barra sería otra estrategia, no ésta);
* ``hi_close``  — el máximo de cierres desde la entrada;
* ``trail_px``  — el stop, que **sólo sube**.

Que el trailing nunca baje es la propiedad que hace que esto necesite memoria: con sólo
la barra de hoy es imposible saber dónde está el stop.

**Causalidad.** La señal que decide entrar es la de AYER (``signal_prev``), igual que en
el simulador — el bucle lee ``d["sig"].iloc[i-1]``. Usar la de hoy adelantaría la
información un día entero, que es el look-ahead clásico de esta familia.

Esta policy **no está cableada a producción**: existe para cerrar §15.2 y para que el
contrato de estado tenga un consumidor real. Cablearla es una decisión aparte, con sus
trials.

Contract: CTR-POLICY-001 (BL-45 §15.2) · Date: 2026-08-05
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.policy import PolicyContext, StrategyDecision
from src.strategies.policies.base import CodedPolicy

#: Claves del store. Se declaran para que el runner sepa qué debe persistir entre
#: corridas: si se pierden, la policy no puede reanudar un trade abierto.
STATE_KEYS = ("in_trade", "size", "hi_close", "trail_px")


class GoldDynamicExitPolicy(CodedPolicy):
    """Entrada por voto de tendencia; salida por señal muerta o trailing stop ATR."""

    required = ("open", "high", "low", "close", "atr_14", "signal_prev", "size_prev")

    def decide(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        trail_mult = float(self.params["trail_mult"])

        apertura = self._get(snapshot, "open")
        minimo = self._get(snapshot, "low")
        cierre = self._get(snapshot, "close")
        atr = self._get(snapshot, "atr_14")
        senal_ayer = int(self._get(snapshot, "signal_prev"))
        size_ayer = float(self._get(snapshot, "size_prev"))

        estado = context.state
        en_trade = bool(estado.get("in_trade", False))
        entradas = []

        if not en_trade:
            # La señal de AYER decide la entrada de HOY: usar la de hoy sería
            # look-ahead de un día entero.
            abre = senal_ayer == 1 and _finito(size_ayer) and _finito(atr)
            entradas.append(
                self._entry(
                    "entry_on_prev_signal",
                    "Señal de ayer activa",
                    {"signal_prev": senal_ayer, "size_prev": size_ayer},
                    abre,
                    "ENTRY_TREND_VOTE" if abre else "NO_SIGNAL",
                    {},
                )
            )
            if not abre:
                estado.update(in_trade=False, size=0.0, hi_close=0.0, trail_px=0.0)
                return self._decide_out(context, "FLAT", 0.0, ("NO_SIGNAL",), entradas, snapshot)

            estado["in_trade"] = True
            # El sizing se FIJA en la entrada y no se recalcula: recalcularlo cada
            # barra sería otra estrategia.
            estado["size"] = size_ayer
            estado["hi_close"] = max(apertura, cierre)
            estado["trail_px"] = max(
                apertura - trail_mult * atr, estado["hi_close"] - trail_mult * atr
            )
            return self._decide_out(
                context, "LONG", size_ayer, ("ENTRY_TREND_VOTE",), entradas, snapshot
            )

        # En trade: las salidas se evalúan ANTES de actualizar el trailing, igual que el
        # simulador. Actualizar primero dejaría que el stop de hoy persiguiera al precio
        # de hoy y la salida no dispararía nunca.
        size = float(estado.get("size", 0.0))
        trail_px = float(estado.get("trail_px", 0.0))

        senal_muerta = senal_ayer == 0
        toca_trailing = minimo <= trail_px
        entradas.append(
            self._entry(
                "exit_signal_off", "Señal murió ayer",
                {"signal_prev": senal_ayer}, senal_muerta, "EXIT_SIGNAL_OFF", {},
            )
        )
        entradas.append(
            self._entry(
                "exit_trailing_stop", "Mínimo tocó el trailing",
                {"low": minimo, "trail_px": trail_px}, toca_trailing,
                "EXIT_TRAILING_STOP", {"trail_px": trail_px},
            )
        )

        if senal_muerta or toca_trailing:
            razon = "EXIT_SIGNAL_OFF" if senal_muerta else "EXIT_TRAILING_STOP"
            estado.update(in_trade=False, size=0.0, hi_close=0.0, trail_px=0.0)
            return self._decide_out(context, "FLAT", 0.0, (razon,), entradas, snapshot)

        # Sigue dentro: el trailing SÓLO sube.
        estado["hi_close"] = max(float(estado.get("hi_close", cierre)), cierre)
        estado["trail_px"] = max(trail_px, estado["hi_close"] - trail_mult * atr)
        return self._decide_out(context, "LONG", size, ("HOLD_IN_TREND",), entradas, snapshot)

    # ------------------------------------------------------------------ helpers
    def _decide_out(self, context, direccion, exposicion, razones, entradas, snapshot):
        """Envuelve el helper de la base con la traza ya construida."""
        from src.contracts.rule_trace import RuleTrace

        return self._decision(
            context,
            direction=direccion,
            exposure=exposicion,
            reason_codes=razones,
            components={k: snapshot[k] for k in self.required if k in snapshot},
            # `winning_rule_id` debe apuntar a una regla que DISPARO: el contrato de
            # RuleTrace lo exige y tiene razon -- señalar como ganadora una condicion
            # falsa haria ilegible la explicacion. Si ninguna disparo, la decision vino
            # del default y eso es `fallback_applied`.
            trace=RuleTrace(
                rules=tuple(entradas),
                winning_rule_id=next(
                    (e.rule_id for e in entradas if e.result), None
                ),
                fallback_applied=not any(e.result for e in entradas),
            ),
        )


def _finito(valor: float) -> bool:
    return valor == valor and abs(valor) != float("inf")
