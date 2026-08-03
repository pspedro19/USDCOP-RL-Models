---
kind: audit
status: PAUSED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/harness/harness_engine.py
  - scripts/validation/commerce_rbac_harness.py
  - scripts/validation/run_quant_harness.py
  - scripts/validation/run_production_harness.py
  - .claude/codex/evidence/harness-latest.json
---

### Quantitative unblock update (2026-07-20)

The quant harness now accepts manifest-supplied `trial_count`, `dsr`, `pbo` and
`benchmark_return` and emits explicit checks/metrics. A four-asset evidence
contract lives at `config/quant_evidence/assets.json`; all entries are
intentionally `blocked_external_real_data` until PIT OOS artifacts are attached.
This prevents synthetic data from being promoted while making the final gate
deterministic and machine-readable.
# Harness engineering: estado consolidado

`harness_engine.py` reúne gates deterministas y produce un manifest JSON con estado por dominio,
razones, comandos, activos y evidencia. Los sub-harnesses cubren commerce/RBAC, cuantitativo
multi-activo y producción/retraining.

Última ejecución:

- Unit harness suite: **9 passed**.
- Commerce harness: **7 checks passed**.
- Contratos, commerce, assets, quant y production: **PASS**.
- `real-data-oos`: **BLOCKED** — faltan datasets PIT reales y manifests OOS de los cuatro activos.
- `provider-e2e`: **BLOCKED** — faltan credenciales sandbox/tenant desechable y E2E reales.
- Decisión agregada: **NO-GO**.

Cada ejecución escribe `.claude/codex/evidence/harness-latest.json`. No se permite convertir
`BLOCKED` en `PASS` manualmente ni presentar una métrica sintética como alfa.

El plan ejecutable para eliminar los bloqueos está en [PRODUCTION-UNBLOCK-PLAN.md](../plans/PRODUCTION-UNBLOCK-PLAN.md).

## Resolución en curso

Se están cerrando los bloqueos internos con tres agentes: commerce/RBAC (refund, chargeback,
órdenes y BOLA), quant (fixtures PIT explícitos y manifests por activo) y production (release,
rollback, SLO y observabilidad). Los datos reales y credenciales sandbox siguen siendo dependencias
externas; el harness debe conservarlas como `BLOCKED` hasta recibirlas.

## Resolución aplicada

- Commerce: quote/order server-side, estados paid/failed/refunded/charged_back, billing event
  idempotente y revocación de entitlements.
- Quant: contrato `config/quant_evidence/assets.json` para los cuatro activos; manifests quedan
  `blocked_external_real_data` hasta adjuntar PIT/OOS reales.
- Release: hashes de artefactos, rollback rehearsal, champion/challenger y SLO/telemetría.
- Suite agregada: **9 tests passed**; contratos y sub-harnesses PASS.
- Estado global: **NO-GO únicamente por dependencias externas** (`real-data-oos`, `provider-e2e`).

## Verificación de dependencias externas

La inspección actual confirma que existen seeds históricos para USD/COP, Gold y BTC, pero no
existe un feed PIT/versionado ni un seed real SPY/SPX con `available_at`; convertir timestamps
de cierre en disponibilidad sería fabricar evidencia y queda prohibido. Tampoco se encontraron
credenciales sandbox Wompi/PSP en el workspace. Por eso ambos bloqueos permanecen legítimamente
`BLOCKED` y requieren provisión externa.

## Última verificación de adaptadores

Se localizaron adaptadores TwelveData, FRED, yfinance, GDELT y scrapers de noticias. Ninguno aporta
por sí solo un registro histórico de revisiones/`available_at` para los cuatro activos. Los seeds
actuales son útiles para smoke/backtest de cableado, no para promoción OOS. El plan no los eleva de
categoría automáticamente.

Se corrigieron además los runners `run_quant_harness.py` y `run_production_harness.py` para funcionar
desde la raíz del repositorio; ambos exponen CLI reproducible y la suite permanece en verde.

Se ejecutó `scripts/data/acquire_public_snapshots.py --asset all --start 2020-01-01`: produjo snapshots
para los cuatro activos con checksum y manifests en `data/snapshots/public_daily/`. Todos están
marcados `pit_vintage=false`, `promotion_eligible=false`; sirven para validación de cableado y
reproducibilidad, no para cerrar el gate OOS.
