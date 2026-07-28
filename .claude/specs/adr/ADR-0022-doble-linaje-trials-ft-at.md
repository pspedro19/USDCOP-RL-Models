---
kind: adr
status: IMPLEMENTED
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/rules/quant-constitution.md
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - .claude/specs/planes/02-usdcop-doble-rol-y-trials.md
  - services/common/metrics.py
---

# ADR-0022 — Doble linaje de trials: FT-xxxx (predictivos) / AT-xxxx (económicos)

## Contexto

La constitución (§2) exige registrar cada mirada como trial, pero el ledger de cada activo
mezcla dos preguntas de naturaleza distinta en un solo conteo plano:

- **Pregunta predictiva**: "¿este modelo/horizonte/feature-set predice?" (métricas: MSE, DA,
  calibración). El zoo de forecasting vive aquí.
- **Pregunta económica**: "¿esta combinación predictor + gate + sizing + salidas gana dinero?"
  (métricas: Calmar, DSR, PnL forward).

El caso límite que forzó la decisión: **convertir un forecast en señal económica**. En
H-META-01 (COP, N=88) la práctica ya se siguió correctamente — el consenso del zoo era un
artefacto predictivo existente, y probarlo como input de sizing se **cobró como un trial
nuevo**, no como "reuso gratis" de trabajo ya pagado. Pero la regla no estaba escrita como
dura, y no existía forma estructurada de trazar de qué trials predictivos desciende una
hipótesis económica (`planes/02-usdcop-doble-rol-y-trials.md` §3-§4).

## Decisión

1. **Dos linajes con prefijo explícito.** Cada trial del `HYPOTHESIS-REGISTRY` de un activo se
   etiqueta **FT-xxxx** (forecast trial: pregunta predictiva) o **AT-xxxx** (action trial:
   pregunta económica). Los trials existentes se etiquetan retroactivamente por familia **sin
   cambiar el conteo**.

2. **Cruzar la muralla cuesta un AT.** Convertir un forecast (o cualquier artefacto
   predictivo: modelo, consenso, ranking) en señal económica = **+1 AT nuevo**, con provenance
   de los FT heredados. La combinación económica es OTRA hipótesis y su juez es el forward de
   la estrategia, no el DA del predictor. (Precedente: H-META-01.)

3. **La separación NO reduce el DSR.** `n_trials_total` del activo = **suma de ambos
   linajes**. Mirar una celda predictiva sigue quemando presupuesto del activo — el DSR
   (`services/common/metrics.py::deflated_sharpe_ratio`) se deflacta con el total, nunca con
   un solo linaje. Lo que este ADR añade es trazabilidad, no descuento.

4. **Campo `provenance` obligatorio en pre-registros nuevos** que crucen la muralla:

   ```yaml
   provenance:
     forecast_trial_ids: [FT-0041, FT-0042]   # FT heredados (entran al N del cluster)
     action_trial_id: AT-0113                  # el trial económico que se cobra ahora
     research_cluster: usdcop_directional_models
   ```

## Qué NO cambia

- La lista de qué cobra y qué no cobra trials (plan 02 §3): regenerar predicciones de un
  modelo congelado, recomputar métricas declaradas, publicar forward, monitorear drift —
  siguen siendo gratis. Modelo nuevo, horizonte nuevo, features post-hoc, escoger el mejor de
  N, o probar el cruce forecast→señal — siguen cobrando.
- El bar DSR > 0.95 y el resto de la constitución §2.

## Consecuencias

- `quant-constitution.md` §2 gana dos viñetas (la regla dura + el DO NOT de que FT también
  deflacta) y sube a Version 1.1.0 — este ADR es el requisito formal para tocarla.
- Los pre-registros nuevos que promuevan un artefacto predictivo a hipótesis económica deben
  declarar `provenance`; un pre-registro sin él que herede forecasts es inválido.
- El presupuesto de miradas queda auditable por familia: cuánto se gastó en predicción pura
  (FT) vs. hipótesis económicas (AT), sin que la partición permita jamás "resetear" el N.
- Revertir este ADR significa volver al ledger plano sin trazabilidad FT→AT.
