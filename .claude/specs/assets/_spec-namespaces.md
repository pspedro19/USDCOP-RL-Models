# SPEC numbering — namespaces por activo (desambiguación)

> **Creado 2026-07-06 (G4, hallazgo I-6 de la auditoría).** Existen DOS familias `SPEC-01..13`
> con numeración solapada. En el repo viven en carpetas distintas (no hay colisión de paths),
> pero al aplanar/exportar los documentos (Projects, zips, prompts) el nombre solo no dice a
> qué activo pertenece. **Regla: al citar una spec fuera de su carpeta, SIEMPRE prefijar con
> el namespace** — `btc:SPEC-XX` o `xau:SPEC-XX`.

| Namespace | Carpeta | Familia | Ejemplos |
|---|---|---|---|
| `btc:` | `btcusdt/design/` | **Diseño de estrategia BTC** (ciclo/posicionamiento/exposure engine) | `btc:SPEC-01` cycle-regime (HMM) · `btc:SPEC-05` event gate LLM · `btc:SPEC-09` RL táctico · `btc:SPEC-11` validation · `btc:SPEC-12` withdrawal · `btc:SPEC-13` integración |
| `xau:` | `xauusd/` | **Implementación pipeline Gold** (datos→features→modelo→backtest) | `xau:SPEC-01` data-ingestion · `xau:SPEC-02` feature-engineering · `xau:SPEC-09` validation-backtest |

Colisiones que ya causaron confusión: `SPEC-11-validation` (BTC) vs `SPEC-09-validation-backtest`
(Gold); `SPEC-01` es HMM en BTC y data-ingestion en Gold.

**Al exportar a un Project externo:** renombrar los archivos con el prefijo (`btc-SPEC-XX-*.md`,
`xau-SPEC-XX-*.md`). Dentro del repo los nombres quedan como están (renombrar = churn en
referencias sin beneficio local).
