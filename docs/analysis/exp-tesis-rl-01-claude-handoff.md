---
kind: analysis
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-11
supersedes: []
code_anchors:
  - scripts/diagnostics/audit_thesis_e2e_status.py
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/run_thesis_llm.py
  - .claude/specs/planes/06-PRE-REGISTRATION-v3.md
---

# Handoff Claude — EXP-TESIS-RL-01

Este documento es una orden de verificación, no una autorización para inventar resultados.
El objetivo es completar D–G solo después de que el operador firme el prerregistro v3 y
proporcione las credenciales fuera del repositorio.

## Hecho y verificado por Codex

- La identidad macro v2 pasa: DXY Investing `942611`, Brent/DGS2 FRED e IBR BanRep.
  Esto verifica reproducibilidad con la fuente declarada; no certifica independencia ICE.
- USD/COP diario TwelveData vs Investing `2112`: 1.727 fechas comunes; mediana 0,07782 %,
  P95 0,551359 %, máximo 5,192771 %, `agreement_flag=OK`.
- `causality_gate=True`, portable v2 vigente y conteos 488/226/570.
- El exportador causal de contextos de selección generó 13.334 contextos en
  `outputs/thesis-repair/llm_selection_contexts_v2.jsonl` después de reconstruir el portable;
  el auditor lo marca `DIAGNOSTIC_ONLY` y no lo confunde con un ledger ni con evidencia.
- Freshness macro por serie pasa para forward; las sesiones incompletas se excluyen y se
  registran, no se rellenan.
- S2 y S3 sintéticas ya no son la misma fixture: S2 tiene costo cero y S3 costo 3 pips.
  `test_sanity_fixtures_are_distinct.py` pasa y el agregado corregido
  `outputs/thesis-repair/sanity_protocol_v2.json` fue regenerado. Ningún resultado sintético
  es evidencia de mercado.
- Los tests de investigación ejecutados por Codex pasan: 90 passed, 1 skipped; los grupos
  de paridad/live/HMM pasan 20/20. `ruff` pasa en los archivos modificados, salvo el generador
  histórico de figuras que conserva deuda previa.

## No confirmado y que Claude debe demostrar

1. PPO v2: diez corridas confirmatorias (5 semillas × 2 configuraciones), no el smoke de
   1.000/20.000 pasos. Cada JSON debe declarar `dataset_version=v2`, hash portable, pasos,
   seed y evaluación por semilla. No sobreescribir v1 ni usar selección como confirmación.
2. S2 corregida: conservar y volver a verificar el protocolo de sanidad agregado con hashes;
   comprobar que S2/S3 difieren por el costo y que los criterios ex-ante se cumplen.
3. LLM: DeepSeek primario y Azure robustez, con modelo/deployment, prompt hash, parámetros,
   latencia, respuesta hash y decisión por contexto. Ninguna selección retrospectiva del mejor.
   Requiere prerregistro `SIGNED`; si no, el runner debe abortar antes de llamar al proveedor.
4. Híbrido: fijar antes de correr si es `PPO + FinMA-ES` (diseño original) o `PPO + LLM`.
   No mezclar ambas definiciones en una sola tabla.
5. Figuras v2: generar únicamente desde artefactos validados; no reutilizar las 21 figuras v1.
   Publicar curvas de capital, drawdown, Sharpe móvil, costos, acciones por régimen y semillas.
6. DSR/bootstrap/estadística: calcular después de las corridas, con trials actualizado,
   por semilla y cartera separada. No reportar confirmación con N menor que el preregistrado.

## Orden operativo obligatorio

1. Firmar `06-PRE-REGISTRATION-v3.md` y registrar modelo/deployment/prompt/costos.
2. Ejecutar `audit_thesis_e2e_status.py`; debe seguir mostrando macro/portable/freshness PASS.
3. Ejecutar S2 corregida y registrar `sanity_protocol_v2.json` nuevo.
4. Ejecutar PPO v2 completo con `--all`; guardar en `outputs/thesis-repair/ppo_v2_full/`.
5. Generar contextos confirmatorios desde portable v2, nunca desde el diagnóstico.
6. Ejecutar DeepSeek y Azure con ledgers separados; validar y liquidar ambos.
7. Ejecutar el híbrido previamente definido.
8. Generar figuras/tablas y correr DSR/bootstrap; actualizar el ledger de trials.
9. Ejecutar el juez forward una sola vez desde el freeze; no reajustar después.

## Comandos de control

```powershell
python scripts/diagnostics/audit_thesis_e2e_status.py --output outputs/thesis-repair/e2e_status_current.json
python scripts/diagnostics/audit_research_source_lineage.py --output outputs/thesis-repair/source_lineage_v2.json
python scripts/diagnostics/verify_macro_declared_identity.py --output outputs/thesis-repair/macro_identity_research_v2_latest.json
python scripts/diagnostics/verify_usdcop_daily_cross_source.py --output outputs/thesis-repair/usdcop_daily_cross_source_v2.json
```

Si una corrida falla, publicar el log y marcarla como no ejecutada. No sustituir cifras por
el handoff ni generar una conclusión de rentabilidad. El cierre debe devolver: hashes,
comandos, salida real, artefactos, conteo de trials, tests y una lista explícita de pendientes.
