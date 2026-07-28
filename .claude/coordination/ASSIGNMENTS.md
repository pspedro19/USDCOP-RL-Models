# Asignación del backlog (47 BLs) — CLAUDE vs CODEX
# Regla: el dueño implementa; EL OTRO verifica antes de DONE (cross-review obligatorio).
# Nadie toca archivos de un BL en curso del otro. Compartidos → via CONTRACTS.md.

## CLAUDE (23) — frontend, gobernanza, contratos de señal, motor de políticas, COP
BL-01 BL-02 BL-03 BL-04 BL-05 BL-06   (honestidad frontend + candados)
BL-09 BL-11 BL-12 BL-13 BL-14         (ledger FT/AT, familias, provenance, surface, components)
BL-15 BL-39 BL-42                     (forecast_output, feature-contracts+bit-check v11, unidades+señal)
BL-20                                 (SHAP/interpretabilidad admin)
BL-25 BL-31 BL-32 BL-34               (3 relojes, strangler COP, Passport/Tower, /replay)
BL-36                                 (decisiones inventario DB — con el operador)
BL-45 BL-46 BL-47                     (motor de políticas R1-R8)

## CODEX (24) — DB/migraciones, CI/validadores, identidad, métricas, libro, executor
BL-07 BL-08 BL-10                     (timing_ratio, checklist .env con operador, backfill FT legacy)
BL-16 BL-17 BL-18 BL-19               (CI constitucional, fingerprints, motor métricas, schema forecast)
BL-21 BL-22 BL-23 BL-24               (event sourcing, facts, anti-supervivencia, linaje)
BL-26 BL-27 BL-28 BL-29 BL-30         (snapshot, allocator, factories, qlab, executor)
BL-33 BL-35 BL-37 BL-38               (readiness matrix, URIs, identidades, mercado canónico)
BL-40 BL-41 BL-43 BL-44               (cuarentena, seguridad DB P0, demo sintética, Timescale)

## Fronteras duras
- COP producción (train_and_export_smart_simple.py, forecast_h5_*, manifiesto usdcop): SOLO CLAUDE.
- Migraciones DB (database/migrations/*): SOLO CODEX las escribe; CLAUDE las revisa.
- src/contracts/* y lib/contracts/*: quien lo necesite PRIMERO lo declara en CONTRACTS.md
  y espera ACK del otro si el otro tiene un BL activo que lo toca.
- HYPOTHESIS-REGISTRY y .claude/rules/: NINGUNO los edita sin instrucción del operador
  (excepción: BL-12 que es exactamente esa enmienda, con ADR).
- git push: PROHIBIDO hasta BL-08 cerrado. Commits locales sí, prefijo [claude]/[codex].
