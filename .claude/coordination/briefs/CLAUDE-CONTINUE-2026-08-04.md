# Handoff CLAUDE — retomar el 2026-08-04

Escrito por CLAUDE `claude-root-152c263e-r2` al cierre de la jornada (23:xx COT del 2026-08-03).
Orden del operador: *"hagamos lo que podamos hasta las 11:45 pm hora colombia y mañana seguimos"*.

## Cómo retomar (en este orden)

1. `INBOX-CLAUDE.md` (cola de Codex) y `LEASES.md` — **y volver a mirar `LEASES.md` justo antes
   de cada patch**; hubo una colisión real por no hacerlo.
2. `PROGRESS.md` (corte cofirmado) y `CONTRACTS.md` (estado de C-010).
3. Este fichero.

## Estado al corte

- **Corte: 11 IMPLEMENTED / 36 PARTIAL / 0 PLANNED = 47.** Invariante en toda la jornada pese a
  ~20 commits: casi todo lo arreglado hoy **no lo rastreaba ningún BL**.
- Gates verdes al cierre: frontmatter + honestidad + inventario **1112 passed / 47 skipped**;
  grafo 401/551; links 680.
- Stack **vivo** (Docker levantado por el operador). `usdcop-backtest-api` sigue **Exited(1)** a
  propósito: su imagen conserva el entrypoint que replaya migraciones. **No arrancarlo** sin
  rebuild.

## Lo que quedó EN VUELO (primero al retomar)

**Codex está haciendo cross-review adversarial de `3078ce06` (C-010 R3)** con leases sobre
`asset_pipeline_factory.py`, `btc_hodl_b1.yaml` y `smart_simple_v11.yaml`. Ataques acordados:
(a) `policy_id` real pero `SPEC_ONLY` ⇒ cero tareas; (b) spec elegible con engine soportado ⇒
exactamente tres tareas; (c) `engine.type: composite` elegible ⇒ falla cerrado; (d) retirar el
caller ⇒ 2F. **Recoger su veredicto antes de abrir nada nuevo.**

## Decisiones del OPERADOR (dos avanzaron hoy, cuatro siguen abiertas)

| # | Decisión | Estado |
|---|---|---|
| 1 | Pin de `fabric-v1` | **AUTORIZADO Y HECHO** (`98cefd2d`, revisado en CLD-350) |
| 2 | Implementar C-010 R3 | **AUTORIZADO Y HECHO** (`3078ce06`, en review) |
| 3 | **Aplicar** `fabric-v1` | **ABIERTA** — pin ≠ apply |
| 4 | Crear usuario admin | ABIERTA — `sb_users=0`; sin él ninguna página es alcanzable y no hay dump que restaurar |
| 5 | Contrato real del News Engine | ABIERTA — `CLAUDE.md:140` declara uno que `src/news_engine/` no importa |
| 6 | Deriva 20-vs-15 | ABIERTA — bloquea `get_feature_builder("current")`, `ObservationBuilder`, 4F de parity y 1F de determinism |

**La 3 es la de mayor rendimiento:** aplicar `fabric-v1` crea **48 tablas en 11 esquemas** (hoy
sólo existe `demo`) y saca de golpe a los módulos que tienen tests verdes sin proteger nada.
Único movimiento de **datos**: `081` copia los `algorithm='SYNTHETIC'` de `config.models` al
esquema `demo` y luego los borra de origen, añadiendo `CHECK`. Revisado: no hay pérdida.

## Trampas verificadas hoy (no repetirlas)

- `db_migrate.py --status` **hace DDL** (`CREATE TABLE IF NOT EXISTS _migrations`). No usarlo como
  consulta bajo régimen operator-gated.
- `pin != apply`: con el pin puesto, `plan_is_authorized(..., None)` sigue **False**; el operador
  debe pasar `--reviewed-digest`.
- Mi **reloj va desviado** respecto a Codex y al contenedor, que coinciden entre sí. Comparar con
  `docker exec usdcop-airflow-scheduler date -u` antes de juzgar si algo es residuo.
- Los `M` de `git status` sobre ficheros que no toqué suelen ser **CRLF worktree vs LF índice**:
  confirmar con `git diff HEAD --stat` (vacío = contenido idéntico) antes de alarmarse.

## Lo sellado hoy por CLAUDE

`f6b8a211` triage re-medido · `6d3a123c` errata BL-45 #6 · `d3220099`+`343cd02f` auditoría de
cableado (y su corrección de forma) · `4cff73d2` `make test` ya no aborta · `85ce2a83`+`68575bbe`+
`2768cf25`+`1ffc95bc` DLQ · `69b0c632`+`835f836b`+`20a73bf0`+`5ec5e732` sombra de `services` ·
`c665b539`+`6aaae855` determinism · `6a556c3e` normalizer `_meta`/`_metadata` · `286ca56b`+
`d3a75061` erratas BL-45 R2/R3 · `ea8ce071` cierre de corte · `3078ce06` C-010 R3 · `74a4f4f2`
actualización de la auditoría.

## Regla que me llevo escrita

**Una búsqueda por nombre no es una medición de capacidad.** Cinco afirmaciones falsas en un día
por contar apariciones en vez de leer el punto exacto; tres fueron mías. Antes de afirmar sobre el
código: ejecutar o abrir el sitio concreto.
