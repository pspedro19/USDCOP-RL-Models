# Handoff — «continúa con Claude»

Actualizado por CODEX: 2026-08-03T20:33:00-05:00.

Cuando el operador diga **«continúa con Claude»**, retomar sin pedir que repita contexto:

1. Leer `PROTOCOL.md`, ambos inboxes, `LEASES.md`, ambos status, `CONTRACTS.md` y este handoff.
2. Consultar otra vez `LEASES.md` inmediatamente antes de cada patch; hubo una colisión real
   durante el review de migraciones y no debe repetirse.
3. Preservar los canales runtime y `data/health/metric_events.jsonl`; no commitearlos ni
   revertirlos.
4. Coordinar por `INBOX-CLAUDE.md`, implementar incrementos acotados y pedir cross-review.

## Estado cofirmado

- Backlog: **10 IMPLEMENTED / 36 PARTIAL / 1 PLANNED**. Solo BL-23 sigue PLANNED.
- No push, DDL, pin de plan, `down -v`, rebuild ni reinicio de Docker sin autorización expresa.
- `fabric-v1` continúa bloqueado por digest fijado distinto del contenido actual; su test rojo
  conocido no debe maquillarse.
- Docker usa un junction hacia `E:` para su disco. No romperlo ni reiniciar Docker sin necesidad.
- El stack estaba levantado; el backtest API fue parado deliberadamente por Claude y no debe
  iniciarse con la imagen vieja, porque conserva el entrypoint que replaya migraciones.

## Trabajo sellado más reciente

- `e4c9d538`: plan review-gated `platform-bootstrap-v1`, sin pin. Allowlist mínima y ordenada:
  045, 046, 050, 051, 053, 054, 055. Excluye 047/pgvector, 052/crypto y rutas H5 redundantes.
  `DATABASE_URL` tiene precedencia; fallback exige password explícito.
- `51b0fb3e`: el runtime ya no replaya `legacy-init`; solo valida el esquema. El bootstrap
  exige `sb_users`, `usdcop_m5_ohlcv` y `macro_indicators_daily` antes de cualquier DDL.
- Claude aprobó ambos commits en CLD-309: **6/6 ataques adversariales muerden**, restauración
  byte-exacta y sin mutaciones vivas. El plan sigue inejecutable deliberadamente por falta de pin.
- `4aa160d2`: segunda línea de tests separa exclusiones y orden 053→055 de la allowlist exacta.
  Focal 5P; suite safety 26P con el único test FABRIC conocido excluido. Se pidió review a Claude
  en CXD-280; recoger su respuesta al volver.
- `7f2bd3ad`: BL-35 registra `airflow dags list-import-errors` exit 0 / `No data found`, cofirmado
  por Claude. Permanece PARTIAL porque falta observar el DAG sintético forecast→exec como import
  error dentro del scheduler real.
- BL-40 orden temporal quedó aprobado en CLD-306; `a07459a3` mata primer/último match.

## Hechos y límites sobre cold boot

- El coldboot observado quedó rojo por esquema y datos faltantes; no declarar `STACK_HEALTHY`.
- Los scripts 02 y 25 fallaron durante **replay** sobre una DB ya transformada. No está probado
  que fallen en el orden one-shot de un volumen fresco. No editarlos: son migraciones aplicadas.
- Claude ofreció ejecutar el ciclo destructivo y entregar salida cruda solo después de pin y
  autoridad del operador. La palabra genérica «continúa» no se trató como autorización de pin
  o borrado de volumen.
- El negativo BL-35 requiere escritura runtime temporal. CXD-279 pidió cesión explícita a Claude;
  recoger ACK antes de crear un DAG sintético, y retirarlo/verificar limpieza si se autoriza.

## Siguiente acción exacta

1. Leer el último `CLD-*` y confirmar que no haya leases activos o mutaciones vivas.
2. Obtener review adversarial de `4aa160d2` y respuesta a la cesión BL-35 de CXD-279.
3. Si el operador autoriza explícitamente el pin/coldboot destructivo, crear el pin en un commit
   separado y revisado; después Claude ejecuta el ciclo acordado. Sin esa autorización, no hacerlo.
4. Si se cede BL-35, ejecutar el negativo temporal dentro del scheduler sin reinicios ni DDL.

## Verificación reciente

- Bootstrap/no-replay: adversarial_remediations 10P; safety 26P con un test FABRIC excluido;
  compileall y diff-check verdes.
- Conocimiento tras BL-35: inventory y doc-index checks verdes; bloque 1054P; enlaces 680 + 3P;
  graph 401/551; graph+honesty 110P/47S; diff-check verde.
- Ruff no estaba instalado en el entorno Python disponible.
